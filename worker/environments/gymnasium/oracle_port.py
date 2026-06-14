#!/usr/bin/env python3
"""
Persistent Python oracle for the Synthex Hub worker.

Reads one JSON job per line on stdin, writes one JSON response per
line on stdout. ALWAYS produces a response containing the original
`job_id`, even on errors, so the Elixir port never hangs waiting.

Supported commands:
  - score_bit       (default; binary-weighted continuous control)
  - collect_states  (rollout episodes with a fixed bit-policy and
                     return the visited states; one job-unit per seed)

Future commands map onto the same protocol; just add a dispatch case.
"""

import json
import logging
import os
import sys
import traceback

import numpy as np
import gymnasium as gym

from feature_kernels import FEATURE_KERNELS

LOG_PATH = os.environ.get("ORACLE_LOG", "/tmp/synthex_hub_worker.log")
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("oracle")
log.info("oracle_port starting (pid=%d)", os.getpid())

ENV_CONFIGS = {
    "InvertedPendulum-v5": {
        "n_action_dims": 1, "action_low": -3.0, "action_high": 3.0,
        "max_steps": 1000, "success_threshold": 950,
    },
    # Classic-control Pendulum swing-up (not MuJoCo). obs = [cos, sin,
    # angular velocity]; single torque actuator in [-2, 2]; 200-step
    # episodes. Per-step reward in ~[-16.27, 0], so a swing-up-and-hold
    # policy totals roughly -150 to -250; success_threshold marks a
    # cleanly held balance.
    "Pendulum-v1": {
        "n_action_dims": 1, "action_low": -2.0, "action_high": 2.0,
        "max_steps": 200, "success_threshold": -250,
    },
    # Gymnasium 1.x InvertedDoublePendulum-v5: 9-dim obs (matches the
    # Warp lineage's layout), single normalized actuator in [-1, 1].
    # Reward ~9-10/step alive bonus minus distance/velocity penalties;
    # a balanced episode tops out near 9300.
    "InvertedDoublePendulum-v5": {
        "n_action_dims": 1, "action_low": -1.0, "action_high": 1.0,
        "max_steps": 1000, "success_threshold": 9000,
    },
    "Swimmer-v5": {
        "n_action_dims": 2, "action_low": -1.0, "action_high": 1.0,
        "max_steps": 1000, "success_threshold": 50,
    },
    "Hopper-v5": {
        "n_action_dims": 3, "action_low": -1.0, "action_high": 1.0,
        "max_steps": 1000, "success_threshold": 500,
        "env_kwargs": {"healthy_reward": 0.0},
    },
    "HalfCheetah-v5": {
        "n_action_dims": 6, "action_low": -1.0, "action_high": 1.0,
        "max_steps": 1000, "success_threshold": 1000,
    },
    "Walker2d-v5": {
        "n_action_dims": 6, "action_low": -1.0, "action_high": 1.0,
        "max_steps": 1000, "success_threshold": 500,
    },
    "Ant-v5": {
        "n_action_dims": 8, "action_low": -1.0, "action_high": 1.0,
        "max_steps": 1000, "success_threshold": 1000,
    },
    "Humanoid-v5": {
        "n_action_dims": 17, "action_low": -0.4, "action_high": 0.4,
        "max_steps": 1000, "success_threshold": 1000,
    },
}


def resolve_cfg(job):
    """Per-job environment config.

    Prefers a hub-pushed ``env_spec`` so brand-new environments need NO
    worker rebuild — the hub is the single source of truth and ships the
    handful of scalars (action dims/range/steps/threshold) the oracle
    needs; ``gym.make(env_name)`` supplies the physics for any registered
    Gymnasium id. Falls back to the baked ``ENV_CONFIGS`` table when the
    payload carries no spec (older hubs), so nothing regresses.
    """
    env_name = job["env_name"]
    spec = job.get("env_spec")

    if isinstance(spec, dict) and spec.get("n_action_dims") is not None:
        return {
            "n_action_dims": int(spec["n_action_dims"]),
            "action_low": float(spec.get("action_low", -1.0)),
            "action_high": float(spec.get("action_high", 1.0)),
            "max_steps": int(spec.get("max_steps", 1000)),
            # No threshold => never count an episode as a "success"; this
            # only affects the secondary landings metric, not the reward.
            "success_threshold": float(spec.get("success_threshold", float("inf"))),
            "env_kwargs": spec.get("env_kwargs") or {},
        }

    if env_name in ENV_CONFIGS:
        return ENV_CONFIGS[env_name]

    raise ValueError(
        f"unknown env_name: {env_name} and no env_spec in payload; "
        f"baked={list(ENV_CONFIGS)}"
    )


# ── Predicate evaluation ────────────────────────────────────────────


def eval_feature(feat, state):
    # Single state: column accessor returns a scalar. Feature semantics
    # live in the shared registry (feature_kernels) so the CPU and GPU
    # oracles can never drift apart. See feature_kernels.py.
    kern = FEATURE_KERNELS.get(feat[0])
    if kern is None:
        return False
    return bool(kern(feat, lambda i: state[i]))


def eval_pred(pred, state):
    if pred is None or pred == "truep":
        return True
    if pred == "falsep":
        return False
    kind = pred[0]
    if kind == "feat":
        return eval_feature(pred[1], state)
    if kind == "not":
        return not eval_pred(pred[1], state)
    if kind == "and":
        return eval_pred(pred[1], state) and eval_pred(pred[2], state)
    if kind == "or":
        return eval_pred(pred[1], state) or eval_pred(pred[2], state)
    return False


# ── score_bit command ───────────────────────────────────────────────


def bit_policy_action(bit_preds, obs, cfg, bits_per_dim):
    bits = [1 if eval_pred(p, obs) else 0 for p in bit_preds]
    weights = [2 ** i for i in range(bits_per_dim)]
    max_sum = sum(weights)
    n = cfg["n_action_dims"]
    lo, hi = cfg["action_low"], cfg["action_high"]
    actions = np.zeros(n)
    for d in range(n):
        s = sum(weights[i] * bits[d * bits_per_dim + i] for i in range(bits_per_dim))
        actions[d] = lo + (hi - lo) * s / max_sum
    return actions


def score_bit_candidate(env_name, cfg, candidate, bit_preds, target_bit, seeds, max_steps, bits_per_dim):
    test_preds = list(bit_preds)
    test_preds[target_bit] = candidate
    total = 0.0
    successes = 0
    env_kwargs = cfg.get("env_kwargs", {})

    for seed in seeds:
        env = gym.make(env_name, **env_kwargs)
        try:
            obs, _ = env.reset(seed=int(seed))
            ep_r = 0.0
            for _ in range(max_steps):
                action = bit_policy_action(test_preds, obs.tolist(), cfg, bits_per_dim)
                obs, r, term, trunc, _ = env.step(action)
                ep_r += float(r)
                if term or trunc:
                    break
            total += ep_r
            if ep_r > cfg["success_threshold"]:
                successes += 1
        finally:
            env.close()

    return {"reward": total, "landings": successes}


# ── dispatch ────────────────────────────────────────────────────────


def handle_score_bit(job):
    env_name = job["env_name"]
    cfg = resolve_cfg(job)

    candidates = job["candidates"]
    bit_preds = job["bit_predicates"]
    target_bit = int(job["target_bit"])
    seeds = job.get("seeds", [0])
    max_steps = int(job.get("max_steps", 1000))
    bits_per_dim = int(job.get("bits_per_dim", 3))

    results = []
    for i, cand in enumerate(candidates):
        try:
            r = score_bit_candidate(
                env_name, cfg, cand, bit_preds, target_bit, seeds, max_steps, bits_per_dim
            )
            r["idx"] = i
        except Exception as e:
            log.exception("candidate %d failed", i)
            r = {"idx": i, "error": f"{type(e).__name__}: {e}"}
        results.append(r)
    return results


# ── collect_states command ──────────────────────────────────────────
#
# `candidates` is a list of seeds. For each seed we roll out a single
# episode under the given `bit_predicates` and emit one result entry
# carrying the visited states + a per-episode success flag. The master
# concatenates results across all chunks to reconstruct the full state
# pool.


def collect_states_one(env_name, cfg, seed, bit_preds, max_steps, bits_per_dim, stride):
    """
    Roll out one episode; record at most `max_steps // stride` evenly
    spaced states. The master only needs a representative sample for
    feature generation — sending every timestep would push 1-2 MB per
    seed through the hub for nothing.
    """
    env_kwargs = cfg.get("env_kwargs", {})
    env = gym.make(env_name, **env_kwargs)
    try:
        obs, _ = env.reset(seed=int(seed))
        states = []
        ep_r = 0.0
        for step in range(max_steps):
            if step % stride == 0:
                states.append(obs.tolist())
            action = bit_policy_action(bit_preds, obs.tolist(), cfg, bits_per_dim)
            obs, r, term, trunc, _ = env.step(action)
            ep_r += float(r)
            if term or trunc:
                break
    finally:
        env.close()

    return {
        "seed": int(seed),
        "states": states,
        "reward": ep_r,
        "success": ep_r > cfg["success_threshold"],
    }


def handle_collect_states(job):
    env_name = job["env_name"]
    cfg = resolve_cfg(job)

    seeds = job.get("candidates") or job.get("seeds") or [0]
    bit_preds = job.get("bit_predicates", [])
    max_steps = int(job.get("max_steps", 1000))
    bits_per_dim = int(job.get("bits_per_dim", 3))

    # Subsample: keep one of every N timesteps. 10 gives ~100 states
    # per 1000-step episode, plenty for feature generation, and cuts
    # the JSON payload that crosses the hub by 10×. Master can request
    # a different stride via the param.
    stride = max(1, int(job.get("state_stride", 10)))

    results = []
    for i, seed in enumerate(seeds):
        try:
            r = collect_states_one(
                env_name, cfg, seed, bit_preds, max_steps, bits_per_dim, stride
            )
            r["idx"] = i
        except Exception as e:
            log.exception("collect_states seed=%s failed", seed)
            r = {"idx": i, "seed": int(seed), "error": f"{type(e).__name__}: {e}",
                 "states": [], "reward": 0.0, "success": False}
        results.append(r)
    return results


COMMANDS = {
    "score_bit": handle_score_bit,
    "collect_states": handle_collect_states,
}


def handle(job):
    cmd = job.get("cmd", "score_bit")
    handler = COMMANDS.get(cmd)
    if handler is None:
        raise ValueError(f"unknown cmd: {cmd}; known={list(COMMANDS)}")
    return handler(job)


def reply(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        job_id = None
        try:
            job = json.loads(line)
            job_id = job.get("job_id")
            results = handle(job)
            reply({"job_id": job_id, "results": results})
        except Exception as e:
            tb = traceback.format_exc()
            log.error("job %s failed: %s\n%s", job_id, e, tb)
            reply(
                {
                    "job_id": job_id,
                    "error": f"{type(e).__name__}: {e}",
                }
            )

    log.info("oracle_port stdin closed; exiting")


if __name__ == "__main__":
    main()
