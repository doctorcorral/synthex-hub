#!/usr/bin/env python3
"""
Batched MuJoCo-Warp oracle for the Synthex Hub worker.

Speaks the IDENTICAL line protocol to `oracle_port.py` (one JSON job
per stdin line, one JSON response per stdout line, always echoing
`job_id`), so the Elixir worker treats it as just another adapter.
Set the worker's ORACLE_SCRIPT to this file and advertise the
`mujoco_warp` capability.

The difference from `oracle_port.py` is the execution model: instead of
`gym.make()` + a pure-Python step loop PER (candidate, seed), an entire
chunk's rollouts become one batch of `nworld` worlds advanced in
lockstep. Physics steps run batched on the GPU via `mujoco_warp`; the
binary bit-policy is a cheap vectorised-numpy step on the host (trivial
vs. contact solving). One world per rollout:

  * score_bit:      nworld = len(candidates) * len(seeds)
  * collect_states: nworld = len(seeds)

Environments are the "*-warp-v5" specs in `warp_core.py`; these are
DISTINCT environments from the Gymnasium lineages by design.

Backend is chosen by ORACLE_BACKEND (auto|cpu|warp, default auto):
falls back to a plain-`mujoco` CPU backend when no CUDA device is
present, so this script also runs (slowly) on a CPU box for testing.
"""

import json
import logging
import os
import sys
import traceback

import numpy as np

import warp_core as core
from warp_backends import make_backend

# Fallback oracle for non-warp envs. A GPU worker advertising
# ["mujoco_warp", "mujoco"] prefers Warp chunks but may be handed
# plain MuJoCo chunks when no Warp work is queued; those env_names
# aren't in warp_core.ENV_SPECS, so we delegate them to the original
# per-rollout CPU oracle. This makes oracle_warp_port a strict
# superset, so one worker can serve both adapters.
import oracle_port as cpu_oracle

LOG_PATH = os.environ.get("ORACLE_LOG", "/tmp/synthex_hub_warp_worker.log")
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("oracle_warp")

BACKEND_PREF = os.environ.get("ORACLE_BACKEND", "auto")
USE_GRAPH = os.environ.get("ORACLE_WARP_GRAPH", "1") not in ("0", "false", "False")

log.info("oracle_warp_port starting (pid=%d, backend=%s)", os.getpid(), BACKEND_PREF)


# ── Caches ──────────────────────────────────────────────────────────
#
# Model compilation (via gymnasium) and backend construction (GPU data
# alloc + CUDA-graph capture) are expensive; reuse them across jobs.
# Backends are keyed by (env_name, nworld) since a captured graph is
# tied to a fixed world count.

_MODELS = {}
_BACKENDS = {}


def get_model(env_name, spec=None):
    if env_name not in _MODELS:
        _MODELS[env_name] = core.load_mj_model(env_name, spec)
    return _MODELS[env_name]


def get_backend(env_name, nworld, spec=None):
    key = (env_name, nworld)
    b = _BACKENDS.get(key)
    if b is None:
        model, _, _ = get_model(env_name, spec)
        b = make_backend(model, nworld, prefer=BACKEND_PREF, use_graph=USE_GRAPH)
        _BACKENDS[key] = b
        log.info("built %s backend for %s nworld=%d", b.name, env_name, nworld)
    return b


# ── Batched rollout ─────────────────────────────────────────────────


def _read_state(b, needs):
    """Snapshot the backend into the state dict the env-rule functions
    consume: qpos/qvel plus any derived `needs` (e.g. site_xpos)."""
    qpos, qvel = b.get_state()
    extras = b.read_fields(needs) if needs else {}
    return {"qpos": qpos, "qvel": qvel, "extras": extras}


def rollout(env_name, spec, seeds, bit_preds_per_world, max_steps, bits_per_dim,
            record_stride=None):
    """Advance `len(seeds)` worlds under per-world bit-policies.

    Returns a dict with per-world arrays:
      ep_return (N,)         summed reward while the episode was live
      success   (N,) bool    ep_return > success_threshold
      states    list[N]      subsampled obs (only if record_stride set)
    """
    _, iqp, iqv = get_model(env_name, spec)
    n = len(seeds)
    fs = spec["frame_skip"]
    needs = spec.get("needs", [])
    n_bits = spec["n_action_dims"] * bits_per_dim

    qpos, qvel = core.reset_states(env_name, seeds, iqp, iqv, spec)
    b = get_backend(env_name, n, spec)
    b.set_state(qpos, qvel)

    state = _read_state(b, needs)
    obs = spec["obs_fn"](state)

    ep_return = np.zeros(n, dtype=np.float64)
    done = np.zeros(n, dtype=bool)
    states = [[] for _ in range(n)] if record_stride else None

    for step in range(max_steps):
        if record_stride is not None and step % record_stride == 0:
            live = np.where(~done)[0]
            for i in live:
                states[i].append(obs[i].tolist())

        bits = core.policy_bits(obs, bit_preds_per_world, n_bits)
        actions = core.decode_actions(bits, spec, bits_per_dim)

        prev_state = state
        b.set_ctrl(actions)
        b.step(fs)
        state = _read_state(b, needs)

        step_r = spec["reward_fn"](prev_state, state, actions)
        # Reward only accrues to still-live worlds; the terminating
        # step itself is counted (we mark done AFTER adding it).
        ep_return += np.where(done, 0.0, step_r)

        obs = spec["obs_fn"](state)
        newly = spec["terminated_fn"](state)
        done = done | newly
        if done.all():
            break

    success = ep_return > spec["success_threshold"]
    return {"ep_return": ep_return, "success": success, "states": states}


# ── score_bit ───────────────────────────────────────────────────────


def handle_score_bit(job):
    env_name = job["env_name"]
    spec = core.resolve_spec(env_name, job.get("env_spec"))

    candidates = job["candidates"]
    bit_preds = job["bit_predicates"]
    target_bit = int(job["target_bit"])
    seeds = job.get("seeds", [0])
    max_steps = int(job.get("max_steps", 1000))
    bits_per_dim = int(job.get("bits_per_dim", 3))

    c = len(candidates)
    s = len(seeds)

    # World layout: world index = ci*s + si  -> candidate ci, seed si.
    world_seeds = [seeds[wi % s] for wi in range(c * s)]

    # Per-bit predicates across worlds: every bit is the shared
    # bit_predicates entry EXCEPT target_bit, which takes each world's
    # candidate. `per_world` groups identical candidates so the eval
    # cost is ~#distinct-candidates, not nworld.
    bit_preds_per_world = list(bit_preds)
    target_per_world = [candidates[wi // s] for wi in range(c * s)]
    bit_preds_per_world[target_bit] = core.per_world(target_per_world)

    out = rollout(env_name, spec, world_seeds, bit_preds_per_world, max_steps, bits_per_dim)
    ep = out["ep_return"]
    succ = out["success"]

    results = []
    for ci in range(c):
        block = slice(ci * s, (ci + 1) * s)
        results.append({
            "idx": ci,
            "reward": float(ep[block].sum()),
            "landings": int(succ[block].sum()),
        })
    return results


# ── collect_states ──────────────────────────────────────────────────


def handle_collect_states(job):
    env_name = job["env_name"]
    spec = core.resolve_spec(env_name, job.get("env_spec"))

    seeds = job.get("candidates") or job.get("seeds") or [0]
    bit_preds = job.get("bit_predicates", [])
    max_steps = int(job.get("max_steps", 1000))
    bits_per_dim = int(job.get("bits_per_dim", 3))
    stride = max(1, int(job.get("state_stride", 10)))

    n_bits = spec["n_action_dims"] * bits_per_dim
    # collect_states uses the CURRENT policy for every world; predicates
    # are shared (or default truep when bit_preds is short / empty).
    bit_preds_per_world = [
        bit_preds[b] if b < len(bit_preds) else "truep" for b in range(n_bits)
    ]

    out = rollout(env_name, spec, list(seeds), bit_preds_per_world, max_steps,
                  bits_per_dim, record_stride=stride)
    ep = out["ep_return"]
    succ = out["success"]
    states = out["states"]

    results = []
    for i, seed in enumerate(seeds):
        results.append({
            "idx": i,
            "seed": int(seed),
            "states": states[i],
            "reward": float(ep[i]),
            "success": bool(succ[i]),
        })
    return results


COMMANDS = {
    "score_bit": handle_score_bit,
    "collect_states": handle_collect_states,
}


def handle(job):
    env_name = job.get("env_name", "")
    # Route to the Warp path when the env is baked OR the hub pushed a
    # Warp descriptor (base_env present) — the latter lets brand-new Warp
    # envs run with no rebuild. Everything else falls back to the
    # per-rollout CPU oracle (which itself honours a pushed env_spec).
    if not (core.is_warp_env(env_name) or core.has_warp_descriptor(job.get("env_spec"))):
        return cpu_oracle.handle(job)

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
            reply({"job_id": job_id, "error": f"{type(e).__name__}: {e}"})

    log.info("oracle_warp_port stdin closed; exiting")


if __name__ == "__main__":
    main()
