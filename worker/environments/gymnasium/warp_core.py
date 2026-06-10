#!/usr/bin/env python3
"""
Backend-agnostic core for the MuJoCo-Warp Synthex environments.

This module owns everything about a "*-warp-v5" environment EXCEPT the
physics stepping itself:

  * the MuJoCo model (loaded from the corresponding Gymnasium env, so
    the dynamics are identical to the CPU lineage's),
  * the seeded reset-noise distribution,
  * the observation layout,
  * the reward function,
  * the early-termination predicate,
  * the binary-weighted bit-policy (vectorised across worlds).

Two backends plug into it (see `warp_backends.py`): a plain-`mujoco`
CPU backend that runs anywhere (used for correctness validation and as
a fallback), and a `mujoco_warp` GPU backend that batches thousands of
worlds on an NVIDIA device. Both step the SAME core, so a Warp rollout
and a CPU rollout of the same policy from the same seed agree up to
solver numerics.

IMPORTANT — these are *distinct environments* from the Gymnasium ones.
We reproduce the reward STRUCTURE (and, where cheap, the exact reset
RNG) faithfully, but no attempt is made to unify Warp and CPU lineages
or to compare their rewards. Each has its own env_name / policy
lineage on the hub.

All policy / predicate / reward math is vectorised over a leading
"world" axis of size N (one world per rollout), so a whole chunk of
candidate×seed rollouts advances in lockstep.
"""

from __future__ import annotations

import numpy as np

try:  # gymnasium is only needed to source the MJCF model + init pose
    import gymnasium as gym
    from gymnasium.utils import seeding
except Exception:  # pragma: no cover - surfaced lazily in load_model
    gym = None
    seeding = None


# ── Vectorised predicate / feature evaluation ───────────────────────
#
# Mirrors `oracle_port.py::eval_feature / eval_pred` exactly, but every
# operation is over an observation BATCH of shape (N, obs_dim) and
# returns a boolean mask of shape (N,). This is the hot path of the
# policy, so it must stay pure-numpy and allocation-light.


def eval_feature_batch(feat, obs):
    kind = feat[0]
    if kind == "axis":
        return obs[:, feat[1]] < feat[2]
    if kind == "diag":
        return feat[3] * obs[:, feat[1]] + obs[:, feat[2]] < 0
    if kind == "sq_diag":
        return feat[3] * obs[:, feat[1]] ** 2 + obs[:, feat[2]] < 0
    if kind == "prod":
        return obs[:, feat[1]] * obs[:, feat[2]] < feat[3]
    if kind == "tridiag":
        return (
            feat[4] * obs[:, feat[1]]
            + feat[5] * obs[:, feat[2]]
            + obs[:, feat[3]]
            < 0
        )
    return np.zeros(obs.shape[0], dtype=bool)


def eval_pred_batch(pred, obs):
    """Boolean mask (N,) for predicate `pred` over obs batch (N, dim)."""
    n = obs.shape[0]
    if pred is None or pred == "truep":
        return np.ones(n, dtype=bool)
    if pred == "falsep":
        return np.zeros(n, dtype=bool)
    kind = pred[0]
    if kind == "feat":
        return eval_feature_batch(pred[1], obs)
    if kind == "not":
        return ~eval_pred_batch(pred[1], obs)
    if kind == "and":
        return eval_pred_batch(pred[1], obs) & eval_pred_batch(pred[2], obs)
    if kind == "or":
        return eval_pred_batch(pred[1], obs) | eval_pred_batch(pred[2], obs)
    return np.zeros(n, dtype=bool)


# ── Environment specifications ──────────────────────────────────────
#
# Each Warp env maps onto a base Gymnasium env (for model + init pose)
# plus the reward/obs/termination rules that Gymnasium implements in
# Python wrappers (and which MuJoCo Warp does NOT provide — Warp only
# gives us the physics). We reimplement those rules here, vectorised.


def _obs_exclude_x(qpos, qvel):
    # qpos[:, 1:] drops the root x-slide, matching
    # exclude_current_positions_from_observation=True.
    return np.concatenate([qpos[:, 1:], qvel], axis=1)


def _obs_full(qpos, qvel):
    return np.concatenate([qpos, qvel], axis=1)


def _forward_reward(weight, ctrl_weight, dt):
    def reward(x_before, x_after, ctrl):
        forward = weight * (x_after - x_before) / dt
        ctrl_cost = ctrl_weight * np.sum(np.square(ctrl), axis=1)
        return forward - ctrl_cost

    return reward


def _never_terminates(qpos, qvel):
    return np.zeros(qpos.shape[0], dtype=bool)


def _z_angle_healthy(z_idx, z_lo, z_hi, ang_idx, ang_lo, ang_hi):
    # Hopper/Walker-style healthy check: root height in [z_lo, z_hi]
    # and torso angle in [ang_lo, ang_hi]; terminate when unhealthy.
    def terminated(qpos, qvel):
        z = qpos[:, z_idx]
        ang = qpos[:, ang_idx]
        healthy = (z > z_lo) & (z < z_hi) & (ang > ang_lo) & (ang < ang_hi)
        return ~healthy

    return terminated


def _z_healthy(z_idx, z_lo, z_hi):
    def terminated(qpos, qvel):
        z = qpos[:, z_idx]
        return ~((z >= z_lo) & (z <= z_hi))

    return terminated


# spec fields:
#   base_env      Gymnasium id to source the model + init pose from
#   frame_skip    physics substeps per policy action
#   dt            frame_skip * model.opt.timestep (for forward reward)
#   n_action_dims actuator count
#   action_low/high  symmetric action clamp the bit-policy decodes into
#   x_index       qpos index used for forward progress (root x-slide)
#   obs_fn(qpos,qvel) -> (N, obs_dim)
#   reward_fn(x_before, x_after, ctrl) -> (N,)
#   terminated_fn(qpos, qvel) -> (N,) bool
#   success_threshold  episode-return cutoff counted as a "landing"
ENV_SPECS = {
    "HalfCheetah-warp-v5": {
        "base_env": "HalfCheetah-v5",
        "frame_skip": 5,
        "n_action_dims": 6,
        "action_low": -1.0,
        "action_high": 1.0,
        "x_index": 0,
        "obs_fn": _obs_exclude_x,
        "reward_fn": _forward_reward(1.0, 0.1, 0.05),
        "terminated_fn": _never_terminates,
        "success_threshold": 1000.0,
        "reset_noise_scale": 0.1,
    },
}


def is_warp_env(env_name):
    return env_name in ENV_SPECS


# ── Model loading + seeded reset ────────────────────────────────────


def load_mj_model(env_name):
    """Compiled `mujoco.MjModel` + init pose for a Warp env, sourced
    from its base Gymnasium env so the physics match the CPU lineage."""
    spec = ENV_SPECS[env_name]
    if gym is None:
        raise RuntimeError("gymnasium is required to load the base model")
    base = gym.make(spec["base_env"]).unwrapped
    model = base.model
    init_qpos = np.array(base.init_qpos, dtype=np.float64)
    init_qvel = np.array(base.init_qvel, dtype=np.float64)
    # Cache dt off the compiled model so a spec needn't hardcode it.
    spec["dt"] = spec["frame_skip"] * float(model.opt.timestep)
    base.close()
    return model, init_qpos, init_qvel


def reset_states(env_name, seeds, init_qpos, init_qvel):
    """Per-seed initial (qpos, qvel) batches of shape (N, nq)/(N, nv).

    Reproduces Gymnasium's MuJoCo reset noise EXACTLY (validated
    bit-for-bit): SeedSequence->PCG64 per seed, uniform(-s, s) on
    qpos and s*N(0,1) on qvel, in that draw order.
    """
    spec = ENV_SPECS[env_name]
    ns = spec["reset_noise_scale"]
    nq = init_qpos.shape[0]
    nv = init_qvel.shape[0]
    qpos = np.empty((len(seeds), nq), dtype=np.float64)
    qvel = np.empty((len(seeds), nv), dtype=np.float64)
    for i, seed in enumerate(seeds):
        rng, _ = seeding.np_random(int(seed))
        qpos[i] = init_qpos + rng.uniform(low=-ns, high=ns, size=nq)
        qvel[i] = init_qvel + ns * rng.standard_normal(nv)
    return qpos, qvel


# ── Vectorised bit-policy ───────────────────────────────────────────


def bit_weights(bits_per_dim):
    return np.array([2 ** i for i in range(bits_per_dim)], dtype=np.float64)


def decode_actions(bits, spec, bits_per_dim):
    """Binary-weighted decode. `bits` is (N, n_dims*bits_per_dim) of
    0/1; returns actions (N, n_dims) in [action_low, action_high].

    Identical decode to oracle_port.bit_policy_action, batched."""
    weights = bit_weights(bits_per_dim)
    max_sum = weights.sum()
    n_dims = spec["n_action_dims"]
    lo, hi = spec["action_low"], spec["action_high"]
    n = bits.shape[0]
    actions = np.empty((n, n_dims), dtype=np.float64)
    for d in range(n_dims):
        seg = bits[:, d * bits_per_dim : (d + 1) * bits_per_dim]
        s = seg @ weights
        actions[:, d] = lo + (hi - lo) * s / max_sum
    return actions


def policy_bits(obs, bit_preds_per_world, n_bits):
    """Evaluate the per-world bit predicates against the obs batch.

    `bit_preds_per_world` is a list of length n_bits; each entry is
    EITHER a single predicate shared by all worlds (fast path, one
    vectorised eval) OR a list of length N giving a distinct predicate
    per world, which we evaluate grouped by identity to stay cheap
    (used for the score_bit target bit, where only ~chunk_size
    distinct candidates exist across the batch).

    Returns float array (N, n_bits) of 0.0/1.0.
    """
    n = obs.shape[0]
    bits = np.zeros((n, n_bits), dtype=np.float64)
    for b in range(n_bits):
        entry = bit_preds_per_world[b]
        if isinstance(entry, _PerWorld):
            for pred, idx in entry.groups():
                bits[idx, b] = eval_pred_batch(pred, obs[idx]).astype(np.float64)
        else:
            bits[:, b] = eval_pred_batch(entry, obs).astype(np.float64)
    return bits


class _PerWorld:
    """Wraps a length-N list of predicates, grouping identical ones so
    `policy_bits` evaluates each distinct predicate once over its
    world-index block rather than per world.

    The grouping is STATIC for the lifetime of a rollout (the per-world
    predicate assignment never changes between steps), so it is computed
    once and cached. Recomputing it per step was an O(N) `json.dumps`
    storm — millions of calls per chunk — that dwarfed the GPU physics
    and pinned throughput to CPU-fallback levels."""

    def __init__(self, preds_per_world):
        self._preds = preds_per_world
        self._groups = None

    def groups(self):
        if self._groups is None:
            import json

            buckets = {}
            order = []
            for i, p in enumerate(self._preds):
                key = json.dumps(p, sort_keys=True)
                if key not in buckets:
                    buckets[key] = (p, [])
                    order.append(key)
                buckets[key][1].append(i)
            self._groups = [
                (buckets[key][0], np.array(buckets[key][1], dtype=np.intp))
                for key in order
            ]
        return self._groups


def per_world(preds_per_world):
    return _PerWorld(preds_per_world)
