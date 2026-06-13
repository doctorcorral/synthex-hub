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

from feature_kernels import FEATURE_KERNELS

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
    # Batch of N worlds: column accessor returns the (N,) slice obs[:, i].
    # Feature semantics live in the shared registry (feature_kernels) so
    # this vectorised path can never drift from the CPU oracle. The
    # kernel bodies are pure-numpy and allocation-light, matching this
    # hot path's requirements. See feature_kernels.py.
    kern = FEATURE_KERNELS.get(feat[0])
    if kern is None:
        return np.zeros(obs.shape[0], dtype=bool)
    return kern(feat, lambda i: obs[:, i])


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


#
# Env-rule functions take a `state` dict so reward/obs/termination can
# depend on derived quantities (site positions, constraint forces) that
# live outside qpos/qvel:
#     state = {"qpos": (N,nq), "qvel": (N,nv), "extras": {name: array}}
# `extras` is populated by the rollout from the backend's read_fields
# for whatever the spec declares in "needs".


def _obs_exclude_x(state):
    # qpos[:, 1:] drops the root x-slide, matching
    # exclude_current_positions_from_observation=True.
    qpos, qvel = state["qpos"], state["qvel"]
    return np.concatenate([qpos[:, 1:], qvel], axis=1)


def _obs_exclude_xy(state):
    # qpos[:, 2:] drops the root x/y slides (Ant/Humanoid-style
    # exclude_current_positions_from_observation=True), keeping the
    # torso z + orientation quaternion + joint angles, then qvel.
    qpos, qvel = state["qpos"], state["qvel"]
    return np.concatenate([qpos[:, 2:], qvel], axis=1)


def _obs_full(state):
    return np.concatenate([state["qpos"], state["qvel"]], axis=1)


def _forward_reward(weight, ctrl_weight, dt, x_index):
    def reward(prev, cur, ctrl):
        x_before = prev["qpos"][:, x_index]
        x_after = cur["qpos"][:, x_index]
        forward = weight * (x_after - x_before) / dt
        ctrl_cost = ctrl_weight * np.sum(np.square(ctrl), axis=1)
        return forward - ctrl_cost

    return reward


def _ant_reward(dt, ctrl_weight, healthy_reward, z_lo, z_hi):
    # Ant-v5 reward, minus the (small, default 5e-4) contact-cost term —
    # contact_cost needs cfrc_ext, which we deliberately omit so this
    # Warp env stays a clean, self-contained physics task:
    #   forward_x_velocity + healthy_bonus - ctrl_cost.
    def reward(prev, cur, ctrl):
        x_before = prev["qpos"][:, 0]
        x_after = cur["qpos"][:, 0]
        forward = (x_after - x_before) / dt
        ctrl_cost = ctrl_weight * np.sum(np.square(ctrl), axis=1)
        z = cur["qpos"][:, 2]
        healthy = (z >= z_lo) & (z <= z_hi)
        alive = np.where(healthy, healthy_reward, 0.0)
        return forward + alive - ctrl_cost

    return reward


def _never_terminates(state):
    return np.zeros(state["qpos"].shape[0], dtype=bool)


def _z_angle_healthy(z_idx, z_lo, z_hi, ang_idx, ang_lo, ang_hi):
    # Hopper/Walker-style healthy check: root height in [z_lo, z_hi]
    # and torso angle in [ang_lo, ang_hi]; terminate when unhealthy.
    def terminated(state):
        qpos = state["qpos"]
        z = qpos[:, z_idx]
        ang = qpos[:, ang_idx]
        healthy = (z > z_lo) & (z < z_hi) & (ang > ang_lo) & (ang < ang_hi)
        return ~healthy

    return terminated


def _z_healthy(z_idx, z_lo, z_hi):
    def terminated(state):
        z = state["qpos"][:, z_idx]
        return ~((z >= z_lo) & (z <= z_hi))

    return terminated


# ── InvertedDoublePendulum-v5: tip-based reward/termination. The pole
#    tip Cartesian position (site_xpos) and the constraint force
#    (qfrc_constraint, obs[8]) are NOT in qpos/qvel, so this spec
#    declares them in "needs" and the backend batches them each step.


def _idp_obs(state):
    # Matches Gymnasium InvertedDoublePendulum._get_obs exactly:
    #   [cart_x, sin(θ1,θ2), cos(θ1,θ2), clip(qvel,-10,10),
    #    clip(qfrc_constraint,-10,10)[:1]]
    qpos, qvel = state["qpos"], state["qvel"]
    qfrc = state["extras"]["qfrc_constraint"]  # (N, nv)
    return np.concatenate(
        [
            qpos[:, :1],
            np.sin(qpos[:, 1:]),
            np.cos(qpos[:, 1:]),
            np.clip(qvel, -10.0, 10.0),
            np.clip(qfrc[:, :1], -10.0, 10.0),
        ],
        axis=1,
    )


def _idp_reward(prev, cur, ctrl):
    # Gymnasium IDP reward, vectorised. The tip Cartesian (x, y=height)
    # comes from the "tip" site; v1,v2 are the two hinge velocities.
    site = cur["extras"]["site_xpos"]  # (N, nsite, 3)
    x = site[:, 0, 0]
    y = site[:, 0, 2]
    v1 = cur["qvel"][:, 1]
    v2 = cur["qvel"][:, 2]
    dist_penalty = 0.01 * x ** 2 + (y - 2.0) ** 2
    vel_penalty = 1e-3 * v1 ** 2 + 5e-3 * v2 ** 2
    terminated = y <= 1.0
    alive_bonus = np.where(terminated, 0.0, 10.0)
    return alive_bonus - dist_penalty - vel_penalty


def _idp_terminated(state):
    y = state["extras"]["site_xpos"][:, 0, 2]
    return y <= 1.0


# spec fields:
#   base_env      Gymnasium id to source the model + init pose from
#   frame_skip    physics substeps per policy action
#   dt            frame_skip * model.opt.timestep (cached at load)
#   n_action_dims actuator count
#   action_low/high  symmetric action clamp the bit-policy decodes into
#   needs         extra MjData fields to batch each step (default [])
#   obs_fn(state) -> (N, obs_dim)
#   reward_fn(prev_state, state, ctrl) -> (N,)
#   terminated_fn(state) -> (N,) bool
#   success_threshold  episode-return cutoff counted as a "success"
ENV_SPECS = {
    "HalfCheetah-warp-v5": {
        "base_env": "HalfCheetah-v5",
        "frame_skip": 5,
        "n_action_dims": 6,
        "action_low": -1.0,
        "action_high": 1.0,
        "needs": [],
        "obs_fn": _obs_exclude_x,
        "reward_fn": _forward_reward(1.0, 0.1, 0.05, 0),
        "terminated_fn": _never_terminates,
        "success_threshold": 1000.0,
        "reset_noise_scale": 0.1,
    },
    "InvertedDoublePendulum-warp-v5": {
        "base_env": "InvertedDoublePendulum-v5",
        "frame_skip": 5,
        "n_action_dims": 1,
        "action_low": -1.0,
        "action_high": 1.0,
        "needs": ["site_xpos", "qfrc_constraint"],
        "obs_fn": _idp_obs,
        "reward_fn": _idp_reward,
        "terminated_fn": _idp_terminated,
        "success_threshold": 9100.0,
        "reset_noise_scale": 0.1,
    },
    # Ant: 8 actuators, obs = qpos[2:]=13 + qvel=14 = 27 (no contact
    # forces). Reward = forward x-velocity + healthy bonus - ctrl cost;
    # terminate when torso z leaves [0.2, 1.0]. dt = 5 * 0.01 = 0.05.
    "Ant-warp-v5": {
        "base_env": "Ant-v5",
        "frame_skip": 5,
        "n_action_dims": 8,
        "action_low": -1.0,
        "action_high": 1.0,
        "needs": [],
        "obs_fn": _obs_exclude_xy,
        "reward_fn": _ant_reward(0.05, 0.5, 1.0, 0.2, 1.0),
        "terminated_fn": _z_healthy(2, 0.2, 1.0),
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
