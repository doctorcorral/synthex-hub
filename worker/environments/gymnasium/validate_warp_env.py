#!/usr/bin/env python3
"""
Validation for the Warp env-core against Gymnasium, on CPU.

Runs anywhere `mujoco` + `gymnasium` are installed (no GPU needed).
Proves the parts we reimplemented — reset noise, observation layout,
reward, termination, and the vectorised bit-policy (incl. sin/cos
features) — match Gymnasium when driven identically. The Warp GPU
backend steps the same core, so passing here means a GPU rollout is
correct up to solver numerics.

Usage:  python3 validate_warp_env.py [ENV_NAME]
          ENV_NAME defaults to HalfCheetah-warp-v5; pass e.g.
          InvertedDoublePendulum-warp-v5 to validate that lineage.
Exit code 0 on success, 1 on any mismatch.
"""

import sys
import numpy as np
import gymnasium as gym

import warp_core as core
from warp_backends import CpuBackend
from oracle_port import eval_pred as scalar_eval_pred, bit_policy_action

BITS_PER_DIM = 3


def fail(msg):
    print(f"  FAIL: {msg}")
    return False


def ok(msg):
    print(f"  ok: {msg}")
    return True


def _read_state(b, needs):
    qpos, qvel = b.get_state()
    extras = b.read_fields(needs) if needs else {}
    return {"qpos": qpos, "qvel": qvel, "extras": extras}


def gym_policy_rollout(base_env, bit_preds, seed, max_steps, cfg, bits_per_dim):
    """Reference rollout in Gymnasium under the scalar bit-policy."""
    e = gym.make(base_env)
    obs, _ = e.reset(seed=int(seed))
    total = 0.0
    obs_trace = [obs.copy()]
    try:
        for _ in range(max_steps):
            a = bit_policy_action(bit_preds, obs.tolist(), cfg, bits_per_dim)
            obs, r, term, trunc, _ = e.step(a)
            total += float(r)
            obs_trace.append(obs.copy())
            if term or trunc:
                break
    finally:
        e.close()
    return total, np.array(obs_trace)


def core_policy_rollout(env_name, bit_preds, seed, max_steps, n_bits):
    """Rollout under warp_core + CpuBackend (single world)."""
    spec = core.ENV_SPECS[env_name]
    needs = spec.get("needs", [])
    model, iqp, iqv = core.load_mj_model(env_name)
    qpos, qvel = core.reset_states(env_name, [seed], iqp, iqv)
    b = CpuBackend(model, nworld=1)
    b.set_state(qpos, qvel)
    fs = spec["frame_skip"]
    total = 0.0
    state = _read_state(b, needs)
    obs = spec["obs_fn"](state)
    obs_trace = [obs[0].copy()]
    shared = [pred for pred in bit_preds]  # all-shared, single world
    for _ in range(max_steps):
        bits = core.policy_bits(obs, shared, n_bits)
        actions = core.decode_actions(bits, spec, BITS_PER_DIM)
        prev_state = state
        b.set_ctrl(actions)
        b.step(fs)
        state = _read_state(b, needs)
        total += float(spec["reward_fn"](prev_state, state, actions)[0])
        obs = spec["obs_fn"](state)
        obs_trace.append(obs[0].copy())
        if bool(spec["terminated_fn"](state)[0]):
            break
    return total, np.array(obs_trace)


def main():
    env = sys.argv[1] if len(sys.argv) > 1 else "HalfCheetah-warp-v5"
    if env not in core.ENV_SPECS:
        print(f"unknown env {env}; known={list(core.ENV_SPECS)}")
        return 1
    spec = core.ENV_SPECS[env]
    base = spec["base_env"]
    needs = spec.get("needs", [])
    n_act = spec["n_action_dims"]
    n_bits = n_act * BITS_PER_DIM
    cfg = {
        "n_action_dims": n_act,
        "action_low": spec["action_low"],
        "action_high": spec["action_high"],
    }
    print(f"=== validating {env}  (base={base}, actions={n_act}) ===")

    passed = True
    model, iqp, iqv = core.load_mj_model(env)

    # 1. Reset noise matches Gymnasium bit-for-bit
    print("[1] reset noise vs gymnasium")
    e = gym.make(base).unwrapped
    for seed in (0, 1, 42, 123):
        e.reset(seed=seed)
        gqp, gqv = e.data.qpos.copy(), e.data.qvel.copy()
        cqp, cqv = core.reset_states(env, [seed], iqp, iqv)
        if not (np.allclose(cqp[0], gqp) and np.allclose(cqv[0], gqv)):
            passed &= fail(f"reset mismatch seed={seed}")
            break
    else:
        passed &= ok("reset qpos/qvel identical for all seeds")
    e.close()

    # 2. Observation layout matches Gymnasium (extras included via backend)
    print("[2] observation layout vs gymnasium")
    e = gym.make(base)
    gobs, _ = e.reset(seed=7)
    cqp, cqv = core.reset_states(env, [7], iqp, iqv)
    b = CpuBackend(model, nworld=1)
    b.set_state(cqp, cqv)
    cobs = spec["obs_fn"](_read_state(b, needs))[0]
    if cobs.shape == gobs.shape and np.allclose(cobs, gobs):
        passed &= ok(f"obs shape {cobs.shape} and values identical")
    else:
        passed &= fail(f"obs mismatch: core {cobs.shape} vs gym {gobs.shape} "
                       f"maxΔ={np.abs(cobs[:len(gobs)]-gobs[:len(cobs)]).max():.2e}")
    e.close()

    # 3. Vectorised predicate eval matches scalar oracle eval (incl. sin/cos)
    print("[3] vectorised predicate eval vs scalar (incl. sin/cos)")
    obs_dim = gobs.shape[0]
    rng = np.random.default_rng(0)
    obs_batch = rng.standard_normal((64, obs_dim))
    preds = [
        ["feat", ["axis", 3, 0.5]],
        ["feat", ["diag", 1, 2, -1]],
        ["not", ["feat", ["axis", 0, 0.0]]],
        ["and", ["feat", ["axis", 5, 0.1]], ["feat", ["prod", 2, 4, 0.0]]],
        ["or", ["feat", ["tridiag", 6, 7, 8, 1, -1]], "falsep"],
        ["feat", ["sin_axis", 1, 0.0]],
        ["feat", ["cos_axis", 3, 0.5]],
        ["not", ["feat", ["sin_axis", 7, -0.2]]],
        "truep",
    ]
    mism = 0
    for p in preds:
        vec = core.eval_pred_batch(p, obs_batch)
        sca = np.array([scalar_eval_pred(p, obs_batch[i].tolist()) for i in range(64)])
        mism += int(np.sum(vec != sca))
    passed &= ok("all predicate kinds match scalar") if mism == 0 else fail(f"{mism} mismatches")

    # 4. decode_actions matches scalar bit_policy_action arithmetic
    print("[4] vectorised action decode vs scalar")
    bits = rng.integers(0, 2, size=(32, n_bits)).astype(np.float64)
    dec = core.decode_actions(bits, spec, BITS_PER_DIM)
    weights = [2 ** k for k in range(BITS_PER_DIM)]
    max_sum = sum(weights)
    mism = 0
    for i in range(32):
        for d in range(n_act):
            s = sum(weights[k] * bits[i, d * BITS_PER_DIM + k] for k in range(BITS_PER_DIM))
            expect = -1.0 + 2.0 * s / max_sum
            if abs(expect - dec[i, d]) > 1e-9:
                mism += 1
    passed &= ok("decode arithmetic matches") if mism == 0 else fail(f"{mism} decode mismatches")

    # 5. Full bit-policy rollout reward + termination matches Gymnasium.
    #    Predicates reference only low obs indices (valid for every env)
    #    and exercise axis/sin/cos/tridiag end-to-end.
    print("[5] full rollout reward vs gymnasium (200 steps)")
    bit_preds = []
    for i in range(n_bits):
        k = i % 4
        if k == 0:
            bit_preds.append(["feat", ["axis", 2, 0.0]])
        elif k == 1:
            bit_preds.append(["feat", ["sin_axis", 1, 0.0]])
        elif k == 2:
            bit_preds.append(["feat", ["cos_axis", 3, 0.5]])
        else:
            bit_preds.append(["feat", ["tridiag", 5, 6, 7, 1, -1]])
    for seed in (0, 3):
        gr, gtr = gym_policy_rollout(base, bit_preds, seed, 200, cfg, BITS_PER_DIM)
        cr, ctr = core_policy_rollout(env, bit_preds, seed, 200, n_bits)
        rdiff = abs(gr - cr)
        n_cmp = min(len(gtr), len(ctr))
        odiff = np.abs(gtr[:n_cmp] - ctr[:n_cmp]).max()
        if rdiff < 1e-4 and odiff < 1e-6 and len(gtr) == len(ctr):
            passed &= ok(f"seed={seed}: reward {cr:.4f} (Δ={rdiff:.2e}), "
                         f"obs Δ={odiff:.2e}, len={len(ctr)}")
        else:
            passed &= fail(f"seed={seed}: reward Δ={rdiff:.4f} core={cr:.4f} "
                           f"gym={gr:.4f}, obs Δ={odiff:.2e}, "
                           f"len core={len(ctr)} gym={len(gtr)}")

    print()
    print("RESULT:", "ALL PASS" if passed else "FAILURES")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
