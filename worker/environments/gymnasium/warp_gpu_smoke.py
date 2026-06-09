#!/usr/bin/env python3
"""
GPU smoke test for the MuJoCo-Warp backend.

Run this ON the GPU box (the M1200/Omarchy 5520). It:

  1. confirms mujoco_warp sees a CUDA device,
  2. runs the SAME fixed bit-policy rollout batch on both the CPU
     backend and the Warp backend and compares episode returns —
     they should agree CLOSELY (not bit-for-bit; Warp and C use
     different solvers, so small numeric drift is expected and fine),
  3. reports throughput (rollouts/sec) for each, and a Warp-only
     batch-size sweep to show the GPU batching win.

Exit 0 if the GPU backend runs and agrees with CPU within tolerance.
"""

import sys
import time
import numpy as np

import warp_core as core
from warp_backends import CpuBackend, WarpBackend, warp_available

ENV = "HalfCheetah-warp-v5"
BITS_PER_DIM = 3
N_BITS = 6 * BITS_PER_DIM

# A fixed, all-shared policy (every world runs the same predicates).
BIT_PREDS = [
    (["feat", ["axis", 8, 0.0]] if i % 3 == 0 else
     ["feat", ["diag", 1, 9, 1]] if i % 3 == 1 else "truep")
    for i in range(N_BITS)
]


def rollout(backend, spec, iqp, iqv, seeds, max_steps):
    n = len(seeds)
    qpos, qvel = core.reset_states(ENV, seeds, iqp, iqv)
    backend.set_state(qpos, qvel)
    qp, qv = backend.get_state()
    obs = spec["obs_fn"](qp, qv)
    ep = np.zeros(n, dtype=np.float64)
    done = np.zeros(n, dtype=bool)
    fs = spec["frame_skip"]
    for _ in range(max_steps):
        bits = core.policy_bits(obs, BIT_PREDS, N_BITS)
        actions = core.decode_actions(bits, spec, BITS_PER_DIM)
        xb = qp[:, spec["x_index"]].copy()
        backend.set_ctrl(actions)
        backend.step(fs)
        qp, qv = backend.get_state()
        xa = qp[:, spec["x_index"]]
        ep += np.where(done, 0.0, spec["reward_fn"](xb, xa, actions))
        obs = spec["obs_fn"](qp, qv)
        done = done | spec["terminated_fn"](qp, qv)
        if done.all():
            break
    return ep


def main():
    print("[0] warp_available:", warp_available())
    if not warp_available():
        print("  FAIL: no CUDA device visible to Warp. Check nvidia driver + warp-lang/mujoco-warp install.")
        return 1

    spec = core.ENV_SPECS[ENV]
    model, iqp, iqv = core.load_mj_model(ENV)

    # 1. Correctness: CPU vs Warp on the same N seeds.
    n = 64
    max_steps = 300
    seeds = list(range(n))

    print(f"[1] correctness CPU vs Warp (N={n}, steps={max_steps})")
    cpu = CpuBackend(model, n)
    t0 = time.time()
    ep_cpu = rollout(cpu, spec, iqp, iqv, seeds, max_steps)
    t_cpu = time.time() - t0

    warp = WarpBackend(model, n, use_graph=True)
    # one warm rollout (kernel JIT + graph capture) excluded from timing
    rollout(warp, spec, iqp, iqv, seeds, 5)
    t0 = time.time()
    ep_warp = rollout(warp, spec, iqp, iqv, seeds, max_steps)
    t_warp = time.time() - t0

    diff = np.abs(ep_cpu - ep_warp)
    print(f"  CPU  : {t_cpu:.3f}s  ({n / t_cpu:.1f} rollouts/s)  mean_return={ep_cpu.mean():.3f}")
    print(f"  Warp : {t_warp:.3f}s  ({n / t_warp:.1f} rollouts/s)  mean_return={ep_warp.mean():.3f}")
    print(f"  return agreement: max|Δ|={diff.max():.4f}  mean|Δ|={diff.mean():.4f}")
    # Tolerance: solver drift over 300 steps. Returns are O(10-100);
    # allow a few % relative or a small absolute band.
    tol = 1.0 + 0.05 * np.abs(ep_cpu).mean()
    agree = diff.max() < tol
    print(f"  within tolerance ({tol:.3f}): {agree}")

    # 2. Warp-only throughput sweep (where batching pays off).
    print("[2] Warp throughput sweep")
    for nb in (64, 256, 1024, 4096):
        try:
            wb = WarpBackend(model, nb, use_graph=True)
            sds = list(range(nb))
            rollout(wb, spec, iqp, iqv, sds, 5)  # warm
            t0 = time.time()
            rollout(wb, spec, iqp, iqv, sds, max_steps)
            dt = time.time() - t0
            print(f"  N={nb:>5}: {dt:.3f}s  ({nb / dt:.1f} rollouts/s)")
        except Exception as e:
            print(f"  N={nb:>5}: FAILED ({type(e).__name__}: {e})")
            break

    print()
    print("RESULT:", "PASS" if agree else "DISAGREE (investigate solver/config)")
    return 0 if agree else 1


if __name__ == "__main__":
    sys.exit(main())
