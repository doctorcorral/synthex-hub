#!/usr/bin/env python3
"""One-shot GPU self-test, run by the installer inside the worker
container right after it starts.

Confirms the worker will ACTUALLY use the GPU for real chunks instead
of silently falling back to CPU, AND that the GPU computes the SAME
reward/termination as the CPU reference. A backend that steps cleanly
can still produce wrong rewards (e.g. the CUDA-graph warmup off-by-one
that zeroed short InvertedDoublePendulum episodes), so a step-only test
is not enough — each env is also checked for CPU/WARP reward parity.

For every env tested it:
  1. Builds the backend via the exact make_backend() decision the oracle
     uses, at a representative batch size (reproduces large-nworld
     failures like CUDA-graph capture or OOM).
  2. Steps once to prove end-to-end GPU execution.
  3. Rolls out the `falsep` baseline policy on CPU and on WARP through the
     exact loop the oracle uses and requires WARP to agree with CPU.

Prints single-line verdicts and exits:
    0  -> all tested envs: GPU (Warp) active and reward matches CPU
    1  -> a failure (CPU fallback / step error / reward disagreement)
    2  -> setup/import error

Usage:  python3 warp_selftest.py [ENV_NAME] [NWORLD]
        With no ENV_NAME, sweeps DEFAULT_ENVS (long- and short-episode
        envs) so reward-correctness regressions are caught on both.
"""

import sys

import numpy as np

# Long-episode (throughput regression) + very-short-episode (exposes
# CUDA-graph / reward-correctness bugs) coverage.
DEFAULT_ENVS = ["HalfCheetah-warp-v5", "InvertedDoublePendulum-warp-v5"]

NWORLD = int(sys.argv[2]) if len(sys.argv) > 2 else 4096

try:
    import warp_core as core
    from warp_backends import make_backend, warp_available, CpuBackend, WarpBackend
except Exception as e:  # noqa: BLE001
    print(f"SELFTEST: FAIL - import error: {e}", flush=True)
    sys.exit(2)


def _baseline_return(env, spec, iqp, iqv, bk):
    """Mean return of the `falsep` baseline policy (constant action_low)
    over a small seed batch, via the exact rollout the oracle uses."""
    seeds = list(range(16))
    n = len(seeds)
    needs = spec.get("needs", [])
    fs = spec["frame_skip"]
    bpd = 4
    n_bits = spec["n_action_dims"] * bpd
    preds = [["falsep"] * n_bits for _ in range(n)]

    q, v = core.reset_states(env, seeds, iqp, iqv)
    bk.set_state(q, v)

    def _state():
        qq, vv = bk.get_state()
        extras = bk.read_fields(needs) if needs else {}
        return {"qpos": qq, "qvel": vv, "extras": extras}

    st = _state()
    obs = spec["obs_fn"](st)
    ep = np.zeros(n)
    done = np.zeros(n, dtype=bool)
    for _ in range(1000):
        bits = core.policy_bits(obs, preds, n_bits)
        act = core.decode_actions(bits, spec, bpd)
        prev = st
        bk.set_ctrl(act)
        bk.step(fs)
        st = _state()
        ep += np.where(done, 0.0, spec["reward_fn"](prev, st, act))
        obs = spec["obs_fn"](st)
        done = done | spec["terminated_fn"](st)
        if done.all():
            break
    return float(ep.mean())


def selftest_env(env):
    """Return True iff `env` runs on the GPU with CPU-matching rewards."""
    print(
        f"SELFTEST: env={env} nworld={NWORLD} warp_available={warp_available()}",
        flush=True,
    )

    try:
        model, iqp, iqv = core.load_mj_model(env)
    except Exception as e:  # noqa: BLE001
        print(f"SELFTEST: FAIL - could not load model {env}: {e}", flush=True)
        return False

    spec = core.ENV_SPECS[env]

    # make_backend prints the canonical [warp-backend] verdict (and, on a
    # build failure, the full traceback) to stderr itself.
    b = make_backend(model, NWORLD, prefer="auto")
    if b.name != "warp":
        print(
            "SELFTEST: FAIL - worker is on the CPU fallback (see the "
            "[warp-backend] line above for the reason). Real chunks will run "
            "~50x slower.",
            flush=True,
        )
        return False

    # Prove it actually steps end-to-end on the GPU at this batch size.
    try:
        b.set_state(np.tile(iqp, (NWORLD, 1)), np.tile(iqv, (NWORLD, 1)))
        b.set_ctrl(np.zeros((NWORLD, spec["n_action_dims"])))
        b.step(spec["frame_skip"])
        b.get_state()
    except Exception as e:  # noqa: BLE001
        import traceback

        print(traceback.format_exc(), file=sys.stderr, flush=True)
        print(
            f"SELFTEST: FAIL - Warp backend built but stepping {NWORLD} worlds "
            f"of {env} raised {type(e).__name__}: {e}",
            flush=True,
        )
        return False

    # Reward correctness: WARP must agree with the CPU reference.
    try:
        cpu_ret = _baseline_return(env, spec, iqp, iqv, CpuBackend(model, 16))
        warp_ret = _baseline_return(env, spec, iqp, iqv, WarpBackend(model, 16))
    except Exception as e:  # noqa: BLE001
        import traceback

        print(traceback.format_exc(), file=sys.stderr, flush=True)
        print(
            f"SELFTEST: FAIL - reward-parity check for {env} raised "
            f"{type(e).__name__}: {e}",
            flush=True,
        )
        return False

    tol = max(1.0, 0.05 * abs(cpu_ret))
    print(
        f"SELFTEST: reward-parity env={env} CPU={cpu_ret:.3f} "
        f"WARP={warp_ret:.3f} tol={tol:.3f}",
        flush=True,
    )
    if abs(warp_ret - cpu_ret) > tol:
        print(
            f"SELFTEST: FAIL - WARP reward disagrees with CPU for {env} "
            f"(CPU={cpu_ret:.3f}, WARP={warp_ret:.3f}). The GPU steps but "
            f"computes wrong reward/termination — real chunks will be garbage.",
            flush=True,
        )
        return False

    print(
        f"SELFTEST: PASS - {env}: GPU (Warp) active on {b.device}; stepped "
        f"{NWORLD} worlds OK; reward matches CPU.",
        flush=True,
    )
    return True


envs = [sys.argv[1]] if len(sys.argv) > 1 else DEFAULT_ENVS
# Evaluate every env (don't short-circuit) so all verdicts are printed.
results = [selftest_env(e) for e in envs]
all_ok = all(results)

if all_ok:
    print(f"SELFTEST: PASS - all envs OK ({', '.join(envs)}).", flush=True)
    sys.exit(0)

print("SELFTEST: FAIL - one or more envs failed (see lines above).", flush=True)
sys.exit(1)
