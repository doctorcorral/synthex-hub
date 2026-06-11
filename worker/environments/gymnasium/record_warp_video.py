#!/usr/bin/env python3
"""
Capture a Synthex bit-policy rollout AS IT ACTUALLY RAN IN WARP.

MuJoCo Warp has no renderer, but a pose (`qpos`) fully determines the
picture. So the honest way to "film Warp" is:

  1. roll the policy out on the real `WarpBackend` (the SAME GPU physics
     + reward/termination accounting the hub used to score it, copied
     from oracle_warp_port.rollout),
  2. capture `qpos` at every control step,
  3. replay those exact Warp states through `mujoco.Renderer`
     (plain MuJoCo just drawing poses — no dynamics, no GPU needed).

This step does (1) and (2): it runs the rollout on the GPU and writes a
small trace file. It needs only numpy + the worker's existing deps, so
it runs in the current worker image with NO rebuild. Rendering (3) is
done separately by scripts/render_warp_trace.py, which can run anywhere
(e.g. your Mac) since it just draws the saved poses.

Run it inside the GPU worker container:

    docker exec -w /app/environments/gymnasium synthex-worker-gpu \
        python3 record_warp_video.py --out /app/warp_trace.npz

then copy the (tiny) trace out and render it locally:

    docker cp synthex-worker-gpu:/app/warp_trace.npz ./warp_trace.npz
    python3 scripts/render_warp_trace.py warp_trace.npz

Pass --render to also produce the mp4 inside the container (requires
imageio in the image; falls back to trace-only if unavailable).

Defaults to the InvertedDoublePendulum-warp-v5 v15 policy (best=11193.6).
Override with --policy policy.json (a JSON list of bit predicates, same
shape as a score_bit job's `bit_predicates`).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

import warp_core as core
from warp_backends import make_backend


DEFAULT_ENV = "InvertedDoublePendulum-warp-v5"
DEFAULT_BITS_PER_DIM = 4
# InvertedDoublePendulum-warp-v5, lineage v15 (exp 635dabe5, best 11193.6).
DEFAULT_PREDS = [
    ["and", ["feat", ["axis", 2, -0.1]], ["feat", ["axis", 2, -0.110559]]],
    ["or", ["feat", ["axis", 2, -0.03]], ["feat", ["tridiag", 4, 3, 2, 2, -2]]],
    ["or", ["feat", ["axis", 2, 0.11]], ["feat", ["axis", 2, 0.13]]],
    ["or",
     ["feat", ["tridiag", 2, 3, 7, 2, 1]],
     ["feat", ["tridiag", 7, 4, 6, 2, 1]]],
]


def _read_state(b, needs):
    qpos, qvel = b.get_state()
    extras = b.read_fields(needs) if needs else {}
    return {"qpos": qpos, "qvel": qvel, "extras": extras}


def rollout_capture(env_name, seeds, preds, max_steps, bits_per_dim):
    """Batched Warp rollout that ALSO records the per-step qpos trace.

    Mirrors oracle_warp_port.rollout's return/termination accounting
    exactly (so reported returns match the hub) and additionally keeps a
    (T, N, nq) qpos trace + per-step (T, N) tip-height and done masks.
    """
    spec = core.ENV_SPECS[env_name]
    model, iqp, iqv = core.load_mj_model(env_name)
    n = len(seeds)
    fs = spec["frame_skip"]
    needs = spec.get("needs", [])
    n_bits = spec["n_action_dims"] * bits_per_dim

    bit_preds_per_world = [
        preds[b] if b < len(preds) else "truep" for b in range(n_bits)
    ]

    qpos, qvel = core.reset_states(env_name, seeds, iqp, iqv)
    b = make_backend(model, n,
                     prefer=os.environ.get("ORACLE_BACKEND", "auto"),
                     use_graph=os.environ.get("ORACLE_WARP_GRAPH", "1")
                     not in ("0", "false", "False"))
    b.set_state(qpos, qvel)
    print(f"[record] backend={b.name} nworld={n}", file=sys.stderr)

    state = _read_state(b, needs)
    obs = spec["obs_fn"](state)

    ep_return = np.zeros(n, dtype=np.float64)
    done = np.zeros(n, dtype=bool)

    def tip_y(st):
        sx = st["extras"].get("site_xpos")
        return sx[:, 0, 2] if sx is not None else np.full(n, np.nan)

    qpos_trace = [state["qpos"].copy()]   # includes the reset pose
    done_trace = [done.copy()]
    y_trace = [tip_y(state)]

    for _ in range(max_steps):
        bits = core.policy_bits(obs, bit_preds_per_world, n_bits)
        actions = core.decode_actions(bits, spec, bits_per_dim)

        prev_state = state
        b.set_ctrl(actions)
        b.step(fs)
        state = _read_state(b, needs)

        step_r = spec["reward_fn"](prev_state, state, actions)
        ep_return += np.where(done, 0.0, step_r)

        obs = spec["obs_fn"](state)
        done = done | spec["terminated_fn"](state)

        qpos_trace.append(state["qpos"].copy())
        done_trace.append(done.copy())
        y_trace.append(tip_y(state))

        if done.all():
            break

    success = ep_return > spec["success_threshold"]
    return {
        "ep_return": ep_return,
        "success": success,
        "qpos_trace": np.array(qpos_trace),   # (T, N, nq)
        "done_trace": np.array(done_trace),   # (T, N)
        "y_trace": np.array(y_trace),         # (T, N)
        "dt": spec["dt"],
        "base_env": spec["base_env"],
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", default=DEFAULT_ENV)
    ap.add_argument("--policy", help="JSON file: list of bit predicates")
    ap.add_argument("--bits-per-dim", type=int, default=DEFAULT_BITS_PER_DIM)
    ap.add_argument("--seeds", default="0-19",
                    help="seeds to evaluate, e.g. '0-19' or '3,8,11'")
    ap.add_argument("--top", type=int, default=3,
                    help="how many top seeds' traces to save/film")
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--out", default="warp_trace.npz",
                    help="trace output (.npz)")
    ap.add_argument("--render", action="store_true",
                    help="also write mp4 in-container (needs imageio)")
    ap.add_argument("--fps", type=int, default=20)
    args = ap.parse_args()

    if not core.is_warp_env(args.env):
        sys.exit(f"{args.env} is not a warp env; known={list(core.ENV_SPECS)}")

    preds = DEFAULT_PREDS
    if args.policy:
        with open(args.policy) as f:
            preds = json.load(f)
    elif args.env == DEFAULT_ENV:
        print("using built-in IDP v15 policy (4 bits, best=11193.6)")

    if "-" in args.seeds and "," not in args.seeds:
        a, z = args.seeds.split("-")
        seeds = list(range(int(a), int(z) + 1))
    else:
        seeds = [int(s) for s in args.seeds.split(",")]

    print(f"rolling out {len(seeds)} seeds on {args.env} (Warp)...")
    out = rollout_capture(args.env, seeds, preds, args.max_steps, args.bits_per_dim)
    ep, succ = out["ep_return"], out["success"]
    done_trace, y_trace = out["done_trace"], out["y_trace"]

    term_step = []
    for i in range(len(seeds)):
        d = np.where(done_trace[:, i])[0]
        term_step.append(int(d[0]) if d.size else -1)

    order = sorted(range(len(seeds)), key=lambda i: -ep[i])
    print(f"\n{'seed':>5} {'return':>10} {'steps_survived':>15} {'success':>8}")
    for i in order:
        survived = args.max_steps if term_step[i] < 0 else term_step[i]
        print(f"{seeds[i]:>5} {ep[i]:>10.2f} {survived:>15} {str(bool(succ[i])):>8}")
    print(f"\nmean return = {ep.mean():.2f} over {len(seeds)} seeds")

    best = order[0]
    if not np.all(np.isnan(y_trace[:, best])):
        last = term_step[best] + 1 if term_step[best] >= 0 else None
        ys = y_trace[:last, best]
        print(f"\nbest seed {seeds[best]} tip height y: min={np.nanmin(ys):.3f} "
              f"max={np.nanmax(ys):.3f} mean={np.nanmean(ys):.3f} "
              f"(termination fires at y<=1.0 — y>1 throughout = genuine balance)")

    # Save the top-K traces (trimmed to each world's terminating frame).
    top = order[: args.top]
    nq = out["qpos_trace"].shape[2]
    T = out["qpos_trace"].shape[0]
    lengths = np.array([(term_step[i] + 1) if term_step[i] >= 0 else T for i in top])
    qpos_top = np.stack([out["qpos_trace"][:, i, :] for i in top])  # (K, T, nq)

    np.savez_compressed(
        args.out,
        env=args.env,
        base_env=out["base_env"],
        dt=out["dt"],
        fps=args.fps,
        seeds=np.array([seeds[i] for i in top]),
        returns=np.array([ep[i] for i in top]),
        lengths=lengths,
        qpos=qpos_top,   # (K, T, nq); render only the first `lengths[k]` frames
    )
    print(f"\nsaved trace -> {args.out} "
          f"(top {len(top)} seeds, {nq} dof, up to {T} frames each)")
    print("render locally with:  python3 scripts/render_warp_trace.py "
          f"{os.path.basename(args.out)}")

    if args.render:
        try:
            _render_inline(args, out, top, term_step, seeds, ep)
        except Exception as e:  # imageio / GL not in image — trace still saved
            print(f"[record] inline render skipped ({type(e).__name__}: {e}); "
                  "use scripts/render_warp_trace.py on the saved trace",
                  file=sys.stderr)


def _render_inline(args, out, top, term_step, seeds, ep):
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    import imageio
    import mujoco

    model, _, _ = core.load_mj_model(args.env)
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), 640)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), 480)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=640)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = 0
    cam.distance, cam.elevation, cam.azimuth = 3.5, -12, 90

    out_dir = os.path.dirname(args.out) or "."
    T = out["qpos_trace"].shape[0]
    for i in top:
        last = term_step[i] + 1 if term_step[i] >= 0 else T
        frames = []
        for t in range(last):
            data.qpos[:] = out["qpos_trace"][t, i, :]
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render())
        tag = "survived" if term_step[i] < 0 else f"fell{term_step[i]}"
        path = os.path.join(out_dir, f"warp_seed{seeds[i]}_{tag}_r{int(ep[i])}.mp4")
        imageio.mimsave(path, frames, fps=args.fps)
        print(f"recorded -> {path} ({len(frames)} frames)")
    renderer.close()


if __name__ == "__main__":
    main()
