#!/usr/bin/env python3
"""
Fetch the latest policy snapshot for a MuJoCo env from a Synthex Hub
and render it as an MP4. Works against any hub that exposes the
`/api/public-status/policies/:env_name` endpoint (default
https://synthex.fit/api).

Usage:

    python scripts/record_from_snapshot.py Ant-v5
    python scripts/record_from_snapshot.py Humanoid-v5 \\
        --seeds 0,1,2,42 --output-dir humanoid_snapshot_videos

No auth required: the snapshot endpoint is public. The script
materializes the snapshot's `policy_code` string into a callable
Python `policy(obs) -> actions` function and runs it through
Gymnasium with `RecordVideo`.

The snapshot is whatever the master last pushed via
`Synthex.Hub.Client.push_policy_snapshot/2` — typically the
latest accepted CEGAR bit. Re-running this script as the
experiment progresses produces a chronological "policy through
time" record without having to coordinate with the master.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import json
import urllib.request
import urllib.error

import numpy as np
import gymnasium as gym


DEFAULT_HUB = os.getenv("SYNTHEX_HUB_URL", "https://synthex.fit/api")


def fetch_snapshot(hub_url: str, env_name: str) -> dict:
    url = f"{hub_url.rstrip('/')}/public-status/policies/{env_name}"
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            sys.exit(
                f"no snapshot for {env_name!r} at {hub_url}. "
                "the master may not have pushed one yet."
            )
        raise


def compile_policy(code: str):
    """Materialize the snapshot's `policy_code` into a callable.

    The generated code defines a top-level `policy(obs)` function;
    we exec into a fresh namespace and return that reference.
    """
    ns: dict = {}
    exec(code, ns)
    if "policy" not in ns or not callable(ns["policy"]):
        sys.exit("snapshot's policy_code did not define a callable `policy(obs)`")
    return ns["policy"]


def classify(env_name: str, reward: float, terminated: bool) -> str:
    if env_name.startswith("Humanoid") or env_name.startswith("Ant"):
        if terminated:
            return f"fell_r{int(reward)}"
        if reward > 500:
            return f"walked_r{int(reward)}"
        return f"survived_r{int(reward)}"
    return f"r{int(reward)}"


def record_one(env_name: str, policy, seed: int, out_dir: str, max_steps: int) -> tuple[float, int, bool]:
    base = gym.make(env_name, render_mode="rgb_array")
    name_prefix = f"{env_name.lower().replace('-', '_')}_snapshot_seed{seed}"
    env = gym.wrappers.RecordVideo(
        base,
        video_folder=out_dir,
        episode_trigger=lambda _i: True,
        name_prefix=name_prefix,
    )

    obs, _ = env.reset(seed=seed)
    total_r = 0.0
    terminated = False
    truncated = False
    steps = 0
    for _ in range(max_steps):
        action = np.asarray(policy(list(obs)), dtype=np.float32)
        # Clip to the action space — defensive, the generated policy
        # already targets the env's [lo, hi] but float math can
        # drift outside by ~eps.
        action = np.clip(action, env.action_space.low, env.action_space.high)
        obs, r, terminated, truncated, _ = env.step(action)
        total_r += float(r)
        steps += 1
        if terminated or truncated:
            break
    env.close()
    return total_r, steps, terminated


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("env", help="Gymnasium env name, e.g. Ant-v5")
    ap.add_argument("--hub", default=DEFAULT_HUB, help=f"hub base URL (default: {DEFAULT_HUB})")
    ap.add_argument("--seeds", default="0,1,2",
                    help="comma-separated episode seeds (default: 0,1,2)")
    ap.add_argument("--output-dir", default=None,
                    help="where to write MP4s (default: <env>_snapshot_videos)")
    ap.add_argument("--max-steps", type=int, default=1000,
                    help="max steps per episode (default: 1000)")
    ap.add_argument("--show-code", action="store_true",
                    help="print the snapshot's policy_code before rolling out")
    args = ap.parse_args()

    snap = fetch_snapshot(args.hub, args.env)
    code = snap.get("policy_code")
    if not code:
        sys.exit(f"snapshot for {args.env} has no policy_code yet")

    print(f"snapshot: env={args.env}")
    print(f"  CEGAR round = {snap.get('cegar_iter')}, iter = {snap.get('iter')}")
    print(f"  target_bit  = {snap.get('target_bit')}/{snap.get('n_bits')}")
    print(f"  best_reward = {snap.get('best_reward')}")
    print(f"  submitter   = {snap.get('submitter')}")
    print(f"  updated_at  = {snap.get('updated_at')}")
    if args.show_code:
        print("\n---- policy_code ----")
        print(code)
        print("---------------------\n")

    policy = compile_policy(code)
    out_dir = args.output_dir or f"{args.env.lower().replace('-', '_')}_snapshot_videos"
    os.makedirs(out_dir, exist_ok=True)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    print(f"rolling out {len(seeds)} episode(s), writing to {out_dir}/")
    t0 = time.time()
    for seed in seeds:
        reward, steps, terminated = record_one(
            args.env, policy, seed, out_dir, args.max_steps
        )
        label = classify(args.env, reward, terminated)
        print(f"  seed={seed:>4d}  reward={reward:>+8.1f}  steps={steps:>4d}  {label}")
    print(f"done in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
