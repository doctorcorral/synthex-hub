#!/usr/bin/env python3
"""Regression test: rollout lookahead must not inherit TimeLimit state.

`snapshot_advantage_rollout` scores two counterfactual branches (v0, v1)
on ONE env instance. Stepping the wrapped env would let the TimeLimit
wrapper accumulate steps across branches, so with full-horizon lookaheads
(lookahead >= max_episode_steps) the second branch starts already
truncated and scores garbage. `_lookahead_value` therefore steps the
unwrapped env; this test pins that behaviour with a tiny time limit.

Reacher is used because it never terminates naturally, so branch length
is fully determined by the lookahead (not by a fall), and its per-step
rewards are strictly negative, so longer branches are strictly more
negative — an observable proxy for "the rollout ran past the wrapper cap".
"""

import unittest

import gymnasium as gym
import numpy as np

from oracle_port import _lookahead_value


class RolloutLookaheadTests(unittest.TestCase):
    def test_branches_not_truncated_by_shared_time_limit(self):
        env = gym.make("Reacher-v5", max_episode_steps=5)
        try:
            env.reset(seed=0)
            qpos = env.unwrapped.data.qpos.copy()
            qvel = env.unwrapped.data.qvel.copy()

            cfg = {"n_action_dims": 2, "action_low": -1.0, "action_high": 1.0}
            preds = ["falsep"] * 6  # constant action, deterministic branch
            action = np.array([0.0, 0.0])

            # Two identical branches from the same snapshot must score
            # identically. With wrapped stepping, the first call would
            # exhaust the 5-step TimeLimit and the second would truncate
            # after a single step (v1 != v0).
            v_full_a = _lookahead_value(env, qpos, qvel, action, preds, 20, cfg, 3)
            v_full_b = _lookahead_value(env, qpos, qvel, action, preds, 20, cfg, 3)
            self.assertAlmostEqual(v_full_a, v_full_b, places=9)

            # And the branch must actually run past the 5-step wrapper cap:
            # Reacher pays a strictly negative reward every step, so a
            # 20-step branch is strictly more negative than a 5-step one.
            v_short = _lookahead_value(env, qpos, qvel, action, preds, 5, cfg, 3)
            self.assertLess(v_full_a, v_short)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
