#!/usr/bin/env python3
"""Unit tests for Agda-faithful successor solve traces."""

import unittest

from successor_solve import max_feasible_horizon, trace_advantage, trace_le, trace_lex_score


class TraceOrderTests(unittest.TestCase):
    def test_lexicographic_order(self):
        self.assertTrue(trace_le([1.0, 0.0], [2.0, 0.0]))
        self.assertFalse(trace_le([2.0, 0.0], [1.0, 0.0]))
        self.assertTrue(trace_le([1.0, 0.0], [1.0, 5.0]))
        self.assertFalse(trace_le([1.0, 5.0], [1.0, 0.0]))

    def test_scalar_preserves_lex_order(self):
        t_better = [1.0, 0.0]
        t_worse = [0.5, 100.0]
        self.assertGreater(trace_advantage(t_worse, t_better), 0.0)

    def test_horizon_clamp(self):
        # 64 actions: 64 + 64^2 + 64^3 = 266_304 <= 1M, +64^4 overflows -> k=3
        self.assertEqual(max_feasible_horizon(64, 10, 1_000_000), 3)
        # 729 actions: 729 + 729^2 = 532_170 <= 1M, +729^3 overflows -> k=2
        self.assertEqual(max_feasible_horizon(729, 10, 1_000_000), 2)
        # Requested horizon below the budget cap passes through unchanged
        self.assertEqual(max_feasible_horizon(64, 2, 1_000_000), 2)
        # Never returns less than 1 even on a tiny budget
        self.assertEqual(max_feasible_horizon(4096, 5, 100), 1)


if __name__ == "__main__":
    unittest.main()
