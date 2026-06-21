#!/usr/bin/env python3
"""Unit tests for Agda-faithful successor solve traces."""

import unittest

from successor_solve import trace_advantage, trace_le, trace_lex_score


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


if __name__ == "__main__":
    unittest.main()
