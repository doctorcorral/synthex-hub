"""Agda-faithful successor traces for CoinductiveHomomorphism scoring.

Implements SuccessorDeterministicMDP's finder:

  successor-trace s a k = value-trace (next s a) k
  value-trace s k       = [solve(s,0), solve(s,1), ..., solve(s,k-1)]

where solve(s,0) = max_a r(s,a) and solve(s,n+1) = max_a solve(next(s,a), n).

Traces are compared lexicographically (Agda ``_≤ₜ_``). For predicate scoring we
map a trace to a scalar with a base-``M`` positional encoding so lex order is
preserved when ``M`` exceeds the reward dynamic range.

This replaces the previous policy-rollout surrogate
``sum of rewards over lookahead steps under the current bit-policy``.
"""

from __future__ import annotations

import itertools
from functools import lru_cache
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


def _state_key(qpos: np.ndarray, qvel: np.ndarray, decimals: int = 3) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    return (
        tuple(np.round(qpos, decimals).tolist()),
        tuple(np.round(qvel, decimals).tolist()),
    )


def _action_from_bits(bits, cfg, bits_per_dim):
    weights = [2 ** i for i in range(bits_per_dim)]
    max_sum = sum(weights)
    n = int(cfg["n_action_dims"])
    lo, hi = float(cfg["action_low"]), float(cfg["action_high"])
    actions = np.zeros(n, dtype=np.float64)
    for d in range(n):
        s = sum(weights[i] * bits[d * bits_per_dim + i] for i in range(bits_per_dim))
        actions[d] = lo + (hi - lo) * s / max_sum
    return actions


def build_action_grid(cfg: dict, bits_per_dim: int, grid_levels) -> List[np.ndarray]:
    """Discrete action set for the solve operator.

    ``grid_levels``:
      * int >= 2 — ``grid_levels`` evenly spaced values per actuator (Cartesian product)
      * ``"full_bits"`` — all ``2^(n_dims * bits_per_dim)`` bit-decoded actions
    """
    n = int(cfg["n_action_dims"])
    lo, hi = float(cfg["action_low"]), float(cfg["action_high"])

    if grid_levels == "full_bits":
        n_bits = n * bits_per_dim
        return [
            _action_from_bits([(code >> i) & 1 for i in range(n_bits)], cfg, bits_per_dim)
            for code in range(2 ** n_bits)
        ]

    levels = int(grid_levels)
    if levels < 2:
        levels = 2
    vals = np.linspace(lo, hi, levels, dtype=np.float64)
    return [np.array(combo, dtype=np.float64) for combo in itertools.product(vals, repeat=n)]


def trace_le(trace_a: Sequence[float], trace_b: Sequence[float]) -> bool:
    """Agda ``_≤ₜ_``: True when trace_a is dominated by trace_b (b is at least as good)."""
    i = 0
    while i < len(trace_a) and i < len(trace_b):
        ra, rb = trace_a[i], trace_b[i]
        if ra < rb:
            return True
        if ra > rb:
            return False
        i += 1
    return i >= len(trace_a)


def trace_lex_score(trace: Sequence[float], reward_ceiling: float = 1000.0) -> float:
    """Lexicographic-max scalarization: earlier positions dominate later ones."""
    base = 2.0 * max(float(reward_ceiling), 1.0) + 1.0
    score = 0.0
    mult = 1.0
    for r in reversed(trace):
        score += float(r) * mult
        mult *= base
    return score


def trace_advantage(trace0: Sequence[float], trace1: Sequence[float], reward_ceiling: float = 1000.0) -> float:
    """Signed gap for predicate scoring: strict lex dominance, else trace-mass diff."""
    if list(trace0) == list(trace1):
        return 0.0
    if trace_le(trace0, trace1) and not trace_le(trace1, trace0):
        return 1.0
    if trace_le(trace1, trace0) and not trace_le(trace0, trace1):
        return -1.0
    return float(sum(trace1) - sum(trace0))


def max_feasible_horizon(n_actions: int, horizon: int, step_budget: int) -> int:
    """Largest trace length k whose exact solve cost fits in ``step_budget``.

    solve(s, d) recurses over the full action grid: computing depth d costs
    ``n_actions^(d+1)`` sim steps (the state cache almost never hits on
    continuous states), so a k-trace costs ``sum_{d<k} n_actions^(d+1)``.
    Without this clamp a request like lookahead=50 on a 729-action grid is
    ~729^50 steps — the job would simply never return.
    """
    cost = 0
    k = 0
    while k < horizon:
        cost += n_actions ** (k + 1)
        if cost > step_budget:
            break
        k += 1
    return max(k, 1)


class SolveEngine:
    """Memoized finite-horizon solve on a single MuJoCo env instance."""

    __slots__ = ("env", "raw", "action_grid", "bottom", "decimals", "step_budget", "_cache")

    def __init__(
        self,
        env,
        action_grid: List[np.ndarray],
        *,
        bottom: float = 0.0,
        state_decimals: int = 3,
        step_budget: int = 1_000_000,
    ):
        self.env = env
        # Step through the UNWRAPPED env. gym.make wraps in TimeLimit whose
        # step counter accumulates across counterfactual branches (set_state
        # does not reset it), so after max_episode_steps cumulative steps
        # every branch reports truncated=True and solve sees bottom
        # everywhere — silently zeroing all advantages. The raw MuJoCo env
        # has no step budget; genuine termination (falls) still surfaces
        # via `terminated`.
        self.raw = env.unwrapped
        self.action_grid = action_grid
        self.bottom = float(bottom)
        self.decimals = int(state_decimals)
        self.step_budget = int(step_budget)
        self._cache: Dict[Tuple[Tuple[float, ...], Tuple[float, ...], int], float] = {}

    def _step(self, qpos: np.ndarray, qvel: np.ndarray, action: np.ndarray):
        self.raw.set_state(qpos, qvel)
        obs, reward, term, trunc, _ = self.raw.step(action)
        nq = self.raw.data.qpos.copy()
        nv = self.raw.data.qvel.copy()
        return nq, nv, obs, float(reward), bool(term or trunc)

    def _immediate_best(self, qpos: np.ndarray, qvel: np.ndarray) -> float:
        best = self.bottom
        for action in self.action_grid:
            _, _, _, reward, _ = self._step(qpos, qvel, action)
            if reward > best:
                best = reward
        return best

    def solve(self, qpos: np.ndarray, qvel: np.ndarray, depth: int) -> float:
        if depth <= 0:
            return self._immediate_best(qpos, qvel)

        key = (*_state_key(qpos, qvel, self.decimals), depth)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        best = self.bottom
        for action in self.action_grid:
            nq, nv, _, _, done = self._step(qpos, qvel, action)
            if done:
                val = self.bottom
            else:
                val = self.solve(nq, nv, depth - 1)
            if val > best:
                best = val

        self._cache[key] = best
        return best

    def _clamped_k(self, horizon: int) -> int:
        return max_feasible_horizon(len(self.action_grid), max(int(horizon), 0), self.step_budget)

    def value_trace(self, qpos: np.ndarray, qvel: np.ndarray, horizon: int) -> List[float]:
        k = self._clamped_k(horizon)
        return [self.solve(qpos, qvel, d) for d in range(k)]

    def successor_trace(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        branch_action: np.ndarray,
        horizon: int,
    ) -> List[float]:
        """``successor-trace s a k`` — trace from ``next(s,a)``, excluding ``r(s,a)``."""
        k = self._clamped_k(horizon)
        nq, nv, _, _, done = self._step(qpos, qvel, branch_action)
        if done:
            # Dead branch: bottom at every depth, same length as the live
            # branch so lex comparison stays positional.
            return [self.bottom] * k
        return [self.solve(nq, nv, d) for d in range(k)]


def snapshot_trace_advantage(
    env_name: str,
    cfg: dict,
    snapshot: dict,
    bit_preds,
    target_bit: int,
    horizon: int,
    bits_per_dim: int,
    *,
    grid_levels=3,
    reward_ceiling: float = 1000.0,
    solve_step_budget: int = 1_000_000,
    eval_pred_fn,
    action_from_bits_fn,
) -> float:
    """Compute A(s) from solve-based successor traces for one snapshot."""
    import gymnasium as gym

    obs = snapshot["obs"]
    qpos = np.array(snapshot.get("qpos"), dtype=np.float64)
    qvel = np.array(snapshot.get("qvel"), dtype=np.float64)
    if qpos.size == 0 or qvel.size == 0:
        return 0.0

    base_bits = [1 if eval_pred_fn(p, obs) else 0 for p in bit_preds]
    b0, b1 = list(base_bits), list(base_bits)
    b0[target_bit] = 0
    b1[target_bit] = 1
    a0 = action_from_bits_fn(b0, cfg, bits_per_dim)
    a1 = action_from_bits_fn(b1, cfg, bits_per_dim)

    env_kwargs = cfg.get("env_kwargs", {})
    env = gym.make(env_name, **env_kwargs)
    try:
        env.reset()
        grid = build_action_grid(cfg, bits_per_dim, grid_levels)
        engine = SolveEngine(env, grid, step_budget=solve_step_budget)
        t0 = engine.successor_trace(qpos, qvel, a0, horizon)
        t1 = engine.successor_trace(qpos, qvel, a1, horizon)
        return trace_advantage(t0, t1, reward_ceiling)
    finally:
        env.close()
