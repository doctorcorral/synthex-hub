#!/usr/bin/env python3
"""
Pluggable feature-kernel registry shared by the CPU oracle
(``oracle_port``) and the GPU oracle (``warp_core``).

A *feature* is a JSON list ``[kind, *params]`` (e.g. ``["axis", 2, 0.1]``).
A *kernel* turns that spec into a boolean test of the observation. The
only difference between the two oracles is how a single observation
coordinate is accessed:

  * CPU  (one state):     ``col(i) -> state[i]``         (a scalar)
  * GPU  (a batch of N):  ``col(i) -> obs[:, i]``        (an ndarray)

numpy's arithmetic and ufuncs are polymorphic over scalars and arrays,
so each kernel body is written ONCE here and serves both paths. Callers
supply the appropriate ``col`` accessor and coerce the result
(``bool(...)`` for the scalar path; the mask is already an ndarray for
the batch path).

To add an arbitrary new feature class, register one kernel here — no
edits to either oracle's dispatch are needed.
"""

import numpy as np

FEATURE_KERNELS = {}


def feature_kernel(kind):
    """Register a kernel ``fn(feat, col) -> bool|ndarray`` under ``kind``."""

    def deco(fn):
        FEATURE_KERNELS[kind] = fn
        return fn

    return deco


# ── Linear / polynomial features ────────────────────────────────────


@feature_kernel("axis")
def _axis(feat, col):
    # obs[i] < t
    return col(feat[1]) < feat[2]


@feature_kernel("diag")
def _diag(feat, col):
    # c*obs[i] + obs[j] < 0
    return feat[3] * col(feat[1]) + col(feat[2]) < 0


@feature_kernel("sq_diag")
def _sq_diag(feat, col):
    # c*obs[i]**2 + obs[j] < 0
    return feat[3] * col(feat[1]) ** 2 + col(feat[2]) < 0


@feature_kernel("prod")
def _prod(feat, col):
    # obs[i]*obs[j] < t
    return col(feat[1]) * col(feat[2]) < feat[3]


@feature_kernel("tridiag")
def _tridiag(feat, col):
    # c1*obs[i] + c2*obs[j] + obs[k] < 0
    return feat[4] * col(feat[1]) + feat[5] * col(feat[2]) + col(feat[3]) < 0


# ── Periodic features ───────────────────────────────────────────────


@feature_kernel("sin_axis")
def _sin_axis(feat, col):
    # sin(obs[i]) < t
    return np.sin(col(feat[1])) < feat[2]


@feature_kernel("cos_axis")
def _cos_axis(feat, col):
    # cos(obs[i]) < t
    return np.cos(col(feat[1])) < feat[2]


# ── Wavelet / localized features ────────────────────────────────────
#
# These respond only inside a bounded region of state space (unlike the
# global hyperplane/periodic features above), so a single predicate can
# carve out a localized pocket without spending compositional depth.


@feature_kernel("wavelet_box")
def _wavelet_box(feat, col):
    # Haar/box indicator: TRUE when lo <= obs[i] < hi.
    # feat = ["wavelet_box", i, lo, hi]
    x = col(feat[1])
    return (x >= feat[2]) & (x < feat[3])


@feature_kernel("wavelet_ricker")
def _wavelet_ricker(feat, col):
    # Ricker / Mexican-hat bump psi((obs[i]-b)/a) < t, where
    # psi(z) = (1 - z**2) * exp(-z**2 / 2). Localized around b with
    # width set by scale a. feat = ["wavelet_ricker", i, b, a, t].
    z = (col(feat[1]) - feat[2]) / feat[3]
    zz = z * z
    return (1.0 - zz) * np.exp(-zz / 2.0) < feat[4]
