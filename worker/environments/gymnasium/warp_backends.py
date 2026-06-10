#!/usr/bin/env python3
"""
Physics-stepping backends for the Warp Synthex environments.

Both expose the same tiny interface, consumed by `oracle_warp_port.py`:

    b = make_backend(model, nworld)      # or CpuBackend / WarpBackend
    b.set_state(qpos, qvel)              # (N, nq), (N, nv)
    b.set_ctrl(ctrl)                     # (N, nu)
    b.step(frame_skip)                   # advance all N worlds
    qpos, qvel = b.get_state()           # (N, nq), (N, nv)

`CpuBackend` (plain `mujoco`) runs anywhere and is the correctness
reference + fallback. `WarpBackend` (`mujoco_warp`) batches all N
worlds on an NVIDIA GPU and is the throughput path. They step the
identical compiled model, so a Warp rollout and a CPU rollout of the
same policy from the same seed agree up to solver numerics.
"""

from __future__ import annotations

import logging
import sys
import traceback

import numpy as np

# Inherits the oracle's logging config (same process), so these land in
# ORACLE_LOG (/tmp/synthex_hub_warp_worker.log) alongside the oracle's
# own lines.
_log = logging.getLogger("warp_backends")

# Deduped one-line verdict, printed to STDERR so it ALWAYS shows up in
# `docker logs <container>` regardless of the file-logging config — this
# is the canonical place to confirm GPU vs CPU-fallback.
_announced = set()


def _verdict(msg):
    if msg not in _announced:
        _announced.add(msg)
        print(f"[warp-backend] {msg}", file=sys.stderr, flush=True)


# ── CPU backend (plain mujoco) ──────────────────────────────────────


class CpuBackend:
    name = "cpu"

    def __init__(self, model, nworld):
        import mujoco

        self._mj = mujoco
        self.model = model
        self.nworld = nworld
        # One MjData per world. Plenty for the modest batch sizes the
        # CPU path handles (it's the reference/fallback, not the
        # high-throughput lane).
        self.datas = [mujoco.MjData(model) for _ in range(nworld)]

    def set_state(self, qpos, qvel):
        for i, d in enumerate(self.datas):
            d.qpos[:] = qpos[i]
            d.qvel[:] = qvel[i]
            self._mj.mj_forward(self.model, d)

    def set_ctrl(self, ctrl):
        for i, d in enumerate(self.datas):
            d.ctrl[:] = ctrl[i]

    def step(self, frame_skip):
        for d in self.datas:
            self._mj.mj_step(self.model, d, nstep=frame_skip)

    def get_state(self):
        qpos = np.array([d.qpos for d in self.datas], dtype=np.float64)
        qvel = np.array([d.qvel for d in self.datas], dtype=np.float64)
        return qpos, qvel

    def read_fields(self, names):
        # Batch arbitrary MjData fields across worlds, e.g. "site_xpos"
        # (N, nsite, 3) or "qfrc_constraint" (N, nv). Used by env specs
        # whose reward/obs depend on derived quantities not in qpos/qvel.
        out = {}
        for nm in names:
            out[nm] = np.array([getattr(d, nm) for d in self.datas], dtype=np.float64)
        return out


# ── Warp backend (mujoco_warp, batched on GPU) ──────────────────────


class WarpBackend:
    name = "warp"

    def __init__(self, model, nworld, device="cuda", use_graph=True):
        import warp as wp
        import mujoco_warp as mjw

        self._wp = wp
        self._mjw = mjw
        self.device = device
        self.nworld = nworld
        self.use_graph = use_graph
        self._graph = None
        self._graph_frame_skip = None

        with wp.ScopedDevice(device):
            self.m = mjw.put_model(model)
            self.d = mjw.make_data(model, nworld=nworld)

    def _assign(self, field, arr):
        # Copy a (N, k) numpy block into a leading-batched warp field,
        # matching the field's dtype to avoid an implicit cast inside
        # the kernels.
        wp = self._wp
        src = wp.array(np.ascontiguousarray(arr), dtype=field.dtype, device=self.device)
        wp.copy(field, src)

    def set_state(self, qpos, qvel):
        wp = self._wp
        mjw = self._mjw
        with wp.ScopedDevice(self.device):
            self._assign(self.d.qpos, qpos)
            self._assign(self.d.qvel, qvel)
            # Recompute derived quantities (xpos, contacts, ...) from
            # the freshly-set state, exactly as mj_forward does on the
            # CPU side after a reset.
            mjw.forward(self.m, self.d)

    def set_ctrl(self, ctrl):
        wp = self._wp
        with wp.ScopedDevice(self.device):
            self._assign(self.d.ctrl, ctrl)

    def step(self, frame_skip):
        wp = self._wp
        mjw = self._mjw
        with wp.ScopedDevice(self.device):
            if self.use_graph and self.device.startswith("cuda"):
                self._step_graph(frame_skip)
            else:
                for _ in range(frame_skip):
                    mjw.step(self.m, self.d)

    def _step_graph(self, frame_skip):
        wp = self._wp
        mjw = self._mjw
        if self._graph is None or self._graph_frame_skip != frame_skip:
            # Warm up once so any lazy allocation happens outside the
            # capture, then record `frame_skip` steps into a CUDA graph.
            # d.ctrl lives in device memory and is read by the captured
            # kernels, so updating it between launches (set_ctrl) feeds
            # new actions without re-capturing.
            mjw.step(self.m, self.d)
            with wp.ScopedCapture() as cap:
                for _ in range(frame_skip):
                    mjw.step(self.m, self.d)
            self._graph = cap.graph
            self._graph_frame_skip = frame_skip
        wp.capture_launch(self._graph)

    def get_state(self):
        with self._wp.ScopedDevice(self.device):
            qpos = self.d.qpos.numpy().astype(np.float64)
            qvel = self.d.qvel.numpy().astype(np.float64)
        return qpos, qvel

    def read_fields(self, names):
        # Same contract as CpuBackend.read_fields, reading the batched
        # mujoco_warp Data arrays (leading world axis) back to host.
        # mujoco_warp mirrors MjData field names, so "site_xpos" /
        # "qfrc_constraint" resolve directly.
        out = {}
        with self._wp.ScopedDevice(self.device):
            for nm in names:
                out[nm] = getattr(self.d, nm).numpy().astype(np.float64)
        return out


# ── Backend selection ───────────────────────────────────────────────


def warp_available():
    """True iff mujoco_warp + a usable CUDA device are importable."""
    try:
        import warp as wp
        import mujoco_warp  # noqa: F401

        wp.init()
        return any(d.is_cuda for d in wp.get_devices())
    except Exception:
        return False


def make_backend(model, nworld, prefer="auto", use_graph=True):
    """Return a stepping backend.

    prefer: "warp" (force GPU, error if unavailable),
            "cpu"  (force CPU),
            "auto" (GPU when available, else CPU).
    """
    if prefer == "cpu":
        _verdict(f"CPU (forced ORACLE_BACKEND=cpu), nworld={nworld}")
        return CpuBackend(model, nworld)
    if prefer == "warp":
        # Forced GPU: do NOT swallow — let the real error surface so a
        # misconfigured "warp" worker fails loudly instead of silently
        # grinding on the CPU.
        b = WarpBackend(model, nworld, use_graph=use_graph)
        _verdict(f"WARP/GPU (forced), nworld={nworld}, device={b.device}")
        return b
    # auto
    if not warp_available():
        msg = (
            f"CPU FALLBACK, nworld={nworld}: warp_available()=False — no CUDA "
            f"device visible to Warp (~50x slower)."
        )
        _verdict(msg)
        _log.warning("make_backend(auto): %s", msg)
        return CpuBackend(model, nworld)
    try:
        b = WarpBackend(model, nworld, use_graph=use_graph)
        _verdict(f"WARP/GPU built OK, nworld={nworld}, device={b.device}")
        return b
    except Exception as e:
        # The silent version of this fallback is exactly what masked a
        # GPU worker quietly running every chunk on a single CPU core
        # (correct, but ~50x slower). Surface the reason on stderr (docker
        # logs) AND the full traceback in ORACLE_LOG.
        _verdict(
            f"CPU FALLBACK, nworld={nworld}: WarpBackend build FAILED "
            f"({type(e).__name__}: {e}) — ~50x slower. Full traceback in "
            f"ORACLE_LOG (/tmp/synthex_hub_warp_worker.log)."
        )
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        _log.exception(
            "make_backend(auto): WarpBackend(nworld=%d) construction FAILED; "
            "falling back to CPU backend (this chunk will run ~50x slower).",
            nworld,
        )
        return CpuBackend(model, nworld)
