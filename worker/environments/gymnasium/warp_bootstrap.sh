#!/usr/bin/env bash
#
# One-shot bootstrap for a MuJoCo-Warp GPU worker on the Omarchy 5520
# (or any NVIDIA/CUDA Linux box). Run it FROM this directory:
#
#     cd worker/environments/gymnasium
#     ./warp_bootstrap.sh
#
# It is staged and safe: it installs deps into a local venv, runs the
# CPU correctness validations, then the real GPU smoke test, and STOPS.
# It does not touch your running worker. At the end it prints the exact
# env vars to flip the worker into a Warp worker.
#
# Re-runnable: the venv is reused if it already exists.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${WARP_VENV:-$HERE/.warp-venv}"
PY="$VENV/bin/python"

echo "=============================================================="
echo " MuJoCo-Warp worker bootstrap"
echo " dir : $HERE"
echo " venv: $VENV"
echo "=============================================================="

# ── 0. GPU / driver visibility ──────────────────────────────────────
echo
echo "[0] GPU / driver"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader || nvidia-smi || true
else
  echo "  WARNING: nvidia-smi not found. Install the NVIDIA driver first."
  echo "  On Arch/Omarchy (Maxwell M1200 is supported by the main 'nvidia' pkg):"
  echo "      sudo pacman -S nvidia nvidia-utils cuda"
  echo "  then reboot and re-run this script."
fi

# ── 1. venv + deps ──────────────────────────────────────────────────
echo
echo "[1] Python venv + dependencies"
if [ ! -x "$PY" ]; then
  python3 -m venv "$VENV"
fi
"$PY" -m pip install --upgrade pip >/dev/null
echo "  installing mujoco, gymnasium, numpy ..."
"$PY" -m pip install --upgrade mujoco gymnasium numpy >/dev/null
echo "  installing warp-lang, mujoco-warp (GPU) ..."
# These pull a CUDA runtime; the host only needs a compatible driver.
"$PY" -m pip install --upgrade warp-lang mujoco-warp
echo "  installed:"
"$PY" - <<'PY'
import importlib.metadata as m
for p in ("mujoco","gymnasium","numpy","warp-lang","mujoco-warp"):
    try: print(f"    {p}=={m.version(p)}")
    except Exception: print(f"    {p}: NOT INSTALLED")
PY

# ── 2. CPU correctness (must pass before trusting the GPU) ───────────
echo
echo "[2] CPU correctness validations"
( cd "$HERE" && "$PY" validate_warp_env.py )
( cd "$HERE" && "$PY" test_warp_oracle_parity.py )

# ── 3. GPU smoke (the real test on this hardware) ───────────────────
echo
echo "[3] GPU smoke test (CPU-vs-Warp agreement + throughput)"
set +e
( cd "$HERE" && "$PY" warp_gpu_smoke.py )
SMOKE=$?
set -e

echo
echo "=============================================================="
if [ "$SMOKE" -eq 0 ]; then
  cat <<EOF
 GPU path is working on this box. To turn the worker into a Warp
 worker, point it at the venv python + the warp oracle and advertise
 the capability, then restart your worker process:

     export PYTHON="$PY"
     export ORACLE_SCRIPT="$HERE/oracle_warp_port.py"
     export WORKER_CAPABILITIES="mujoco_warp,mujoco"   # prefer warp, fall back
     export ORACLE_BACKEND="warp"
     # keep your existing WORKER_NAME / API_TOKEN / SERVER_URL

 Then submit a Warp experiment with:
     "env_name": "HalfCheetah-warp-v5", "adapter": "mujoco_warp"

 (To make it Warp-only with no CPU fallback, use
  WORKER_CAPABILITIES="mujoco_warp".)
EOF
else
  cat <<EOF
 GPU smoke did NOT pass (exit $SMOKE). The CPU path is fine, but the
 Warp backend either can't see the GPU or disagreed with CPU. Common
 causes on a Maxwell M1200:
   - NVIDIA driver / CUDA not installed or too new/old for this GPU
   - compute capability 5.0 unsupported by the installed warp-lang
 Scroll up for the smoke output and share it; we'll diagnose from there.
 Meanwhile the worker can keep running as a plain mujoco (CPU) worker.
EOF
fi
echo "=============================================================="
