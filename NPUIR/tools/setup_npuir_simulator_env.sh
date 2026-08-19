#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${NPU_IR_SIM_VENV:-$HOME/.venv/npuir-sim-system}"
PYTHON_BIN="${NPU_IR_SIM_PYTHON:-/usr/bin/python3.10}"
PATCH_FILE="$SCRIPT_DIR/triton-ascend-3.2-cann91-simulator.patch"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python 3.10 was not found at $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  'torch==2.7.1+cpu'
"$VENV_DIR/bin/python" -m pip install \
  'torch_npu==2.7.1.post8' \
  'triton-ascend==3.2.0' \
  'attrs==24.2.0' \
  'numpy==1.26.4' \
  'scipy==1.13.1' \
  'decorator==5.1.1' \
  'psutil==6.0.0' \
  pyyaml \
  pybind11

SITE_PACKAGES="$($VENV_DIR/bin/python -c 'import site; print(site.getsitepackages()[0])')"
COMPILER_PY="$SITE_PACKAGES/triton/backends/ascend/compiler.py"
NPU_UTILS_CPP="$SITE_PACKAGES/triton/backends/ascend/npu_utils.cpp"

if rg -q 'TRITON_BISHENGIR_DISABLE_LIB_CALL_NOINLINE' "$COMPILER_PY" && \
   ! rg -q '\{"WARP_STACK_SIZE",' "$NPU_UTILS_CPP"; then
  echo "Triton simulator compatibility patch is already applied."
else
  patch -d "$SITE_PACKAGES" -p1 < "$PATCH_FILE"
fi

echo "NPU-IR simulator environment is ready at $VENV_DIR"
