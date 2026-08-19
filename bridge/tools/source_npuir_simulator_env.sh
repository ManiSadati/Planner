#!/usr/bin/env bash

# Source this file before running msprof-based Triton/NPU-IR simulator commands.
# It configures the runtime environment only; it does not install packages.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Source this script instead of executing it:" >&2
  echo "  source bridge/tools/source_npuir_simulator_env.sh" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLANNER_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

export CANN_ROOT="${CANN_ROOT:-${ASCEND_HOME_PATH:-/home/a84369921/Ascend/cann-9.1.0-beta.3}}"
export NPU_IR_ROOT="${NPU_IR_ROOT:-$HOME/AscendNPU-IR}"
export NPU_IR_SIM_VENV="${NPU_IR_SIM_VENV:-$HOME/.venv/npuir-sim-system}"
export TMPDIR="${TMPDIR:-$HOME/tmp}"

if [[ ! -f "$CANN_ROOT/set_env.sh" ]]; then
  echo "CANN set_env.sh not found: $CANN_ROOT/set_env.sh" >&2
  return 1
fi

if [[ ! -x "$NPU_IR_SIM_VENV/bin/python" ]]; then
  echo "NPU-IR simulator venv not found: $NPU_IR_SIM_VENV" >&2
  echo "Run bridge/tools/setup_npuir_simulator_env.sh first." >&2
  return 1
fi

if [[ ! -x "$NPU_IR_ROOT/build/install/bin/bishengir-compile" ]]; then
  echo "NPU-IR install not found: $NPU_IR_ROOT/build/install" >&2
  return 1
fi

source "$NPU_IR_SIM_VENV/bin/activate"
case "$-" in
  *u*) _npuir_sim_had_nounset=1 ;;
  *) _npuir_sim_had_nounset=0 ;;
esac
set +u
source "$CANN_ROOT/set_env.sh"
if [[ "$_npuir_sim_had_nounset" == "1" ]]; then
  set -u
else
  set +u
fi
unset _npuir_sim_had_nounset

export PATH="$NPU_IR_ROOT/build/install/bin:$PATH"
export PYTHONNOUSERSITE=1
export TRITON_ASCEND_ARCH="${TRITON_ASCEND_ARCH:-Ascend910_9589}"
export TRITON_BISHENGIR_DISABLE_LIB_CALL_NOINLINE="${TRITON_BISHENGIR_DISABLE_LIB_CALL_NOINLINE:-1}"
export TRITON_DISABLE_FFTS="${TRITON_DISABLE_FFTS:-1}"
export TRITON_SIMULATOR_CLEAN_EXIT="${TRITON_SIMULATOR_CLEAN_EXIT:-1}"
export TRITON_ALWAYS_COMPILE="${TRITON_ALWAYS_COMPILE:-1}"
export TRITON_DEBUG="${TRITON_DEBUG:-1}"
export PYTHONPATH="${PYTHONPATH:-}"

export PLANNER_DIR
