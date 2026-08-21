#!/usr/bin/env bash

# Source this file before running bridge/tools/run_comparison_flow.sh.
# It sets the testcase/kernel/output variables used by every comparison flow.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Source this script instead of executing it:" >&2
  echo "  source bridge/tools/source_comparison_env.sh [options]" >&2
  exit 1
fi

_comparison_usage() {
  cat >&2 <<'EOF'
Usage:
  source bridge/tools/source_comparison_env.sh [options]

Options:
  --testcase NAME       Bridge testcase name. Default: vadd
  --kernel-name NAME    Triton kernel function name. Default: vector_add_kernel
  --python-file PATH    Repo-relative or absolute Python testcase. Default: bridge/triton-example/vector_add.py
  --cann-root PATH      CANN install root containing set_env.sh. Default: existing CANN_ROOT or ASCEND_HOME_PATH
  --soc-version SOC     NPU-IR msprof simulator SOC. Default: Ascend950PR_9589
  --core-id ID          NPU-IR msprof simulator core id. Default: 0
  --ptoas-sim-soc SOC   PTOAS fixture simulator SOC. Default: same as --soc-version
  --out-root PATH       Output root. Default: $HOME/tmp/npuir-ptoas-comparison/<testcase>-<timestamp>
  --run-id ID           Run id used when --out-root is omitted. Default: UTC timestamp
EOF
}

_comparison_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PLANNER_DIR="$(cd "$_comparison_script_dir/../.." && pwd)"

export TESTCASE="${TESTCASE:-vadd}"
export KERNEL_NAME="${KERNEL_NAME:-vector_add_kernel}"
export PY_FILE="${PY_FILE:-bridge/triton-example/vector_add.py}"
export CANN_ROOT="${CANN_ROOT:-${ASCEND_HOME_PATH:-}}"
export SOC_VERSION="${SOC_VERSION:-Ascend950PR_9589}"
export CORE_ID="${CORE_ID:-0}"
_comparison_ptoas_sim_soc_explicit=0
if [[ -n "${PTOAS_SIM_SOC_VERSION:-}" ]]; then
  _comparison_ptoas_sim_soc_explicit=1
fi
export PTOAS_SIM_SOC_VERSION="${PTOAS_SIM_SOC_VERSION:-}"
export RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"

_comparison_require_arg() {
  if [[ $# -lt 2 || -z "${2:-}" ]]; then
    echo "error: $1 requires an argument" >&2
    _comparison_usage
    return 1
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --testcase)
      _comparison_require_arg "$@" || return 1
      TESTCASE="$2"
      shift 2
      ;;
    --kernel-name|--kernel)
      _comparison_require_arg "$@" || return 1
      KERNEL_NAME="$2"
      shift 2
      ;;
    --python-file|--py-file)
      _comparison_require_arg "$@" || return 1
      PY_FILE="$2"
      shift 2
      ;;
    --cann-root)
      _comparison_require_arg "$@" || return 1
      CANN_ROOT="$2"
      shift 2
      ;;
    --soc-version|--soc)
      _comparison_require_arg "$@" || return 1
      SOC_VERSION="$2"
      shift 2
      ;;
    --core-id)
      _comparison_require_arg "$@" || return 1
      CORE_ID="$2"
      shift 2
      ;;
    --ptoas-sim-soc)
      _comparison_require_arg "$@" || return 1
      PTOAS_SIM_SOC_VERSION="$2"
      _comparison_ptoas_sim_soc_explicit=1
      shift 2
      ;;
    --out-root)
      _comparison_require_arg "$@" || return 1
      OUT_ROOT="$2"
      shift 2
      ;;
    --run-id)
      _comparison_require_arg "$@" || return 1
      RUN_ID="$2"
      shift 2
      ;;
    -h|--help)
      _comparison_usage
      return 0
      ;;
    *)
      echo "error: unknown option: $1" >&2
      _comparison_usage
      return 1
      ;;
  esac
done

export TESTCASE
export KERNEL_NAME
export PY_FILE
export CANN_ROOT
export SOC_VERSION
export CORE_ID
if [[ "$_comparison_ptoas_sim_soc_explicit" == "0" ]]; then
  PTOAS_SIM_SOC_VERSION="$SOC_VERSION"
fi
export PTOAS_SIM_SOC_VERSION
export RUN_ID
export OUT_ROOT="${OUT_ROOT:-$HOME/tmp/npuir-ptoas-comparison/${TESTCASE}-${RUN_ID}}"
export EARLY_OUT="${EARLY_OUT:-$OUT_ROOT/early-ir}"
export BASELINE_SIM_OUT="${BASELINE_SIM_OUT:-$OUT_ROOT/baseline-npuir-sim}"
export BASELINE_IR_OUT="${BASELINE_IR_OUT:-$OUT_ROOT/baseline-npuir-ir}"
export BRIDGE_IR_OUT="${BRIDGE_IR_OUT:-$OUT_ROOT/bridge-ptoas-vmi}"
export BRIDGE_RUNNER_OUT="${BRIDGE_RUNNER_OUT:-$OUT_ROOT/bridge-runner}"
export BRIDGE_SIM_OUT="${BRIDGE_SIM_OUT:-$OUT_ROOT/bridge-sim}"
export PTOAS_ONLY_OUT="${PTOAS_ONLY_OUT:-$OUT_ROOT/ptoas-only}"

mkdir -p "$OUT_ROOT"

cat <<EOF
Comparison environment:
  TESTCASE=$TESTCASE
  KERNEL_NAME=$KERNEL_NAME
  PY_FILE=$PY_FILE
  OUT_ROOT=$OUT_ROOT
  SOC_VERSION=$SOC_VERSION
  CORE_ID=$CORE_ID
  PTOAS_SIM_SOC_VERSION=$PTOAS_SIM_SOC_VERSION
EOF

if [[ -z "$CANN_ROOT" ]]; then
  echo "  CANN_ROOT is not set; simulator flows will ask for --cann-root or CANN_ROOT." >&2
else
  echo "  CANN_ROOT=$CANN_ROOT"
fi

unset -f _comparison_usage
unset -f _comparison_require_arg
unset _comparison_script_dir
unset _comparison_ptoas_sim_soc_explicit
