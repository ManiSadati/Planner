#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

case_name="${CASE_NAME:-${TESTCASE_NAME:-$(basename -- "$script_dir")}}"
build_dir="${BUILD_DIR:-$script_dir/out/build/ptoas-sim}"
run_dir="${RUN_DIR:-$build_dir/run}"
soc_version="${SOC_VERSION:-Ascend950PR_9599}"
ptoas_bin="${PTOAS_BIN:?PTOAS_BIN is required}"
kernel_mlir="${KERNEL_MLIR:?KERNEL_MLIR is required}"
ascend_home="${ASCEND_HOME_PATH:?ASCEND_HOME_PATH is required}"
use_msprof="${USE_MSPROF:-0}"
kernel_name="${KERNEL_NAME:-}"
core_id="${CORE_ID:-0}"
msprof_output="${MSPROF_OUTPUT:-$build_dir/profile}"
cmake_bin="${CMAKE_BIN:-cmake}"
python_bin="${PYTHON_BIN:-python3}"

if [[ ! -x "$ptoas_bin" ]]; then
  echo "error: PTOAS_BIN is not executable: $ptoas_bin" >&2
  exit 1
fi
if [[ ! -f "$kernel_mlir" ]]; then
  echo "error: KERNEL_MLIR does not exist: $kernel_mlir" >&2
  exit 1
fi
if ! "$cmake_bin" --version >/dev/null 2>&1; then
  echo "error: CMAKE_BIN is not a working cmake binary: $cmake_bin" >&2
  exit 1
fi
if ! "$python_bin" -c 'import numpy' >/dev/null 2>&1; then
  echo "error: PYTHON_BIN cannot import numpy: $python_bin" >&2
  exit 1
fi

sim_lib_dir="${SIM_LIB_DIR:-$ascend_home/tools/simulator/$soc_version/lib}"
export LD_LIBRARY_PATH="$sim_lib_dir:$ascend_home/runtime/lib64/stub:$ascend_home/lib64:${LD_LIBRARY_PATH:-}"

"$cmake_bin" -S "$script_dir" -B "$build_dir" \
  -DSOC_VERSION="$soc_version" \
  -DPTOAS_BIN="$ptoas_bin" \
  -DKERNEL_MLIR="$kernel_mlir"
"$cmake_bin" --build "$build_dir" --parallel "${BUILD_JOBS:-$(nproc)}"

executable="$build_dir/${case_name}_vpto"
if [[ ! -x "$executable" ]]; then
  echo "error: simulator executable not found: $executable" >&2
  exit 1
fi

mkdir -p "$run_dir"
cd "$run_dir"
"$python_bin" "$script_dir/gen_data.py"

if [[ "$use_msprof" == "1" ]]; then
  [[ -n "$kernel_name" ]] || {
    echo "error: KERNEL_NAME is required when USE_MSPROF=1" >&2
    exit 1
  }
  mkdir -p "$msprof_output"
  chmod 700 "$msprof_output" 2>/dev/null || true
  msprof op simulator \
    --kernel-name="$kernel_name" \
    --soc-version="$soc_version" \
    --core-id="$core_id" \
    --output="$msprof_output" \
    --application="$executable"
else
  "$executable"
fi

"$python_bin" "$script_dir/compare.py"
