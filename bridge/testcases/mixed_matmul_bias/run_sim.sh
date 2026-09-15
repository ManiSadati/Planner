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
cmake_bin="${CMAKE_BIN:-cmake}"
python_bin="${PYTHON_BIN:-python3}"

sim_lib_dir="${SIM_LIB_DIR:-$ascend_home/tools/simulator/$soc_version/lib}"
export LD_LIBRARY_PATH="$sim_lib_dir:$ascend_home/runtime/lib64/stub:$ascend_home/lib64:${LD_LIBRARY_PATH:-}"

"$cmake_bin" -S "$script_dir" -B "$build_dir" \
    -DSOC_VERSION="$soc_version" \
    -DPTOAS_BIN="$ptoas_bin" \
    -DKERNEL_MLIR="$kernel_mlir"
"$cmake_bin" --build "$build_dir" --parallel "${BUILD_JOBS:-$(nproc)}"

mkdir -p "$run_dir"
cd "$run_dir"
"$python_bin" "$script_dir/gen_data.py"

if [[ "${USE_MSPROF:-0}" == "1" ]]; then
    kernel_name="${KERNEL_NAME:?KERNEL_NAME is required when USE_MSPROF=1}"
    profile_dir="${MSPROF_OUTPUT:-$build_dir/profile}"
    mkdir -p "$profile_dir"
    msprof op simulator \
        --kernel-name="$kernel_name" \
        --soc-version="$soc_version" \
        --core-id="${CORE_ID:-0}" \
        --output="$profile_dir" \
        --application="$build_dir/${case_name}_vpto"
else
    "$build_dir/${case_name}_vpto"
fi

"$python_bin" "$script_dir/compare.py"
