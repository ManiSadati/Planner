#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CASE_NAME="${TESTCASE_NAME:-$(basename "${SCRIPT_DIR}")}"
BUILD_DIR="${BUILD_DIR:-${SCRIPT_DIR}/build}"
RUN_DIR="${RUN_DIR:-${BUILD_DIR}/run}"
SOC_VERSION="${SOC_VERSION:-Ascend950PR_9599}"
PTOAS_BIN="${PTOAS_BIN:-${SCRIPT_DIR}/../../../../PTOAS/build/tools/ptoas/ptoas}"
KERNEL_MLIR="${KERNEL_MLIR:-${SCRIPT_DIR}/../../out/npuir-ptoas-bridge/${CASE_NAME}/${CASE_NAME}.vpto.mlir}"
if [[ ! -x "${PTOAS_BIN}" ]]; then if command -v ptoas >/dev/null 2>&1; then PTOAS_BIN=$(command -v ptoas); else echo "[ERROR] PTOAS_BIN is not executable: ${PTOAS_BIN}" >&2; exit 1; fi; fi
if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then echo "[ERROR] ASCEND_HOME_PATH is not set." >&2; exit 1; fi
if [[ ! -f "${KERNEL_MLIR}" ]]; then echo "[ERROR] VPTO MLIR file is missing: ${KERNEL_MLIR}" >&2; exit 1; fi
SIM_LIB_DIR="${SIM_LIB_DIR:-${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib}"
export LD_LIBRARY_PATH="${SIM_LIB_DIR}:${ASCEND_HOME_PATH}/runtime/lib64/stub:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}"
cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" -DSOC_VERSION="${SOC_VERSION}" -DPTOAS_BIN="${PTOAS_BIN}" -DKERNEL_MLIR="${KERNEL_MLIR}"
cmake --build "${BUILD_DIR}" --parallel "${BUILD_JOBS:-$(nproc)}"
mkdir -p "${RUN_DIR}"
cd "${RUN_DIR}"
python3 "${SCRIPT_DIR}/gen_data.py"
"${BUILD_DIR}/${CASE_NAME}_vpto"
python3 "${SCRIPT_DIR}/compare.py"
