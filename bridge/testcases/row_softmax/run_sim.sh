#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BUILD_DIR="${BUILD_DIR:-${SCRIPT_DIR}/build}"
RUN_DIR="${RUN_DIR:-${BUILD_DIR}/run}"
SOC_VERSION="${SOC_VERSION:-Ascend950PR_9599}"
PTOAS_BIN="${PTOAS_BIN:-${SCRIPT_DIR}/../../../../PTOAS/build/tools/ptoas/ptoas}"
KERNEL_MLIR="${KERNEL_MLIR:-${SCRIPT_DIR}/../../out/npuir-ptoas-bridge/row_softmax/row_softmax.vpto.mlir}"

if [[ ! -x "${PTOAS_BIN}" ]]; then
    if command -v ptoas >/dev/null 2>&1; then
        PTOAS_BIN=$(command -v ptoas)
    else
        echo "[ERROR] PTOAS_BIN is not executable: ${PTOAS_BIN}" >&2
        echo "[ERROR] Set PTOAS_BIN=/path/to/ptoas or put ptoas in PATH." >&2
        exit 1
    fi
fi

if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    echo "[ERROR] ASCEND_HOME_PATH is not set. Source the CANN environment first." >&2
    exit 1
fi

if [[ ! -f "${KERNEL_MLIR}" ]]; then
    echo "[ERROR] VPTO MLIR file is missing: ${KERNEL_MLIR}" >&2
    echo "[ERROR] Generate it first with the bridge script, for example:" >&2
    echo "[ERROR]   Planner/bridge/tools/run_npuir_ptoas_bridge_tests.sh --emit-vpto row_softmax" >&2
    exit 1
fi
KERNEL_MLIR=$(cd -- "$(dirname -- "${KERNEL_MLIR}")" && pwd)/$(basename -- "${KERNEL_MLIR}")

SIM_LIB_DIR="${SIM_LIB_DIR:-${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib}"
export LD_LIBRARY_PATH="${SIM_LIB_DIR}:${ASCEND_HOME_PATH}/runtime/lib64/stub:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}"

cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" \
    -DSOC_VERSION="${SOC_VERSION}" \
    -DPTOAS_BIN="${PTOAS_BIN}" \
    -DKERNEL_MLIR="${KERNEL_MLIR}"
cmake --build "${BUILD_DIR}" --parallel "${BUILD_JOBS:-$(nproc)}"

mkdir -p "${RUN_DIR}"
cd "${RUN_DIR}"
python3 "${SCRIPT_DIR}/gen_data.py"
"${BUILD_DIR}/row_softmax_vpto"
python3 "${SCRIPT_DIR}/compare.py"
