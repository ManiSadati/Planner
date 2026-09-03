#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <testcase-directory>" >&2
    exit 2
fi

CASE_DIR=$(cd -- "$1" && pwd)
CASE_NAME=$(basename -- "${CASE_DIR}")
BUILD_DIR="${BUILD_DIR:-${CASE_DIR}/build-npu}"
RUN_DIR="${RUN_DIR:-${BUILD_DIR}/run}"
KERNEL_OBJ="${KERNEL_OBJ:-${CASE_DIR}/${CASE_NAME}_kernel_ptoas_fatobj.o}"
ASCEND_DRIVER_PATH="${ASCEND_DRIVER_PATH:-/usr/local/Ascend/driver}"
BISHENG_BIN="${BISHENG_BIN:-bisheng}"
LIB_NAME="${CASE_NAME}_kernel"
EXECUTABLE="${CASE_NAME}_vpto"

if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    echo "[ERROR] ASCEND_HOME_PATH is not set. Source the A5 CANN environment first." >&2
    exit 1
fi
if [[ ! -f "${KERNEL_OBJ}" ]]; then
    echo "[ERROR] prebuilt fat object not found: ${KERNEL_OBJ}" >&2
    exit 1
fi
if ! command -v "${BISHENG_BIN}" >/dev/null 2>&1; then
    echo "[ERROR] bisheng not found. Source the A5 CANN environment or set BISHENG_BIN." >&2
    exit 1
fi

mkdir -p "${BUILD_DIR}" "${RUN_DIR}"

"${BISHENG_BIN}" \
    -c -fPIC -xcce \
    -Xhost-start -Xhost-end \
    -mllvm -cce-aicore-stack-size=0x8000 \
    -mllvm -cce-aicore-function-stack-size=0x8000 \
    -mllvm -cce-aicore-record-overflow=true \
    -mllvm -cce-aicore-addr-transform \
    -mllvm -cce-aicore-dcci-insert-for-scalar=false \
    --cce-aicore-arch=dav-c310-cube \
    -std=c++17 \
    -Wno-macro-redefined -Wno-ignored-attributes -Wno-unknown-attributes \
    -I "${ASCEND_HOME_PATH}/include" \
    -I "${ASCEND_HOME_PATH}/pkg_inc" \
    -I "${ASCEND_HOME_PATH}/pkg_inc/profiling" \
    -I "${ASCEND_HOME_PATH}/pkg_inc/runtime/runtime" \
    "${CASE_DIR}/launch.cpp" \
    -o "${BUILD_DIR}/launch.o"

"${BISHENG_BIN}" \
    -fPIC -s -Wl,-z,relro -Wl,-z,now --cce-fatobj-link \
    -shared -Wl,-soname,"lib${LIB_NAME}.so" \
    -L "${ASCEND_HOME_PATH}/lib64" \
    -Wl,-rpath,"${ASCEND_HOME_PATH}/lib64" \
    -o "${BUILD_DIR}/lib${LIB_NAME}.so" \
    "${KERNEL_OBJ}" \
    "${BUILD_DIR}/launch.o" \
    -Wl,--no-as-needed -lruntime

"${BISHENG_BIN}" \
    -xc++ -include stdint.h -include stddef.h -std=c++17 \
    "${CASE_DIR}/main.cpp" \
    -I "${CASE_DIR}" \
    -I "${ASCEND_HOME_PATH}/include" \
    -I "${ASCEND_DRIVER_PATH}/kernel/inc" \
    -L "${BUILD_DIR}" \
    -L "${ASCEND_HOME_PATH}/lib64" \
    -Wl,-rpath,"${BUILD_DIR}" \
    -Wl,-rpath,"${ASCEND_HOME_PATH}/lib64" \
    -o "${BUILD_DIR}/${EXECUTABLE}" \
    -l"${LIB_NAME}" \
    -Wl,--allow-shlib-undefined -lruntime \
    -lstdc++ -lascendcl -lm -ltiling_api -lplatform -lc_sec -ldl -lnnopbase -lpthread

cd "${RUN_DIR}"
python3 "${CASE_DIR}/gen_data.py"
LD_LIBRARY_PATH="${BUILD_DIR}:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}" \
    "${BUILD_DIR}/${EXECUTABLE}"
python3 "${CASE_DIR}/compare.py"
