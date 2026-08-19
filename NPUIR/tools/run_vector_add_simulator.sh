#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_ROOT="${NPU_IR_SIM_OUTPUT:-$HOME/tmp/npuir-simulator/vector-add-$RUN_ID}"

source "$SCRIPT_DIR/source_npuir_simulator_env.sh"

mkdir -p \
  "$OUTPUT_ROOT/cache" \
  "$OUTPUT_ROOT/dump" \
  "$OUTPUT_ROOT/logs" \
  "$OUTPUT_ROOT/profile"
chmod 700 \
  "$OUTPUT_ROOT" \
  "$OUTPUT_ROOT/cache" \
  "$OUTPUT_ROOT/dump" \
  "$OUTPUT_ROOT/logs" \
  "$OUTPUT_ROOT/profile"

export TRITON_CACHE_DIR="$OUTPUT_ROOT/cache"
export TRITON_DUMP_DIR="$OUTPUT_ROOT/dump"
export ASCEND_PROCESS_LOG_PATH="$OUTPUT_ROOT/logs"

echo "Simulator output: $OUTPUT_ROOT"
exec msprof op simulator \
  --kernel-name=vector_add_kernel \
  --soc-version=Ascend950PR_9589 \
  --core-id=0 \
  --output="$OUTPUT_ROOT/profile" \
  python3 "$PLANNER_DIR/bridge/triton-example/vector_add.py"
