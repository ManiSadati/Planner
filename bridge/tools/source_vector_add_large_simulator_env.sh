#!/usr/bin/env bash

# Source this before running the vector_add_large msprof command.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Source this script instead of executing it:" >&2
  echo "  source bridge/tools/source_vector_add_large_simulator_env.sh" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "$SCRIPT_DIR/source_npuir_simulator_env.sh"

cd "$PLANNER_DIR"

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
export OUT="${NPU_IR_SIM_OUTPUT:-$HOME/tmp/npuir-simulator/vector-add-large-$RUN_ID}"

mkdir -p "$OUT"/{cache,dump,logs,profile}
chmod 700 "$OUT" "$OUT"/{cache,dump,logs,profile}

export TRITON_CACHE_DIR="$OUT/cache"
export TRITON_DUMP_DIR="$OUT/dump"
export ASCEND_PROCESS_LOG_PATH="$OUT/logs"

echo "Planner dir: $PLANNER_DIR"
echo "Simulator output: $OUT"
