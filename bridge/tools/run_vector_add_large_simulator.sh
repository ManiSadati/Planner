#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "$SCRIPT_DIR/source_vector_add_large_simulator_env.sh"

exec msprof op simulator \
  --kernel-name=vector_add_large_kernel \
  --soc-version=Ascend950PR_9589 \
  --core-id=0 \
  --output="$OUT/profile" \
  python3 "$PLANNER_DIR/bridge/triton-example/vector_add_large.py"
