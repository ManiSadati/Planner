#!/usr/bin/env bash

# Kept so old shells and notes fail gently.
# The bridge runner no longer needs a sourced setup script.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  cat >&2 <<'EOF'
Do not execute this script.

Set the three root paths directly, then run one option and one testcase:

  export ASCEND_NPU_IR_ROOT=/path/to/AscendNPU-IR
  export CANN_ROOT=/path/to/CANN
  export PTOAS_ROOT=/path/to/PTOAS
  bridge/tools/run_comparison_flow.sh emit-vpto vadd
EOF
  exit 1
fi

export ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-${CANN_ROOT:-}}"
echo "source_comparison_env.sh is no longer needed; use bridge/tools/run_comparison_flow.sh directly."
