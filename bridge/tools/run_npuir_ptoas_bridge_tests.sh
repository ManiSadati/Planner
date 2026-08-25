#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat >&2 <<'EOF'
Usage:
  bridge/tools/run_npuir_ptoas_bridge_tests.sh <option> <testcase>

This is a compatibility wrapper. The implementation lives in:
  bridge/tools/run_comparison_flow.sh

Useful options:
  early-ir
  print-all
  npu-sim
  emit-vmi
  emit-vpto
  bridge-sim
EOF
}

if [[ $# -eq 0 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

case "${1:-}" in
  --emit-vpto)
    shift
    exec "$script_dir/run_comparison_flow.sh" emit-vpto "$@"
    ;;
  --run-simulator|--sim)
    shift
    exec "$script_dir/run_comparison_flow.sh" bridge-sim "$@"
    ;;
  --all)
    shift
    "$script_dir/run_comparison_flow.sh" early-ir "$@"
    "$script_dir/run_comparison_flow.sh" print-all "$@"
    "$script_dir/run_comparison_flow.sh" emit-vpto "$@"
    exec "$script_dir/run_comparison_flow.sh" bridge-sim "$@"
    ;;
  *)
    exec "$script_dir/run_comparison_flow.sh" "$@"
    ;;
esac
