#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  NPUIR/tools/dump_early_ir_from_triton.sh <kernel-name> <python-file> [output-root]

Example:
  NPUIR/tools/dump_early_ir_from_triton.sh \
    vector_add_large_kernel \
    bridge/triton-example/vector_add_large.py

Environment:
  CANN_ROOT        CANN installation containing set_env.sh.
  SOC_VERSION      Simulator SOC. Default: Ascend950PR_9589.
  CORE_ID          Simulator core id. Default: 0.
  NPU_IR_SIM_OUTPUT Optional output root.
  EARLY_IR_STOP_AFTER_DUMP
                   If 1, stop the simulator after the first TTAdapter MLIR
                   dump is observed. Default: 0.
  EARLY_IR_CAPTURE_TIMEOUT_SEC
                   Timeout for stop-after-dump mode. Default: 600.

This script uses msprof op simulator to trigger the Triton/NPU-IR compiler
dump path, then collects generated MLIR/LLVM files. Its purpose is early IR
capture; simulator metrics are secondary.
EOF
}

if [[ $# -lt 2 || $# -gt 3 ]]; then
  usage
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/source_npuir_simulator_env.sh"

kernel_name="$1"
python_file="$2"
if [[ "${python_file}" != /* ]]; then
  python_file="$PLANNER_DIR/${python_file}"
fi

if [[ ! -f "$python_file" ]]; then
  echo "error: python file not found: $python_file" >&2
  exit 1
fi

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
safe_kernel_name="${kernel_name//[^A-Za-z0-9_.-]/_}"
output_root="${3:-${NPU_IR_SIM_OUTPUT:-$HOME/tmp/npuir-early-ir/${safe_kernel_name}-${RUN_ID}}}"
soc_version="${SOC_VERSION:-Ascend950PR_9589}"
core_id="${CORE_ID:-0}"
stop_after_dump="${EARLY_IR_STOP_AFTER_DUMP:-0}"
capture_timeout_sec="${EARLY_IR_CAPTURE_TIMEOUT_SEC:-600}"

mkdir -p \
  "$output_root/cache" \
  "$output_root/dump" \
  "$output_root/logs" \
  "$output_root/profile" \
  "$output_root/early-ir"
chmod 700 \
  "$output_root" \
  "$output_root/cache" \
  "$output_root/dump" \
  "$output_root/logs" \
  "$output_root/profile" \
  "$output_root/early-ir"

export TRITON_CACHE_DIR="$output_root/cache"
export TRITON_DUMP_DIR="$output_root/dump"
export ASCEND_PROCESS_LOG_PATH="$output_root/logs"

cmd=(
  msprof op simulator
  "--kernel-name=${kernel_name}"
  "--soc-version=${soc_version}"
  "--core-id=${core_id}"
  "--output=${output_root}/profile"
  python3
  "$python_file"
)

{
  printf 'working_directory=%q\n' "$PLANNER_DIR"
  printf 'command='
  printf '%q ' "${cmd[@]}"
  printf '\n'
} >"$output_root/command.txt"

echo "Output root: $output_root"
echo "Running simulator compile path for early IR capture..."

sim_status=0
if [[ "$stop_after_dump" == "1" ]]; then
  echo "Stop-after-dump mode: enabled"
  if command -v setsid >/dev/null 2>&1; then
    setsid "${cmd[@]}" >"$output_root/msprof.log" 2>&1 &
    sim_pid=$!
    kill_target="-$sim_pid"
  else
    "${cmd[@]}" >"$output_root/msprof.log" 2>&1 &
    sim_pid=$!
    kill_target="$sim_pid"
  fi

  found_dump=0
  deadline=$((SECONDS + capture_timeout_sec))
  while kill -0 "$sim_pid" 2>/dev/null; do
    if find "$output_root/dump" -type f -name '*kernel.ttadapter.mlir' -print -quit | grep -q .; then
      found_dump=1
      sleep 2
      break
    fi
    if (( SECONDS >= deadline )); then
      echo "warning: timed out waiting for TTAdapter MLIR; letting simulator exit or be terminated" >&2
      break
    fi
    sleep 1
  done

  if kill -0 "$sim_pid" 2>/dev/null; then
    kill -TERM -- "$kill_target" 2>/dev/null || kill -TERM "$sim_pid" 2>/dev/null || true
    sleep 2
    kill -KILL -- "$kill_target" 2>/dev/null || kill -KILL "$sim_pid" 2>/dev/null || true
  fi

  set +e
  wait "$sim_pid"
  sim_status=$?
  set -e
  if [[ "$found_dump" == "1" ]]; then
    echo "Stopped simulator after initial TTAdapter MLIR dump was captured."
  fi
else
  if ! "${cmd[@]}" >"$output_root/msprof.log" 2>&1; then
    sim_status=$?
  fi
fi
echo "$sim_status" >"$output_root/msprof-exit-code.txt"

manifest="$output_root/early-ir-manifest.txt"
: >"$manifest"

count=0
while IFS= read -r dump_file; do
  count=$((count + 1))
  dump_dir="$(basename "$(dirname "$dump_file")")"
  dump_base="$(basename "$dump_file")"
  copied_name="$(printf '%03d-%s-%s' "$count" "$dump_dir" "$dump_base")"
  cp -a "$dump_file" "$output_root/early-ir/$copied_name"
  printf '%s -> early-ir/%s\n' "$dump_file" "$copied_name" >>"$manifest"
done < <(
  find "$output_root/dump" -type f \
    \( -name '*.mlir' -o -name '*.ll' \) \
    | sort
)

echo "Collected IR files: $count"
echo "Manifest: $manifest"
echo "msprof log: $output_root/msprof.log"

if [[ "$count" -eq 0 ]]; then
  echo "error: no MLIR/LLVM dump files were captured" >&2
  exit 1
fi

if [[ "$sim_status" -ne 0 ]]; then
  echo "note: msprof exited $sim_status, but IR dumps were captured"
fi
