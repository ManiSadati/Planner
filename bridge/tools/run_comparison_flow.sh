#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
planner_root="$(cd -- "${script_dir}/../.." && pwd)"
workspace_root="$(cd -- "${planner_root}/.." && pwd)"

usage() {
  cat >&2 <<'EOF'
Usage:
  bridge/tools/run_comparison_flow.sh <flow>

Flows:
  record-versions   Write repo branch/commit records into $OUT_ROOT/versions.md.
  early-ir          Run Triton Python through msprof compile path and collect early IR.
  baseline-sim      Run full baseline NPU-IR simulator from Triton Python.
  baseline-ir       Compile early IR without bridge passes and dump AVE intrinsic IR.
  bridge-ir         Compile early IR with bridge passes and dump PTOAS VMI IR.
  bridge-lower      Run bridge testcase through NPU-IR -> PTOAS VPTO and LLVM IR.
  ptoas-lower       Re-run PTOAS VPTO and LLVM lowering from bridge-lower VMI.
  bridge-sim        Run bridge testcase through PTOAS simulator fixture.
  all-ir            Run early-ir, baseline-ir, bridge-ir, bridge-lower.

Expected environment:
  Source bridge/tools/source_comparison_env.sh first, or set TESTCASE,
  KERNEL_NAME, PY_FILE, OUT_ROOT, CANN_ROOT, SOC_VERSION, and CORE_ID manually.
EOF
}

flow="${1:-}"
if [[ -z "$flow" || "$flow" == "-h" || "$flow" == "--help" ]]; then
  usage
  exit 0
fi

TESTCASE="${TESTCASE:-vadd}"
KERNEL_NAME="${KERNEL_NAME:-vector_add_kernel}"
PY_FILE="${PY_FILE:-bridge/triton-example/vector_add.py}"
CANN_ROOT="${CANN_ROOT:-${ASCEND_HOME_PATH:-}}"
SOC_VERSION="${SOC_VERSION:-Ascend950PR_9589}"
CORE_ID="${CORE_ID:-0}"
PTOAS_SIM_SOC_VERSION="${PTOAS_SIM_SOC_VERSION:-Ascend950PR_9599}"
OUT_ROOT="${OUT_ROOT:-$HOME/tmp/npuir-ptoas-comparison/${TESTCASE}-manual}"
EARLY_OUT="${EARLY_OUT:-$OUT_ROOT/early-ir}"
BASELINE_SIM_OUT="${BASELINE_SIM_OUT:-$OUT_ROOT/baseline-npuir-sim}"
BASELINE_IR_OUT="${BASELINE_IR_OUT:-$OUT_ROOT/baseline-npuir-ir}"
BRIDGE_IR_OUT="${BRIDGE_IR_OUT:-$OUT_ROOT/bridge-ptoas-vmi}"
BRIDGE_RUNNER_OUT="${BRIDGE_RUNNER_OUT:-$OUT_ROOT/bridge-runner}"
BRIDGE_SIM_OUT="${BRIDGE_SIM_OUT:-$OUT_ROOT/bridge-sim}"
PTOAS_ONLY_OUT="${PTOAS_ONLY_OUT:-$OUT_ROOT/ptoas-only}"
NPU_IR_ROOT="${NPU_IR_ROOT:-$workspace_root/AscendNPU-IR}"
BISHENGIR_COMPILE="${BISHENGIR_COMPILE:-$NPU_IR_ROOT/build/install/bin/bishengir-compile}"
PTOAS_ROOT="${PTOAS_ROOT:-$workspace_root/PTOAS/PTOAS_Markham}"

if [[ "$PY_FILE" != /* ]]; then
  PY_FILE_ABS="$planner_root/$PY_FILE"
else
  PY_FILE_ABS="$PY_FILE"
fi

log() {
  printf '[comparison] %s\n' "$*"
}

require_cann() {
  if [[ -z "$CANN_ROOT" || ! -f "$CANN_ROOT/set_env.sh" ]]; then
    echo "error: set CANN_ROOT to a CANN install containing set_env.sh" >&2
    exit 1
  fi
}

ensure_private_dirs() {
  mkdir -p "$@"
  chmod 700 "$@" 2>/dev/null || true
}

find_early_ir() {
  if [[ -n "${EARLY_IR:-}" && -f "$EARLY_IR" ]]; then
    printf '%s\n' "$EARLY_IR"
    return 0
  fi

  local found
  found="$(find "$EARLY_OUT/early-ir" -type f -name '*kernel.ttadapter.mlir' 2>/dev/null | sort | tail -1)"
  if [[ -z "$found" ]]; then
    echo "error: no early TTAdapter MLIR found under $EARLY_OUT/early-ir" >&2
    echo "       run: bridge/tools/run_comparison_flow.sh early-ir" >&2
    exit 1
  fi
  printf '%s\n' "$found"
}

compile_args() {
  local input="$1"
  local output_path="$2"
  local temps="$3"
  local target_pass="$4"

  printf '%s\0' \
    "$BISHENGIR_COMPILE" "$input" \
    --target=Ascend910_9589 \
    --enable-auto-multi-buffer=true \
    --enable-auto-bind-sub-block=true \
    --disable-ffts \
    --limit-auto-multi-buffer-of-local-buffer=no-limit \
    --enable-auto-blockify-loop \
    --enable-hfusion-compile=true \
    --enable-hivm-compile=true \
    --enable-triton-kernel-compile=true \
    --mlir-disable-threading \
    --mlir-print-stacktrace-on-diagnostic \
    --enable-vf-merge-level=1 \
    "--mlir-print-ir-after=$target_pass" \
    "--save-temps=$temps" \
    -o "$output_path"
}

run_compile_dump() {
  local mode="$1"
  local target_pass="$2"
  local out_dir="$3"
  local bridge_flag="$4"
  local input
  input="$(find_early_ir)"

  mkdir -p "$out_dir"
  mapfile -d '' cmd < <(compile_args "$input" "$out_dir/$KERNEL_NAME" "$out_dir/temps" "$target_pass")
  printf '%q ' "${cmd[@]}" >"$out_dir/command.txt"
  printf '\n' >>"$out_dir/command.txt"

  log "$mode compile: $input"
  set +e
  if [[ "$bridge_flag" == "1" ]]; then
    BISHENGIR_ENABLE_PTOAS_BRIDGE=1 "${cmd[@]}" >"$out_dir/compile.log" 2>&1
  else
    env -u BISHENGIR_ENABLE_PTOAS_BRIDGE "${cmd[@]}" >"$out_dir/compile.log" 2>&1
  fi
  local status=$?
  set -e
  echo "$status" >"$out_dir/exit-code.txt"

  if rg -q "IR Dump After .*${target_pass}|${target_pass}" "$out_dir/compile.log"; then
    log "$mode dump captured: $out_dir/compile.log"
    if [[ "$status" -ne 0 ]]; then
      log "$mode compiler exited $status after/around dump; keeping output for IR comparison"
    fi
    return 0
  fi

  log "$mode compile failed before requested dump; see $out_dir/compile.log"
  return "$status"
}

activate_ptoas_if_available() {
  if command -v ptoas >/dev/null 2>&1; then
    return 0
  fi
  if [[ -f "$HOME/.bashrc" ]]; then
    # shellcheck disable=SC1090
    source "$HOME/.bashrc" >/dev/null 2>&1 || true
  fi
  if declare -F activate_ptoas >/dev/null 2>&1; then
    activate_ptoas >/dev/null
  fi
}

find_ptoas_bin() {
  activate_ptoas_if_available
  if [[ -n "${PTOAS_BIN:-}" && -x "$PTOAS_BIN" ]]; then
    printf '%s\n' "$PTOAS_BIN"
    return 0
  fi
  if command -v ptoas >/dev/null 2>&1; then
    command -v ptoas
    return 0
  fi
  echo "error: ptoas not found; run activate_ptoas or set PTOAS_BIN" >&2
  exit 1
}

run_record_versions() {
  mkdir -p "$OUT_ROOT"
  {
    echo "# Comparison Versions"
    echo
    date -u '+Generated: %Y-%m-%dT%H:%M:%SZ'
    echo
    for repo in "$NPU_IR_ROOT" "$PTOAS_ROOT" "$planner_root"; do
      if [[ -d "$repo/.git" ]]; then
        echo "## $repo"
        git -C "$repo" rev-parse --abbrev-ref HEAD
        git -C "$repo" rev-parse HEAD
        git -C "$repo" status --short
        echo
      fi
    done
  } >"$OUT_ROOT/versions.md"
  log "wrote $OUT_ROOT/versions.md"
}

run_early_ir() {
  require_cann
  log "early IR: $KERNEL_NAME from $PY_FILE"
  CANN_ROOT="$CANN_ROOT" \
  SOC_VERSION="$SOC_VERSION" \
  CORE_ID="$CORE_ID" \
  "$planner_root/NPUIR/tools/dump_early_ir_from_triton.sh" \
    "$KERNEL_NAME" \
    "$PY_FILE" \
    "$EARLY_OUT"
}

run_baseline_sim() {
  require_cann
  ensure_private_dirs \
    "$BASELINE_SIM_OUT" \
    "$BASELINE_SIM_OUT/cache" \
    "$BASELINE_SIM_OUT/dump" \
    "$BASELINE_SIM_OUT/logs" \
    "$BASELINE_SIM_OUT/profile"

  log "baseline simulator: $KERNEL_NAME from $PY_FILE"
  # shellcheck disable=SC1091
  source "$planner_root/NPUIR/tools/source_npuir_simulator_env.sh"

  export TRITON_CACHE_DIR="$BASELINE_SIM_OUT/cache"
  export TRITON_DUMP_DIR="$BASELINE_SIM_OUT/dump"
  export ASCEND_PROCESS_LOG_PATH="$BASELINE_SIM_OUT/logs"
  unset BISHENGIR_ENABLE_PTOAS_BRIDGE

  msprof op simulator \
    --kernel-name="$KERNEL_NAME" \
    --soc-version="$SOC_VERSION" \
    --core-id="$CORE_ID" \
    --output="$BASELINE_SIM_OUT/profile" \
    python3 "$PY_FILE_ABS" \
    >"$BASELINE_SIM_OUT/msprof.stdout.log" \
    2>"$BASELINE_SIM_OUT/msprof.stderr.log"

  log "baseline simulator output: $BASELINE_SIM_OUT"
}

run_bridge_lower() {
  local ptoas_bin
  ptoas_bin="$(find_ptoas_bin)"
  log "bridge lower: $TESTCASE"
  BISHENGIR_ENABLE_PTOAS_BRIDGE=1 \
  OUTPUT_ROOT="$BRIDGE_RUNNER_OUT" \
  BISHENGIR_COMPILE="$BISHENGIR_COMPILE" \
  PTOAS_BIN="$ptoas_bin" \
  "$planner_root/bridge/tools/run_npuir_ptoas_bridge_tests.sh" \
    --from-bishengir-compile \
    --clean \
    --emit-vpto \
    --emit-llvmir \
    "$TESTCASE"
}

run_ptoas_lower() {
  local ptoas_bin
  ptoas_bin="$(find_ptoas_bin)"
  local vmi="$BRIDGE_RUNNER_OUT/$TESTCASE/$TESTCASE.vmi.mlir"
  if [[ ! -f "$vmi" ]]; then
    echo "error: missing VMI file: $vmi" >&2
    echo "       run: bridge/tools/run_comparison_flow.sh bridge-lower" >&2
    exit 1
  fi

  mkdir -p "$PTOAS_ONLY_OUT"
  log "PTOAS lower: $vmi"
  "$ptoas_bin" --pto-arch=a5 --pto-backend=vpto --emit-vpto \
    "$vmi" -o "$PTOAS_ONLY_OUT/$TESTCASE.vpto.mlir"
  "$ptoas_bin" --pto-arch=a5 --pto-backend=vpto --emit-vpto-llvm-ir \
    "$vmi" -o "$PTOAS_ONLY_OUT/$TESTCASE.vpto.ll"
}

run_bridge_sim() {
  require_cann
  local ptoas_bin
  ptoas_bin="$(find_ptoas_bin)"
  log "bridge simulator: $TESTCASE"
  BISHENGIR_ENABLE_PTOAS_BRIDGE=1 \
  ASCEND_HOME_PATH="$CANN_ROOT" \
  SOC_VERSION="$PTOAS_SIM_SOC_VERSION" \
  SIM_LIB_DIR="$CANN_ROOT/tools/simulator/$PTOAS_SIM_SOC_VERSION/lib" \
  BUILD_JOBS="${BUILD_JOBS:-16}" \
  OUTPUT_ROOT="$BRIDGE_SIM_OUT" \
  BISHENGIR_COMPILE="$BISHENGIR_COMPILE" \
  PTOAS_BIN="$ptoas_bin" \
  "$planner_root/bridge/tools/run_npuir_ptoas_bridge_tests.sh" \
    --from-bishengir-compile \
    --clean \
    --run-simulator \
    "$TESTCASE"
}

case "$flow" in
  record-versions)
    run_record_versions
    ;;
  early-ir)
    run_early_ir
    ;;
  baseline-sim)
    run_baseline_sim
    ;;
  baseline-ir)
    run_compile_dump baseline convert-hivmave-to-ave-intrin "$BASELINE_IR_OUT" 0
    ;;
  bridge-ir)
    run_compile_dump bridge convert-hivmave-to-ptoas-vmi "$BRIDGE_IR_OUT" 1
    ;;
  bridge-lower)
    run_bridge_lower
    ;;
  ptoas-lower)
    run_ptoas_lower
    ;;
  bridge-sim)
    run_bridge_sim
    ;;
  all-ir)
    run_record_versions
    run_early_ir
    run_compile_dump baseline convert-hivmave-to-ave-intrin "$BASELINE_IR_OUT" 0
    run_compile_dump bridge convert-hivmave-to-ptoas-vmi "$BRIDGE_IR_OUT" 1
    run_bridge_lower
    ;;
  *)
    echo "error: unknown flow: $flow" >&2
    usage
    exit 1
    ;;
esac
