#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
planner_root="$(cd -- "${script_dir}/../.." && pwd)"
testcase_root="${planner_root}/bridge/testcases"

npuir_root="${ASCEND_NPU_IR_ROOT:-${NPU_IR_ROOT:-}}"
cann_root="${CANN_ROOT:-${ASCEND_HOME_PATH:-}}"
ptoas_root="${PTOAS_ROOT:-}"

soc_version="Ascend950PR_9599"
core_id="0"

usage() {
  cat >&2 <<EOF
Usage:
  bridge/tools/run_comparison_flow.sh [--clean-build] [--print-ir-after-all] <option> <testcase>

Options:
  npu-sim      Run the Triton Python testcase through the native NPU-IR flow.
  bridge-sim   Run the same Python testcase through the NPU-IR-to-PTOAS flow.
  emit-vmi     Emit PTOAS VMI MLIR from the testcase MLIR with --emit-ptoas-vmi.
  emit-vpto    Emit PTOAS VPTO MLIR from the VMI MLIR.
  print-ir     Emit PTOAS VMI and log the IR after every NPU-IR pass.

Flags:
  --clean-build        Remove testcase build directories before running.
  --print-ir-after-all Add --mlir-print-ir-after-all to emit-vmi.

Required environment:
  ASCEND_NPU_IR_ROOT=/path/to/AscendNPU-IR
  CANN_ROOT=/path/to/CANN
  PTOAS_ROOT=/path/to/PTOAS

Optional environment:
  NPU_SIM_VENV=/path/to/simulator/venv
    default: $HOME/.venv/npuir-sim-system

Testcase layout:
  Planner/bridge/testcases/<testcase>/
    <one Triton Python file with one @triton.jit kernel>
    compile-input.mlir         preferred input for emit-vmi/emit-vpto inspection
    input.mlir                 fallback inspection input

Outputs:
  Planner/bridge/testcases/<testcase>/out/
  Planner/bridge/testcases/<testcase>/out/build/
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
}

log() {
  printf '[bridge] %s\n' "$*"
}

clean_build_dirs() {
  local case_dir="$1"
  local out_dir="$2"

  rm -rf "$out_dir/build" "$case_dir/build"
  log "removed build directories"
}

abs_path() {
  local path="$1"
  if [[ "$path" == /* ]]; then
    printf '%s\n' "$path"
  else
    printf '%s\n' "$(cd -- "$(dirname -- "$path")" && pwd)/$(basename -- "$path")"
  fi
}

prepend_path() {
  local path="$1"
  if [[ -d "$path" ]]; then
    case ":${PATH:-}:" in
      *":$path:"*) ;;
      *) export PATH="$path:${PATH:-}" ;;
    esac
  fi
}

prioritize_path() {
  local path="$1"
  [[ -d "$path" ]] || die "PATH directory does not exist: $path"
  export PATH="$path:${PATH:-}"
}

prepend_ld_library_path() {
  local path="$1"
  if [[ -d "$path" ]]; then
    case ":${LD_LIBRARY_PATH:-}:" in
      *":$path:"*) ;;
      *) export LD_LIBRARY_PATH="$path${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
    esac
  fi
}

require_npuir_root() {
  [[ -n "$npuir_root" ]] || die "set ASCEND_NPU_IR_ROOT to the AscendNPU-IR checkout"
  [[ -e "$npuir_root" ]] || die "ASCEND_NPU_IR_ROOT does not exist: $npuir_root"
}

require_cann_root() {
  [[ -n "$cann_root" ]] || die "set CANN_ROOT to the CANN install root"
  [[ -f "$cann_root/set_env.sh" ]] || die "CANN_ROOT must contain set_env.sh: $cann_root"
}

require_ptoas_root() {
  [[ -n "$ptoas_root" ]] || die "set PTOAS_ROOT to the PTOAS checkout or ptoas binary"
  [[ -e "$ptoas_root" ]] || die "PTOAS_ROOT does not exist: $ptoas_root"
}

find_bishengir_compile() {
  require_npuir_root
  local candidates=(
    "$npuir_root"
    "$npuir_root/build/bin/bishengir-compile"
    "$npuir_root/build/install/bin/bishengir-compile"
  )
  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -x "$candidate" && "$(basename -- "$candidate")" == "bishengir-compile" ]]; then
      abs_path "$candidate"
      return 0
    fi
  done
  die "cannot find bishengir-compile under $npuir_root"
}

find_ptoas() {
  require_ptoas_root
  local candidates=(
    "$ptoas_root"
    "$ptoas_root/build/tools/ptoas/ptoas"
    "$ptoas_root/PTOAS_Markham/build/tools/ptoas/ptoas"
    "$ptoas_root/tools/ptoas/ptoas"
  )
  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -x "$candidate" && "$(basename -- "$candidate")" == "ptoas" ]]; then
      abs_path "$candidate"
      return 0
    fi
  done
  die "cannot find ptoas under $ptoas_root"
}

find_cann_bishengir_compile() {
  require_cann_root
  local candidates=(
    "$cann_root/tools/bishengir/bin/bishengir-compile"
    "$cann_root/x86_64-linux/bin/bishengir-compile"
  )
  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -x "$candidate" ]]; then
      abs_path "$candidate"
      return 0
    fi
  done
  die "cannot find the CANN bishengir-compile under $cann_root"
}

activate_simulator_venv() {
  local requested_venv="${NPU_SIM_VENV:-}"
  local candidates=()
  local venv

  if [[ -n "$requested_venv" ]]; then
    candidates+=("$requested_venv")
  else
    candidates+=("$HOME/.venv/npuir-sim-system")
    if [[ -n "$npuir_root" ]]; then
      candidates+=("$npuir_root/.venv")
    fi
  fi

  for venv in "${candidates[@]}"; do
    if [[ -f "$venv/bin/activate" && -x "$venv/bin/python3" ]]; then
      # shellcheck disable=SC1090
      source "$venv/bin/activate"
      export NPU_SIM_VENV="$venv"
      export PYTHONNOUSERSITE=1
      python3 -c 'import torch, torch_npu, triton; from triton.runtime import driver; assert driver.active' \
        >/dev/null 2>&1 || die "simulator venv cannot load Torch-NPU and the Triton Ascend backend: $venv"
      return 0
    fi
  done

  if [[ -n "$requested_venv" ]]; then
    die "NPU_SIM_VENV is not a usable virtual environment: $requested_venv"
  fi
  die "cannot find the simulator venv; set NPU_SIM_VENV"
}

select_npu_compiler() {
  local compile_flow="$1"
  local compiler

  case "$compile_flow" in
    npuir)
      compiler="$(find_cann_bishengir_compile)"
      ;;
    ptoas)
      compiler="$(find_bishengir_compile)"
      ;;
    *)
      die "unsupported Triton compile flow: $compile_flow"
      ;;
  esac

  prioritize_path "$(dirname -- "$compiler")"
  export TRITON_NPU_COMPILER_PATH="$(dirname -- "$compiler")"
  printf '%s\n' "$compiler"
}

source_cann_env() {
  require_cann_root
  export ASCEND_HOME_PATH="$cann_root"

  local had_nounset=0
  case $- in
    *u*) had_nounset=1; set +u ;;
  esac
  # shellcheck disable=SC1090
  source "$cann_root/set_env.sh"
  if [[ "$had_nounset" == "1" ]]; then
    set -u
  fi

  prepend_path "$cann_root/tools/bisheng_compiler/bin"

  export PYTHONNOUSERSITE=1
  export TRITON_BISHENGIR_DISABLE_LIB_CALL_NOINLINE=1
  export TRITON_DISABLE_FFTS=1
  export TRITON_SIMULATOR_CLEAN_EXIT=1
  export TRITON_ALWAYS_COMPILE=1
  export TRITON_DEBUG=1
  unset TRITON_COMPILE_ONLY
}

configure_ptoas_env() {
  local ptoas_bin="$1"
  local ptoas_bin_dir
  ptoas_bin_dir="$(dirname -- "$ptoas_bin")"

  prepend_path "$ptoas_bin_dir"
  prepend_ld_library_path "$ptoas_bin_dir/../lib"
  prepend_ld_library_path "$ptoas_root/build/lib"
  prepend_ld_library_path "$ptoas_root/PTOAS_Markham/build/lib"
  prepend_ld_library_path "$cann_root/tools/simulator/$soc_version/lib"
  prepend_ld_library_path "$cann_root/runtime/lib64/stub"
  prepend_ld_library_path "$cann_root/lib64"

  export TRITON_OBJCOPY_PATH="${TRITON_OBJCOPY_PATH:-/usr/bin/objcopy}"
  export TRITON_AICORE_LD_PATH="${TRITON_AICORE_LD_PATH:-$cann_root/tools/bisheng_compiler/bin/ld.lld}"
  [[ -x "$TRITON_OBJCOPY_PATH" ]] || die "TRITON_OBJCOPY_PATH is not executable: $TRITON_OBJCOPY_PATH"
  [[ -x "$TRITON_AICORE_LD_PATH" ]] || die "TRITON_AICORE_LD_PATH is not executable: $TRITON_AICORE_LD_PATH"
}

case_dir_for() {
  local case_name="$1"
  local case_dir="$testcase_root/$case_name"
  [[ -d "$case_dir" ]] || die "testcase directory not found: $case_dir"
  printf '%s\n' "$case_dir"
}

find_python_file() {
  local case_dir="$1"
  local matches=()
  local file
  shopt -s nullglob
  for file in "$case_dir"/*.py; do
    if grep -q '@triton\.jit' "$file"; then
      matches+=("$file")
    fi
  done
  shopt -u nullglob

  [[ ${#matches[@]} -gt 0 ]] || die "no Triton Python file with @triton.jit found in $case_dir"
  [[ ${#matches[@]} -eq 1 ]] || die "multiple Triton Python files found in $case_dir; keep one per testcase"
  printf '%s\n' "${matches[0]}"
}

find_kernel_name() {
  local python_file="$1"
  local kernels=()
  local name
  while IFS= read -r name; do
    kernels+=("$name")
  done < <(
    awk '
      /@triton\.jit/ { want_def = 1; next }
      want_def && /^[[:space:]]*def[[:space:]]+[A-Za-z_][A-Za-z0-9_]*/ {
        sub(/^[[:space:]]*def[[:space:]]+/, "")
        sub(/\(.*/, "")
        print
        want_def = 0
      }
    ' "$python_file"
  )

  [[ ${#kernels[@]} -gt 0 ]] || die "could not infer Triton kernel name from $python_file"
  [[ ${#kernels[@]} -eq 1 ]] || die "multiple @triton.jit kernels found in $python_file; keep one per testcase"
  printf '%s\n' "${kernels[0]}"
}

write_command() {
  local path="$1"
  shift
  {
    printf 'working_directory=%q\n' "$planner_root"
    printf 'command='
    printf '%q ' "$@"
    printf '\n'
  } >"$path"
}

run_logged() {
  local log_file="$1"
  local status
  shift
  if "$@" >"$log_file" 2>&1; then
    return 0
  else
    status=$?
    die "command failed with exit code $status; see $log_file"
  fi
}

compile_flags() {
  printf '%s\0' \
    "--target=$soc_version" \
    "--enable-auto-multi-buffer=true" \
    "--enable-auto-bind-sub-block=true" \
    "--disable-ffts" \
    "--limit-auto-multi-buffer-of-local-buffer=no-limit" \
    "--enable-auto-blockify-loop" \
    "--enable-hfusion-compile=true" \
    "--enable-hivm-compile=true" \
    "--enable-triton-kernel-compile=true" \
    "--mlir-disable-threading" \
    "--mlir-print-stacktrace-on-diagnostic" \
    "--enable-vf-merge-level=1"
}

run_python_simulator() {
  local case_dir="$1"
  local build_dir="$2"
  local compile_flow="$3"
  local python_file kernel_name sim_out sim_log ptoas_bin

  case "$compile_flow" in
    npuir)
      sim_out="$build_dir/npu-python"
      ;;
    ptoas)
      sim_out="$build_dir/ptoas-python"
      ;;
    *)
      die "unsupported Triton compile flow: $compile_flow"
      ;;
  esac

  python_file="$(find_python_file "$case_dir")"
  kernel_name="$(find_kernel_name "$python_file")"
  sim_log="$sim_out/msprof.log"

  source_cann_env
  activate_simulator_venv
  select_npu_compiler "$compile_flow" >/dev/null
  if [[ "$compile_flow" == "ptoas" ]]; then
    ptoas_bin="$(find_ptoas)"
    configure_ptoas_env "$ptoas_bin"
    export TRITON_PTOAS_PATH="$ptoas_bin"
  else
    unset TRITON_PTOAS_PATH
  fi

  mkdir -p "$sim_out/cache" "$sim_out/dump" "$sim_out/logs" "$sim_out/profile"
  chmod 700 "$sim_out" "$sim_out/cache" "$sim_out/dump" "$sim_out/logs" "$sim_out/profile" 2>/dev/null || true

  export TRITON_CACHE_DIR="$sim_out/cache"
  export TRITON_DUMP_DIR="$sim_out/dump"
  export ASCEND_PROCESS_LOG_PATH="$sim_out/logs"
  export TRITON_ASCEND_COMPILE_FLOW="$compile_flow"
  export TRITON_ASCEND_ARCH="$soc_version"
  unset BISHENGIR_ENABLE_PTOAS_BRIDGE

  local cmd=(
    msprof op simulator
    "--kernel-name=$kernel_name"
    "--soc-version=$soc_version"
    "--core-id=$core_id"
    "--output=$sim_out/profile"
    python3
    "$python_file"
  )
  write_command "$sim_out/command.txt" "${cmd[@]}"
  {
    printf 'NPU_SIM_VENV=%q\n' "$NPU_SIM_VENV"
    printf 'python3=%q\n' "$(command -v python3)"
    printf 'bishengir-compile=%q\n' "$(command -v bishengir-compile)"
    printf 'TRITON_ASCEND_COMPILE_FLOW=%q\n' "$TRITON_ASCEND_COMPILE_FLOW"
    printf 'TRITON_ASCEND_ARCH=%q\n' "$TRITON_ASCEND_ARCH"
    if [[ -n "${TRITON_PTOAS_PATH:-}" ]]; then
      printf 'TRITON_PTOAS_PATH=%q\n' "$TRITON_PTOAS_PATH"
    fi
  } >>"$sim_out/command.txt"

  log "$compile_flow simulator: $(basename -- "$python_file")"
  run_logged "$sim_log" "${cmd[@]}"
  log "simulator log: $sim_log"
}

run_emit_vmi() {
  local case_dir="$1"
  local case_name="$2"
  local print_ir_after_all="$3"
  local out_dir="$case_dir/out"
  local build_dir="$out_dir/build"
  local log_file="$build_dir/emit-vmi.log"
  local vmi_file="$out_dir/$case_name.vmi.mlir"
  local input="$case_dir/compile-input.mlir"
  local bishengir_compile

  if [[ ! -f "$input" ]]; then
    input="$case_dir/input.mlir"
  fi
  [[ -f "$input" ]] || die "missing compile-input.mlir or input.mlir in $case_dir"
  source_cann_env
  bishengir_compile="$(find_bishengir_compile)"
  prioritize_path "$(dirname -- "$bishengir_compile")"
  mkdir -p "$build_dir/temps-vmi"

  mapfile -d '' flags < <(compile_flags)
  local cmd=(
    "$bishengir_compile"
    "$input"
    "${flags[@]}"
    "--emit-ptoas-vmi"
  )
  if [[ "$print_ir_after_all" == "1" ]]; then
    cmd+=("--mlir-print-ir-after-all")
    log_file="$out_dir/after-all.log"
  fi
  cmd+=(
    "--save-temps=$build_dir/temps-vmi"
    "-o"
    "$vmi_file"
  )
  write_command "$build_dir/emit-vmi.command.txt" "${cmd[@]}"

  log "bishengir-compile -> PTOAS VMI"
  run_logged "$log_file" env -u BISHENGIR_ENABLE_PTOAS_BRIDGE "${cmd[@]}"
  [[ -s "$vmi_file" ]] || die "bishengir-compile did not create $vmi_file; see $log_file"
  log "wrote $vmi_file"
  if [[ "$print_ir_after_all" == "1" ]]; then
    log "IR dump log: $log_file"
  fi
}

run_emit_vpto() {
  local case_dir="$1"
  local case_name="$2"
  local out_dir="$case_dir/out"
  local build_dir="$out_dir/build"
  local vmi_file="$out_dir/$case_name.vmi.mlir"
  local vpto_file="$out_dir/$case_name.vpto.mlir"
  local ptoas_bin

  [[ -f "$vmi_file" ]] || run_emit_vmi "$case_dir" "$case_name" 0
  source_cann_env
  ptoas_bin="$(find_ptoas)"
  configure_ptoas_env "$ptoas_bin"

  local cmd=(
    "$ptoas_bin"
    "--pto-arch=a5"
    "--pto-backend=vpto"
    "--emit-vpto"
    "$vmi_file"
    "-o"
    "$vpto_file"
  )
  mkdir -p "$build_dir"
  write_command "$build_dir/emit-vpto.command.txt" "${cmd[@]}"
  log "PTOAS VMI -> VPTO"
  run_logged "$build_dir/emit-vpto.log" "${cmd[@]}"
  log "wrote $vpto_file"
}

clean_build=0
print_ir_after_all=0
args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --clean-build|clean-build)
      clean_build=1
      shift
      ;;
    --print-ir-after-all|print-ir-after-all)
      print_ir_after_all=1
      shift
      ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        args+=("$1")
        shift
      done
      ;;
    *)
      args+=("$1")
      shift
      ;;
  esac
done
set -- "${args[@]}"

if [[ $# -ne 2 ]]; then
  usage
  exit 1
fi

option="${1#--}"
case_name="$2"
case_dir="$(case_dir_for "$case_name")"
out_dir="$case_dir/out"
build_dir="$out_dir/build"
if [[ "$clean_build" == "1" ]]; then
  clean_build_dirs "$case_dir" "$out_dir"
fi
mkdir -p "$build_dir"

if [[ "$print_ir_after_all" == "1" && "$option" != "emit-vmi" && "$option" != "vmi" ]]; then
  die "--print-ir-after-all is supported with emit-vmi only; use print-ir <testcase>"
fi

case "$option" in
  npu-sim|baseline-sim)
    run_python_simulator "$case_dir" "$build_dir" npuir
    ;;
  bridge-sim|ptoas-sim)
    run_python_simulator "$case_dir" "$build_dir" ptoas
    ;;
  emit-vmi|vmi)
    run_emit_vmi "$case_dir" "$case_name" "$print_ir_after_all"
    ;;
  emit-vpto|vpto)
    run_emit_vpto "$case_dir" "$case_name"
    ;;
  print-ir|print-all|after-all)
    run_emit_vmi "$case_dir" "$case_name" 1
    ;;
  *)
    die "unknown option: $1"
    ;;
esac
