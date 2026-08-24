#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
planner_root="$(cd -- "${script_dir}/../.." && pwd)"
workspace_root="$(cd -- "${planner_root}/.." && pwd)"

testcase_root="${TESTCASE_ROOT:-${planner_root}/bridge/testcases}"
output_root="${OUTPUT_ROOT:-${planner_root}/bridge/out/npuir-ptoas-bridge}"
bishengir_opt="${BISHENGIR_OPT:-${workspace_root}/AscendNPU-IR/build/bin/bishengir-opt}"
bishengir_compile="${BISHENGIR_COMPILE:-${workspace_root}/AscendNPU-IR/build/bin/bishengir-compile}"
ptoas_bin="${PTOAS_BIN:-${workspace_root}/PTOAS/build/tools/ptoas/ptoas}"
npu_target="${NPU_TARGET:-Ascend910_9589}"
pto_arch="${PTO_ARCH:-a5}"
pto_backend="${PTO_BACKEND:-vpto}"
target_pass="${TARGET_PASS:-convert-hivmave-to-ptoas-vmi}"

use_bishengir_compile=1
emit_vpto=0
emit_llvmir=0
run_simulator=0
clean=0
cases=()

if [[ -z "${NO_COLOR:-}" && ( -t 1 || -t 2 || "${FORCE_COLOR:-0}" == "1" ) ]]; then
  color_reset=$'\033[0m'
  color_red=$'\033[31m'
  color_yellow=$'\033[33m'
  color_green=$'\033[32m'
  color_cyan=$'\033[36m'
  color_dim=$'\033[2m'
else
  color_reset=""
  color_red=""
  color_yellow=""
  color_green=""
  color_cyan=""
  color_dim=""
fi

log_info() {
  printf '%s[bridge]%s %s\n' "${color_cyan}" "${color_reset}" "$*"
}

log_output() {
  printf '  %s%s%s\n' "${color_green}" "$*" "${color_reset}"
}

log_note() {
  printf '  %snote:%s %s\n' "${color_yellow}" "${color_reset}" "$*"
}

log_error() {
  printf '%serror:%s %s\n' "${color_red}" "${color_reset}" "$*" >&2
}

log_detail() {
  printf '  %s%s%s\n' "${color_dim}" "$*" "${color_reset}"
}

usage() {
  cat <<EOF
usage: $(basename "$0") [options] [case ...]

Runs Planner bridge testcases through:
  NPU-IR MLIR --convert-hivmave-to-ptoas-vmi -> PTOAS

Options:
  --from-bishengir-compile    Run compile-input.mlir, compile_input.mlir, or
                              input.mlir through
                              bishengir-compile and extract the dump after
                              ${target_pass} (default).
  --from-bishengir-opt         Run input.mlir through bishengir-opt and the bridge
                              pass directly.
  --emit-vpto                 Ask PTOAS to write final VPTO MLIR.
  --emit-llvmir               Ask PTOAS to write translated VPTO LLVM IR.
  --run-simulator, --sim      Run the testcase simulator fixture after VPTO emission.
  --all                       Enable --emit-vpto, --emit-llvmir, and --run-simulator.
  --clean, --clean-build      Remove testcase output/build directory before running.
  --testcase-root DIR         Testcase root. Default: ${testcase_root}
  --output-root DIR           Output root. Default: ${output_root}
  --bishengir-opt PATH        bishengir-opt path. Default: ${bishengir_opt}
  --bishengir-compile PATH    bishengir-compile path. Default: ${bishengir_compile}
  --ptoas PATH                ptoas path. Default: ${ptoas_bin}
  --npu-target TARGET         bishengir-compile target. Default: ${npu_target}
  --pto-arch ARCH             PTOAS architecture. Default: ${pto_arch}
  --pto-backend BACKEND       PTOAS backend. Default: ${pto_backend}
  -h, --help                  Show this help.

Environment overrides:
  TESTCASE_ROOT, OUTPUT_ROOT, BISHENGIR_OPT, BISHENGIR_COMPILE, PTOAS_BIN,
  NPU_TARGET, PTO_ARCH, PTO_BACKEND, EXTRA_BISHENGIR_COMPILE_FLAGS,
  NO_COLOR, FORCE_COLOR

Color output is enabled for terminals by default. Set NO_COLOR=1 to disable it
or FORCE_COLOR=1 to enable it when output is redirected.

If no emission/execution option is selected, --emit-vpto is used.
If no case is named, every direct child of TESTCASE_ROOT containing input.mlir or
a compile input is run in compile mode. Compile mode prefers compile-input.mlir,
then compile_input.mlir, then input.mlir. Opt mode requires input.mlir.
EOF
}

require_option_arg() {
  local option="$1"
  if [[ $# -lt 2 || -z "${2:-}" ]]; then
    log_error "${option} requires an argument"
    usage >&2
    exit 1
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --from-bishengir-compile|--use-bishengir-compile)
      use_bishengir_compile=1
      shift
      ;;
    --from-bishengir-opt|--use-bishengir-opt)
      use_bishengir_compile=0
      shift
      ;;
    --emit-vpto)
      emit_vpto=1
      shift
      ;;
    --emit-llvmir|--emit-vpto-llvm-ir)
      emit_llvmir=1
      shift
      ;;
    --run-simulator|--sim)
      run_simulator=1
      shift
      ;;
    --all)
      emit_vpto=1
      emit_llvmir=1
      run_simulator=1
      shift
      ;;
    --clean|--clean-build)
      clean=1
      shift
      ;;
    --testcase-root)
      require_option_arg "$@"
      testcase_root="$2"
      shift 2
      ;;
    --output-root)
      require_option_arg "$@"
      output_root="$2"
      shift 2
      ;;
    --bishengir-opt)
      require_option_arg "$@"
      bishengir_opt="$2"
      shift 2
      ;;
    --bishengir-compile)
      require_option_arg "$@"
      bishengir_compile="$2"
      shift 2
      ;;
    --ptoas)
      require_option_arg "$@"
      ptoas_bin="$2"
      shift 2
      ;;
    --npu-target)
      require_option_arg "$@"
      npu_target="$2"
      shift 2
      ;;
    --pto-arch)
      require_option_arg "$@"
      pto_arch="$2"
      shift 2
      ;;
    --pto-backend)
      require_option_arg "$@"
      pto_backend="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        cases+=("$1")
        shift
      done
      ;;
    -*)
      log_error "unknown option: $1"
      usage >&2
      exit 1
      ;;
    *)
      cases+=("$1")
      shift
      ;;
  esac
done

if [[ "${emit_vpto}" == "0" && "${emit_llvmir}" == "0" && "${run_simulator}" == "0" ]]; then
  emit_vpto=1
fi

if [[ "${run_simulator}" == "1" ]]; then
  emit_vpto=1
fi

if [[ "${use_bishengir_compile}" == "0" && ! -x "${bishengir_opt}" ]]; then
  log_error "bishengir-opt not found or not executable: ${bishengir_opt}"
  log_detail "Set BISHENGIR_OPT=/path/to/bishengir-opt."
  exit 1
fi

if [[ "${use_bishengir_compile}" == "1" && ! -x "${bishengir_compile}" ]]; then
  log_error "bishengir-compile not found or not executable: ${bishengir_compile}"
  log_detail "Set BISHENGIR_COMPILE=/path/to/bishengir-compile."
  exit 1
fi

if [[ ! -x "${ptoas_bin}" ]]; then
  if command -v ptoas >/dev/null 2>&1; then
    ptoas_bin="$(command -v ptoas)"
  else
    log_error "ptoas not found or not executable: ${ptoas_bin}"
    log_detail "Set PTOAS_BIN=/path/to/ptoas or put ptoas in PATH."
    exit 1
  fi
fi

mkdir -p "${output_root}"

if [[ ${#cases[@]} -eq 0 ]]; then
  shopt -s nullglob
  for case_dir in "${testcase_root}"/*; do
    if [[ "${use_bishengir_compile}" == "1" &&
          ( -f "${case_dir}/compile-input.mlir" ||
            -f "${case_dir}/compile_input.mlir" ||
            -f "${case_dir}/input.mlir" ) ]]; then
      cases+=("$(basename "${case_dir}")")
    elif [[ "${use_bishengir_compile}" == "0" && -f "${case_dir}/input.mlir" ]]; then
      cases+=("$(basename "${case_dir}")")
    fi
  done
  shopt -u nullglob
fi

if [[ ${#cases[@]} -eq 0 ]]; then
  log_error "no bridge testcases found under ${testcase_root}"
  exit 1
fi

print_command_file() {
  local path="$1"
  shift
  {
    printf 'working_directory=%q\n' "${workspace_root}"
    printf 'command='
    printf '%q ' "$@"
    printf '\n'
  } >"${path}"
}

run_logged() {
  local log_file="$1"
  shift
  if "$@" >"${log_file}" 2>&1; then
    return 0
  else
    local status=$?
    log_error "command failed with exit code ${status}; see ${log_file}"
    return "${status}"
  fi
}

extract_target_dump() {
  local log_file="$1"
  local output_file="$2"
  local pass_name="$3"
  local count_file="$4"
  local tmp_file="${output_file}.tmp"
  local tmp_count_file="${count_file}.tmp"

  if awk -v pass="(${pass_name})" -v count_file="${tmp_count_file}" '
    function flush_candidate(  i) {
      if (capture && n > 0) {
        for (i = 1; i <= n; i++) {
          last[i] = buf[i]
        }
        last_n = n
      }
      capture = 0
      n = 0
    }

    BEGIN {
      capture = 0
      count = 0
      last_n = 0
      n = 0
    }

    /^\/\/ -----\/\/ IR Dump (After|Before)/ {
      flush_candidate()
      if ($0 ~ /^\/\/ -----\/\/ IR Dump After/ &&
          $0 !~ / Failed / &&
          index($0, pass) != 0) {
        capture = 1
        count++
        next
      }
      next
    }

    /^\[/ {
      flush_candidate()
      next
    }

    /^(hivmc|error:|warning:|loc\()/ {
      flush_candidate()
      next
    }

    capture {
      n++
      buf[n] = $0
      next
    }

    END {
      flush_candidate()
      if (last_n == 0) {
        exit 1
      }
      for (i = 1; i <= last_n; i++) {
        print last[i]
      }
      print count > count_file
    }
  ' "${log_file}" >"${tmp_file}"; then
    mv "${tmp_file}" "${output_file}"
    mv "${tmp_count_file}" "${count_file}"
  else
    rm -f "${tmp_file}"
    rm -f "${tmp_count_file}"
    return 1
  fi
}

bishengir_compile_flags=(
  "--target=${npu_target}"
  "--enable-auto-multi-buffer=true"
  "--enable-auto-bind-sub-block=true"
  "--disable-ffts"
  "--limit-auto-multi-buffer-of-local-buffer=no-limit"
  "--enable-auto-blockify-loop"
  "--enable-hfusion-compile=true"
  "--enable-hivm-compile=true"
  "--enable-triton-kernel-compile=true"
  "--mlir-disable-threading"
  "--mlir-print-stacktrace-on-diagnostic"
  "--enable-vf-merge-level=1"
  "--mlir-print-ir-after=${target_pass}"
)

if [[ -n "${AICORE_LIBDEVICE_BC:-}" ]]; then
  bishengir_compile_flags+=("--append-bisheng-options=-cce-link-aicore-ll-module ${AICORE_LIBDEVICE_BC}")
fi

if [[ -n "${EXTRA_BISHENGIR_COMPILE_FLAGS:-}" ]]; then
  read -r -a extra_flags <<<"${EXTRA_BISHENGIR_COMPILE_FLAGS}"
  bishengir_compile_flags+=("${extra_flags[@]}")
fi

for case_name in "${cases[@]}"; do
  case_dir="${testcase_root}/${case_name}"
  input_mlir="${case_dir}/input.mlir"
  compile_input_mlir="${case_dir}/compile-input.mlir"
  compile_input_alt_mlir="${case_dir}/compile_input.mlir"
  case_out="${output_root}/${case_name}"
  vmi_mlir="${case_out}/${case_name}.vmi.mlir"
  vpto_mlir="${case_out}/${case_name}.vpto.mlir"
  llvmir_file="${case_out}/${case_name}.vpto.ll"

  if [[ "${clean}" == "1" ]]; then
    rm -rf "${case_out}"
  fi

  if [[ "${use_bishengir_compile}" == "0" && ! -f "${input_mlir}" ]]; then
    log_error "missing testcase input: ${input_mlir}"
    exit 1
  fi

  compile_source_mlir="${compile_input_mlir}"
  if [[ "${use_bishengir_compile}" == "1" &&
        ! -f "${compile_source_mlir}" ]]; then
    compile_source_mlir="${compile_input_alt_mlir}"
  fi
  if [[ "${use_bishengir_compile}" == "1" &&
        ! -f "${compile_source_mlir}" ]]; then
    compile_source_mlir="${input_mlir}"
  fi
  if [[ "${use_bishengir_compile}" == "1" &&
        ! -f "${compile_source_mlir}" ]]; then
    log_error "missing testcase compile input: ${compile_input_mlir}, ${compile_input_alt_mlir}, or ${input_mlir}"
    exit 1
  fi

  mkdir -p "${case_out}"

  if [[ "${use_bishengir_compile}" == "1" ]]; then
    cp "${compile_source_mlir}" "${case_out}/compile-input.mlir"
    log_info "${case_name}: bishengir-compile -> VMI dump"
    npuir_cmd=(
      "${bishengir_compile}"
      "${compile_source_mlir}"
      "${bishengir_compile_flags[@]}"
      "--save-temps=${case_out}/temps"
      "-o"
      "${case_out}/${case_name}"
    )
    print_command_file "${case_out}/npuir-command.txt" "${npuir_cmd[@]}"
    compile_status=0
    if "${npuir_cmd[@]}" >"${case_out}/npuir.log" 2>&1; then
      echo "0" >"${case_out}/npuir-exit-code.txt"
    else
      compile_status=$?
      echo "${compile_status}" >"${case_out}/npuir-exit-code.txt"
    fi

    dump_count_file="${case_out}/after-${target_pass}-dump-count.txt"
    if ! extract_target_dump "${case_out}/npuir.log" "${vmi_mlir}" "${target_pass}" "${dump_count_file}"; then
      log_error "no successful dump after ${target_pass}; see ${case_out}/npuir.log"
      exit 1
    fi

    if [[ "${compile_status}" != "0" ]]; then
      log_note "bishengir-compile exited ${compile_status}; continuing with extracted ${target_pass} dump"
    fi
  else
    cp "${input_mlir}" "${case_out}/input.mlir"
    log_info "${case_name}: NPU-IR -> VMI"
    npuir_cmd=(
      "${bishengir_opt}"
      "${input_mlir}"
      "--convert-hivmave-to-ptoas-vmi"
      "-o"
      "${vmi_mlir}"
    )
    print_command_file "${case_out}/npuir-command.txt" "${npuir_cmd[@]}"
    run_logged "${case_out}/npuir.log" "${npuir_cmd[@]}"
  fi
  log_output "vmi: ${vmi_mlir}"

  if [[ "${emit_vpto}" == "1" ]]; then
    log_info "${case_name}: VMI -> VPTO"
    vpto_cmd=(
      "${ptoas_bin}"
      "--pto-arch=${pto_arch}"
      "--pto-backend=${pto_backend}"
      "--emit-vpto"
      "${vmi_mlir}"
      "-o"
      "${vpto_mlir}"
    )
    print_command_file "${case_out}/ptoas-vpto-command.txt" "${vpto_cmd[@]}"
    run_logged "${case_out}/ptoas-vpto.log" "${vpto_cmd[@]}"
    log_output "vpto: ${vpto_mlir}"
  fi

  if [[ "${emit_llvmir}" == "1" ]]; then
    log_info "${case_name}: VMI -> VPTO LLVM IR"
    llvmir_cmd=(
      "${ptoas_bin}"
      "--pto-arch=${pto_arch}"
      "--pto-backend=${pto_backend}"
      "--emit-vpto-llvm-ir"
      "${vmi_mlir}"
      "-o"
      "${llvmir_file}"
    )
    print_command_file "${case_out}/ptoas-llvmir-command.txt" "${llvmir_cmd[@]}"
    run_logged "${case_out}/ptoas-llvmir.log" "${llvmir_cmd[@]}"
    log_output "llvmir: ${llvmir_file}"
  fi

  if [[ "${run_simulator}" == "1" ]]; then
    log_info "${case_name}: simulator"
    sim_src="${case_out}/sim"
    vpto_mlir_for_sim="$(cd -- "$(dirname -- "${vpto_mlir}")" && pwd)/$(basename -- "${vpto_mlir}")"
    rm -rf "${sim_src}"
    mkdir -p "${sim_src}"
    cp -R "${case_dir}/." "${sim_src}/"
    rm -rf "${sim_src}/build"
    sim_cmd=(
      env
      "PTOAS_BIN=${ptoas_bin}"
      "KERNEL_MLIR=${vpto_mlir_for_sim}"
      "TESTCASE_NAME=${case_name}"
      "BUILD_DIR=${sim_src}/build"
      "RUN_DIR=${sim_src}/build/run"
      bash
      "${sim_src}/run_sim.sh"
    )
    print_command_file "${case_out}/sim-command.txt" "${sim_cmd[@]}"
    run_logged "${case_out}/sim.log" "${sim_cmd[@]}"
    log_output "simulator log: ${case_out}/sim.log"
  fi
done
