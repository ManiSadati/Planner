#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
planner_root="$(cd -- "${script_dir}/../.." && pwd)"

input_dir="${INPUT_DIR:-${planner_root}/bridge/triton-example}"
output_root="${OUTPUT_ROOT:-${planner_root}/bridge/examples/npuir-early-ir/replay}"
bishengir_compile="${BISHENGIR_COMPILE:-${HOME}/AscendNPU-IR/build/install/bin/bishengir-compile}"
npu_target="${NPU_TARGET:-Ascend910_9589}"
target_pass="${TARGET_PASS:-convert-hivmave-to-ave-intrin}"
print_after_all="${PRINT_AFTER_ALL:-1}"

mkdir -p "${output_root}"

if [[ ! -x "${bishengir_compile}" ]]; then
  cat >&2 <<EOF
error: bishengir-compile not found or not executable:
  ${bishengir_compile}

Set BISHENGIR_COMPILE to the local build/install binary path.
EOF
  exit 1
fi

shopt -s nullglob
inputs=("${input_dir}"/*_kernel.mlir)

if [[ ${#inputs[@]} -eq 0 ]]; then
  echo "error: no *_kernel.mlir files found in ${input_dir}" >&2
  exit 1
fi

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
      if ($0 ~ /^\/\/ -----\/\/ IR Dump After/ && index($0, pass) != 0) {
        capture = 1
        count++
        n = 1
        buf[n] = $0
      }
      next
    }

    /^\[/ {
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

common_flags=(
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
)

if [[ "${print_after_all}" == "1" ]]; then
  common_flags+=("--mlir-print-ir-after-all")
else
  common_flags+=("--mlir-print-ir-after=${target_pass}")
fi

if [[ -n "${AICORE_LIBDEVICE_BC:-}" ]]; then
  common_flags+=("--append-bisheng-options=-cce-link-aicore-ll-module ${AICORE_LIBDEVICE_BC}")
fi

if [[ -n "${EXTRA_BISHENGIR_COMPILE_FLAGS:-}" ]]; then
  # Intentional shell-word split for simple extra flag experiments.
  # Do not put paths with spaces in this variable.
  read -r -a extra_flags <<<"${EXTRA_BISHENGIR_COMPILE_FLAGS}"
  common_flags+=("${extra_flags[@]}")
fi

for input in "${inputs[@]}"; do
  name="$(basename "${input}" .mlir)"
  case_dir="${output_root}/${name}"
  mkdir -p "${case_dir}/temps"

  cp "${input}" "${case_dir}/source-after-append-device-spec.mlir"

  cmd=(
    "${bishengir_compile}"
    "${input}"
    "${common_flags[@]}"
    "--save-temps=${case_dir}/temps"
    "-o"
    "${case_dir}/${name}"
  )

  {
    printf 'working_directory=%q\n' "${planner_root}"
    printf 'command='
    printf '%q ' "${cmd[@]}"
    printf '\n'
  } >"${case_dir}/command.txt"

  echo "[replay] ${name}"
  compile_status=0
  if "${cmd[@]}" >"${case_dir}/compile.log" 2>&1; then
    echo "0" >"${case_dir}/compile-exit-code.txt"
  else
    compile_status=$?
    echo "${compile_status}" >"${case_dir}/compile-exit-code.txt"
  fi

  target_dump="${case_dir}/after-${target_pass}.mlir"
  dump_count_file="${case_dir}/after-${target_pass}-dump-count.txt"
  if extract_target_dump "${case_dir}/compile.log" "${target_dump}" "${target_pass}" "${dump_count_file}"; then
    dump_count="$(cat "${dump_count_file}")"
    echo "  target dump: ${target_dump}"
    if [[ "${dump_count}" != "1" ]]; then
      echo "  note: extracted last of ${dump_count} matching target-pass dumps"
    fi
    if [[ "${compile_status}" != "0" ]]; then
      echo "  note: compiler exited ${compile_status} after/around replay; target dump was still captured"
    fi
  else
    echo "  missing target dump for ${target_pass}: ${case_dir}/compile.log" >&2
    if [[ "${compile_status}" != "0" ]]; then
      echo "  compiler exit ${compile_status}" >&2
    fi
  fi
done
