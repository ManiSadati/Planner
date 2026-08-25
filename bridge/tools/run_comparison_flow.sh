#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
planner_root="$(cd -- "${script_dir}/../.." && pwd)"
testcase_root="${planner_root}/bridge/testcases"

npuir_root="${ASCEND_NPU_IR_ROOT:-${NPU_IR_ROOT:-}}"
cann_root="${CANN_ROOT:-${ASCEND_HOME_PATH:-}}"
ptoas_root="${PTOAS_ROOT:-}"

npu_target="Ascend910_9589"
npu_sim_soc="Ascend950PR_9589"
ptoas_sim_soc="Ascend950PR_9599"
core_id="0"
target_pass="convert-hivmave-to-ptoas-vmi"

usage() {
  cat >&2 <<EOF
Usage:
  bridge/tools/run_comparison_flow.sh <option> <testcase>

Options:
  early-ir     Generate <testcase>/input.mlir from the Triton Python testcase.
  print-all    Run bishengir-compile and save the print-after-all log.
  npu-sim      Run the Triton Python testcase through the NPU-IR simulator.
  emit-vmi     Emit PTOAS VMI MLIR from <testcase>/input.mlir.
  emit-vpto    Emit PTOAS VPTO MLIR from the VMI MLIR.
  bridge-sim   Run input.mlir -> VMI -> VPTO -> PTOAS simulator fixture.

Required environment:
  ASCEND_NPU_IR_ROOT=/path/to/AscendNPU-IR
  CANN_ROOT=/path/to/CANN
  PTOAS_ROOT=/path/to/PTOAS

Testcase layout:
  Planner/bridge/testcases/<testcase>/
    <one Triton Python file with one @triton.jit kernel>
    input.mlir                 created by early-ir, used by compile options
    run_sim.sh                 auto-created by bridge-sim when fixture files exist

Outputs:
  Planner/bridge/testcases/<testcase>/input.mlir
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

find_cmake() {
  local candidates=()
  local candidate

  if [[ -n "${CMAKE_BIN:-}" ]]; then
    candidates+=("$CMAKE_BIN")
  fi
  while IFS= read -r candidate; do
    candidates+=("$candidate")
  done < <(type -P -a cmake 2>/dev/null || true)
  candidates+=("/usr/bin/cmake" "/bin/cmake")

  for candidate in "${candidates[@]}"; do
    if [[ -x "$candidate" ]] && "$candidate" --version >/dev/null 2>&1; then
      abs_path "$candidate"
      return 0
    fi
  done

  die "cannot find a working cmake binary"
}

find_python_with_numpy() {
  local candidates=()
  local candidate root

  if [[ -n "${PYTHON_BIN:-}" ]]; then
    candidates+=("$PYTHON_BIN")
  fi
  while IFS= read -r candidate; do
    candidates+=("$candidate")
  done < <(type -P -a python3 python 2>/dev/null || true)

  shopt -s nullglob
  for root in "$HOME/miniconda3" "$HOME/anaconda3" /home/*/miniconda3 /home/*/anaconda3; do
    candidates+=(
      "$root/bin/python3"
      "$root/envs/ptoas/bin/python3"
      "$root/envs/triton/bin/python3"
      "$root/envs/pypto/bin/python3"
    )
  done
  shopt -u nullglob

  for candidate in "${candidates[@]}"; do
    if [[ -x "$candidate" ]] && "$candidate" -c 'import numpy' >/dev/null 2>&1; then
      abs_path "$candidate"
      return 0
    fi
  done

  die "cannot find a Python that can import numpy"
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

  prepend_path "$(dirname -- "$(find_bishengir_compile)")"
  prepend_path "$cann_root/tools/bisheng_compiler/bin"

  local venv
  for venv in "$npuir_root/.venv" "$HOME/.venv/npuir-sim-system"; do
    if [[ -f "$venv/bin/activate" ]]; then
      # shellcheck disable=SC1090
      source "$venv/bin/activate"
      break
    fi
  done

  export PYTHONNOUSERSITE=1
  export TRITON_ASCEND_ARCH="$npu_target"
  export TRITON_BISHENGIR_DISABLE_LIB_CALL_NOINLINE=1
  export TRITON_DISABLE_FFTS=1
  export TRITON_SIMULATOR_CLEAN_EXIT=1
  export TRITON_ALWAYS_COMPILE=1
  export TRITON_DEBUG=1
}

configure_ptoas_env() {
  local ptoas_bin="$1"
  local ptoas_bin_dir
  ptoas_bin_dir="$(dirname -- "$ptoas_bin")"

  prepend_path "$ptoas_bin_dir"
  prepend_ld_library_path "$ptoas_bin_dir/../lib"
  prepend_ld_library_path "$ptoas_root/build/lib"
  prepend_ld_library_path "$ptoas_root/PTOAS_Markham/build/lib"
  prepend_ld_library_path "$cann_root/tools/simulator/$ptoas_sim_soc/lib"
  prepend_ld_library_path "$cann_root/runtime/lib64/stub"
  prepend_ld_library_path "$cann_root/lib64"
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

maybe_kernel_name() {
  local case_dir="$1"
  local python_file
  if python_file="$(find_python_file "$case_dir" 2>/dev/null)"; then
    find_kernel_name "$python_file"
  fi
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
    "--target=$npu_target" \
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

extract_pass_dump() {
  local log_file="$1"
  local output_file="$2"
  local pass_name="$3"
  local count_file="$4"
  local tmp_file="${output_file}.tmp"
  local tmp_count_file="${count_file}.tmp"

  if awk -v pass="(${pass_name})" -v count_file="$tmp_count_file" '
    function flush_candidate(  i) {
      if (capture && n > 0) {
        for (i = 1; i <= n; i++) last[i] = buf[i]
        last_n = n
      }
      capture = 0
      n = 0
    }

    BEGIN { capture = 0; count = 0; last_n = 0; n = 0 }

    /^\/\/ -----\/\/ IR Dump (After|Before)/ {
      flush_candidate()
      if ($0 ~ /^\/\/ -----\/\/ IR Dump After/ &&
          $0 !~ / Failed / &&
          index($0, pass) != 0) {
        capture = 1
        count++
      }
      next
    }

    /^\[/ || /^(hivmc|error:|warning:|loc\()/ {
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
      if (last_n == 0) exit 1
      for (i = 1; i <= last_n; i++) print last[i]
      print count > count_file
    }
  ' "$log_file" >"$tmp_file"; then
    mv "$tmp_file" "$output_file"
    mv "$tmp_count_file" "$count_file"
  else
    rm -f "$tmp_file" "$tmp_count_file"
    return 1
  fi
}

latest_ttadapter_dump() {
  local dump_root="$1"
  find "$dump_root" -type f -name '*kernel.ttadapter.mlir' 2>/dev/null | sort | tail -n 1
}

run_python_simulator() {
  local case_dir="$1"
  local build_dir="$2"
  local stop_after_dump="$3"
  local python_file kernel_name sim_out sim_log dump_file

  python_file="$(find_python_file "$case_dir")"
  kernel_name="$(find_kernel_name "$python_file")"
  sim_out="$build_dir/npu-python"
  sim_log="$sim_out/msprof.log"

  source_cann_env
  mkdir -p "$sim_out/cache" "$sim_out/dump" "$sim_out/logs" "$sim_out/profile"
  chmod 700 "$sim_out" "$sim_out/cache" "$sim_out/dump" "$sim_out/logs" "$sim_out/profile" 2>/dev/null || true

  export TRITON_CACHE_DIR="$sim_out/cache"
  export TRITON_DUMP_DIR="$sim_out/dump"
  export ASCEND_PROCESS_LOG_PATH="$sim_out/logs"
  unset BISHENGIR_ENABLE_PTOAS_BRIDGE

  local cmd=(
    msprof op simulator
    "--kernel-name=$kernel_name"
    "--soc-version=$npu_sim_soc"
    "--core-id=$core_id"
    "--output=$sim_out/profile"
    python3
    "$python_file"
  )
  write_command "$sim_out/command.txt" "${cmd[@]}"

  if [[ "$stop_after_dump" == "0" ]]; then
    log "NPU-IR simulator: $(basename -- "$python_file")"
    run_logged "$sim_log" "${cmd[@]}"
    log "simulator log: $sim_log"
    return 0
  fi

  log "early IR: $(basename -- "$python_file")"
  local sim_pid kill_target found_dump=0 deadline status
  if command -v setsid >/dev/null 2>&1; then
    setsid "${cmd[@]}" >"$sim_log" 2>&1 &
    sim_pid=$!
    kill_target="-$sim_pid"
  else
    "${cmd[@]}" >"$sim_log" 2>&1 &
    sim_pid=$!
    kill_target="$sim_pid"
  fi

  deadline=$((SECONDS + 600))
  while kill -0 "$sim_pid" 2>/dev/null; do
    if [[ -n "$(latest_ttadapter_dump "$sim_out/dump")" ]]; then
      found_dump=1
      sleep 2
      break
    fi
    (( SECONDS < deadline )) || break
    sleep 1
  done

  if kill -0 "$sim_pid" 2>/dev/null; then
    kill -TERM -- "$kill_target" 2>/dev/null || kill -TERM "$sim_pid" 2>/dev/null || true
    sleep 2
    kill -KILL -- "$kill_target" 2>/dev/null || kill -KILL "$sim_pid" 2>/dev/null || true
  fi

  set +e
  wait "$sim_pid"
  status=$?
  set -e
  echo "$status" >"$sim_out/msprof-exit-code.txt"

  if [[ -n "$(latest_ttadapter_dump "$sim_out/dump")" ]]; then
    found_dump=1
  fi
  [[ "$found_dump" == "1" ]] || die "no TTAdapter MLIR dump captured; see $sim_log"
  dump_file="$(latest_ttadapter_dump "$sim_out/dump")"
  cp -a "$dump_file" "$case_dir/input.mlir"
  log "wrote $case_dir/input.mlir"
}

run_compile() {
  local case_dir="$1"
  local build_dir="$2"
  local mode="$3"
  local print_arg="$4"
  local output_stem="$5"
  local log_file="$6"
  local input="$case_dir/input.mlir"
  local bishengir_compile status

  [[ -f "$input" ]] || die "missing $input; run early-ir first"
  source_cann_env
  bishengir_compile="$(find_bishengir_compile)"
  mkdir -p "$build_dir/temps-$mode" "$(dirname -- "$output_stem")"

  mapfile -d '' flags < <(compile_flags)
  local cmd=(
    "$bishengir_compile"
    "$input"
    "${flags[@]}"
    "$print_arg"
    "--save-temps=$build_dir/temps-$mode"
    "-o"
    "$output_stem"
  )
  write_command "$build_dir/$mode.command.txt" "${cmd[@]}"

  set +e
  if [[ "$mode" == "vmi" ]]; then
    BISHENGIR_ENABLE_PTOAS_BRIDGE=1 "${cmd[@]}" >"$log_file" 2>&1
  else
    env -u BISHENGIR_ENABLE_PTOAS_BRIDGE "${cmd[@]}" >"$log_file" 2>&1
  fi
  status=$?
  set -e
  echo "$status" >"$build_dir/$mode.exit-code.txt"
  return "$status"
}

run_print_all() {
  local case_dir="$1"
  local out_dir="$case_dir/out"
  local build_dir="$out_dir/build"
  local log_file="$out_dir/after-all.log"
  mkdir -p "$build_dir"

  log "bishengir-compile print-after-all"
  if run_compile "$case_dir" "$build_dir" "after-all" "--mlir-print-ir-after-all" "$build_dir/after-all/kernel" "$log_file"; then
    log "wrote $log_file"
    return 0
  fi

  if grep -q "IR Dump After" "$log_file"; then
    log "compiler exited nonzero, but pass dumps were captured: $log_file"
    return 0
  fi
  die "compiler failed before pass dumps; see $log_file"
}

run_emit_vmi() {
  local case_dir="$1"
  local case_name="$2"
  local out_dir="$case_dir/out"
  local build_dir="$out_dir/build"
  local log_file="$build_dir/emit-vmi.log"
  local vmi_file="$out_dir/$case_name.vmi.mlir"
  local count_file="$build_dir/after-$target_pass.dump-count.txt"
  mkdir -p "$build_dir"

  log "bishengir-compile -> PTOAS VMI"
  if ! run_compile "$case_dir" "$build_dir" "vmi" "--mlir-print-ir-after=$target_pass" "$build_dir/vmi/kernel" "$log_file"; then
    log "compiler exited nonzero; trying to extract the requested pass dump"
  fi

  extract_pass_dump "$log_file" "$vmi_file" "$target_pass" "$count_file" ||
    die "could not extract a successful dump after $target_pass; see $log_file"
  log "wrote $vmi_file"
}

run_emit_vpto() {
  local case_dir="$1"
  local case_name="$2"
  local out_dir="$case_dir/out"
  local build_dir="$out_dir/build"
  local vmi_file="$out_dir/$case_name.vmi.mlir"
  local vpto_file="$out_dir/$case_name.vpto.mlir"
  local ptoas_bin

  [[ -f "$vmi_file" ]] || run_emit_vmi "$case_dir" "$case_name"
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

write_generated_run_sim() {
  local script_path="$1"

  cat >"$script_path" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

case_name="${CASE_NAME:-${TESTCASE_NAME:-$(basename -- "$script_dir")}}"
build_dir="${BUILD_DIR:-$script_dir/out/build/ptoas-sim}"
run_dir="${RUN_DIR:-$build_dir/run}"
soc_version="${SOC_VERSION:-Ascend950PR_9599}"
ptoas_bin="${PTOAS_BIN:?PTOAS_BIN is required}"
kernel_mlir="${KERNEL_MLIR:?KERNEL_MLIR is required}"
ascend_home="${ASCEND_HOME_PATH:?ASCEND_HOME_PATH is required}"
use_msprof="${USE_MSPROF:-0}"
kernel_name="${KERNEL_NAME:-}"
core_id="${CORE_ID:-0}"
msprof_output="${MSPROF_OUTPUT:-$build_dir/profile}"
cmake_bin="${CMAKE_BIN:-cmake}"
python_bin="${PYTHON_BIN:-python3}"

if [[ ! -x "$ptoas_bin" ]]; then
  echo "error: PTOAS_BIN is not executable: $ptoas_bin" >&2
  exit 1
fi
if [[ ! -f "$kernel_mlir" ]]; then
  echo "error: KERNEL_MLIR does not exist: $kernel_mlir" >&2
  exit 1
fi
if ! "$cmake_bin" --version >/dev/null 2>&1; then
  echo "error: CMAKE_BIN is not a working cmake binary: $cmake_bin" >&2
  exit 1
fi
if ! "$python_bin" -c 'import numpy' >/dev/null 2>&1; then
  echo "error: PYTHON_BIN cannot import numpy: $python_bin" >&2
  exit 1
fi

sim_lib_dir="${SIM_LIB_DIR:-$ascend_home/tools/simulator/$soc_version/lib}"
export LD_LIBRARY_PATH="$sim_lib_dir:$ascend_home/runtime/lib64/stub:$ascend_home/lib64:${LD_LIBRARY_PATH:-}"

"$cmake_bin" -S "$script_dir" -B "$build_dir" \
  -DSOC_VERSION="$soc_version" \
  -DPTOAS_BIN="$ptoas_bin" \
  -DKERNEL_MLIR="$kernel_mlir"
"$cmake_bin" --build "$build_dir" --parallel "${BUILD_JOBS:-$(nproc)}"

executable=""
for candidate in \
  "$build_dir/${case_name}_vpto" \
  "$build_dir/lowered_${case_name}_vpto"; do
  if [[ -x "$candidate" && ! -d "$candidate" ]]; then
    executable="$candidate"
    break
  fi
done

if [[ -z "$executable" ]]; then
  mapfile -t executable_candidates < <(
    find "$build_dir" -maxdepth 1 -type f -perm -111 \
      ! -name '*.so' \
      ! -name '*.a' \
      ! -name 'cmake*' \
      | sort
  )
  if [[ ${#executable_candidates[@]} -eq 1 ]]; then
    executable="${executable_candidates[0]}"
  fi
fi

if [[ -z "$executable" ]]; then
  echo "error: could not infer the built simulator executable in $build_dir" >&2
  echo "       either name it ${case_name}_vpto/lowered_${case_name}_vpto or edit run_sim.sh" >&2
  exit 1
fi

mkdir -p "$run_dir"
cd "$run_dir"
"$python_bin" "$script_dir/gen_data.py"

if [[ "$use_msprof" == "1" ]]; then
  [[ -n "$kernel_name" ]] || {
    echo "error: KERNEL_NAME is required when USE_MSPROF=1" >&2
    exit 1
  }
  mkdir -p "$msprof_output"
  chmod 700 "$msprof_output" 2>/dev/null || true
  msprof op simulator \
    --kernel-name="$kernel_name" \
    --soc-version="$soc_version" \
    --core-id="$core_id" \
    --output="$msprof_output" \
    --application="$executable"
else
  "$executable"
fi

"$python_bin" "$script_dir/compare.py"
EOF
  chmod +x "$script_path"
}

ensure_bridge_sim_fixture() {
  local case_dir="$1"
  local missing=()
  local required_file

  for required_file in CMakeLists.txt main.cpp launch.cpp gen_data.py compare.py; do
    if [[ ! -f "$case_dir/$required_file" ]]; then
      missing+=("$required_file")
    fi
  done

  if [[ ${#missing[@]} -gt 0 ]]; then
    die "bridge-sim needs a simulator fixture; first generate these files: ${missing[*]}"
  fi

  if [[ ! -f "$case_dir/run_sim.sh" ]]; then
    write_generated_run_sim "$case_dir/run_sim.sh"
    log "created $case_dir/run_sim.sh"
  fi
}

run_bridge_sim() {
  local case_dir="$1"
  local case_name="$2"
  local out_dir="$case_dir/out"
  local build_dir="$out_dir/build"
  local vpto_file="$out_dir/$case_name.vpto.mlir"
  local ptoas_bin kernel_name cmake_bin python_bin

  ensure_bridge_sim_fixture "$case_dir"
  [[ -f "$vpto_file" ]] || run_emit_vpto "$case_dir" "$case_name"

  source_cann_env
  ptoas_bin="$(find_ptoas)"
  cmake_bin="$(find_cmake)"
  python_bin="$(find_python_with_numpy)"
  configure_ptoas_env "$ptoas_bin"
  kernel_name="$(maybe_kernel_name "$case_dir" || true)"

  mkdir -p "$build_dir/ptoas-sim"
  local env_cmd=(
    env
    "PTOAS_BIN=$ptoas_bin"
    "KERNEL_MLIR=$vpto_file"
    "TESTCASE_NAME=$case_name"
    "CASE_NAME=$case_name"
    "BUILD_DIR=$build_dir/ptoas-sim"
    "RUN_DIR=$build_dir/ptoas-sim/run"
    "ASCEND_HOME_PATH=$cann_root"
    "SOC_VERSION=$ptoas_sim_soc"
    "SIM_LIB_DIR=$cann_root/tools/simulator/$ptoas_sim_soc/lib"
    "BUILD_JOBS=$(nproc)"
    "USE_MSPROF=1"
    "CORE_ID=$core_id"
    "MSPROF_OUTPUT=$build_dir/ptoas-profile"
    "CMAKE_BIN=$cmake_bin"
    "PYTHON_BIN=$python_bin"
    "PATH=$(dirname -- "$python_bin"):$(dirname -- "$cmake_bin"):${PATH:-}"
  )
  if [[ -n "$kernel_name" ]]; then
    env_cmd+=("KERNEL_NAME=$kernel_name")
  fi
  env_cmd+=(bash "$case_dir/run_sim.sh")

  write_command "$build_dir/bridge-sim.command.txt" "${env_cmd[@]}"
  log "VPTO simulator fixture"
  run_logged "$build_dir/bridge-sim.log" "${env_cmd[@]}"
  log "simulator log: $build_dir/bridge-sim.log"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ $# -ne 2 ]]; then
  usage
  exit 1
fi

option="${1#--}"
case_name="$2"
case_dir="$(case_dir_for "$case_name")"
out_dir="$case_dir/out"
build_dir="$out_dir/build"
mkdir -p "$build_dir"

case "$option" in
  early-ir|input-mlir|generate-mlir)
    run_python_simulator "$case_dir" "$build_dir" 1
    ;;
  print-all|after-all)
    run_print_all "$case_dir"
    ;;
  npu-sim|baseline-sim)
    run_python_simulator "$case_dir" "$build_dir" 0
    ;;
  emit-vmi|vmi)
    run_emit_vmi "$case_dir" "$case_name"
    ;;
  emit-vpto|vpto)
    run_emit_vpto "$case_dir" "$case_name"
    ;;
  bridge-sim|ptoas-sim)
    run_bridge_sim "$case_dir" "$case_name"
    ;;
  *)
    die "unknown option: $1"
    ;;
esac
