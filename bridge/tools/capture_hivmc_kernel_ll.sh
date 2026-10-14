#!/usr/bin/env bash
set -uo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  capture_hivmc_kernel_ll.sh --capture-dir <directory> -- <hivmc-a5 command...>

Runs the command with an isolated TMPDIR and copies transient kernel.ll files
to the capture directory before the command removes its temporary files.
EOF
}

capture_dir=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --capture-dir)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      capture_dir="$2"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

[[ -n "$capture_dir" ]] || { echo "error: --capture-dir is required" >&2; usage; exit 2; }
[[ $# -gt 0 ]] || { echo "error: a command is required after --" >&2; usage; exit 2; }

mkdir -p "$capture_dir"
capture_dir="$(cd -- "$capture_dir" && pwd)"
temp_root="$(mktemp -d "${TMPDIR:-/tmp}/capture-hivmc-kernel-ll.XXXXXX")"
stop_file="$temp_root/.stop-capture"
poll_interval="${KERNEL_LL_CAPTURE_POLL_INTERVAL:-0.01}"
command_args=("$@")
watch_dirs=("$temp_root")

for ((index = 0; index < ${#command_args[@]}; index++)); do
  argument="${command_args[$index]}"
  output_file=""
  case "$argument" in
    -o|--output)
      if ((index + 1 < ${#command_args[@]})); then
        output_file="${command_args[$((index + 1))]}"
      fi
      ;;
    -o=*|--output=*)
      output_file="${argument#*=}"
      ;;
  esac

  if [[ -n "$output_file" ]]; then
    output_dir="$(dirname -- "$output_file")"
    if [[ "$output_dir" != /* ]]; then
      output_dir="$PWD/$output_dir"
    fi
    watch_dirs+=("$output_dir")
  fi
done

capture_files() {
  local watch_dir watch_name source_file relative_name destination_file temporary_file

  for watch_dir in "${watch_dirs[@]}"; do
    [[ -d "$watch_dir" ]] || continue
    watch_name="$(basename -- "$watch_dir")"
    while IFS= read -r -d '' source_file; do
      relative_name="${source_file#"$watch_dir"/}"
      relative_name="${relative_name//\//.}"
      destination_file="$capture_dir/${watch_name}.${relative_name%.ll}.kernel.ll"
      temporary_file="$destination_file.tmp.$$"

      if cp -- "$source_file" "$temporary_file" 2>/dev/null; then
        mv -f -- "$temporary_file" "$destination_file"
      else
        rm -f -- "$temporary_file"
      fi
    done < <(find "$watch_dir" -type f -name kernel.ll -print0 2>/dev/null)
  done
}

cleanup() {
  touch "$stop_file" 2>/dev/null || true
  if [[ -n "${watcher_pid:-}" ]]; then
    wait "$watcher_pid" 2>/dev/null || true
  fi
  rm -rf -- "$temp_root"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

(
  while [[ ! -e "$stop_file" ]]; do
    capture_files
    sleep "$poll_interval"
  done
  capture_files
) &
watcher_pid=$!

command_status=0
TMPDIR="$temp_root" "$@" || command_status=$?

touch "$stop_file"
wait "$watcher_pid" 2>/dev/null || true
watcher_pid=""

echo "Captured kernel.ll files:"
find "$capture_dir" -maxdepth 1 -type f -name '*.kernel.ll' -print | sort

exit "$command_status"
