#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  apply.sh <check|apply|revert> <installed|a5-files> <root>

Layouts:
  installed  root is the Python site-packages directory containing triton/
  a5-files   root contains compile.py and utils.py copied from the A5 venv

Examples:
  apply.sh check installed "$VIRTUAL_ENV/lib/python3.10/site-packages"
  apply.sh apply installed "$VIRTUAL_ENV/lib/python3.10/site-packages"
  apply.sh apply a5-files "$HOME/ta-3.2.2-dev"
EOF
}

if [[ $# -ne 3 ]]; then
  usage
  exit 2
fi

action=$1
layout=$2
root=$3
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

case "$layout" in
  installed)
    patch_file="$script_dir/installed-site-packages.patch"
    required_files=(
      "triton/backends/ascend/compiler.py"
      "triton/backends/ascend/utils.py"
      "triton/tools/get_ascend_devices.py"
    )
    marker_file="$root/triton/backends/ascend/compiler.py"
    ;;
  a5-files)
    patch_file="$script_dir/a5-file-set.patch"
    required_files=("compile.py" "utils.py")
    marker_file="$root/compile.py"
    ;;
  *)
    echo "error: unknown layout: $layout" >&2
    usage
    exit 2
    ;;
esac

for relative_path in "${required_files[@]}"; do
  if [[ ! -f "$root/$relative_path" ]]; then
    echo "error: expected file not found: $root/$relative_path" >&2
    exit 1
  fi
done

can_apply() {
  patch --batch --forward --dry-run -p1 -d "$root" < "$patch_file" >/dev/null 2>&1
}

has_patch_marker() {
  grep -q 'TRITON_ASCEND_COMPILE_FLOW' "$marker_file"
}

can_revert() {
  has_patch_marker \
    && patch --batch --reverse --dry-run -p1 -d "$root" < "$patch_file" >/dev/null 2>&1
}

case "$action" in
  check)
    if can_apply; then
      echo "compatible: patch can be applied"
    elif can_revert; then
      echo "applied: this exact patch is already present"
    else
      echo "incompatible: files are neither the expected baseline nor this patched version" >&2
      exit 1
    fi
    ;;
  apply)
    if has_patch_marker; then
      echo "unchanged: patch is already applied"
      exit 0
    fi
    if ! can_apply; then
      echo "error: patch does not apply cleanly; files were not changed" >&2
      exit 1
    fi
    patch --batch --forward -p1 -d "$root" < "$patch_file"
    echo "applied: $patch_file"
    ;;
  revert)
    if can_apply; then
      echo "unchanged: patch is not applied"
      exit 0
    fi
    if ! can_revert; then
      echo "error: patch cannot be cleanly reverted; files were not changed" >&2
      exit 1
    fi
    patch --batch --reverse -p1 -d "$root" < "$patch_file"
    echo "reverted: $patch_file"
    ;;
  *)
    echo "error: unknown action: $action" >&2
    usage
    exit 2
    ;;
esac
