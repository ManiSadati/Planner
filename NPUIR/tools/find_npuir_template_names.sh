#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: find_npuir_template_names.sh [NPU_IR_ROOT] [OUTPUT_DIR]

Extract the exported NPU-IR A5 template names from C310 LLVM bitcode.

Defaults:
  NPU_IR_ROOT  $HOME/AscendNPU-IR
  OUTPUT_DIR   $HOME/tmp/npuir-template-inventory

Environment:
  LLVM_NM      Path to an LLVM 15-compatible llvm-nm. The script also checks
               common build and CANN locations automatically.
  CCEC         Path to ccec. Used to emit textual LLVM IR when a compatible
               llvm-nm is unavailable.
EOF
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
  usage
  exit 0
fi

if (( $# > 2 )); then
  usage >&2
  exit 2
fi

npu_ir_root=${1:-"$HOME/AscendNPU-IR"}
output_dir=${2:-"$HOME/tmp/npuir-template-inventory"}
bundle_dir="$npu_ir_root/build/install/lib"
per_source_root="$npu_ir_root/build/tools/bishengir/bishengir/lib/Template/lib/RegBase"

bundles=(
  "$bundle_dir/meta_op.aic.c310.bc"
  "$bundle_dir/meta_op.aiv.c310.bc"
  "$bundle_dir/meta_op.mix.aic.c310.bc"
  "$bundle_dir/meta_op.mix.aiv.c310.bc"
)

for bundle in "${bundles[@]}"; do
  if [[ ! -f $bundle ]]; then
    echo "error: missing A5 template bundle: $bundle" >&2
    echo "build and install NPU-IR template products first" >&2
    exit 1
  fi
done

llvm_nm_candidates=()
if [[ -n ${LLVM_NM:-} ]]; then
  llvm_nm_candidates+=("$LLVM_NM")
fi

for command_name in llvm-nm-15 llvm-nm; do
  if command_path=$(command -v "$command_name" 2>/dev/null); then
    llvm_nm_candidates+=("$command_path")
  fi
done

llvm_nm_candidates+=(
  "$npu_ir_root/build/bin/llvm-nm"
  "$npu_ir_root/build-ptoas-llvm/bin/llvm-nm"
)

for root_var in "${CANN_ROOT:-}" "${ASCEND_HOME_PATH:-}" "${ASCEND_TOOLKIT_HOME:-}"; do
  [[ -n $root_var ]] || continue
  llvm_nm_candidates+=(
    "$root_var/tools/bisheng_compiler/bin/llvm-nm"
    "$root_var/compiler/bin/llvm-nm"
    "$root_var/bin/llvm-nm"
  )
done

llvm_nm=
for candidate in "${llvm_nm_candidates[@]}"; do
  [[ -x $candidate ]] || continue
  if "$candidate" --defined-only --extern-only --format=posix \
      "${bundles[0]}" >/dev/null 2>&1; then
    llvm_nm=$candidate
    break
  fi
done

ccec_candidates=()
if [[ -n ${CCEC:-} ]]; then
  ccec_candidates+=("$CCEC")
fi
if ccec_path=$(command -v ccec 2>/dev/null); then
  ccec_candidates+=("$ccec_path")
fi
for root_var in "${CANN_ROOT:-}" "${ASCEND_HOME_PATH:-}" "${ASCEND_TOOLKIT_HOME:-}"; do
  [[ -n $root_var ]] || continue
  ccec_candidates+=(
    "$root_var/tools/bisheng_compiler/bin/ccec"
    "$root_var/compiler/ccec_compiler/bin/ccec"
    "$root_var/bin/ccec"
  )
done

shopt -s nullglob
ccec_candidates+=(
  "$HOME"/Ascend/cann-*/tools/bisheng_compiler/bin/ccec
  /home/*/Ascend/cann-*/tools/bisheng_compiler/bin/ccec
)
shopt -u nullglob

ccec=
for candidate in "${ccec_candidates[@]}"; do
  if [[ -x $candidate ]]; then
    ccec=$candidate
    break
  fi
done

if [[ -n $llvm_nm ]]; then
  symbol_reader="llvm-nm: $llvm_nm"
elif [[ -n $ccec ]]; then
  symbol_reader="ccec: $ccec"
else
  cat >&2 <<EOF
error: no compatible symbol reader was found for:
  ${bundles[0]}

These bundles were produced by the Bisheng/CANN LLVM 15 toolchain. GNU nm and
the LLVM 19 tools in the NPU-IR checkout cannot reliably read them. Set LLVM_NM
to an LLVM 15-compatible llvm-nm or set CCEC to the Bisheng compiler driver:

  LLVM_NM=/path/to/llvm-15/bin/llvm-nm $0 "$npu_ir_root" "$output_dir"
  CCEC=/path/to/ccec $0 "$npu_ir_root" "$output_dir"
EOF
  exit 2
fi

list_interface_symbols() {
  local bitcode=$1

  if [[ -n $llvm_nm ]]; then
    "$llvm_nm" --defined-only --extern-only --format=posix "$bitcode" |
      awk '$1 ~ /^_mlir_ciface_/ { print $1 }'
    return
  fi

  "$ccec" --target=hiipu64-hisilicon-cce -x ir -S -emit-llvm \
    "$bitcode" -o - 2>/dev/null |
    awk '
      /^define / && match($0, /@_mlir_ciface_[A-Za-z0-9_.$-]+/) {
        print substr($0, RSTART + 1, RLENGTH - 1)
      }
    '
}

mkdir -p "$output_dir"

physical="$output_dir/all-physical-template-symbols.tsv"
unique="$output_dir/all-unique-template-symbols.txt"
normal_unique="$output_dir/normal-unique-template-symbols.txt"
by_source="$output_dir/by-source-template-symbols.tsv"
operation_unique="$output_dir/operation-template-symbols.txt"

printf 'bundle\ttemplate_symbol\tc_interface_symbol\n' >"$physical"
for bundle in "${bundles[@]}"; do
  bundle_name=$(basename "$bundle")
  list_interface_symbols "$bundle" |
    awk -v bundle="$bundle_name" '
      $1 ~ /^_mlir_ciface_/ {
        interface = $1
        canonical = interface
        sub(/^_mlir_ciface_/, "", canonical)
        print bundle "\t" canonical "\t" interface
      }
    ' >>"$physical"
done

tail -n +2 "$physical" | cut -f2 | sort -u >"$unique"
tail -n +2 "$physical" |
  awk -F '\t' '$1 == "meta_op.aic.c310.bc" || $1 == "meta_op.aiv.c310.bc" { print $2 }' |
  sort -u >"$normal_unique"

printf 'domain\tsource_bitcode\ttemplate_symbol\tc_interface_symbol\n' >"$by_source"
if [[ -d $per_source_root ]]; then
  while IFS= read -r bc; do
    relative=${bc#"$per_source_root"/}
    domain=${relative%%/*}
    case "$relative" in
      Vector/SIMT*) domain=SIMT ;;
      Vector/*) domain=Vector ;;
      Cube/*) domain=Cube ;;
      Debug/*) domain=Support ;;
    esac

    list_interface_symbols "$bc" |
      awk -v domain="$domain" -v source="$relative" '
        $1 ~ /^_mlir_ciface_/ {
          interface = $1
          canonical = interface
          sub(/^_mlir_ciface_/, "", canonical)
          print domain "\t" source "\t" canonical "\t" interface
        }
      ' >>"$by_source"
  done < <(
    find "$per_source_root" -type f \
      \( -name '*.aic.c310.bc' -o -name '*.aiv.c310.bc' \) \
      ! -name '*.mix.*' -print | sort
  )

  tail -n +2 "$by_source" |
    awk -F '\t' '$1 == "Cube" || $1 == "Vector" || $1 == "SIMT" { print $3 }' |
    sort -u >"$operation_unique"
else
  : >"$operation_unique"
  echo "warning: per-source bitcode directory not found: $per_source_root" >&2
fi

physical_count=$(( $(wc -l <"$physical") - 1 ))
unique_count=$(wc -l <"$unique")
normal_count=$(wc -l <"$normal_unique")
operation_count=$(wc -l <"$operation_unique")

cat <<EOF
Template inventory complete.

symbol reader:      $symbol_reader
physical symbols:  $physical_count
unique symbols:    $unique_count
normal bundles:    $normal_count
operation symbols: $operation_count

Outputs:
  $physical
  $unique
  $normal_unique
  $by_source
  $operation_unique
EOF
