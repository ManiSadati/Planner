#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
planner_root=$(cd -- "$script_dir/../.." && pwd)

inventory_dir=${1:-"$HOME/tmp/npuir-template-inventory"}
output_dir=${2:-"$planner_root/bridge/inventory"}
source_inventory="$inventory_dir/by-source-template-symbols.tsv"
physical_inventory="$inventory_dir/all-physical-template-symbols.tsv"
assignments="$output_dir/npuir-template-pack-assignments.tsv"
report="$output_dir/npuir-template-pack-classification.md"

mkdir -p "$output_dir"

for input in "$source_inventory" "$physical_inventory"; do
  if [[ ! -f $input ]]; then
    echo "error: missing inventory input: $input" >&2
    echo "run $planner_root/NPUIR/tools/find_npuir_template_names.sh first" >&2
    exit 1
  fi
done

tmp_assignments=$(mktemp "$inventory_dir/.assignments.XXXXXX")
tmp_sorted=$(mktemp "$inventory_dir/.assignments-sorted.XXXXXX")
trap 'rm -f "$tmp_assignments" "$tmp_sorted"' EXIT

awk -F '\t' '
  function append_unique(current, value, key) {
    key = "," current ","
    if (index(key, "," value ",") != 0)
      return current
    return current == "" ? value : current "," value
  }

  function rank_group(name) {
    if (name ~ /_[12]d_/)
      return "low"
    return "high"
  }

  function classify(name, source, domain) {
    if (domain ~ /Support/ || source == "") {
      if (name ~ /^assert_/)
        return "Support/assertions.mlir"
      if (name ~ /^print_/)
        return "Support/print-lifecycle.mlir"
      if (name ~ /^(free_lock_var|sync_block_)/)
        return "Support/runtime-sync.mlir"
      if (name ~ /\.cube$/ || source ~ /Debug\.aic/)
        return "Support/cube-debug.mlir"
      if (name ~ /\.vector$/ || source ~ /Debug\.aiv/)
        return "Support/vector-debug.mlir"
      return "Support/print-lifecycle.mlir"
    }

    if (source ~ /Cube\/LocalMmadMx/) {
      if (name ~ /with_.*bias/)
        return "Cube/MmadL1/bias.mlir"
      return "Cube/MmadL1/fp8-mx.mlir"
    }
    if (source ~ /Cube\/LocalMmad/) {
      if (name ~ /with_.*bias/)
        return "Cube/MmadL1/bias.mlir"
      if (name ~ /float8|fp8|fp4/)
        return "Cube/MmadL1/fp8-mx.mlir"
      if (name ~ /int8/)
        return "Cube/MmadL1/integer.mlir"
      if (name ~ /float_to_float/)
        return "Cube/MmadL1/fp32-ieee-hf32.mlir"
      return "Cube/MmadL1/classic-float.mlir"
    }
    if (source ~ /Cube\/GlobalMmad/) {
      if (name ~ /int8|int32/)
        return "Cube/GlobalMmad/integer.mlir"
      if (name ~ /float8|fp8|fp4/)
        return "Cube/GlobalMmad/fp8-mx.mlir"
      return "Cube/GlobalMmad/classic-float.mlir"
    }
    if (source ~ /Cube\/nd2nz/) {
      if (name ~ /forbias/)
        return "Cube/Nd2Nz/bias.mlir"
      if (name ~ /float8|fp8|fp4|hifloat/)
        return "Cube/Nd2Nz/fp8-mx.mlir"
      if (name ~ /half|float|bfloat/)
        return "Cube/Nd2Nz/standard-float.mlir"
      return "Cube/Nd2Nz/integer.mlir"
    }
    if (source ~ /Cube\/Fixpipe/) {
      if (name ~ /_dual_/)
        return "Cube/Fixpipe/dual-output.mlir"
      if (name ~ /_nz2nd_/)
        return "Cube/Fixpipe/nz2nd.mlir"
      if (name ~ /_nz2dn_/)
        return "Cube/Fixpipe/nz2dn.mlir"
      return "Cube/Fixpipe/normal.mlir"
    }
    if (source ~ /Cube\/Copy1D/)
      return "Cube/Copy/copy1d.mlir"
    if (source ~ /Cube\/l12ub/)
      return "Cube/Copy/l1-to-ub.mlir"
    if (source ~ /Cube\/loadMXScale/)
      return "Cube/Setup/mx-scale.mlir"
    if (source ~ /Cube\/set2d/)
      return "Cube/Setup/set2d-initialize.mlir"

    if (source ~ /Vector\/Copy[123]D/) {
      if (name ~ /^load_gm_to_ubuf_/)
        return "Vector/DMA/gm-to-ub.mlir"
      if (name ~ /^store_ubuf_to_gm_/)
        return "Vector/DMA/ub-to-gm.mlir"
      if (name ~ /^copy_ubuf_to_ubuf_/)
        return "Vector/DMA/ub-to-ub.mlir"
      if (name ~ /^copy_ubuf_to_cbuf_/)
        return "Vector/DMA/ub-to-l1.mlir"
      return "Vector/DMA/unaligned-layout.mlir"
    }
    if (source ~ /Vector\/VecOpI64/) {
      if (name ~ /^cast_/)
        return "Vector/Integer64/conversion.mlir"
      if (name ~ /^vc(add|max|min)_/)
        return "Vector/Integer64/reduction.mlir"
      if (name ~ /^vcmp|^vcmps|^vsel/)
        return "Vector/Integer64/compare-select.mlir"
      if (name ~ /^vload|^masked_store|^vgather|^binary_dma|^vbr|^vdup|^vci|^vslide|^vintlv|^vdintlv/)
        return "Vector/Integer64/dma-addressing.mlir"
      return "Vector/Integer64/arithmetic.mlir"
    }
    if (source ~ /Vector\/(Atan|Cos|Sin|Tan)\./)
      return "Vector/Math/trigonometric.mlir"
    if (source ~ /Vector\/(Erf|Relu|Tanh)\./)
      return "Vector/Math/nonlinear.mlir"
    if (source ~ /Vector\/(Log1p|Log2|Pow)\./)
      return "Vector/Math/logarithm-power.mlir"
    if (source ~ /Vector\/(Recip|Rsqrt|Round|Ldexp|Ilogb)\./)
      return "Vector/Math/reciprocal-rounding.mlir"
    if (source ~ /Vector\/(DivMagic|DivRn|Vdiv|Vmodui|Umulhi|FloatAsInt|NumChecks)\./)
      return "Vector/Math/integer-special.mlir"
    if (source ~ /Vector\/(Add1D|BroadcastScalar)\./)
      return "Vector/Math/basic-arithmetic.mlir"
    if (source ~ /Vector\/(Cumsum|Cumprod|CumsumSimtSklansky)\./)
      return "Vector/Collective/prefix-sum-product.mlir"
    if (source ~ /Vector\/Cumminmax\./)
      return "Vector/Collective/prefix-minmax.mlir"
    if (source ~ /Vector\/ReductionWithIndex\./)
      return "Vector/Collective/reduction-with-index.mlir"
    if (source ~ /Vector\/Sort[12]D\./)
      return "Vector/Collective/sorting.mlir"
    if (source ~ /Vector\/Flip1D\./)
      return "Vector/Collective/rearrangement.mlir"

    if (source ~ /Vector\/SIMTLoadStore/)
      return "SIMT/Direct/load-store.mlir"
    if (source ~ /Vector\/SIMTStride(Load|Store)/) {
      if (name ~ /_1d_/)
        return "SIMT/Direct/strided-rank1.mlir"
      if (name ~ /_2d_/)
        return "SIMT/Direct/strided-rank2.mlir"
      return "SIMT/Direct/strided-rank3.mlir"
    }
    if (source ~ /Vector\/SIMTIndirectLoad/) {
      if (name ~ /_[12]d_/)
        return "SIMT/IndirectLoad/rank1-rank2.mlir"
      if (name ~ /_3d_/)
        return "SIMT/IndirectLoad/rank3.mlir"
      return "SIMT/IndirectLoad/rank4-rank5.mlir"
    }
    if (source ~ /Vector\/SIMTIndirectStore/) {
      if (name ~ /no_mask/)
        return rank_group(name) == "low" ? "SIMT/IndirectStore/unmasked-low-rank.mlir" : "SIMT/IndirectStore/unmasked-high-rank.mlir"
      return rank_group(name) == "low" ? "SIMT/IndirectStore/masked-low-rank.mlir" : "SIMT/IndirectStore/masked-high-rank.mlir"
    }
    if (source ~ /Vector\/SIMTIndexSelect/)
      return name ~ /index_select_[23]d_/ ? "SIMT/Indexing/select-low-rank.mlir" : "SIMT/Indexing/select-high-rank.mlir"
    if (source ~ /Vector\/SIMTIndexPut/)
      return "SIMT/Indexing/index-put.mlir"
    if (source ~ /Vector\/SIMTGather/)
      return "SIMT/Indexing/gather.mlir"
    if (source ~ /Vector\/SIMTScatter/)
      return "SIMT/Indexing/scatter.mlir"
    if (source ~ /Vector\/SIMTIndirectAtomicBlock/)
      return "SIMT/Atomic/block.mlir"
    if (source ~ /Vector\/SIMTIndirectAtomicSoftAccel/)
      return "SIMT/Atomic/software.mlir"
    if (source ~ /Vector\/SIMTIndirectAtomic/) {
      if (name ~ /_cas_|_xchg_/)
        return "SIMT/Atomic/compare-swap.mlir"
      if (name ~ /_min_|_max_/)
        return "SIMT/Atomic/minmax.mlir"
      return "SIMT/Atomic/arithmetic.mlir"
    }
    if (source ~ /Vector\/SIMTHistogram/)
      return "SIMT/Collective/histogram.mlir"

    return "Unclassified/review-required.mlir"
  }

  NR == FNR {
    if (FNR == 1)
      next
    name = $3
    sources[name] = append_unique(sources[name], $2)
    domains[name] = append_unique(domains[name], $1)
    next
  }

  FNR == 1 { next }
  {
    name = $2
    physical_count[name]++
    bundles[name] = append_unique(bundles[name], $1)
  }

  END {
    for (name in physical_count) {
      pack = classify(name, sources[name], domains[name])
      if (pack ~ /^Cube\//)
        order = 1
      else if (pack ~ /^Vector\//)
        order = 2
      else if (pack ~ /^SIMT\//)
        order = 3
      else if (pack ~ /^Support\//)
        order = 4
      else
        order = 9
      source = sources[name] == "" ? "bundle-only" : sources[name]
      domain = domains[name] == "" ? "Support" : domains[name]
      print order "\t" pack "\t" name "\t" physical_count[name] "\t" domain "\t" source "\t" bundles[name]
    }
  }
' "$source_inventory" "$physical_inventory" >"$tmp_assignments"

sort -t $'\t' -k1,1n -k2,2 -k3,3 "$tmp_assignments" >"$tmp_sorted"

printf 'order\tproposed_mlir_pack\ttemplate_symbol\tphysical_occurrences\tdomain\tsource_bitcode\tbundles\n' >"$assignments"
cut -f1-7 "$tmp_sorted" >>"$assignments"

unclassified_count=$(awk -F '\t' '$2 ~ /^Unclassified\// { count++ } END { print count + 0 }' "$tmp_sorted")
unique_count=$(wc -l <"$tmp_sorted")
physical_count=$(awk -F '\t' '{ total += $4 } END { print total + 0 }' "$tmp_sorted")
pack_count=$(cut -f2 "$tmp_sorted" | sort -u | wc -l)

{
  cat <<EOF
# Proposed PTODSL Pack Classification

Generated from the exported C310 bitcode symbols by:

\`NPUIR/tools/find_npuir_template_names.sh\`
\`bridge/tools/classify_npuir_template_names.sh\`

This is a proposed packaging assignment, not a statement that the PTODSL
implementations already exist. The catalog strips the \`_mlir_ciface_\` prefix,
lists each canonical symbol once, and records duplicate normal/MIX definitions
as physical occurrences rather than separate PTODSL implementations.

## Coverage

- Physical bitcode definitions: $physical_count
- Unique canonical symbols: $unique_count
- Proposed MLIR packs used: $pack_count
- Unclassified symbols: $unclassified_count

The companion machine-readable mapping is
\`npuir-template-pack-assignments.tsv\`. It includes source bitcode and bundle
provenance for every symbol.

## Pack Summary

| Proposed MLIR pack | Unique symbols | Physical definitions |
|---|---:|---:|
EOF

  awk -F '\t' '
    { unique[$2]++; physical[$2] += $4 }
    END {
      for (pack in unique)
        print pack "\t" unique[pack] "\t" physical[pack]
    }
  ' "$tmp_sorted" |
    sort |
    awk -F '\t' '{ print "| `" $1 "` | " $2 " | " $3 " |" }'

  awk -F '\t' '
    $2 != current {
      current = $2
      print "\n## `" current "`\n"
    }
    {
      suffix = $4 > 1 ? " (" $4 " physical definitions)" : ""
      print "- `" $3 "`" suffix
    }
  ' "$tmp_sorted"
} >"$report"

if (( unclassified_count != 0 )); then
  echo "warning: $unclassified_count symbols require classification review" >&2
fi

cat <<EOF
Classification complete.

physical definitions: $physical_count
unique symbols:       $unique_count
MLIR packs:           $pack_count
unclassified:         $unclassified_count

Outputs:
  $report
  $assignments
EOF
