# Q2 native template instances: Triton-Ascend 3.6 census

Recorded: 2026-10-23. **42 exact native template instances** were observed in the completed Q2 compile-only campaign. This is the dependency list encountered by the retained test functions; the full Q2/CI requirement set remains unknown.

The sweep attempted **28,751 declared configurations across 33 test-function paths (30 files)**: **22,684** completed with usable dumps and **6,067** failed during native HIVM lowering. **59 of 92 catalogued function paths were excluded** before the sweep. There were **315 distinct usable native compilations**. A configuration can request multiple JITs, and usable captures remain counted if another JIT in that configuration fails.

This document sits beside the [earlier mixed Q2/NPU-IR-Test census](native-template-census-2026-10-19.md); it does not replace or merge that inventory. The new scope is Q2 only. Different selected functions, inputs, compiler versions and exclusions prevent attributing the changed symbol list to Triton-Ascend 3.6 alone.

Machine-readable companions: [symbols](q2-native-template-census-triton36-2026-10-23-symbols.csv), [category counts](q2-native-template-census-triton36-2026-10-23-categories.csv), and [provenance, exclusions and evidence](q2-native-template-census-triton36-2026-10-23-evidence.json).

## Unique instances per category

| Category | Unique instances |
|---|---:|
| MmadL1 | 4 |
| Fixpipe | 5 |
| Nd2Nz | 2 |
| Cube setup | 2 |
| Vector DMA | 19 |
| Vector Integer64 | 4 |
| Vector collective | 5 |
| SIMT indirect load | 1 |
| **Total** | **42** |

Categories group the exact native symbols; specialization suffixes, datatypes, ranks and layout names are preserved. `indirect_load_nonvolatile_1d_float_int64_t` is catalogued under SIMT indirect load; that classification alone does not establish that its caller ran in pure-SIMT compilation mode.

## Exact instances, sorted by category

Names below are native IR identities with only the `_mlir_ciface_` wrapper prefix removed. Each symbol has a matching C-interface definition in the frozen campaign C310 bitcode libraries. No symbol is missing from that library set. This confirms definition availability, not that each caller links the right library or runs correctly. Historical local C310 patches are included.

Count definitions:

- **Static sites:** call sites summed over distinct usable native compilations, deduplicated globally.
- **Across configs:** static sites summed once per distinct native compilation per manifest configuration; repeated identical JIT requests within one configuration count once. These are configuration-weighted static counts, not dynamic invocation frequencies.
- **Files / functions:** distinct Q2 relative file paths / file-qualified test-function paths containing the symbol. Per-symbol file/function counts cannot be added to obtain category unions.

Declared configuration tuples include the whole-test Cartesian product of source autotune groups. Inactive branch choices and identical specializations reuse compilation; configuration count therefore exceeds unique native compilation count.

### MmadL1 — 4 instances

| Exact native template instance | Static sites | Across configs | Files | Functions |
|---|---:|---:|---:|---:|
| `mma_tile_bfloat16_t_to_float` | 10 | 720 | 4 | 6 |
| `mma_tile_bfloat16_t_to_float_ta` | 24 | 720 | 2 | 3 |
| `mma_tile_bfloat16_t_to_float_tb` | 39 | 1,260 | 4 | 5 |
| `mma_tile_float_to_float` | 112 | 896 | 4 | 4 |

### Fixpipe — 5 instances

| Exact native template instance | Static sites | Across configs | Files | Functions |
|---|---:|---:|---:|---:|
| `fixpipe_normal_float_to_bfloat16_t_4d_to_4d_cbuf` | 6 | 180 | 2 | 3 |
| `fixpipe_normal_float_to_float_4d_to_4d_cbuf` | 48 | 384 | 4 | 4 |
| `fixpipe_nz2dn_dual_float_to_float_4d_to_2d_ubuf` | 6 | 180 | 2 | 3 |
| `fixpipe_nz2nd_dual_float_to_float_4d_to_2d_ubuf` | 103 | 2,036 | 8 | 9 |
| `fixpipe_nz2nd_float_to_bfloat16_t_4d_to_2d_gm` | 4 | 540 | 2 | 3 |

### Nd2Nz — 2 instances

| Exact native template instance | Static sites | Across configs | Files | Functions |
|---|---:|---:|---:|---:|
| `nd2nz_bfloat16_t` | 41 | 1,530 | 6 | 8 |
| `nd2nz_float` | 24 | 192 | 4 | 4 |

### Cube setup — 2 instances

| Exact native template instance | Static sites | Across configs | Files | Functions |
|---|---:|---:|---:|---:|
| `set_l1_2d_bfloat16_t` | 41 | 1,530 | 6 | 8 |
| `set_l1_2d_float` | 24 | 192 | 4 | 4 |

### Vector DMA — 19 instances

| Exact native template instance | Static sites | Across configs | Files | Functions |
|---|---:|---:|---:|---:|
| `copy_ubuf_to_cbuf_1d_float` | 24 | 192 | 2 | 2 |
| `copy_ubuf_to_cbuf_2d_bfloat16_t` | 40 | 1,620 | 4 | 6 |
| `copy_ubuf_to_cbuf_2d_float` | 80 | 640 | 4 | 4 |
| `copy_ubuf_to_ubuf_1d_float` | 239 | 137,817 | 15 | 16 |
| `copy_ubuf_to_ubuf_2d_float` | 22 | 5,384 | 6 | 6 |
| `load_gm_to_ubuf_1d_bfloat16_t` | 105 | 2,305 | 13 | 16 |
| `load_gm_to_ubuf_1d_float` | 243 | 35,094 | 16 | 16 |
| `load_gm_to_ubuf_1d_half` | 129 | 193,372 | 10 | 10 |
| `load_gm_to_ubuf_1d_int32_t` | 2 | 2 | 1 | 1 |
| `load_gm_to_ubuf_2d_bfloat16_t` | 187 | 1,874 | 11 | 14 |
| `load_gm_to_ubuf_2d_float` | 135 | 570 | 11 | 12 |
| `load_gm_to_ubuf_2d_half` | 110 | 13,353 | 8 | 8 |
| `store_ubuf_to_gm_1d_bfloat16_t` | 26 | 539 | 6 | 8 |
| `store_ubuf_to_gm_1d_float` | 303 | 94,189 | 20 | 21 |
| `store_ubuf_to_gm_1d_half` | 35 | 58,550 | 6 | 6 |
| `store_ubuf_to_gm_2d_bfloat16_t` | 149 | 1,499 | 11 | 13 |
| `store_ubuf_to_gm_2d_float` | 130 | 508 | 9 | 9 |
| `store_ubuf_to_gm_2d_half` | 82 | 8,023 | 8 | 8 |
| `store_ubuf_to_gm_3d_float` | 1 | 2 | 2 | 2 |

### Vector Integer64 — 4 instances

| Exact native template instance | Static sites | Across configs | Files | Functions |
|---|---:|---:|---:|---:|
| `cast_int32_t_to_int64_t` | 24 | 27 | 1 | 1 |
| `masked_store_NORM_B64_int64_t_rank1` | 24 | 27 | 1 | 1 |
| `vadd_int64_t` | 24 | 27 | 1 | 1 |
| `vbr_int64_t` | 24 | 27 | 1 | 1 |

### Vector collective — 5 instances

| Exact native template instance | Static sites | Across configs | Files | Functions |
|---|---:|---:|---:|---:|
| `cumsum_1d_float_dim0` | 2 | 2 | 1 | 1 |
| `cumsum_2d_float_dim0` | 4 | 369 | 3 | 3 |
| `cumsum_2d_float_dim0_comp` | 3 | 273 | 3 | 3 |
| `cumsum_2d_float_dim1` | 6 | 16 | 3 | 3 |
| `cumsum_3d_float_dim1` | 1 | 2 | 2 | 2 |

### SIMT indirect load — 1 instance

| Exact native template instance | Static sites | Across configs | Files | Functions |
|---|---:|---:|---:|---:|
| `indirect_load_nonvolatile_1d_float_int64_t` | 24 | 27 | 1 | 1 |

## Capture, versions and limits

The harness used source-derived, shape-only inputs and fixed existing autotune choices, intercepted JIT launches, and compiled through native NPU-IR. It retained structured IR immediately before `convert-hivm-to-std` and parse-checked dumps after `convert-hivm-to-std` and `convert-hivmave-to-std`. Only usable, complete checkpoints contribute symbols. A private guard stopped the external binary backend. **No kernels, simulator runs, numerical checks, final native linking or PTODSL implementation validation were performed.**

| Component | Recorded campaign value |
|---|---|
| Target / flow | `Ascend950PR_9589` / native `npuir`; bridge disabled |
| Triton-Ascend / Triton | `3.6.0+gite9f9b169` / `3.6.0` |
| Published candidate integration baseline | `461aa8f086c6bf26229ee324b8f4b5d3902e002d` |
| Q2 source | `ebf78b205a87ecc5cd935a214bd2ef0fc176dcd9`; clean at snapshot |
| NPU-IR compiler self-report | `75525e826a7b`; LLVM 19.1.7 |
| NPU-IR source checkout at freeze | `58f887f7688ba26e17ea02548558113abb702c19`; dirty runtime-test edits recorded in evidence |
| Compiler SHA-256 | `765cf8da946a15272dceb4a37f789394c783590814102f9bd49331845668fd8c` |

The compiler was built before source publication, so its self-reported stamp and source checkout revision are recorded separately. The sweep used verified private copies of that compiler/resources and the candidate packages, with `$HOME/.venv/npuir-sim-3.6-candidate/bin/python`. Pinned PTOAS and its LLVM revisions are provenance only; this native campaign did not execute PTOAS. See the [shared 3.6 baseline](../memory/triton-ascend-3.6-shared-a5-2026-10-22.md) and the evidence JSON for all four revisions and installed-library hashes.

All 6,067 in-sweep failures were recorded at `native_hivm_lowering`: 6,048 configurations in gated layer-norm backward (accuracy and performance copies), 17 GELU configurations, and 2 Swiglu configurations. Sampled diagnostics report UB overflow; the failing-stage aggregate is not a complete root-cause classification. Failures may still contain a different usable JIT capture.

Preflight exclusions include 11 functions with the native Fixpipe bytecode-import failure (5 previously known and 6 found in the fresh compile preflight). Other exclusions cover missing inputs/imports, host preparation/runtime dependencies, negative tests and non-kernel tests. Exact function paths and reasons are in the evidence JSON. Missing data is **unknown coverage**, not an unsupported-feature finding.

The launch settings were preserved as declared; unit flags were not forcibly enabled. This inventory does not classify all unit-flag, dynamic descriptor or scratch-reuse combinations within a symbol and does not update any PTODSL adapter/test support status.

Campaign artifacts on the originating server:

```text
$HOME/template-census/triton36/q2-full/run-20261022/
  summary.json
  manifest.jsonl
  scheduled-records.json
  all-exclusions.json
  observed-template-counts.csv
  category-counts.json
  completed.jsonl
  native-captures/
  compile-preflight/
  bitcode-symbol-audit/
```

Next coverage work is to resolve excluded paths and inspect failed configurations independently; absence from this observed list does not remove an earlier native dependency. Bridge compilation and numerical checks remain separate follow-ups.
