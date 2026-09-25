# Q2 Triton Kernel PTOAS Failure Categorization

## Scope

This report categorizes the results of running the Q2 Triton Kernel tests
through the AscendNPU-IR-to-PTOAS bridge flow. The bridge starts with
AscendNPU-IR, emits PTOAS VMI through `convert-hivmave-to-ptoas-vmi` and the
structured HIVM template conversion, then runs the remaining PTOAS VMI-to-VPTO
and object-file pipeline.

The input corpus is recorded in [results.csv](../../../Logs/results.csv) and
[results.json](../../../Logs/results.json). Detailed reduced failure logs are
under [`collected/`](../../../Logs/collected/).

The corpus contains 140 parameterized testcases:

| Status | Testcases |
|---|---:|
| Passed | 2 |
| Failed | 138 |
| Total | 140 |

## Passing Testcase Performance

Both passing testcases are layer-normalization forward cases. PTOAS is slower
than the baseline NPU-IR flow in both measurements.

| Testcase | NPU-IR | PTOAS | Difference | PTOAS overhead |
|---|---:|---:|---:|---:|
| `layer_norm_gated_fwd[2-4096-1200-silu]` | 442.245 us | 489.96 us | +47.715 us | +10.79% |
| `layer_norm_gated_fwd[2-2-64-silu]` | 4.3 us | 4.9 us | +0.6 us | +13.95% |

Every failed testcase is assigned exactly one top-level classification:

| Top-level classification | Unique failed testcases |
|---|---:|
| Template coverage | 95 |
| `HIVMAVEToPTOASVMI` coverage | 33 |
| Neither: failure occurs before the bridge | 10 |
| **Total** | **138** |

Counts in the detailed table are not additive. A single conversion can report
several missing structured templates, such as ND2NZ, MmadL1, and Fixpipe, in
the same testcase.

## Shape-Deduplicated Logical Testcases

The 140 result rows are parameterized shape instantiations of 17 pytest test
functions. The table below counts each test function once. A test is marked
**uniform** only when every one of its shapes has the same failure signature;
solving that signature is therefore expected to unblock all listed shapes.
Mixed tests retain separate shape groups because fixing one group does not
necessarily fix the other groups in that test function.

| Logical testcase | Parameterized results | Shape-level behavior | Same issue for all shapes? | Deduplicated testcase count |
|---|---:|---|---|---:|
| `test_causal_conv1d_bwd_with_initial_state_perf` | 5 failed | All 5: `memref.reinterpret_cast` legalization | Yes | 1 |
| `test_causal_conv1d_bwd_varlen_perf` | 5 failed | All 5: `memref.reinterpret_cast` legalization | Yes | 1 |
| `test_chunk_bwd_dqkwg_perf` | 10 failed | All 10: MmadL1 normalization/contract, BF16 ND2NZ, and BF16 Fixpipe | Yes | 1 |
| `test_chunk_bwd_dv_local_perf` | 10 failed | All 10: MmadL1 dimension/init normalization | Yes | 1 |
| `test_chunk_gated_delta_rule_bwd_dhu_perf` | 10 failed | All 10: MmadL1 normalization/contract and BF16 Fixpipe | Yes | 1 |
| `test_chunk_gated_delta_rule_fwd_h_perf` | 10 failed | All 10: MmadL1 normalization/contract and BF16 Fixpipe | Yes | 1 |
| `test_chunk_local_cumsum_perf` | 10 failed | 7 **[CS7]**: `vextf` mask; 3 **[CU3]**: unresolved Cumsum helper | **No: 2 groups** | 1 |
| `test_chunk_fwd_o_perf` | 10 failed | All 10: MmadL1 contract and BF16 Fixpipe | Yes | 1 |
| `test_chunk_scaled_dot_kkt_fwd_perf` | 10 failed | All 10: missing BF16/FP32 B-transposed MmadL1 specialization | Yes | 1 |
| `test_layer_norm_gated_bwd_perf` | 10 failed | 7 **[LN7]**: VADD attributes; 2 **[VI2]**: `vintlv`; 1 `[2-100-50-sigmoid]`: unresolved 2-D helpers | **No: 3 groups** | 1 |
| `test_layer_norm_gated_fwd_perf` | 8 failed, 2 passed | 7 **[LN7]**: VADD attributes; 1 `[2-100-50-sigmoid]`: unresolved 2-D helpers; 2 **[VI2]**: pass | **No: 2 failure groups and 1 passing group** | 1 |
| `test_prepare_wy_repr_bwd_perf` | 10 failed | All 10: BF16 `tl.exp` frontend rejection | Yes | 1 |
| `test_recompute_w_u_fwd_perf` | 10 failed | All 10: MmadL1 normalization and BF16 ND2NZ | Yes | 1 |
| `test_solve_tril_mxr_perf` | 8 failed | All 8: FP32 IEEE MmadL1 specialization and FP32 Fixpipe | Yes | 1 |
| `test_solve_tril_mxr_varlen_perf` | 2 failed | Both: FP32 IEEE MmadL1 specialization and FP32 Fixpipe | Yes | 1 |
| `test_solve_tril_perf` | 8 failed | All 8: FP32 IEEE MmadL1 specialization, FP32 ND2NZ, and FP32 Fixpipe | Yes | 1 |
| `test_solve_tril_varlen_perf` | 2 failed | Both: FP32 IEEE MmadL1 specialization, FP32 ND2NZ, and FP32 Fixpipe | Yes | 1 |
| **Total** | **140 results** | **14 uniform functions; 3 mixed functions; 21 homogeneous failure groups** |  | **17** |

The deduplicated count answers a different question from the 138-failure
count: there are 17 logical test functions to unblock, but 21 distinct
within-function failure groups because the three mixed functions follow
shape-dependent lowering paths. Counts in later error rows still represent
parameterized results so the impact of each fix remains visible.

## Detailed Categorization

All testcase paths in this table implicitly start with
`q2::tests/fla_perf/`. Parameter-set abbreviations are expanded in
[Exact parameter sets](#exact-parameter-sets).

| Error | Testcases | Testcase count | Classification | Template or converter work required |
|---|---|---:|---|---|
| `memref.reinterpret_cast` failed legalization. Initial-state cases additionally report that the cast must be rank-1 unit-stride or have static higher-rank strides while preserving element type. | `test_causal_conv1d_bwd_perf.py::test_causal_conv1d_bwd_with_initial_state_perf` **[CI5]**; `test_causal_conv1d_bwd_perf.py::test_causal_conv1d_bwd_varlen_perf` **[CV5]** | 10 | `HIVMAVEToPTOASVMI` coverage | N/A. Extend the bridge's `memref.reinterpret_cast` conversion contract. |
| `ave.hir.vadd`: extra AVE operation attributes are unsupported in Phase 1. | `test_layer_norm_gated_bwd_perf.py::test_layer_norm_gated_bwd_perf` **[LN7]**; `test_layer_norm_gated_fwd_perf.py::test_layer_norm_gated_fwd_perf` **[LN7]** | 14 | `HIVMAVEToPTOASVMI` coverage | N/A. Preserve or lower the observed AVE attributes when creating the PTO VMI add. |
| `ave.hir.vextf`: currently requires a direct `pge <ALL>` mask. | `test_chunk_local_cumsum_perf.py::test_chunk_local_cumsum_perf` **[CS7]** | 7 | `HIVMAVEToPTOASVMI` coverage | N/A. Add support for the observed non-direct or partial predicate producer. |
| Failed to legalize `ave.hir.vintlv`. | `test_layer_norm_gated_bwd_perf.py::test_layer_norm_gated_bwd_perf` **[VI2]** | 2 | `HIVMAVEToPTOASVMI` coverage | N/A. Add AVE `vintlv` to PTO VMI `vintlv` conversion, including its two-result and mask contract. |
| `hivm.hir.mmadL1`: requires an `i1` init condition and normalized M/K/N in `[1,64]`. | `test_chunk_bwd_dqkwg_perf` **[K10]**; `test_chunk_bwd_dv_local_perf` **[K10]**; `test_chunk_gated_delta_rule_bwd_dhu_perf` **[KV10]**; `test_chunk_gated_delta_rule_fwd_h_perf` **[KV10]**; `test_recompute_w_u_fwd_perf` **[R10]** | 50 | Template coverage | Generalize `__pto_mmadl1_bf16_f32_nn` and add/use `__pto_mmadl1_bf16_f32_tb` where B is transposed. The converter must normalize init to `i1` and split dimensions larger than 64 into supported microtiles. A helper alone cannot bypass the current pre-selection rejection. |
| `hivm.hir.mmadL1`: unsupported result, bias, unit flag, A-transpose, or I4 contract. | `test_chunk_bwd_dqkwg_perf` **[K10]**; `test_chunk_gated_delta_rule_bwd_dhu_perf` **[KV10]**; `test_chunk_gated_delta_rule_fwd_h_perf` **[KV10]**; `test_chunk_fwd_o_perf` **[O10]** | 40 | Template coverage | For the observed BF16 workloads, add A-transpose support, nominally `__pto_mmadl1_bf16_f32_ta`, plus accumulator/result-enabled NN/TA variants as required. The converter must also normalize result and unit-flag forms before helper selection. The reduced logs do not identify which member of the broad rejection applies to every operation. |
| `hivm.hir.mmadL1`: type, transpose, and precision have no pre-generated specialization. | `test_chunk_scaled_dot_kkt_fwd_perf` **[SD10]** | 10 | Template coverage | Add `__pto_mmadl1_bf16_f32_tb`: BF16 inputs, FP32 accumulation, transposed B. |
| `hivm.hir.mmadL1`: type, transpose, and precision have no pre-generated specialization. | Both `test_solve_tril_mxr_perf.py` test functions and both `test_solve_tril_perf.py` test functions **[Z10]** | 20 | Template coverage | Add `__pto_mmadl1_f32_f32_ieee_nn`: FP32 inputs and accumulation, IEEE precision, non-transposed operands. |
| `hivm.hir.nd2nz`: no compatible datatype, shape, layout, or padding contract. | `test_chunk_bwd_dqkwg_perf` **[K10]**; `test_recompute_w_u_fwd_perf` **[R10]** | 20 | Template coverage | Extend `__pto_nd2nz_bf16_gm_l1` for the rejected shape/layout/padding or result form. BF16 datatype coverage already exists. The full rejected operation is required to determine whether separate strided, padded, or non-GM variants are needed. |
| `hivm.hir.nd2nz`: no compatible datatype, shape, layout, or padding contract. | Both `test_solve_tril_perf.py` test functions **[Z10]** | 10 | Template coverage | Extend `__pto_nd2nz_f32_gm_l1`. If the rejected source is UB rather than GM, add `__pto_nd2nz_f32_ub_l1`. The reduced logs do not retain enough operation detail to select between these contracts conclusively. |
| `hivm.hir.fixpipe`: no compatible destination, datatype, or conversion contract. | `test_chunk_bwd_dqkwg_perf` **[K10]**; `test_chunk_gated_delta_rule_bwd_dhu_perf` **[KV10]**; `test_chunk_gated_delta_rule_fwd_h_perf` **[KV10]**; `test_chunk_fwd_o_perf` **[O10]** | 40 | Template coverage | Add `__pto_fixpipe_nz2nd_f32_bf16_row_split_ub`; add a `__pto_fixpipe_nz2nd_f32_bf16_single_ub` variant if the full IR confirms single-destination cases. No BF16 UB Fixpipe helper exists in the current catalog. |
| `hivm.hir.fixpipe`: no compatible destination, datatype, or conversion contract. | Both `test_solve_tril_mxr_perf.py` test functions and both `test_solve_tril_perf.py` test functions **[Z10]** | 20 | Template coverage | Generalize `__pto_fixpipe_nz2nd_f32_f32_row_split_ub` beyond its current exact `32x64` destination restriction, including the solve kernels' `16x16` subblocks. |
| Linker cannot resolve `_mlir_ciface_cumsum_2d_float_dim0`. | `test_chunk_local_cumsum_perf.py::test_chunk_local_cumsum_perf` **[CU3]** | 3 | Template coverage | Add a PTO-native composite helper, suggested symbol `__pto_cumsum_2d_f32_dim0`, using the supported vector load/add/store and predicate operations. |
| Linker cannot resolve `_mlir_ciface_load_gm_to_ubuf_2d_half` and `_mlir_ciface_copy_ubuf_to_ubuf_2d_float`. | BWD and FWD layer-norm-gated tests at `[2-100-50-sigmoid]` | 2 | Template coverage | Extend selection of the existing `__pto_load_gm_to_ubuf_2d_half` PTODSL helper beyond its current descriptor contract, and implement a composite 2-D FP32 UB-to-UB copy helper. Both paths are expressible with existing PTO operations. |
| Triton rejects BF16 input to `tl.exp`: expected `fp16`, `fp32`, or `fp64`. | `test_prepare_wy_repr_bwd_perf.py::test_prepare_wy_repr_bwd_perf` **[BF10]** | 10 | Neither: pre-bridge Triton frontend failure | N/A. Cast the input to a supported floating-point type before `tl.exp`, or add BF16 frontend semantics. This failure never reaches BishengIR or PTOAS. |

## Exact Parameter Sets

### Causal convolution

- **CI5:**
  - `B2-T65535-D128-silu`
  - `B1-T1024-D64-swish`
  - `B2-T2-D64-silu`
  - `B2-T1024-D128-swish`
  - `B2-T4096-D1200-silu`
- **CV5:**
  - `N2-T100-D50-swish`
  - `N16-T8192-D128-swish`
  - `N32-T8192-D128-silu`
  - `N16-T16384-D128-swish`
  - `N4-T16384-D128-silu`

### Layer normalization and vector conversion

- **LN7:**
  - `1-2097152-128-silu`
  - `1-2097152-128-sigmoid`
  - `2-1024-128-sigmoid`
  - `16-8192-128-sigmoid`
  - `32-8192-128-silu`
  - `16-16384-128-sigmoid`
  - `4-16384-128-silu`
- **VI2:**
  - `2-2-64-silu`
  - `2-4096-1200-silu`
- **CS7:**
  - `1-1024-32-False`
  - `4-16384-32-True`
  - `8-16384-32-True`
  - `1-65536-32-False`
  - `4-65536-32-False`
  - `8-65536-32-True`
  - `1-131072-32-True`
- **CU3:**
  - `1-8192-64-False`
  - `4-131072-64-False`
  - `8-131072-64-False`

### Cube and structured-template workloads

- **K10:**
  - `B1-T1024-H32-K128`
  - `B4-T1024-H32-K128`
  - `B16-T1024-H32-K128`
  - `B1-T8192-H32-K128`
  - `B4-T8192-H32-K128`
  - `B16-T8192-H8-K128`
  - `B1-T131072-H4-K128`
  - `B4-T131072-H2-K128`
  - `B16-T131072-H2-K128`
  - `B1-T16384-H4-K128`
- **KV10:** the corresponding **K10** dimensions with `-V128` appended.
- **R10:**
  - `B1-T1024-H32-HV32-K128-V128-BT64-torch.bfloat16`
  - `B4-T1024-H32-HV32-K128-V128-BT64-torch.bfloat16`
  - `B16-T1024-H32-HV32-K128-V128-BT64-torch.bfloat16`
  - `B1-T8192-H32-HV32-K128-V128-BT64-torch.bfloat16`
  - `B4-T8192-H32-HV32-K128-V128-BT64-torch.bfloat16`
  - `B16-T8192-H32-HV32-K128-V128-BT64-torch.bfloat16`
  - `B1-T131072-H32-HV32-K128-V128-BT64-torch.bfloat16`
  - `B4-T131072-H32-HV32-K128-V128-BT64-torch.bfloat16`
  - `B16-T131072-H8-HV8-K128-V128-BT64-torch.bfloat16`
  - `B1-T16384-H32-HV32-K128-V128-BT64-torch.bfloat16`
- **O10:**
  - `B1-T1024-H32-D128-V128`
  - `B4-T1024-H32-D128-V128`
  - `B16-T1024-H32-D128-V128`
  - `B1-T8192-H32-D128-V128`
  - `B4-T8192-H32-D128-V128`
  - `B16-T8192-H8-D128-V128`
  - `B1-T131072-H4-D128-V128`
  - `B1-T65536-H2-D128-V128`
  - `B4-T65536-H2-D128-V128`
  - `B1-T16384-H4-D128-V128`
- **SD10:**
  - `B1-T1024-H32-D128`
  - `B1-T8192-H64-D128`
  - `B4-T16384-H32-D128`
  - `B8-T16384-H32-D128`
  - `B1-T65536-H32-D128`
  - `B4-T65536-H8-D128`
  - `B8-T65536-H4-D128`
  - `B1-T131072-H2-D128`
  - `B4-T131072-H2-D128`
  - `B8-T16384-H64-D128`
- **Z10:**
  - Fixed-shape: `B1-T1024-H32-chunk_size64`
  - Fixed-shape: `B1-T1024-H64-chunk_size64`
  - Fixed-shape: `B8-T1024-H32-chunk_size64`
  - Fixed-shape: `B16-T1024-H64-chunk_size64`
  - Fixed-shape: `B1-T8192-H64-chunk_size64`
  - Fixed-shape: `B8-T8192-H32-chunk_size64`
  - Fixed-shape: `B16-T8192-H64-chunk_size64`
  - Fixed-shape: `B4-T131072-H32-chunk_size64`
  - Varlen: `H16-D128-chunk_size64`
  - Varlen: `H2-D128-chunk_size64`

### Pre-bridge BF16 exponential failure

- **BF10:**
  - `B1-T1024-H32-HV32-K128-V128-BT64-torch.bfloat16`
  - `B4-T1024-H32-HV32-K128-V128-BT64-torch.bfloat16`
  - `B16-T1024-H32-HV32-K128-V128-BT64-torch.bfloat16`
  - `B1-T8192-H32-HV32-K128-V128-BT64-torch.bfloat16`
  - `B4-T8192-H32-HV32-K128-V128-BT64-torch.bfloat16`
  - `B16-T8192-H8-HV8-K128-V128-BT64-torch.bfloat16`
  - `B1-T65536-H8-HV8-K128-V128-BT64-torch.bfloat16`
  - `B4-T65536-H8-HV8-K128-V128-BT64-torch.bfloat16`
  - `B16-T65536-H8-HV8-K128-V128-BT64-torch.bfloat16`
  - `B1-T16384-H32-HV32-K128-V128-BT64-torch.bfloat16`

## Template Findings and Confidence

The current PTODSL Cube catalog contains these relevant MmadL1
specializations:

- `__pto_mmadl1_f16_f32_nn`
- `__pto_mmadl1_f16_f32_tb`
- `__pto_mmadl1_bf16_f32_nn`
- `__pto_mmadl1_i8_i32_nn`
- `__pto_mmadl1_f32_f32_hf32_nn`

Therefore the two directly evidenced missing MmadL1 instantiations are:

```text
__pto_mmadl1_bf16_f32_tb
__pto_mmadl1_f32_f32_ieee_nn
```

The directly evidenced high-impact missing Fixpipe specialization is:

```text
__pto_fixpipe_nz2nd_f32_bf16_row_split_ub
```

The `requires i1 init and normalized M/K/N` and `unsupported result, bias,
unit flag, A transpose, or I4 contract` diagnostics occur before helper
selection. For those rows, implementing a helper without extending the C++
matcher and normalizing the rejected operation will not make the testcase
convert.

Similarly, BF16 and FP32 GM-to-L1 ND2NZ helpers already exist. The generic
ND2NZ diagnostic means the operation failed a shape, memory-space, rank,
continuity, result, initialization, stride, or padding requirement. The
reduced logs omit enough of the rejected operation that a new helper name
cannot be selected conclusively; a full IR capture should precede
implementation.

## CCE Template Rewrite Feasibility in PTODSL/PTO

This table separates three questions that the linker or template-selection
diagnostic conflates: whether a CCE helper exists, whether a PTODSL template
matches the exact operation, and whether the PTO dialect contains the
instructions needed to express the rewrite.

| Missing or rejected CCE/template path | PTODSL/PTO operations available | PTO rewrite support | Required bridge/template work |
|---|---|---|---|
| BF16 MmadL1 with transposed B: `__pto_mmadl1_bf16_f32_tb` | `pto.mte_l1_l0b` supports transpose; `pto.mad*` supports BF16 input and FP32 accumulation | **Supported** | Add the BF16 TB helper and its matcher specialization. |
| FP32 IEEE MmadL1: `__pto_mmadl1_f32_f32_ieee_nn` | `pto.mad*` supports FP32; omitting `tf32_mode` selects ordinary IEEE FP32 rather than HF32/TF32 | **Supported** | Add the FP32 IEEE NN helper and specialization. |
| Broad MmadL1 rejection: result, bias, unit flag, A-transpose, or I4 | `pto.mad`, `pto.mad_acc`, `pto.mad_bias`, L1-to-L0 A transpose, and unit-flag controls exist | **Supported except for integer I4.** The I4 attribute is not proven to occur in these logs; see the [detailed I4 support analysis](mmadl1_i4_pto_support_analysis.md). | Normalize result/init/dimensions and select accumulator, bias, transpose, and unit-flag forms before template matching. Integer I4 additionally requires new PTO type, load, and MAD coverage. The reduced log does not identify which condition rejected each operation. |
| BF16 ND2NZ GM-to-L1 | `pto.mte_gm_l1_frac` supports ND2NZ with BF16 | **Supported** for GM source | Generalize the existing `__pto_nd2nz_bf16_gm_l1` matcher for the rejected shape, layout, stride, padding, or result contract. |
| FP32 ND2NZ GM-to-L1 | `pto.mte_gm_l1_frac` supports ND2NZ with FP32 | **Supported** for GM source | Generalize the existing `__pto_nd2nz_f32_gm_l1` matcher. If full IR shows a UB source, see the direct-instruction gap below. |
| FP32-to-BF16 row-split Fixpipe to UB: `__pto_fixpipe_nz2nd_f32_bf16_row_split_ub` | `pto.mte_l0c_ub` supports the F32-to-BF16 conversion mode | **Supported** | Add the BF16 destination helper and matcher; add a single-destination variant only if full IR demonstrates it is required. |
| FP32-to-FP32 row-split Fixpipe for solve subblocks | Existing `pto.mte_l0c_ub` helper path supports FP32 output | **Supported** | Relax the current helper/matcher restriction from exactly `32x64` to the observed `16x16` subblocks and other valid dynamic rows/columns. |
| `_mlir_ciface_cumsum_2d_float_dim0` | `pto.vlds`, `pto.vadd`, `pto.vsts`, predicate/mask operations, and UB copy operations | **Supported as a composition** | Port the CCE Sklansky cumulative-add sequence to a PTODSL helper; no native Cumsum instruction is required. |
| `_mlir_ciface_load_gm_to_ubuf_2d_half` | Existing `__pto_load_gm_to_ubuf_2d_half` uses `pto.mte_gm_ub`, including looped/grouped transfer support | **Already implemented; current match is too narrow** | Extend the matcher/helper branches to accept the layer-normalization descriptor rather than falling back to the unresolved legacy CCE symbol. |
| `_mlir_ciface_copy_ubuf_to_ubuf_2d_float` | `pto.mte_ub_ub` supports grouped 2-D DMA; vector/scalar load-store operations cover unaligned row tails | **Supported as a composition** | Implement aligned DMA and unaligned-tail branches equivalent to the CCE `Copy2D` helper. |

### Instructions not supported or not established

No PTO instruction gap is confirmed for the templates directly identified by
these logs. Every evidenced BF16/FP32 MmadL1, Fixpipe, Cumsum, 2-D load, and
2-D copy path can be written using current PTO operations. The remaining two
caveats are conditional because the reduced logs do not retain the rejected
operation operands and attributes:

| Potential instruction gap | Status | Consequence |
|---|---|---|
| Packed signed-I4 MmadL1 selected by `enable_I4` | **Unsupported by the complete current PTO typed load/MAD path if present; not proven to occur in these failures.** CCE uses packed-S4 loads and `mad_s4` with I32 accumulation. PTO restricts its S4 bridge loads to packed FP4 and has no integer `s4 x s4 -> s32` typed MAD route. See the [detailed CCE/PTO analysis](mmadl1_i4_pto_support_analysis.md). | Add a signed packed-I4 type contract, integer-S4 Cube loads, S4-by-S4 MAD lowering, packing-aware dimension handling, and a matching PTODSL helper. Capture the rejected MmadL1 attributes first to determine whether this gap affects the current corpus. |
| Direct UB-to-L1 ND2NZ transfer | **No direct one-operation form found.** PTO has GM-to-L1 ND2NZ and ordinary UB-to-L1 DMA. | If full IR shows that the rejected ND2NZ source is UB, use a composed repack/transfer or add a PTO primitive. If the source is GM, the existing instruction is sufficient. |

Therefore the currently evidenced blockers are primarily PTODSL template and
C++ matcher coverage, plus bridge normalization. Integer-I4 MmadL1 has a PTO
type/instruction-lowering gap if present, while UB-to-L1 ND2NZ has no direct
one-operation form. Capturing the full rejected MmadL1 and ND2NZ operations is
required to determine whether either conditional gap affects this testcase
corpus.

## Classification Basis

The categorization follows the bridge's documented division of responsibility:

- AVE and memref operations rejected by `convert-hivmave-to-ptoas-vmi` are
  `HIVMAVEToPTOASVMI` coverage gaps.
- Structured `hivm.hir.nd2nz`, `hivm.hir.mmadL1`, and `hivm.hir.fixpipe`
  conversions, together with unresolved legacy `_mlir_ciface_*` helpers, are
  template coverage gaps.
- The BF16 `tl.exp` failure occurs in Triton's frontend before BishengIR or
  PTOAS and is therefore neither bridge category.

Relevant implementation and design references:

- [Bridge overview](../README.md)
- [FlashAttention bridge summary](../designs/flash-attention-bridge-summary.md)
- [Cube conversion status](../memory/cube-conversion-status.md)
- [Cube template conversion matcher](../../../AscendNPU-IR/bishengir/lib/Conversion/HIVMTemplatesToPTO/Cube/MatmulRegionToPTO.cpp)
- [PTODSL Cube template source and catalog](../../../AscendNPU-IR/bishengir/lib/Template/lib/RegBase/Cube/nd2nz_mmadl1_64_ptodsl.py)
- [MmadL1 integer-I4 PTO support analysis](mmadl1_i4_pto_support_analysis.md)
- [PTODSL 2-D GM-to-UB load](../../../AscendNPU-IR/bishengir/lib/Template/lib/RegBase/Vector/load_gm_to_ub_ptodsl.py)
- [CCE Cumsum implementation](../../../AscendNPU-IR/bishengir/lib/Template/lib/RegBase/Vector/Cumsum.cpp)
- [CCE 2-D copy implementation](../../../AscendNPU-IR/bishengir/lib/Template/lib/RegBase/Vector/Copy2D.cpp)
