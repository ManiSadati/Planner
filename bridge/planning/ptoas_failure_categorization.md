# Q2 Triton Kernel PTOAS Failure Categorization

## Scope

This report categorizes the results of running the Q2 Triton Kernel tests
through the AscendNPU-IR-to-PTOAS bridge flow. The bridge starts with
AscendNPU-IR, emits PTOAS VMI through `convert-hivmave-to-ptoas-vmi` and the
structured HIVM template conversion, then runs the remaining PTOAS VMI-to-VPTO
and object-file pipeline.

The input corpus is recorded in [results.csv](results.csv) and
[results.json](results.json). Detailed reduced failure logs are under
[`collected/`](collected/).

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
| Linker cannot resolve `_mlir_ciface_cumsum_2d_float_dim0`. | `test_chunk_local_cumsum_perf.py::test_chunk_local_cumsum_perf` **[CU3]** | 3 | Template coverage | Implement/import a PTO-native `cumsum_2d_float_dim0` equivalent, suggested symbol `__pto_cumsum_2d_f32_dim0`. |
| Linker cannot resolve `_mlir_ciface_load_gm_to_ubuf_2d_half` and `_mlir_ciface_copy_ubuf_to_ubuf_2d_float`. | BWD and FWD layer-norm-gated tests at `[2-100-50-sigmoid]` | 2 | Template coverage | Implement/import a 2-D FP16 GM-to-UB load, suggested symbol `__pto_load_gm_to_ub_2d_f16`, and a 2-D FP32 UB-to-UB copy, suggested symbol `__pto_copy_ub_to_ub_2d_f32`. Both helpers are needed by each testcase. |
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

- [Bridge overview](../Planner/bridge/README.md)
- [FlashAttention bridge summary](../Planner/bridge/designs/flash-attention-bridge-summary.md)
- [Cube conversion status](../Planner/bridge/memory/cube-conversion-status.md)
- [Cube template conversion matcher](../AscendNPU-IR/bishengir/lib/Conversion/HIVMTemplatesToPTO/Cube/MatmulRegionToPTO.cpp)
- [PTODSL Cube template source and catalog](../AscendNPU-IR/bishengir/lib/Template/lib/RegBase/Cube/nd2nz_mmadl1_64_ptodsl.py)
