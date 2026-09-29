# Missing or Incomplete Templates for `q2_triton_kernels` Coverage

Based on [`ptoas_failure_categorization.md`](ptoas_failure_categorization.md).

## Confirmed Missing Templates

| Template | Required coverage |
|---|---|
| `__pto_mmadl1_bf16_f32_tb` | BF16 inputs, FP32 accumulation, transposed B |
| `__pto_mmadl1_f32_f32_ieee_nn` | FP32 inputs and accumulation, IEEE precision, non-transposed operands |
| `__pto_fixpipe_nz2nd_f32_bf16_row_split_ub` | FP32 accumulator to BF16 UB output using row-split Fixpipe |
| `__pto_cumsum_2d_f32_dim0` | PTO-native composite implementation of 2-D FP32 cumulative sum along dimension 0 |
| Unnamed 2-D FP32 UB-to-UB copy helper | Replacement for `_mlir_ciface_copy_ubuf_to_ubuf_2d_float` |

## Existing Templates with Missing Configurations

| Existing template | Missing configuration or extension |
|---|---|
| `__pto_mmadl1_bf16_f32_nn` | Generalize init/result handling and M/K/N dimensions; support splitting dimensions larger than 64 into valid microtiles |
| `__pto_nd2nz_bf16_gm_l1` | Support the rejected shape, layout, stride, padding, result, or initialization forms |
| `__pto_nd2nz_f32_gm_l1` | Support the rejected FP32 shape, layout, stride, padding, result, or initialization forms |
| `__pto_fixpipe_nz2nd_f32_f32_row_split_ub` | Relax the exact `32x64` destination restriction to support `16x16` solve-kernel subblocks and other valid dimensions |
| `__pto_load_gm_to_ubuf_2d_half` | Extend matcher/helper branches to accept the layer-normalization descriptor currently falling back to the unresolved CCE helper |

## Conditional Templates Requiring Full-IR Confirmation

| Template | Condition |
|---|---|
| `__pto_mmadl1_bf16_f32_ta` | Required if the rejected BF16 MmadL1 operations use transposed A |
| Accumulator/result-enabled BF16 MmadL1 NN/TA variants | Required according to the exact result and accumulator forms in the rejected operations |
| `__pto_fixpipe_nz2nd_f32_bf16_single_ub` | Required if full IR confirms single-destination Fixpipe cases |
| `__pto_nd2nz_f32_ub_l1` | Required if the rejected FP32 ND2NZ source is UB rather than GM |
| Packed signed-I4 MmadL1 template | Required only if `enable_I4` is present; the reduced logs do not prove that it is used |

## Required Matcher and Converter Work

Adding templates alone will not unblock every testcase. The bridge and C++
matcher must also:

- Normalize MmadL1 init values to `i1`.
- Split M/K/N dimensions larger than 64 into supported microtiles.
- Normalize result and accumulator forms before template selection.
- Handle bias, unit flags, and A/B transpose configurations.
- Add matcher specializations for the new MmadL1 and Fixpipe templates.
- Broaden ND2NZ matching for valid shapes, layouts, strides, padding,
  initialization, and result forms.
- Broaden the 2-D GM-to-UB load matcher for the layer-normalization descriptor.

## High-Confidence Priority Order

1. `__pto_mmadl1_bf16_f32_tb`
2. `__pto_mmadl1_f32_f32_ieee_nn`
3. `__pto_fixpipe_nz2nd_f32_bf16_row_split_ub`
4. Generalize `__pto_mmadl1_bf16_f32_nn`
5. Generalize `__pto_fixpipe_nz2nd_f32_f32_row_split_ub`
6. Generalize `__pto_nd2nz_bf16_gm_l1`
7. Generalize `__pto_nd2nz_f32_gm_l1`
8. Add `__pto_cumsum_2d_f32_dim0`
9. Add the 2-D FP32 UB-to-UB copy helper
10. Extend `__pto_load_gm_to_ubuf_2d_half`
