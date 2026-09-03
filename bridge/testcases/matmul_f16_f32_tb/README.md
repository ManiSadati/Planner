# F16/F32 B-Transpose Matmul

This testcase isolates a `64x64x64` F16 matmul with F32 accumulation and F16
output. B is physically stored and loaded as `[N,K]`; an explicit `tl.trans`
feeds its transpose to `tl.dot`. Asymmetric data makes a lost transpose
observable. It is intended to require the B-transpose F16/F32 `mmadL1` PTODSL
variant.

```bash
cd "$HOME/Planner"
bridge/tools/run_comparison_flow.sh early-ir matmul_f16_f32_tb
bridge/tools/run_comparison_flow.sh emit-vmi matmul_f16_f32_tb
bridge/tools/run_comparison_flow.sh emit-vpto matmul_f16_f32_tb
bridge/tools/run_comparison_flow.sh --clean-build bridge-sim matmul_f16_f32_tb
bridge/tools/run_comparison_flow.sh fatobj matmul_f16_f32_tb
```

The bridge runner uses PTODSL mode by default. Generated artifacts are under
`out/`; `input.mlir` is the captured early NPU-IR input.

## Current Result

Verified on 2026-09-03:

- `early-ir` succeeds; the pass trace confirms
  `hivm.hir.mmadL1 {b_transpose}` with M/K/N `64/64/64`.
- The dedicated F16/F32 B-transpose helper emits VMI and VPTO without retaining
  the CCE `mma_tile_half_to_float_tb` call.
- `bridge-sim` passes with zero maximum absolute difference.
- `fatobj` publishes `matmul_f16_f32_tb_kernel_ptoas_fatobj.o` in this
  directory.

On an A5 server, source its CANN environment and run:

```bash
bash run_npu_from_fatobj.sh
```
