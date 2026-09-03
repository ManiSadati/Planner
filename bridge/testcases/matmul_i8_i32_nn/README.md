# INT8 to INT32 NN Matmul

This testcase isolates a `64x64x64` row-major INT8 matmul with INT32
accumulation and output. It is intended to require an INT8/INT32 NN `mmadL1`
PTODSL instantiation and matching INT8 DMA helpers.

```bash
cd "$HOME/Planner"
bridge/tools/run_comparison_flow.sh early-ir matmul_i8_i32_nn
bridge/tools/run_comparison_flow.sh emit-vmi matmul_i8_i32_nn
bridge/tools/run_comparison_flow.sh emit-vpto matmul_i8_i32_nn
bridge/tools/run_comparison_flow.sh --clean-build bridge-sim matmul_i8_i32_nn
bridge/tools/run_comparison_flow.sh fatobj matmul_i8_i32_nn
```

The bridge runner uses PTODSL mode by default. Generated artifacts are under
`out/`; `input.mlir` is the captured early NPU-IR input.

## Current Result

Verified on 2026-09-03:

- `early-ir` succeeds and the NPU-IR path selects `nd2nz_int8_t` and
  `mma_tile_int8_t_to_int32_t`.
- The signed INT8 ND2NZ and INT8/INT32 NN MMAD helpers emit VMI and VPTO
  successfully.
- `bridge-sim` passes with an exact INT32 output match.
- `fatobj` publishes `matmul_i8_i32_nn_kernel_ptoas_fatobj.o` in this directory.

On an A5 server, source its CANN environment and run:

```bash
bash run_npu_from_fatobj.sh
```
