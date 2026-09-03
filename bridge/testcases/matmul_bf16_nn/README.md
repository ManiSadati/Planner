# BF16 NN Matmul

This testcase isolates a `64x64x64` row-major BF16 matmul with F32
accumulation and BF16 output. It is intended to require a BF16/F32 NN
`mmadL1` PTODSL instantiation and matching BF16 DMA helpers.

```bash
cd "$HOME/Planner"
bridge/tools/run_comparison_flow.sh early-ir matmul_bf16_nn
bridge/tools/run_comparison_flow.sh emit-vmi matmul_bf16_nn
bridge/tools/run_comparison_flow.sh emit-vpto matmul_bf16_nn
bridge/tools/run_comparison_flow.sh --clean-build bridge-sim matmul_bf16_nn
bridge/tools/run_comparison_flow.sh fatobj matmul_bf16_nn
```

The bridge runner uses PTODSL mode by default. Generated artifacts are under
`out/`; `input.mlir` is the captured early NPU-IR input.

## Current Result

Verified on 2026-09-03:

- `early-ir` succeeds and the NPU-IR path selects `nd2nz_bfloat16_t`,
  `mma_tile_bfloat16_t_to_float`, and BF16 Fixpipe writeback.
- The typed ND2NZ and BF16/F32 NN MMAD helpers emit VMI and VPTO successfully.
- `bridge-sim` passes with an exact BF16 output match.
- `fatobj` publishes `matmul_bf16_nn_kernel_ptoas_fatobj.o` in this directory.

On an A5 server, source its CANN environment and run:

```bash
bash run_npu_from_fatobj.sh
```
