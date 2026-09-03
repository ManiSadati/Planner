# F32 HF32 NN Matmul

This testcase isolates a `64x64x64` row-major F32 matmul using Triton's
`input_precision="hf32"` contract, with F32 accumulation and output. It is
intended to require the HF32-enabled F32/F32 NN `mmadL1` PTODSL variant.

```bash
cd "$HOME/Planner"
bridge/tools/run_comparison_flow.sh early-ir matmul_f32_hf32_nn
bridge/tools/run_comparison_flow.sh emit-vmi matmul_f32_hf32_nn
bridge/tools/run_comparison_flow.sh emit-vpto matmul_f32_hf32_nn
bridge/tools/run_comparison_flow.sh --clean-build bridge-sim matmul_f32_hf32_nn
bridge/tools/run_comparison_flow.sh fatobj matmul_f32_hf32_nn
```

The bridge runner uses PTODSL mode by default. Generated artifacts are under
`out/`; `input.mlir` is the captured early NPU-IR input.

## Current Result

Verified on 2026-09-03:

- The original `early-ir` output contained `input_precison = "hf32"` as
  emitted by the current Triton frontend. The tracked `input.mlir` was manually
  corrected to `input_precision` for this focused bridge experiment; rerunning
  `early-ir` will overwrite it with the misspelled form.
- With that correction, the pass trace contains `mmadL1 {enable_HF32}` and
  selects `mma_tile_float_to_float_hf32`, proving the NPU-IR handoff works.
- The typed F32 ND2NZ and F32/HF32 NN MMAD helpers emit VMI and VPTO
  successfully.
- `bridge-sim` passes with zero maximum absolute difference.
- `fatobj` publishes `matmul_f32_hf32_nn_kernel_ptoas_fatobj.o` in this
  directory.

On an A5 server, source its CANN environment and run:

```bash
bash run_npu_from_fatobj.sh
```
