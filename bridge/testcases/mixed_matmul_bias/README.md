# Mixed Matmul Bias

This is a minimal mixed Cube/Vector bridge testcase. One logical program computes
a fixed `64x64x64` FP16 matrix multiplication, applies the elementwise epilogue
`matmul * bias + bias`, and stores FP16 output. The dot product is intended for
Cube execution and the bias epilogue for Vector execution. The multiplication
prevents the bias from being folded into the matmul accumulator.

Generate the captured NPU-IR input and run the complete bridge simulator flow:

```bash
cd /home/m00967009/Workspace/Planner
bridge/tools/run_comparison_flow.sh early-ir mixed_matmul_bias
bridge/tools/run_comparison_flow.sh \
  --clean-build --bridge-mode ptodsl bridge-sim mixed_matmul_bias
```

## Current Bridge Status

The captured `input.mlir` lowers to one AIC function and one outlined AIV
function. NPU-IR transfers the FP32 Cube accumulator to the two Vector
subblocks with an FP32-to-FP16, row-split `hivm.hir.fixpipe`, followed by
cross-core synchronization.

The bridge lowers that fixpipe to `pto.mte_l0c_ub` with `dst_mode(split_m)`,
`nz2nd`, logical `m = 64`, logical `n = 64`, and f32 L0C to f16 UB pointer
types. PTOAS selects the FP32-to-FP16 backend intrinsic from the destination
type in split mode, so the VMI form intentionally carries no explicit
`pre_quant` operand.

The testcase emits VMI and VPTO through the NPU-IR to PTOAS bridge path. The
simulator fixture is present for end-to-end mixed Cube/Vector validation.
