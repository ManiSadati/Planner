# NPU-IR Device-Spec Replay Results

Last updated: 2026-08-11

## Summary

The Codex-server NPU-IR build can replay the current A5-generated
`bridge/triton-example/*_kernel.mlir` inputs far enough to capture IR after:

```text
convert-hivmave-to-ave-intrin
```

All six current examples reached that target pass:

```text
dma_copy_kernel
matmul64_kernel
rmsnorm_kernel
row_softmax_kernel
vector_dma_pipeline_kernel
vector_elementwise_kernel
```

The compiler still exits nonzero afterward on the non-A5 server because the
downstream A5 backend executable is unavailable:

```text
Cannot find hivmc-a5 under $PATH
```

This is not a blocker for the current bridge investigation because the target
IR dump is captured before that downstream failure.

## Generated Artifacts

Replay outputs live under:

```text
bridge/examples/npuir-early-ir/replay/
```

For each case, the useful files are:

```text
after-convert-hivmave-to-ave-intrin.mlir
compile.log
compile-exit-code.txt
after-convert-hivmave-to-ave-intrin-dump-count.txt
```

Each compile log printed the target pass six times. The replay wrapper now
extracts the last matching target-pass dump into
`after-convert-hivmave-to-ave-intrin.mlir` and records the count separately.

Clean target dump sizes after extraction:

```text
dma_copy_kernel:            129 lines
matmul64_kernel:            114 lines
rmsnorm_kernel:             283 lines
row_softmax_kernel:         398 lines
vector_dma_pipeline_kernel: 368 lines
vector_elementwise_kernel:  279 lines
```

The cleaned target dumps do not contain the downstream `[ERROR]` lines.

## Operation Evidence At This Boundary

`dma_copy_kernel` reaches a late state where GM/UB DMA has already become:

```text
func.call @load_gm_to_ubuf_1d_float
func.call @store_ubuf_to_gm_1d_float
```

The surrounding schedule still contains explicit HIVM pipe synchronization,
including `PIPE_MTE3`, `PIPE_MTE2`, `PIPE_V`, `set_flag`, `wait_flag`, and
`pipe_barrier`.

`matmul64_kernel` reaches a late cube/template state with:

```text
func.call @nd2nz_half
func.call @mma_tile_half_to_float
func.call @fixpipe_nz2nd_float_to_half_4d_to_2d_gm
```

The surrounding schedule includes CBUF/CC pointer casts and explicit pipe
sync across MTE, M, and FIX pipes.

The vector examples and vector parts of mixed examples contain outlined vector
functions and regbase vector intrinsics, including:

```text
hivm_regbaseintrins.intr.hivm.pge.b32
hivm_regbaseintrins.intr.hivm.vldsx1.v64f32
hivm_regbaseintrins.intr.hivm.vstsx1.v64f32
hivm_regbaseintrins.intr.hivm.vadd.s.x
hivm_regbaseintrins.intr.hivm.vmuls.s.x
hivm_regbaseintrins.intr.hivm.vsel
```

Reduction examples add operations such as:

```text
hivm_regbaseintrins.intr.hivm.vcadd.s.x
hivm_regbaseintrins.intr.hivm.vcmax.s.x
hivm_regbaseintrins.intr.hivm.vexpdif
hivm_regbaseintrins.intr.hivm.vdiv.s.x
hivm_regbaseintrins.intr.hivm.vsqrt.x
```

## Implication

`convert-hivmave-to-ave-intrin` is useful as a readable endpoint for current
compiler behavior, but it is too late as the primary bridge interception point
for structured DMA and cube conversion.

At this endpoint:

- DMA is visible mainly as template/helper calls;
- cube/matmul is visible mainly as `nd2nz`, `mma_tile`, and `fixpipe` helper
  calls;
- vector bodies are already lowered to regbase vector intrinsics;
- sync/control facts are still visible as HIVM HIR pipe operations.

Therefore the first real AscendNPU-IR implementation should add an exploratory
bridge-analysis/export pass earlier in the pipeline, while structured HIVM ops
are still available. The target should be before the lowering that turns
structured DMA/cube operations into template/helper calls.

The late endpoint remains useful for checking what NPU-IR currently emits and
for deriving expected DMA/cube/sync categories, but it should not be the only
source for bridge mapping.
