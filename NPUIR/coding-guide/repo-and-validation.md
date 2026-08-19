# AscendNPU-IR Repo And Validation Notes

Last updated: 2026-08-19

Scope: initial local planning note for bridge development. Do not treat this as
a full build recipe yet.

## Repo State

Local development repo:

```text
$HOME/AscendNPU-IR
```

Historical local branch at the first scan:

```text
mani/fuse-explore
4254b5dec90a4d3d92f581f3fe32b79ea1a82d9a Add more info about autvectorizev2
```

Historical local remote at the first scan:

```text
origin git@gitcode.com:manisadati/AscendNPU-IR.git
```

Current implementation rule: use the Wilson fork as the main bridge
implementation source and check its current active branch before coding.
Before compatibility decisions, compare with upstream:

```text
https://gitcode.com/wilsoncxfeng/AscendNPU-IR
https://gitcode.com/Ascend/AscendNPU-IR
```

A GitHub mirror may be useful as a secondary read-only remote if one exists, but
GitCode upstream should remain the primary upstream source.

## Where To Code

Expected bridge work should start on the AscendNPU-IR side unless the human asks
for PTOAS changes.

Likely source areas:

- `bishengir/include/bishengir/Conversion/`
- `bishengir/lib/Conversion/`
- `bishengir/include/bishengir/Dialect/HIVM/IR/`
- `bishengir/include/bishengir/Dialect/HIVMAVE/IR/`
- `bishengir/lib/Tools/bishengir-compile/regbase/PassPipeline.cpp`
- `bishengir/include/bishengir/Conversion/Passes.td`

Likely existing conversion examples:

- `bishengir/lib/Conversion/VectorToHIVMAVE/VectorToHIVMAVE.cpp`
- `bishengir/lib/Conversion/ArithToHIVMAVE/ArithToHIVMAVE.cpp`
- `bishengir/lib/Conversion/HIVMToStandard/regbase/HIVMToStandard.cpp`
- `bishengir/lib/Conversion/HIVMAVEToAVEIntrin/HIVMAVEToAVEIntrin.cpp`

Template and synchronization areas that matter for mapping:

- `bishengir/lib/Template/lib/Cube/LocalMmad.cpp`
- `bishengir/lib/Template/include/Cube/LocalMmad/LocalMmadUtils.h`
- `bishengir/lib/Template/lib/DMA/Cbuf/nd2nz.cpp`
- `bishengir/lib/Template/include/DMA/ND2NZ.h`
- `bishengir/include/bishengir/Dialect/HIVM/IR/HIVMSynchronizationOps.td`
- `bishengir/lib/Dialect/HIVM/Transforms/HIVMDecomposeOp.cpp`
- `bishengir/lib/Dialect/HIVM/Transforms/InjectSync/`
- `bishengir/lib/Dialect/HIVM/Transforms/GraphSyncSolver/`
- `bishengir/lib/Dialect/HIVM/Transforms/regbase/PlanMemory.cpp`

## Test And Validation Layers

Local checks that may be possible on this server:

- Build-only or compile-only checks if dependencies/submodules are ready.
- `bishengir-opt` pass tests for new conversion patterns.
- `llvm-lit` tests under `bishengir/test`.
- Focused MLIR FileCheck-style tests for before/after IR.

Documented full test target:

```bash
cmake --build . --target "check-mlir;check-bishengir"
```

Documented direct lit command from the build directory:

```bash
./bin/llvm-lit ../bishengir/test
```

Relevant existing test areas:

- `bishengir/test/Conversion/HFusionToHIVM/`
- `bishengir/test/Conversion/HFusionToVector/`
- `bishengir/test/Conversion/HIVMToStandard/`
- `bishengir/test/Conversion/HIVMToStandard/RegBase/`
- `bishengir/test/Dialect/HIVM/`
- `bishengir/test/Dialect/HIVMAVE/`
- `bishengir/test/regbase/`

Bridge-relevant sync tests to inspect before changing sync behavior:

- `bishengir/test/Dialect/HIVM/IR/sync-ops.mlir`
- `bishengir/test/Dialect/HIVM/inject-sync.mlir`
- `bishengir/test/Dialect/HIVM/sync-solver*.mlir`
- `bishengir/test/Dialect/HIVM/graph-sync-solver*.mlir`

A5/server validation that likely needs the human loop:

- End-to-end `bishengir-compile` output intended for `hivmc-a5`.
- Template-library/E2E cases that require CANN and the BiSheng compiler path.
- Runtime correctness and performance on actual A5 hardware.

When proposing validation, separate these three buckets:

```text
local static / lit checks
compile-only checks without A5 runtime
A5 hardware checks requiring human feedback
```

## First Bridge Development Rules

- Create the source-backed mapping table before implementation.
- Do not assume `convert-hivmave-to-ave-intrin` is sufficient for all ops.
- Treat `HIVMToStandard` as a likely earlier boundary for DMA, cube, and sync.
- Do not implement cube or DMA mapping by translating only final library-call
  names unless the needed semantic operands have already been captured.
- Preserve vector length, mask, access pattern, memory scope, sync ownership,
  and accumulator lifetime facts for PTOAS/VMI planning.
- Track sync generation separately from sync lowering. NPU-IR may already have
  generated `set_flag`, `wait_flag`, `sync_block_set`, and `sync_block_wait`
  before the bridge sees the IR.
- Keep PTOAS current reality separate from PTOAS direction-of-travel: current
  upstream TileLib is not VMI-based by default, but future VMI-based TileOp
  expansion is plausible and important.
- Any new pass should come with a small MLIR test that shows the intended IR
  before and after the pass.
