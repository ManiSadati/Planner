# AscendNPU-IR Lowering Pipeline

Last updated: 2026-08-10

Scope: initial Stage 3 local source scan of `/home/m84446336/AscendNPU-IR`
on branch `mani/fuse-explore` at
`4254b5dec90a4d3d92f581f3fe32b79ea1a82d9a`. This pass did not fetch
`https://gitcode.com/Ascend/AscendNPU-IR`, and the local repo currently has
only the `manisadati` fork as `origin`.

## Short Version

AscendNPU-IR has at least two bridge-relevant lowering layers:

```text
high-level input / Torch / Triton / linalg
  -> HFusion scheduling, fusion, tiling, and vectorization preparation
  -> MLIR vector / arith
  -> HIVMAVE vector/predicate operations
  -> HIVM and HIVMAVE lowering toward Standard/library calls
  -> HIVMAVE lowering to AVE/regbase intrinsic dialect
  -> downstream hivmc-a5 / CCE-oriented output
```

The PTOAS bridge probably cannot live at one single point for every operation.
Vector arithmetic and predicates may fit around or before
`convert-hivmave-to-ave-intrin`, but DMA, cube, and sync operations may need to
be intercepted before `convert-hivm-to-std`, because that pass rewrites many
structured `hivm.hir.*` operations into external library calls.

## Current Local Pipeline Shape

The regbase compile driver first builds the HIVMAVE optimization pipeline when
HIVM compilation is enabled, then optionally lowers to LLVM-oriented IR.

Key source:

- `/home/m84446336/AscendNPU-IR/bishengir/lib/Tools/bishengir-compile/regbase/PassPipeline.cpp`

In the local source, the late lowering order is:

```text
annotation lowering
hivm alloc-to-alloca
debug init/finish insertion
mark-disable-load
convert-hivm-to-std
convert-hivmave-to-std
expand-strided-metadata
convert-hivmave-to-ave-intrin
hoist-vstas
convert-ascend-dpx-to-hivmregbaseintrins
SCF/affine/arith lowering cleanup
```

The key lines are around `PassPipeline.cpp:268-312`.

## HIVM Pipeline Placement

Before late conversion to Standard/library calls, the HIVM pipeline already does
substantial semantic work: memory-scope inference, op decomposition, data-layout
inference, buffer sizing, flatten/reduce-rank cleanup, extra-buffer allocation,
multi-buffer marking, memory planning, lower-to-loops, sync pipeline insertion,
memref-ext lowering, and FFTS metadata attachment.

Key source:

- `/home/m84446336/AscendNPU-IR/bishengir/lib/Dialect/HIVM/Pipelines/HIVMPipelines.cpp:360`
- `/home/m84446336/AscendNPU-IR/bishengir/lib/Dialect/HIVM/Pipelines/HIVMPipelines.cpp:437`
- `/home/m84446336/AscendNPU-IR/bishengir/lib/Dialect/HIVM/Pipelines/HIVMPipelines.cpp:455`

Bridge implication: "before `convert-hivm-to-std`" is still too broad for the
final design. Mapping rows should distinguish whether they need pre-memory-plan,
post-memory-plan, post-sync, or late-template-lowering semantics.

## HFusion And Vectorization Front End

For regbase HFusion, the source pipeline runs a vectorization lane before final
HIVM/HIVMAVE lowering:

```text
canonicalization
fold extract/insert
sink op to consumer in loop
manual-scope vectorization pipeline
optional vf-fusion
fuse transpose into load
hfusion-pre-vectorization-fusion
prepare i1 nx1 for vectorization
hfusion-auto-vectorize-v2 or legacy hfusion-auto-vectorize
outline vector function
tree-reduce-v2
convert-hfusion-to-vector
vector mask lowering
optional inliner for vf-fusion
pull slice into vector function
simplify VF args
loop/subset/canonical cleanup
```

Key source:

- `/home/m84446336/AscendNPU-IR/bishengir/lib/Dialect/HFusion/Pipelines/regbase/HFusionRegbasePipelines.cpp:397`
- `/home/m84446336/AscendNPU-IR/bishengir/lib/Dialect/HFusion/Pipelines/regbase/HFusionRegbasePipelines.cpp:574`
- `/home/m84446336/AscendNPU-IR/bishengir/include/bishengir/Conversion/Passes.td:215`

Bridge implication: there may be a high-value option to branch after
`convert-hfusion-to-vector`, before vector/arith are converted to HIVMAVE, if
PTOAS VMI wants MLIR-vector-like semantics. That is only a hypothesis until the
mapping table compares examples.

## HIVMAVE Vector Layer

`lower-ave-pipeline` converts vector and arith operations into HIVMAVE for
regbase targets, then applies HIVMAVE-specific optimization/legalization:

```text
convert-vector-to-hivmave
convert-arith-to-hivmave
i1 soft implementation
complex reduction intermediate lowering
process vsstb
optimize reduction loop
AVE loop optimize
legalize HIVMAVE
replace vector-scalar
process membar
combine AVE ops
scalar broadcast to vload
PLT/PGE rewrites
vector layout analysis
alignment bitwidth analysis
AVE normalize
remove vector layout attr
```

Key source:

- `/home/m84446336/AscendNPU-IR/bishengir/lib/Dialect/HIVMAVE/Pipelines/HIVMAVEPipelines.cpp:37`
- `/home/m84446336/AscendNPU-IR/bishengir/include/bishengir/Conversion/Passes.td:355`
- `/home/m84446336/AscendNPU-IR/bishengir/include/bishengir/Conversion/Passes.td:384`

Important source facts:

- `convert-vector-to-hivmave` rewrites `vector.load`, `vector.store`,
  masked load/store, transfer read/write, gather/scatter, masks, broadcasts,
  shape casts, and reductions into `hivmave` operations.
- `vector.store` creates a full-lane `ave.hir.pge` mask before creating
  `ave.hir.masked_store`.
- `convert-arith-to-hivmave` maps arith/math operations such as add, sub, mul,
  div, min/max, comparisons, casts, select, shifts, sqrt/rsqrt/exp/log/abs, and
  fma into HIVMAVE vector ops.

Key sources:

- `/home/m84446336/AscendNPU-IR/bishengir/lib/Conversion/VectorToHIVMAVE/VectorToHIVMAVE.cpp:75`
- `/home/m84446336/AscendNPU-IR/bishengir/lib/Conversion/VectorToHIVMAVE/VectorToHIVMAVE.cpp:960`
- `/home/m84446336/AscendNPU-IR/bishengir/lib/Conversion/ArithToHIVMAVE/ArithToHIVMAVE.cpp:1976`

Bridge implication: this layer is the closest local analogue to PTOAS VMI for
vector and mask semantics. The future-compatible path should try to preserve
logical vector length, mask, access pattern, and accumulator lifetime facts so
PTOAS VMI layout assignment can make physical register decisions later.

## HIVM DMA, Cube, Sync, And Structured Ops

HIVM is the broader hardware-aware dialect. It models data movement, local/global
memory scopes, cube/vector core type, synchronization, and macro operations.

Initial key op locations:

- `hivm.hir.load`: `/home/m84446336/AscendNPU-IR/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td:62`
- `hivm.hir.store`: `/home/m84446336/AscendNPU-IR/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td:156`
- `hivm.hir.nd2nz`: `/home/m84446336/AscendNPU-IR/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td:356`
- `hivm.hir.pointer_cast`: `/home/m84446336/AscendNPU-IR/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMOps.td:224`
- `hivm.hir.set_flag`: `/home/m84446336/AscendNPU-IR/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMSynchronizationOps.td:43`
- `hivm.hir.sync_block`: `/home/m84446336/AscendNPU-IR/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMSynchronizationOps.td:87`
- `hivm.hir.sync_block_set`: `/home/m84446336/AscendNPU-IR/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMSynchronizationOps.td:129`
- `hivm.hir.mmadL1`: `/home/m84446336/AscendNPU-IR/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMMacroOps.td:173`

The regbase `HIVMToStandard` conversion rewrites many of these operations into
external library calls. Its pattern list includes `MmadL1Op`, `ND2NZOp`,
`LoadOp`, `StoreOp`, many `hivm` vector ops, sync-lock helpers, SIMT-style
indirect/stride/gather/scatter ops, and custom ops.

Key source:

- `/home/m84446336/AscendNPU-IR/bishengir/lib/Conversion/HIVMToStandard/HIVMToStandard.cpp:1936`
- `/home/m84446336/AscendNPU-IR/bishengir/lib/Conversion/HIVMToStandard/regbase/HIVMToStandard.cpp:1902`
- `/home/m84446336/AscendNPU-IR/bishengir/lib/Conversion/HIVMToStandard/regbase/HIVMToStandard.cpp:1978`

Concrete source-backed examples:

- `hivm.hir.mmadL1` is a structured cube macro op, not a scalar vector op. It
  carries A/B/C matrix operands, init condition, real `m/k/n`, sync-related
  arguments, unit-flag mode, transpose attributes, and optional bias. Its
  library-call name is generated from the `mma_tile` op family, element types,
  transpose flags, HF32/I4 flags, and bias mode.
- The `mma_tile` template loads L1 data into L0A/L0B, manages double buffering,
  sync flags, unit flags, optional bias movement, and emits CCE MAD intrinsics.
- `hivm.hir.nd2nz` is a GM-to-CBUF ND-to-NZ data movement template. The template
  maps a GM `(n,d)` view into an L1 `(n1,d1,d0,n0)` layout where
  `n0 * sizeof(T) = 32B` and `d0 = 16`, with different copy intrinsic variants
  by element byte width.
- `hivm.hir.sync_block` is decomposed before late lowering into
  `sync_block_set`, `sync_block_wait`, and/or `pipe_barrier` forms. Graph sync
  and inject-sync passes also generate `set_flag` and `wait_flag` directly.

Key sources:

- `/home/m84446336/AscendNPU-IR/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMMacroOps.td:58`
- `/home/m84446336/AscendNPU-IR/bishengir/lib/Dialect/HIVM/IR/LibraryFunctionOpInterface/LibraryFunctionOpInterfaceImpl.cpp:1104`
- `/home/m84446336/AscendNPU-IR/bishengir/lib/Template/lib/Cube/LocalMmad.cpp:314`
- `/home/m84446336/AscendNPU-IR/bishengir/lib/Template/include/Cube/LocalMmad/LocalMmadUtils.h:95`
- `/home/m84446336/AscendNPU-IR/bishengir/lib/Dialect/HIVM/IR/LibraryFunctionOpInterface/LibraryFunctionOpInterfaceImpl.cpp:1204`
- `/home/m84446336/AscendNPU-IR/bishengir/lib/Template/include/DMA/ND2NZ.h:23`
- `/home/m84446336/AscendNPU-IR/bishengir/lib/Template/lib/DMA/Cbuf/nd2nz.cpp:19`
- `/home/m84446336/AscendNPU-IR/bishengir/lib/Dialect/HIVM/Transforms/HIVMDecomposeOp.cpp:400`
- `/home/m84446336/AscendNPU-IR/bishengir/lib/Dialect/HIVM/Transforms/InjectSync/InjectSync.cpp:75`
- `/home/m84446336/AscendNPU-IR/bishengir/lib/Dialect/HIVM/Transforms/GraphSyncSolver/SyncSolverCodeGen.cpp:197`
- `/home/m84446336/AscendNPU-IR/bishengir/lib/Dialect/HIVM/Transforms/regbase/PlanMemory.cpp:171`

Bridge implication: for DMA/cube/sync, the bridge may need to run before
`convert-hivm-to-std` or add a sibling lowering that consumes HIVM directly.
Waiting until after library-call lowering risks losing shape, scope, pipe, and
sync-event structure that PTOAS would need.

## `convert-hivmave-to-ave-intrin`

This pass lowers HIVMAVE to the regbase intrinsic dialect and LLVM-compatible
pieces. Its pattern population includes vector load/store, gather/scatter,
PGE/PLT masks, select, reductions, compare, broadcast, type conversion, shift,
interleave/deinterleave, membar, vpack/vunpack, predicate casts, plus binary,
unary, ternary, vector-scalar, and broadcast registries.

Key source:

- `/home/m84446336/AscendNPU-IR/bishengir/lib/Conversion/HIVMAVEToAVEIntrin/HIVMAVEToAVEIntrin.cpp:3430`
- `/home/m84446336/AscendNPU-IR/bishengir/lib/Conversion/HIVMAVEToAVEIntrin/HIVMAVEToAVEIntrin.cpp:3586`
- `/home/m84446336/AscendNPU-IR/bishengir/include/bishengir/Conversion/Passes.td:401`

Detailed source facts:

- Binary vector lowering selects typed intrinsics from element type and vector
  length, then casts values to hardware vector-length size where required.
- `ave.hir.vload` lowering rejects non-1-D vectors, adjusts vector or predicate
  storage to `util::VL_BITS` or `util::PREDICATE_BITS`, computes element
  pointers, and lowers to typed/arch-specific intrinsic paths.
- `ave.hir.masked_store` has similar 1-D and hardware-size assumptions.
- `ave.hir.pge` lowers into a hardware PGE-like intrinsic path with mask
  bit-width attributes.
- Scatter lowering is explicitly not implemented in the inspected local source,
  so gather/scatter rows need separate confirmation.

Key sources:

- `/home/m84446336/AscendNPU-IR/bishengir/lib/Conversion/HIVMAVEToAVEIntrin/HIVMAVEToAVEIntrin.cpp:120`
- `/home/m84446336/AscendNPU-IR/bishengir/lib/Conversion/HIVMAVEToAVEIntrin/HIVMAVEToAVEIntrin.cpp:780`
- `/home/m84446336/AscendNPU-IR/bishengir/lib/Conversion/HIVMAVEToAVEIntrin/HIVMAVEToAVEIntrin.cpp:1018`
- `/home/m84446336/AscendNPU-IR/bishengir/lib/Conversion/HIVMAVEToAVEIntrin/HIVMAVEToAVEIntrin.cpp:1448`

Bridge implication: this is still a plausible first vector bridge boundary, but
only for operations that have not already been lowered by `HIVMToStandard` or
`HIVMAVEToStandard`. It should be treated as a vector/intrinsic boundary, not as
the complete NPU-IR-to-PTOAS boundary.

## Current Bridge Hypotheses

- `ave.hir.vload`, `ave.hir.masked_store`, `ave.hir.pge`, and arithmetic
  `ave.hir.vf*` ops are likely candidates for PTOAS VMI mapping.
- `hivm.hir.load`, `hivm.hir.store`, and `hivm.hir.nd2nz` are likely earlier
  HIVM-to-PTO tile/DMA/MTE mapping candidates, not VMI-only vector rows.
- `hivm.hir.mmadL1`/`mma*` are likely earlier HIVM-to-cube/PTO-ISA/tile mapping
  candidates. They should not be forced through a VMI-only abstraction unless a
  separate cube representation is proven sufficient.
- `hivm.hir.pointer_cast` is kept legal by regbase `HIVMToStandard`, so pointer
  representation may survive later than other HIVM ops.
- `hivm.hir.set_flag` and `hivm.hir.sync_block*` need explicit ownership:
  either preserve NPU-IR-authored sync in PTO level3/manual-sync style, or hand
  sync over to PTOAS level2 auto-sync after proving the semantic gap is safe.
- Sync mapping needs a generation/ownership column. `sync_block` decomposition,
  graph sync, and injected set/wait flags happen before final library-call
  lowering, so the bridge must know whether it is preserving generated sync or
  asking PTOAS to regenerate it.
- Current PTOAS upstream TileLib does not emit VMI by default, but branch
  evidence points toward logical VMI-based TileOp/PTODSL expansion. For this
  bridge, preserve the facts VMI will need even when today's compile path needs
  a lower fallback.

## Open Questions For Mapping Table

- For each key `hivm.hir.*` and `ave.hir.*` op, what is the smallest source test
  that exercises the current lowering?
- Which operations are still present immediately before
  `convert-hivm-to-std`, `convert-hivmave-to-std`, and
  `convert-hivmave-to-ave-intrin` on realistic kernels?
- Which NPU-IR memory scopes correspond cleanly to PTOAS memory spaces and which
  need explicit allocation/planning rows?
- Can vector-only work map to PTOAS VMI directly, or is a PTODSL/tile wrapper
  needed to enter the current VPTO pipeline?
- For cube/matmul paths, should PTOAS receive tile ops, PTO-ISA concepts, or a
  lower micro-op sequence?
- How should A5 validation report IR dumps back to Planner so local development
  can iterate without direct hardware access?
