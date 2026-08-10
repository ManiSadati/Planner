# PTOAS Lowering Pipeline

Last updated: 2026-08-07

Scope: Planner-side Stage 2 summary of PTOAS lowering state. This summarizes the local checkout docs/code and should be cross-checked against upstream PRs before implementation decisions.

## Short Version

PTOAS currently has two broad backend paths:

- `emitc`: older/default C++ codegen path that lowers PTO IR toward EmitC/C++ and PTO-ISA-style calls.
- `vpto`: native VPTO path. This is the design path most relevant to the NPU-IR bridge because it lowers tile-level PTO into VPTO authoring IR, runs the VMI semantic pipeline, then emits VPTO/LLVM/device objects.

For the bridge, the key PTOAS boundary is:

```text
tile-native PTO IR
  -> ExpandTileOp
  -> helper inline/fold
  -> VPTO authoring IR, possibly including pto.vmi.* ops
  -> VMI semantic pipeline for any remaining VMI ops
  -> physical VPTO IR / existing VPTO micro ops
  -> VPTO LLVM/device emission
```

`ExpandTileOp` is a hard boundary in current docs. Older `View2Memref` / `PTOToA5VM` mainline assumptions should not be used as current source of truth.

## Inputs And User-Facing Layers

PTOAS serves several input surfaces:

- textual `.pto` IR for the `ptoas` CLI;
- Python bindings for constructing PTO IR;
- PTODSL / PyPTO style `@pto.jit` kernels;
- TileOps and lower-level micro/VMI operations.

PTODSL `mode` changes ownership:

- `mode="auto"`: author mostly TileOps; PTOAS handles staging, instruction ordering, and sync insertion defaults.
- `mode="explicit"`: author can use MTE DMA, explicit sync, and raw pointer controls. Native launch maps this toward `--pto-level=level3`, where the caller owns local addresses and synchronization by default.

PTODSL `backend` changes final compilation:

- `backend="vpto"` is the default native path and works with `auto` and `explicit`.
- `backend="emitc"` generates C++ and only works with `mode="auto"`.
- PTODSL can mix `emitc` and `vpto` modules in one compilation unit.

## Shared Pre-Backend Pipeline

The main `tools/ptoas/ptoas.cpp` pipeline first normalizes frontend/tile-native PTO IR before choosing the final backend.

Important shared passes include:

```text
PTOMaterializeTileOpSections
PTONormalizeUncoveredTileSections
PTOValidatePhysicalSectionBoundaries

PTOCanonicalizeIR                    # VPTO only today
SerialFrontendPipeLowering
PTOInferValidatePipeInit
LoweringSyncToPipe
InferPTOLayout
PTOA5NormalizeTMov                   # non-A2/A3
PTOValidateIntToPtrUses
InsertTemplateAttributes             # VPTO + PTODSL TileLib path
FusionPlan / OpScheduling / fusion markers
PTORematerializeFixpipeVectorQuant
PTOPlanMemory or PTOPlanMemoryModern # skipped at level3
PTOResolveReservedBuffers
PTORemoveIdentityTMov
optional auto-sync mode
PTOResolveBufferSelect
PTOInlineBackendHelpers
Canonicalizer / CSE
```

Auto-sync mode is mutually selected from:

- `PTOInsertSync`
- `PTOBufidSync`
- `PTOInjectBarrierAllSync`
- `PTOGraphSyncSolver`

Sync runs before `PTOResolveBufferSelect` so it can still see per-use multi-tile slot identity.

## Build Levels And Ownership

The level split matters for the NPU-IR bridge:

| Level | PTOAS behavior | Bridge meaning |
| --- | --- | --- |
| `level2` | Runs memory planning; may run auto sync when requested. | Good when PTOAS should own local address planning and/or sync insertion. |
| `level3` | Skips `PlanMemory`; explicit local addresses required; explicit sync contract preserved. | Good when NPU-IR already owns memory/sync and we translate that structure directly. |

Current memory planning notes:

- legacy memplan is still default;
- modern memplan exists behind `--plan-memory-impl=modern`;
- current tile-native memplan writes addresses back onto `pto.alloc_tile addr`;
- docs say the historical `pto.pointer_cast` bridge is no longer the ordinary memplan materialization result.

That last point matters because the human overview's rough mapping mentioned `hivm.hir.pointer_cast -> pto.castptr`. `pto.castptr` is still a real pointer op, but planner-owned tile allocation may now materialize addresses directly on `pto.alloc_tile` instead of via an old pointer-cast bridge.

## PTODSL TileLib And `ExpandTileOp`

For VPTO tile-op expansion, PTODSL TileLib is the default backend.

Important nuance: `ExpandTileOp` is the tile-op expansion boundary, not a
guarantee that every expanded tile op becomes VMI. Current TileLib helpers can
contain existing VPTO/vector-style ops such as `pto.vlds`, `pto.vadd`, and
`pto.vsts`. Separately, PTOAS also accepts and tests direct `pto.vmi.*` surface
ops. The VMI semantic pipeline always runs in the VPTO backend, but it is only
substantive for functions that contain VMI ops after tile expansion, helper
inlining, or direct user/frontend generation.

Planning takeaway: current upstream implementation and likely design direction
are not the same thing. On current upstream `988d50e24`, `ptodsl/ptodsl/tilelib`
does not use `pto.vmi.*` in its templates, while PTODSL itself exposes a VMI
namespace. WenboCodes' `new-vf-fusion-design` branch is the strongest signal
that TileOp/PTODSL expansion may move toward VMI template bodies. For the
NPU-IR bridge, prefer mappings that preserve enough information for logical VMI
when possible, but record current compile-time fallback requirements explicitly.

The PTODSL path has two Python-daemon interactions:

```text
TileOp in MLIR
  -> InsertTemplateAttributes
       query legal PTODSL template candidates
       store compact candidates attr
  -> ExpandTileOp
       pick first remaining candidate
       render helper with current operands/attrs
       replace TileOp with func.call
  -> PTOInlineLibCall
  -> FoldTileBufIntrinsics
  -> VPTO-facing IR, with VMI ops only if the helper/frontend produced them
```

The specialization key must include op name, architecture, tile/view/vector/scalar operand metadata, and forwarded context attrs. This is a core correctness rule because different shapes/strides/layouts can require different rendered helper bodies.

## VPTO Backend Pipeline

When backend is VPTO and tile ops are still present:

```text
LowerPTOToUBufOps
ExpandTileOp
PTOInlineLibCall
FoldTileBufIntrinsics(shape-only)
optional post-lowering tile fusion
FoldTileBufIntrinsics(addr-only)
SCCP
Canonicalizer
```

Then the unified VPTO pipeline runs:

```text
VPTOSplitCVModule
VPTONormalizeContainer
ApplySIMTEntryNoInline
Func inliner
VMI semantic pipeline
VPTO emission preparation
```

The VPTO emission preparation includes pointer/wrapper normalization, `PTOInferVPTOVecScope`, optional soft-post-update optimization, loop counter narrowing, and `PTOValidateVPTOEmissionIR`.

## Branch Watch: VMI-Level VF Fusion

`WenboCodes/PTOAS` has a branch-local design package at:

```text
https://github.com/WenboCodes/PTOAS/tree/new-vf-fusion-design/docs/new-vf-fusion-design
```

This package is not current upstream `main`, but it is important for bridge planning. Its RFC proposes replacing PyPTO2's LLVM/physical-layer VF fusion pipeline with a VMI-layer VF-fusion pass plus a post-fusion `mem2reg` pass.

The proposed logical flow is:

```text
TileOP template expansion
  -> independent scf.for loops over N x VL with pto.vmi ops
  -> VMI-level VF fusion eligibility checks
  -> compatible loops merged into shared loops
  -> VMI mem2reg removes UB vstore/vload handoff
  -> pto.as layout assignment and lowering
  -> physical pto.mi / VPTO
```

Key ideas to keep visible:

- fusion eligibility should use VMI-layer trip count, adjacency, alias, and access-pattern information rather than reconstructing it from physical `pto.mi` / LLVM IR;
- `!pto.ptr` may need shape/stride information and `vload`/`vstore` may need multidimensional index expressions to preserve structured access information;
- `N` is static tile buffer capacity, while `valid row` and lane-level tail are dynamic concepts carried separately;
- reduce/elementwise fusion depends on dtype-dependent `VL` and accumulator lifetime, not only on traversal direction;
- post-fusion `mem2reg` is the intended way to remove tileop-to-tileop UB round-trips.

Bridge implication: if NPU-IR lowering erases shaped access, loop, mask, or accumulator-lifetime facts before PTOAS sees them, this future VMI fusion direction becomes harder or impossible to exploit.

## Branch Watch: Legacy Tile-Fusion And VMI Contracts

Focused re-exploration found older branch-local docs that are useful for context but should not override current upstream/mainline evidence.

`zhendong404/PTOAS` tile-fusion branches describe an older A5 Level-3 OpLib/tile-fusion path. Important ideas include `pto.fusion_region`, tile-fusion planning before structure is erased, DFG/lifetime analysis, template lowering, register passing to remove UB store/load handoffs, sync placement, and register-pressure/cost-model hazards.

However, those docs also rely on older pipeline pieces such as OpLib concrete templates, `PTOViewToMemref`, `pto.simd.*`, and EmitC-style bridging. For current bridge planning, treat them as hazard context, not as the active PTOAS architecture.

`mouliangyu/PTOAS:vmi-per-block-cast` contains a branch-only `vmi-dialect-design.md` that reinforces the current VMI contract:

- VMI is a logical semantic vector layer, not a physical VPTO dialect.
- Producers should emit VMI ops or VMI compositions after entering VMI.
- Physical layout choices and target fallbacks belong to explicit layout assignment/lowering, not hidden producer-side assumptions.
- After layout assignment, VMI data and masks should carry enough concrete layout information for `VMIToVPTO` without re-solving hidden context.

Bridge implication: the NPU-IR-to-PTOAS conversion should preserve logical vector semantics cleanly and avoid hard-coding physical VPTO layout too early. If a mapping requires a physical choice, the mapping table should call that out as a risk instead of burying it in code.

## VMI Semantic Pipeline

VMI is the logical vector layer before physical VPTO. It represents logically contiguous vector registers and masks, while layout assignment decides how those values are split across physical 256B vector registers and predicate registers.

Current driver pipeline:

```text
VMINormalizeSignlessIntToUnsigned
VMILowerUnifiedToLegacy
Canonicalizer
VMILegalizeArithSelect
PTOValidateVMIIR
Canonicalizer
CSE
VMIPreAssignmentCombine
Canonicalizer
CSE
VMILegalizeArithSelect
VMIMaskGranularityAssignment
VMILayoutAssignment
Canonicalizer
CSE
VMILayoutRematerialize
Canonicalizer
CSE
VMILayoutFold
Canonicalizer
CSE
VMILayoutSinkMaterialization
Canonicalizer
CSE
VMILegalizeArithSelect
PTOValidateVMILayoutIR
VMIToVPTO
```

The design rule is strict:

- surface VMI should not carry concrete layouts;
- layout decisions are written into `!pto.vmi.*` types or explicit helper ops;
- `vmi-to-vpto` must not redo layout solving or recover hidden context;
- after `vmi-to-vpto`, no `pto.vmi.*`, `!pto.vmi.*`, or residual conversion casts should remain.

## VMI Concepts Relevant To NPU-IR Mapping

VMI ISA folder coverage check: on 2026-08-07, Codex verified the local
`upstream/main` VMI ISA directory has these files and that the current local
branch has no diff from `upstream/main` under `docs/isa/vmi-isa`:

```text
00-architecture-overview.md
01-load-store.md
02-index-gen.md
03-eltwise-compute.md
04-broadcast.md
05-reduce.md
06-convert.md
07-sfu.md
08-predicate-ops.md
09-data-rearrange.md
10-appendices.md
```

The Stage 2 summary initially cited the most directly relevant files
(`00`, `01`, and `08`), but the mapping table should use the full folder. The
extra files matter for NPU-IR rows covering index generation, eltwise ops,
broadcast, reductions, conversion, SFU/fused ops, gather/scatter, data
rearrange, `vselr`, and A5 merge-mode emulation.

VMI types:

- data: `!pto.vmi.vreg<LxT>`
- mask: `!pto.vmi.mask<Lxpred>` on the surface, then concrete `b8`/`b16`/`b32` granularity after assignment

Important operations for first mapping table:

- `pto.vmi.vload`: logical UB load. A5 loads are unpredicated; tail masks are migrated to consumers/stores or expressed through shorter loads.
- `pto.vmi.vstore`: logical UB store, predicated on A5.
- `pto.vmi.vadd` and similar elementwise ops: logical compute, mask governed.
- `pto.vmi.create_mask`: first-N/all-active mask generation.
- `pto.vmi.create_group_mask`: grouped mask generation.
- VMI lowering categories: passthrough, layout-rewritable, and contiguous-required.

For NPU-IR bridge planning, this means a rough `ave.hir.pge -> pto.vmi.create_mask` mapping is plausible, but mask granularity/layout must be left to PTOAS VMI assignment rather than hard-coded early.

## Sync And Memory Planning

PTOAS has both explicit sync primitives and auto-sync passes.

Micro sync primitives include:

- `pto.set_flag`
- `pto.wait_flag`
- `pto.pipe_barrier`
- `pto.get_buf`
- `pto.rls_buf`
- dynamic `get_buf`/`rls_buf` variants

InsertSync's job is to turn real cross-pipeline memory dependencies into minimal necessary sync. The documented pipeline is:

```text
PTOIRTranslator
InsertSyncAnalysis
MoveSyncState
RemoveRedundantSync
SyncEventIdAllocation
SyncCodegen
```

Bridge implication: if the NPU-IR side already provides explicit `set_flag` / `sync_block_*` semantics, we need to choose whether the generated PTO uses:

- level3/manual sync, preserving NPU-IR ownership;
- PTOAS auto-sync, requiring us to provide memory effects/addresses in a way PTOAS can analyze;
- a deliberately mixed mode for specific patterns.

This should become a table column in the NPU-IR mapping, not an afterthought.

## Tile DMA And Cube/Tile Boundary

Tile-level data movement is represented with `pto.tload` and `pto.tstore` over tensor views and tile buffers:

- `pto.tload`: GM/partition view to local tile, usually `PIPE_MTE2`.
- `pto.tstore`: tile to partition view, `PIPE_MTE3` for vec/mat sources and `PIPE_FIX` for acc sources.

Tile instructions are defined as a high-performance layer built on micro instructions. A tile op can expand into vector loads/stores, compute ops, masks, and sync flags.

This supports the current bridge hypothesis:

- vector-side HIVM-AVE operations should likely map through VMI;
- DMA/tile movement may map better through tile ops or lower PTO micro ops depending on how much structure remains at the chosen NPU-IR interception point;
- cube instructions/templates may need earlier interception or explicit PTO-ISA/PTO tile mappings, because late CCE-template calls may hide intent.
- future VMI-level VF fusion, as sketched in WenboCodes' branch, makes structured loop/access/mask preservation a performance requirement, not just a cleanliness preference.

## Current Watch Risks

Recent upstream PRs/issues show active movement in:

- `scf.while` / `scf.if` support and branch merge correctness;
- PTODSL scalar and SIMT surface cleanup;
- VMI predicate folding and mask spill behavior;
- `vscatter`, gather/scatter, GlobalTensor casts;
- explicit L1-to-L0 loads and MX quant movement;
- allocator/sync/bufid/event-id design;
- possible LLVM19 downgrade / VPTO branch adaptation.

Before implementing the conversion pass, cross-check these active areas against the first NPU-IR mapping table.

## Key Local Source References

- `$HOME/PTOAS/PTOAS_Markham/tools/ptoas/ptoas.cpp`
- `$HOME/PTOAS/PTOAS_Markham/docs/ptoas-tile-fusion-design.md`
- `$HOME/PTOAS/PTOAS_Markham/docs/designs/vmi-introduction.md`
- `$HOME/PTOAS/PTOAS_Markham/docs/designs/vmi-implementation-manual.md`
- `$HOME/PTOAS/PTOAS_Markham/docs/designs/ptodsl-tilelib-template-selection-design.md`
- `$HOME/PTOAS/PTOAS_Markham/docs/designs/ptoas-auto-sync-design.md`
- `$HOME/PTOAS/PTOAS_Markham/docs/designs/ptoas-largest-first-fit-four-gates-memplan-design.md`
- `$HOME/PTOAS/PTOAS_Markham/ptodsl/README.md`
- `$HOME/PTOAS/PTOAS_Markham/ptodsl/docs/user_guide/01-introduction.md`
- `$HOME/PTOAS/PTOAS_Markham/docs/isa/vmi-isa/00-architecture-overview.md`
- `$HOME/PTOAS/PTOAS_Markham/docs/isa/vmi-isa/01-load-store.md`
- `$HOME/PTOAS/PTOAS_Markham/docs/isa/vmi-isa/08-predicate-ops.md`
- `$HOME/PTOAS/PTOAS_Markham/docs/isa/micro-isa/01-pipeline-sync.md`
- `$HOME/PTOAS/PTOAS_Markham/docs/isa/tile-op/03-pointer-and-view.md`
- `$HOME/PTOAS/PTOAS_Markham/docs/isa/tile-op/04-dma-data-movement.md`

## External Source References

- `https://github.com/hw-native-sys/PTOAS`
- `https://github.com/hw-native-sys/PTOAS/issues`
- `https://github.com/hw-native-sys/PTOAS/pulls`
- `https://github.com/hw-native-sys/PTOAS/branches`
- `https://github.com/hw-native-sys/PTOAS/tree/main/docs/isa/vmi-isa`
- `https://github.com/mouliangyu/PTOAS`
- `https://github.com/WenboCodes/PTOAS/tree/new-vf-fusion-design/docs/new-vf-fusion-design`
- `https://raw.githubusercontent.com/zhendong404/PTOAS/tile-fusion-stage2/docs/tile_fusion/oplib_lowering_tile_fusion_design_v1.md`
- `https://raw.githubusercontent.com/zhendong404/PTOAS/rewrite/tile-fusion-2-pr-ready-20260319/docs/tile_fusion/tile_fusion_design_spec.md`
- `https://raw.githubusercontent.com/mouliangyu/PTOAS/vmi-per-block-cast/docs/designs/vmi-dialect-design.md`
