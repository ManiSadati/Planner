# NPU-IR To PTOAS Mapping Draft

Last updated: 2026-08-26

Scope: local-first mapping draft from Stage 2 PTOAS context, Stage 3 local
NPU-IR source scan, and Stage 4 local PTO-ISA source scan. This table has not
yet been reconciled with current Ascend upstream or a fresh PR/branch scan.

Prototype review note: `soyu-wilson/AscendNPU-IR:codex/ave-to-vmi` was reviewed
as vector-side prototype context. It supports the idea of an HIVMAVE-to-VMI pass
before `convert-hivmave-to-ave-intrin`, but should not be continued directly or
treated as implementation authority. See
`bridge/planning/soyu-wilson-ave-to-vmi-branch-review.md`.

Vector implementation update: the interception point and VMI-first direction
are implemented in AscendNPU-IR. Vector add, row softmax, and RMSNorm establish
good practical vector coverage for the current stage, with the accepted
softmax and RMSNorm fixtures supported at performance on par with NPU-IR.
Vector rows in this broad table are now maintenance context; authoritative
implemented mappings, restrictions, and non-direct expansions are in
`bridge/planning/ave-to-ptoas-vmi-implementation-plan.md` and
`bridge/designs/ave-to-ptoas-vmi-conversion-design.md`.

Current Cube update: `bridge/triton-example/cube_dotproduct.py` is the first
fixture, but no Cube row is confirmed from that fixture yet. The active task is
to compare each selected NPU-IR CCE template implementation against PTOAS/PTO
semantics and classify it as direct, PTO composition, template rewrite, or
unsupported. See `bridge/planning/cube-conversion-exploration.md`.

DMA rewrite note: the current DMA/template-specific plan is tracked in
`bridge/planning/dma-template-rewrite-plan.md`, with compact memory in
`bridge/memory/dma-template-mapping.md`. The source-backed `dma_copy_kernel`
trace is `bridge/memory/dma-copy-conversion-trace.md`.

Status meanings:

- `confirmed`: directly established by local source and unlikely to change per
  row semantics.
- `likely`: source evidence supports the mapping direction, but details remain.
- `hypothesis`: plausible design direction requiring examples or upstream check.
- `unknown`: do not implement from this row yet.

## Table

| NPU-IR op/pattern | Current NPU-IR lowering | Proposed PTO/PTOAS target | Best interception point | Status | Risk | Source references | Notes |
|---|---|---|---|---|---|---|---|
| `ave.hir.vload` | Lowered from MLIR vector load/transfer into HIVMAVE; later `convert-hivmave-to-ave-intrin` lowers 1-D loads to typed A5 intrinsic paths and hardware vector/predicate widths. | Preferred: PTOAS VMI logical vector load. Fallback/direct PTO-ISA: `TLOAD` only if the bridge is emitting tile/AS with a `GlobalTensor` + `TileType::Vec` model. | After HIVMAVE normalization, before `convert-hivmave-to-ave-intrin`. | likely | VMI and PTO-ISA tile load have different memory-view and mask models; A5 intrinsic lowering already assumes 1-D/vector-length decisions. | N1, N2, N8, P6 | Preserve logical vector length, mask, stride/access pattern, and memory view before physical layout selection. |
| `ave.hir.masked_store` | MLIR vector store path creates a full-lane `ave.hir.pge` and `ave.hir.masked_store`; intrinsic lowering lowers masked store under 1-D/hardware-size assumptions. | Preferred: PTOAS VMI masked store/predicate store. PTO-ISA `TSTORE` is only a fallback for valid-region-style stores, not arbitrary masks. | After HIVMAVE normalization, before `convert-hivmave-to-ave-intrin`. | likely | PTO-ISA valid region is rectangular/prefix-like; VMI can better preserve predicate semantics. | N1, N2, N8, P7 | Do not collapse predicate masks into tile valid-region unless equivalence is proven. |
| `ave.hir.pge` | Predicate generation lowers to hardware PGE-like intrinsic path with mask bit-width attributes. | PTOAS VMI predicate/mask op. PTO-ISA may represent simple masks through valid regions or mask tiles, but this needs case-by-case proof. | Before `convert-hivmave-to-ave-intrin`. | likely | Physical mask bit width may leak too early if intercepted after intrinsic lowering. | N2, N8, P2 | Preserve logical predicate coverage separately from physical mask granularity. |
| `ave.hir.vf*` arithmetic, e.g. `ave.hir.vfadd` | `convert-arith-to-hivmave` maps arith/math ops to HIVMAVE vector ops; intrinsic lowering selects typed vector intrinsics by dtype and vector length. | PTOAS VMI arithmetic for future-compatible VPTO path. PTO-ISA tile ops such as `TADD` are possible only when the bridge chooses a tile-ISA entry point. | Before `convert-hivmave-to-ave-intrin`; possibly earlier after `convert-hfusion-to-vector` for MLIR-vector-like semantics. | likely | Need choose VMI vs tile entry point; reductions/broadcasts/selects may need separate rows. | N1, N2, P3, P6 | Good first vector prototype row if memory/sync is kept minimal. |
| `hivm.hir.load` | In `dma_copy_kernel`, remains structured through `hivm-mark-disable-load`; `convert-hivm-to-std` rewrites it to `load_gm_to_ubuf_1d_float`. Current regbase templates route the contiguous C310/A5 path to `copy_gm_to_ubuf_align_v2`. | First target: low-level VPTO `pto.mte_gm_ub` with explicit sync preserved. Longer-term target: PTOAS tile load / PTO-ISA `TLOAD` when tile/view ownership is resolved. | After `hivm-mark-disable-load`, before `convert-hivm-to-std`. | confirmed for simple GM->UB row | Need preserve memory scope, UB placement, dynamic valid length, dtype, stride, padding facts, and explicit sync ownership. | N3, N4, N9, P3, P6, P13 | First dry-run pass should match rank-1 contiguous GM->UB only and record zero-pad separately from unsupported nonzero padding. |
| `hivm.hir.store` | In `dma_copy_kernel`, remains structured through `hivm-mark-disable-load`; `convert-hivm-to-std` rewrites it to `store_ubuf_to_gm_1d_float`. Current regbase templates route the contiguous C310/A5 path to `copy_ubuf_to_gm_align_v2`. | First target: low-level VPTO `pto.mte_ub_gm` with explicit sync preserved. Longer-term target: PTOAS tile store / PTO-ISA `TSTORE` when tile/view ownership is resolved. | After `hivm-mark-disable-load`, before `convert-hivm-to-std`. | confirmed for simple UB->GM row | A5 `TSTORE` has different Vec vs Acc constraints, atomic/quantization modes, and valid-region assumptions. Atomic store is out of scope for the first row. | N3, N4, N9, P7, P13 | Accumulator stores should be split from vector stores in the implementation table. |
| `hivm.hir.nd2nz` | Regbase `HIVMToStandard` lowers `ND2NZOp` to a library call; template maps GM `(n,d)` to L1 `(n1,d1,d0,n0)` with `n0 * sizeof(T) = 32B`, `d0 = 16`. | PTO-ISA `TLOAD` Mat ND-to-NZ style path if source is GM; PTO-ISA `TMOV` ND-to-NZ path if source/destination are tiles; PTOAS tile/DMA row otherwise. | Before `convert-hivm-to-std`, probably after layout/memory facts are known. | likely | Need know whether the bridge sees GM-to-L1 movement or tile-to-tile movement. | N4, N5, P6, P8 | Not a VMI row. Preserve source layout, target L1/NZ layout, dtype, and shape. |
| `hivm.hir.mmadL1` | Structured cube macro op lowers toward `mma_tile` library/template. Carries A/B/C operands, init condition, real `m/k/n`, sync args, unit-flag mode, transpose/HF32/I4/bias attributes. | PTO-ISA `TMATMUL`, `TMATMUL_ACC`, `TMATMUL_BIAS`, or `TMATMUL_MX`, plus surrounding `TLOAD`/`TMOV`/`TSTORE`. | Before `convert-hivm-to-std`; exact point depends on whether NPU-IR or PTOAS owns sync/memory planning. | likely | VMI-only mapping loses cube roles, accumulator init/source, unit-flag phase, and sync. | N4, N6, P4, P9 | Use this as first cube row, not `ave.hir.*`. |
| Global `hivm.hir.matmul` / `mix_matmul` family | `HIVMToStandard` pattern list includes global matmul and mix matmul ops; library-call naming exists for global MMAD-like ops. | Either lower/decompose to PTO-ISA `TMATMUL*` tile rows or map to a higher PTOAS tile matmul op if one exists. | Unknown until decomposition path is traced on examples. | hypothesis | High-level global tiling/process-size semantics may not map one-to-one to `TMATMUL` without schedule decomposition. | N3, N6, P4 | Needs an example IR dump before implementation. |
| `hivm.hir.pointer_cast` | Kept legal by regbase `HIVMToStandard`, unlike many HIVM ops. | Possible PTOAS/PTO-ISA address/view binding helper; manual placement may relate to `TASSIGN`, but this is not confirmed. | Later than many HIVM ops, but before address semantics are lost. | unknown | Pointer representation, memory scope, and manual/auto placement semantics are not mapped yet. | N3, P2, P3 | Do not implement until pointer/view ownership is explicit. |
| `hivm.hir.set_flag` / `hivm.hir.wait_flag` | Generated by InjectSync and GraphSyncSolver; carry source/wait pipe and event-id semantics. | PTO event model: `Event<SrcOp,DstOp>`, AS `record_event`/`wait_event`, or `TSYNC(events...)` if pipe pair/token semantics match. | After NPU-IR sync generation, before final lowering consumes/rewrites sync. | hypothesis | Mechanical mapping can be wrong if pipe IDs, event IDs, token lifetime, or auto/manual mode differ. | N7, P5, P10 | Mapping table needs a sync ownership column before implementation. |
| `hivm.hir.sync_block` / `sync_block_set` / `sync_block_wait` | `sync_block` decomposes into set/wait/barrier forms; memory planning treats only `SyncBlockWaitOp` as a cross-core RECEIVE sync guarantee in the inspected path. | PTO `SYNCALL` only if participant-set and barrier semantics match; otherwise explicit PTO-AS sync or NPU-IR-owned lower sync. | After `sync_block` decomposition if preserving NPU-IR sync; before decomposition only if PTOAS will own sync globally. | unknown | `SYNCALL` is a participant rendezvous, not a generic replacement for every NPU-IR cross-core sync. | N7, P10, P11 | Needs example kernels and A5 validation path. |
| HIVMAVE gather/scatter-like vector ops | HIVMAVE intrinsic pattern list includes gather/scatter, but local scan found scatter lowering explicitly not implemented in one path. | PTOAS VMI gather/scatter for vector semantics; PTO-ISA `MGATHER`/`MSCATTER` for GM indexed tile movement. | Before `convert-hivmave-to-ave-intrin`, but confirm per op. | unknown | Gather/scatter semantics vary by row/element coalesce, OOB, conflict/atomic policy, and target. | N2, P12 | Defer until a source test identifies exact NPU-IR op shape. |

## Source References

- N1: `$HOME/AscendNPU-IR/bishengir/lib/Conversion/VectorToHIVMAVE/VectorToHIVMAVE.cpp:75`, `:960`; `$HOME/AscendNPU-IR/bishengir/lib/Conversion/ArithToHIVMAVE/ArithToHIVMAVE.cpp:1976`
- N2: `$HOME/AscendNPU-IR/bishengir/lib/Conversion/HIVMAVEToAVEIntrin/HIVMAVEToAVEIntrin.cpp:120`, `:780`, `:1018`, `:1448`, `:3430`
- N3: `$HOME/AscendNPU-IR/bishengir/lib/Conversion/HIVMToStandard/HIVMToStandard.cpp:1936`; `$HOME/AscendNPU-IR/bishengir/lib/Conversion/HIVMToStandard/regbase/HIVMToStandard.cpp:1902`, `:1978`
- N4: `$HOME/AscendNPU-IR/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td:62`, `:156`, `:356`; `$HOME/AscendNPU-IR/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMMacroOps.td:58`, `:173`
- N5: `$HOME/AscendNPU-IR/bishengir/lib/Template/include/DMA/ND2NZ.h:23`; `$HOME/AscendNPU-IR/bishengir/lib/Template/lib/DMA/Cbuf/nd2nz.cpp:19`
- N6: `$HOME/AscendNPU-IR/bishengir/lib/Dialect/HIVM/IR/LibraryFunctionOpInterface/LibraryFunctionOpInterfaceImpl.cpp:1104`, `:1204`; `$HOME/AscendNPU-IR/bishengir/lib/Template/lib/Cube/LocalMmad.cpp:314`
- N7: `$HOME/AscendNPU-IR/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMSynchronizationOps.td:43`, `:87`, `:129`; `$HOME/AscendNPU-IR/bishengir/lib/Dialect/HIVM/Transforms/HIVMDecomposeOp.cpp:400`; `$HOME/AscendNPU-IR/bishengir/lib/Dialect/HIVM/Transforms/InjectSync/InjectSync.cpp:75`; `$HOME/AscendNPU-IR/bishengir/lib/Dialect/HIVM/Transforms/GraphSyncSolver/SyncSolverCodeGen.cpp:197`; `$HOME/AscendNPU-IR/bishengir/lib/Dialect/HIVM/Transforms/regbase/PlanMemory.cpp:171`
- N8: `$HOME/AscendNPU-IR/bishengir/include/bishengir/Dialect/HIVMAVE/IR/HIVMAVEOps.td:35`, `:404`, `:444`, `:455`, `:542`, `:744`
- N9: `bridge/memory/dma-copy-conversion-trace.md`
- P1: `$HOME/pto-isa/docs/mkdocs/src/manual/01-overview.md:5`, `:14`, `:22`; `$HOME/pto-isa/docs/mkdocs/src/manual/08-virtual-isa-and-ir.md:8`, `:40`
- P2: `$HOME/pto-isa/docs/mkdocs/src/manual/03-state-and-types.md:7`, `:18`, `:29`, `:40`
- P3: `$HOME/pto-isa/include/pto/common/type.hpp:122`, `:169`, `:224`; `$HOME/pto-isa/include/pto/common/pto_tile.hpp:261`, `:1394`, `:1428`, `:1597`
- P4: `$HOME/pto-isa/docs/isa/TMATMUL.md:8`; `$HOME/pto-isa/include/pto/npu/a5/TMatmul.hpp:28`, `:130`, `:159`, `:257`
- P5: `$HOME/pto-isa/include/pto/common/event.hpp:21`, `:115`, `:259`; `$HOME/pto-isa/include/pto/npu/a5/TSync.hpp:38`, `:73`, `:98`
- P6: `$HOME/pto-isa/docs/isa/TLOAD.md:8`; `$HOME/pto-isa/include/pto/npu/a5/TLoad.hpp:302`, `:599`, `:651`
- P7: `$HOME/pto-isa/docs/isa/TSTORE.md:8`; `$HOME/pto-isa/include/pto/npu/a5/TStore.hpp:115`, `:182`
- P8: `$HOME/pto-isa/docs/isa/TMOV.md:8`; `$HOME/pto-isa/include/pto/npu/a5/TMov.hpp:451`, `:553`, `:622`
- P9: `$HOME/pto-isa/include/pto/common/pto_instr.hpp:658`; `$HOME/pto-isa/include/pto/common/type.hpp:232`
- P10: `$HOME/pto-isa/docs/isa/TSYNC.md:8`; `$HOME/pto-isa/include/pto/common/pto_instr.hpp:47`, `:131`; `$HOME/pto-isa/include/pto/npu/a5/TSync.hpp:17`
- P11: `$HOME/pto-isa/docs/isa/SYNCALL.md:1`; `$HOME/pto-isa/include/pto/common/type.hpp:267`
- P12: `$HOME/pto-isa/docs/isa/MGATHER.md:1`; `$HOME/pto-isa/docs/isa/MSCATTER.md:1`; `$HOME/pto-isa/include/pto/npu/a5/MGather.hpp:425`, `:452`; `$HOME/pto-isa/include/pto/npu/a5/MScatter.hpp:391`, `:456`
- P13: `$HOME/PTOAS/PTOAS_Markham/include/PTO/IR/VPTOOps.td`; `$HOME/PTOAS/PTOAS_Markham/lib/PTO/Transforms/VPTOExpandWrapperOps.cpp`; `$HOME/PTOAS/PTOAS_Markham/lib/PTO/Transforms/VPTOCANN900LLVMEmitter.cpp`

## Next Work

- Use `cube_dotproduct.py` to trace the first Cube row and its relevant DMA,
  staging, accumulator, result, and sync rows.
- Locate the actual selected CCE template implementations; do not decide from
  operation or function names alone.
- Compare each contract with current PTOAS/PTO operations and record direct,
  composition, rewrite, or unsupported status.
- Reconcile the resulting rows with current Ascend upstream and the active
  Wilson branch before implementation.
- Split rows into implementation issues only after the mapping table and bridge
  entry points have been reviewed.
