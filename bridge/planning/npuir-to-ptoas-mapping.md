# NPU-IR To PTOAS Mapping Draft

Last updated: 2026-08-27

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

Current Cube update: `bridge/triton-example/cube_dotproduct.py` has been traced
through the last structured HIVM form and all three selected CCE templates. The
64x64 fixture uses two `hivm.hir.nd2nz` operations, one
`hivm.hir.mmadL1`, and one NZ2ND/F322F16 `hivm.hir.fixpipe`. Existing PTO
primitives cover the region, but generic `mmadL1` requires a composition or
template rewrite because it also owns L1-to-L0 staging, K scheduling, sync, and
barriers. See `bridge/planning/cube-conversion-exploration.md`.
The strict 64x64 composition is now implemented and verified through
real-fixture VMI/VPTO emission and PTOAS simulator numerical comparison. Direct
CCE performance comparison and A5 validation remain pending.

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
| `hivm.hir.nd2nz` | In `cube_dotproduct`, two ops move 64x64 f16 GM ND matrices to 4x4x16x16 CBUF/L1 NZ buffers. The selected template emits `copy_gm_to_cbuf_multi_nd2nz_b16`, with a row-loop fallback for a large GM leading stride. | Tile route: `pto.tload` into MAT. First low-level route: `pto.mte_gm_l1_frac` in `nd2nz` mode; its PTOAS wrapper expansion emits `pto.copy_gm_to_cbuf_multi_nd2nz`. | After `hivm-mark-disable-load` and sync finalization, before `convert-hivm-to-std`. | confirmed for the fixture; guarded-direct family | Preserve source/destination layout, rank, dtype, shape, continuous destination, intrinsic stride limits, and fallback behavior. | N4, N5, P6, P8, P14 | Not a VMI row. The first patch may specialize the observed b16/64x64 form and reject the large-stride fallback. |
| `hivm.hir.mmadL1` | In `cube_dotproduct`, consumes two f16 L1 NZ buffers and initializes a 64x64 f32 L0C result. The selected `mma_tile_half_to_float` template performs K partitioning, L1-to-L0A/B moves, optional ping-pong, MTE1/M sync, MAD, and barriers. | Fixture composition: `pto.mte_l1_l0a`, `pto.mte_l1_l0b`, and `pto.mad`, or tile-level `pto.textract` plus `pto.tmatmul`. Generic forms need accumulate/bias/MX variants and reconstructed scheduling. | Existing optional bridge slot after sync finalization and before `convert-hivm-to-std`. | PTO composition for strict fixture; template rewrite for generic family | A bare `pto.tmatmul` is not equivalent. Preserve K splitting, init/accumulate, transpose, HF32/I4/MX, unit flags, physical addresses, and event ownership. | N4, N6, P4, P9, P14 | First route should preserve NPU-IR memory and sync ownership; PTOAS tile ownership is a separate longer-term design. |
| `hivm.hir.fixpipe`, L0C->GM NZ2ND F322F16 | In `cube_dotproduct`, `fixpipe_nz2nd_float_to_half_4d_to_2d_gm` configures ND parameters and emits `copy_matrix_cc_to_gm` with NZ2ND and f32-to-f16 conversion. | Tile route: `pto.tstore` from ACC. Low-level route: `pto.mte_l0c_gm` or `pto.copy_matrix_cc_to_gm`. | Same structured Cube boundary, before `convert-hivm-to-std`. | confirmed candidate direct mapping for the exact fixture mode | Verify packed mode, source/destination strides, conversion, pipe assignment, and absence of relu, dual destination, channel split, or extra quantization. | N3, N4, P7, P13, P14 | Keep other fixpipe destinations and modes as separate guarded rows. |
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
- N5: `$HOME/AscendNPU-IR/bishengir/lib/Template/lib/RegBase/Cube/CMakeLists.txt:1`; `$HOME/AscendNPU-IR/bishengir/lib/Template/lib/RegBase/Cube/include/DMA/ND2NZ.h`; `$HOME/AscendNPU-IR/bishengir/lib/Template/lib/RegBase/Cube/compat/DMA/Cbuf/nd2nz.cpp:17`
- N6: `$HOME/AscendNPU-IR/bishengir/lib/Dialect/HIVM/IR/LibraryFunctionOpInterface/LibraryFunctionOpInterfaceImpl.cpp:1104`, `:1204`; `$HOME/AscendNPU-IR/bishengir/lib/Template/lib/RegBase/Cube/LocalMmad.cpp:55`, `:139`, `:156`, `:239`
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
- P14: `bridge/planning/cube-conversion-exploration.md`; current PTOAS
  `include/PTO/IR/PTOOps.td`, `include/PTO/IR/VPTOOps.td`, and
  `ptodsl/examples/fa_dn_matmul.py`

## Next Work

- Run the unchanged CCE simulator path for direct trace/tick comparison with
  the passing strict 64x64 bridge fixture.
- Validate NPU-IR-owned memory addresses and explicit sync on A5 hardware.
- Keep the CCE path as the fallback and baseline while generalizing K
  partitioning, accumulation, transpose, bias, precision, and fixpipe modes.
