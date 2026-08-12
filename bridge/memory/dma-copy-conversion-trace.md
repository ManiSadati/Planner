# DMA Copy Conversion Trace

Last updated: 2026-08-12

## Scope

This note records the source-backed DMA trace for `dma_copy_kernel` and chooses
the first bridge boundary for converting AscendNPU-IR DMA to something PTOAS can
understand.

Local evidence:

- input fixture: `bridge/triton-example/dma_copy_kernel.mlir`
- replay log: `bridge/examples/npuir-early-ir/replay/dma_copy_kernel/compile.log`
- endpoint dump: `bridge/examples/npuir-early-ir/replay/dma_copy_kernel/after-convert-hivmave-to-ave-intrin.mlir`
- replay wrapper: `bridge/tools/replay_npuir_from_device_spec.sh`

The replay was run with `--mlir-disable-threading` and
`--mlir-print-ir-after-all`. The compiler still exits nonzero after the target
dump on this non-A5 server because `hivmc-a5` is unavailable, but the needed
pass dumps are captured before that downstream failure.

## Decision

The first DMA bridge should intercept after `MarkDisableLoad
(hivm-mark-disable-load)` and before `ConvertHIVMToStandard
(convert-hivm-to-std)`.

This is the best current sweet spot because:

- `hivm.hir.load` and `hivm.hir.store` are still structured operations;
- source and destination address spaces are explicit: `gm` and `ub`;
- dtype, rank, dynamic valid length, and stride facts are still visible;
- NPU-IR memory planning has already exposed UB pointer casts and
  double-buffering;
- NPU-IR sync planning has already exposed the MTE/V/MTE event sequence;
- the next pass rewrites DMA into template/helper calls and loses clean
  structured-DMA operands.

The first implementation should preserve NPU-IR-authored memory placement and
sync. Treat this as a level3/manual-sync style bridge until a separate decision
hands ownership to PTOAS.

## Pass Trace

| Step | Pass boundary | Representation | Mapping value |
|---|---|---|---|
| 0 | `AppendTargetDeviceSpec (hacc-append-device-spec)` | `memref.copy` for GM-like input to local buffer, then `bufferization.materialize_in_destination` for output | Useful source intent only. No HIVM memory spaces, no pipe sync. |
| 1 | `ConvertToHIVMOp (convert-to-hivm-op)` | First `hivm.hir.load` and `hivm.hir.store` appear | Structured DMA exists, but address spaces and final sync are not concrete enough. |
| 2 | `PlanMemoryRegBase (hivm-plan-memory-regbase)` | `gm` and `ub` address spaces appear; UB buffer is represented by `hivm.hir.pointer_cast`; `hivm.multi_buffer = 2` is visible | Usable backup boundary. Still earlier than final sync placement. |
| 3 | `HIVMLowerToLoops` / `GraphSyncSolver` | Structured load/store remain; sync begins to become explicit | Useful for confirming schedule evolution. |
| 4 | `MarkDisableLoad (hivm-mark-disable-load)` | Last structured-DMA point with final visible memory/sync context | Selected sweet spot. |
| 5 | `ConvertHIVMToStandard (convert-hivm-to-std)` | `hivm.hir.load/store` are replaced with `func.call @load_gm_to_ubuf_1d_float` and `func.call @store_ubuf_to_gm_1d_float` | Good comparison/fallback evidence, too late as primary bridge input. |
| 6 | `convert-hivmave-to-ave-intrin` endpoint | DMA is still helper calls; vector body is lower-level intrinsic form | Too late for structured DMA/cube mapping. |

## Key IR Shapes

At the A5 input boundary, movement is still frontend-ish:

```mlir
memref.copy %subview, %subview_0
  : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>

bufferization.materialize_in_destination %extracted_slice in writable %subview_2
  : (tensor<?xf32>, memref<?xf32, strided<[1], offset: ?>>) -> ()
```

At first structured HIVM conversion:

```mlir
hivm.hir.load
  ins(%subview : memref<?xf32, strided<[1], offset: ?>>)
  outs(%subview_0 : memref<?xf32>)
  pad_mode = <PadValue> pad_value = %cst : f32
  left_padding_num = %c0 : index eviction_policy = <EvictFirst>

hivm.hir.store
  ins(%extracted_slice : tensor<?xf32>)
  outs(%subview_2 : memref<?xf32, strided<[1], offset: ?>>)
```

At memory planning:

```mlir
%5 = hivm.hir.pointer_cast(%c0_i64, %c1024_i64)
  : memref<256xf32, #hivm.address_space<ub>>
annotation.mark %5 {hivm.multi_buffer = 2 : i32}

hivm.hir.load
  ins(%subview : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
  outs(%subview_0 : memref<?xf32, #hivm.address_space<ub>>)
  pad_mode = <PadValue> pad_value = %cst : f32
  eviction_policy = <EvictFirst> core_type = <VECTOR>

hivm.hir.store
  ins(%subview_2 : memref<?xf32, strided<[1]>, #hivm.address_space<ub>>)
  outs(%subview_3 : memref<?xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>)
```

At the selected boundary, the same structured DMA is surrounded by explicit
sync:

```mlir
hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, %14]
hivm.hir.load ...
hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]

hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]
hivm.hir.pipe_barrier[<PIPE_MTE3>]
hivm.hir.store ...
hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, %14]
```

The full loop also contains startup and cleanup tokens:

```mlir
hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
...
hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID0>]
hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_MTE2>, <EVENT_ID1>]
hivm.hir.pipe_barrier[<PIPE_ALL>]
```

After `convert-hivm-to-std`, the structured DMA becomes:

```mlir
func.call @load_gm_to_ubuf_1d_float(%cast, %cast_1, %c2_i32, %cst, %c0, %c0_i32)

func.call @store_ubuf_to_gm_1d_float(%cast_5, %cast_6, %c0_i32)
```

This is too late for the primary bridge because rank, stride, padding, atomic,
and sync ownership have to be recovered indirectly from helper names and
operands.

## Current `dma_copy_kernel` Row

Observed DMA shape:

- source: dynamic GM `memref<?xf32, #hivm.address_space<gm>>`
- destination: dynamic GM `memref<?xf32, #hivm.address_space<gm>>`
- local buffer: UB `memref<256xf32, #hivm.address_space<ub>>`
- double buffer: two 1024-byte UB pointer casts
- valid element count: dynamic, up to 256 `f32` elements
- transfer layout: rank-1, contiguous stride 1
- load pipe: `PIPE_MTE2`
- store pipe: `PIPE_MTE3`
- vector/tail-fill pipe: `PIPE_V`

The current kernel includes a tail-fill path when the valid count is smaller
than 256. For a pure copy, the store writes only the valid subview, so the fill
is probably not externally observable. Do not generalize that to kernels with
vector compute between load and store: there, the tail fill can become part of
the value semantics.

## NPU-IR Source Anchors

Structured op definitions:

- `$HOME/AscendNPU-IR/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td`
  defines `hivm.hir.load` as `PIPE_MTE2` and `hivm.hir.store` as `PIPE_MTE3`.

Library-call name selection:

- `$HOME/AscendNPU-IR/bishengir/lib/Dialect/HIVM/IR/LibraryFunctionOpInterface/LibraryFunctionOpInterfaceImpl.cpp`
  maps address spaces to library names: `GM -> gm`, `UB -> ubuf`,
  `L1 -> cbuf`.
- The copy-like library name builder forms names like
  `load_gm_to_ubuf_1d_float` and `store_ubuf_to_gm_1d_float`.

Structured-DMA to helper-call conversion:

- `$HOME/AscendNPU-IR/bishengir/lib/Conversion/HIVMToStandard/regbase/HIVMToStandard.cpp`
  registers `CopyOpToLibraryCallPattern<hivm::LoadOp>` and
  `CopyOpToLibraryCallPattern<hivm::StoreOp>`.
- The same converter marks `hivm::LoadOp` and `hivm::StoreOp` illegal so they
  must be rewritten before the pass completes.
- Load operands add pad mode/value, left padding, and eviction policy.
- Store operands add atomic kind.

Template implementation evidence:

- Current regbase vector templates:
  `$HOME/AscendNPU-IR/bishengir/lib/Template/lib/RegBase/Vector/Copy1D.cpp`
  and `$HOME/AscendNPU-IR/bishengir/lib/Template/include/RegBase/DMAUtils.h`.
- Non-regbase/older DMA templates:
  `$HOME/AscendNPU-IR/bishengir/lib/Template/lib/DMA/Ubuf/Copy1D.cpp`
  and `$HOME/AscendNPU-IR/bishengir/lib/Template/include/DMA/DMAUtils.h`.

For the regbase C310/A5 path, contiguous 1D GM->UB eventually calls
`load_gm_to_ubuf_intrin_core`, which emits `copy_gm_to_ubuf_align_v2`.
Contiguous 1D UB->GM calls `store_ubuf_to_gm_intrin_core`, which emits
`copy_ubuf_to_gm_align_v2`. Non-contiguous or unaligned variants may route
through 2D promotion or scalar fallback; those are not part of the first
implementation slice.

## PTOAS Mapping Target Evidence

Low-level VPTO/MTE surface:

- `$HOME/PTOAS/PTOAS_Markham/include/PTO/IR/VPTOOps.td` defines
  `pto.mte_gm_ub`, `pto.mte_ub_gm`, `pto.copy_gm_to_ubuf`, and
  `pto.copy_ubuf_to_gm`.
- `$HOME/PTOAS/PTOAS_Markham/lib/PTO/Transforms/VPTOExpandWrapperOps.cpp`
  expands `pto.mte_gm_ub` and `pto.mte_ub_gm` to loop-register setup plus
  `pto.copy_gm_to_ubuf` / `pto.copy_ubuf_to_gm`.
- `$HOME/PTOAS/PTOAS_Markham/lib/PTO/Transforms/VPTOCANN900LLVMEmitter.cpp`
  lowers the copy ops to `llvm.hivm.MOV.OUT.TO.UB.ALIGN.V2.*.DV` and
  `llvm.hivm.MOV.UB.TO.OUT.ALIGN.V2.DV`.
- `$HOME/PTOAS/PTOAS_Markham/include/PTO/IR/PTOOps.td` defines
  `pto.set_flag`, `pto.wait_flag`, `pto.set_flag_dyn`, and
  `pto.wait_flag_dyn`.

PTODSL/TileLib surface:

- `$HOME/PTOAS/PTOAS_Markham/lib/TileOps/a5/tload.py` uses `pto.mte_load`.
- `$HOME/PTOAS/PTOAS_Markham/lib/TileOps/a5/tstore.py` uses `pto.mte_store`.

PTO-ISA tile surface:

- `$HOME/pto-isa/docs/isa/TLOAD.md`
- `$HOME/pto-isa/docs/isa/TSTORE.md`
- `$HOME/pto-isa/docs/isa/TSYNC.md`

For the first code patch, the most concrete PTOAS target is low-level VPTO MTE,
not high-level tile `TLOAD`/`TSTORE`. Tile-level mapping remains the better
long-term interface if we can preserve enough tile shape/layout intent.

## First Mapping Table

| NPU-IR source | PTOAS low-level target | Operand mapping for first slice | Notes |
|---|---|---|---|
| `hivm.hir.load` from GM to UB | `pto.mte_gm_ub` | source GM ptr, destination UB ptr, `l2_cache_ctl`, `len_burst = valid_elems * elem_bytes`, `nburst(1, len_burst, len_burst)` for contiguous 1D | If UB row stride must remain tile-size bytes rather than valid bytes, record that explicitly before emitting. |
| `hivm.hir.store` from UB to GM | `pto.mte_ub_gm` | source UB ptr, destination GM ptr, `len_burst = valid_elems * elem_bytes`, `nburst(1, len_burst, len_burst)` for contiguous 1D | Atomic store is out of scope for the first slice. |
| `hivm.hir.set_flag` / `wait_flag` | `pto.set_flag` / `pto.wait_flag` | static event attrs map directly if pipe names and event ids match | Preserve order exactly. Do not combine with PTOAS auto-sync in this first path. |
| dynamic event `%14` flag/wait | `pto.set_flag_dyn` / `pto.wait_flag_dyn` | dynamic event id value maps to dynamic sync op | Needs a minimal PTOAS lit test before relying on it in generated IR. |
| tail fill outlined vector function | likely VMI/vector fill, PTO fill, or preserve as existing function | not part of simple DMA row | For pure copy it may be unobservable; for compute kernels it must be preserved. |

## First Code Patch Recommendation

Implement a dry-run bridge/export pass in `$HOME/AscendNPU-IR`, placed after
`hivm-mark-disable-load` and before `convert-hivm-to-std`.

The pass should initially:

1. Walk functions after memory/sync planning.
2. Match only rank-1 contiguous `hivm.hir.load` GM->UB and rank-1 contiguous
   `hivm.hir.store` UB->GM.
3. Accept only no-atomic store and no nonzero left padding.
4. Record zero pad mode separately rather than treating every `PadValue` as an
   unsupported case.
5. Record surrounding sync ops in program order.
6. Emit a stable mapping record, not a destructive lowering, with explicit
   rejection reasons.

Suggested first record fields:

```text
function
op_name
direction
source_address_space
destination_address_space
element_type
rank
shape_or_dynamic_extent
source_stride
destination_stride
ub_allocation_or_pointer_cast
multi_buffer_factor
pad_mode
pad_value
left_padding
atomic_kind
sync_before
sync_after
ptoas_candidate_op
status
rejection_reason
```

After the export is stable, generate one PTOAS-facing lit test that contains
`pto.mte_gm_ub`, `pto.mte_ub_gm`, and explicit sync. Only then replace an
existing NPU-IR lowering path.

## Rejected First Steps

- Do not start from `convert-hivmave-to-ave-intrin`: structured DMA is already
  gone.
- Do not parse final `load_gm_to_ubuf_1d_float` names as the primary source:
  useful for checking current behavior, too lossy for the bridge.
- Do not start inside CCE templates: they are reference evidence for legality,
  alignment, and fallback, but keeping that as the implementation boundary keeps
  the bridge coupled to the template backend.
- Do not use PTOAS auto-sync while preserving NPU-IR sync in the same row.
