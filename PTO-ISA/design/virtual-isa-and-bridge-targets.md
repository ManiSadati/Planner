# PTO-ISA Virtual ISA And Bridge Targets

Last updated: 2026-08-10

Scope: local Stage 4 scan of `$HOME/pto-isa` on branch `master` at
`896d8ec69aaf5b623fead5afcae7a657fa784a2b`. This pass did not fetch
`git@gitcode.com:cann/pto-isa.git`. The local PTO-ISA worktree already has a
modified `include/pto/npu/a5/TBinOp.hpp`; this note does not modify or depend
on that dirty file.

## Short Version

PTO-ISA is a tile-level virtual ISA with explicit valid-region, tile-location,
layout, GlobalTensor, event, and backend-legality concepts. It is higher level
than raw CCE intrinsics, but lower and more hardware-shaped than generic MLIR
vector IR.

The bridge should treat PTO-ISA as a useful semantic target for tile, DMA,
matrix/cube, and explicit-sync rows:

- vector rows may map either to PTOAS VMI or PTO tile-vector ops depending on
  the chosen PTOAS entry point;
- GM-to-tile movement rows can often be described by `TLOAD`/`TSTORE`;
- ND-to-NZ and layout movement rows likely need `TLOAD` Mat paths or `TMOV`
  layout paths, not a VMI-only abstraction;
- cube rows should consider `TMATMUL`, `TMATMUL_ACC`, `TMATMUL_BIAS`, and
  `TMATMUL_MX`;
- explicit sync rows need careful mapping to PTO events, `TSYNC`, `record_event`
  / `wait_event`, and possibly `SYNCALL`.

## Architecture Contract

The PTO manual describes the virtual ISA goals as a stable cross-generation
contract with tile-centric semantics, explicit valid-region behavior, a boundary
between architecture-defined and implementation-defined behavior, and a bridge
from intrinsics to IR/backend codegen.

Key sources:

- `$HOME/pto-isa/docs/mkdocs/src/manual/01-overview.md:5`
- `$HOME/pto-isa/docs/mkdocs/src/manual/01-overview.md:14`
- `$HOME/pto-isa/docs/mkdocs/src/manual/01-overview.md:22`
- `$HOME/pto-isa/docs/mkdocs/src/manual/01-overview.md:38`

PTO uses a three-layer contract:

```text
Virtual ISA layer
  -> AS structured representation for verification/transforms
  -> backend lowering and target-specific legalization/codegen
```

The AS model is expected to carry operation schema, attributes, effects,
explicit synchronization, and memory effects. Lowering must preserve
valid-region semantics, explicit ordering dependencies, and operation meaning.

Key source:

- `$HOME/pto-isa/docs/mkdocs/src/manual/08-virtual-isa-and-ir.md:8`
- `$HOME/pto-isa/docs/mkdocs/src/manual/08-virtual-isa-and-ir.md:18`
- `$HOME/pto-isa/docs/mkdocs/src/manual/08-virtual-isa-and-ir.md:40`

## Type And State Model

PTO architectural state includes tile values/metadata, scalar/immediate values,
global memory views, and synchronization/event state. Tile legality is shaped by
dtype, shape, valid-region compatibility, location role, layout, and alignment.

Key source:

- `$HOME/pto-isa/docs/mkdocs/src/manual/03-state-and-types.md:7`
- `$HOME/pto-isa/docs/mkdocs/src/manual/03-state-and-types.md:18`
- `$HOME/pto-isa/docs/mkdocs/src/manual/03-state-and-types.md:29`
- `$HOME/pto-isa/docs/mkdocs/src/manual/03-state-and-types.md:40`

Important public type facts:

- `TileType` includes `Vec`, `Mat`, `Left`, `Right`, `Acc`, `Bias`, `Scaling`,
  `ScaleLeft`, `ScaleRight`, and `Ctrl`.
- `Layout` includes `ND`, `DN`, `NZ`, MX layouts, and conv-oriented layouts such
  as `NC1HWC0`, `NCHW`, `NHWC`, and fractal forms.
- `STPhase` and `AccPhase` encode partial/final unit-flag phase concepts.
- `SyncAllMode` and `SyncCoreType` encode hard/soft and AIV/AIC/mixed barrier
  modes.

Key sources:

- `$HOME/pto-isa/include/pto/common/type.hpp:122`
- `$HOME/pto-isa/include/pto/common/type.hpp:169`
- `$HOME/pto-isa/include/pto/common/type.hpp:224`
- `$HOME/pto-isa/include/pto/common/type.hpp:267`

`GlobalTensor` carries a 5-D shape/stride contract plus layout. `Tile` carries
location, dtype, static rows/cols, row/col stride, valid row/col, base layout,
boxed/fractal layout, fractal size, padding policy, and dynamic valid-region
setters.

Key sources:

- `$HOME/pto-isa/include/pto/common/pto_tile.hpp:261`
- `$HOME/pto-isa/include/pto/common/pto_tile.hpp:272`
- `$HOME/pto-isa/include/pto/common/pto_tile.hpp:1394`
- `$HOME/pto-isa/include/pto/common/pto_tile.hpp:1428`
- `$HOME/pto-isa/include/pto/common/pto_tile.hpp:1508`
- `$HOME/pto-isa/include/pto/common/pto_tile.hpp:1597`

## Instruction Families

PTO instruction families include synchronization/resource binding, elementwise
tile ops, tile-scalar ops, reductions/expands, GM-tile memory ops, matrix/GEMV
ops, layout/data-movement transforms, and complex/irregular ops.

Key source:

- `$HOME/pto-isa/docs/mkdocs/src/manual/06-instructions.md:8`
- `$HOME/pto-isa/docs/PTOISA.md:9`

Bridge-relevant families:

- `TLOAD`: GM `GlobalTensor` to `Tile`, with AS forms `pto.tload` and DPS
  `pto.tload ins(...) outs(...)`. A5 supports Vec ND/DN/NZ matching paths and
  Mat/cube paths including ND/DN/NZ and MX layout cases.
- `TSTORE`: `Tile` to GM `GlobalTensor`, with Vec and Acc paths on A5, optional
  atomic add, quantization/scaling, and unit-flag store phases.
- `TMOV`: tile-to-tile movement and transformation. The AS design recommends
  split forms such as `tmov.m2l`, `tmov.m2r`, `tmov.m2b`, `tmov.m2s`,
  `tmov.a2v`, and `tmov.v2v`.
- `TMATMUL*`: matrix/cube family over `Left`, `Right`, and `Acc` tiles, with
  valid-region-derived `m/k/n`, accumulator input variants, bias, MX scale
  variants, and `AccPhase`.
- `MGATHER`/`MSCATTER`: indexed GM/tile movement with row/element coalesce,
  out-of-bounds policies, and A5 SIMT behavior.
- `TSYNC`/events/`SYNCALL`: explicit pipe/event ordering and cross-core
  barriers.

Key sources:

- `$HOME/pto-isa/docs/isa/TLOAD.md:8`
- `$HOME/pto-isa/docs/isa/TSTORE.md:8`
- `$HOME/pto-isa/docs/isa/TMOV.md:8`
- `$HOME/pto-isa/docs/isa/TMATMUL.md:8`
- `$HOME/pto-isa/docs/isa/MGATHER.md:1`
- `$HOME/pto-isa/docs/isa/MSCATTER.md:1`
- `$HOME/pto-isa/docs/isa/TSYNC.md:8`
- `$HOME/pto-isa/docs/isa/SYNCALL.md:1`

## A5 Implementation Anchors

The public API wrappers in `include/pto/common/pto_instr.hpp` call
`TSYNC(events...)` before issuing most instructions, then dispatch through
`*_IMPL` functions. This means event ownership is part of the call shape, not
only an external scheduling concern.

Key sources:

- `$HOME/pto-isa/include/pto/common/pto_instr.hpp:47`
- `$HOME/pto-isa/include/pto/common/pto_instr.hpp:256`
- `$HOME/pto-isa/include/pto/common/pto_instr.hpp:351`
- `$HOME/pto-isa/include/pto/common/pto_instr.hpp:658`
- `$HOME/pto-isa/include/pto/common/pto_instr.hpp:1246`

A5 event implementation maps PTO `Op` values to hardware pipes such as MTE2,
MTE3, V, FIX, M, and MTE1. `Event<SrcOp,DstOp>` emits `set_flag`/`wait_flag`
for ordinary cross-pipe dependencies and `set_intra_block`/`wait_intra_block`
for selected cross-core cases.

Key sources:

- `$HOME/pto-isa/include/pto/common/event.hpp:21`
- `$HOME/pto-isa/include/pto/common/event.hpp:115`
- `$HOME/pto-isa/include/pto/common/event.hpp:259`
- `$HOME/pto-isa/include/pto/npu/a5/TSync.hpp:17`
- `$HOME/pto-isa/include/pto/npu/a5/TSync.hpp:38`
- `$HOME/pto-isa/include/pto/npu/a5/TSync.hpp:73`
- `$HOME/pto-isa/include/pto/npu/a5/TSync.hpp:98`

A5 implementation details:

- `TLOAD_IMPL` dispatches between normal tile and ConvTile paths. Normal tile
  load supports Vec and Mat cases; Mat paths call cube/MX checks and load
  routines. ConvTile paths include NC1HWC0/FRACTAL_Z/NHWC/NCHW/NCDHW layout
  conversions.
- `TSTORE_IMPL` accepts Vec or Acc sources on A5. Acc paths carry ND/NZ/conv
  layout restrictions, quantization/scaling variants, atomic add, and
  unit-flag store phase.
- `TMATMUL_IMPL` validates Left/Right/Acc dtype/layout roles, derives `m/k/n`
  from valid regions, and calls `mad`; MX variants call `mad_mx`.
- `TMOV_IMPL` handles Mat-to-Left/Right/Bias/Scaling/ScaleLeft/ScaleRight,
  Acc-to-Vec/Mat, Vec-to-Vec/Mat, and Vec ND-to-NZ movement.
- `MGATHER` and `MSCATTER` enforce Vec tile/index contracts for UB paths; A5
  also has a GM-to-L1 NZ Mat gather path.

Key sources:

- `$HOME/pto-isa/include/pto/npu/a5/TLoad.hpp:302`
- `$HOME/pto-isa/include/pto/npu/a5/TLoad.hpp:599`
- `$HOME/pto-isa/include/pto/npu/a5/TLoad.hpp:651`
- `$HOME/pto-isa/include/pto/npu/a5/TStore.hpp:115`
- `$HOME/pto-isa/include/pto/npu/a5/TStore.hpp:182`
- `$HOME/pto-isa/include/pto/npu/a5/TMatmul.hpp:28`
- `$HOME/pto-isa/include/pto/npu/a5/TMatmul.hpp:130`
- `$HOME/pto-isa/include/pto/npu/a5/TMatmul.hpp:159`
- `$HOME/pto-isa/include/pto/npu/a5/TMatmul.hpp:257`
- `$HOME/pto-isa/include/pto/npu/a5/TMov.hpp:451`
- `$HOME/pto-isa/include/pto/npu/a5/TMov.hpp:553`
- `$HOME/pto-isa/include/pto/npu/a5/TMov.hpp:622`
- `$HOME/pto-isa/include/pto/npu/a5/MGather.hpp:425`
- `$HOME/pto-isa/include/pto/npu/a5/MGather.hpp:452`
- `$HOME/pto-isa/include/pto/npu/a5/MScatter.hpp:391`
- `$HOME/pto-isa/include/pto/npu/a5/MScatter.hpp:456`

## Bridge Implications

- `hivm.hir.mmadL1` should first be compared against PTO `TMATMUL`,
  `TMATMUL_ACC`, `TMATMUL_BIAS`, and `TMATMUL_MX`, not VMI. Preserve
  `m/k/n`, accumulator init/source, bias, MX scale, dtype, tile roles, layouts,
  and unit-flag phase.
- `hivm.hir.nd2nz` should first be compared against `TLOAD` Mat ND-to-NZ style
  paths or `TMOV` ND-to-NZ movement, depending whether the source is GM or tile
  storage at the bridge point.
- `hivm.hir.load`/`store` can potentially map to `TLOAD`/`TSTORE`, but the row
  must preserve GlobalTensor shape/stride/layout and the tile valid region.
- `ave.hir.vf*`, `ave.hir.pge`, and vector load/store rows may still prefer
  PTOAS VMI when the target is VPTO/VMI. If the target is PTO-ISA directly,
  they need `TileType::Vec`, valid-region, predicate/mask, and event mapping.
- NPU-IR `set_flag`/`wait_flag` rows cannot be mechanically replaced with
  `TSYNC` unless the source/destination pipe pair, token lifetime, and event ID
  semantics match PTO `Event<SrcOp,DstOp>`.
- NPU-IR `sync_block*` rows may relate to `SYNCALL` only when the participant
  set and cross-core visibility semantics match. Otherwise they may need an
  explicit lower-level sync representation or remain NPU-IR-owned.

## Open Questions For Mapping Table

- Does the intended bridge emit PTO-AS/Virtual ISA directly, PTOAS tile ops, or
  a PTOAS dialect that later aligns with PTO-ISA?
- Which PTOAS path currently consumes or can preserve PTO-style `record_event`
  / `wait_event` / `barrier` semantics?
- Can NPU-IR `mmadL1` unit-flag modes map cleanly to PTO `AccPhase`/`STPhase`,
  or are there missing phases/ownership details?
- Is NPU-IR `nd2nz` always GM-to-L1 at the bridge point, or can it also appear
  as tile-to-tile layout movement?
- Which A5-only PTO-ISA constraints must be reflected in the first verifier
  instead of deferred to a backend legality failure?
