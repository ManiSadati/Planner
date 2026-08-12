# DMA Template Rewrite Plan

Last updated: 2026-08-12

Scope: first practical bridge plan for replacing the CCE template/library-call
DMA path in AscendNPU-IR with something PTOAS can understand. This is based on
local source inspection of `$HOME/AscendNPU-IR`, `$HOME/PTOAS/PTOAS_Markham`,
and `$HOME/pto-isa`.

Role: broad DMA/category strategy. For the current focused `dma_copy_kernel`
trace, use `bridge/planning/dma-copy-conversion-exploration.md`.

Active focused exploration: `bridge/planning/dma-copy-conversion-exploration.md`.
Use `dma_copy_kernel` first to locate the exact conversion sweet spot and trace
each major load/store syntax change before changing NPU-IR code.

## Short Version

Do not start by rewriting every CCE DMA template.

Start by intercepting structured HIVM DMA operations before
`convert-hivm-to-std`, record the exact movement/layout semantics, and implement
one narrow proof of concept that emits PTOAS-compatible DMA/tile movement.

Recommended first slice:

- source op: `hivm.hir.load` from GM to UB, contiguous 1D or 2D, no padding;
- second op: `hivm.hir.store` from UB to GM, contiguous, no atomic;
- bridge target: either PTOAS low-level VPTO MTE ops (`mte_gm_ub`,
  `mte_ub_gm`) or PTO tile-level `TLOAD`/`TSTORE` style ops, depending on the
  review decision below;
- boundary: before `HIVMToStandard` lowers these ops into CCE-style library
  calls.

This slice is intentionally small. It proves the bridge mechanics without
starting with ND-to-NZ layout conversion, L0C fixpipe, atomics, or cube sync.

## Why This Boundary

`HIVMToStandard` currently rewrites many structured `hivm.hir.*` operations to
library calls. That includes load, store, copy, ND2NZ, NZ2ND, L12UB, fixpipe,
load-scale, cube, and several sync-related paths.

After that conversion, the bridge mostly sees names like:

- `load_gm_to_ubuf_*`
- `store_ubuf_to_gm_*`
- `copy_gm_to_cbuf_multi_nd2nz_*`
- `nz2nd_*`
- `l12ub_*`
- fixpipe/cube template entry points

Those names are useful as reference evidence, but they are not the best
implementation boundary. The better boundary is the structured HIVM op, because
it still carries source/destination memory spaces, shape, stride, dtype, rank,
padding, layout, atomic mode, and sync context.

## What The Human Should Review

These decisions should be made before code starts:

| Review item | Choice | Why it matters |
|---|---|---|
| Primary bridge target | PTOAS tile ops / PTO-ISA style vs low-level VPTO MTE ops vs mixed | Tile ops preserve intent better. VPTO MTE maps closer to CCE intrinsics and may be easier for a first smoke test. |
| Memory ownership | NPU-IR owns placement/sync vs PTOAS owns placement/sync | If NPU-IR owns addresses and explicit sync, target a level3-like path. If PTOAS owns memory planning and auto-sync, target a level2-like path. Mixing this implicitly is risky. |
| First DMA row | GM->UB load/store vs GM->L1 ND2NZ vs UB->L1 | GM->UB is simplest. ND2NZ is more representative for cube/tile layout, but has more legality details. |
| Output format for prototype | Real PTOAS IR immediately vs neutral mapping/export file first | A neutral export helps validate semantics quickly. Real IR proves compiler integration sooner. |
| A5 validation loop | Local PTOAS compile only vs human-run A5 execution | PTOAS can be compiled locally. Full NPU-IR/A5 correctness still needs the A5 server loop. |

## DMA Categories

This table is the working classification for CCE-template DMA replacement.

| Category | NPU-IR source shape | Current CCE/template path | Likely PTOAS/PTO target | First status |
|---|---|---|---|---|
| GM->UB normal load | `hivm.hir.load`, GM source, UB destination | `load_gm_to_ubuf_*`, `copy_gm_to_ubuf_align_b*` | Preferred high-level: `TLOAD` to Vec tile. Low-level fallback: `pto.mte_gm_ub`. | Good first PoC. |
| UB->GM normal store | `hivm.hir.store`, UB source, GM destination | `store_ubuf_to_gm_*`, `copy_ubuf_to_gm_align_b*` | Preferred high-level: `TSTORE` from Vec tile. Low-level fallback: `pto.mte_ub_gm`. | Good first PoC. |
| UB->GM atomic store | `hivm.hir.store` with atomic add/max/min | atomic setup around UB->GM store template | `TSTORE` atomic if legal, or low-level VPTO store sequence if available. | Defer until plain store works. |
| UB->UB copy | `hivm.hir.copy`, UB to UB | `copy_ubuf_to_ubuf`, plus vector/scalar fallback for tails | `TMOV` Vec->Vec or `pto.mte_ub_ub`. Tail fallback may need VMI/VPTO vector ops. | Medium risk. |
| UB->L1 copy | `hivm.hir.copy`, UB to CBUF/L1 | `copy_ubuf_to_cbuf`, MTE UB->L1 path | `TMOV` Vec->Mat or `pto.mte_ub_l1`. | Good second wave. |
| GM->L1 normal load | `hivm.hir.load`, GM to CBUF/L1 | `copy_gm_to_cbuf` | `TLOAD` to Mat tile or `pto.mte_gm_l1`. | Good second wave. |
| GM->L1 ND2NZ | `hivm.hir.nd2nz`, GM ND source to L1 NZ target | `copy_gm_to_cbuf_multi_nd2nz_b8/b16/b32s` | `TLOAD` Mat ND->NZ if using tile level, or low-level `copy_gm_to_cbuf_multi_nd2nz`. | Important, but not first if we want a fast PoC. |
| L1->GM NZ2ND | `hivm.hir.nz2nd`, CBUF/L1 NZ source to GM ND target | `nz2nd_4d_to_2d_*`, `nz2nd_5d_to_3d_*` | Possible route: L1/Mat -> UB/Vec `TMOV`, then `TSTORE`; direct PTOAS low-level support must be verified. | High risk/unknown. |
| L1->UB with layout conversion | `hivm.hir.l12ub`, CBUF/L1 to UB | `l12ub_4d_to_2d_*`, `l12ub_5d_to_3d_*` | `TMOV` Mat->Vec or `pto.mte_l1_ub`. | Important second wave. |
| L1->L0A/L0B cube operand load | cube/template staging around `hivm.hir.mmadL1` | cube MTE/template staging | `TMOV` Mat->Left/Right or low-level `mte_l1_l0a/b`. | Handle with cube rewrite, not as first DMA-only PoC. |
| L0C fixpipe to GM/L1/UB | `hivm.hir.fixpipe` | fixpipe templates with relu/quant/layout/dual-destination modes | `TSTORE` from Acc, `TSTORE_FP`, `TMOV` Acc->Vec/Mat, or low-level `mte_l0c_*`. | High risk; needs separate design. |
| MX scale load | `hivm.hir.load_scale`, GM to L1 scale layout | `load_scale_gm_to_cbuf_*` | PTOAS/PTO-ISA MX scale load path if legal; exact target needs verification. | Defer. |
| L1 fill/init | SET2D-like template paths | `set_l1_2d` style initialization | PTO tile fill/pad op if available, otherwise low-level fill sequence. | Defer. |
| Indirect/stride/gather/scatter memory | HIVM/SIMT or HIVMAVE indexed movement | intrinsic/template-specific lowering | VMI gather/scatter for vector rows; PTO-ISA `MGATHER`/`MSCATTER` for tile/GM indexed movement. | Separate from core DMA. |

## Mapping Rules By Category

Use these rules when filling the implementation mapping table.

| NPU-IR memory movement | If bridge chooses tile-level target | If bridge chooses low-level VPTO target | Notes |
|---|---|---|---|
| GM -> UB | `TLOAD` into Vec tile | `mte_gm_ub` | Preserve shape, stride, dtype, valid region, pad mode/value. |
| UB -> GM | `TSTORE` from Vec tile | `mte_ub_gm` | Split atomic and non-atomic rows. |
| GM -> L1 | `TLOAD` into Mat tile | `mte_gm_l1` or `copy_gm_to_cbuf` | Layout role matters: Mat/Bias/Scaling may not be interchangeable. |
| GM ND -> L1 NZ | `TLOAD` with ND->NZ Mat semantics | `copy_gm_to_cbuf_multi_nd2nz` | Preserve `n0`, `d0`, rank, bias variant, and continuous-destination requirement. |
| UB -> UB | `TMOV` Vec->Vec | `mte_ub_ub` or `copy_ubuf_to_ubuf` | Unaligned tails may require extra vector/scalar fallback. |
| UB -> L1 | `TMOV` Vec->Mat | `mte_ub_l1` or `copy_ubuf_to_cbuf` | Useful bridge toward cube operands. |
| L1 -> UB | `TMOV` Mat->Vec | `mte_l1_ub` or `copy_cbuf_to_ubuf` | Layout conversion details must be explicit. |
| L1/NZ -> GM/ND | likely two-step `TMOV` then `TSTORE`, unless direct support exists | unknown/direct support not confirmed | Do not implement from assumption. |
| L0C -> GM | `TSTORE` from Acc, maybe `TSTORE_FP` | `mte_l0c_gm` | Quant/relu/dtype modes need separate tests. |
| L0C -> L1/UB | `TMOV` Acc->Mat/Vec | `mte_l0c_l1` or `mte_l0c_ub` | Fixpipe dual destination is a separate risk. |

## First Implementation Step

The first code step should be a small experimental pass or export mode in
`$HOME/AscendNPU-IR`, before `HIVMToStandard`, with strict matching:

1. Match only simple `hivm.hir.load` GM->UB and `hivm.hir.store` UB->GM.
2. Reject padding, atomics, layout conversion, non-contiguous edge cases, and
   unsupported ranks with a clear diagnostic.
3. Emit a stable mapping record with:
   - source op name;
   - source/destination address spaces;
   - dtype;
   - rank/shape;
   - strides;
   - padding/atomic/layout flags;
   - selected PTOAS target;
   - rejection reason if unsupported.
4. Use that record to generate or hand-check a minimal PTOAS/VPTO test.
5. Only after the first row compiles should the bridge replace a real lowering
   path or emit real PTOAS IR.

This keeps the first implementation reversible and measurable.

## What Not To Do First

- Do not continue the Soyu-Wilson AVE-to-VMI branch as the DMA starting point.
  It is useful vector context, but it does not solve DMA/cube/sync.
- Do not parse every CCE template as the primary source of truth. The templates
  are evidence for legality and fallback behavior; the structured HIVM ops are
  the cleaner bridge source.
- Do not map every DMA to VMI. VMI is a logical vector layer. ND2NZ, L1/L0
  staging, cube operand movement, and fixpipe are tile/DMA/cube rows.
- Do not mix NPU-IR-authored sync with PTOAS auto-sync without an explicit row
  decision.

## Immediate Evidence Needed

Before starting the first code patch, collect one or two real IR examples:

- an IR dump before `convert-hivm-to-std` containing `hivm.hir.load` GM->UB;
- an IR dump before `convert-hivm-to-std` containing `hivm.hir.store` UB->GM;
- preferably one `hivm.hir.nd2nz` example for the second wave;
- the corresponding post-`HIVMToStandard` library-call names for comparison.

The full Python/Triton-to-NPU-IR lowering path should be run on an A5 machine,
not on the Codex-accessible server. If these real examples are not already in
Planner, Codex should ask the human to generate early MLIR / early NPU-IR dumps
on the A5 server and place them in a Planner-accessible examples folder.

The goal is not to exhaustively test the whole compiler yet. The goal is to
make sure the first bridge row matches real NPU-IR, not just operation
definitions.

## Source Anchors

- NPU-IR DMA op definitions:
  `$HOME/AscendNPU-IR/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td`
- NPU-IR CCE DMA templates:
  `$HOME/AscendNPU-IR/bishengir/lib/Template/include/DMA/`
  and `$HOME/AscendNPU-IR/bishengir/lib/Template/lib/DMA/`
- NPU-IR library-call conversion:
  `$HOME/AscendNPU-IR/bishengir/lib/Dialect/HIVM/IR/LibraryFunctionOpInterface/LibraryFunctionOpInterfaceImpl.cpp`
  and
  `$HOME/AscendNPU-IR/bishengir/lib/Conversion/HIVMToStandard/regbase/HIVMToStandard.cpp`
- PTOAS low-level VPTO/MTE ops:
  `$HOME/PTOAS/PTOAS_Markham/include/PTO/IR/VPTOOps.td`
- PTOAS VMI ops:
  `$HOME/PTOAS/PTOAS_Markham/include/PTO/IR/VMIOps.td`
- PTO-ISA movement docs:
  `$HOME/pto-isa/docs/isa/TLOAD.md`,
  `$HOME/pto-isa/docs/isa/TSTORE.md`,
  `$HOME/pto-isa/docs/isa/TMOV.md`,
  `$HOME/pto-isa/docs/isa/TSYNC.md`
