# Planning Overview

Last updated: 2026-08-12

This file is the high-level index for active bridge planning. Codex should read
this file at the start of each meaningful Planner task before choosing which
lower-level planning documents matter.

## Current Direction

The project goal is to build an open backend path from AscendNPU-IR to
PTOAS/PTO-ISA, replacing as much of the CCE-template/backend path as is
practical.

Current implementation bias:

- implement bridge work on the AscendNPU-IR side first;
- avoid changing PTOAS unless explicitly needed;
- start with DMA because `dma_copy_kernel` gives a small, source-backed example;
- find the correct NPU-IR pass boundary before writing conversion code;
- keep PTOAS/PTO-ISA as the mapping target and compatibility check.

## Planning Document Roles

| File | Role | Status |
|---|---|---|
| `bridge/planning/README.md` | high-level plan index and short/long-term roadmap | active entry point |
| `bridge/planning/dma-copy-conversion-exploration.md` | current focused investigation for `dma_copy_kernel` | active short-term work |
| `bridge/planning/dma-template-rewrite-plan.md` | broader DMA category strategy and first PoC constraints | active strategy |
| `bridge/planning/npuir-device-spec-replay.md` | how to replay A5-generated IR locally and inspect pass dumps | active support workflow |
| `bridge/planning/npuir-to-ptoas-mapping.md` | cross-domain mapping draft from NPU-IR concepts to PTOAS/PTO-ISA | active but needs updates from DMA trace |
| `bridge/planning/soyu-wilson-ave-to-vmi-branch-review.md` | review of Wilson/Soyu vector prototype branch | reference only |
| `bridge/planning/local-repo-baseline.md` | local repo inventory and initial state | reference baseline |
| `bridge/planning/initial-exploration-plan.md` | one-time exploration/backfill plan | mostly historical |

The two DMA planning docs are intentionally different:

- `dma-template-rewrite-plan.md` is the category-level strategy for all DMA and
  template families.
- `dma-copy-conversion-exploration.md` is the immediate experiment for the
  first kernel, `dma_copy_kernel`.

## Short-Term Plan

Immediate objective: implement the first dry-run DMA bridge/export pass for
`dma_copy_kernel`.

Current DMA decision: the first bridge boundary should be after
`hivm-mark-disable-load` and before `convert-hivm-to-std`. At that point
`hivm.hir.load` / `hivm.hir.store` are still structured, memory spaces are
explicit, UB double-buffering is visible, and the relevant MTE/V/MTE sync
sequence has already been emitted. The supporting trace is
`bridge/memory/dma-copy-conversion-trace.md`.

Completed exploration:

1. Regenerated and inspected the full `dma_copy_kernel` replay log with
   `--mlir-disable-threading` and `--mlir-print-ir-after-all`.
2. Traced GM->UB load and UB->GM store through the major representation changes.
3. Recorded the first appearance and last useful appearance of structured
   `hivm.hir.load` / `hivm.hir.store`.
4. Identified the pass that converts structured DMA into template/library calls
   such as `load_gm_to_ubuf_1d_float` and `store_ubuf_to_gm_1d_float`.
5. Found the NPU-IR DMA template declaration, selection logic, implementation,
   and the actual low-level instruction sequence used by those templates.
6. Inspected PTOAS/PTO-ISA movement options for the same semantics:
   `TLOAD`, `TSTORE`, VPTO MTE ops, and sync/event support.
7. Wrote the result into `bridge/memory/dma-copy-conversion-trace.md`.

Expected first code patch:

- an AscendNPU-IR dry-run analysis/export pass or mode;
- no broad rewrite yet;
- strict matching for simple GM->UB load and UB->GM store;
- clear rejection reasons for padding, atomics, layout conversion, unsupported
  rank/stride, and unclear sync ownership.
- preserve NPU-IR-authored explicit sync for the first slice; do not combine it
  with PTOAS auto-sync.

## Medium-Term Plan

After the `dma_copy_kernel` trace is complete:

1. Implement the narrow GM->UB / UB->GM bridge record in AscendNPU-IR.
2. Generate a minimal PTOAS-facing test from the bridge record or emitted IR.
3. Validate locally where possible.
4. Send the necessary A5 validation command/log request to the human.
5. Expand only after the first row is stable.

Likely second-wave DMA rows:

- GM->L1 normal load;
- GM->L1 ND2NZ;
- UB->L1 / L1->UB movement;
- sync/event mapping around DMA;
- UB->GM atomic store only after plain store is understood.

## Long-Term Plan

The broader bridge should eventually handle:

- vector-side mapping to PTO/VMI while preserving masks, predicates, and
  accumulator lifetimes;
- DMA/tile movement mapping to PTO tile abstractions or low-level VPTO movement;
- cube/matmul mapping around ND2NZ, L1/L0 staging, MMAD, and fixpipe;
- explicit sync ownership between NPU-IR and PTOAS;
- performance-preserving lowering, not just syntactic translation;
- A5 validation through the human-run server loop;
- daily explorer updates so PTOAS upstream/fork movement does not invalidate the
  bridge assumptions.

## Current Blocking Questions

- Where exactly in the current AscendNPU-IR pass pipeline should the dry-run
  bridge/export pass be inserted so it runs after `hivm-mark-disable-load`?
- What output format should the dry-run pass use first: JSON, MLIR comments, or
  a small bridge dialect/module dump?
- Should the first generated PTOAS test use low-level VPTO `pto.mte_gm_ub` /
  `pto.mte_ub_gm` only, or also include a tile-level `TLOAD`/`TSTORE`
  comparison?
- How should dynamic event ids from NPU-IR be represented in the first PTOAS
  sync test?

## Rule For Planning Updates

When a planning document becomes too specific, keep it local to the experiment.
When a fact affects the overall project direction, update this file and
`bridge/memory/project-state.md`.
