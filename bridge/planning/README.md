# Planning Overview

Last updated: 2026-08-19

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
- treat `https://gitcode.com/wilsoncxfeng/AscendNPU-IR` as the main bridge
  implementation fork, but check the most up-to-date branch each time instead
  of assuming `master`;
- prioritize the comparison structure now, because the first conversion pieces
  exist and need a clean way to compare baseline NPU-IR against the
  NPU-IR-to-PTOAS path;
- keep the DMA boundary decision as current implementation context, but do not
  let it block the comparison harness cleanup;
- keep PTOAS/PTO-ISA as the mapping target and compatibility check.

## Planning Document Roles

| File | Role | Status |
|---|---|---|
| `bridge/planning/README.md` | high-level plan index and short/long-term roadmap | active entry point |
| `bridge/planning/dma-copy-conversion-exploration.md` | current focused investigation for `dma_copy_kernel` | active short-term work |
| `bridge/planning/dma-template-rewrite-plan.md` | broader DMA category strategy and first PoC constraints | active strategy |
| `NPUIR/coding-guide/device-spec-replay.md` | how to replay A5-generated IR locally and inspect pass dumps | active support workflow |
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

Immediate objective: clean up the comparison structure so baseline NPU-IR and
the NPU-IR-to-PTOAS path can be run and compared consistently. The DMA
conversion work is the first implementation row feeding into that comparison,
but comparison harness clarity is the current priority.

Comparison requirements:

- run baseline NPU-IR and the bridge path on the same simulator target when
  using simulator numbers;
- keep one-line runnable commands for the human;
- record output directories, IR dumps, tick/cycle interpretation, and runtime
  caveats;
- ensure PTOAS host-side fixtures are behaviorally equivalent to the Triton
  host path used by NPU-IR;
- separate simulator functional/tick checks from real A5 runtime checks.

Current local simulator status:

- this server can run selected NPU-IR/Triton cases through the CANN operator
  simulator;
- `vector_add_kernel` and `vector_add_large_kernel` have both run
  successfully in simulator;
- simulator results are useful for functional checks and rough comparison, but
  hardware runtime/performance still needs the A5 server.

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
8. Implemented `convert-hivm-templates-to-pto` in AscendNPU-IR for contiguous rank-one
   GM-to-UB loads and UB-to-GM non-atomic stores, including dynamic lengths and
   `PadValue` loads.
9. Ran the pass on the real `MarkDisableLoad` dump. The result contains
   `pto.mte_gm_ub` and `pto.mte_ub_gm`, preserves the surrounding NPU-IR sync,
   and remains valid when `convert-hivm-to-std` runs afterward.

Current first code patch:

- an AscendNPU-IR conversion pass, available as
  `--convert-hivm-templates-to-pto`;
- strict matching for contiguous rank-one GM->UB load and UB->GM store;
- dynamic byte counts derived from the runtime memref dimension;
- explicit casts from HIVM memory spaces to PTO memory spaces;
- `PadValue` support for loads and rejection of unsupported atomic/layout cases;
- preserve NPU-IR-authored explicit sync for the first slice; do not combine it
  with PTOAS auto-sync.

## Medium-Term Plan

After the comparison structure is clean:

1. Re-run the baseline NPU-IR and bridge-path comparison for `vadd` and the
   large vector-add case.
2. Decide whether to enable the DMA pass through a dedicated compiler option or a
   separate PTO pipeline; do not enable it unconditionally in the CCE pipeline.
3. Convert or preserve the explicit sync operations in PTO form.
4. Run PTO wrapper expansion/lowering on the mixed NPU-IR/PTO module.
5. Send the necessary A5 validation command/log request to the human.
6. Expand only after the first row is stable.

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

- Which existing bridge scripts are truly cross-repo comparison scripts, and
  which should move under `NPUIR/` or `PTOAS/` because they are repo-specific?
- Should the pass be enabled through a dedicated compiler option or a separate
  PTO-target pipeline?
- Which PTO wrapper-expansion and pointer-lowering passes should immediately
  follow this conversion inside AscendNPU-IR?
- How should dynamic event ids from NPU-IR be represented in the first PTOAS
  sync test?

## Rule For Planning Updates

When a planning document becomes too specific, keep it local to the experiment.
When a fact affects the overall project direction, update this file and
`bridge/memory/project-state.md`.
