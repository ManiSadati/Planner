# Planning Overview

Last updated: 2026-08-28

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
- treat vector conversion as a completed project milestone for the current
  stage; add new vector instructions incrementally when future kernels expose
  a missing operation;
- make Cube plus its required DMA/staging path the active investigation;
- compare each NPU-IR CCE Cube/DMA template contract against PTOAS/PTO dialect
  operations before choosing an implementation boundary;
- keep two explicit Cube routes available: a PTO-native rewrite and a
  CCE-template compatibility route;
- make preservation of external CCE calls and their memref descriptor ABI the
  active experiment before generalizing the PTO-native rewrite;
- keep PTOAS/PTO-ISA as the mapping target and compatibility check.

Completed vector-bridge milestone:

- `lowered_vector_add_kernel.mlir` and the Planner `row_softmax` testcase lower
  through `convert-hivmave-to-ptoas-vmi` and unmodified PTOAS to VPTO;
- the complete 64-by-256 row-softmax fixture passes PTOAS simulator numerical
  comparison;
- the Planner row-softmax and RMSNorm kernels are supported, with accepted
  performance on par with the NPU-IR path for the tested fixtures;
- this gives good practical coverage of the vector instruction path. Remaining
  signed min/max, `-inf`, partial-mask, shape, and datatype limits stay recorded
  as maintenance debt rather than the active project milestone.

Current Cube milestone:

- first fixture: `bridge/triton-example/cube_dotproduct.py`;
- the first source trace, mapping decision, and strict compiler-side conversion
  slice are complete;
- the observed path is two GM-to-L1 ND2NZ loads, one `hivm.hir.mmadL1`, and one
  NZ2ND f32-to-f16 fixpipe store;
- existing PTO primitives cover the fixture, but `mmadL1` is not equivalent to
  one bare `pto.tmatmul` because its CCE template also owns L1-to-L0 staging, K
  partitioning, double buffering, sync, and barriers;
- the first implemented route is a strict low-level PTO composition in
  `convert-hivm-templates-to-pto`, preserving NPU-IR memory and sync ownership;
- that direct route is implemented for the exact 64x64 f16/f16/f32/f16,
  init=true, no-transpose, NZ2ND fixture and remains guarded by the existing
  default-off bridge switch;
- the real fixture now emits PTOAS VMI, lowers through unmodified PTOAS to
  VPTO, and passes the PTOAS simulator numerical comparison for all 4096 f16
  outputs;
- `matmul_64` is now the active experiment: retain the selected CCE template
  calls and their memrefs through VMI and VPTO, use standard memref-to-LLVM
  lowering to recover the original descriptor ABI, and link the matching CCE
  template object or bitcode into the PTOAS device result;
- longer-term route: `pto.tload -> pto.textract -> pto.tmatmul -> pto.tstore`
  with PTOAS owning tile allocation and sync.

## Two Cube Paths

| Path | Contract | Why keep it | Main risks |
| --- | --- | --- | --- |
| External CCE calls, active experiment | Skip the early Cube rewrite. Preserve `nd2nz_half`, `mma_tile_half_to_float`, and fixpipe calls with their ranked memref operands through PTOAS. Let standard MLIR lowering create the C-interface descriptors, then link the matching CCE device implementation. | Reuses the broad, mature NPU-IR template behavior and tests the bridge without first recreating every layout, stride, K-loop, precision, and sync case. It is the fastest way to test general Cube compatibility. | PTOAS passes may reject or accidentally erase memref descriptors; address spaces and descriptor ABI must match NPU-IR exactly; CANN/PTOAS device objects must be linked with compatible tools; MIX modules need separate AIC descriptor and AIV pointer policies; this route retains a CCE dependency and is not the final fully open backend. |
| Direct PTO rewrite, validated narrow alternative | Intercept structured `nd2nz -> mmadL1 -> fixpipe` before CCE calls and emit PTO MTE/MAD operations directly. | Removes the CCE-template dependency and exposes the complete Cube path to PTOAS optimization and scheduling. The strict 64x64 fixture already passes simulation. | Generalization requires faithfully rebuilding template loops, layouts, large-stride fallbacks, K partitioning, accumulation, precision modes, double buffering, and sync. A superficially similar PTO sequence can be numerically or physically wrong. |

The external-call path is being explored first because it preserves behavior
already implemented and validated by NPU-IR. It will establish whether PTOAS
can carry the existing descriptor ABI and link the CCE device code. The direct
rewrite remains valuable, but broadening it before this compatibility test
would require reimplementing many template variants without first knowing
whether that work is necessary.

## Planning Document Roles

| File | Role | Status |
|---|---|---|
| `bridge/planning/README.md` | high-level plan index and short/long-term roadmap | active entry point |
| `bridge/planning/cube-conversion-exploration.md` | staged Cube/template/DMA mapping plan using `cube_dotproduct.py` | active short-term plan |
| `bridge/memory/cube-conversion-status.md` | compact current Cube decisions, conversion point, and review gate | active memory |
| `bridge/planning/ave-to-ptoas-vmi-implementation-plan.md` | completed vector-add, row-softmax, and RMSNorm-era vector contract | maintenance/reference |
| `bridge/designs/ave-to-ptoas-vmi-conversion-design.md` | durable AVE/HIVM-to-PTOAS VMI implementation decisions and non-direct mapping index | active design log |
| `bridge/planning/dma-copy-conversion-exploration.md` | completed focused investigation for `dma_copy_kernel` | reference foundation |
| `bridge/planning/dma-template-rewrite-plan.md` | broader DMA category strategy, including Cube staging rows | supporting strategy |
| `NPUIR/coding-guide/device-spec-replay.md` | how to replay A5-generated IR locally and inspect pass dumps | active support workflow |
| `bridge/planning/npuir-to-ptoas-mapping.md` | cross-domain mapping draft from NPU-IR concepts to PTOAS/PTO-ISA | active; next update comes from Cube trace |
| `bridge/planning/soyu-wilson-ave-to-vmi-branch-review.md` | review of Wilson/Soyu vector prototype branch | reference only |
| `bridge/planning/local-repo-baseline.md` | local repo inventory and initial state | reference baseline |
| `bridge/planning/initial-exploration-plan.md` | one-time exploration/backfill plan | mostly historical |

The two DMA planning docs are intentionally different:

- `dma-template-rewrite-plan.md` is the category-level strategy for all DMA and
  template families.
- `dma-copy-conversion-exploration.md` records the completed first focused DMA
  experiment, `dma_copy_kernel`.

## Short-Term Plan

Immediate objective: broaden the now-passing `matmul_64` external-call result
without changing the passing direct rewrite.

Review order:

1. Preserve the verified ABI invariant: public kernel entries use PTO pointers,
   while external Cube calls use standard memref descriptors.
2. Run a genuine split MIX fixture before calling the per-function type policy
   general; AIC external calls keep descriptors and AIV VMI stays pointer-based.
3. Exercise another shape/template branch, preferably one with dynamic or
   nontrivial strides.
4. Compare external-call, direct-PTO, and unchanged NPU-IR results and traces
   under identical simulator options.
5. Validate the linked external-call fat object on real A5 hardware.
6. Decide which template families should remain external and which justify a
   direct PTO rewrite.

The detailed plan is `bridge/planning/cube-conversion-exploration.md`.

## Completed Foundations

The comparison harness and first vector/DMA bridge pieces provide the reusable
validation foundation for Cube work.

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
- the NPU-IR-to-PTOAS `row_softmax` bridge fixture has run successfully through
  VMI, VPTO, and PTOAS simulator numerical comparison;
- the strict 64x64 `cube_dotproduct` bridge fixture passes the PTOAS simulator
  numerical comparison with maximum absolute error `0.001953125`;
- accepted RMSNorm fixtures are also supported and provide the second complex
  vector-kernel milestone;
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

Existing first DMA code patch:

- an AscendNPU-IR conversion pass, available as
  `--convert-hivm-templates-to-pto`;
- strict matching for contiguous rank-one GM->UB load and UB->GM store;
- dynamic byte counts derived from the runtime memref dimension;
- explicit casts from HIVM memory spaces to PTO memory spaces;
- `PadValue` support for loads and rejection of unsupported atomic/layout cases;
- preserve NPU-IR-authored explicit sync for the first slice; do not combine it
  with PTOAS auto-sync.

## Medium-Term Plan

After the external-call experiment is functionally validated:

1. Decide whether external calls are an acceptable transitional backend or
   only a compatibility/reference route.
2. Measure how much fixture and mode coverage the external path inherits from
   the CCE templates.
3. Continue the direct PTO route where removing the CCE dependency or exposing
   optimization opportunities justifies the implementation cost.
4. Generalize direct mappings one observed contract at a time while retaining
   the external path and unchanged NPU-IR as comparison baselines.
5. Validate IR legality and PTOAS lowering locally, then compare simulator
   output and request A5 runtime/performance validation from the human.

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

## Current Open Questions

- Can PTOAS preserve NPU-IR ranked memref descriptors through VMI, VPTO, and
  pre-CCE LLVM without a custom descriptor ABI?
- Where should the matching CCE template object enter PTOAS device linking?
- Can a split MIX kernel keep descriptor-carrying AIC calls and pointer-based
  AIV VMI operations in the same compilation flow?
- Does its preserved event/address plan remain correct on A5 hardware?
- Which next `mmadL1` mode should drive generalization: K partitioning,
  accumulation, transpose, or bias/precision behavior?
- How should the bridge preserve full IEEE behavior for the row-softmax
  negative-infinity initializer on the current PTOAS path?
- How should signed integer scalar min/max be generalized beyond constant
  scalars and full masks?

## Rule For Planning Updates

When a planning document becomes too specific, keep it local to the experiment.
When a fact affects the overall project direction, update this file and
`bridge/memory/project-state.md`.
