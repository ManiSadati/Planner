# Planning Overview

Last updated: 2026-09-02

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
- make PTODSL/TileLib-style, PTO-visible Cube expansion the intended production
  default so PTOAS can optimize and evolve the generated operations;
- keep preservation of external CCE calls and their memref descriptor ABI as a
  working compatibility/reference route, not the intended default;
- do not duplicate the PTODSL Cube body in a hand-written C++ rewrite; C++ may
  match/adapt NPU-IR operations and import pre-generated helpers whose authoring
  source is Python;
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
- the first source trace and mapping decision are complete;
- the observed path is two GM-to-L1 ND2NZ loads, one `hivm.hir.mmadL1`, and one
  NZ2ND f32-to-f16 fixpipe store;
- existing PTO primitives cover the fixture, but `mmadL1` is not equivalent to
  one bare `pto.tmatmul` because its CCE template also owns L1-to-L0 staging, K
  partitioning, double buffering, sync, and barriers;
- the earlier strict 64x64 C++ rewrite proved the low-level mapping and passed
  simulation, but has been removed to avoid maintaining a second Cube-template
  implementation;
- the external-call `matmul_64` route is complete enough to serve as a working
  compatibility and numerical reference;
- commit `9d97eff1240434e537e45ee9154c65df80208e2e` added the PTODSL Cube source.
  Its explicit MmadL1 and ND2NZ MLIR instantiations are now checked in and
  installed; `bishengir-compile` parses those files and replaces the structured
  operations with internal calls without executing Python;
- PTOAS inlines and lowers the imported body, and the 64x64 simulator fixture
  passes. The Python template is the implementation source; the C++ bridge owns
  matching, argument adaptation, and materialization rather than duplicating
  the template body.

## Cube Paths

| Path | Contract | Why keep it | Main risks |
| --- | --- | --- | --- |
| PTODSL/PTO-native expansion, preferred | Preserve the structured Cube contract before `convert-hivm-to-std`, import pre-generated PTO helpers, and call them with NPU-IR's pointers, dimensions, init state, strides, and event IDs. | The 64x64 path works and keeps Cube semantics visible to PTOAS instead of treating CCE code as a black box. | Only the observed f16 64x64x64 path is numerically proven; memory/sync ownership, tails, K segmentation, precision modes, bias, transpose, and fallbacks remain. |
| External CCE calls, compatibility/reference | Preserve `nd2nz_half`, `mma_tile_half_to_float`, and fixpipe calls with their ranked memrefs through PTOAS, then link the matching CCE implementation. | Already works and provides broad mature behavior plus a strong numerical/ABI baseline. | Cube remains opaque to PTOAS; the route retains CCE and descriptor/linker dependencies and complicates MIX packaging. |

The external path answered the compatibility question successfully. The next
work is therefore the preferred PTODSL/PTO-native path, using the external path
and unchanged NPU-IR as references.

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

Immediate objective: generalize the now-connected PTODSL Cube template without
regressing the external-call compatibility route.

Review order:

1. Preserve the passing 64x64 contract: caller-owned local buffers, M/K/N,
   init/accumulate, and NPU-IR event IDs enter the imported Python helper.
2. Add numerical tests for a partial tile and `init = false` accumulation,
   comparing each result with unchanged NPU-IR and the CCE template route.
3. Define the longer-term memory/sync ownership transition. Do not run both the NPU-IR
   physical event plan and PTOAS automatic allocation/sync for the same region.
4. Complete numerical validation for a partial tile and accumulating K
   iteration. The multi-tile `q_kt_matmul` PTODSL path already reaches VMI,
   VPTO, fat-object generation, and 32-core simulator launch; run its published
   object on A5 for practical full-shape validation.
5. Compare PTODSL, external-call, and unchanged NPU-IR output and traces under
   identical options.
6. Run a genuine split MIX fixture and request A5 hardware validation before
   making the PTODSL bridge mode operationally default.

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
- the PTODSL 64x64 `cube_dotproduct` bridge fixture passes the PTOAS simulator
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

After the first PTODSL-integrated Cube fixture passes:

1. Move K segmentation, double buffering, and init/accumulate scheduling into
   the PTO-visible template contract one observed branch at a time.
2. Add ND2NZ stride fallback, tails/padding, transpose, bias, and precision
   variants based on concrete NPU-IR fixtures.
3. Measure optimization and scheduling differences against external CCE calls;
   this is the main reason to prefer the PTODSL route.
4. Keep external calls as the compatibility/reference backend for unsupported
   variants, with explicit fallback reasons rather than silent behavior changes.
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

- Should the first compiler-integrated PTODSL route emit PTO tile operations
  for `ExpandTileOp`, or materialize and inline generated PTO bodies directly?
- Which layer owns local-memory addresses and event scheduling during the
  transition from NPU-IR's explicit plan to PTOAS TileLib expansion?
- What is the smallest tile-level representation that preserves ND2NZ layout,
  partial M/N/K, and the first-versus-accumulating K iteration?
- Can a split MIX kernel keep descriptor-carrying AIC calls and pointer-based
  AIV VMI operations when an unsupported Cube variant falls back to CCE?
- Which next `mmadL1` mode should follow the normalized f16 path: internal K
  segmentation, transpose, or bias/precision behavior?
- How should the bridge preserve full IEEE behavior for the row-softmax
  negative-infinity initializer on the current PTOAS path?
- How should signed integer scalar min/max be generalized beyond constant
  scalars and full masks?

## Rule For Planning Updates

When a planning document becomes too specific, keep it local to the experiment.
When a fact affects the overall project direction, update this file and
`bridge/memory/project-state.md`.
