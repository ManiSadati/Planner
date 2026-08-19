# Initial Exploration Plan

Last updated: 2026-08-10

Status: historical. This plan seeded the current bridge, PTOAS, NPU-IR,
PTO-ISA, and explorer docs. Use `bridge/planning/README.md` and
`bridge/memory/project-state.md` for current planning.

## Objective

Build enough source-backed context to create the first trustworthy NPU-IR-to-PTOAS mapping table and implementation plan.

This is Codex-led work, not scheduled explorer-agent work. Do not use the OpenAI API key for the one-time backfill.

## Outputs

- `explorer/reports/backfill/YYYY-MM-DD.md`: one-month ecosystem backfill report.
- `bridge/memory/project-state.md`: updated durable state.
- `bridge/memory/upstream-watch.md`: updated upstream/fork watch notes.
- `bridge/planning/npuir-to-ptoas-mapping.md`: source-backed mapping table.
- `bridge/planning/soyu-wilson-ave-to-vmi-branch-review.md`: reviewed
  prototype context for the AVE-to-VMI vector bridge branch.
- `PTOAS/design/*.md`: Planner-side PTOAS design summaries.
- `PTOAS/coding-guide/*.md`: PTOAS coding/build/pipeline guide notes.
- `NPUIR/design/*.md`: Planner-side NPU-IR design summaries.
- `NPUIR/coding-guide/*.md`: NPU-IR coding/build/pipeline guide notes.
- `PTO-ISA/design/*.md`: Planner-side PTO-ISA design summaries as needed.

## Stage 1: Local Repo Baseline

Status: complete. See `bridge/planning/local-repo-baseline.md`.

Check local repo state and structure without changing branches:

- `$HOME/PTOAS/PTOAS_Markham`
- `$HOME/AscendNPU-IR`
- `$HOME/pto-isa`

Collect:

- current branch and remotes;
- major doc directories;
- major compiler/pass directories;
- build/test entry points;
- existing AI/design docs already present in the repos.

## Stage 2: PTOAS Design Context

Status: complete enough to move to Stage 3. See:

- `PTOAS/design/ecosystem-inventory-2026-08-07.md`
- `PTOAS/design/lowering-pipeline.md`
- `PTOAS/coding-guide/pipeline-and-validation.md`
- `explorer/reports/backfill/2026-08-07-ptoas-stage2-snapshot.md`
- `explorer/reports/backfill/2026-08-07-ptoas-reexploration.md`

Follow-up note: after human verification caught the missed `WenboCodes/PTOAS:new-vf-fusion-design` branch, Codex reran focused branch triage. The follow-up added the branch-triage policy, captured Wenbo VMI-level VF fusion context, recorded legacy zhendong tile-fusion docs as hazard context, and recorded mouliangyu branch-local VMI contract context. A later four-day explorer lookback used the GitHub token successfully and produced current configured-scope PTOAS reports at `explorer/reports/README.md` and `explorer/reports/daily/2026-08-10.md`.

Answer:

- What is the current PTOAS lowering pipeline?
- Where do tileops, VMI, VPTO, PTODSL tilelib, emit-C, and PTO-ISA fit?
- Which parts are current versus transitional or legacy?
- Where is `expandtile` in the pipeline, and what IR exists immediately before and after it?
- What synchronization and memory-planning passes are required versus optional when input already carries planning information?

Before writing mapping-table rows for vector operations, use the complete
`docs/isa/vmi-isa/00-10` file set, not only the overview/load-store/predicate
subset.

Important planning take: current upstream TileLib templates are not VMI-based
by default, but branch-local design evidence points toward PTODSL/TileOp
expansion into logical VMI. The mapping table should therefore distinguish:

- preferred semantic target for future-compatible design, usually VMI for vector
  work;
- current upstream implementation target or fallback needed to compile today;
- information that must be preserved so a later VMI-based TileLib/fusion path is
  still usable.

## Stage 3: NPU-IR Design Context

Status: in progress. Initial local source scan is recorded in:

- `NPUIR/design/lowering-pipeline.md`
- `NPUIR/coding-guide/repo-and-validation.md`

Finding so far: `convert-hivmave-to-ave-intrin` is useful for vector-side
mapping, but it is not sufficient as the only bridge boundary. The driver runs
`convert-hivm-to-std` before `convert-hivmave-to-ave-intrin`, and the regbase
`HIVMToStandard` conversion lowers many DMA, cube, vector, sync-lock, SIMT, and
custom HIVM ops to external library calls. Mapping table rows should therefore
record both a vector/HIVMAVE boundary and an earlier HIVM boundary where needed.

Additional local-source findings:

- The HIVM pipeline performs memory planning, lower-to-loops, sync pipeline
  insertion/decomposition, memref-ext lowering, and FFTS metadata work before
  late conversion. "Before `convert-hivm-to-std`" should be split into finer
  rows when sync or memory ownership matters.
- `hivm.hir.mmadL1`/`mma*` are cube/template rows, not VMI-only vector rows.
- `hivm.hir.nd2nz` is a GM-to-CBUF ND-to-NZ template-backed DMA/layout row.
- `sync_block`, `set_flag`, `wait_flag`, `sync_block_set`, and
  `sync_block_wait` need an explicit sync-ownership column in the mapping table.
- These are local `mani/fuse-explore` findings only. Before claiming
  compatibility with the current Ascend upstream, fetch or inspect
  `https://gitcode.com/Ascend/AscendNPU-IR`.
- `soyu-wilson/AscendNPU-IR:codex/ave-to-vmi` is reviewed prototype context,
  not a branch to continue directly. It supports the vector-side boundary idea,
  but must be ported or reimplemented against current AscendNPU-IR and current
  PTOAS VMI. See `bridge/planning/soyu-wilson-ave-to-vmi-branch-review.md`.

Answer:

- What is the current NPU-IR lowering pipeline from Triton/high-level input to HIVM-AVE and CCE-oriented lowering?
- What does `convert-hivmave-to-ave-intrin` consume and produce?
- What does `HIVMToStandard` consume and produce?
- Where are `hir.load`, `hir.store`, `nd2nz`, `pge`, `set_flag`, `sync_block_set`, `mmadL1`, and `mma*` defined and lowered?
- Which stage preserves the most useful structure for PTOAS/PTO-ISA mapping?

## Stage 4: PTO-ISA Context

Status: local source-backed baseline complete enough for an initial mapping-table
draft. See:

- `PTO-ISA/design/virtual-isa-and-bridge-targets.md`
- `PTO-ISA/coding-guide/repo-and-validation.md`

Finding so far: PTO-ISA is a tile-level virtual ISA with explicit valid-region,
tile-location, layout, GlobalTensor, event, and backend-legality concepts. It is
a plausible semantic target for NPU-IR tile/DMA/cube/sync rows, while PTOAS VMI
remains the preferred future-compatible target for logical vector semantics when
the bridge enters PTOAS through VPTO/VMI.

Important local-source implications:

- `mmadL1`/`mma*` should be compared against `TMATMUL`, `TMATMUL_ACC`,
  `TMATMUL_BIAS`, and `TMATMUL_MX`, with surrounding `TLOAD`/`TMOV`/`TSTORE`.
- `nd2nz` should be compared against `TLOAD` Mat ND-to-NZ paths or `TMOV`
  ND-to-NZ movement depending on source/destination storage at the bridge point.
- `set_flag`/`wait_flag` should be compared against PTO `Event<SrcOp,DstOp>` /
  `TSYNC`, but only after pipe pair and event-token semantics are explicit.
- `sync_block*` may map to `SYNCALL` only when participant-set semantics match.

Answer:

- What level of abstraction does PTO-ISA expose?
- Which tile, DMA, sync, vector, and cube concepts are represented directly?
- Does PTOAS currently target PTO-ISA, VPTO, VMI, or multiple paths?
- What parts of PTO-ISA are useful for the NPU-IR bridge versus only for final CCE intrinsic expansion?

## Stage 5: One-Month Ecosystem Backfill

Status: partially covered by the 2026-08-10 four-day configured-scope explorer lookback. The current PTOAS watcher scope is up to date enough for near-term planning, and GitHub depth-2 fork discovery is now implemented for configured GitHub repos. The full one-month ecosystem backfill and GitCode issue/PR discovery are still pending.

Review recent design-relevant activity for:

- PTOAS upstream and named forks;
- active forks under `mouliangyu/PTOAS`;
- AscendNPU-IR upstream, the Wilson implementation fork, and the human
  `manisadati` fork when needed;
- relevant GitHub mirrors if available.

Focus on:

- branch movement that affects IR design, lowering pipelines, sync, memory planning, VMI, VPTO, PTO-ISA, or backend behavior;
- issues and PRs with design implications;
- work that could conflict with the bridge plan;
- work that confirms or updates the current hypothesis.

Current 2026-08-10 PTOAS watch items that should feed mapping-table review:

- LLVM19 / VPTO `feature-vpto` migration;
- cross-block versus intra-block sync API split;
- implicit tmp materialization pass ordering;
- PTO Common ops and `PTOLowerScalarToStandard`;
- VPTO scheduler framework;
- SoftLibService / late SoftLib expansion;
- FP4 L1-to-L0 S4 staging;
- large elementwise 1D/2D TileLib refactor in the Markham fork.

Do not summarize routine CI churn, typo fixes, or unrelated cleanup unless it changes design assumptions.

## Stage 6: Mapping Table

Status: local-first draft created. See:

- `bridge/planning/npuir-to-ptoas-mapping.md`

Important caveat: the draft is useful for planning, but it is not implementation
authority until Stage 5 upstream/fork reconciliation and example IR dumps are
added for at least one vector row, one DMA/layout row, one cube row, and one
sync row.

Finalize the source-backed mapping table after Stages 1-5.

Initial rows should include at least:

- `hir.vload`
- `hir.vadd`
- `ave.hir.pge`
- `hivm.hir.set_flag`
- `hivm.hir.pointer_cast`
- `hivm.hir.sync_block_set`
- `hivm.hir.nd2nz`
- `hivm.hir.load`
- `hivm.hir.store`
- `mmadL1`
- `mma*`

Columns:

```text
NPU-IR op/pattern | Current NPU-IR lowering | Proposed PTO/PTOAS target | Best interception point | Status | Risk | Source references | Notes
```

Use statuses like:

- `confirmed`
- `likely`
- `hypothesis`
- `blocked`
- `unknown`

## Stage 7: Implementation Plan

Only after the mapping table exists, draft the first staged implementation plan.

The plan should identify:

- the first minimal pass or prototype;
- where it lives in AscendNPU-IR;
- smallest useful test input;
- expected IR before/after;
- build/test commands;
- performance-sensitive assumptions.
