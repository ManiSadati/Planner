# Initial Exploration Plan

Last updated: 2026-08-07

## Objective

Build enough source-backed context to create the first trustworthy NPU-IR-to-PTOAS mapping table and implementation plan.

This is Codex-led work, not scheduled explorer-agent work. Do not use the OpenAI API key for the one-time backfill.

## Outputs

- `explorer/reports/backfill/YYYY-MM-DD.md`: one-month ecosystem backfill report.
- `bridge/memory/project-state.md`: updated durable state.
- `bridge/memory/upstream-watch.md`: updated upstream/fork watch notes.
- `bridge/planning/npuir-to-ptoas-mapping.md`: source-backed mapping table.
- `PTOAS/design/*.md`: Planner-side PTOAS design summaries.
- `PTOAS/coding-guide/*.md`: PTOAS coding/build/pipeline guide notes.
- `NPUIR/design/*.md`: Planner-side NPU-IR design summaries.
- `NPUIR/coding-guide/*.md`: NPU-IR coding/build/pipeline guide notes.
- `PTO-ISA/design/*.md`: Planner-side PTO-ISA design summaries as needed.

## Stage 1: Local Repo Baseline

Status: complete. See `bridge/planning/local-repo-baseline.md`.

Check local repo state and structure without changing branches:

- `/home/m84446336/PTOAS/PTOAS_Markham`
- `/home/m84446336/AscendNPU-IR`
- `/home/m84446336/pto-isa`

Collect:

- current branch and remotes;
- major doc directories;
- major compiler/pass directories;
- build/test entry points;
- existing AI/design docs already present in the repos.

## Stage 2: PTOAS Design Context

Answer:

- What is the current PTOAS lowering pipeline?
- Where do tileops, VMI, VPTO, PTODSL tilelib, emit-C, and PTO-ISA fit?
- Which parts are current versus transitional or legacy?
- Where is `expandtile` in the pipeline, and what IR exists immediately before and after it?
- What synchronization and memory-planning passes are required versus optional when input already carries planning information?

## Stage 3: NPU-IR Design Context

Answer:

- What is the current NPU-IR lowering pipeline from Triton/high-level input to HIVM-AVE and CCE-oriented lowering?
- What does `convert-hivmave-to-ave-intrin` consume and produce?
- What does `HIVMToStandard` consume and produce?
- Where are `hir.load`, `hir.store`, `nd2nz`, `pge`, `set_flag`, `sync_block_set`, `mmadL1`, and `mma*` defined and lowered?
- Which stage preserves the most useful structure for PTOAS/PTO-ISA mapping?

## Stage 4: PTO-ISA Context

Answer:

- What level of abstraction does PTO-ISA expose?
- Which tile, DMA, sync, vector, and cube concepts are represented directly?
- Does PTOAS currently target PTO-ISA, VPTO, VMI, or multiple paths?
- What parts of PTO-ISA are useful for the NPU-IR bridge versus only for final CCE intrinsic expansion?

## Stage 5: One-Month Ecosystem Backfill

Review recent design-relevant activity for:

- PTOAS upstream and named forks;
- active forks under `mouliangyu/PTOAS`;
- AscendNPU-IR upstream and `manisadati` fork;
- relevant GitHub mirrors if available.

Focus on:

- branch movement that affects IR design, lowering pipelines, sync, memory planning, VMI, VPTO, PTO-ISA, or backend behavior;
- issues and PRs with design implications;
- work that could conflict with the bridge plan;
- work that confirms or updates the current hypothesis.

Do not summarize routine CI churn, typo fixes, or unrelated cleanup unless it changes design assumptions.

## Stage 6: Mapping Table

Create the first source-backed mapping table after Stages 1-5.

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
