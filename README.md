# Planner

Planner is the coordination repo for the AscendNPU-IR, PTOAS, PTO-ISA, and
NPU-IR-to-PTOAS bridge work.

Start here:

- `AGENT.md`: Codex operating contract.
- `human/HighLevelOverview.md`: human-owned project intent.
- `bridge/README.md`: cross-repo bridge and comparison entry point.
- `bridge/planning/README.md`: current plan and next steps.
- `bridge/memory/project-state.md`: durable current state.
- `explorer/reports/README.md`: latest upstream/fork monitoring summary.

Folder ownership:

- `human/`: human-owned intent. Codex should not edit it unless explicitly told.
- `bridge/`: cross-repo bridge planning, shared fixtures, comparison harness,
  and durable working memory.
- `NPUIR/`: AscendNPU-IR-specific design notes, build/run docs, and helper
  scripts.
- `PTOAS/`: PTOAS-specific design notes and build/run docs.
- `PTO-ISA/`: PTO-ISA-specific design notes and build/run docs.
- `explorer/`: scheduled monitoring agent and reports.

Current short-term priority:

```text
clean comparison structure -> rerun baseline NPU-IR and bridge-path tests ->
continue guarded conversion work from the Wilson AscendNPU-IR fork
```
