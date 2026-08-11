# Bridge

`bridge/` is the working connection between the human project owner and Codex.

- `memory/`: compact durable project facts, decisions, risks, and workflow notes.
- `planning/`: active plans, mapping tables, staged work, and exploration outputs.

Human intent lives in `human/`. If a bridge note conflicts with `human/`, the human document wins and Codex should report the mismatch.

## Current Status

- Explorer bot is installed as a user systemd timer and runs daily at 7:00am Eastern.
- Latest configured-scope PTOAS report: `explorer/reports/README.md`.
- Latest big-change report: `explorer/reports/daily/2026-08-11.md`.
- Durable state summary: `bridge/memory/project-state.md`.
- A5/early-IR workflow memory: `bridge/memory/a5-ir-workflow.md`.
- DMA template mapping memory: `bridge/memory/dma-template-mapping.md`.
- Upstream/fork watch list: `bridge/memory/upstream-watch.md`.
- Current mapping draft: `bridge/planning/npuir-to-ptoas-mapping.md`.
- DMA rewrite plan: `bridge/planning/dma-template-rewrite-plan.md`.

The configured PTOAS watcher scope is current as of 2026-08-11. GitHub
direct-fork/fork-of-fork discovery is implemented for configured GitHub repos.
GitCode issue/PR tracking is still pending.
