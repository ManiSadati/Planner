# Bridge

`bridge/` is the working connection between the human project owner and Codex.

- `memory/`: compact durable project facts, decisions, risks, and workflow notes.
- `planning/`: active plans, mapping tables, staged work, and exploration outputs.

Human intent lives in `human/`. If a bridge note conflicts with `human/`, the human document wins and Codex should report the mismatch.

## Current Status

- Explorer bot is installed as a user systemd timer and runs daily at 7:00am Eastern.
- Latest configured-scope PTOAS report: `explorer/reports/README.md`.
- Latest big-change report: `explorer/reports/daily/2026-08-10.md`.
- Durable state summary: `bridge/memory/project-state.md`.
- Upstream/fork watch list: `bridge/memory/upstream-watch.md`.
- Current mapping draft: `bridge/planning/npuir-to-ptoas-mapping.md`.

The configured PTOAS watcher scope is current as of 2026-08-10. GitHub
direct-fork/fork-of-fork discovery is implemented for configured GitHub repos.
GitCode issue/PR tracking is still pending.
