# AscendNPU-IR Planner Notes

This folder contains Planner-side AscendNPU-IR notes and NPU-IR-only tools.

- `design/`: pipeline and bridge-relevant IR summaries.
- `coding-guide/`: build, simulator, replay, install, and validation notes.
- `tools/`: NPU-IR-only helper scripts.

Bridge-level comparison, PTOAS handoff, and cross-repo test orchestration stay
under `bridge/`.

Current implementation source is the Wilson fork:

```text
https://gitcode.com/wilsoncxfeng/AscendNPU-IR
```

Check the current active branch before starting code work. `master` is expected
to become the integration branch, but active work may temporarily live on a
branch such as `melika/ave-to-vmi`.
