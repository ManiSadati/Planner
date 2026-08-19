# A5 IR Workflow

Last updated: 2026-08-11

## Rule

Do not assume the full AscendNPU-IR Python/Triton lowering workflow can run on
the Codex-accessible server.

That workflow should be run on a server with actual A5 hardware. The local
server can still be used for code inspection, pass development, source-backed
planning, compile-only checks when they do not require A5 runtime support, and
analysis of saved IR files.

## Practical Workflow

1. The human runs the Triton/Python entry workflow on an A5 machine.
2. The human lowers the example to early MLIR / early NPU-IR dumps.
3. Those IR dumps are copied into a Planner-accessible examples folder.
4. Codex uses those stable IR inputs locally to inspect operation shapes,
   compare pass boundaries, design mappings, and run local pass-level checks
   where possible.
5. When hardware validation is needed again, Codex asks the human for the exact
   A5 command/output/log needed.

## Current State

Source Triton fixtures live in:

```text
bridge/triton-example/
```

The early-IR dump landing folder is:

```text
bridge/examples/npuir-early-ir/
```

It currently contains only workflow documentation unless the human has added
actual A5-generated dumps.

The current `*_kernel.mlir` files in `bridge/triton-example/` are A5-generated
dumps right after `AppendTargetDeviceSpec (hacc-append-device-spec)`. They can
be used as local replay inputs after NPU-IR is built here. See
`NPUIR/coding-guide/device-spec-replay.md`.

Use `NPUIR/coding-guide/a5-installation.md` for the A5 install/runtime workflow.
Use `NPUIR/coding-guide/codex-server-build.md` for the non-A5 Codex-server
build/replay workflow. Keep these paths separate: the A5 machine produces the
runtime-backed dumps, while the Codex server replays saved dumps where possible.

For local replay, the current endpoint is `convert-hivmave-to-ave-intrin`. We
do not need compiler stages after that for the current bridge investigation.

If conversion work needs real NPU-IR examples, Codex should explicitly ask the
human to generate or place those IR dumps. Codex should not silently substitute
synthetic examples for source-backed conclusions.

## Dump Contents

Each example should ideally include:

- original source/kernel name;
- A5 machine/compiler version, if known;
- early IR before `convert-hivm-to-std`;
- IR after `convert-hivm-to-std`, when useful for comparison;
- command or pipeline notes used to produce the dump.
