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

There is not yet a dedicated folder of early IR examples.

If conversion work needs real NPU-IR examples, Codex should explicitly ask the
human to generate or place those IR dumps. Codex should not silently substitute
synthetic examples for source-backed conclusions.

## Suggested Future Folder

Use a repo-relative folder name when it is created, for example:

```text
bridge/examples/npuir-early-ir/
```

Each example should ideally include:

- original source/kernel name;
- A5 machine/compiler version, if known;
- early IR before `convert-hivm-to-std`;
- IR after `convert-hivm-to-std`, when useful for comparison;
- command or pipeline notes used to produce the dump.
