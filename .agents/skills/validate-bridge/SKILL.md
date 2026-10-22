---
name: validate-bridge
description: Validate an AscendNPU-IR to PTOAS bridge build or runtime failure using matching toolchain revisions, patched Triton Python entry points, and bounded native-versus-bridge checks.
---

# Validate the Bridge

Read [Planner's contract](../../../AGENT.md), its required reading, the
[current plan](../../../bridge/planning/README.md), and the commands in the
[development guide](../../../bridge/designs/ptodsl-template-development-and-fallback.md).
Use its environment setup rather than adding exports to kernels.

Use the [published 3.6 source baseline](../../../bridge/memory/triton-ascend-3.6-shared-a5-2026-10-22.md#shared-source-baseline)
and its recorded setup; this server's preferred candidate is only one installation.
On other servers, verify equivalent source/patch provenance and report missing setup.
Verify the actual interpreter, patch and compiler paths; do not silently switch
to an older environment or modify the shared installation after a failure.

## Establish Provenance

1. Record NPU-IR and all relevant submodule revisions: Ascend LLVM, PTOAS, and
   PTO LLVM. Resolve the actual compiler executables, venv, and installed MLIR
   resources. A branch checkout does not rebuild installed tools.
2. Use `bishengir-compile` from the intended NPU-IR build/install, `hivmc-a5`
   from CANN, and PTOAS from this checkout's pinned submodule build. Do not use
   Triton-bundled compiler copies or an unrelated standalone PTOAS checkout.
3. Check the installed Triton driver patch, not just its package version.
   Keep the two LLVM toolchains separate; textual MLIR is their boundary.

## Bounded Validation

- Run cheap compiler/IR checks first. If sources/resources changed, use the
  guide's incremental build then install commands; do not rebuild LLVM by habit.
  Coordinate one writer to shared build/install directories and do not replace
  binaries while another agent's tests are using them.
- For runtime checks, use patched Triton `python kernel.py` under
  `msprof op simulator` on this server. Start with one relevant small case and
  a fresh private cache/output directory, not a full model or shape matrix.
- Compare native and PTODSL flows with the same kernel/data/configuration.
  Record compile-flow, bridge-mode, fallback, SoC, and case-filter settings.
  `FALLBACK=1` permits native dependencies; unset/zero is the no-fallback path.
- Inspect child/compiler errors, selected symbols, and numerical PASS evidence.
  A successful profiler exit or a trace file is not proof of a successful test.
- Preserve logs and first failing IR. Report simulator timings as simulator
  measurements, not hardware performance. Physical A5 testing is a human handoff.

Return the revision/tool manifest, exact reproduction command, tested scope,
results, first actionable failure, and evidence locations. Do not silently
repair unrelated environments, update pins, or launch background suites.
