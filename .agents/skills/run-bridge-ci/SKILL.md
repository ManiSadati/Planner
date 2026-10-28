---
name: run-bridge-ci
description: Run an agreed AscendNPU-IR and PTOAS bridge CI check set, preserve revision-specific evidence, and classify build, compiler, and runtime failures without automatically changing code.
---

# Run Bridge CI

Read [Planner's contract](../../../AGENT.md), its required reading, the
[current plan](../../../bridge/planning/README.md), and the
[development guide](../../../bridge/designs/ptodsl-template-development-and-fallback.md).
Use `validate-bridge` for toolchain provenance and runtime validation details.
This skill runs checks; it does not install hosted CI or schedule jobs.

For local runs, use the short
[current environment note](../../../bridge/memory/current-local-environment.md)
and its already-wired candidate, not an installation recipe. The
[published 3.6 source baseline](../../../bridge/memory/triton-ascend-3.6-shared-a5-2026-10-22.md#shared-source-baseline)
records frontend provenance; other servers need a verified equivalent.
Record its actual patch and tool provenance; failures do not authorize switching
environments or reinstalling shared tools. Other servers need their own verified setup.

## Select the Check Set

1. Record the requested repository, revision/diff, check set, runtime budget,
   and allowed build/install work. Record dirty files rather than cleaning them.
   If scope is unspecified, start with focused compile checks and report that
   limited scope; do not silently launch a complete runtime matrix.
2. Locate existing lit, pytest, and workflow commands before inventing a runner.
   The development guide lists the bridge conversion lit directories. Runtime
   tests are separate from the normal CMake/lit suite and need explicit scope.
3. Record all four toolchain revisions, resolved binaries, installed template
   resources, patched Triton environment, and case filters. A stale build is an
   infrastructure blocker, not a source regression or permission to reinstall.

## Execute and Preserve Evidence

- Coordinate ownership before building/installing or using another job's tools.
  Do not update shared binaries during tests. Independent runs need separate
  private output/cache directories, a bounded worker count, and time limits.
- Run the selected compile checks first. For authorized simulator coverage, use
  patched Triton Python tests with explicit instance/shape filters. No filters
  can mean over a hundred expensive cases, not one smoke test.
- Preserve commands, logs, exit statuses, failed parameter IDs, and failing IR
  when available. Capture the actual test exit status through `tee`; profiler
  completion alone is not a pass. Distinguish timeouts from numerical failures.
- Check native baseline and selected PTODSL symbols where applicable. An allowed
  native fallback validates fallback behavior, not the missing implementation.
- A failing stage is evidence, not necessarily the cause. Classify failures as
  environment/build, native limitation, bridge regression, suspected design
  problem, or unknown. Explain evidence and uncertainty; do not patch to green,
  weaken assertions, change test configs, or hide failures with retries.

## Report

Return revision/tool provenance, scope and exclusions, pass/fail/skip/timeout
counts, elapsed time, failure reproductions, and artifact paths. Label blocked
or incomplete runs honestly. Do not edit sources, tests, workflows, pins,
global configuration, or native libraries; commit/push nothing. Planner updates
and full/nightly/hardware suites require an explicit assignment.
