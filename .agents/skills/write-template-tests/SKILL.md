---
name: write-template-tests
description: Write focused compiler and pytest-style runtime tests for AscendNPU-IR template instances, verifying exact selection and native-supported configurations before PTODSL numerical parity.
---

# Write Template Tests

Read [Planner's contract](../../../AGENT.md), its required reading, the
[current plan](../../../bridge/planning/README.md), and the
[runtime coverage policy](../../../bridge/inventory/ptodsl-template-test-triage.md#runtime-test-selection-standard).
Use the actual code checkout, not legacy Planner fixture copies.

For new local runtime tests, use the preferred 3.6 candidate in the
[environment handoff](../../../bridge/memory/triton-ascend-3.6-shared-a5-2026-10-22.md#preferred-local-use).
Verify the selected interpreter, patch and compiler provenance for both flows;
do not silently change environments or patch the shared installation to pass.

## Test Design

1. Inspect existing tests under
   `bishengir/test/Integration/PTOASBridge/template_runtime/` and reuse its
   `utils.py`. Keep pytest parameter IDs and expected native symbols explicit;
   shared compiler discovery/IR inspection belongs in utilities, not each test.
2. Select configurations using the roughly 12-case coverage policy: small odd
   tensors, tile-boundary cases, larger odd tensors, legal small/large and
   asymmetric blocks, tails, reduction iterations, and varied program grids.
   Do not force a Cartesian product or silently clamp unsupported block sizes.
3. Run the unchanged kernel/configuration through native NPU-IR first. Record
   native failures as baseline limitations; do not patch native templates or
   count those cases as evidence against a PTODSL instance.
4. Check the expected native instance or `__pto_<native-symbol>` in actual
   generated IR before claiming runtime coverage. Existing utilities can allow
   native fallback: inspect their returned symbol and distinguish fallback
   coverage from proof that the PTODSL implementation was exercised.
   For replacement validation with `FALLBACK=1`, require the target's exact
   `__pto_` call and materialized PTO body (or verified inlined provenance).
   Reject native fallback for that target while allowing unrelated dependencies
   to fall back; a fallback-tolerant helper's numerical PASS is insufficient.
5. Compare against actual reference computation using identical data/configs
   for native and bridge paths. Keep numerical tolerances justified by datatype.
6. Put focused lit/FileCheck selection, adapter, and diagnostic checks under
   the relevant `bishengir/test/Conversion/` directory. Compile success alone
   does not establish numerical correctness.

## Run and Report

- Use patched Triton `python kernel.py` or the test's pytest entry point under
  the simulator environment; do not substitute legacy comparison-runner tests.
- Run only the requested scope. Obtain direction before launching a full costly
  simulator matrix; use separate output/cache directories for independent runs.
- Return instance/configuration, selected symbol, native result, bridge result,
  and evidence path for each case. Distinguish unrun, skipped, failed, and passed.
