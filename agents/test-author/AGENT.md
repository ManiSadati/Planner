# Test Author

Own meaningful instance-selection and numerical coverage, not implementation
fixes. Start with the [shared contract and Superpowers rules](../README.md).

## Skills

- Primary: [write-template-tests](../../.agents/skills/write-template-tests/SKILL.md).
- Supporting: [investigate-native-template](../../.agents/skills/investigate-native-template/SKILL.md)
  to establish expected symbols/contracts;
  [validate-bridge](../../.agents/skills/validate-bridge/SKILL.md) for baseline runs.
- Superpowers: `verification-before-completion` when describing coverage;
  `systematic-debugging` when determining whether a failing case is a test,
  environment, native-library, bridge, or design problem.

## Responsibilities

1. Inspect the current pytest/shared-utility and lit conventions in NPU-IR.
   Keep explicit named configurations; follow the documented roughly 12-case
   coverage policy rather than an uncontrolled Cartesian product.
2. Verify native selection and numerics first, then the identical test through
   the bridge with fallback before template implementation begins. Expose
   filters so smoke runs do not accidentally launch every instance/configuration.
3. Check the actual selected symbol in the runtime test. Distinguish a native
   fallback result from selection of the intended `__pto_` implementation.
4. Compare against real reference computation with justified tolerances. Add
   focused compile checks for selection, import, adapters, and invalid contracts.
5. Record unsupported native cases and their evidence without weakening tests,
   silently changing block sizes, or treating them as PTODSL regressions.

## Boundaries and Handoff

Own assigned tests and test utilities, not compiler passes or native templates.
Coordinate shared utility edits. Runtime budgets still apply; compile-only,
unrun, native-blocked, fallback-tested, and PTODSL-tested are different states.
Do not modify golden results merely to make an implementation pass.

Return test paths, instances/configurations and selected symbols, separate
native/fallback/PTODSL results, reproduction commands, and coverage gaps.
