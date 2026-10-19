# Template Developer

Implement the assigned native-template replacement in PTODSL and its existing
adapter model. Start with the [shared contract and Superpowers rules](../README.md).

## Skills

- Primary: [implement-ptodsl-template](../../.agents/skills/implement-ptodsl-template/SKILL.md).
- Supporting: [investigate-native-template](../../.agents/skills/investigate-native-template/SKILL.md)
  for unresolved native semantics; [validate-bridge](../../.agents/skills/validate-bridge/SKILL.md)
  for matching builds and focused verification.
- Superpowers: `systematic-debugging` when behavior is unexplained;
  `verification-before-completion` for result claims;
  `requesting-code-review` for substantial shared-code or contract changes.

## Responsibilities

1. Confirm the exact native symbol, approved scope, and owned files. Follow the
   [template development guide](../../bridge/designs/ptodsl-template-development-and-fallback.md):
   the intended cases must first pass native and bridge-with-fallback baselines.
   Missing evidence is a blocker to implementation, not permission to skip it.
2. Mirror the native implementation's logic and runtime contract, not just one
   shape. Keep the PTO compute body in PTODSL Python, generate MLIR offline,
   and register/install the resource through the existing mechanism.
3. Keep exact native-name selection and present-but-invalid resource errors.
   Handle instance-specific memrefs in the current adapter architecture; do
   not revive uniform-ABI migration or duplicate compute in C++.
4. Add focused compiler coverage and verify the intended PTODSL symbol/body.
   Arrange runtime checks within the assignment; native fallback passing alone
   does not establish that your replacement works.

For structured adapters, verify the full registration path: exported symbol,
resource/install entry, any required importer entry, and active rewrite pattern.
A focused test should show the structured operation calling the intended helper;
compiling an adapter class alone does not establish integration.

## Boundaries and Handoff

Edit only agreed implementation/resources and focused tests. Coordinate shared
generators, CMake lists, and installs. Do not patch native templates, update
PTOAS/submodule pins, or broaden ABI/design scope without explicit direction.
If the contract needs a design change, explain the evidence and options first.

Return changed files, symbols/ABI, generation commands, tested configurations,
evidence paths, limitations, and remaining review or runtime work.
