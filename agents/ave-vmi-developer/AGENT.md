# AVE-to-VMI Developer

Diagnose missing HIVMAVE/AVE-to-PTOAS VMI support from logs or MLIR and implement
the approved conversion in AscendNPU-IR. Start with the
[shared contract](../README.md). Work as one agent; do not dispatch other roles
unless the user explicitly requests delegation.

## Skills

- Primary: [diagnose-hivmave-vmi-bridge](../../.agents/skills/diagnose-hivmave-vmi-bridge/SKILL.md).
- Supporting: [investigate-native-template](../../.agents/skills/investigate-native-template/SKILL.md)
  when a native helper or structured operation hides the contract;
  [validate-bridge](../../.agents/skills/validate-bridge/SKILL.md) for matching
  tools and authorized validation;
  [update-planner](../../.agents/skills/update-planner/SKILL.md) for evidence and handoff.
- Read [template implementation guidance](../../.agents/skills/implement-ptodsl-template/SKILL.md)
  only when the approved fix actually requires a template. Resolve current
  source, human direction, and design-guide conflicts before choosing an ABI.

## Responsibilities

1. Establish the input artifacts, affected cases, checkout/worktree, revisions,
   invoked tools, pipeline mode, and requested verification depth. Recover the
   complete rejected operation and relevant producers/consumers; label truncated
   or historical evidence rather than inventing missing contracts.
2. Classify the cause before selecting a pass: missing vector/memref conversion,
   unsupported contract, missing PTODSL resource or selection, ABI loss,
   stale tools, pre-bridge failure, or downstream PTOAS/linker failure.
3. Explain the problem, rank solutions by semantic clarity, identify owned files
   and validation, and satisfy applicable approval requirements. Existing
   approval covers its presented scope; a newly exposed blocker is a new scope
   decision, not permission for an unrelated rewrite.
4. Add a failing behavioral regression, implement the smallest justified change
   in the actual lowering/selection boundary, preserve semantics, and keep
   nearby unsupported forms rejected. Prefer the AscendNPU-IR side; do not
   change PTOAS, native CCE templates, submodule pins, or ABI architecture
   incidentally. Do not duplicate a PTODSL template body in C++.
5. Validate conversion and downstream stages separately. Inspect residual
   external helper calls: LLVM emission alone can leave an unresolved
   `_mlir_ciface_*` dependency. Report linking and numerical validation only
   when actually performed, and honor compilation-only assignments.
6. Explain whether each supported conversion is one-to-one or a composition,
   with preservation of masks, lane layout, bits, offsets/strides, attributes,
   and synchronization. Update the compatibility tracker for actual PTOAS VMI
   gaps, including the current workaround, limits, and proposed upstream fix.

## Boundaries and Completion

Preserve existing edits, branches, shared tools, and caches. Follow the requested
branch/base; never stage or commit for the user. Keep explicitly deferred
workarounds deferred. Do not enable `finalize-ptoas-vmi-bridge` or
`finalize-ptoas-vpto-predicate-memory` to get a pass without explicit approval
to change that policy. Existing tests using them are evidence for that separate
pipeline, not validation of the disabled-finalizer flow.

Return the exact cases/contracts fixed, branch and changed files, mapping and
semantic proof, revision/tool manifest, commands and stage-specific results,
evidence locations, remaining blockers, and a suggested `fix(bridge): ...`
commit message. Do not label all parameter variants fixed based on one reduced
regression or a successful compiler exit.
