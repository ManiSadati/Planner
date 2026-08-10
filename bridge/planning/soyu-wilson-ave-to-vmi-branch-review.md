# Soyu-Wilson AVE-To-VMI Branch Review

Last updated: 2026-08-10

## Branch Identified

The branch is:

- Repo: `https://github.com/soyu-wilson/AscendNPU-IR`
- Branch: `codex/ave-to-vmi`
- Head: `4f6bc0b34045f225b9aed1d15a53a8538f84111a`
- Commit message: `Add AVE to VMI conversion pass`
- Related issue: `https://github.com/soyu-wilson/AscendNPU-IR/issues/1`

This is the branch referenced from `human/HighLevelOverview.md`.

## What The Branch Adds

Compared with Soyu-Wilson fork `master` at `1bae10b805e450bce0efbae095e8b144a2283428`,
the branch adds one small prototype:

- 20 files changed.
- 979 inserted lines.
- New `bishengir/Conversion/HIVMAVEToVMI` pass.
- New `bishengir/Dialect/PTOBridge` textual bridge dialect.
- Pass/CMake/dialect registration.
- Two lit tests:
  - `bishengir/test/Conversion/HIVMAVEToVMI/hivmave-to-vmi.mlir`
  - `bishengir/test/Conversion/HIVMAVEToVMI/unsupported.mlir`

The pass consumes `ave.hir.*` operations immediately before
`convert-hivmave-to-ave-intrin` and emits textual `pto.vmi.*` operations with
logical VMI vector and mask types.

## Immediate Verdict

Do not continue this branch directly as the main implementation branch.

Use it as a prototype and reference for the first vector-side bridge pass.
Port selected ideas into a fresh branch based on the current AscendNPU-IR
development baseline and current PTOAS VMI contract.

Reliability by area:

| Area | Reliability | Notes |
| --- | --- | --- |
| Branch identification | high | Branch and issue were verified directly. |
| Pass placement idea | medium-high | The `HIVMAVE` boundary before intrinsic lowering is still a plausible vector boundary. |
| AVE-to-VMI operation mapping table | medium | Useful starting point for elementwise, predicate, load/store rows, but incomplete. |
| Current PTOAS compatibility | low-medium | Current PTOAS has moved since the issue comparison; the branch was not built against PTOAS VMI verifiers. |
| Full NPU-IR bridge coverage | low | It is AVE/vector only and does not solve HIVM DMA, cube, memory planning, or sync rows. |
| Direct merge/cherry-pick safety | low | Patch check against local current AscendNPU-IR failed at `InitAllDialects.h`; more importantly, it is based on a fork branch, not the current source-of-truth baseline. |

## What Is Helpful

The branch is useful because it validates several planning assumptions:

- A vector-only prototype can be placed before `convert-hivmave-to-ave-intrin`.
- The first bridge slice can stay strict and fail on unmapped AVE semantics
  instead of silently approximating behavior.
- Logical vector/mask conversion can preserve lane count and element type before
  physical PTOAS layout assignment.
- Basic AVE families have straightforward VMI candidates:
  - `vadd`, `vsub`, `vmul`
  - `vabs`, `vneg`, `vexp`, `vsqrt`, `vln`, `vrelu`
  - `vadds`, `vmuls`, `vmaxs`, `vmins`
  - `vload`, `masked_store`
  - `pge`, `plt`
  - `vcmp`, `vcmps`
  - `vsel`, `vci`
  - predicate boolean ops
  - `vintlv`, `vdintlv`
- The negative diagnostics are useful. A bridge pass should explicitly reject
  unsupported conversions, reductions, complex memory modes, and layout casts
  until each has a PTOAS-backed contract.

These pieces are worth reading before implementing the vector prototype:

- `bishengir/lib/Conversion/HIVMAVEToVMI/HIVMAVEToVMI.cpp`
- `bishengir/lib/Conversion/HIVMAVEToVMI/README.md`
- `bishengir/test/Conversion/HIVMAVEToVMI/hivmave-to-vmi.mlir`
- `bishengir/test/Conversion/HIVMAVEToVMI/unsupported.mlir`

## What Should Not Be Trusted As-Is

The branch is not an implementation authority.

Reasons:

- The branch name is AI-prefixed: `codex/ave-to-vmi`. Per branch triage policy,
  that is not disqualifying, but it means the code needs source and verifier
  confirmation.
- It creates a local `PTOBridge` dialect that allows unknown operations. That
  is acceptable as a temporary text-emission shim, but it is not equivalent to
  linking against or verifying with PTOAS.
- Its tests check only `bishengir-opt` textual output and unsupported-op
  diagnostics. They do not prove PTOAS can parse, verify, lower, or run the
  emitted IR.
- It covers only the AVE vector layer. It does not cover `hivm.hir.load`,
  `hivm.hir.store`, `hivm.hir.nd2nz`, `hivm.hir.mmadL1`, `mma*`,
  `set_flag`, `wait_flag`, or `sync_block*`.
- Its related issue compares against an older PTOAS state. Current PTOAS
  upstream already has a first-class `pto.vmi` layer, so some of the issue's
  namespace/type mismatch conclusions are stale.
- The branch was compared against Soyu-Wilson fork `master`, not against the
  official Ascend GitCode upstream. The local manisadati master is at a
  different baseline.

## Updated PTOAS Compatibility Read

The related GitHub issue says the branch emits a proposed textual interchange
schema, not directly consumable PTOAS IR. That was a fair warning for the
PTOAS state it inspected.

Current local PTOAS upstream `upstream/main` at
`988d50e245217669a27448c96641bb7eaf26baed1` contains:

- `include/PTO/IR/VMIOps.td`
- `include/PTO/IR/VMITypeDefs.td`
- `include/PTO/IR/VMIAttrs.td`
- `lib/PTO/IR/VMI.cpp`
- `lib/PTO/Transforms/VMIToVPTO.cpp`
- `docs/isa/vmi-isa/00-architecture-overview.md`

This changes the assessment:

- The old "PTOAS has no `pto.vmi` namespace" concern is stale.
- Several Soyu-emitted names now exist in current PTOAS VMI, including
  `pto.vmi.vload`, `pto.vmi.vstore`, `pto.vmi.pset`, `pto.vmi.pge`,
  `pto.vmi.plt`, and the main `vadd`/`vmul`/`vabs` family.
- Current PTOAS VMI types have optional layout metadata. Soyu's no-layout
  logical types may be acceptable as pre-layout text, but this must be verified
  through PTOAS parsing and VMI verification.
- Current PTOAS VMI has more target operations than Soyu's pass uses,
  including conversion, reduction, gather/scatter, group, and layout support.
  Those are future mapping work, not reasons to trust the existing branch.

So the modern conclusion is narrower:

Soyu's branch is closer to current PTOAS VMI than its issue originally made it
look, but it is still only a partial AVE-vector prototype with no end-to-end
PTOAS validation.

## Recommended Reuse

Take these ideas:

- The pass boundary: `HIVMAVE` after AVE normalization and before
  `convert-hivmave-to-ave-intrin`.
- The strict unsupported-op audit before rewriting.
- The conversion table as a seed for the first vector rows.
- The test structure: one positive file and one expected-diagnostic file.
- The idea of keeping layout logical and letting PTOAS assign physical layout.

Do not take these unchanged:

- The separate `PTOBridge` dialect as the long-term contract.
- The assumption that AVE/vector coverage is enough for the full bridge.
- The issue's older PTOAS compatibility table without regenerating it against
  current PTOAS VMI source.
- The exact branch history or patch as the implementation base.

## Recommended Implementation Direction

Start a fresh branch from the current AscendNPU-IR development baseline.

First prototype:

1. Build a narrow `HIVMAVEToPTOASVMI` or similarly named pass.
2. Handle only:
   - `pge`/`plt`/all-active mask creation;
   - `vload`/`masked_store` continuous modes;
   - `vadd`/`vmul`/`vabs` or another minimal arithmetic trio.
3. Emit current PTOAS VMI-compatible text.
4. Add an NPU-IR lit test proving the pass rewrites AVE correctly.
5. Add a PTOAS-side parse/verify smoke test for the emitted text.
6. Keep unsupported operations fail-fast.

Do not include DMA/cube/sync rows in this first AVE pass. Those need earlier
HIVM interception points and separate mapping rows.

## Open Checks Before Coding

- Re-run the AVE operation inventory on current Ascend upstream and
  `manisadati` development branch.
- Regenerate the Soyu mapping table against current PTOAS VMI docs/source, not
  the older issue comparison.
- Verify whether NPU-IR can emit a textual `.mlir` file acceptable to PTOAS
  without linking PTOAS into the AscendNPU-IR build.
- Decide whether the temporary bridge dialect is acceptable for prototyping or
  whether the first pass should emit through a neutral textual export path.
- Capture one real pre-`convert-hivmave-to-ave-intrin` IR dump from an example
  kernel before implementing beyond the toy tests.

