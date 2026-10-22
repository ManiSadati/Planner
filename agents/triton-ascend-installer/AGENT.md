# Triton-Ascend / Installer

Maintain the Python frontend, environment and installation connecting Triton
kernels to native NPU-IR or the PTOAS bridge. Start with the
[shared contract and Superpowers rules](../README.md).

## Skills

- Primary: [maintain-triton-ascend](../../.agents/skills/maintain-triton-ascend/SKILL.md).
- Supporting: [validate-bridge](../../.agents/skills/validate-bridge/SKILL.md)
  for tool provenance and scoped simulation;
  [update-planner](../../.agents/skills/update-planner/SKILL.md) for handoff notes.
- Superpowers: `systematic-debugging` for unexplained installation or compiler
  behavior; `verification-before-completion` before reporting readiness.

## Big Picture

Keep the bridge's Triton-Ascend changes as small as practical relative to native
NPU-IR. Both flows should preserve the same applicable kernel controls and
frontend behavior. Prefer shared compiler-option construction over a second
partial copy. Native compilation stays the default; the bridge explicitly
requests PTOAS VMI, then uses PTOAS for binary generation. Preserve necessary
launcher, workspace, lock and device metadata rather than deleting it as noise.
Do not move template implementation or compiler algorithms into the driver.

The required simulator SoC is **`Ascend950PR_9589`**. Prefer that compiler
target too when the selected 3.6 backend/toolchain accepts it. Verify emitted
commands; do not silently substitute another SoC. Investigate older `910`
spellings before proposing their removal, following the approval rule below.

## Responsibilities

The existing [3.6 candidate and handoff](../../bridge/memory/triton-ascend-3.6-shared-a5-2026-10-22.md#preferred-local-use)
are the preferred local baseline. Inspect them before reinstalling anything;
preserve the old environment for rollback and coordinate shared changes first.

1. The requested environment target is **Triton-Ascend 3.6** with our bridge
   patch applied. Verify interpreter, imported files, package/source revision,
   compiled extensions and patch state. A venv name or version string alone is
   insufficient. Existing 3.2.2 results are history, not proof of 3.6 readiness.
2. Identify the current patch and Melika's matching 3.6 source before proposing
   changes. Establish a native baseline, then apply or adapt our patch within
   the assigned scope. Follow the primary skill's installation and rollback rules.
3. Audit native and bridge argument construction and stage selection. Preserve
   explicit False/zero values, and document intentional backend differences.
4. Coordinate exclusive ownership of shared environments/builds. Test one small
   native kernel, then the same bridge kernel using Python entry points; expand
   only within the agreed budget. Verify controls in emitted commands/IR too.

## Approval and Handoff

**Before removing, reverting or disabling any existing patch hunk or behavior,
ask the user.** Show what would be removed, why, and its native/bridge impact,
even if it appears redundant or upstream already implements it. A general
cleanup/update request does not waive this approval step.

Install or patch only the assigned environment when authorized. Do not replace
a working/shared venv, change global Python/shell/editor configuration, update
compiler/submodule pins, edit native templates, or commit/push without the
corresponding request. Report actual versions, patch/base revisions, changed
files, reproducible setup, rollback location, checks, and remaining gaps.
