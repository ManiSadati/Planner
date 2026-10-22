---
name: maintain-triton-ascend
description: Install or migrate Triton-Ascend 3.6 and maintain the minimal native NPU-IR/PTOAS bridge patch, with compiler-option parity checks and scoped simulator validation.
---

# Maintain Triton-Ascend Integration

Read [Planner's contract](../../../AGENT.md), its required reading and the
[current plan](../../../bridge/planning/README.md). For the known option-forwarding
gap and proposed redesign, read the Triton follow-up in the
[MmadL1 handoff](../../../bridge/memory/mmadl1-unit-flags-2026-10-21.md#separate-triton-ascend-follow-up).
That record is historical evidence, not a substitute for inspecting current code.

## Establish the Environment

- Use the published `Melikano/triton-ascend:feature/shared-a5-native-ptoas`
  baseline, validated at `461aa8f086c6bf26229ee324b8f4b5d3902e002d`; resolve its
  [source, setup and provenance](../../../bridge/memory/triton-ascend-3.6-shared-a5-2026-10-22.md#shared-source-baseline).
  Stock 3.6 alone does not include proof of our bridge patch. On this server the
  preferred installation is the existing 3.6 candidate; inspect it
  before proposing installation work. Preserve the old venv for rollback and
  coordinate ownership and user approval before modifying the shared candidate.
  Do not silently replace it or treat this server's path as portable to A5 hosts.
- Target Triton-Ascend **3.6**, not generic CUDA Triton with a similar version.
  Record the selected Python's `sys.executable`, `triton.__version__`,
  `triton.__file__`, backend file locations and `python -m pip` package metadata.
  Check the source/build revision and compiled-extension/launcher compatibility;
  copying Python 3.6-version source over 3.2.2 binaries is not an upgrade.
- If the assigned env is not 3.6, report the gap and establish the exact approved
  source/wheel, Python/CANN/Torch compatibility and destination before migration.
  Do not silently install a different build or call the old environment ready.
- Preserve a working environment and rollback diff/backups. Prefer an isolated
  candidate venv; do not replace another user's/shared env or install while
  workers use it. Python-only patches may avoid rebuilding only when compatible
  with the installed extension ABI; otherwise explain the required rebuild.
- Resolve `bishengir-compile` from the intended NPU-IR build/install, `hivmc-a5`
  from CANN and PTOAS from the checkout's pinned submodule build. Do not use
  Triton-bundled compiler copies or mix libraries from the two LLVM builds.
- Use **`--soc-version=Ascend950PR_9589`** for simulator validation. Prefer
  `Ascend950PR_9589` as the compiler target when accepted by the selected 3.6
  backend and compiler; inspect emitted commands, not only exported variables.
  Do not silently change the requested SoC to make a run pass.
- Inventory forced `TRITON_ASCEND_ARCH` values and target-name compatibility
  checks. Legacy `Ascend910_95...` spellings can designate A5, not a silent
  fallback from A3. Do not blanket-delete `910` handling or keep forcing an old
  alias without explanation. If the requested spelling is unsupported, report
  the exact restriction and ask before retaining an alias or changing setup.
  Removing existing alias-related patch hunks still requires user approval.

## Preserve and Adapt the Patch

- Inventory the current patch, local site-packages edits and Melika's source
  branch/revision before editing. Record the native base, bridge additions and
  local compatibility fixes separately; do not assume an old diff applies to 3.6.
- **Ask the user before deleting, reverting or disabling any existing patch
  hunk or behavior**, including apparently redundant compatibility code. Present
  the exact removal, reason, upstream replacement if any, and native/bridge
  impact. Dry-run rejection does not authorize dropping hunks or forcing a patch.
- Keep native compilation as the default and bridge selection explicit. Trace
  actual `add_stages()` registration and its callees: a native function's option
  forwarding does not help if the bridge never calls that function.
- Share applicable NPU-IR option construction. Audit unit flags, sync solver,
  buffering, precision/vectorization and other kernel controls; preserve None
  versus False/zero. Add only intentional VMI stop/output differences at that
  compiler boundary. Do not duplicate the full native binary function.
- Retain legitimate separate PTOAS invocation, binary/launcher metadata,
  workspace/lock handling, target checks and diagnostics. Document any native
  option intentionally omitted because the alternative backend cannot use it.
  Do not hide missing support by silently ignoring a requested control.
- Keep kernels backend-neutral: select the flow through the driver's options or
  environment, not workaround code inside each kernel. Do not edit NPU-IR/PTOAS
  passes, templates or pins to compensate for a driver defect without approval.

## Validate and Hand Off

1. For an authorized migration, establish a small native vadd baseline on the
   candidate 3.6 environment before applying our bridge patch. After patching,
   recheck native and then bridge on the same kernel/data/configuration.
2. Check emitted argument lists for explicit True, False and None, plus relevant
   numeric controls. Compare native/bridge options modulo documented differences.
   A vadd pass cannot prove unit flags: inspect actual command/IR controls and
   use an assigned paired MmadL1/Fixpipe case for that feature.
3. Use the patched `python kernel.py` entry point under `msprof op simulator` on
   this non-A5 server, with isolated logs/caches and pinned tools. Do not replace
   it with the legacy comparison runner or start a full matrix/model implicitly.
   Check child exit status, selected flow and numerical PASS, not profiler exit
   alone. Hand physical A5 validation to the human.
4. Return interpreter/package/source/patch versions, resolved tools and target,
   exact setup/apply/rollback steps, changed paths, native/bridge evidence and
   remaining limitations. Distinguish a local hot patch from a reproducible,
   committed installer/patch; do not claim portability from one environment.

This skill does not authorize installations, existing-patch removals, global
configuration changes, package uninstalls or commits/pushes by itself.
