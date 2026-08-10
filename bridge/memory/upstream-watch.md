# Upstream Watch

Last updated: 2026-08-07

## Purpose

Track upstream and fork movement that can affect the NPU-IR-to-PTOAS bridge plan.

## Initial Exploration Ownership

The one-time initial exploration and one-month backfill are owned by Codex. They should not use the OpenAI API key.

## Daily Monitoring Ownership

The scheduled explorer agent will later run at 7:00am Eastern time, Monday through Saturday. It should use the OpenAI API key only for daily summarization of new changes since the last successful scan.

## PTOAS Tracking Targets

- `https://github.com/hw-native-sys/PTOAS`
- `https://github.com/zhendong404/PTOAS`
- `https://github.com/mouliangyu/PTOAS`
- `https://github.com/WenboCodes/PTOAS`
- forks of `https://github.com/mouliangyu/PTOAS`, especially active forks
- forks that are themselves forked from active forks, not only direct forks of upstream
- local fork: `$HOME/PTOAS/PTOAS_Markham`

## PTOAS Source-Of-Truth Rule

The local PTOAS checkout is useful for building, running, and inspecting available code, but it is not authoritative for design state. Its `origin` remote is a personal fork, not upstream, and the checked-out branch may be far behind the active PTOAS ecosystem.

Explorer/backfill work must compare upstream, active forks, forks-of-forks, branches, issues, and PRs before deciding whether a PTOAS change matters for the bridge plan.

## NPU-IR Tracking Targets

- upstream source of truth: `https://gitcode.com/Ascend/AscendNPU-IR`
- main development fork: `https://gitcode.com/manisadati/AscendNPU-IR`
- local fork: `$HOME/AscendNPU-IR`
- GitHub mirrors if found and useful

## Current Watch Status

- PTOAS Stage 2 snapshot produced: `explorer/reports/backfill/2026-08-07-ptoas-stage2-snapshot.md`.
- PTOAS focused re-exploration produced: `explorer/reports/backfill/2026-08-07-ptoas-reexploration.md`.
- PTOAS ecosystem inventory produced: `PTOAS/design/ecosystem-inventory-2026-08-07.md`.
- Explorer branch triage policy produced: `explorer/docs/branch-triage-policy.md`.
- No daily explorer report has been produced yet.
- No acknowledgement state exists yet because there has been no meaningful explorer report.

## Latest PTOAS Snapshot

Collected: 2026-08-07 20:14:33 UTC.

Current high-priority watch areas:

- upstream `hw-native-sys/PTOAS` is the strongest source-of-truth signal;
- `mouliangyu/PTOAS` remains important because it has 15 forks and active `feature-vpto-backend` descendants;
- active direct forks and fork-of-fork branches are moving, but several default branches are behind upstream, so compare branch heads before trusting fork design state;
- current upstream issues/PRs touch VMI/VPTO lowering, PTODSL scalar/control flow, sync/allocation, gather/scatter, L1/L0 loads, quant, and possible LLVM19/VPTO branch adaptation.
- `WenboCodes/PTOAS` branch `new-vf-fusion-design` contains important VMI-level VF fusion docs and should be tracked as design context even though it is not upstream `main`.
- older `zhendong404/PTOAS` tile-fusion branches contain useful legacy design context around fusion planning, OpLib/template lowering, UB handoff removal, sync placement, and register pressure, but should not be treated as current architecture without new upstream evidence.
- `mouliangyu/PTOAS:vmi-per-block-cast` contains branch-only VMI contract context; it reinforces logical VMI producer boundaries and explicit layout/lowering ownership.

Bridge-relevant live upstream PR/issue topics:

- `scf.while` / `scf.if` support and branch merge correctness;
- VMI predicate fold, mask granularity, mask spill behavior, and vscatter memory effects;
- PTODSL scalar interface unification and SIMT/scalar boundary cleanup;
- explicit L1-to-L0 loads and MX quant movement;
- allocator, bufid, event-id, and auto-sync behavior;
- gather/scatter and GlobalTensor cast legality.
- VMI-level VF fusion ideas from `WenboCodes/PTOAS:new-vf-fusion-design`: shaped pointers, multidimensional VMI load/store indices, post-fusion mem2reg, `N x VL` tile-shape constraints, and accumulator-lifetime/cost-model tradeoffs.

When the mapping table starts, re-check these topics if more than a few days have passed.

Known explorer limitation from focused re-exploration:

- unauthenticated GitHub API rate limiting blocked a fresh PR file-list pass;
- future explorer implementation should support an optional GitHub token and should record rate-limit failures explicitly rather than silently dropping PR triage.
