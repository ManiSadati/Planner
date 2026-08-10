# Upstream Watch

Last updated: 2026-08-10

## Purpose

Track upstream and fork movement that can affect the NPU-IR-to-PTOAS bridge plan.

## Initial Exploration Ownership

The one-time initial exploration and one-month backfill are owned by Codex. They should not use the OpenAI API key.

## Daily Monitoring Ownership

The scheduled explorer agent is installed as a user systemd timer and runs daily at 7:00am Eastern time. It uses the OpenAI API key only for daily summarization of new changes since the last successful scan. A read-only GitHub token is configured for reliable GitHub issue/PR collection.

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
- Explorer bot is installed as a user systemd timer: daily at 7:00am Eastern.
- Four-day configured-scope lookback report produced: `explorer/reports/daily/2026-08-10.md`.
- Latest generated overview: `explorer/reports/README.md`.
- GitHub token is configured; the previous unauthenticated PR file-list rate-limit failure did not recur on rerun.

## Latest PTOAS Snapshot

Collected: 2026-08-10 18:50:12 UTC by one-off four-day explorer lookback.

Configured-scope coverage:

- PTOAS Markham fork/remotes: 14 changed branches;
- `hw-native-sys/PTOAS`: 22 updated issues and 50 updated PRs;
- AscendNPU-IR local branch tracking: 0 changed branches.

Current high-priority watch areas:

- upstream `hw-native-sys/PTOAS` remains the strongest source-of-truth signal;
- upstream main advanced with implicit tmp materialization, TFILLPAD unification, VPTO vscatter memory-effect fixes, and broad IR/emitter/test changes;
- LLVM19 / VPTO `feature-vpto` is now a major environment watch item through `codex/downgrade-llvm19` and PR #1156;
- cross-block versus intra-block sync APIs are splitting through `codex/sync-block-interfaces` and PR #1204;
- PTO Common ops and `PTOLowerScalarToStandard` in PR #1189 may affect bridge op-surface assumptions;
- PR #1202 introduces an analysis-only VPTO scheduler framework;
- issue #1200 and PR #1203 add explicit FP4 L1-to-L0 S4 staging;
- TileLib/SoftLib movement remains active: PR #1196 landed native integer vdiv support and PR #1193 adds SoftLibService / late expansion;
- Markham `origin/main` has in-process PTODSL materialization through TileLibService;
- Markham `origin/elemntwise-1d-2d-versions` is a large elementwise 1D/2D TileLib refactor relevant to template selection and predicate behavior, but it is not authoritative upstream truth.

Standing branch-local design context remains important:

- `WenboCodes/PTOAS:new-vf-fusion-design` contains VMI-level VF fusion docs and should stay in the watch inventory;
- older `zhendong404/PTOAS` tile-fusion branches remain useful legacy hazard context, not current architecture;
- `mouliangyu/PTOAS:vmi-per-block-cast` reinforces logical VMI producer boundaries and layout/lowering ownership.

When implementation planning resumes, re-check these targets first: LLVM19 environment, sync API split, implicit tmp pass ordering, PTO Common ops, SoftLibService pass ordering, VPTO scheduler framework, and FP4 S4 staging.

Current explorer limitations:

- automatic direct-fork and fork-of-fork discovery is not implemented yet, though the policy caps future discovery at depth 2;
- GitCode issue/PR tracking for AscendNPU-IR is not implemented yet;
- daily explorer uses metadata-first summaries and does not deeply read every changed source file;
- deterministic PR/branch scoring is documented but not fully implemented before the OpenAI summary step.
