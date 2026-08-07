# PTOAS Stage 2 Snapshot

Collected: 2026-08-07 20:14:33 UTC

This is a Codex-led initial exploration snapshot. It did not use the OpenAI API key and is not a scheduled daily explorer run.

## Why This Snapshot Exists

The local PTOAS checkout is not source of truth. Before summarizing PTOAS design, Codex checked upstream GitHub metadata for repo/fork/issue/PR movement and then inspected local PTOAS docs/source for pipeline facts.

## Current External Signals

- `hw-native-sys/PTOAS` is authoritative upstream. It had 74 forks and repo activity on 2026-08-07.
- `mouliangyu/PTOAS` remains important because it has its own fork network: 15 forks, with several active `feature-vpto-backend` defaults.
- `zhendong404/PTOAS` is active through PR heads, but its default branch is behind upstream.
- `WenboCodes/PTOAS` is a fork-of-fork and should remain in the watch set. Its default branch was mainly CI divergence in this snapshot, but its `new-vf-fusion-design` branch contains important VMI-level VF fusion design docs.

## Design-Relevant Upstream Movement

Current issue/PR topics that can affect the NPU-IR bridge:

- VMI/VPTO lowering and layout: `vscatter`, gather/scatter, VCI rematerialization, FP4/f4x2 conversion, predicate folding, mask spill behavior.
- PTODSL surface: scalar unification, SIMT/scalar refactor, struct/member access, runtime casts, control-flow branch merge.
- Control flow: `scf.while`, `scf.if`, prefix-sum reproducer.
- Memory/sync: unified allocator, bufid/event-id design, flag-hoisting deadlock, explicit L1-to-L0 loads, memory planning around temp buffers.
- Compatibility: possible LLVM19 downgrade / VPTO branch adaptation.

## PTOAS Pipeline Facts Learned

- `ExpandTileOp` is the current hard boundary from tile-native PTO IR into VPTO-facing helper IR.
- PTODSL TileLib is the default VPTO tile-op expansion backend.
- The VPTO backend always appends the VMI semantic pipeline before VPTO emission.
- VMI represents logical vectors/masks. Layout assignment decides physical register layout; `vmi-to-vpto` consumes assigned layouts and must not infer hidden context.
- `level3` skips `PlanMemory` and preserves explicit address/manual-sync ownership.
- auto-sync is selectable between InsertSync, BufidSync, barrier-all, and GraphSyncSolver; sync runs before buffer selection so it can see multi-tile slot identity.

## Correction From Human Verification

The first Stage 2 pass did not inspect `WenboCodes/PTOAS` branch `new-vf-fusion-design`.

That branch was checked after human review. Its docs propose VMI-level VF fusion plus post-fusion `mem2reg`, with shaped pointers, multidimensional `vload`/`vstore`, static `N` vs dynamic `valid row`, lane-tail masks, and `N x VL` tile-shape constraints for reduce/elementwise fusion.

This does not replace upstream `main` as source of truth, but it is an important watch item for performance planning and for preserving structured loop/access/mask information in the NPU-IR bridge.

## Planner Docs Updated By This Snapshot

- `PTOAS/design/ecosystem-inventory-2026-08-07.md`
- `PTOAS/design/lowering-pipeline.md`
- `PTOAS/coding-guide/pipeline-and-validation.md`
- `bridge/memory/upstream-watch.md`
- `bridge/memory/project-state.md`
- `bridge/planning/initial-exploration-plan.md`

## Follow-Up For Initial Exploration

Next should be Stage 3: NPU-IR design context. That stage should confirm exactly what `HIVMToStandard` and `convert-hivmave-to-ave-intrin` consume/produce before we create the mapping table.

## Sources

- `https://github.com/hw-native-sys/PTOAS`
- `https://github.com/hw-native-sys/PTOAS/issues`
- `https://github.com/hw-native-sys/PTOAS/pulls`
- `https://github.com/hw-native-sys/PTOAS/branches`
- `https://github.com/mouliangyu/PTOAS`
- `https://github.com/WenboCodes/PTOAS/tree/new-vf-fusion-design/docs/new-vf-fusion-design`
- local PTOAS checkout: `/home/m84446336/PTOAS/PTOAS_Markham`
