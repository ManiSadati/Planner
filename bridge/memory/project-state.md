# Project State

Last updated: 2026-08-07

## Current Goal

Create an open backend path from AscendNPU-IR through PTOAS/PTO-ISA, replacing the CCEC-style low-level backend segment where feasible.

## Current Working Hypothesis

- Main implementation should happen on the AscendNPU-IR side, not by modifying PTOAS.
- The likely integration point is before `convert-hivmave-to-ave-intrin`.
- Some patterns may need to be intercepted earlier, possibly around `HIVMToStandard`, especially if loads/stores, DMA, or cube templates lose useful structure after CCE-template lowering.
- PTO/VMI is expected to cover vector-side semantics.
- PTO tile abstractions or PTO-ISA may be needed for tile/cube/DMA behavior.

## Development Target

- Main local repo: `/home/m84446336/AscendNPU-IR`
- Main fork: `https://gitcode.com/manisadati/AscendNPU-IR`
- Upstream source of truth: `https://gitcode.com/Ascend/AscendNPU-IR`
- PTOAS local checkout `/home/m84446336/PTOAS/PTOAS_Markham` is not source of truth; its `origin` is a personal fork and `mani/fix_ptodsl` may be behind the active ecosystem.
- PTOAS design truth must come from upstream `hw-native-sys/PTOAS` plus active forks, branches, PRs, and issues.

## Build/Test Reality

- PTOAS can be built and run on this server.
- AscendNPU-IR can be coded on this server, but full A5 validation may require another server with actual A5 hardware.
- Expected workflow for A5-dependent validation: Codex edits/plans locally, the human runs on the A5 server, then returns logs/results for the next debugging pass.

## Planner Status

- `AGENT.md` is approved as the current Codex contract.
- `human/HighLevelOverview.md` is the human-owned project overview.
- `explorer/` exists but has not run a real daily scan yet.
- Initial Codex-led exploration has started.
- Stage 1 local repo baseline is complete: see `bridge/planning/local-repo-baseline.md`.
- Stage 2 PTOAS context is complete enough for NPU-IR exploration: see `PTOAS/design/lowering-pipeline.md`, `PTOAS/design/ecosystem-inventory-2026-08-07.md`, and `PTOAS/coding-guide/pipeline-and-validation.md`.
- Source-backed NPU-IR to PTOAS mapping table has not been created yet.

## Open Technical Risks

- The exact NPU-IR interception point is not confirmed.
- PTOAS VMI/VPTO pipeline is active and moving. Current design centerpieces are `ExpandTileOp`, PTODSL TileLib expansion, VMI layout assignment, and `VMIToVPTO`.
- PTOAS local branch state must not be confused with upstream/fork design state. The 2026-08-07 ecosystem snapshot found active upstream PRs/issues in VMI, VPTO, PTODSL scalar/control flow, sync, memory planning, gather/scatter, and L1/L0 movement.
- DMA and cube-template mappings may not be clean one-to-one mappings at the late HIVM-AVE level.
- Synchronization and memory-planning ownership between NPU-IR and PTOAS must be kept explicit.
- Performance parity is a requirement, not a nice-to-have.

## Current PTOAS Understanding

- PTOAS has `emitc` and `vpto` backend paths. The bridge should primarily reason about `vpto`.
- Current VPTO tile-op lowering crosses the `ExpandTileOp` boundary from tile-native PTO IR to VPTO-facing helper IR.
- PTODSL TileLib is the default VPTO tile-op expansion backend.
- The VPTO backend always runs a VMI semantic pipeline before physical VPTO emission.
- VMI represents logical vectors/masks; layout assignment owns physical register layout and mask granularity.
- `level2` allows PTOAS memory planning and optional auto-sync; `level3` skips memory planning and preserves explicit address/manual-sync ownership.
- The mapping table must include sync/memory ownership per row, because blindly mixing NPU-IR-authored sync with PTOAS auto-sync is risky.
- `WenboCodes/PTOAS:new-vf-fusion-design` is important branch-local design context for future VMI-level VF fusion. It strengthens the need to preserve shaped access, loop, mask, and accumulator-lifetime facts across the NPU-IR bridge.
