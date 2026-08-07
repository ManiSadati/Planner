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

## Planner Status

- `AGENT.md` is approved as the current Codex contract.
- `human/HighLevelOverview.md` is the human-owned project overview.
- `explorer/` exists but has not run a real daily scan yet.
- Initial Codex-led exploration has started.
- Stage 1 local repo baseline is complete: see `bridge/planning/local-repo-baseline.md`.
- Source-backed NPU-IR to PTOAS mapping table has not been created yet.

## Open Technical Risks

- The exact NPU-IR interception point is not confirmed.
- PTOAS VMI/VPTO pipeline status needs fresh source inspection.
- DMA and cube-template mappings may not be clean one-to-one mappings at the late HIVM-AVE level.
- Synchronization and memory-planning ownership between NPU-IR and PTOAS must be kept explicit.
- Performance parity is a requirement, not a nice-to-have.
