# PTOAS Focused Re-Exploration

Collected: 2026-08-07 20:48:38 UTC

This is a Codex-led follow-up to the PTOAS Stage 2 snapshot. It did not use the OpenAI API key and it was prompted by the missed `WenboCodes/PTOAS:new-vf-fusion-design` branch.

## Method

- Re-read the approved Planner contract and branch triage policy.
- Scored branches across watched PTOAS repos, named forks, and active forks-of-forks.
- Inspected changed file lists, changed-line scale, branch names, and fork relationships.
- Fetched selected branch-local design docs from the strongest candidates.
- Checked the local PTOAS checkout to see which docs are present locally and which are branch-only.

## Limitation

The unauthenticated GitHub API hit a rate limit before the open-PR file-list pass completed. Branch scoring and selected branch-doc inspection completed, but PR-file re-triage should be rerun after rate reset or with an optional GitHub token.

This limitation does not invalidate the branch findings below, but it means the PR watch list should still be treated as the Stage 2 snapshot plus follow-up work, not as a fresh complete PR scan.

## Branch Candidates

| Candidate | Classification | Why it matters | Planner action |
| --- | --- | --- | --- |
| `WenboCodes/PTOAS:new-vf-fusion-design` | Investigate | Small branch, but very high design signal: 8 files under `docs/new-vf-fusion-design`, about 2k+ lines of VMI-level VF fusion design. | Keep as a standing watch item. Preserve loop/access/mask/accumulator-lifetime lessons in mapping work. |
| `zhendong404/PTOAS:tile-fusion-stage2` | Investigate as legacy context | Branch-only OpLib/tile-fusion docs describe concrete design problems around loop fusion, template lowering, register passing, and sync placement. | Do not treat as current source of truth. Use as design archaeology for hazards that current VMI/VPTO work still needs to avoid. |
| `zhendong404/PTOAS:tile-fusion-2` and related rewrite branches | Watch / legacy context | Similar tile-fusion material, including `pto.fusion_region`, scheduling, DFG/lifetime, and cost-model concerns. | Record lessons only; re-check only if revived by a current PR or issue. |
| `mouliangyu/PTOAS:vmi-per-block-cast` | Investigate | Contains branch-only `docs/designs/vmi-dialect-design.md`, with a clear VMI producer/layout contract. | Use to cross-check VMI mapping assumptions, especially "logical VMI before physical VPTO" boundaries. |
| `mouliangyu/PTOAS:vmi-examples` | Watch | Many VMI docs are already present in the local checkout, but this branch reinforces lane-stride/layout themes. | Treat as supporting context unless a diff shows new unmapped design material. |

## Key Findings

The stricter triage would have caught `WenboCodes/PTOAS:new-vf-fusion-design`. Its branch name, design-specific docs folder, relevant file names, and large coherent docs diff all score strongly even though the branch is not recent and has only two commits.

The zhendong tile-fusion branches are important but mostly legacy. Their docs use older concepts such as OpLib lowering, `PTOViewToMemref`, `pto.simd.*`, imported concrete templates, and EmitC-style bridging. Current Planner docs should continue to treat `ExpandTileOp`, PTODSL TileLib, VMI, and VPTO as the active center. The old branches still preserve useful hazards: tile/access facts are easiest to analyze before lowering erases them, UB store/load round-trips are performance-critical, register pressure can block fusion, and sync inside fused loops can destroy the win.

The mouliangyu VMI design branches align with the current VMI direction. The branch-only `vmi-dialect-design.md` is especially useful because it states the producer boundary clearly: after entering VMI, logical vector semantics should be represented as VMI ops or VMI compositions, while physical VPTO layout decisions belong to layout assignment and lowering.

The branch scoring produced some false-positive inflation. Old divergent branches can inherit many unrelated AI docs or stale generated files when compared against a modern default branch. The explorer should score all changed files, separate relevant source/docs from low-value AI/process docs, and record when a compare is heavily diverged.

## Design Lessons To Carry Forward

- Preserve loop structure, shaped access information, masks, and accumulator lifetime when choosing the NPU-IR interception point.
- Keep VMI producer output logical. Do not smuggle physical VPTO assumptions into a bridge pass too early.
- Treat sync and memory-planning ownership as explicit mapping-table columns.
- Reuse old tile-fusion docs as hazard context, not as current PTOAS architecture.
- Keep old but human-looking design branches visible even when they are outside the daily recency window.

## Explorer Implementation Follow-Up

- Store first-seen branch state with score, classification, changed files, and skip reason.
- Scan all changed file paths available from a compare, not only the first displayed sample.
- Downweight AI-prefixed branches and low-value paths unless they also touch relevant source or a live PR/issue.
- Mark heavily diverged compares as lower confidence until a better base or merge-base is found.
- Add optional authenticated GitHub API support so PR file-list collection is less likely to hit rate limits.

## Sources

- `https://github.com/WenboCodes/PTOAS/tree/new-vf-fusion-design/docs/new-vf-fusion-design`
- `https://raw.githubusercontent.com/WenboCodes/PTOAS/new-vf-fusion-design/docs/new-vf-fusion-design/RFC-vf-fusion-on-vmi.md`
- `https://raw.githubusercontent.com/WenboCodes/PTOAS/new-vf-fusion-design/docs/new-vf-fusion-design/PyPTO2-vf-fusion-analysis.md`
- `https://raw.githubusercontent.com/WenboCodes/PTOAS/new-vf-fusion-design/docs/new-vf-fusion-design/adr/0001-tile-shape-n-times-vl.md`
- `https://raw.githubusercontent.com/WenboCodes/PTOAS/new-vf-fusion-design/docs/new-vf-fusion-design/adr/0002-static-n-dynamic-valid-data-via-mask.md`
- `https://raw.githubusercontent.com/WenboCodes/PTOAS/new-vf-fusion-design/docs/new-vf-fusion-design/adr/0003-vload-vstore-multidim-index-shaped-ptr.md`
- `https://raw.githubusercontent.com/WenboCodes/PTOAS/new-vf-fusion-design/docs/new-vf-fusion-design/adr/0004-vmi-mem2reg-after-fusion.md`
- `https://raw.githubusercontent.com/zhendong404/PTOAS/tile-fusion-stage2/docs/tile_fusion/oplib_lowering_tile_fusion_design_v1.md`
- `https://raw.githubusercontent.com/zhendong404/PTOAS/tile-fusion-stage2/docs/tile_fusion/oplib_ir_spec.md`
- `https://raw.githubusercontent.com/zhendong404/PTOAS/tile-fusion-stage2/docs/tile_fusion/a5_oplib_v1_authoring.md`
- `https://raw.githubusercontent.com/zhendong404/PTOAS/rewrite/tile-fusion-2-pr-ready-20260319/docs/tile_fusion/tile_fusion_design_spec.md`
- `https://raw.githubusercontent.com/mouliangyu/PTOAS/vmi-per-block-cast/docs/designs/vmi-dialect-design.md`
- `https://raw.githubusercontent.com/mouliangyu/PTOAS/vmi-examples/docs/designs/vmi-lane-stride-generalization-design.md`
