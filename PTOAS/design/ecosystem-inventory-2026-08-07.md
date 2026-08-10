# PTOAS Ecosystem Inventory

Collected: 2026-08-07 20:14:33 UTC

Scope: initial Codex-led PTOAS Stage 2 inventory. This is not the full one-month backfill yet; it is the source-of-truth snapshot used before summarizing PTOAS design.

## Source-Of-Truth Rule

Do not use `/home/m84446336/PTOAS/PTOAS_Markham`, its `origin`, or local branch `mani/fix_ptodsl` as the authoritative PTOAS state.

For design truth, prefer:

- upstream: `https://github.com/hw-native-sys/PTOAS`
- upstream PRs/issues/branches
- active fork network, especially `mouliangyu/PTOAS` and forks of that fork
- local checkout only for currently available docs, code inspection, and build experiments

## Current Repo Snapshot

GitHub API snapshot:

| Repo | Relationship | Default | Last pushed | Forks | Open issues count | Notes |
| --- | --- | --- | --- | ---: | ---: | --- |
| `hw-native-sys/PTOAS` | upstream | `main` | 2026-08-07T09:16:00Z | 74 | 135 | Authoritative upstream repo. |
| `zhendong404/PTOAS` | fork of upstream | `main` | 2026-08-07T09:01:12Z | 0 | 1 | Active through PR heads, but default branch is far behind upstream. |
| `mouliangyu/PTOAS` | fork of upstream | `main` | 2026-08-07T10:17:13Z | 15 | 48 | Important fork, with its own fork network. |
| `WenboCodes/PTOAS` | fork of `mouliangyu/PTOAS` | `main` | 2026-07-08T04:26:56Z | 0 | 0 | Fork-of-fork; default branch diverged from upstream. |
| `PTO-ISA/PTOAS` | fork of upstream | `main` | 2026-03-30T09:00:22Z | 0 | 1 | Older fork; low current priority. |

Default branch heads queried separately:

| Repo/branch | HEAD date | HEAD summary |
| --- | --- | --- |
| `hw-native-sys/PTOAS main` | 2026-08-05T07:00:15Z | `fix(ptoas): reuse embedded compiler state safely` |
| `zhendong404/PTOAS main` | 2026-04-17T03:55:46Z | `Merge pull request #501...` |
| `mouliangyu/PTOAS main` | 2026-07-29T05:55:16Z | `fix/vmi-compact-reduction-store` merged |
| `WenboCodes/PTOAS main` | 2026-06-07T07:42:41Z | CI watchdog hardening |

Important inference: some repos have recent `pushed_at` because non-default refs are moving. Do not assume a fork default branch is current just because the repo was pushed today.

## Fork Network Watch

Most active direct forks of upstream by `pushed_at` included:

- `KurrinQu/PTOAS`
- `TaoTao-real/PTOAS`
- `afshinarefi/PTOAS_Markham`
- `mouliangyu/PTOAS`
- `FangRui0/PTOAS`
- `jimmychou0/PTOAS`
- `Zhendong404/PTOAS`
- `and0d0/PTOAS`
- `Likai-19/PTOAS`
- `HecreReed/PTOAS`

Active forks of `mouliangyu/PTOAS` included:

| Fork | Default branch | Last pushed | HEAD date | HEAD summary |
| --- | --- | --- | --- | --- |
| `liuzidi/PTOAS` | `feature-vpto-backend` | 2026-08-07T02:15:07Z | 2026-05-26T06:07:27Z | TileLang daemon RPC/stable key work |
| `TelGome/PTOAS` | `feature-vpto-backend` | 2026-08-07T01:27:47Z | 2026-05-16T06:12:50Z | HP support for TDiv |
| `peanutchan/PTOAS-official` | `feature-vpto-backend` | 2026-08-06T07:38:56Z | 2026-03-24T10:46:50Z | VPTO spec A5 merged ISA draft |
| `erhsh/PTOAS` | `feature-vpto-backend` | 2026-08-04T10:18:17Z | 2026-06-03T01:47:59Z | `tinsert` support |
| `sundyCoder/PTOAS` | `feature-vpto-backend` | 2026-07-28T09:00:08Z | 2026-06-07T07:42:41Z | CI watchdog hardening |
| `dj87-dot/PTOAS` | `feature-vmi` | 2026-07-14T07:02:50Z | 2026-07-14T06:35:40Z | rope kernel implementation fix |

Compare results:

- `zhendong404/main` is behind upstream `main` by 2146 commits.
- `mouliangyu/main` is behind upstream `main` by 401 commits.
- `WenboCodes/main` diverged: 8 commits ahead, 1364 behind upstream.
- active `feature-vpto-backend` forks of `mouliangyu` diverged heavily, with 125-315 commits ahead but 1861-2598 behind upstream depending fork.

Working interpretation: upstream `hw-native-sys/PTOAS` is the strongest current design signal, but divergent `feature-vpto-backend` branches can still contain older design context for VMI/VPTO/PTODSL. Explorer should track them as watch targets, not as automatic source of truth.

## WenboCodes `new-vf-fusion-design` Branch

Important correction: the first Stage 2 pass recorded `WenboCodes/PTOAS` as a fork-of-fork but did not inspect its `new-vf-fusion-design` branch. That branch contains design docs under:

```text
https://github.com/WenboCodes/PTOAS/tree/new-vf-fusion-design/docs/new-vf-fusion-design
```

Files present:

- `CONTEXT.md`
- `PTO-vmi-design.md`
- `PyPTO2-vf-fusion-analysis.md`
- `RFC-vf-fusion-on-vmi.md`
- `adr/0001-tile-shape-n-times-vl.md`
- `adr/0002-static-n-dynamic-valid-data-via-mask.md`
- `adr/0003-vload-vstore-multidim-index-shaped-ptr.md`
- `adr/0004-vmi-mem2reg-after-fusion.md`

Design signal:

- proposes reconstructing VF fusion on top of `pto.vmi`, with eligibility checks moved from physical `pto.mi` / LLVM IR into the logical VMI layer;
- treats TileOP as a template library over VMI ops, not a separate dialect;
- each TileOP expands into a self-contained `scf.for` over `N x VL` plus VMI ops;
- supports partial fusion: compatible loop groups can fuse while incompatible groups remain separate;
- proposes a VMI-level `mem2reg` pass after fusion to eliminate tileop-to-tileop UB `vstore`/`vload` round-trips;
- introduces shaped `!pto.ptr` plus multidimensional `vload`/`vstore` index expressions so access-pattern and alias checks retain structured information;
- separates static tile buffer capacity `N`, dynamic `valid row`, and lane-level tail masks;
- constrains reduce-friendly tile shape around dtype-dependent `VL`, with ColMax/RowMax tradeoffs left to cost modeling.

Bridge impact:

- This is not current upstream `main`, so do not treat it as implemented source of truth.
- It is highly relevant design context because it frames future PTOAS/VMI fusion around the same performance concerns we care about: avoiding UB round-trips, preserving structured access information, making masks first-class, and keeping physical layout decisions in `pto.as`.
- For the NPU-IR mapping table, this strengthens the need to preserve loop structure, shaped access information, masks, and reduce accumulator lifetime if we want PTOAS/VMI to recover deep fusion opportunities.

## Focused Re-Exploration Addendum

Collected: 2026-08-07 20:48:38 UTC.

After the missed Wenbo branch, Codex reran PTOAS branch triage using `explorer/docs/branch-triage-policy.md`. The follow-up confirmed that the new policy would surface `WenboCodes/PTOAS:new-vf-fusion-design` as an `Investigate` branch despite its age, because it has a human-looking branch name, a design-specific docs folder, relevant VMI/VF/fusion terms, and a large coherent docs diff.

The follow-up also found useful but older branch-local design context:

| Branch family | Status for Planner | Notes |
| --- | --- | --- |
| `WenboCodes/PTOAS:new-vf-fusion-design` | High-value watch item | Keep as future VMI-level fusion context. It is not current implementation truth, but its shaped pointer, multidimensional access, mask, `N x VL`, and VMI `mem2reg` ideas directly affect bridge performance planning. |
| `zhendong404/PTOAS:tile-fusion-stage2` and related tile-fusion branches | Legacy design archaeology | Contains branch-only OpLib/tile-fusion docs about `pto.fusion_region`, template lowering, scheduling, DFG/lifetime, UB handoff removal, and loop/sync hazards. Much of the concrete pipeline is older than current `ExpandTileOp`/PTODSL TileLib/VMI/VPTO architecture. |
| `mouliangyu/PTOAS:vmi-per-block-cast` | VMI contract context | Contains branch-only `vmi-dialect-design.md`, reinforcing that VMI producers should emit logical VMI semantics while layout assignment owns physical layout/VPTO lowering. |
| `mouliangyu/PTOAS:vmi-examples` | Supporting context | Reinforces lane-stride and layout themes; many related docs are already present in the local checkout. |

Explorer implementation lesson: old divergent branches can inflate scores because compare results include inherited AI docs, `openspec/`, `.claude/`, or other process files. Future explorer runs should score from the complete changed-file list, separate relevant source/design files from AI/process paths, record ahead/behind divergence, and classify old useful branches as "legacy context" rather than current design truth when appropriate.

## Recent Upstream Issues To Watch

Issue titles updated since 2026-07-07 that are design-relevant:

| Issue | State at collection | Why it matters to bridge |
| --- | --- | --- |
| `#1190` | open | TileOps/a5 code changes should auto-affect examples; relevant to local template experimentation. |
| `#1187` | open | `scf.while` + `scf.if` unsupported pattern; affects control-flow mapping from NPU-IR. |
| `#1185` | open | MX quant/TMOV movement; relevant to low precision and data movement. |
| `#1175` | open | PTODSL SIMT/scalar refactor; changes public scalar/SIMT semantics. |
| `#1183` | open | runtime `if` SSA merge bug; relevant to control-flow correctness. |
| `#1181` | open | `pto.vdiv` int32 support gap. |
| `#1179` | open | runtime float/integer scalar cast gap. |
| `#1060` | open | MXFP4 L1-to-L0 offset/scale stride bug; relevant to DMA/cube staging. |
| `#1148` | open | VMI vs ASC quant performance gaps. |
| `#1135` | open | dynamic VCI rematerialization/base offset bug. |
| `#1150` | open | VMI predicate fold/DCE for expert-pad masks. |
| `#1168` | open | mask register spill around vscatter. |
| `#1165` | open | invalid static-to-dynamic GlobalTensor cast for MGATHER/MSCATTER. |
| `#1143` | open | VPTO resource/register-pressure-aware scheduling framework. |
| `#1117` | open | SIMT FP32 scalar fastmath/divf differences. |

Closed but still informative:

- `#1149`: deadlock from flag hoisting out of `scf.if` plus `simt_entry noinline`.
- `#1152`: `pto.jit` kernel kind coverage bug.
- `#1046`: VMI PTODSL `vadd`/`vadds` interface unification.
- `#1021` / `#990`: HF32 mode support.
- `#1095`: PTODSL SSA leak across physical sections.
- `#1058`: VMI register type validation in PTODSL/TileLang frontends.

## Recent Upstream PRs To Watch

PR titles updated since 2026-07-07 that are design-relevant:

| PR | State at collection | Head | Why it matters |
| --- | --- | --- | --- |
| `#1188` | open | `mouliangyu:codex/fix-vscatter-cse-regression` | VScatter memory effects during lowering. |
| `#1189` | open | `mouliangyu:codex/unify-pto-scalar-surface` | PTODSL scalar interface and PTO dialect unification. |
| `#1140` | open | `and0d0:mte_l1_l0` | explicit L1-to-L0 loads; relevant to DMA/cube handoff. |
| `#1178` | open | `Likai-19:feature-vmi-vcvt-f4` | VMI `vcvt` support for f4x2. |
| `#1180` | open | `FangRui0:fix_issue1165_mgather_static_stride_cast` | MGATHER/MSCATTER lowering/type cast fix. |
| `#1182` | open | `jimmychou0:zjm/ptodsl-struct-member-access` | PTODSL structure surface. |
| `#1131` | open | `FangRui0:tmp_tile_memory_plan` | no-tmp frontend ops to PTO-ISA tmp-parameterized interfaces. |
| `#1156` | open | upstream branch `codex/downgrade-llvm19` | potential LLVM version/backend compatibility change. |
| `#1110` | open | `frank-deng:mscatter` | MSCATTER addition. |
| `#1177` | open | `mouliangyu:codex/issue-990-hf32-mode` | global HF32 mode config. |
| `#1158` | open | `frank-deng:tgather` | TGATHER compare support. |
| `#1137` | open | `Zhendong404:codex/issue-1126` | PTODSL boolean branch merge normalization. |
| `#1173` | open | `Zhendong404:codex/scf-while-support` | generic `scf.while`; relevant to NPU-IR control flow. |
| `#1174` | open | `Adamkh329:pr3b-unified-allocator` | allocator unifying bufid/event id. |
| `#1169` | open | `learning-chip:psum-issue` | VMI reproducer for prefix-sum `scf.while`. |
| `#1122` | open | `HecreReed:codex/unify-tfillpad-mode` | TFILLPAD and FP operand forms. |
| `#971` | open | `castigli:feature/a2a3-tgather-mgather-vpto` | A2/A3 gather lowering. |
| `#1164` | open | `peanutchan:proposal/vmi-assert-and-trap-llvm` | `pto.trap` and `pto.vmi.assert` LLVM lowering. |
| `#1147` | open | `learning-chip:zjw/quant_vf_parity` | quant performance gaps for VMI vs ASC. |
| `#1151` | open | `mouliangyu:feat/vmi-predicate-fold-main` | predicate/mask optimization. |
| `#1100` | open | `Adamkh329:pr1-affine-disjoint` | dependence removal by disjointness. |
| `#1027` | open | `HecreReed:codex/issue527-nz-layout-design` | NZ layout inference aligned with PTO-ISA 5D form. |

## Bridge-Relevant Takeaways

- PTOAS is actively changing exactly where this bridge is sensitive: VMI layout/lowering, masks, scalar surface, control flow, memory planning, sync, gather/scatter, and L1/L0 movement.
- `ExpandTileOp` and the VMI-to-VPTO semantic pipeline must be treated as current design centerpieces, not optional side details.
- WenboCodes' `new-vf-fusion-design` branch is a watch item for future VMI-level fusion design, especially shaped pointers, post-fusion mem2reg, and `N x VL` tile-shape constraints.
- For NPU-IR mapping, control-flow rows should not wait until implementation. Recent `scf.while` and `scf.if` bugs/PRs show this is a live edge.
- Predicate/mask mapping is higher risk than the first rough mapping suggested. VMI mask granularity, spill behavior, predicate fold, and vscatter effects are all active work.
- Memory planning/sync ownership must stay explicit. Recent allocator, bufid/event-id, flag-hoisting, and multi-buffer work can conflict with blindly preserving NPU-IR sync while also running PTOAS auto-sync.

## Sources

- GitHub API repo snapshots: `https://api.github.com/repos/hw-native-sys/PTOAS`, `https://api.github.com/repos/mouliangyu/PTOAS`, `https://api.github.com/repos/zhendong404/PTOAS`, `https://api.github.com/repos/WenboCodes/PTOAS`
- GitHub API fork snapshots: `https://api.github.com/repos/hw-native-sys/PTOAS/forks?per_page=100&sort=newest`, `https://api.github.com/repos/mouliangyu/PTOAS/forks?per_page=100&sort=newest`
- GitHub API issue snapshot: `https://api.github.com/repos/hw-native-sys/PTOAS/issues?state=all&since=2026-07-07T00:00:00Z&per_page=100&sort=updated&direction=desc`
- GitHub API PR snapshot: `https://api.github.com/repos/hw-native-sys/PTOAS/pulls?state=all&sort=updated&direction=desc&per_page=100`
- Compare checks: `https://github.com/hw-native-sys/PTOAS/compare/main...mouliangyu:main`, `https://github.com/hw-native-sys/PTOAS/compare/main...zhendong404:main`, `https://github.com/hw-native-sys/PTOAS/compare/main...liuzidi:feature-vpto-backend`
- WenboCodes VF fusion branch docs: `https://github.com/WenboCodes/PTOAS/tree/new-vf-fusion-design/docs/new-vf-fusion-design`
- Focused re-exploration report: `explorer/reports/backfill/2026-08-07-ptoas-reexploration.md`
- Zhendong tile-fusion design docs: `https://raw.githubusercontent.com/zhendong404/PTOAS/tile-fusion-stage2/docs/tile_fusion/oplib_lowering_tile_fusion_design_v1.md`, `https://raw.githubusercontent.com/zhendong404/PTOAS/rewrite/tile-fusion-2-pr-ready-20260319/docs/tile_fusion/tile_fusion_design_spec.md`
- Mouliangyu VMI dialect branch doc: `https://raw.githubusercontent.com/mouliangyu/PTOAS/vmi-per-block-cast/docs/designs/vmi-dialect-design.md`
