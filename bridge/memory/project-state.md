# Project State

Last updated: 2026-08-12

## Current Goal

Create an open backend path from AscendNPU-IR through PTOAS/PTO-ISA, replacing the CCEC-style low-level backend segment where feasible.

## Current Working Hypothesis

- Main implementation should happen on the AscendNPU-IR side, not by modifying PTOAS.
- The bridge likely needs row-specific integration points, not one global pass boundary.
- Vector rows may fit before or around `convert-hivmave-to-ave-intrin`.
- DMA, cube, and sync rows likely need to be intercepted earlier around HIVM memory/sync planning and before `HIVMToStandard` loses structured operands to CCE-template/library-call lowering.
- PTO/VMI is expected to cover vector-side semantics.
- PTO tile abstractions or PTO-ISA may be needed for tile/cube/DMA behavior.

## Development Target

- Main local repo: `$HOME/AscendNPU-IR`
- Main fork: `https://gitcode.com/manisadati/AscendNPU-IR`
- Upstream source of truth: `https://gitcode.com/Ascend/AscendNPU-IR`
- PTOAS local checkout `$HOME/PTOAS/PTOAS_Markham` is not source of truth; its `origin` is a personal fork and `mani/fix_ptodsl` may be behind the active ecosystem.
- PTOAS design truth must come from upstream `hw-native-sys/PTOAS` plus active forks, branches, PRs, and issues.

## Build/Test Reality

- PTOAS can be built and run on this server.
- AscendNPU-IR can be coded on this server, but full A5 validation may require another server with actual A5 hardware.
- The full AscendNPU-IR Python/Triton lowering workflow should be treated as A5-machine-only. The Codex-accessible server can inspect code and analyze saved IR dumps, but should not assume it can lower Python/Triton examples locally.
- Triton source fixtures live in `bridge/triton-example/`; A5-generated early IR dumps should go under `bridge/examples/npuir-early-ir/`.
- The early-IR folder currently contains workflow documentation unless the human has added actual dumps. If real examples are needed, Codex should ask the human to generate early MLIR / early NPU-IR dumps on an A5 server and place them in Planner.
- Current A5-generated `*_kernel.mlir` files are dumps right after `AppendTargetDeviceSpec`; local replay from that boundary is planned in `bridge/planning/npuir-device-spec-replay.md` and scripted by `bridge/tools/replay_npuir_from_device_spec.sh`. The current replay endpoint is `convert-hivmave-to-ave-intrin`; compiler stages after that are not required for this bridge investigation. A nonzero replay exit caused by missing `hivmc-a5` is expected on the non-A5 server if the target-pass dump was captured.
- First local replay result is recorded in `bridge/memory/npuir-device-spec-replay-results-2026-08-11.md`. All six current examples reached `convert-hivmave-to-ave-intrin`; the endpoint confirms that DMA/cube are already helper/template calls by then, so the first bridge analysis/export pass should run earlier while structured HIVM ops are still available.
- Active implementation focus is DMA conversion through `dma_copy_kernel`. The conversion sweet spot has been selected: after `hivm-mark-disable-load` and before `convert-hivm-to-std`. The supporting source-backed trace is `bridge/memory/dma-copy-conversion-trace.md`.
- Next code step: add a dry-run AscendNPU-IR bridge/export pass on `mani/DMA` that records simple rank-1 contiguous `hivm.hir.load` GM->UB and `hivm.hir.store` UB->GM mappings, preserves surrounding explicit sync, and emits rejection reasons instead of changing production lowering.
- Expected workflow for A5-dependent validation: Codex edits/plans locally, the human runs on the A5 server, then returns logs/results for the next debugging pass.
- The A5 installation/runtime workflow is tracked in `bridge/memory/npu_ir_installation.md`; the non-A5 Codex-server build/replay workflow is tracked separately in `bridge/memory/npuir-codex-server-build.md`.
- Current Codex-server AscendNPU-IR build status: `$HOME/AscendNPU-IR` is on `mani/DMA` tracking `wilsoncxfeng/master` at `08031590`; the pinned LLVM submodule is present at `third-party/llvm-project`; the local Release build completed and installed `bishengir-compile` / `bishengir-opt`.

## Planner Status

- `AGENT.md` is approved as the current Codex contract.
- `bridge/planning/README.md` is the planning overview Codex should read at the start of each meaningful Planner task.
- `human/HighLevelOverview.md` is the human-owned project overview.
- `explorer/` is installed as a user systemd timer and scheduled daily at 7:00am Eastern time.
- Initial Codex-led exploration has started.
- Stage 1 local repo baseline is complete: see `bridge/planning/local-repo-baseline.md`.
- Stage 2 PTOAS context is complete enough for NPU-IR exploration: see `PTOAS/design/lowering-pipeline.md`, `PTOAS/design/ecosystem-inventory-2026-08-07.md`, `PTOAS/coding-guide/pipeline-and-validation.md`, and `explorer/reports/backfill/2026-08-07-ptoas-reexploration.md`.
- Stage 3 NPU-IR context has a local source-backed baseline: see `NPUIR/design/lowering-pipeline.md` and `NPUIR/coding-guide/repo-and-validation.md`.
- Stage 4 PTO-ISA context has a local source-backed baseline: see `PTO-ISA/design/virtual-isa-and-bridge-targets.md` and `PTO-ISA/coding-guide/repo-and-validation.md`.
- A5/early-IR workflow is tracked at `bridge/memory/a5-ir-workflow.md`.
- Codex-server NPU-IR build/replay workflow is tracked at `bridge/memory/npuir-codex-server-build.md`.
- First local-source-backed NPU-IR to PTOAS mapping draft exists at `bridge/planning/npuir-to-ptoas-mapping.md`; it still needs upstream/fork reconciliation and example IR dumps before implementation.
- DMA/template rewrite planning exists at `bridge/planning/dma-template-rewrite-plan.md`, with compact memory at `bridge/memory/dma-template-mapping.md`.
- Focused DMA-copy conversion exploration plan exists at `bridge/planning/dma-copy-conversion-exploration.md`.
- Focused DMA-copy conversion trace exists at `bridge/memory/dma-copy-conversion-trace.md`. It confirms low-level VPTO `pto.mte_gm_ub` / `pto.mte_ub_gm` plus explicit `pto.set_flag` / `pto.wait_flag` as the most concrete first PTOAS target for the simple DMA row.
- `soyu-wilson/AscendNPU-IR:codex/ave-to-vmi` has been reviewed as vector-pass prototype context. Do not continue it directly; port selected ideas into a fresh current-baseline branch if used. See `bridge/planning/soyu-wilson-ave-to-vmi-branch-review.md`.
- Latest configured-scope explorer lookback completed on 2026-08-11. Reports: `explorer/reports/README.md` and `explorer/reports/daily/2026-08-11.md`. The scan used the GitHub token and no longer hit the previous GitHub rate-limit failure.
- GitHub fork-discovery state was bootstrapped on 2026-08-10 for PTOAS:
  74 forks and 801 branch heads recorded, with no bootstrap errors.

## Latest Explorer/PTOAS State

As of the 2026-08-11 daily report and the 2026-08-10 four-day lookback, the configured watcher scope is current enough for near-term bridge planning. Scope covered configured PTOAS local/remotes, `hw-native-sys/PTOAS` issues/PRs, and local AscendNPU-IR branch tracking. GitHub direct-fork/fork-of-fork discovery is implemented and bootstrapped for configured GitHub repos. GitCode issue/PR tracking is still pending.

Latest PTOAS signals:

- upstream `hw-native-sys/PTOAS` main advanced with implicit tmp materialization, TFILLPAD unification, VPTO vscatter memory-effect fixes, and broad IR/emitter/test updates;
- `codex/downgrade-llvm19` and PR #1156 make LLVM 19 / VPTO `feature-vpto` the major toolchain watch item;
- PR #1204 and `codex/sync-block-interfaces` split cross-block and intra-block sync APIs;
- PR #1189 introduces PTO Common ops and `PTOLowerScalarToStandard`, which may affect bridge import/export assumptions;
- PR #1202 adds an analysis-only VPTO scheduler framework;
- issue #1200 and PR #1203 add explicit FP4 L1-to-L0 S4 staging;
- PR #1193 adds SoftLibService / late SoftLib expansion, while PR #1196 landed native integer vdiv TileLib support;
- local Markham fork `origin/main` now has in-process PTODSL materialization through TileLibService;
- local Markham branch `origin/elemntwise-1d-2d-versions` is a large elementwise 1D/2D TileLib refactor and should be treated as relevant but not authoritative source-of-truth.

Immediate review targets before implementation: LLVM19 environment alignment, sync API split, implicit tmp pass ordering, PTO Common ops, SoftLibService pass ordering, and VPTO scheduler implications.

## Open Technical Risks

- The DMA-copy interception point is confirmed for the first slice: after
  `hivm-mark-disable-load`, before `convert-hivm-to-std`. Other operation
  families still need their own boundary checks.
- PTOAS VMI/VPTO pipeline is active and moving. Current design centerpieces are `ExpandTileOp`, PTODSL TileLib expansion, VMI layout assignment, and `VMIToVPTO`.
- PTOAS local branch state must not be confused with upstream/fork design state. The 2026-08-10 configured-scope scan found active upstream/fork movement in LLVM19 migration, sync interface split, implicit tmp materialization, PTO Common ops, VPTO scheduler work, SoftLib, FP4 staging, VMI/TileLib, and PTODSL behavior.
- DMA and cube-template mappings may not be clean one-to-one mappings at the late HIVM-AVE level.
- Synchronization and memory-planning ownership between NPU-IR and PTOAS must be kept explicit.
- Performance parity is a requirement, not a nice-to-have.

## Current PTOAS Understanding

- PTOAS has `emitc` and `vpto` backend paths. The bridge should primarily reason about `vpto`.
- Current VPTO tile-op lowering crosses the `ExpandTileOp` boundary from tile-native PTO IR to VPTO-facing helper IR.
- PTODSL TileLib is the default VPTO tile-op expansion backend.
- `ExpandTileOp` does not necessarily mean "lower to VMI." Current helpers can produce existing VPTO/vector-style ops such as `pto.vlds`/`pto.vadd`/`pto.vsts`, while direct `pto.vmi.*` frontend/test paths also exist.
- The VPTO backend always runs a VMI semantic pipeline before physical VPTO emission, but that pipeline is substantive only when VMI ops are present after expansion/inlining or direct frontend generation.
- Direction-of-travel note: current upstream TileLib is not VMI-based by default, but branch evidence points toward future TileOp/PTODSL expansion into logical VMI so fusion and `mem2reg` happen before physical VPTO. For NPU-IR mapping, preserve loop/access/mask/accumulator facts for VMI even when current compile tests require lower VPTO/vector fallbacks.
- VMI represents logical vectors/masks; layout assignment owns physical register layout and mask granularity.
- `level2` allows PTOAS memory planning and optional auto-sync; `level3` skips memory planning and preserves explicit address/manual-sync ownership.
- The mapping table must include sync/memory ownership per row, because blindly mixing NPU-IR-authored sync with PTOAS auto-sync is risky.
- `WenboCodes/PTOAS:new-vf-fusion-design` is important branch-local design context for future VMI-level VF fusion. It strengthens the need to preserve shaped access, loop, mask, and accumulator-lifetime facts across the NPU-IR bridge.
- Older zhendong tile-fusion branches are useful as legacy hazard context, especially for UB handoff removal, fusion scheduling, sync placement, and register pressure, but their OpLib/`PTOViewToMemref`/EmitC-style details should not override current `ExpandTileOp`/PTODSL/VMI/VPTO evidence.
- Mouliangyu VMI branch-local docs reinforce that bridge output should preserve logical VMI semantics and avoid hard-coding physical VPTO layout before VMI layout assignment.

## Current NPU-IR Understanding

- Local NPU-IR repo is `$HOME/AscendNPU-IR`. It has `origin` pointing at the `manisadati` GitCode fork and a `wilsoncxfeng` remote pointing at `git@gitcode.com:wilsoncxfeng/AscendNPU-IR.git`.
- Current DMA development branch: `mani/DMA`, based on and tracking `wilsoncxfeng/master` as of commit `08031590`.
- Upstream source of truth is still `https://gitcode.com/Ascend/AscendNPU-IR`; compare with upstream before making compatibility claims.
- Regbase late lowering runs `convert-hivm-to-std`, then `convert-hivmave-to-std`, then `convert-hivmave-to-ave-intrin`.
- `lower-ave-pipeline` creates/optimizes HIVMAVE via `convert-vector-to-hivmave` and `convert-arith-to-hivmave` before final intrinsic lowering.
- The HIVM pipeline performs memory-scope inference, decomposition, data-layout inference, buffer sizing, memory planning, lower-to-loops, sync insertion/decomposition, memref-ext lowering, and FFTS metadata work before late lowering. Mapping rows should distinguish pre-memory-plan, post-memory-plan, post-sync, and final-conversion boundaries.
- `HIVMToStandard` dispatches to the regbase converter for regbased targets and rewrites many `hivm.hir.*` DMA, cube, vector, sync-lock, SIMT, and custom ops to library calls. That makes it a no-later-than boundary for many non-vector bridge rows.
- `convert-hivmave-to-ave-intrin` remains a plausible vector-side boundary, but it already makes hardware vector-length, predicate-width, and intrinsic-selection decisions. It is not the whole bridge boundary.
- `hivm.hir.mmadL1`/`mma*` are structured cube/template operations. `mmadL1` carries matrix operands, real `m/k/n`, L0C init, sync-related args, unit-flag mode, transpose/HF32/I4/bias attributes, then lowers toward `mma_tile` templates. It should not be treated as a VMI-only vector row.
- `hivm.hir.nd2nz` is GM-to-CBUF ND-to-NZ data movement with template-backed layout and copy-intrinsic behavior. It is closer to PTO tile/DMA/MTE mapping than to VMI arithmetic.
- DMA/template rewriting should start after `hivm-mark-disable-load` and before `convert-hivm-to-std` from structured HIVM DMA ops, not by parsing final CCE library-call names as the main source. First recommended proof of concept is strict GM->UB `hivm.hir.load` plus UB->GM `hivm.hir.store`, preserving explicit NPU-IR sync and emitting a mapping/export record before replacing any lowering path.
- In `dma_copy_kernel`, structured DMA turns into `load_gm_to_ubuf_1d_float` / `store_ubuf_to_gm_1d_float` at `convert-hivm-to-std`. Current regbase template evidence routes the contiguous C310/A5 path to `copy_gm_to_ubuf_align_v2` and `copy_ubuf_to_gm_align_v2`. PTOAS has corresponding low-level VPTO MTE wrappers and copy lowering.
- Sync ownership is an explicit bridge dimension. NPU-IR can decompose `sync_block` and generate `set_flag`/`wait_flag`/`sync_block_set`/`sync_block_wait` before final lowering; PTOAS level2 auto-sync versus level3/manual-sync must be chosen per row.
- Candidate operation families for the first mapping table: `ave.hir.vload`, `ave.hir.masked_store`, `ave.hir.pge`, `ave.hir.vf*`; `hivm.hir.load`, `hivm.hir.store`, `hivm.hir.nd2nz`, `hivm.hir.pointer_cast`, `hivm.hir.set_flag`, `hivm.hir.sync_block*`, and `hivm.hir.mmadL1`.
- The Soyu-Wilson `codex/ave-to-vmi` branch adds a narrow `HIVMAVEToVMI`
  prototype and useful unsupported-op diagnostics. It is not a full bridge:
  it covers only AVE/vector rows, uses a local textual bridge dialect, lacks
  PTOAS parse/verify/run tests, and does not address HIVM DMA/cube/sync rows.

## Current PTO-ISA Understanding

- Local PTO-ISA repo is `$HOME/pto-isa` on `master` at `896d8ec69aaf5b623fead5afcae7a657fa784a2b`; its remote is `git@gitcode.com:cann/pto-isa.git`.
- The local PTO-ISA worktree has an existing modified file, `include/pto/npu/a5/TBinOp.hpp`; do not revert or treat it as upstream truth without review.
- PTO-ISA is a tile-level virtual ISA with a three-layer contract: Virtual ISA semantics, AS representation, and backend lowering/legalization.
- Core contract dimensions are tile dtype, shape, valid region, location role, layout/fractal layout, GlobalTensor shape/stride/layout, events, and backend legality.
- PTO exposes bridge-relevant operation families for GM/tile movement (`TLOAD`, `TSTORE`), tile layout/resource movement (`TMOV`), cube/matmul (`TMATMUL*`), indexed movement (`MGATHER`, `MSCATTER`), events/pipe sync (`TSYNC`), and cross-core barrier (`SYNCALL`).
- A5 `TMATMUL` validates `Left`/`Right`/`Acc` tile roles and derives `m/k/n` from valid regions before calling `mad`; MX variants call `mad_mx`.
- A5 `TMOV` covers Mat-to-Left/Right/Bias/Scaling/ScaleLeft/ScaleRight, Acc-to-Vec/Mat, Vec-to-Vec/Mat, and Vec ND-to-NZ movement.
- A5 event logic maps PTO ops to hardware pipes and uses `set_flag`/`wait_flag` or intra-block sync depending on event type. NPU-IR sync mapping must match pipe pair, token lifetime, and participant semantics.
