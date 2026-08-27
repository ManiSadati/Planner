# Project State

Last updated: 2026-08-27

## Current Goal

Create an open backend path from AscendNPU-IR through PTOAS/PTO-ISA, replacing the CCEC-style low-level backend segment where feasible.

## Current Milestone

- Vector conversion is complete enough for the current project stage. Row
  softmax and RMSNorm are supported for the accepted fixtures, with performance
  accepted as on par with the NPU-IR path.
- New vector instructions are maintenance work driven by concrete future
  kernels, not the active exploration target.
- Active focus is Cube plus its relevant DMA/staging sequence, starting from
  `bridge/triton-example/cube_dotproduct.py`.
- The first 64x64 Cube trace, mapping decision, and strict conversion slice are
  complete. The real fixture emits PTOAS VMI, lowers to VPTO, and passes PTOAS
  simulator numerical comparison for all 4096 f16 outputs.
- The next gate is a direct trace/performance comparison with the unchanged CCE
  simulator path, followed by A5 validation.

## Current Working Hypothesis

- Main implementation should happen on the AscendNPU-IR side, not by modifying PTOAS.
- The Wilson fork is the main bridge implementation source. Check which Wilson
  branch is newest/relevant before assuming `master` is current.
- The bridge likely needs row-specific integration points, not one global pass boundary.
- Vector rows may fit before or around `convert-hivmave-to-ave-intrin`.
- DMA, cube, and sync rows likely need to be intercepted earlier around HIVM memory/sync planning and before `HIVMToStandard` loses structured operands to CCE-template/library-call lowering.
- PTO/VMI is expected to cover vector-side semantics.
- The accepted softmax and RMSNorm fixtures provide good practical evidence for
  that vector-side strategy; remaining vector limitations stay explicit but do
  not block Cube exploration.
- Many DMA rows may map to concrete PTO dialect movement operations. Some DMA
  and cube/template rows may require rewriting NPU-IR template lowering to emit
  PTO-compatible operations rather than mapping a final CCE call one-to-one.
- PTO tile abstractions or PTO-ISA may be needed for tile/cube/DMA behavior.

## Development Target

- Main local repo: `$HOME/AscendNPU-IR`
- Main bridge implementation fork: `https://gitcode.com/wilsoncxfeng/AscendNPU-IR`
- Human personal fork: `https://gitcode.com/manisadati/AscendNPU-IR`
- Upstream source of truth: `https://gitcode.com/Ascend/AscendNPU-IR`
- PTOAS local checkout `$HOME/PTOAS/PTOAS_Markham` is not source of truth; its `origin` is a personal fork and `mani/fix_ptodsl` may be behind the active ecosystem.
- PTOAS design truth must come from upstream `hw-native-sys/PTOAS` plus active forks, branches, PRs, and issues.

## Build/Test Reality

- PTOAS can be built and run on this server.
- AscendNPU-IR can be coded on this server.
- The Codex-accessible server can run selected Triton/NPU-IR cases with the
  CANN A5 operator simulator. `vector_add_kernel` and
  `vector_add_large_kernel` are verified through local NPU-IR compilation and
  simulated execution with exact output. Real A5 validation and authoritative
  performance results still require A5 hardware.
- Triton source fixtures live in `bridge/triton-example/`; A5-generated early IR dumps should go under `bridge/examples/npuir-early-ir/`.
- The early-IR folder currently contains workflow documentation unless the human has added actual dumps. If real examples are needed, Codex should ask the human to generate early MLIR / early NPU-IR dumps on an A5 server and place them in Planner.
- Current A5-generated `*_kernel.mlir` files are dumps right after `AppendTargetDeviceSpec`; local replay from that boundary is planned in `NPUIR/coding-guide/device-spec-replay.md` and scripted by `NPUIR/tools/replay_npuir_from_device_spec.sh`. The current replay endpoint is `convert-hivmave-to-ave-intrin`; compiler stages after that are not required for this bridge investigation. A nonzero replay exit caused by missing `hivmc-a5` is expected on the non-A5 server if the target-pass dump was captured.
- First local replay result is recorded in `NPUIR/coding-guide/device-spec-replay-results-2026-08-11.md`. All six current examples reached `convert-hivmave-to-ave-intrin`; the endpoint confirms that DMA/cube are already helper/template calls by then, so the first bridge analysis/export pass should run earlier while structured HIVM ops are still available.
- The first DMA conversion through `dma_copy_kernel` is now supporting
  foundation rather than the active milestone. Its selected sweet spot remains
  after `hivm-mark-disable-load` and before `convert-hivm-to-std`; see
  `bridge/memory/dma-copy-conversion-trace.md`.
- AscendNPU-IR now has a standalone `convert-hivm-templates-to-pto` pass on
  `mani/DMA`.
  It converts contiguous rank-one GM->UB `hivm.hir.load` and non-atomic UB->GM
  `hivm.hir.store`, supports static or dynamic lengths, maps compatible
  `PadValue` loads, and emits explicit HIVM-to-PTO memory-space casts.
- The pass was verified on the real `dma_copy_kernel` dump after
  `hivm-mark-disable-load`. The generated ignored artifact is
  `bridge/examples/npuir-early-ir/replay/dma_copy_kernel/after-convert-hivm-templates-to-pto.mlir`;
  both PTO MTE operations remain valid through a following
  `convert-hivm-to-std` invocation.
- Current priority: compare the validated strict `cube_dotproduct.py` PTO
  composition with the unchanged CCE baseline, then generalize from concrete
  fixtures.
- The conversion is guarded and default-off, so the CCE path remains the
  fallback and comparison baseline.
- Expected workflow for A5-dependent validation: Codex edits/plans locally, the human runs on the A5 server, then returns logs/results for the next debugging pass.
- The A5 installation/runtime workflow is tracked in `NPUIR/coding-guide/a5-installation.md`; the non-A5 Codex-server build/replay workflow is tracked in `NPUIR/coding-guide/codex-server-build.md`; the simulator workflow is tracked in `NPUIR/coding-guide/simulator-workflow.md`.
- Temporary LLVM IR capture after `hivmc-a5` and before CCE `bisheng` is
  tracked in `NPUIR/coding-guide/llvm-ir-capture.md`. The current CANN 9.1 beta
  path requires watching the `-o` output directory for `kernel*.ll` while
  `bishengir-compile --save-linked-ir` is running.
- Current Codex-server AscendNPU-IR build status: `$HOME/AscendNPU-IR` has a
  working local Release build with `bishengir-compile`, `bishengir-opt`, and
  the C220/C310 template bitcode needed for end-to-end Triton/simulator
  compilation. Branch status must be checked before each code task because
  Wilson fork development may move across branches.

## Planner Status

- `AGENT.md` is approved as the current Codex contract.
- `bridge/planning/README.md` is the planning overview Codex should read at the start of each meaningful Planner task.
- `human/HighLevelOverview.md` is the human-owned project overview.
- `explorer/` is installed as a user systemd timer and scheduled daily at 7:00am Eastern time.
- Initial Codex-led exploration and baseline mapping are complete enough for
  operation-family implementation work.
- Stage 1 local repo baseline is complete: see `bridge/planning/local-repo-baseline.md`.
- Stage 2 PTOAS context is complete enough for NPU-IR exploration: see `PTOAS/design/lowering-pipeline.md`, `PTOAS/design/ecosystem-inventory-2026-08-07.md`, `PTOAS/coding-guide/pipeline-and-validation.md`, and `explorer/reports/backfill/2026-08-07-ptoas-reexploration.md`.
- Stage 3 NPU-IR context has a local source-backed baseline: see `NPUIR/design/lowering-pipeline.md` and `NPUIR/coding-guide/repo-and-validation.md`.
- Stage 4 PTO-ISA context has a local source-backed baseline: see `PTO-ISA/design/virtual-isa-and-bridge-targets.md` and `PTO-ISA/coding-guide/repo-and-validation.md`.
- A5/early-IR workflow is tracked at `NPUIR/coding-guide/a5-ir-workflow.md`.
- Codex-server NPU-IR build/replay workflow is tracked at `NPUIR/coding-guide/codex-server-build.md`.
- Codex-server A5 simulator workflow is tracked at `NPUIR/coding-guide/simulator-workflow.md`.
- The local-source-backed NPU-IR-to-PTOAS mapping draft exists at
  `bridge/planning/npuir-to-ptoas-mapping.md`. Vector and first DMA rows have
  concrete evidence; the strict first Cube row is now implemented and verified
  through VMI-to-VPTO lowering.
- DMA/template rewrite planning exists at `bridge/planning/dma-template-rewrite-plan.md`, with compact memory at `bridge/memory/dma-template-mapping.md`.
- Focused DMA-copy conversion exploration plan exists at `bridge/planning/dma-copy-conversion-exploration.md`.
- Focused DMA-copy conversion trace exists at `bridge/memory/dma-copy-conversion-trace.md`. It confirms low-level VPTO `pto.mte_gm_ub` / `pto.mte_ub_gm` plus explicit `pto.set_flag` / `pto.wait_flag` as the most concrete first PTOAS target for the simple DMA row.
- The vector milestone is complete for the current stage: accepted row-softmax
  and RMSNorm fixtures are supported with performance on par with NPU-IR.
- Active Cube planning is `bridge/planning/cube-conversion-exploration.md`, with
  compact memory in `bridge/memory/cube-conversion-status.md`.
- `soyu-wilson/AscendNPU-IR:codex/ave-to-vmi` has been reviewed as vector-pass prototype context. Do not continue it directly; port selected ideas into a fresh current-baseline branch if used. See `bridge/planning/soyu-wilson-ave-to-vmi-branch-review.md`.
- Latest configured-scope explorer report completed on 2026-08-26. Reports:
  `explorer/reports/README.md` and `explorer/reports/daily/2026-08-26.md`.
  The scan used the GitHub token and no longer hit the previous GitHub
  rate-limit failure.
- GitHub fork-discovery state was bootstrapped on 2026-08-10 for PTOAS:
  74 forks and 801 branch heads recorded, with no bootstrap errors.

## Latest Explorer/PTOAS State

As of the 2026-08-26 daily report, PTOAS remains active in areas that can
affect bridge assumptions. GitHub direct-fork and fork-of-fork discovery is
implemented; GitCode issue/PR tracking is still pending. Use
`explorer/reports/README.md` and `explorer/reports/daily/2026-08-26.md` for the
full current report.

Current signals most relevant to the Cube stage:

- VMI fusion and `ExpandTileOp` changes remain active, but Cube cannot be
  treated as a VMI-only mapping;
- VPTO scheduling, tied-copy materialization, pointer/alias modeling, and sync
  work may affect the eventual mixed Cube/DMA pipeline;
- ND-to-NZ extraction, layout inference, L1/L0 movement, and ConvTile work are
  relevant target-side signals for Cube staging and result movement;
- `TaoTao-real/PTOAS:feature-vmi` is important experimental vector-template
  evidence but is not upstream Cube implementation truth;
- bridge-specific PTOAS compatibility watches remain in
  `bridge/designs/ave-ptoas-vmi-compatibility-tracker.md`.

Before Cube implementation, re-check current PTOAS Cube/tile/DMA operations and
any open PRs that touch L1/L0 staging, ND2NZ, fixpipe, matmul, or sync.

## Open Technical Risks

- The DMA-copy interception point is confirmed for the first slice: after
  `hivm-mark-disable-load`, before `convert-hivm-to-std`. Other operation
  families still need their own boundary checks.
- PTOAS VMI/VPTO pipeline is active and moving. Current design centerpieces are `ExpandTileOp`, PTODSL TileLib expansion, VMI layout assignment, and `VMIToVPTO`.
- PTOAS local branch state must not be confused with upstream/fork design state. The 2026-08-10 configured-scope scan found active upstream/fork movement in LLVM19 migration, sync interface split, implicit tmp materialization, PTO Common ops, VPTO scheduler work, SoftLib, FP4 staging, VMI/TileLib, and PTODSL behavior.
- DMA and cube-template mappings may not be clean one-to-one mappings at the late HIVM-AVE level.
- Similar names are insufficient for Cube mapping. The actual NPU-IR CCE
  template implementation and the PTO operation contract must agree on layout,
  valid shape, accumulation, precision, movement, and sync semantics.
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

- Local NPU-IR repo is `$HOME/AscendNPU-IR`. It has `origin` pointing at the
  `manisadati` GitCode fork and a `wilsoncxfeng` remote pointing at
  `git@gitcode.com:wilsoncxfeng/AscendNPU-IR.git`.
- Wilson fork branches are the main implementation source. Check current branch
  freshness before using any specific branch as the base. `master` is normally
  expected to be current, but current bridge work may temporarily live on a
  different Wilson branch such as `melika/ave-to-vmi`.
- `soyu-wilson` prototype work is historical context only. Other Wilson fork
  branches may still be active and should not be treated as historical without
  checking current branch state.
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
