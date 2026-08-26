# PTOAS State

Last updated: 2026-08-26T11:02:47+00:00

## Upstream PTOAS lands A2/A3 gather in VPTO, loop-hint SCF→CF pass, and Python core DSO decouple; scheduler/tied-copy pipeline PRs advance; AVE→VMI expands in AscendNPU-IR

Upstream main moved to fc8db5e. Net new in-tree functionality: (1) A2/A3 VPTO gather lowering (tgather/tgatherb) with revised tgatherb contracts documented and negative/shape/layout checks; (2) SCF→CF conversion pass that preserves unroll hints plus LLVM metadata plumbing; (3) C/Python binding refactor that decouples the _core Python extension from the version-bound compiler DSO via C API coverage; (4) composed dense layout materialization support and VMI/VPTO test updates; (5) minor CI/simulator pin/TCI semantics/test tightenings. Core areas touched include include/PTO/IR (PTO, VPTOUbOps), lib/PTO/Transforms (LowerPTOToUBufOps, VPTOLLVMEmitter, new PTOConvertSCFToCFWithLoopHintsPass), lib/CAPI/Bindings, docs/designs, and lit/PTODSL tests.

## Scan Coverage

- PTOAS Markham fork: 3 changed branches
- PTOAS Markham fork GitHub fork network: 19 changed branches
- AscendNPU-IR fork: 3 changed branches
- hw-native-sys/PTOAS: 12 updated issues, 25 updated PRs

## Persistent Watch Context

### `TaoTao-real/PTOAS:feature-vmi`

Status manually verified on 2026-08-26:

- This is a direct fork of `hw-native-sys/PTOAS` and is now an explicit
  high-priority Explorer target. Explorer must query `feature-vmi` by name even
  when GitHub's normal per-fork branch limit would omit it.
- The fork's default `main` is not the important implementation line. The
  relevant integration branch is
  [`feature-vmi`](https://github.com/TaoTao-real/PTOAS/tree/feature-vmi), whose
  verified head was
  [`fd4568a`](https://github.com/TaoTao-real/PTOAS/commit/fd4568a2a4cf408c86215680a5ffe727ad273c79)
  from 2026-08-14.
- The branch contains the VMI TileLib work represented by
  [`ce4cc93`](https://github.com/hw-native-sys/PTOAS/commit/ce4cc93310615b4f5bad7dc9aa9e4d37d018afa1)
  and the later DSv4 lowering fixes in
  [`3948337`](https://github.com/TaoTao-real/PTOAS/commit/394833704c2c9411bf82eb912198e3af7422dcc5).
- It implements a real TileOp-to-VMI route. For a legal A5 vector candidate,
  `pto.tadds` can select the PTODSL `vmi_tadds` template, expand to VMI
  load/mask/`pto.vmi.vadds`/store operations, and then lower through VMI to
  physical VPTO. This is template selection and expansion, not a direct
  one-operation rename pass.
- This work is not in current `hw-native-sys/PTOAS:main`. At verification time,
  `feature-vmi` had 147 commits not in upstream main while upstream main had 795
  commits not in `feature-vmi`, so it is important implementation evidence but
  cannot be treated as current upstream source truth without a rebase or careful
  port.
- Current PyPTO emits tile-level PTO such as `pto.tadds`, but its normal PTOAS
  invocation does not enable this experimental VPTO/VMI route. A raw PyPTO
  `.pto` can exercise it through the fork's PTOAS pipeline; seamless PyPTO
  runtime integration remains separate work.

Explorer should report changes to `feature-vmi`, movement of this work into an
upstream PR/branch, or upstream changes that supersede its template-selection
and VMI-fusion design.

### Bridge-Owned PTOAS Compatibility Watches

The detailed source is
`bridge/designs/ave-ptoas-vmi-compatibility-tracker.md`. As of 2026-08-26, keep
these four PTOAS contract areas visible:

- signed integer VMI vector-scalar min/max and signless-to-unsigned
  normalization;
- reduction result shapes when AVE expects the original vector width;
- one-lane `pto.vmi.vload {dist_mode = "brc"}` layout/lowering support;
- one-point, singleton, or rank-zero VMI store support.

All four bridge workarounds remain active. Explorer should report proposed or
merged PTOAS support against the exact tracker row, but should call it
bridge-validated only after the corresponding NPU-IR conversion and PTOAS
lowering tests pass.
