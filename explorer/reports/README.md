# PTOAS State

Last updated: 2026-09-03T11:03:48+00:00

## Upstream reverts VPTO scheduler; VMI convert/layout patch lands in PR; L2-bypass tload branch advances; new cast/signless bug filed

- Upstream PTOAS head: 75e4a224 (2026-09-03). PR #1451 reverted PR #1310 (VPTO scheduler phase two). Pipeline returns to the pre-scheduler state; keep scheduler off/analysis-only for A5.
- Persistent fragment auto-promotion (PR #1341) is merged and documented; pass pto-promote-persistent-fragment-loops runs before pto-unroll-loops.
- VMI convert/layout: an upstream PR (#1452) preserves compact VL4 i8/i16→ui32 widening layout and clarifies signless-int convert uses unsigned semantics; a lit test was added.
- DMA: a branch adds an optional L2-bypass cache policy to pto.tload, using PTO-ISA L2 hint API (not yet merged upstream).
- Open issues affecting bridge: #1454 cross-width signedness cast (i64→si32) survives and fails LLVM translation; #1374 vmula accumulator misbinding; #1446 VPTO scheduler overflow (now mitigated by revert).

## Scan Coverage

- PTOAS Markham fork: 7 changed branches
- PTOAS Markham fork GitHub fork network: 25 changed branches
  - warning: compare Zhendong404/PTOAS: HTTP Error 404: Not Found
- AscendNPU-IR fork: 2 changed branches
- hw-native-sys/PTOAS: 8 updated issues, 22 updated PRs

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
