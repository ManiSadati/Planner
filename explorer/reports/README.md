# PTOAS State

Last updated: 2026-08-11T14:03:33+00:00

## IR sync API alignment, VMI carry-chain ops, A5 SoftLib path, FP4 vbitcast; scalar-store lowering under investigation

PTOAS upstream is actively changing IR and lowering boundaries relevant to the bridge: named sync ops now align with PTODSL; VMI introduces carry-chain ops with VMI→VPTO support; an A5 PTODSL SoftLib expansion path for sin/cos and i32 vdiv is landing; VPTO now permits low-precision FP4 vbitcast to ui8; scalar-store lowering semantics are being corrected. Several items are large and need Codex backfill before the bridge can lock interfaces.

## Scan Coverage

- PTOAS Markham fork: 8 changed branches
- PTOAS Markham fork GitHub fork network: 10 changed branches
- AscendNPU-IR fork: 0 changed branches
- hw-native-sys/PTOAS: 3 updated issues, 19 updated PRs
