# PTOAS State

Last updated: 2026-08-14T11:03:39+00:00

## VMI unaligned access lands (spec + lowering), VPTO address-recurrence normalization enabled, PTODSL surface unification and IR family split continue; large scheduler/docs drop on Zhendong’s branch

- Upstream/main advanced with broad C++ codecheck refactors; functional surface largely unchanged but many files touched.
- New VMI capability: continuous vload/vstore unaligned access documented and implemented in VMI→VPTO lowering (branch + PR #1260).
- VPTO pipeline change: address-recurrence normalization pass introduced/enabled ahead of soft post-update (PR #1244), with shared utils and substantial test coverage.
- PTODSL/IR evolution: unifying scalar/builtin vector/SIMT interfaces (PR #1189) and splitting PTO op families (PR #1233) remain active; forks mirror these efforts.
- Markham/Zhendong feature branch aggregates substantial scheduler scaffolding (VPTOScheduler headers) and SoftLib/ptodsl fixes alongside cast and VMI/V PTO cleanups.
- No new AscendNPU-IR changes.

## Scan Coverage

- PTOAS Markham fork: 8 changed branches
- PTOAS Markham fork GitHub fork network: 16 changed branches
- AscendNPU-IR fork: 0 changed branches
- hw-native-sys/PTOAS: 11 updated issues, 28 updated PRs
