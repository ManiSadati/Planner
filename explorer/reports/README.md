# PTOAS State

Last updated: 2026-08-21T11:03:01+00:00

## Upstream lands reusable VPTO Address Analysis; new Tile Fusion; forks push scheduler/sync/VMI gather; high-risk VMI unaligned refactor in-flight

Upstream PTOAS main advanced (e32488) to merge vexpdif fusion and, more importantly, closed PR #1290 to land reusable VPTO address/value-evolution analyses and a VPTOAddressSemantics op interface. Emitters and VPTOSoftPostUpdate were refactored to consume the analyses; tests and a print pass added. A large refactor branch (codex/vmi-unaligned-load-store) moved further with VMI address consumer unification and broad repo churn; requires backfill. Active PRs add a first-class VPTO scheduler analysis (PR #1310), broaden VMI vgather dtypes/widths (PR #1315), and extend PTODSL sync.set to PIPE_MTE1/PIPE_V (PR #1314). Fork network shows heavy scheduler-pressure and canonical sync reduction work plus gather/test realignments. New issues flag VMI lane-stride FP8 roundtrip semantics and backend stability. Bridge boundary changed at VPTO addressing; review required.

## Scan Coverage

- PTOAS Markham fork: 8 changed branches
- PTOAS Markham fork GitHub fork network: 19 changed branches
- AscendNPU-IR fork: 0 changed branches
- hw-native-sys/PTOAS: 10 updated issues, 22 updated PRs
