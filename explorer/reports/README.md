# PTOAS State

Last updated: 2026-08-20T11:03:31+00:00

## VMI layout rematerialization lands; VPTO address analysis PR; unified sync proposals; AscendNPU-IR bridges PTO via gated vendored dialect

Upstream PTOAS advanced to 54ba38da on upstream/main. Notable technical change: new VMI transform lib/PTO/Transforms/VMILayoutRematerializeWeakProducers.cpp with tests (test/lit/vmi_new/...). VMI version bumped to v0.1.5 then v0.1.6 via CI scripts; PTODSL gained per-element pto.Vec init=sequence (docs and tests updated). A3/A5 CI/docs updated to use an A3 self-hosted runner and TaskQueue. Multiple high-signal forks and PRs propose VMI fusion/VecScope MemBar, unified sync (buffer-id routing, determinism), PTODSL scalar/builtin unification, and VPTO reusable address analysis. AscendNPU-IR added a vendored PTO dialect snapshot and HIVM→PTO(VMI/VPTO) conversion passes; PTOAS bridge passes are now gated by an env flag.

## Scan Coverage

- PTOAS Markham fork: 9 changed branches
- PTOAS Markham fork GitHub fork network: 22 changed branches
- AscendNPU-IR fork: 5 changed branches
- hw-native-sys/PTOAS: 3 updated issues, 17 updated PRs
