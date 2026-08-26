# PTOAS State

Last updated: 2026-08-26T11:02:47+00:00

## Upstream PTOAS lands A2/A3 gather in VPTO, loop-hint SCF→CF pass, and Python core DSO decouple; scheduler/tied-copy pipeline PRs advance; AVE→VMI expands in AscendNPU-IR

Upstream main moved to fc8db5e. Net new in-tree functionality: (1) A2/A3 VPTO gather lowering (tgather/tgatherb) with revised tgatherb contracts documented and negative/shape/layout checks; (2) SCF→CF conversion pass that preserves unroll hints plus LLVM metadata plumbing; (3) C/Python binding refactor that decouples the _core Python extension from the version-bound compiler DSO via C API coverage; (4) composed dense layout materialization support and VMI/VPTO test updates; (5) minor CI/simulator pin/TCI semantics/test tightenings. Core areas touched include include/PTO/IR (PTO, VPTOUbOps), lib/PTO/Transforms (LowerPTOToUBufOps, VPTOLLVMEmitter, new PTOConvertSCFToCFWithLoopHintsPass), lib/CAPI/Bindings, docs/designs, and lit/PTODSL tests.

## Scan Coverage

- PTOAS Markham fork: 3 changed branches
- PTOAS Markham fork GitHub fork network: 19 changed branches
- AscendNPU-IR fork: 3 changed branches
- hw-native-sys/PTOAS: 12 updated issues, 25 updated PRs
