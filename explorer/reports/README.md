# PTOAS State

Last updated: 2026-08-12T11:04:02+00:00

## PTOAS: vcvt rounding reverted/reopened; LLVM19 downgrade branch moved; FP4 S4 staging + TPRINT merged; new VMI fused ops; big MX design drop; AscendNPU-IR DMA refactor

Upstream main integrated TPRINT and explicit FP4 L1->L0 S4 staging; VMI carry-chain ops are in tree while a related carry-layout fix was reverted. The float->int vcvt rounding fix (#585) was merged then reverted and is now reopened with dedicated tests, so semantics remain in flux. A large LLVM19 downgrade branch is active. New VMI fused vexpdif and packed group-slot widening fixes landed in branches. PTODSL gained generic scf.while; a VPTO ui32 arith.select legalization fix is in flight. MX grpAxis/exponent ZZ design and tests landed on a codex branch. AscendNPU-IR merged a very large DMA/NDDMA refactor and numerous passes.

## Scan Coverage

- PTOAS Markham fork: 18 changed branches
- PTOAS Markham fork GitHub fork network: 20 changed branches
- AscendNPU-IR fork: 2 changed branches
- hw-native-sys/PTOAS: 11 updated issues, 37 updated PRs
