# PTOAS State

Last updated: 2026-08-10T18:50:12+00:00

## PTOAS pivots to LLVM 19, in‑process TileLib and implicit tmp land; sync interfaces split; VPTO scheduler framework introduced

Upstream PTOAS advanced substantially: a codex branch downgrades the toolchain to LLVM 19 (feature-vpto), upstream main merged implicit tmp materialization and TFILLPAD unification, and a feature branch splits cross- vs intra-block sync. New analysis-only VPTO scheduler framework PR was opened. Multiple VMI/TileLib fixes landed or are under review (integer vdiv native/SoftLib, scalar store handling, SCF while). The Markham fork synced the new in-process PTODSL TileLibService on main and is iterating a large Elementwise 1D/2D design branch. No AscendNPU-IR changes reported.

## Scan Coverage

- PTOAS Markham fork: 14 changed branches
- AscendNPU-IR fork: 0 changed branches
- hw-native-sys/PTOAS: 22 updated issues, 50 updated PRs
