# PTOAS State

Last updated: 2026-08-19T11:03:55+00:00

## PTOAS: broad core transforms landed; new VMILayoutRematerialize pass; CV pipelining design; scheduler/memplan/address-analysis PRs; large VMI predicate-fold work in forks

- Local Markham origin/main advanced to 38e7f151 (112 files, +5742/-819). Large surface across VMI/VPTO IR and Transforms; CI added A3/A5 board validation workflows; memplan design doc updated; numerous tests/docs touched.
- Upstream main advanced to 190d64b9. Added VMILayoutRematerializeWeakProducers pass, A3 self-hosted board tests, PTODSL per-element Vec init, ptobc support for tfree entries; multiple PTODSL fixes and test organization updates.
- New upstream doc branch upstream/codex/cv-pipelining-design created with a 1143‑line design for configurable Cube/Vector kernel pipelining.
- Fork network shows heavy movement: new VMI predicate-fold pass (multiple repos), VPTO scheduler refinements, VPTO address-analysis design/impl branches, unified sync modeling, and an A6 target fork.
- Open PRs concentrate on VMI–VF fusion and codegen pipeline (big ExpandTileOp rewrite), deterministic memplan priority, reusable VPTO address analysis, loop‑unroll hints, VecScope‑aware CSE, and CV pipelining design. VMILayoutRematerializeWeakProducers fix merged.

## Scan Coverage

- PTOAS Markham fork: 5 changed branches
- PTOAS Markham fork GitHub fork network: 21 changed branches
- AscendNPU-IR fork: 0 changed branches
- hw-native-sys/PTOAS: 8 updated issues, 22 updated PRs
