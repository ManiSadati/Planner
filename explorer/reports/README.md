# PTOAS State

Last updated: 2026-08-25T11:02:51+00:00

## Upstream VMI API simplification lands; VPTO defaults change; TEXTRACT (ND→2×NZ) enters implementation; scheduler/stream-fusion PRs advance; AscendNPU-IR expands VMI lowering

- upstream/main advanced to 5139d22 with VMI memory API change: repeat_stride removed from VMI vload/vstore and legacy stride ops; lowering fixes physical repeat_stride=0 and uses resultless vsstb. PTODSL, parser/verifier, docs, and tests updated.
- VPTO soft post-update is now enabled by default (configurable flag remains). Address semantics headers/impl, analysis, and tests updated.
- Design-only TEXTRACT ND→2×NZ doc merged; implementation PR (stage 1) adds IR, validation, VPTO lowering, EmitC, tools, and tests.
- Active branches/PRs target VMI unaligned-stream handling and VPTO stateful stream fusion; a VPTO scheduler upgrade PR (phase two) is open and defaults A5 scheduling to on.
- Multiple DSL control-flow/unroll hint passes proposed (enable-hint path restored; auto-promotion of persistent loops).

## Scan Coverage

- PTOAS Markham fork: 4 changed branches
- PTOAS Markham fork GitHub fork network: 16 changed branches
- AscendNPU-IR fork: 3 changed branches
- hw-native-sys/PTOAS: 16 updated issues, 20 updated PRs
