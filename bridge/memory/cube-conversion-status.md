# Cube Conversion Status

Last updated: 2026-08-26

## Current Milestone

The active bridge milestone is Cube compute plus the DMA/staging required by
Cube. The first fixture is `bridge/triton-example/cube_dotproduct.py`.

This stage is planning only. The fixture has not been modified and no new Cube
pipeline trace or conversion implementation has started.

## Completed Foundation

- Row softmax and RMSNorm are supported for the accepted bridge fixtures, with
  performance accepted as on par with NPU-IR.
- The AVE-to-VMI path now provides good practical vector coverage. Future
  vector work is incremental maintenance driven by concrete missing operations.
- Simple GM->UB and UB->GM mappings established the guarded NPU-IR-to-PTO
  conversion pattern and explicit-sync ownership model.

## Current Decision

Inspect the actual NPU-IR CCE template implementation before choosing a Cube
mapping:

1. If one PTO operation preserves the full template contract, use a direct
   mapping.
2. If several PTO operations preserve it, emit a tested PTO composition.
3. If CCE template expansion hides behavior or no equivalent exists, intercept
   the structured NPU-IR form and rewrite that template lowering in PTO dialect.
4. Keep the existing CCE path for unsupported rows and as the comparison
   baseline.

## Unknowns To Resolve

- Exact structured Cube op and selected template family for
  `cube_dotproduct.py`.
- Exact input DMA path, including whether GM->L1 uses normal movement or ND2NZ.
- L1->L0A/L0B staging contracts and tile layouts.
- Accumulator initialization, repeated accumulation, and precision modes.
- L0C result/fixpipe path and final destination.
- Sync, event, memory-placement, and address ownership across Cube and MTE
  pipelines.

## Next Action When Exploration Starts

Follow `bridge/planning/cube-conversion-exploration.md`: trace the real fixture,
locate every selected CCE template and implementation, then complete the
direct/composition/rewrite mapping table before coding.
