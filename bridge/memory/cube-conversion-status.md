# Cube Conversion Status

Last updated: 2026-08-27

## Current Milestone

The active bridge milestone is Cube compute plus the DMA/staging required by
Cube. The first fixture is `bridge/triton-example/cube_dotproduct.py`.

The first fixture has been traced, classified, and implemented as a strict
compiler-side slice. It reaches PTOAS VMI and VPTO; simulator and A5 runtime
equivalence against the Triton-equivalent numerical reference now passes. A5
hardware validation remains pending.

## Completed Foundation

- Row softmax and RMSNorm are supported for the accepted bridge fixtures, with
  performance accepted as on par with NPU-IR.
- The AVE-to-VMI path now provides good practical vector coverage. Future
  vector work is incremental maintenance driven by concrete missing operations.
- Simple GM->UB and UB->GM mappings established the guarded NPU-IR-to-PTO
  conversion pattern and explicit-sync ownership model.

## Confirmed Fixture Path

After `hivm-mark-disable-load`, the 64x64 f16 fixture contains:

```text
2 x hivm.hir.nd2nz (GM ND -> CBUF/L1 NZ)
  -> hivm.hir.mmadL1 (f16 x f16 -> f32 CC/L0C, init=true)
  -> hivm.hir.fixpipe (NZ2ND, F322F16, CC/L0C -> GM)
```

NPU-IR has already assigned ping-pong addresses and explicit MTE/Cube/FIX
events. `convert-hivm-to-std` then selects `nd2nz_half`,
`mma_tile_half_to_float`, and
`fixpipe_nz2nd_float_to_half_4d_to_2d_gm`.

The current A5 build gets these device implementations from
`bishengir/lib/Template/lib/RegBase/Cube/`, including the CATLASS-based
`LocalMmad.cpp`. The similarly named older generic template directories are
not the selected A5 source for this fixture.

## Current Decision

- Do not extend `convert-hivmave-to-ptoas-vmi` for Cube. Its rank-4 pointer
  rejection is expected because that pass is vector-specific.
- Do not replace `mma_tile_half_to_float` or `hivm.hir.mmadL1` with one bare
  `pto.tmatmul`. The CCE template also performs L1-to-L0 staging, K partitioning,
  double buffering, sync, init/accumulate selection, and barriers.
- No new PTO primitive appears necessary for the starter fixture. Existing PTO
  operations cover ND2NZ, L1-to-L0A/B, MAD, and L0C-to-GM fixpipe movement.
- The first code route extends `convert-hivm-templates-to-pto` with a strict
  low-level PTO composition while preserving NPU-IR memory and sync ownership.
- Put Cube conversion code in a new
  `bishengir/lib/Conversion/HIVMTemplatesToPTO/Cube/` subdirectory. It should
  emit PTO MLIR directly and must not be added to the CCE
  `bishengir/lib/Template/` tree.
- Longer term: represent the region with `pto.tload -> pto.textract ->
  pto.tmatmul -> pto.tstore` and let PTOAS own tile allocation and sync.

## Conversion Point

Use the existing bridge slot after `hivm-mark-disable-load` and sync-block
finalization, immediately before `convert-hivm-to-std`. This is the last point
where the three structured operations, memory roles, layouts, addresses, and
events coexist.

## Verified Output

- Cube code lives in
  `$HOME/AscendNPU-IR/bishengir/lib/Conversion/HIVMTemplatesToPTO/Cube/`.
- Focused lit tests cover atomic PTO emission, rejection of an unsupported
  transpose form, and continuation through the AVE-to-VMI stage.
- The real fixture emits two `pto.mte_gm_l1_frac` ops, L1-to-L0A/B movement,
  `pto.mad`, and `pto.mte_l0c_gm` with Cube kernel metadata.
- PTOAS successfully lowers the result to VPTO containing
  `pto.load_cbuf_to_ca`, `pto.load_cbuf_to_cb`, and `pto.mad_raw`.
- The simulator fixture matches the Triton contract: row-major 64x64 f16 A/B,
  f32 accumulation rounded to an f16 output, and one launched program.
- The clean bridge simulator run passes all 4096 output elements with maximum
  absolute error `0.001953125`; the run reports 3250 total ticks.
- PTO's B-side L1-to-L0B `transpose = true` is required to form the physical
  right-tile `nZ` layout. It does not mean source-level `b_transpose=true`.

## Next Validation

- Run the unchanged NPU-IR/CCE simulator path on the same fixture for direct
  trace and performance comparison.
- Validate preserved addresses and explicit event ordering.
- Run the accepted path on A5 hardware through the human validation loop.
- Generalize only from concrete fixtures; all non-matching modes retain the CCE
  path.

The detailed evidence, mapping table, risks, and staged implementation proposal
are in `bridge/planning/cube-conversion-exploration.md`.
