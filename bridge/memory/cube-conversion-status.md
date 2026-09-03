# Cube Conversion Status

Last updated: 2026-09-03

## Current Milestone

The active bridge milestone is Cube compute plus the DMA/staging required by
Cube. The first fixture is `bridge/triton-example/cube_dotproduct.py`.

The first fixture has been traced and validated through the two active routes:
preserved external CCE calls and the preferred PTODSL-template route. In the
PTODSL route, `bishengir-compile` imports pre-generated `MmadL1` and ND2NZ PTO
bodies, calls them from the converted kernel, and PTOAS lowers them successfully.
The 64x64 simulator comparison passes, and the `matmul_513` PTODSL fat object
passes numerical comparison on A5 hardware. Four additional 64x64 contracts
now pass local simulation: F16 B-transpose, BF16/F32 NN, signed INT8/INT32 NN,
and F32/HF32 NN. A-transpose, bias, further datatypes, and mixed-kernel
coverage remain pending.

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

- Make PTODSL/PTO-native expansion the intended default Cube architecture.
  Preserve structured shape/layout/init/dependency facts before
  `convert-hivm-to-std`, then let PTOAS see and optimize the generated PTO
  operations.
- Use the Python PTODSL implementation as the authoring source. Check in its
  explicit MLIR instantiations and load installed copies directly during
  compilation; do not launch Python or a service from `bishengir-compile`.
- Keep external CCE calls as a compatibility/reference route. They work, but
  their implementation remains a black box to PTOAS optimization.
- Do not maintain a second hand-written C++ implementation of the PTODSL Cube
  body. The earlier direct 64x64 rewrite remains historical mapping evidence,
  but its implementation has been removed.
- Do not use raw `!pto.ptr` arguments for CCE calls that consume memref
  descriptors. Let standard memref-to-LLVM lowering create the descriptors.
- Do not replace `mma_tile_half_to_float` or `hivm.hir.mmadL1` with one bare
  `pto.tmatmul`. The CCE template also performs L1-to-L0 staging, K partitioning,
  double buffering, sync, init/accumulate selection, and barriers.
- No new PTO primitive appears necessary for the starter fixture. Existing PTO
  operations cover ND2NZ, L1-to-L0A/B, MAD, and L0C-to-GM fixpipe movement.
- Do not claim that a standalone Python file is integrated. Success requires
  `bishengir-compile` to place its PTO semantics in the kernel module, followed
  by PTOAS lowering and numerical validation.

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
- The removed direct C++ proof passed all 4096 output elements with maximum
  absolute error `0.001953125`; its historical run reported 3250 total ticks.
- PTO's B-side L1-to-L0B `transpose = true` is required to form the physical
  right-tile `nZ` layout. It does not mean source-level `b_transpose=true`.

## PTODSL Template Path

Commit `9d97eff1240434e537e45ee9154c65df80208e2e` added:

```text
bishengir/lib/Template/lib/RegBase/Cube/nd2nz_mmadl1_64_ptodsl.py
bishengir/lib/Template/lib/RegBase/Cube/ptodsl_64x64_blockers.md
```

The source has been generalized from one fixed whole-kernel 64x64 case into a
normalized f16 microtemplate contract. It now accepts actual M/K/N values up to
64, caller-owned L1/L0 buffers, init versus accumulate, and NPU-IR event IDs.
It emits ND2NZ, L1-to-L0A/B, `pto.mad`/`pto.mad_acc`, and a probe-only f32-to-f16
writeback. A hidden non-square ND2NZ layout error was fixed by using the padded
K/N extent rather than the row count for the destination physical extent.

The source-contract checks pass for 64x64x64 initialization and 16x32x48
accumulation. In `--bridge-mode ptodsl`, NPU-IR parses the installed
`mmadl1_f16_f32_nn.mlir` and `nd2nz_f16_gm_l1.mlir` resources, imports
`@__pto_mmadl1_f16_f32_nn` and `@__pto_nd2nz_f16_gm_l1`, and replaces both
structured operations with normal internal calls. The VMI therefore contains
the calls and their visible PTO implementations. PTOAS inlines and lowers them to
`pto.load_cbuf_to_ca`, `pto.load_cbuf_to_cb`, and `pto.mad_raw`.

The latest PTODSL 64x64 simulator run passes all 4096 f16 outputs with maximum
absolute error `0.001953125` and reports 3265 total ticks. This proves only the
observed f16 64x64x64 initialization path.

The `matmul_513` PTODSL fat object also passes on A5. Its 81 logical output
programs and nine K steps validate odd whole-shape boundaries and the
initialize-then-accumulate sequence. NPU-IR pads each generated MMAD microtile
to 64, so this does not yet validate a helper call whose runtime M/K/N operand
is below 64.

Four aligned configuration probes were implemented on 2026-09-03. Each has a
separate pre-generated MMAD helper; BF16, INT8, and F32 additionally have typed
ND2NZ helpers. The bridge validates their A5 rank-4 Cube layouts, dispatches
typed Fixpipe mappings, and preserves signed INT8 across the rank-changing
Cube pointer view. All four emit VMI and VPTO, link, and pass the local
simulator: exact BF16 and INT32 matches, with zero maximum absolute difference
for F16 B-transpose and F32/HF32.

The F32/HF32 input still exposes an upstream frontend contract mismatch: the
captured Linalg op is emitted with `input_precison`, while NPU-IR consumes
`input_precision`. The tracked fixture uses the corrected spelling; regenerating
its early IR will restore the typo until the adapter or runner is fixed.

The multi-tile `q_kt_matmul` fixture also reaches VMI and VPTO in PTODSL mode.
Its one imported helper call remains inside the four-step K loop, and PTOAS
emits both initializing and accumulating `mad_raw` forms. The generated fat
object links and launches all 32 AIC blocks. The full 8192-logical-tile cycle
simulation did not complete during a five-minute smoke window, so its numerical
result remains for A5 hardware or a much longer simulator run.

PTOAS already supplies the model to follow in `lib/TileOps/a5/tmatmul.py`,
`tmatmul_acc.py`, `ptodsl/examples/fa_dn_matmul.py`, and the in-process Python
TileLib service in `tools/ptoas/NativeModule.cpp`.

## External-Call Experiment

`bridge/testcases/matmul_64/` is a copy of the same 64x64 source and early IR,
kept separate for the external-call compatibility experiment. Run it with:

```bash
bridge/tools/run_comparison_flow.sh \
  --bridge-mode external-calls emit-vmi matmul_64
bridge/tools/run_comparison_flow.sh \
  --bridge-mode external-calls emit-vpto matmul_64
bridge/tools/run_comparison_flow.sh \
  --bridge-mode external-calls bridge-sim matmul_64
```

Verified through 2026-08-28:

- the external mode skips `convert-hivm-templates-to-pto` but still runs the
  later VMI pass;
- VMI and VPTO preserve two `nd2nz_half` calls, one
  `mma_tile_half_to_float` call, and one
  `fixpipe_nz2nd_float_to_half_4d_to_2d_gm` call;
- all HIVM/AVE types and operations are removed, and the module is classified
  as a Cube module;
- the VMI kernel entry remains five raw GM pointers plus three scalars, while
  the external CCE declarations and calls retain ranked memrefs;
- standard PTOAS memref-to-LLVM lowering emits descriptor wrappers whose
  signatures and `_mlir_ciface_*` calls match the unchanged NPU-IR pre-CCE
  baseline;
- both the beta and CANN 9.x PTOAS LLVM emitters support the marked external
  path;
- `bridge-sim` supplies the installed NPU-IR AIC template bitcode to PTOAS,
  builds the fat object, and resolves all three CCE template symbols;
- simulator output passes all 4096 elements with maximum absolute error
  `0.001953125`; the external-call run reports 3647 total ticks.

The runner discovers
`$ASCEND_NPU_IR_ROOT/build/install/lib/meta_op.aic.c310.bc` and exports it as
`PTOAS_AICORE_LL_MODULE` only in external-call mode. PTOAS forwards it to Cube
Bisheng compilation with `-cce-link-aicore-ll-module`. Its
`_mlir_ciface_*` entry points receive the ranked memref descriptors created by
standard MLIR lowering; the public kernel entry remains raw pointers.

Main external-path risks are descriptor/address-space ABI mismatch, rejection
of memrefs by PTOAS stages, compatible device linking of the CCE object, MIX
packaging, and continued dependence on CCE templates. The PTODSL path avoids
that dependency, but must faithfully recreate template layouts, loops,
precision modes, buffering, and synchronization.

## Next Validation

- Construct one focused fixture that reaches `MmadL1` with a runtime M/K/N
  value below 64 instead of relying only on padded global boundaries.
- Keep the BF16, INT8, F32/HF32, and F16 B-transpose simulator regressions
  green. Resolve the upstream HF32 attribute spelling mismatch at the
  early-IR generation boundary.
- The multi-tile `q_kt_matmul` compile/build/launch path is complete, but its
  full numerical result still needs A5 hardware or a long simulator run.
- Generalize address/layout and sync ownership beyond the observed A5
  64x64x64 microtile before enabling PTOAS automatic allocation/scheduling.
- Compare the imported Python template against the selected CCE implementation
  for tails, K segmentation, transpose, bias, precision modes, and fallbacks.
- Compare PTODSL, external-call, and unchanged NPU-IR traces and ticks under
  identical simulator options.
- Validate a genuine split MIX fixture and run the resulting fat object on A5.

The detailed evidence, mapping table, risks, and staged implementation proposal
are in `bridge/planning/cube-conversion-exploration.md`.
