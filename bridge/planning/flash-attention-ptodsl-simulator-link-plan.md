# Flash-Attention PTODSL Simulator Link Plan

Last updated: 2026-09-03

## Status

The `flash_atten` testcase successfully lowers through the NPU-IR bridge to
PTOAS VMI and then through PTOAS to VPTO. Simulator device-object generation
currently fails at fat-object link time because several legacy C-interface
helpers are declared and called but are not defined by any linked object.

This is not a failure to propagate `--bridge-mode ptodsl` through the
comparison flow. It is an incomplete PTODSL replacement boundary for this
mixed Cube/Vector kernel.

## Observed Failure

The failing log is:

```text
bridge/testcases/flash_atten/out/build/bridge-sim.log
```

The linker reports unresolved `_mlir_ciface_*` symbols for:

```text
copy_ubuf_to_ubuf_1d_float
load_gm_to_ubuf_2d_half
copy_ubuf_to_cbuf_2d_half
mma_tile_half_to_float
fixpipe_nz2nd_float_to_float_4d_to_2d_ubuf
```

The generated VMI and VPTO contain private `func.func` declarations for these
helpers with `llvm.emit_c_interface`, but no function bodies. PTOAS therefore
emits calls to the corresponding `_mlir_ciface_*` ABI symbols. The fat-object
link cannot resolve those calls because PTODSL mode does not link the NPU-IR
AICore bitcode fallback.

## Verified Mode Behavior

The following evidence confirms that PTODSL mode is active:

- both `flash_atten.vmi.bridge-mode.txt` and
  `flash_atten.vpto.bridge-mode.txt` contain `ptodsl`;
- VMI-to-VPTO lowering is invoked with `--pto-level=level3`;
- generated VMI contains imported PTODSL helper bodies such as
  `__pto_nd2nz_f16_gm_l1` and `__pto_mmadl1_f16_f32_nn`;
- `bridge-sim` consumes the existing mode-matched
  `flash_atten.vpto.mlir`; it only reruns VPTO emission when that artifact or
  its mode marker is absent or mismatched.

The option is therefore working as currently implemented. It replaces only
the helper forms covered by the bridge's PTODSL import and call-rewrite logic.
The five unresolved helpers remain outside that coverage.

## Required Resolution

The preferred fix is to eliminate each remaining external declaration in the
AscendNPU-IR bridge output by mapping the source operation or legacy helper
call to PTO operations or an imported, pre-generated PTODSL helper body. PTOAS
files must remain unchanged.

Handle one helper family at a time:

1. Trace each call back to the structured NPU-IR operation and template that
   produced it.
2. Record its complete contract: memory spaces, element types, dimensions,
   layout, padding, strides, synchronization ownership, and kernel section.
3. Determine whether existing PTO operations express the contract directly.
4. For multi-operation behavior, add a generated PTODSL instantiation and
   import it using the established bridge mechanism rather than duplicating
   the template body in C++.
5. Rewrite the legacy call to the PTO/PTODSL form and remove its declaration
   only when no calls remain.
6. Add focused conversion tests and then rerun the complete flash-attention
   VMI, VPTO, fat-object, simulator, and numerical-comparison flow.

Suggested investigation order:

1. `copy_ubuf_to_ubuf_1d_float`
2. `load_gm_to_ubuf_2d_half`
3. `copy_ubuf_to_cbuf_2d_half`
4. `mma_tile_half_to_float`
5. `fixpipe_nz2nd_float_to_float_4d_to_2d_ubuf`

The copy/load helpers are narrower movement contracts and should establish the
DMA mapping before the Cube compute and fixpipe helpers are changed.

## Compatibility Alternative

The existing `external-calls` bridge mode can provide the NPU-IR AICore
bitcode that defines legacy C-interface helpers. That route remains useful as
a compatibility and behavior reference, but it is not the intended PTODSL
solution because it preserves opaque CCE helper dependencies.

Do not silently add the external bitcode fallback to PTODSL mode. Doing so
would make a nominally PTO-native artifact depend on legacy helper
implementations and would hide incomplete conversion.

## Acceptance Criteria

The PTODSL flash-attention simulator milestone is complete when:

- the generated VMI and VPTO contain no calls or bodyless declarations for the
  five helpers listed above;
- all replacement operations preserve source memory, layout, synchronization,
  and mixed-kernel section semantics;
- PTOAS emits and links the mixed fat object without NPU-IR AICore bitcode;
- the simulator launches both required MIX components successfully;
- numerical comparison passes against the testcase reference;
- focused negative tests reject unsupported shapes, types, layouts, or modes
  instead of leaving unresolved external declarations.

## Open Questions

- Which of the five calls still have a structured representation at the
  current bridge insertion point, and which have already been lowered to
  legacy calls?
- Are existing PTODSL instantiations available for the exact flash-attention
  type, shape, layout, and synchronization contracts?
- Which sync operations are caller-owned versus helper-owned for each DMA and
  Cube helper?
- Can the existing imported MmadL1 helper replace every
  `mma_tile_half_to_float` call in this kernel, or do the remaining calls use a
  distinct contract?
- Does the f32 NZ2ND-to-UB fixpipe form require a new PTODSL instantiation or
  only argument adaptation to an existing one?
