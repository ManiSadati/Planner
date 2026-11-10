# Normal F32-to-BF16 CBUF Fixpipe Compatibility

Last updated: 2026-11-10

## Summary

The native NPU-IR template registers
`fixpipe_normal_float_to_bfloat16_t_4d_to_4d_cbuf`, which transfers an F32
accumulator from L0C into BF16 L1 (CBUF) storage using the normal 4D-to-4D
layout. The corresponding PTODSL helper cannot currently be generated because
PTOAS rejects a BF16 destination for the required `nz2nz` accumulator-store mode.

## Native NPU-IR Behavior

`bishengir/lib/Template/lib/RegBase/Cube/Fixpipe.cpp` registers:

```cpp
REGISTE_FIXPIPE(cc, cbuf, 4, 4, float, bfloat16_t, normal, TransformMode::NORMAL);
```

The intended transfer converts F32 accumulator values to BF16 and preserves
the normal fractal layout in L1. The matching PTO operation is
`pto.mte_l0c_l1` with an F32 L0C source, BF16 L1 destination, `nz2nz`, and
`pre_quant` mode `f32_bf16`.

Native registration establishes that the template exists. It does not establish
that every control configuration has been validated at runtime.

## Observed Failure

Generating the explicit PTODSL instantiation through
`fixpipe_ptodsl.py --emit-instantiation
fixpipe_normal_float_to_bfloat16_t_4d_to_4d_cbuf` reaches MLIR verification and
fails with:

```text
'pto.mte_l0c_l1' op nz2nz requires destination element type to be f32
```

The rejected operation has an F32 L0C pointer, a BF16 L1 pointer, `nz2nz`,
and `pre_quant` mode `f32_bf16`. Verification fails before a helper resource
can be generated successfully, so this attempt provides no runtime correctness
evidence for the PTODSL instance.

## PTOAS Restriction

The shared structured accumulator-store verifier in
`third-party/ptoas/lib/PTO/IR/VPTO/structuredacc/VPTOStructuredAcc.cpp`, in the
`AccStoreMode::Nz2nz` case, contains:

```cpp
if (!isa<FloatType>(destinationElementType) ||
    !cast<FloatType>(destinationElementType).isF32()) {
  return op->emitOpError("nz2nz requires destination element type to be f32");
}
```

This rejects BF16 even when the operation supplies the F32-to-BF16 conversion
mode. The confirmed blocker is the verifier contract. Whether subsequent PTOAS
lowering already supports this precise combination remains unverified.

## Current Status

- Native CCE template: registered.
- PTODSL generation: blocked by the PTOAS `nz2nz` destination-type verifier.
- PTOAS verifier or lowering fix: not implemented in this investigation.
- PTODSL runtime parity: not established.
