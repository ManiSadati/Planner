# Signed Vector-Scalar Min/Max Compatibility

Last updated: 2026-08-24

## Problem

AVE distinguishes signed integer vector-scalar minimum and maximum with
`ave.hir.vsmins` and `ave.hir.vsmaxs`. Their vector and scalar MLIR types are
nevertheless signless `i32`.

The direct bridge mapping originally emitted:

```mlir
pto.vmi.vmins %vector, %scalar, %mask
    : !pto.vmi.vreg<64xi32>, i32, !pto.vmi.mask<64xpred>
      -> !pto.vmi.vreg<64xi32>
```

This VMI is initially well typed. PTOAS then runs
`pto-vmi-normalize-signless-int-to-unsigned`, whose whitelist includes VMI
vector-scalar min/max. The pass changes signless vector operands and results to
unsigned vectors but leaves scalar operands signless. PTOAS subsequently
rejects the resulting `ui32` vector versus `i32` scalar mismatch.

Treating the operation as unsigned is not a valid workaround. A negative signed
value such as `-1` has the unsigned bit-pattern value `4294967295`, which changes
both minimum and maximum semantics.

There are two PTOAS-side issues at this boundary:

- the normalization pass can turn accepted VMI into VMI rejected by PTOAS's
  own verifier;
- the VMI vector-scalar operation requires scalar signedness to match the
  vector, but the normalization pass does not update the scalar.

The MLIR revision used by the bridge requires integer `arith.constant` results
to be signless, so the bridge cannot directly materialize an `si32` constant.
PTOAS already uses `builtin.unrealized_conversion_cast` as a no-op signedness
carrier between signless and explicitly signed or unsigned integer scalars.

The bridge must not modify either the PTOAS repository or the in-tree PTOAS
dialect copy. Its emitted VMI must satisfy the existing PTOAS pipeline.

## Considered Solutions

### Signed vector-vector min/max

Broadcast the scalar, bitcast the source and broadcast value to an explicitly
signed VMI vector type, apply `pto.vmi.vmin` or `pto.vmi.vmax`, and bitcast the
result back to the surrounding signless vector type.

This is general for constant and dynamic scalar operands and keeps min/max as a
single semantic operation, but introduces a broadcast and signed carrier
bitcasts.

### Signed compare/select expansion

Broadcast the scalar and express min/max with a signed `slt` or `sgt` compare
followed by `pto.vmi.vsel`. PTOAS explicitly supports signed comparison modes
and creates signed physical carriers while lowering them.

This avoids signed scalar values but expands one operation into multiple
operations and may produce worse target code.

### Signed VMI vector constants

For an `arith.constant` scalar, create a splat `pto.vmi.constant` whose element
type is explicitly `si32`, bitcast the source vector to `si32`, apply signed
vector-vector `pto.vmi.vmin` or `pto.vmi.vmax`, and bitcast the result back to
signless `i32`.

This is concise for the current row-softmax kernel but does not support dynamic
scalar operands.

### Explicitly signed vector-scalar min/max

For an `arith.constant` scalar, reinterpret its signless `i32` SSA value as
`si32` with a one-value `builtin.unrealized_conversion_cast`, bitcast the source
to an explicitly signed VMI vector, apply `pto.vmi.vmins` or `pto.vmi.vmaxs`
directly, and bitcast the result back to signless `i32`.

This preserves the source operation as one target arithmetic instruction and
satisfies the PTOAS VMI verifier before and after signless normalization. It is
the smallest lowering for the constant-scalar form and matches PTOAS's existing
representation for scalar signedness-only casts.

## Implemented Compatibility Solution

The bridge uses explicit signed vector and scalar types with the direct PTOAS
VMI scalar operation for `ave.hir.vsmins` and `ave.hir.vsmaxs`.

The accepted form is intentionally narrow:

- source and result are rank-one `vector<Nxi32>`;
- the scalar is defined by `arith.constant`;
- the mask is a direct `ave.hir.pge <ALL>`.

The bridge emits:

1. `pto.vmi.bitcast` from the signless source to
   `!pto.vmi.vreg<Nxsi32>`;
2. a `builtin.unrealized_conversion_cast` from the original constant's `i32`
   value to `si32`;
3. direct `pto.vmi.vmins` or `pto.vmi.vmaxs` with matching signed vector and
   scalar types;
4. `pto.vmi.bitcast` back to `!pto.vmi.vreg<Nxi32>`.

Dynamic scalars and partial masks are rejected at the original AVE operation.
Partial masks remain unsupported until the inactive-lane contract of the AVE
and PTOAS scalar operations is verified to match.

The direct mappings for floating-point `ave.hir.vmins` and `ave.hir.vmaxs` are
unchanged. Integer addition also remains signless because addition is
independent of signed interpretation.

## Future Direction

The same signedness-only cast could represent a dynamic signless `i32` scalar,
but that path remains rejected until it is covered by a concrete kernel and an
end-to-end PTOAS test. Partial masks can be enabled after their inactive-lane
semantics are verified. A dedicated PTO scalar bitcast would be preferable if
PTOAS adds one in the future.
