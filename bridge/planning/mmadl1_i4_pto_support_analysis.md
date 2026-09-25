# MmadL1 Integer-I4 Support Analysis

## Conclusion

The `enable_I4` form of `hivm.hir.mmadL1` is not ordinary
`i8 x i8 -> i32` matrix multiplication. It preserves the fact that the
operands originated as signed I4 values and selects a packed-S4 Cube path in
the CCE implementation. That path uses S4-specific L1-to-L0 transfers,
packing-aware dimensions, and the dedicated `mad_s4` intrinsic with I32
accumulation.

The current PTO implementation does not expose an equivalent complete
integer `s4 x s4 -> s32` path. Its S4 Cube bridge-load operations accept
packed FP4 types, and its typed MAD emitter supports `f16 x s4` but not
integer `s4 x s4`. Therefore `enable_I4`, when present, is a PTO
datatype/instruction-lowering gap rather than only a missing PTODSL helper.

The reduced failure logs do **not** prove that any observed MmadL1 operation
has `enable_I4`. The current bridge diagnostic groups I4 with result tensors,
bias, unit flags, and transpose-A, so full rejected IR is required to decide
whether this gap affects the current testcase corpus.

## Meaning of `enable_I4` in HIVM

The `hivm.hir.mmadL1` operation defines `enable_I4` as an optional unit
attribute with the description `Was i4 -> i8 conversion`; see
[`HIVMMacroOps.td`](../../../AscendNPU-IR/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMMacroOps.td),
lines 129-131. The exposed operand element type may consequently be I8, but
the attribute tells the backend that ordinary I8 MMAD semantics are not
sufficient: it must retain the original signed-I4 interpretation.

The legacy library-call name also appends `_i4` when this attribute is
present and asserts that transpose-A is absent; see
[`LibraryFunctionOpInterfaceImpl.cpp`](../../../AscendNPU-IR/bishengir/lib/Dialect/HIVM/IR/LibraryFunctionOpInterface/LibraryFunctionOpInterfaceImpl.cpp),
lines 1154-1170. This creates a distinct CCE specialization rather than
reusing the ordinary I8 specialization.

## CCE implementation

The CCE `LocalMmad` template carries I4 as a compile-time Boolean and changes
both operand movement and the MAD operation when it is enabled.

### Matrix A load

The I4 branch transfers A from L1 to L0A with
`load_cbuf_to_ca_s4`; see
[`LocalMmad.cpp`](../../../AscendNPU-IR/bishengir/lib/Template/lib/Cube/LocalMmad.cpp),
lines 128-145. The normal I8 branch instead follows the regular
`load_cbuf_to_ca` or transpose-load path.

### Matrix B load

For non-transposed B, the I4 branch calls `load2d_transpose_b4`, which uses
the S4 transpose-load intrinsic to produce the L0B layout; see
`LocalMmad.cpp`, lines 93-116 and 251-273.

The I4 plus transposed-B branch is currently empty and marked
`TODO: IMPLEMENT` at lines 264-268. Thus the legacy CCE implementation itself
only provides a complete I4 B-load path for the non-transposed case.

### Transpose and packing constraints

Transpose-A is explicitly forbidden by a compile-time assertion at
`LocalMmad.cpp`, lines 341-343:

```cpp
static_assert(TA == 0 && "i4 mmad doesn't support transpose A");
```

The I4 branch also changes the tiling arithmetic. It accounts for doubled
B-side/K extents, including `k_ceil_b = CEIL_FACTOR(2 * k, ...)`, and adjusts
the B-side partition size; see `LocalMmad.cpp`, lines 349-386. These
adjustments are part of the packed-S4 interpretation and must not be dropped
by mapping the operation to ordinary I8 MMAD.

### S4 MAD and accumulation

The I4 branch invokes `mad_intrin_core_s4` with I32 accumulation and passes
`2 * k_part_actual` and `2 * n_ceil` to the intrinsic; see `LocalMmad.cpp`,
lines 504-515. The wrapper lowers directly to `mad_s4`; see
[`Cube/Utils.h`](../../../AscendNPU-IR/bishengir/lib/Template/include/Cube/Utils.h),
lines 73-78.

Consequently, substituting ordinary `i8 x i8 -> i32` MAD would change at
least three parts of the CCE contract:

- the L1-to-L0 interpretation of A and B;
- the packing-aware K/N dimensions; and
- the selected hardware MAD intrinsic.

## Current PTO support

### S4 load operations are restricted to packed FP4

PTO defines `load_cbuf_to_ca_s4` and `load_cbuf_to_cb_s4`, but their verifier
requires both source and destination element types to be packed
`f4e1m2x2` or `f4e2m1x2`; see
[`VPTO.cpp`](../../../AscendNPU-IR/third-party/ptoas/lib/PTO/IR/VPTO.cpp),
lines 9411-9436. Those are floating-point FP4 formats, not the signed integer
I4 contract selected by HIVM `enable_I4`.

The existence of an operation containing `s4` in its name therefore does not
establish support for CCE integer-I4 MmadL1. The current verifier deliberately
narrows those bridge loads to packed FP4 values.

### No integer `s4 x s4 -> s32` typed MAD route

The PTO LLVM emitter recognizes a signed or signless I4 right-hand operand as
the `s4` fragment; see
[`VPTOLLVMEmitter.cpp`](../../../AscendNPU-IR/third-party/ptoas/lib/PTO/Transforms/VPTOLLVMEmitter.cpp),
lines 530-560. Its typed-callee selection then provides:

- `i8 x i8 -> i32` through `llvm.hivm.MAD.s8.c310`, at lines 626-629; and
- `f16 x s4` through `llvm.hivm.MAD.f16s4.c310`, at lines 650-653.

There is no selection for an integer I4/S4 left-hand operand combined with an
I4/S4 right-hand operand and I32 destination. In particular, the available
`f16 x s4` mixed-type operation is not equivalent to CCE `mad_s4`.

## Bridge behavior and required work

The current structured-template matcher rejects `enable_I4` before selecting
a PTODSL helper. It uses one diagnostic for result tensors, bias, unit flags,
transpose-A, and I4; see
[`MatmulRegionToPTO.cpp`](../../../AscendNPU-IR/bishengir/lib/Conversion/HIVMTemplatesToPTO/Cube/MatmulRegionToPTO.cpp),
lines 192-196.

Full support for the CCE I4 contract would require more than adding a named
MmadL1 PTODSL specialization:

1. Define a packed signed-I4 PTO buffer/element-type contract distinct from
   packed FP4.
2. Permit that integer-I4 contract in the S4 L1-to-L0 load operations and
   lower it to the appropriate S4 Cube load intrinsics.
3. Add an integer `s4 x s4 -> s32` MAD path corresponding to CCE `mad_s4`.
4. Preserve the CCE packing-related K/N dimension and tiling adjustments.
5. Add a PTODSL MmadL1 helper and C++ matcher branch with the same supported
   transpose restrictions. Transpose-A must remain rejected, and
   transpose-B requires separate implementation because it is also missing
   from the current CCE template.

## Impact on the collected failures

Two conclusions must remain separate:

- **PTO capability:** the CCE integer-I4 MmadL1 contract is not currently
  supported by the complete PTO typed load/MAD path.
- **Observed logs:** the collected reduced logs do not show whether
  `enable_I4` is present. Their grouped MmadL1 diagnostic cannot establish
  that I4 caused any current failure.

The next diagnostic step is to inspect or capture the full rejected
`hivm.hir.mmadL1` operations and check explicitly for the `enable_I4`
attribute. If it is absent, the observed failures belong to another member
of the grouped contract diagnostic and no I4 implementation is needed for
those testcases.
