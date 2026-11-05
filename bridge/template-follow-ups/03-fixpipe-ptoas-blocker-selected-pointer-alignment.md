# Fixpipe PTOAS failure: masked-store alignment through `arith.select`

## Issue

Eight `nz2nd-f32-f32-ub` cases fail during PTOAS VMI→VPTO lowering:

- Shapes: `small-b16`, `boundary-b16-32-16`, `boundary-b32-16-32`, `single-col-b64-16-16`.
- Controls: `scale-half`, `scale-truncated`, `normal-relu`, `scale-relu`.

```text
VMI-UNSUPPORTED: pto.vmi.masked_store requires a destination address with a
proven store alignment: a masked store needs per-lane write predicates, the
exact unaligned store form writes a contiguous low-bit prefix only, and a
write must never exceed the semantic footprint
```

The rejected operation is a **masked UB store in the vector consumer**, rather than the Fixpipe instruction itself. Its buffer addresses are aligned, but PTOAS cannot prove that alignment through the `arith.select` choosing the buffer.

## Concrete failing lowering

Example: [control-scale-half saved MLIR](/home/c50066216/local-template-test-runs/fixpipe-parallel/2026-11-04_14-54-52_0dv7m1jg/0015-nz2nd-f32-f32-ub-control-scale-half/dump/GU6TKFAFP6RALWFDZKSG2FQRH3AXN5ZDYKJMGGQLO4NCQC7CG7KA/kernel.ptovmi.mlir).

The saved vector function creates UB pointers at byte addresses `0` and `1024`, selects a buffer, then adds `0.25` using a 16-lane predicate. The following are excerpts from that function; intervening operations are omitted:

```mlir
%1 = pto.castptr %c0_i64 : i64 -> !pto.ptr<f32, ub>
%3 = pto.castptr %c1024_i64 : i64 -> !pto.ptr<f32, ub>

// %12 is a scalar buffer-selection condition.
%31 = arith.select %12, %3, %1 : !pto.ptr<f32, ub>
%32 = arith.select %12, %3, %1 : !pto.ptr<f32, ub>

pto.vecscope {
  scf.for %arg10 = %c0_0 to %c16 step %c1 {
    %c16_1 = arith.constant 16 : index
    %37 = arith.muli %arg10, %c16_1 : index
    %38 = pto.addptr %31, %37 : <f32, ub> -> <f32, ub>
    %c0_2 = arith.constant 0 : index
    %39 = arith.addi %c0_2, %c0_0 : index
    %40 = pto.vmi.load %38[%39]
        : !pto.ptr<f32, ub> -> !pto.vmi.vreg<64xf32>
    %41 = pto.vmi.pge "PAT_VL16" : !pto.vmi.mask<64xpred>
    %42 = pto.vmi.vadds %40, %cst, %41
        : !pto.vmi.vreg<64xf32>, f32, !pto.vmi.mask<64xpred>
          -> !pto.vmi.vreg<64xf32>
    %c16_3 = arith.constant 16 : index
    %43 = arith.muli %arg10, %c16_3 : index
    %44 = pto.addptr %32, %43 : <f32, ub> -> <f32, ub>
    %c0_4 = arith.constant 0 : index
    %45 = arith.addi %c0_4, %c0_0 : index
    %46 = pto.vmi.pge "PAT_VL16" : !pto.vmi.mask<64xpred>
    pto.vmi.vstore %42, %44[%45], %46
        : !pto.vmi.vreg<64xf32>, !pto.ptr<f32, ub>,
          !pto.vmi.mask<64xpred>
  }
}
```

Here `%c0_0` is zero and `%cst` is `0.25`. `pto.addptr` offsets are in **elements**, so `row × 16` f32 elements means `row × 64` bytes. The store destination is therefore either:

```text
0    + row × 64 bytes
1024 + row × 64 bytes
```

Both are divisible by 32 for every row. The failure is an **unknown proof result**, not evidence of a misaligned runtime address. Other shape cases contain nested selections among multiple UB buffers.

## Where the rejection comes from

In [PTOAddressAnalysis.cpp](/home/c50066216/AscendNPU-IR/third-party/ptoas/lib/PTO/Analysis/PTOAddressAnalysis.cpp), `getKnownPointerRemainder` traces pointer casts and `pto.addptr`, but has no `arith.select` case. Its final dispatch is:

```cpp
auto add = pointer.getDefiningOp<AddPtrOp>();
if (!add)
  return std::nullopt;
return pointerRemainderThroughAddPtr(add, alignmentBytes, depth);
```

When tracing `%44`, it follows the addition to `%32`, encounters the selection, and returns unknown. `isKnownAddressAligned` accepts only a known zero remainder; unknown returns false.

In [VMIToVPTOPatternInternals8.cpp](/home/c50066216/AscendNPU-IR/third-party/ptoas/lib/PTO/Transforms/VMIToVPTO/VMIToVPTOPatternInternals8.cpp), `verifySupportedVMIStructuredMaskedStoreOp` explicitly rejects that result:

```cpp
bool unalignedDestination = !isKnownAddressAligned(
    store.getDestination(), store.getOffset(),
    valueVMIType.getElementType(), kMemoryAccessAlignmentBytes);
if (unalignedDestination) {
  // Emits the diagnostic quoted above.
  return WalkResult::interrupt();
}
```

`kMemoryAccessAlignmentBytes` is **32**. The local variable name includes both genuinely unaligned and unknown addresses. Bridge `pto.vmi.vstore` is dispatched to the legacy masked-store form inside PTOAS, explaining why the diagnostic names `pto.vmi.masked_store`.

The check must not simply be removed: an arbitrary selected pointer could be unaligned, and the backend must preserve the exact masked write footprint.

## NPU-IR workaround

Implementation commit: **`2486bca41` — `[Conversion] fix: Branch masked stores on selected UB buffers`**. Includes the bridge-finalization rewrite and its nested-buffer-selection regression test.

File: [HIVMAVEToPTOASVMI.cpp](/home/c50066216/AscendNPU-IR/bishengir/lib/Conversion/HIVMAVEToPTOASVMI/HIVMAVEToPTOASVMI.cpp).

The initial AVE→VMI conversion in `ConvertMaskedStorePattern` emits the store normally:

```cpp
rewriter.replaceOpWithNewOp<pto::VMIvStoreOp>(
    op, adaptor.getVal(), *base, *linearOffset, Value{}, Value{},
    ValueRange{*mask}, IntegerAttr{});
```

The workaround runs in `FinalizePTOASVMIBridgePass`, after pointer normalization made selections visible. Its helper, `lowerSelectedMaskedStore`, handles ordinary masked stores without stride/group modes, checks selected base alignment using the existing `isDirectPointerAligned`, and clones the store and its pointer additions into scalar branches. Nested selections are handled recursively; cloning preserves store attributes.

Applied to the example above, the relevant transformation is the following. This is **normalized explanatory MLIR**, with descriptive SSA names; it shows the implemented transformation, not a verbatim dump. The load/value calculation remains before these stores.

```mlir
// Before: PTOAS loses the alignment proof at %selected.
%selected = arith.select %condition, %buffer1024, %buffer0
    : !pto.ptr<f32, ub>
%dst = pto.addptr %selected, %rowOffset
    : <f32, ub> -> <f32, ub>
pto.vmi.vstore %value, %dst[%zero], %mask
    : !pto.vmi.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.vmi.mask<64xpred>

// After: each store has a directly traceable aligned base.
scf.if %condition {
  %dstTrue = pto.addptr %buffer1024, %rowOffset
      : <f32, ub> -> <f32, ub>
  pto.vmi.vstore %value, %dstTrue[%zero], %mask
      : !pto.vmi.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.vmi.mask<64xpred>
} else {
  %dstFalse = pto.addptr %buffer0, %rowOffset
      : <f32, ub> -> <f32, ub>
  pto.vmi.vstore %value, %dstFalse[%zero], %mask
      : !pto.vmi.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.vmi.mask<64xpred>
}
```

Exactly one branch writes the original value with the original mask and offset. It may introduce branches or duplicate code; a performance impact was not measured.
