# Predicate-memory Store and Load Compatibility

Last updated: 2026-09-18

## Summary

The masked 64-by-64 FlashAttention kernel stores a predicate value in UB and
loads it later. Native NPU-IR and PTOAS VMI do not currently express this
operation in the same way.

Native NPU-IR knows that this memory contains packed predicate bits and emits
predicate load/store instructions. PTOAS VMI has predicate registers, but its
VMI memory operations can only load or store vector-register data. A vector
store's mask says which lanes may write; it does not make the stored value a
predicate.

The current bridge therefore uses an ordinary byte vector only as a temporary
carrier through PTOAS VMI. A bridge-owned pass runs on the physical VPTO after
PTOAS and replaces that carrier with the same class of predicate instructions
used by native NPU-IR.

This makes the final VPTO native-equivalent for the supported forms, but it is
a compatibility repair rather than first-class PTOAS VMI support.

## Why FlashAttention Stores a Mask

The kernel constructs a boolean validity mask for the score tile. Depending on
the full kernel, that mask combines conditions such as valid query positions,
valid key positions, and the causal condition. The mask is produced in one
part of the vector computation, materialized in UB, and loaded again when the
kernel chooses between a valid score and the large negative masked value.

This is different from an ordinary masked store:

- ordinary masked store: store numeric data, and use a predicate to choose
  which numeric lanes write;
- predicate-memory store: the predicate bits are the data being stored.

Treating the second form as the first changes both the memory layout and the
meaning of its address.

## Native NPU-IR Behavior

The source represents predicate memory with an `i1` memref and records its
predicate granularity with `functionType = PB8`, `PB16`, or `PB32`.

Native instruction lowering uses:

- PB8 store: `psts`;
- PB16 or PB32 store: initialize alignment, execute `pstu`, then finish with
  `vstas`;
- load: `plds`, followed by the required predicate layout conversion for PB16
  or PB32.

The source memref offsets are offsets in `i1` elements, so they are bit
offsets. The final predicate instructions address packed bytes. Correct
lowering must therefore convert the accumulated bit offset to a byte offset.

This behavior is implemented by native instruction lowering, not by an
NPU-IR template. A bridge-only PTODSL template would not be a one-to-one copy
of the native design.

## Missing PTOAS VMI Contract

PTOAS VMI can represent a logical predicate register with `!pto.vmi.mask`.
Its VMI loads and stores, however, transfer `!pto.vmi.vreg` values. Their mask
operands control active lanes of a numeric transfer.

Consequently VMI cannot directly retain all of the source information:

- that the value in memory is a predicate;
- whether its granularity is PB8, PB16, or PB32;
- that the pointer offset is measured in bits;
- which physical predicate store/load sequence must be selected.

Physical VPTO already has the necessary `psts`, `pstu`, `vstas`, and `plds`
operations. The gap is between logical VMI and those physical operations.

## Temporary VMI Carrier

The bridge carries the operation through VMI in a recognizable ordinary form:

1. For a store, convert the predicate lanes to zero or one byte values and
   emit a VMI vector store.
2. For a load, emit a VMI byte-vector load and compare the bytes with zero to
   rebuild a logical mask.
3. Surround the pointer with an ordinary cast pair whose intermediate element
   type records PB8, PB16, or PB32.
4. Keep the original `i1` offset in the marked pointer expression so it can be
   recognized as a bit offset later.

The marker uses existing pointer casts instead of adding a private operation
or attribute to PTOAS IR. PTOAS preserves the casts while it assigns physical
layouts and lowers the carrier to VPTO.

At this intermediate point, the generated store may look like an ordinary
`vsts`, for example with a packed byte distribution. That operation is not the
final intended hardware store.

## VPTO Finalization

After PTOAS emits VPTO, the bridge runs:

```text
--finalize-ptoas-vpto-predicate-memory
```

The finalizer performs four tasks:

1. Recognize the pointer cast pair and recover PB8, PB16, or PB32.
2. Rebuild the address by accumulating its bit offsets and dividing by eight.
3. Recover the original predicate from the store's select/conversion carrier,
   or replace the loaded-byte/nonzero-compare carrier with a predicate load.
4. Remove the now-dead byte-vector operations and emit native-equivalent VPTO.

The resulting store is:

- `pto.psts` for PB8;
- `pto.init_align`, `pto.pstu`, then `pto.vstas` for PB16 or PB32.

The resulting load starts with `pto.plds`. PB16 and PB32 loads then interleave
and bitcast the predicate into the required physical mask layout.

The pass validates that a marked store has a recoverable predicate and that
the recovered physical mask width matches the PB marker. It fails instead of
silently retaining a marked but incorrect ordinary vector operation.

## Where the Pass Runs

The comparison runner currently uses this sequence:

```text
AscendNPU-IR bridge emits VMI
        |
        v
PTOAS emits raw physical VPTO
        |
        v
finalize-ptoas-vpto-predicate-memory
        |
        v
final VPTO used for later compilation or simulation
```

The finalizer acts on VPTO, not VMI. It cannot run before PTOAS because the
physical predicate operations and physical mask layouts do not exist yet.
Raw PTOAS VPTO and finalized VPTO are intentionally kept as separate artifacts
by `run_comparison_flow.sh`.

A hardware flow that cannot run a separate `bishengir-opt` step must provide
an equivalent post-PTOAS integration point. Merely placing this pass in the
pre-PTOAS `bishengir-compile` VMI pipeline would run it at the wrong level.

## Relation to the Continuous ONEPT Gap

This issue and the continuous one-point store issue both mention `vstas`, but
they are different:

| Issue | Value stored | Missing VMI meaning | Native physical sequence |
| --- | --- | --- | --- |
| Continuous `ONEPT_B32` | One ordinary FP32 reduction value per iteration | Loop-carried alignment and base-pointer state | `init_align`, repeated `vstus.post`, final `vstas` |
| Predicate memory | The predicate bits themselves | Predicate-valued memory, PB width, and bit addressing | PB8 `psts`, or PB16/PB32 `init_align`, `pstu`, `vstas`; `plds` for loads |

Fixing one does not fix the other.

## Current Status

- The full masked FlashAttention testcase exercises predicate-memory store and
  load.
- The bridge accepts PB8, PB16, and PB32 carrier forms covered by the
  finalizer tests.
- The comparison flow automatically finalizes raw PTOAS VPTO before using its
  final VPTO artifact.
- The mask-free `flash_attention_tiny` testcase avoids this path and passes,
  which isolates predicate memory from the core attention calculation.
- PTOAS VMI still does not directly represent predicate-valued memory.

## Clean Long-term Direction

PTOAS VMI needs predicate-memory load and store operations whose contract
includes:

- a mask value as the stored value or loaded result;
- PB8, PB16, or PB32 granularity;
- bit-addressed source offsets or an explicit conversion contract;
- lowering to the existing physical predicate instructions.

Once that contract exists, the bridge can map the AVE operation directly, and
PTOAS can select `psts`, `pstu`/`vstas`, and `plds` during normal VMI-to-VPTO
lowering. The byte carrier, pointer marker, and post-PTOAS finalizer can then be
removed.
