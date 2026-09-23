# Continuous ONEPT Store Compatibility

Last updated: 2026-09-17

## Summary

The 64x64 FlashAttention kernel contains one-point stores that are marked
continuous. Native NPU-IR and the PTOAS bridge do not currently lower these
stores in the same way.

Native NPU-IR treats a continuous one-point store as a stateful stream. It
carries store state from one loop iteration to the next and finishes the stream
after the loop. The bridge currently lowers the same source operation to an
ordinary one-point store on every iteration.

This is a confirmed compatibility difference. It is not, by itself, proof that
this difference causes a particular numerical error.

## What Is a ONEPT Store?

`ONEPT_B32` stores one 32-bit value. A reduction may compute its result in a
vector register, but only one lane is written to memory.

There are two relevant source forms:

- An ordinary ONEPT store performs one independent one-point write.
- A continuous ONEPT store has the `hivm.is_continuous` attribute. Consecutive
  loop iterations belong to one store stream.

The continuous attribute is important. It changes the native instruction
sequence; it is not only an optimization hint.

The `ave.unaligned_ub_access` attribute does not make an ordinary ONEPT store
continuous. Native NPU-IR explicitly handles continuous ONEPT stores in a
separate path.

## Native NPU-IR Behavior

For an ordinary `ONEPT_B32` store, native NPU-IR emits a one-point `vstsx1`
store.

For a continuous `ONEPT_B32` store, native NPU-IR emits three logical steps:

1. Initialize vector-store alignment state before the loop.
2. Execute `vstus.post.f32` in the loop. This operation returns updated
   alignment state and an updated base pointer.
3. Execute one `vstas` after the loop to finish the stream.

The `HoistVstas` pass rewrites the loop so that the updated alignment state and
base pointer are carried from one iteration to the next:

```text
init alignment
       |
       v
loop(state, base):
    (next state, next base) = vstus.post(value, base, state)
       |
       +---- carried to the next iteration

vstas(final state, final base)
```

This sequence is implemented by native intrinsic lowering. It is not an
NPU-IR template.

## Where It Appears in FlashAttention

The generated 64x64 FlashAttention AVE IR contains both forms.

The continuous forms include:

- the per-row maximum reduction store;
- the per-row sum reduction store.

Both stores have a singleton `vector<1xf32>` value, a `VL1` mask, and the
`hivm.is_continuous` attribute. They execute across a loop of 64 elements.

The same generated kernel also contains an unaligned ONEPT store without
`hivm.is_continuous`. Native NPU-IR correctly keeps that store on the ordinary
`vstsx1` path. This confirms that the continuous attribute, rather than ONEPT
or unaligned access alone, selects the stateful native sequence.

After native lowering, the maximum and sum reduction loops each contain
`vstus.post.f32`, with one corresponding `vstas` after the loop.

## Current Bridge Behavior

The bridge converts the singleton value to a singleton PTOAS VMI register and
emits `pto.vmi.vstore`.

PTOAS later lowers this to a physical store of the following form:

```mlir
pto.vsts ... {dist = "1PT_B32"}
```

The final bridge output uses this same stateless instruction form for ordinary
and continuous ONEPT stores. The continuous alignment state, updated base
pointer, and final `vstas` are no longer represented.

The bridge also inserts `pto.mem_bar "VST_VLD"` before some later loads from
the same UB storage. This barrier orders stores and loads, but it is not a
replacement for `vstas`:

- `mem_bar` lowers to a memory-barrier instruction;
- `vstas` finishes a stateful unaligned-store stream;
- the two operations carry different information and have different target
  instructions.

## Why the Bridge Cannot Express the Native Sequence Today

PTOAS has low-level VPTO operations for all three native steps:

- `pto.init_align`;
- `pto.vstus`;
- `pto.vstas`.

However, the PTOAS VMI dialect used at the bridge boundary has no equivalent
stateful store operation. Its store operations do not accept alignment state
and do not return an updated alignment state or base pointer.

The low-level VPTO operations use physical vector-register and alignment
types. Those types are assigned only after PTOAS processes VMI. Emitting the
VPTO operations directly from the bridge would mix the logical VMI layer with
the later physical-register layer.

## Compatibility Direction

The clean exact mapping is a state-carrying PTOAS VMI operation or sequence
that PTOAS can lower to its existing VPTO `init_align`, `vstus`, and `vstas`
operations.

That direction currently requires a PTOAS VMI change, which is outside the
bridge's present scope. Until such support exists, the bridge can emit an
ordinary one-point store, but it cannot claim instruction-level parity for
continuous ONEPT streams.

The following alternatives should not be used:

- A bridge-only PTODSL template. Native NPU-IR does not implement this behavior
  with a template, so that would introduce a bridge-only abstraction.
- Direct VPTO emission from the AVE-to-VMI conversion. This would bypass the
  VMI layout and physicalization boundary.
- Treating a store/load memory barrier as `vstas`. These are different target
  operations.

## Current Status

- Ordinary `ONEPT_B32`: represented by a PTOAS one-point `vsts` path.
- Continuous `ONEPT_B32`: accepted by the bridge, but lowered through the same
  stateless path as the ordinary form.
- FlashAttention impact: the max and sum reduction streams expose this gap.
- Numerical impact: not established by this investigation.
- Exact parity: blocked on a stateful continuous-store contract at the PTOAS
  VMI level.
