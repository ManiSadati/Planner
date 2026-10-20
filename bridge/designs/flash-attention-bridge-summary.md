# FlashAttention Bridge Bring-up Summary

Last updated: 2026-09-18

## Outcome

The `flash_atten_64x64` testcase now completes the AscendNPU-IR-to-PTOAS
bridge flow and passes its PTOAS simulator output comparison. The bring-up also
produced two smaller diagnostic testcases:

- `flash_atten_64x64_debug` can return one selected intermediate result and
  compare it with the host reference;
- `flash_attention_tiny` computes one exact, non-causal 64-by-64 tile without
  boundary or causal masks and passes the simulator comparison.

Passing the simulator does not mean that every source operation already has an
exact PTOAS VMI representation. Two compatibility gaps remain:

1. Continuous one-point reduction stores lose their stateful
   `vstus`/`vstas` sequence at the VMI boundary.
2. Predicate values cannot be stored or loaded directly through PTOAS VMI, so
   the current flow carries them temporarily as bytes and repairs the physical
   VPTO after PTOAS lowering.

The second repair is required before using the generated VPTO as the
native-equivalent hardware input. Hardware numerical validation after that
repair is separate from the simulator result recorded here.

## Incremental Bring-up

The work started from three smaller paths that were already useful in
isolation:

- `mixed_matmul_bias` exercised a mixed Cube and Vector kernel;
- `qk_matmul` exercised the Q-by-transposed-K product;
- the softmax testcase exercised vector reductions, exponentiation, and
  normalization.

FlashAttention was then reduced to a single 64-by-64 attention tile. This
removed the second key tile and most loop-carried online-softmax state, while
retaining the important QK matmul, scale and mask, row softmax, PV matmul, and
mixed AIC/AIV execution paths.

The bring-up proceeded in the following order:

1. Make the reduced kernel convert completely to PTOAS VMI.
2. Make the VMI lower completely to VPTO and link as one mixed kernel.
3. Run the simulator and isolate incorrect results at intermediate stages.
4. Compare bridge lowering with native NPU-IR lowering for operations that
   behaved differently.
5. Add the mask-free tiny testcase to separate the core attention computation
   from predicate-memory behavior.

## Template Parity

The bridge follows the rule that an operation implemented by an NPU-IR
template is represented by a corresponding PTODSL-generated PTOAS template. It
is not expanded directly into an unrelated bridge-only instruction sequence.

The FlashAttention path required PTODSL coverage for these template families:

- the native GM-to-UB two-dimensional FP16 load used by Q, K, and V movement;
- UB-to-L1 two-dimensional FP16 copies;
- FP16-to-FP32 L1 MMAD variants, including the transposed-B form;
- row-split NZ-to-ND Fixpipe transfers for FP32 and FP16 UB results.

The UB-to-L1 conversion was especially important for hardware. Leaving the
native helper call in the object produced the unresolved symbol
`_mlir_ciface_copy_ubuf_to_cbuf_2d_half`. Importing the matching
PTODSL-generated template removes that external dependency.

An earlier bridge-only PTODSL abstraction for the unaligned one-point FP32
store was removed after comparison with native NPU-IR showed that the native
behavior belongs to instruction lowering rather than a template. Continuous
`ONEPT_B32` stores and predicate-memory stores are also native instruction
lowering. Inventing PTODSL templates for these cases only in the bridge would
not be a one-to-one conversion. Their remaining VMI compatibility issues are
described below.

## VMI and Mixed-kernel Work

The reduced kernel exposed several AVE and PTOAS VMI cases that the smaller
kernels did not exercise together. The bridge work included support or fixes
for:

- predicate logical operations and cross-width predicate reconstruction;
- packed stores, deinterleave, truncation, rank-zero views, broadcast loads,
  and one-point stores;
- preserving masks with the value lane count expected by VMI operations;
- store-to-load ordering and rematerializing VMI values around barriers;
- explicit VMI scopes required by PTOAS;
- mixed AIC/AIV module sections so PTOAS can physicalize both sides and emit a
  single kernel container;
- conversion of the Cube and Vector synchronization forms exercised by the
  reduced kernel; the broader sync contract remains separately tracked.

These changes made the complete kernel representable through VMI instead of
leaving AVE operations or native helper calls in the PTOAS input.

## Intermediate-result Debugging

`flash_atten_64x64_debug` was added because a final-output mismatch does not
show where attention first becomes wrong. The testcase keeps the computation
shape and writes one selected intermediate value to an extra GM output.

The supported checkpoints are:

1. `loaded_q`
2. `loaded_k_transposed`
3. `raw_qk`
4. `scaled_scores`
5. `masked_scores`
6. `row_max`
7. `centered_scores`
8. `probabilities_f32`
9. `row_sum`
10. `loaded_v`
11. `probabilities_f16`
12. `pv_accumulator`
13. `denominator`
14. `normalized_f32`
15. `output_f16`

The fixture and comparison path were corrected so that each checkpoint uses
its real shape and datatype. Because the selected stage is a Triton
`constexpr`, changing `FLASH_ATTN_DEBUG_STAGE` requires regenerating the early
IR. Clearing an old generated IR or build cache is also necessary when moving
between stages.

Checking the load stages independently established that Q and transposed K
were loaded correctly. Advancing through `raw_qk` and the later checkpoints
then separated source/fixture mistakes from actual lowering problems. After
the bridge and fixture corrections, the full `flash_atten_64x64` testcase
passes the final simulator comparison.

## Mask-free Control Test

`flash_attention_tiny` computes the same basic QK-softmax-PV pipeline on one
exact 64-by-64 tile, but deliberately avoids boundary and causal masks. It
therefore does not need to spill a predicate value to UB and load it again.

Its successful VMI, VPTO, link, and simulator run show that the core attention
calculation can work independently of the predicate-memory compatibility
path. It is useful as a control testcase whenever a change to mask handling
causes the full FlashAttention testcase to fail.

## Remaining Compatibility Gaps

### Continuous ONEPT Store State

The row maximum and row sum reductions contain continuous `ONEPT_B32` stores.
Native NPU-IR carries alignment and base-pointer state through the loop with
`vstus.post` and finishes it with `vstas`. PTOAS VMI currently provides only a
stateless one-point vector store, which becomes `vsts {dist = "1PT_B32"}`.

This is an instruction-level mismatch. Its numerical impact on the current
testcase has not been established. The full analysis is in
`continuous-one-point-store-compatibility.md`.

### Predicate-memory Store and Load

FlashAttention materializes a validity mask in UB and reloads it when applying
the score mask. This stores the predicate itself; it is not a normal vector
store controlled by a predicate.

PTOAS VMI has a logical mask type, but its memory operations store vector
register data. The mask operand on a vector store only selects active lanes.
It cannot say that the value being stored is a packed predicate. Native NPU-IR
instead lowers to predicate operations such as `psts`, `pstu`, `vstas`, and
`plds`, with predicate offsets interpreted in bits.

The current bridge uses a temporary byte carrier so PTOAS can process the VMI.
After PTOAS emits physical VPTO, `--finalize-ptoas-vpto-predicate-memory`
recognizes that carrier, restores the native-equivalent predicate operations,
and converts bit offsets to byte addresses. The comparison runner performs
this step between PTOAS VPTO emission and later compilation or simulation.

This preserves the required physical behavior, but it is not first-class
predicate-memory support in PTOAS VMI. The full analysis is in
`predicate-memory-store-compatibility.md`.

## Current Validation Boundary

The following results are established:

- `flash_atten_64x64` emits VMI and finalized VPTO and passes the PTOAS
  simulator comparison;
- the staged debug testcase can validate each named intermediate against its
  host reference;
- `flash_attention_tiny` passes without using predicate-memory storage;
- the predicate finalizer has focused conversion tests for PB8, PB16, and PB32
  store/load carrier forms.

The following claims should not be made yet:

- continuous one-point streams are exactly represented in PTOAS VMI;
- predicate memory is natively represented in PTOAS VMI;
- raw, unfinalized PTOAS VPTO is suitable for the masked hardware kernel;
- final hardware numerical output has been validated solely because the
  simulator passes.

## Clean Long-term Direction

The bridge should remain a semantic mapping, not become a second backend.
Operations implemented by NPU-IR templates should continue to use matching
PTODSL-generated templates. Instruction-lowered behavior should have a VMI
contract capable of carrying the same meaning until PTOAS physicalization.

That requires two PTOAS-facing capabilities:

1. a state-carrying VMI representation for continuous one-point stores;
2. VMI predicate-memory load and store operations that carry PB8/PB16/PB32
   granularity and bit-addressed offsets.

When those exist and lower to the current physical VPTO instructions, the
ordinary-store carriers and the post-PTOAS predicate finalizer can be removed.
