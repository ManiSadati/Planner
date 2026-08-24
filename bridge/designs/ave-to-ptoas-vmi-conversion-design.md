# AVE to PTOAS VMI Conversion Design Decisions

Last updated: 2026-08-24

This document records bridge design decisions as they are made during
implementation. The implementation plan remains in
`Planner/bridge/planning/ave-to-ptoas-vmi-implementation-plan.md`; this file is
the running design log for choices that affect code structure.

The signed integer vector-scalar min/max compatibility problem and its
constant-only explicit signed lowering are documented separately in
`Planner/bridge/designs/signed-vector-scalar-min-max-compatibility.md`.

## Decision 1: Use An In-Tree PTO Dialect Copy

The bridge must emit real PTO VMI operations and types from a dialect built
inside AscendNPU-IR. The PTO dialect TableGen sources will be moved from PTOAS
into the AscendNPU-IR tree and built as a normal in-tree MLIR dialect. The
bridge must not link against PTOAS's installed `libPTOIR.a`.

Implications:

- AscendNPU-IR builds the bridge unconditionally from the in-tree PTO dialect.
- No `find_package(PTOAS)` or `PTOAS::PTOIR` dependency is required for
  AscendNPU-IR.
- The temporary `PTOASMLIRCompat.cpp` shim and pass anchor are removed because
  there is no external PTOAS archive to satisfy.
- The copied PTO dialect must be generated with AscendNPU-IR's MLIR TableGen
  and linked through its eventual in-tree dialect target.
- The conversion pass may continue to use typed C++ APIs such as
  `pto::PTODialect`, `pto::VMIVRegType`, and `pto::VMIVaddOp`; those names
  should resolve from the in-tree PTO dialect once the `.td` files and dialect
  implementation are added.

## Decision 2: Use Conversion-Driven Validation

The pass should not keep a large separate verifier that walks the IR and then
walks it again to perform the conversion. The main legality mechanism should be
MLIR dialect conversion:

- A `TypeConverter` defines the supported AVE-to-VMI type boundary.
- A `ConversionTarget` marks AVE illegal, PTOAS VMI legal, and structural
  dialects dynamically legal only when their types are converted.
- One `ConversionPattern` per supported AVE operation performs the local
  semantic checks for that operation.
- `applyFullConversion` rejects the whole module when any operation, type,
  region edge, or materialization is unsupported.

This keeps code size and maintenance cost lower because the same pattern that
knows how to convert an operation also owns that operation's local legality
rules. It also avoids duplicating dispatch logic between a verifier walk and a
conversion walk. Runtime performance is not the main driver because both
approaches are linear in IR size, but avoiding a full preflight pass removes one
unnecessary traversal and reduces diagnostic drift.

Separate checks are still appropriate when they are genuinely module-wide and
not naturally owned by one conversion pattern. Examples include mixed
Cube/Vector rejection, kernel/container ABI rules, and final residual checks for
AVE operations or `unrealized_conversion_cast`.

## Current Next Step

The vector-add and row-softmax acceptance artifacts now lower through PTOAS to
VPTO. The complete Planner row-softmax fixture also builds, runs on the
simulator, and passes numerical comparison. The next implementation stage
should be driven by another concrete lowered vector kernel; semantic debt from
the row-softmax compatibility mappings remains listed below.

## Decision 3: Phase 1 Conversion Skeleton Shape

The first implementation of `convert-hivmave-to-ptoas-vmi` should convert the
strict full-lane slice directly to surface VMI:

- `ave.hir.pge <ALL>` becomes `pto.vmi.pset "PAT_ALL"`.
- `ave.hir.vload <NORM>` becomes `pto.vmi.load`.
- full-mask `ave.hir.vadd`, `vsub`, `vmul`, and `vabs` become unmasked
  `pto.vmi.vadd`, `vsub`, `vmul`, and `vabs`.
- all-active normal `ave.hir.masked_store` becomes unmasked `pto.vmi.store`.
- supported `vector<Nxf16|bf16|f32>` values become `!pto.vmi.vreg<NxT>`.
- supported `vector<Nxi1>` predicates become `!pto.vmi.mask<Nxpred>`.
- accepted HIVM GM/UB memrefs become PTO pointers.

Local pattern checks are kept for supported operation families whose attributes
or masks can make the operation unsupported. Unmapped AVE operations have no
conversion pattern and are rejected by `applyFullConversion` with the original
operation location. That is intentional for the skeleton; detailed diagnostics
for a new operation family should be added when that family gets its first
conversion pattern.

## Decision 4: Keep Direct Floating Vector-Scalar Ops As One Family

The first operation-coverage expansion keeps the same strict full-lane contract
and adds the direct floating-point vector-scalar arithmetic family:

- `ave.hir.vadds` becomes `pto.vmi.vadds`.
- `ave.hir.vmuls` becomes `pto.vmi.vmuls`.
- `ave.hir.vmaxs` becomes `pto.vmi.vmaxs`.
- `ave.hir.vmins` becomes `pto.vmi.vmins`.

These mappings share one conversion pattern because the source and target
operation shapes are identical: vector source, scalar operand, predicate mask,
and one vector result. The pattern owns only local legality:

- the mask must be a direct all-active `ave.hir.pge <ALL>` converted to
  `pto.vmi.pset`;
- the vector source/result must be a supported rank-1 `f16`, `bf16`, or `f32`
  data vector;
- the scalar operand type must match the AVE vector element type.

This keeps the conversion extensible without adding a separate verifier walk.
Additional vector-scalar operations such as shifts or integer-only forms should
be added only after their scalar type rules and PTOAS VMI verifier constraints
are checked operation by operation.

This direct family is floating-point only. AVE signed integer
`ave.hir.vsmins` and `ave.hir.vsmaxs` do not use this pattern because PTOAS
normalizes their signless vector carrier to unsigned while leaving the scalar
signless. Their explicit signed lowering is Decision 14.

## Decision 5: Support Direct Static Tail Masks Only

The bridge supports non-full-lane masks only when the mask operand is produced
directly by `ave.hir.pge` with a static pattern that has a verified PTO VMI
equivalent:

- `ALL` maps to `pto.vmi.pset "PAT_ALL"`.
- `VL1`, `VL2`, `VL3`, `VL4`, `VL8`, `VL16`, `VL32`, `VL64`, and `VL128` map
  to `pto.vmi.pge "PAT_VL<n>"`.

The bridge intentionally rejects `M3`, `M4`, `H`, `Q`, and `ALLF` for now.
`M3` and `M4` are not prefix masks, `ALLF` has no accepted `pto.vmi.pge`
encoding because `PAT_VL0` is rejected by the PTO verifier, and `H`/`Q` have
conflicting or incomplete source-side documentation and utility behavior.

Arithmetic conversions now preserve the converted mask operand on the emitted
VMI arithmetic op instead of requiring all-active masks or dropping the mask.
This keeps the bridge IR semantically explicit. Predicate-producing or
predicate-combining AVE operations are still unsupported; a supported mask must
be a direct `ave.hir.pge` so conversion cannot silently accept composed masks
that have no mapping.

`ave.hir.masked_store` keeps the existing unmasked `pto.vmi.store` emission for
direct all-active masks. For supported partial `VL*` masks it emits unified
`pto.vmi.vstore` with the mask operand and omits `pmode`, matching the accepted
in-tree PTO verifier and the observed PTOAS lowering path.

Known downstream caveat: PTOAS currently accepts VMI arithmetic with masks, but
its unified-to-legacy lowering was observed to drop arithmetic masks while
preserving the store mask. This is acceptable only for the narrow straight-line
tail case where a masked arithmetic result is immediately consumed by a store
using the same mask. General masked computation should not be claimed correct
until PTOAS preserves arithmetic inactive-lane semantics or the bridge has a
proven workaround.

## Decision 6: Convert MemRef Metadata/View Chains To PTO Pointers

`memref.extract_strided_metadata` and `memref.reinterpret_cast` are structural
memref operations, but the Stage 1 target ABI is pointer-based. The bridge now
folds accepted rank-one unit-stride metadata/view chains into PTO pointer
operations instead of preserving memref descriptors:

- accepted GM and UB memrefs convert to `!pto.ptr<element, gm|ub>`;
- signless `i8` GM memrefs convert to `!pto.ptr<ui8, gm>` to match the proven
  PTOAS hand-authored target;
- `memref.extract_strided_metadata` on a static rank-one unit-stride GM/UB
  memref is replaced with the converted source pointer plus constant
  offset/size/stride values;
- `memref.reinterpret_cast` with rank one, unchanged element type, and static
  unit stride becomes either the source pointer for offset zero or `pto.addptr`
  for a static/dynamic element offset;
- non-unit strides, element-type changes, unsupported spaces, and higher ranks
  remain unsupported.

For a dynamic-layout memref function argument, the converted `!pto.ptr` is
treated as the already-normalized logical base pointer. The original memref
descriptor offset is intentionally not reconstructed because that state is not
available after collapsing the ABI to PTO pointers. This is acceptable for the
Stage 1 artifact, whose helper computes explicit element offsets in SSA.

## Decision 7: Consume Only Semantically Matched Normalized AVE Attributes

The bridge accepts the normalized AVE operation attributes generated in
`lowered_vector_add_kernel.mlir`, but only through per-operation semantic
allowlists:

- `ave.hir.pge` may consume `functionType = #ave.func_dist_type<pb32>`.
- `ave.hir.vload <NORM>` may consume
  `functionType = #ave.func_dist_type<norm>`.
- `ave.hir.masked_store <NORM_B16|NORM_B32>` may consume
  `functionType = #ave.func_dist_type<norm>`.
- `ave.hir.masked_store` may consume the unit attribute
  `hivm.is_continuous` only on the accepted normal contiguous store path.

The conversion does not require these attributes on hand-authored tests, but if
they are present they must match the operation form. Unknown attributes and
mismatched `functionType` values still reject the original AVE operation.

This keeps Phase 1B strict while accepting the normal Ascend lowering output.
After this change, the exact Stage 1 artifact proceeds past the previous
`functionType` failure and the first remaining blocker is
`hivm.hir.set_ctrl`, which belongs to the later synchronization/control phase.

## Decision 8: Convert Existing PTO MTE Ops By Operand Remapping

The lowered vector-add entry function already contains PTO MTE operations with
HIVM/PTO memref operands. Dialect conversion does not automatically update
these existing legal-dialect operations when their operands need type
conversion, so the bridge owns a narrow same-op rewrite for:

- `pto.mte_gm_ub`
- `pto.mte_ub_gm`

The rewrite keeps the original PTO op, attributes, transfer lengths, strides,
and optional operands, but replaces operands with the converted values. The op
is accepted only when its source and destination have become PTO pointer types.
This prevents mixed memref/pointer MTE IR from reaching PTOAS while avoiding a
new interpretation of DMA semantics in the AVE bridge.

Together with the pointer-memory conversion:

- `hivm.hir.pointer_cast(i64)` becomes `pto.castptr`;
- `memref.memory_space_cast` is erased when both sides convert to the same PTO
  pointer type;
- zero-offset GM reinterpret casts fold to the original GM pointer;
- MTE source/destination operands print as `!pto.ptr<..., gm|ub>`.

## Decision 9: Preserve Supported Sync Operations Without Rescheduling

Flag/wait conversion explicitly maps HIVM pipe and event attributes to PTO
typed attributes. Static `EVENT_ID0` through `EVENT_ID7` map to the same PTO
event identity. A dynamic event operand maps to `pto.set_flag_dyn` or
`pto.wait_flag_dyn`, with an `arith.index_cast` when PTO requires an index.

The accepted pipe set includes the observed scalar, vector, and MTE paths plus
`PIPE_ALL` barriers. The bridge preserves source operation order and does not
insert, remove, pair, or reschedule synchronization. This was necessary for
row softmax: scalarizing a multi-lane `BRC_B32` load moved the read to the
scalar pipeline and made the surrounding vector-pipeline flag/wait sequence
ineffective. Decision 16 keeps that load on the vector pipeline instead of
changing the source synchronization contract.

`hivm.hir.set_ctrl` is lowered as a read-modify-write sequence because the HIVM
op updates one bit while PTO `pto.set_ctrl` writes the complete control value:

1. `pto.get_ctrl : i64`
2. `arith.ori` with `1 << idx` for `enable = true`, or `arith.andi` with
   `~(1 << idx)` for `enable = false`
3. `pto.set_ctrl`

This preserves unrelated control bits and supports the observed bits 48 and
60. Indices outside `[0, 63]` are rejected.

## Decision 10: Rewrite Helper Calls With Converted Pointer ABI

The Stage 1 artifact calls an outlined vector helper after UB
`hivm.hir.pointer_cast` operations have been converted to PTO UB pointers.
`func.call` therefore needs an explicit rewrite to use the converted operand
types and callee signature. The bridge keeps the `no_inline` marker on the
rewritten call and drops the source-only `hivm.vector_function` call attribute.

This conversion is scoped to direct `func.call`; indirect calls are not
supported in Stage 1.

## Decision 11: Emit A Minimal PTOAS Kernel Contract

Phase 1F removes the source-only Ascend metadata after successful dialect
conversion and emits the minimal PTOAS contract proven by the hand-authored
target:

- module attributes become `pto.target_arch = "a5"` and
  `pto.kernel_kind = #pto.kernel_kind<vector>`;
- the original `hacc.entry` function is marked with `pto.kernel`;
- helper functions keep `no_inline` when it was present;
- source-only module, function, and argument attributes from HACC, HIVM, TT,
  and DLTI are dropped.

This pass does not yet claim a general Ascend-target-to-PTO-architecture
mapping. The hard-coded `a5` contract is tied to the Stage 1
`lowered_vector_add_kernel.mlir` artifact and the Phase 1A PTOAS validation.

## Decision 12: Use Row Softmax As The Second Acceptance Artifact

`Planner/bridge/testcases/row_softmax/input.mlir` is the Stage 2 source of
truth. It starts after `hacc-append-device-spec`, then relies on the normal
AscendNPU-IR pipeline to produce AVE/HIVM before the bridge pass. The Planner
runner extracts the last successful dump after
`convert-hivmave-to-ptoas-vmi`, even if a later native-backend stage fails.

Acceptance requires unmodified PTOAS VMI-to-VPTO lowering and simulator
comparison. The committed fixture launches 64 rows of 256 `f32` elements and
checks against a NumPy reference with `atol = 2e-4` and `rtol = 2e-4`. This is
evidence for that finite-input fixture, not a general proof for arbitrary
softmax shapes or IEEE exceptional values.

## Decision 13: Adapt AVE Reduction Result Shape Explicitly

AVE reduction results can retain the source vector width, while PTOAS VMI
floating reductions return `!pto.vmi.vreg<1xT>`. The bridge emits
`pto.vmi.reduce_addf` for ADD or `pto.vmi.reduce_maxf` for MAX. If the AVE
result is wider than one lane, it then emits `pto.vmi.broadcast` back to the
converted AVE result width; a one-lane result uses the reduction directly.

Only rank-one `f16`, `bf16`, and `f32` vectors with matching source/result
element types and direct supported masks are accepted. ADD sets the target
reassociation attribute required by the PTO op. MIN, integer, and bitwise
reduction kinds remain unsupported.

## Decision 14: Retype Signed Integer Scalar Min/Max Explicitly

Direct `pto.vmi.vmins`/`vmaxs` is not compatible with PTOAS for AVE signed
integer `vsmins`/`vsmaxs`: PTOAS changes the signless vector carrier to
unsigned but leaves the scalar `i32`, causing a verifier mismatch and, if
forced, incorrect signed ordering.

For the row-softmax form, the bridge requires rank-one `vector<Nxi32>`, a
direct `pge <ALL>` mask, and an `arith.constant` scalar. It emits:

1. `pto.vmi.bitcast` from signless `i32` to signed `si32` lanes;
2. a signedness-only `builtin.unrealized_conversion_cast` from the scalar
   constant's `i32` value to `si32`;
3. direct `pto.vmi.vmins` or `pto.vmi.vmaxs` with matching signed types;
4. `pto.vmi.bitcast` back to the surrounding signless carrier.

Dynamic scalars and partial masks are rejected. Floating-point
`ave.hir.vmins`/`vmaxs` retain their direct mappings.

## Decision 15: Map Floating Vector Max Directly

Floating `ave.hir.vmax` maps directly to masked `pto.vmi.vmax`. PTOAS provides
the matching floating vector operation, and no type or verifier incompatibility
requires an expansion.

An earlier row-softmax debugging version expanded the op to
`pto.vmi.vcmp "gt"` plus `pto.vmi.vsel`. That expansion was rejected because
it adds an instruction and predicate value, can change inactive-lane behavior,
and is not generally equivalent for NaNs or signed zero. The softmax failure
was caused by broadcast-load pipeline synchronization, not direct `vmax`.

## Decision 16: Keep Multi-Lane BRC_B32 Loads On The Vector Pipeline

`ave.hir.vload <BRC_B32>` reads a scalar UB location and distributes it across
a vector result. For a result wider than one lane, the bridge emits unified
`pto.vmi.vload` with `dist_mode = "brc"`. This keeps the operation on the
vector pipeline and preserves the effect of the source vector-pipeline
`set_flag`/`wait_flag` sequence around outlined reduction helpers.

For a one-lane result only, the bridge emits `pto.load_scalar` followed by
`pto.vmi.broadcast`. PTOAS layout assignment rejects the one-lane unified
broadcast-load form. Both mappings require a rank-zero UB `f32` pointer, no
indices, and a supported rank-one floating result.

## Decision 17: Reconstruct Scalar Staging And One-Lane Stores

Row-softmax scalar staging crosses source vector, scalar, and rank-zero memref
representations. The bridge uses narrow structural adaptations:

- rank-zero UB `memref.store` of `f32` becomes zero-offset
  `pto.store_scalar`;
- `masked_store <ONEPT_B32>` to a rank-zero UB destination becomes
  zero-offset unmasked `pto.vmi.store`;
- an unresolved projection from a wider VMI vector to a one-lane value used
  only by stores is replaced with `pto.vmi.pge "PAT_VL1"` plus masked
  `pto.vmi.vstore` of the original wide value;
- scalar-like conversion casts are folded when they form a proven one-lane
  broadcast round trip.

These are use-constrained rewrites, not general vector-shape conversion. Any
unexpected user leaves the cast unresolved and causes pass failure.

## Decision 18: Replace Stored Negative-Infinity Initializers Narrowly

The validated softmax path stores an `f32 -inf` reduction initializer in
rank-zero UB. The current bridge replaces that constant only when it directly
feeds `pto.store_scalar`, using `-FLT_MAX` instead. No PTOAS file is modified.

This is a compatibility workaround with a known semantic limitation:
`-FLT_MAX` is an equivalent max identity only when the row values are finite
and above it. Inputs containing `-inf`, NaNs, or other exceptional cases may
behave differently. The rewrite must remain narrow until PTOAS accepts the
original value through the complete simulator path or a more faithful bridge
representation is proven.

## Decision 19: Stage Simulator Fixtures Into Clean Build Trees

The Planner bridge and comparison runners copy each testcase fixture into its
output directory, remove any copied `build/` directory, and then invoke its
`run_sim.sh` with the generated VPTO path. Testcase source trees can contain
local build output, but copied CMake caches embed absolute source paths and
cannot be reused from the staged directory.

The runner derives the generated kernel path from the testcase name; users do
not need to set `KERNEL_MLIR` for normal bridge runs. Missing simulator fixture
files are reported before attempting the simulator build.

## Non-Direct Mapping Index

This table indexes every currently documented mapping where the bridge does
more than rename an operation while preserving the same operands and results.

| Source contract | Bridge decision |
| --- | --- |
| HIVM GM/UB memrefs and function ABI | Collapse accepted memrefs to PTO pointers; convert signatures and calls (Decisions 6, 8, 10) |
| signless `i8` GM memref element | Use `ui8` PTO pointer element for the proven PTOAS ABI (Decision 6) |
| metadata/reinterpret/memory-space view chain | Materialize constants and `pto.addptr`, or erase an identity space cast (Decisions 6 and 8) |
| `pge <ALL>` | Emit `pto.vmi.pset "PAT_ALL"`; fixed tails use target `PAT_VL*` names (Decisions 3 and 5) |
| masked scalar broadcast | Require all-active mask and emit unmasked VMI broadcast (Decision 4 and broadcast pattern) |
| all-active masked arithmetic/store | Emit the target unmasked form where required; supported partial stores use `vstore` (Decision 5) |
| `set_ctrl` one-bit update | Read-modify-write the complete PTO control register (Decision 9) |
| static/dynamic flag event representation | Rebuild PTO typed enums or choose target dynamic flag/wait op; preserve ordering (Decision 9) |
| AVE reduction result width | Reduce to one lane, then broadcast when AVE requires a wider result (Decision 13) |
| signed integer `vsmins`/`vsmaxs` | Signed vector bitcasts, a scalar signedness-only cast, and direct VMI scalar min/max (Decision 14) |
| `vload <BRC_B32>` | Multi-lane VMI broadcast load; one-lane scalar load plus broadcast (Decision 16) |
| rank-zero scalar and one-point stores | PTO scalar store, zero-offset vector store, or `PAT_VL1` masked store (Decision 17) |
| stored `f32 -inf` initializer | Narrow replacement with `-FLT_MAX` for the finite-input softmax path (Decision 18) |
| source module/function metadata | Remove source-only attributes and emit the minimal PTOAS kernel contract (Decision 11) |
