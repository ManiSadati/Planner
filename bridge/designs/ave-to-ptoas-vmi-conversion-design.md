# AVE to PTOAS VMI Conversion Design Decisions

Last updated: 2026-08-20

This document records bridge design decisions as they are made during
implementation. The implementation plan remains in
`Planner/bridge/planning/ave-to-ptoas-vmi-implementation-plan.md`; this file is
the running design log for choices that affect code structure.

The signed integer vector-scalar min/max compatibility problem and its
temporary constant-only lowering are documented separately in
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

The exact Stage 1 artifact now has an automated conversion regression and has
been manually validated through PTOAS `--emit-vpto`. The next step is to decide
whether to wire PTOAS discovery into automated testing or move on to the next
concrete kernel artifact.

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

## Decision 4: Add Vector-Scalar Ops As One Family

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

## Decision 9: Lower Only Observed Static Sync And Control Forms

Stage 1 synchronization/control conversion is intentionally limited to the
forms present in `lowered_vector_add_kernel.mlir`:

- `hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]` maps to
  `pto.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]`.
- `hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]` maps to
  `pto.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]`.
- `hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]` maps to
  `pto.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]`.
- `hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]` maps to
  `pto.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]`.
- `hivm.hir.pipe_barrier[<PIPE_ALL>]` maps to `pto.barrier <PIPE_ALL>`.

Dynamic event IDs and static event IDs other than `EVENT_ID0` are rejected in
this slice. Pipes other than `PIPE_MTE2`, `PIPE_MTE3`, `PIPE_V`, and
`PIPE_ALL` are also rejected until a concrete artifact requires them.

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
