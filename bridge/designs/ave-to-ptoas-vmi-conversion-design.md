# AVE to PTOAS VMI Conversion Design Decisions

Last updated: 2026-08-17

This document records bridge design decisions as they are made during
implementation. The implementation plan remains in
`Planner/bridge/planning/ave-to-ptoas-vmi-implementation-plan.md`; this file is
the running design log for choices that affect code structure.

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

Refactor the existing `convert-hivmave-to-ptoas-vmi` pass from preflight-only
validation into an MLIR dialect-conversion skeleton:

1. Add the `TypeConverter` for the Phase 1 supported types.
2. Add the `ConversionTarget` legality rules.
3. Move local legality checks into conversion patterns.
4. Keep the first slice focused on rejecting unsupported input cleanly before
   broadening operation coverage.

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
- rank-1 identity HIVM UB memrefs become PTO VEC memrefs.

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
