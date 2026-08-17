# AVE to PTOAS VMI Conversion: Final Implementation Plan

Last reviewed: 2026-08-11

Status: finalized planning baseline. No implementation is authorized by this
document itself. Phase 0 contract checks must pass before conversion code is
written.

## 1. Problem Summary

Add an MLIR conversion pass to AscendNPU-IR that converts the vector portion of
AscendNPU-IR's HIVMAVE/AVE IR (`ave.hir.*`) into PTOAS producer-boundary VMI IR
(`pto.vmi.*`). The pass is for Vector-core kernels only. Cube operations, DMA
between GM/L1/L0/UB, synchronization, and mixed Cube/Vector orchestration are
outside this pass.

The source and target layers are close enough for a staged one-to-one lowering
of common vector operations, but they are not type- or ABI-compatible by name
alone. The bridge must also convert logical vector and predicate types, UB
memory-space attributes, memory indices, function/control-flow type edges, and
module/kernel metadata. It must reject every semantic case it does not preserve.

## 2. Reviewed Baseline

This plan was checked against the following local source-of-truth revisions:

| Repository | Revision reviewed | Relevant source |
| --- | --- | --- |
| AscendNPU-IR | `master` at `08031590767ffd9e29a1a83e1f881c825af32dab` | `HIVMAVEOps.td`, `HIVMAVEAttrs.td`, `HIVMAVETypes.td`, conversion pass registration, regbase pipeline |
| PTOAS | `elemntwise-1d-2d-versions` at `2f2ea5bcab5010f72b8973bd6c4461e2b8b3f866` | `VMIOps.td`, `VMITypeDefs.td`, `VMI.cpp`, VMI validation/lowering pipeline, VMI lit tests |
| Planner | current worktree on 2026-08-11 | mapping draft, lowering summaries, prototype branch review |

The PTOAS branch name is recorded deliberately: VMI is evolving quickly, so
implementation must pin or revalidate the exact PTOAS revision used for the
bridge build.

Key source locations:

- Ascend source operations and contracts:
  `AscendNPU-IR/bishengir/include/bishengir/Dialect/HIVMAVE/IR/`.
- Ascend late regbase pipeline:
  `AscendNPU-IR/bishengir/lib/Tools/bishengir-compile/regbase/PassPipeline.cpp`.
- Ascend pass conventions:
  `AscendNPU-IR/bishengir/include/bishengir/Conversion/Passes.td` and
  `AscendNPU-IR/bishengir/lib/Conversion/CMakeLists.txt`.
- PTOAS VMI operations and types: `PTOAS/include/PTO/IR/VMIOps.td` and
  `PTOAS/include/PTO/IR/VMITypeDefs.td`.
- PTOAS producer-boundary verifier:
  `PTOAS/lib/PTO/Transforms/PTOValidateVMIIR.cpp`.
- PTOAS semantic pipeline: `appendVMISemanticPipeline` in
  `PTOAS/tools/ptoas/ptoas.cpp`.
- PTOAS compiler-facing container contract: `PTOAS/docs/vpto-spec.md`.

The reviewed Soyu-Wilson prototype remains reference material only. Its strict
unsupported-op audit and interception boundary are useful, but its local
unknown-op bridge dialect is not an acceptable final contract.

## 3. Final Architecture Decisions

### 3.1 Source boundary

The semantic interception point is after the HIVMAVE optimization pipeline has
run and before any of these late conversions:

```text
buildLowerAVEPipelines
  -> [new vector-kernel legality check]
  -> [new convert-hivmave-to-ptoas-vmi pass]
  -> [PTOAS VMI output/container preparation]
```

The PTOAS path must branch before:

```text
convert-hivm-to-std
convert-hivmave-to-std
expand-strided-metadata
convert-hivmave-to-ave-intrin
```

This is stricter than merely inserting the pass immediately before
`convert-hivmave-to-ave-intrin`. `convert-hivm-to-std` can erase out-of-scope
HIVM DMA/Cube/sync structure into library calls, making a vector-only legality
check unreliable. `convert-hivmave-to-std` also rewrites selected AVE cases on
some architectures. A dedicated PTOAS branch preserves diagnostics and avoids
running CCE-specific lowering after VMI creation.

The standalone conversion pass must still be independently invocable through
`bishengir-opt` on an already normalized AVE module.

### 3.2 Target contract

Emit real PTOAS dialect operations and types, not string-named unknown
operations and not a local shadow dialect. The target is surface VMI IR:

- `!pto.vmi.vreg<NxT>` without a physical layout;
- `!pto.vmi.mask<Nxpred>` without a physical layout;
- native `pto.vmi.*` semantic operations;
- only ordinary `builtin`, `func`, `scf`, `cf`, `memref`, and scalar support
  operations around VMI values;
- no `pto.vmi.ensure_*`, physical `!pto.vreg`/`!pto.mask`, or VPTO operations.

PTOAS owns signless-integer normalization where supported, mask granularity
assignment, physical vector layout assignment, and VMI-to-VPTO lowering. The
Ascend pass must not preselect a physical VMI layout.

### 3.3 Dependency model

The production pass should use a typed PTO dialect built inside AscendNPU-IR
from PTO TableGen sources copied from PTOAS. Do not link the pass against
PTOAS's installed `libPTOIR.a`, and do not require `find_package(PTOAS)`.

Guard this with an AscendNPU-IR build option such as
`BISHENGIR_ENABLE_PTOAS_VMI`, defaulting to `OFF`, so the normal Ascend build
does not acquire an unconditional PTO dialect/conversion dependency until the
in-tree PTO dialect is available.

The copied PTO dialect must be regenerated with AscendNPU-IR's MLIR TableGen
and compiled against AscendNPU-IR's MLIR runtime. This avoids linking MLIR
objects built against different LLVM/MLIR ABIs.

A textual file may be used as an integration artifact between the two command
line tools. A textual unknown-op shim is not the production implementation. If
AscendNPU-IR cannot build against PTOAS's LLVM revision, stop and resolve the
toolchain alignment or explicitly approve a separate textual-export
architecture.

### 3.4 Driver integration

Add a mutually exclusive output/backend selection rather than a loose boolean
that can coexist with `lower-to-llvm`. The intended behavior is:

```text
backend = cce/llvm       -> existing late pipeline, unchanged
backend = ptoas-vmi      -> normalized HIVMAVE -> VMI -> textual MLIR output
```

For the first implementation milestone, register and test the pass in
`bishengir-opt`. Integrate `bishengir-compile` only after the pass output passes
PTOAS validation and the kernel/container ABI is settled.

## 4. Scope

### In scope

- One-dimensional logical vector values on the Vector side.
- AVE predicates and vector arithmetic with proven VMI semantics.
- AVE UB vector loads and stores whose address can be represented as one PTOAS
  element offset.
- Type conversion across private function, block, SCF, and CF boundaries.
- Conversion of HIVM UB memref memory-space attributes to PTO VEC/UB memory
  space where the underlying storage is unchanged.
- Strict diagnostics for unsupported AVE/HIVM operations and attributes.
- PTOAS producer-boundary verification and VMI-to-VPTO smoke validation.

### Out of scope

- Cube and matrix operations, including `mmadL1` and `mma*`.
- HIVM DMA and layout movement between memory levels, including `hivm.hir.load`,
  `hivm.hir.store`, `nd2nz`, L1/L0 movement, and GM kernel data movement.
- Synchronization and barriers, including `set_flag`, `wait_flag`,
  `sync_block*`, and `ave.hir.membar`.
- Mixed AIC/AIV modules and host-stub generation.
- Selecting physical VMI register layouts or masks.
- Approximating unsupported operations with math sequences unless a separate
  semantic and numerical review approves that expansion.

## 5. Pass Location and Structure

Create the conversion under AscendNPU-IR using existing naming conventions:

```text
bishengir/include/bishengir/Conversion/HIVMAVEToPTOASVMI/
  HIVMAVEToPTOASVMI.h

bishengir/lib/Conversion/HIVMAVEToPTOASVMI/
  HIVMAVEToPTOASVMI.cpp
  CMakeLists.txt

bishengir/test/Conversion/HIVMAVEToPTOASVMI/
```

Register a `ModuleOp` pass named:

```text
convert-hivmave-to-ptoas-vmi
```

Update, behind the PTOAS build option:

- `bishengir/include/bishengir/Conversion/Passes.td`;
- `bishengir/include/bishengir/Conversion/Passes.h`;
- `bishengir/include/bishengir/Conversion/CMakeLists.txt`;
- the relevant top-level CMake package discovery/configuration;
- `bishengir/lib/Tools/bishengir-compile/regbase/PassPipeline.cpp` only in the
  driver-integration phase.

Use MLIR dialect conversion, not a greedy rename pass and not a separate
operation-by-operation preflight verifier:

1. Configure a `TypeConverter` for vectors, masks, and supported memrefs.
2. Configure a `ConversionTarget` that marks AVE illegal, PTOAS VMI legal, and
   structural dialects dynamically legal only when their types are converted and
   acceptable to the producer-boundary contract.
3. Add one conversion pattern per supported AVE operation family. Each pattern
   performs only the local semantic checks required to preserve that operation,
   such as `pge <ALL>`, `vload <NORM>`, all-active masks, UB memory, rank-1
   offsets, and supported element types.
4. Use `rewriter.notifyMatchFailure` and/or operation diagnostics from the
   failing pattern so unsupported operations and attributes report at their
   original locations.
5. Add materializations only when they have defined semantics. Do not leave
   `unrealized_conversion_cast` as successful output.
6. Use standard Func/SCF/CF signature and region conversion patterns for
   converted block arguments and yields.
7. Let `applyFullConversion` reject the whole module when an AVE operation,
   type, attribute, or region edge has no legal conversion.
8. Keep only genuinely module-wide checks outside the conversion patterns, such
   as mixed Cube/Vector legality, entry ABI/container rules, and final residual
   assertions.

Keep container/entry-point adaptation separate from operation conversion if it
requires more than module attributes and memory-space type rewriting. A small
second pass is preferable to coupling VMI op patterns to host/kernel ABI rules.

## 6. Type and Memory Contract

### 6.1 Data vectors

Initial mapping:

```text
vector<NxT> -> !pto.vmi.vreg<NxT>
```

Phase 1 supports only rank-1 built-in vectors with `f16`, `bf16`, or `f32`
elements and no AVE vector-layout encoding. Add integer and FP8 types only
after PTOAS signedness and storage-type verification passes for each case.

Do not initially accept:

- rank-0 or rank-greater-than-1 vectors;
- `!ave.pad_vec`;
- vectors carrying AVE layout metadata;
- vectors wider than the target PTOAS revision accepts;
- `vector<Nxi1>` when it is data rather than a predicate.

### 6.2 Masks

Initial mapping:

```text
vector<Nxi1> used as an AVE predicate -> !pto.vmi.mask<Nxpred>
```

The mask lane count must equal the governed VMI vector lane count. Surface VMI
uses `pred`; do not copy AVE's physical `b8`/`b16`/`b32` mask width into the
surface type. PTOAS assigns concrete mask granularity later.

`!ave.mask<...>` and predicate vectors with AVE layout/part metadata are
deferred until their logical lane interpretation is proven. The AVE type mixes
logical and physical-mask information, so copying its fields mechanically is
unsafe.

### 6.3 Memrefs and address spaces

For an accepted AVE vector load/store, convert:

```text
memref<..., #hivm.address_space<ub>>
  -> memref<..., #pto.address_space<vec>>
```

PTOAS VMI memory verifiers require UB-backed storage. Do not convert GM, L1,
L0A, L0B, L0C, SSBUF, or unknown spaces in this pass.

Phase 1 accepts only rank-1, identity/unit-stride UB memrefs and exactly one
`index` operand. AVE indices and PTOAS offsets are treated as element offsets
only after a source test confirms that contract.

Later multidimensional support must compute a single element offset from
strided memref metadata:

```text
linear = base_offset + sum(index[i] * stride[i])
```

Reject non-strided affine layouts, unsupported subviews, byte-based offsets,
or dynamic metadata that cannot be represented without changing semantics.

### 6.4 Function and control-flow boundaries

- Entry functions must not expose VMI vector/mask values in their public ABI.
- Private helper signatures may be converted if PTOAS inlining and validation
  accept them.
- Convert SCF/CF block arguments, loop-carried values, yields, branches, and
  returns consistently through the `TypeConverter`.
- Scalar `index`, integer, and floating-point values remain standard MLIR
  types unless an individual VMI op requires a narrower type.
- `ave.hir.plt` uses `index`, while `pto.vmi.plt` requires `i32`; this is not a
  direct type conversion and remains deferred until range and truncation
  semantics are established.

## 7. Operation Mapping Plan

### 7.1 Phase 1 vertical slice

The first slice deliberately uses all-active compute and store masks so
inactive-lane semantics cannot be lost.

| AVE source | PTOAS VMI target | Phase 1 restrictions |
| --- | --- | --- |
| `ave.hir.pge <ALL>` | `pto.vmi.pset "PAT_ALL"` | Built-in predicate vector; lane count matches users |
| `ave.hir.vload <NORM>` | `pto.vmi.vload` with `dist_mode = "continuous"` | One result; rank-1 unit-stride UB memref; one element offset |
| `ave.hir.vadd` | `pto.vmi.vadd` | `f16`/`bf16`/`f32`; governing mask is statically all-active and omitted at target |
| `ave.hir.vsub` | `pto.vmi.vsub` | Same restrictions as `vadd` |
| `ave.hir.vmul` | `pto.vmi.vmul` | Same restrictions as `vadd` |
| `ave.hir.vabs` | `pto.vmi.vabs` | Same restrictions as `vadd` |
| `ave.hir.masked_store <NORM_B*>` | unmasked `pto.vmi.vstore` | Store pattern matches element width; mask is statically all-active |

Representative source:

```mlir
%all = ave.hir.pge <ALL> : vector<64xi1>
%a = ave.hir.vload <NORM> %lhs[%off]
    : memref<64xf32, #hivm.address_space<ub>> into vector<64xf32>
%b = ave.hir.vload <NORM> %rhs[%off]
    : memref<64xf32, #hivm.address_space<ub>> into vector<64xf32>
%sum = ave.hir.vadd %a, %b, %all : vector<64xf32>, vector<64xi1>
ave.hir.masked_store <NORM_B32> %dst[%off], %all, %sum
    : memref<64xf32, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
```

Expected semantic VMI body:

```mlir
%all = pto.vmi.pset "PAT_ALL" : !pto.vmi.mask<64xpred>
%a = pto.vmi.vload %lhs[%off] {dist_mode = "continuous"}
    : memref<64xf32, #pto.address_space<vec>> -> !pto.vmi.vreg<64xf32>
%b = pto.vmi.vload %rhs[%off] {dist_mode = "continuous"}
    : memref<64xf32, #pto.address_space<vec>> -> !pto.vmi.vreg<64xf32>
%sum = pto.vmi.vadd %a, %b
    : !pto.vmi.vreg<64xf32>, !pto.vmi.vreg<64xf32>
      -> !pto.vmi.vreg<64xf32>
pto.vmi.vstore %sum, %dst[%off]
    : !pto.vmi.vreg<64xf32>, memref<64xf32, #pto.address_space<vec>>
```

The unused `%all` may be removed by canonicalization after all users are
converted. The conversion itself must not depend on that cleanup for legality.

### 7.2 Phase 2 predicates, tails, and basic control flow

Add only after Phase 1 passes PTOAS validation:

| AVE source | Candidate VMI target | Required proof |
| --- | --- | --- |
| `pge <VL1..VL128>` | `pto.vmi.pge "PAT_VL<n>"` | Active-lane count does not exceed logical mask length |
| tail `masked_store` | `pto.vmi.masked_store`, or corrected unified `vstore` contract | Inactive lanes skip writes exactly as AVE does |
| `preg.and/or/xor/not` | VMI mask logic ops | Governing-mask behavior and lane count match |
| `vcmp`, `vcmps` | `pto.vmi.vcmp`, `pto.vmi.vcmps` | AVE comparison enum, seed mask, signedness, NaN behavior, and inactive result agree |
| `vsel` | `pto.vmi.vsel` | False-value and inactive-lane behavior agree |
| SCF/CF carrying vectors | structural SCF/CF with VMI types | Producer-boundary verifier and VMI layout assignment accept each construct |

The current PTOAS source contains a documentation/implementation conflict for
unified masked stores: `VMIvStoreOp` documentation describes `pmode="merge"`
as skip-write, while its verifier rejects `merge`. The legacy
`pto.vmi.masked_store` has a mask-governed contract and verifier. Do not choose
between these forms until PTOAS owners confirm the intended producer surface or
the verifier/documentation is fixed. Phase 1 avoids the conflict by accepting
only all-active stores.

### 7.3 Phase 3 arithmetic and conversions

Add families in small, independently tested groups:

- Direct candidates: `vdiv`, `vmin`, `vmax`, `vneg`, `vsqrt`, `vexp`, `vln`,
  `vrelu`, `vand`, `vor`, `vxor`, `vnot`.
- Signedness-sensitive candidates: `vsmax`, `vsmin`, `vumax`, `vumin`, right
  shifts, and all integer comparisons.
- Vector-scalar candidates: `vadds`, `vmuls`, `vmaxs`, `vmins`, `vshls`,
  `vshrs`; verify PTOAS scalar-width constraints, especially shift counts.
- Index/broadcast candidates: `vci`, `scalar_broadcast`, and `broadcast`.
- Conversion candidates through `pto.vmi.vcvt`: `vextf`, `vextsi`, `vextui`,
  `vtrunci`, `vtruncf`, `vfptosi`, `vfptoui`, `vsitofp`, and `vuitofp`.
  Map rounding, saturation, signedness, part, and packing attributes explicitly.
- Multiply/accumulate candidates: AVE `mull` to `pto.vmi.vmull` and `vmula` to
  the appropriate VMI accumulator operation, after result width and accumulator
  initialization are confirmed.

Masked arithmetic must remain deferred even when the operation name matches,
unless AVE inactive lanes are proven to match VMI `pmode` behavior. Full-lane
masked forms may be lowered as unmasked operations.

### 7.4 Deferred or currently unmapped

| AVE source family | Reason for deferral |
| --- | --- |
| `pge <M3/M4/H/Q/ALLF>` | No confirmed one-op unified VMI mapping; grouped/false-mask construction needs proof |
| `plt`, `pltm` | Result/update semantics and `index` versus `i32` contract differ |
| `vrsqrt` | No confirmed exact VMI operation; expansion may change numerical behavior |
| `vtrc`, `vrnd`, `vmod`, `vmodui`, `vdivfhp` | Rounding/remainder/high-precision semantics are not mapped |
| `vabs_diff`, `vsadd`, `vssub`, `vprelu`, `vlrelus` | Exact saturation/activation semantics need PTOAS confirmation |
| `vpack`, `vunpack`, `preg.cast` | Logical versus physical part/granularity semantics are unresolved |
| `vslide` | No confirmed direct logical VMI mapping |
| `vintlv`, `vdintlv` | VMI candidates exist, but lane/layout change contracts require dedicated tests |
| `ave.reduction` | AVE result lane, mask, initializer, reassociation, and VMI group semantics require a separate design |
| `vgather`, `vscatter` | Address space, index units/type, masking, OOB, and conflict behavior are unresolved |
| non-`NORM` load/store modes | Each AVE dist mode needs an explicit mapping to VMI `dist_mode`, group, stride, unpack, broadcast, or channel operation |
| `unaligned.masked.store`, `store_with_stride` | Alignment and block/repeat-stride units need proof |
| `vector.layout_cast`, `!ave.pad_vec` | PTOAS must own physical layout; AVE logical content/part must first be reconstructed |
| `membar` | Synchronization is outside the vector conversion pass |

## 8. Container and Kernel ABI

The conversion pass's operation-level output must first satisfy
`pto-validate-vmi-ir`. End-to-end `ptoas` compilation additionally needs a
compiler-facing VPTO container. The expected normalized form is:

```mlir
module attributes {pto.target_arch = "a5"} {
  module attributes {pto.backend = "vpto",
                     pto.kernel_kind = #pto.kernel_kind<vector>} {
    func.func @kernel(...) attributes {pto.kernel} {
      // Surface pto.vmi.* operations.
      return
    }
  }
}
```

Before implementing container wrapping, answer:

- Which Ascend function attribute identifies the launched AIV entry point?
- Are the bridge inputs already UB memrefs, or must a surrounding non-VMI layer
  preserve GM-to-UB movement?
- Should the bridge emit normalized compiler-facing submodules directly, or
  emit `pto.section.vector` source form and let PTOAS normalize it?
- How are Ascend memref arguments represented in PTOAS's external kernel ABI,
  where `!pto.ptr<..., gm/ub>` may be required?

Until these are resolved, Phase 1 is a pass-level VMI conversion, not a claim
that an arbitrary Ascend vector kernel is a launchable PTOAS kernel.

## 9. Phased Implementation

### Phase 0: Freeze the build and semantic contract

Tasks:

1. Pin the AscendNPU-IR and PTOAS revisions used by CI.
2. Copy or vendor the required PTO dialect TableGen sources into AscendNPU-IR
   and build them with AscendNPU-IR's LLVM/MLIR tree.
3. Confirm the in-tree PTO dialect target exports the C++ APIs used by the
   bridge pass.
4. Capture at least two real modules immediately after
   `buildLowerAVEPipelines`: one full-lane vector kernel and one tail kernel.
5. Inventory every remaining dialect/op in those modules before late Standard
   lowering.
6. Confirm AVE load/store indices are element offsets.
7. Resolve the PTOAS unified masked-store conflict.
8. Select and document the compiler-facing container and entry ABI.

Exit criteria:

- One shared LLVM ABI is demonstrated.
- A hand-authored VMI version of the full-lane kernel passes
  `pto-validate-vmi-ir` and lowers to VPTO.
- The real pre-conversion IR contains no unexplained operation needed by the
  first slice.
- Every Phase 1 mapping has written semantic acceptance criteria.

### Phase 1: Build the strict full-lane conversion slice

Tasks:

1. Add the optional PTOAS CMake integration and pass registration.
2. Implement rank-1 float vector, predicate, and UB memref type conversion.
3. Implement the dialect-conversion target and local pattern checks so
   unsupported AVE/HIVM operations reject the whole module during conversion.
4. Convert `pge <ALL>`, `vload <NORM>`, full-mask `vadd`/`vsub`/`vmul`/`vabs`,
   and full-mask normal stores.
5. Convert private Func/SCF/CF type edges needed by the captured kernel.
6. Reject residual AVE/HIVM, unsupported attributes, and unrealized casts.
7. Add Ascend lit tests and a PTOAS parse/verify/lower integration test.

Exit criteria:

- The representative load-add-store kernel converts without AVE residue.
- PTOAS parses the emitted text, passes `pto-validate-vmi-ir`, assigns layout,
  and lowers it to VPTO without VMI residue.
- Every unsupported operation in the test corpus fails with a stable,
  location-bearing diagnostic.
- Existing CCE/LLVM pipelines are unchanged when PTOAS support is disabled.

### Phase 2: Add tails, predicates, and control flow

Tasks:

1. Implement fixed `pge <VLn>` masks.
2. Implement masked stores using the PTOAS-approved representation.
3. Add predicate logic, compare, and select after semantic review.
4. Add SCF/CF cases for loops, `if`, and branch joins carrying VMI values.
5. Add multidimensional strided memref linearization only if required by a real
   kernel and validated against PTOAS memref restrictions.

Exit criteria:

- A real tail kernel preserves skip-write behavior.
- PTOAS mask granularity and layout assignment pass for each predicate path.
- Negative tests cover lane mismatch, offset overflow/range, unsupported
  layouts, and incompatible compare semantics.

### Phase 3: Expand operation coverage

Implement the Phase 3 groups from Section 7 one family at a time. For every
family:

1. Add a source/target mapping note with attributes and edge cases.
2. Add positive float/integer/type-width tests as applicable.
3. Add verifier-negative tests.
4. Run the full PTOAS semantic pipeline.
5. Do not mark the family supported until VMI-to-VPTO succeeds.

Keep reductions, gather/scatter, complex distribution modes, and layout-changing
operations as separate milestones because their blast radius is larger than a
one-to-one elementwise conversion.

### Phase 4: Integrate the compiler driver

Tasks:

1. Add the `ptoas-vmi` backend/output selection to compile configuration.
2. Branch after `buildLowerAVEPipelines` and before late Standard/intrinsic
   lowering.
3. Add vector-kernel legality validation before any HIVM op can become a
   library call.
4. Add container/entry-point adaptation using the Phase 0 ABI decision.
5. Emit textual MLIR suitable for the pinned PTOAS tool.
6. Add an end-to-end test from representative Ascend input to emitted VPTO.

Exit criteria:

- `bishengir-compile` can select the PTOAS VMI path without running the CCE
  intrinsic tail.
- Cube, DMA, sync, mixed-core, and unsupported vector kernels fail before
  partial output is emitted.
- The default backend remains byte-for-byte or FileCheck-equivalent to its
  pre-bridge behavior.

### Phase 5: Hardware and performance validation

Tasks:

1. Compile representative kernels through both the existing CCE path and the
   PTOAS path.
2. Validate numerical results, tails, aliasing, and memory bounds on A5 hardware
   or the approved simulator.
3. Compare VPTO/instruction shape, register parts, mask setup, and memory access
   count.
4. Benchmark representative vector sizes and alignment cases.
5. Record unsupported cases and performance regressions before broadening the
   enabled kernel set.

## 10. Test Strategy

### Ascend pass tests

Create focused lit files under
`bishengir/test/Conversion/HIVMAVEToPTOASVMI/`:

```text
basic-f32-load-add-store.mlir
basic-f16-bf16-arithmetic.mlir
type-and-memory-space-conversion.mlir
control-flow.mlir
tail-mask-and-store.mlir
unsupported-ops.mlir
unsupported-memory.mlir
unsupported-types-and-layouts.mlir
```

Checks must assert types, operation names, mask handling, attributes, offsets,
and the absence of `ave.hir`, AVE types/layouts, and unrealized casts. Avoid
tests that only check that a `pto.vmi` string appeared.

### PTOAS contract tests

Pipe or save the converted module and run, using tools built against the pinned
LLVM revision:

```bash
bishengir-opt --convert-hivmave-to-ptoas-vmi input.mlir \
  | pto-test-opt --pto-validate-vmi-ir
```

Then run the semantic lowering sequence or the PTOAS CLI:

```bash
ptoas --pto-arch=a5 --pto-backend=vpto --emit-vpto converted.pto -o -
```

The full check must confirm:

- producer-boundary validation succeeds;
- signless normalization, unified-to-legacy lowering, mask assignment, and
  layout assignment succeed;
- VMI-to-VPTO succeeds;
- no `pto.vmi.*`, `!pto.vmi.*`, or `unrealized_conversion_cast` remains in
  emitted VPTO.

Make cross-repository tests conditional on an explicitly discovered PTOAS test
tool and lit feature. Missing PTOAS must skip only bridge integration tests, not
silently weaken pass unit tests in a PTOAS-enabled build.

### Required negative cases

- Any Cube, DMA, sync, or mixed-core operation.
- Unsupported AVE operation or distribution mode.
- Non-UB memory and unknown memory spaces.
- Rank-greater-than-1 or non-unit-stride memory in Phase 1.
- Multiple AVE load results in Phase 1.
- Mask/data lane mismatch.
- AVE padded vectors, layout casts, layout attributes, and mask parts.
- Public kernel signatures exposing VMI values.
- Unsupported element types or integer signedness.
- Dynamic/byte offsets whose units cannot be proven.
- Masked compute with non-all-active masks before `pmode` semantics are proven.
- Residual AVE/HIVM ops or unrealized casts after conversion.

### Regression checks

- Run Ascend's conversion and dialect lit suites plus `check-bishengir`.
- Run PTOAS `check-pto`, with emphasis on `test/lit/vmi_new`.
- Run existing CCE pipeline tests with PTOAS support both disabled and enabled
  but unselected.
- Add at least one test proving the CCE path still reaches
  `convert-hivmave-to-ave-intrin` and the PTOAS path does not.

## 11. Assumptions

- The bridge targets the PTOAS VMI revision represented by the pinned PTOAS
  checkout, not an abstract or older VMI specification.
- Vector kernels reach a stable normalized HIVMAVE form before the late Standard
  and AVE intrinsic conversions.
- PTOAS remains responsible for physical vector layout and mask granularity.
- Initial AVE vector memory operations access UB, and GM/L1/L0 movement remains
  outside this pass.
- Phase 1 can be demonstrated with rank-1, unit-stride, full-lane floating-point
  operations.
- PTOAS A5/VPTO is the intended backend. If another architecture is required,
  re-run the target operation and verifier audit.

## 12. Open Questions and Blocking Decisions

Resolve these in order. Items 1-5 block production implementation; later items
block only the corresponding coverage phase.

1. **PTO dialect import:** Which PTOAS `.td` files and support sources are
   required to build the PTO dialect in AscendNPU-IR without linking
   `libPTOIR.a`?
2. **In-tree dialect target:** What will the copied PTO dialect library target be
   named, and should `MLIRHIVMAVEToPTOASVMI` link it directly or depend on an
   aggregate dialect library?
3. **Kernel/container ABI:** Which Ascend functions become PTOAS `pto.kernel`
   entries, and which PTOAS container form should the compiler emit?
4. **Memory ownership:** What mechanism supplies UB buffers in the PTOAS kernel
   when HIVM DMA is out of scope? Are bridge inputs already UB-resident?
5. **Offset units:** Are AVE load/store indices element offsets at the selected
   interception point for all accepted memrefs?
6. **Masked store:** Should the bridge emit legacy `pto.vmi.masked_store`, or
   will unified `pto.vmi.vstore` support skip-write/merge semantics?
7. **Masked compute:** What are AVE inactive result lanes for each unary,
   binary, ternary, compare, and vector-scalar family?
8. **Integer signedness:** When AVE uses signless MLIR integers, which operation
   or attribute is authoritative for signed/unsigned interpretation?
9. **Predicate types:** How should `!ave.mask`, layout-bearing predicate vectors,
   and `part` values be reconstructed as logical masks?
10. **Control flow:** Which SCF/CF constructs occur in real normalized vector
    kernels, and which are accepted by the pinned PTOAS layout pipeline?
11. **PLT range:** Can AVE `index` values be proven to fit PTOAS `i32`, and do
    both operations return the same next-remainder value?
12. **Numerical modes:** How do AVE rounding, saturation, high-precision,
    reciprocal, and conversion-part attributes map to PTOAS?

Every resolved answer should be added to the mapping documentation and encoded
in a verifier or test. Do not leave semantic decisions only in code comments.

## 13. Risks and Mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| VMI changes under the bridge | Build or verifier breakage | Pin PTOAS revision; run cross-repo contract tests; update mapping with each pin change |
| LLVM ABI mismatch | Link-time or runtime corruption | One shared LLVM 21 toolchain; never link cross-major MLIR libraries |
| Broad Standard lowering erases unsupported source ops | Invalid kernels appear vector-only | Branch before `convert-hivm-to-std`; perform legality audit first |
| Masked operations silently change inactive lanes | Wrong numerical results or stores | Full-mask-only Phase 1; prove `pmode`/skip-write semantics before tails |
| AVE physical layout leaks into VMI | PTOAS layout assignment conflict | Emit only surface VMI types; reject AVE layout/part cases initially |
| Address-space attrs share numeric values but differ by dialect | PTOAS verifier rejects memory or accepts the wrong space | Explicit HIVM-UB to PTO-VEC attribute conversion; reject all other spaces |
| Multidimensional indices are flattened incorrectly | Out-of-bounds or wrong-element access | Rank-1 first; use structured memref metadata and element-offset tests later |
| Partial conversion leaves mixed AVE/VMI IR | Downstream failure far from source | AVE dialect illegal, residual-op audit, no unrealized casts |
| Conversion passes but kernel ABI is not launchable | False end-to-end confidence | Separate producer-boundary and container gates; require PTOAS CLI test |
| One-to-one IR mapping regresses performance | Correct but poor code | Compare emitted VPTO/instructions and benchmark before enabling broadly |

## 14. Definition of Done

The bridge is complete for its declared vector subset when:

- AscendNPU-IR contains a typed, optionally built
  `convert-hivmave-to-ptoas-vmi` pass in the standard conversion hierarchy.
- The driver has an explicit PTOAS VMI path that branches before late Standard
  and AVE intrinsic lowering.
- Supported kernels contain only valid surface VMI plus permitted structural
  MLIR at the handoff.
- Unsupported Cube, DMA, sync, memory, type, layout, and vector cases fail early
  with actionable diagnostics.
- Converted modules pass PTOAS producer-boundary verification and lower through
  layout assignment and VMI-to-VPTO without residual VMI or casts.
- At least one full-lane kernel and one tail kernel are numerically validated on
  the approved target/simulator.
- Existing Ascend CCE behavior is unchanged when the PTOAS backend is not
  selected.
- The supported operation/type/attribute matrix and remaining gaps are recorded
  in Planner and match the tests.
