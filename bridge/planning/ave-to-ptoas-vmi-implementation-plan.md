# AVE to PTOAS VMI Conversion Implementation Plan

Last revised: 2026-08-17

Status: Stage 1 conversion implemented and manually validated through PTOAS.
Stage 1 is defined by one concrete end-to-end acceptance artifact,
`AscendNPU-IR/lowered_vector_add_kernel.mlir`.

## 1. Stage 1 Goal

The first stage is complete when this command succeeds:

```bash
AscendNPU-IR/build/bin/bishengir-opt \
  AscendNPU-IR/lowered_vector_add_kernel.mlir \
  --convert-hivmave-to-ptoas-vmi \
  -o /tmp/lowered_vector_add_kernel.pto.mlir
```

and PTOAS can consume the result and lower it through the VMI pipeline:

```bash
PTOAS/build/bin/ptoas \
  --pto-arch=a5 \
  --pto-backend=vpto \
  --emit-vpto \
  /tmp/lowered_vector_add_kernel.pto.mlir \
  -o /tmp/lowered_vector_add_kernel.vpto.mlir
```

The exact PTOAS binary path and architecture option may differ in the local
build. The selected architecture must be verified from the source HACC/DLTI
target metadata; `a5` is not assumed merely because it is shown above.

"Complete conversion" for Stage 1 means:

- the whole module, not only the outlined vector helper, is accepted;
- every `ave.hir.*` and `hivm.hir.*` operation is eliminated;
- every HIVM memory-space type is converted to a PTO representation;
- source-only AVE, HIVM, HACC, TT, and DLTI attributes are either mapped to a
  defined PTOAS contract or removed after their meaning has been discharged;
- vector computation is expressed as surface `pto.vmi.*` operations;
- DMA, synchronization, pointer, and control-register behavior needed by this
  kernel is expressed with non-VMI PTO operations;
- function signatures and `func.call` sites agree after type conversion;
- the result contains no `unrealized_conversion_cast`;
- MLIR verification and PTOAS VMI-to-VPTO lowering both succeed.

This definition intentionally permits ordinary `builtin`, `arith`, `func`,
`scf`, and the subset of `memref` operations accepted by PTOAS. It does not
mean that every operation in the output has a `pto.vmi` prefix.

## 2. Current Baseline

The pass already exists in AscendNPU-IR at:

```text
bishengir/lib/Conversion/HIVMAVEToPTOASVMI/HIVMAVEToPTOASVMI.cpp
```

The PTO dialect is built in-tree from sources under:

```text
bishengir/include/PTOAS/PTO/IR/
bishengir/lib/PTOAS/PTO/IR/
```

There is no external `libPTOIR.a` dependency and no PTOAS feature flag. The
pass is always registered in `bishengir-opt`.

Implemented and tested support currently includes:

- rank-1 `f16`/`bf16`/`f32` data vectors to `!pto.vmi.vreg`;
- predicate vectors to `!pto.vmi.mask`;
- accepted HIVM GM/UB memrefs to PTO pointers;
- `ave.hir.pge` for `ALL` and fixed `VL*` patterns;
- `ave.hir.vload`, selected unary/binary/vector-scalar arithmetic, and normal
  masked stores;
- `memref.extract_strided_metadata` and contiguous rank-1
  `memref.reinterpret_cast` folded to PTO pointer operations;
- `hivm.hir.pointer_cast`, redundant `memref.memory_space_cast`, and existing
  PTO MTE op operand remapping for the exact pointer memory graph;
- static `set_ctrl`, `set_flag`, `wait_flag`, and `PIPE_ALL` barrier
  conversion for the exact artifact;
- direct `func.call` conversion for the outlined vector helper;
- Func/SCF structural type conversion for the focused tests.

The current pass converts the whole acceptance artifact through
`bishengir-opt`, emits the PTOAS kernel/vector metadata contract, and the
result lowers through PTOAS `--emit-vpto`. Phase 1G should turn this into an
acceptance-style regression and record the exact handoff command.

## 3. Acceptance Artifact Inventory

### 3.1 Outlined vector helper

`@vector_add_kernel_outlined_vf_0` contains:

- three static rank-1 UB memref arguments;
- an `scf.for` over four 64-element chunks;
- `memref.extract_strided_metadata` and `memref.reinterpret_cast` view chains;
- two normal AVE vector loads;
- full-lane `pge <ALL>` predicates;
- one AVE vector add;
- one full-lane normal masked store;
- source attributes `functionType` and `hivm.is_continuous`;
- source-only function attributes.

The vector operations themselves are already covered. The remaining helper
work is attribute handling and choosing a pointer/memref representation that
also composes with the entry function and PTOAS ABI.

### 3.2 Entry function

`@vector_add_kernel` additionally contains:

- dynamic GM memref arguments and source argument metadata;
- three static rank-1 GM reinterpret casts;
- three HIVM UB pointer casts from integer byte addresses;
- HIVM-to-PTO `memref.memory_space_cast` operations;
- existing `pto.mte_gm_ub` and `pto.mte_ub_gm` operations;
- static `set_flag` and `wait_flag` synchronization;
- three `set_ctrl` bit updates;
- one `PIPE_ALL` barrier;
- a call to the outlined helper;
- source module, function, and argument attributes.

These operations make a narrow amount of outer-kernel conversion part of Stage
1. This is not a commitment to support general HIVM DMA, synchronization, or
arbitrary kernel ABI conversion.

## 4. Stage 1 Mapping Contract

### 4.1 Confirmed mappings

| Source | Target | Stage 1 restriction |
| --- | --- | --- |
| `vector<64xf32>` | `!pto.vmi.vreg<64xf32>` | Existing mapping |
| `vector<64xi1>` predicate | `!pto.vmi.mask<64xpred>` | Existing mapping |
| `ave.hir.pge <ALL>` | `pto.vmi.pset "PAT_ALL"` | Existing mapping |
| `ave.hir.vload <NORM>` | `pto.vmi.load` | Contiguous UB source |
| `ave.hir.vadd` | `pto.vmi.vadd` | Direct full-lane mask |
| full-lane `masked_store <NORM_B32>` | `pto.vmi.store` | `f32`, contiguous UB destination |
| HIVM static `set_flag` | `pto.set_flag` | Static pipe pair and `EVENT_ID0` |
| HIVM static `wait_flag` | `pto.wait_flag` | Static pipe pair and `EVENT_ID0` |
| `hivm.hir.pipe_barrier[PIPE_ALL]` | `pto.barrier #pto.pipe<PIPE_ALL>` | Explicit enum mapping |
| existing `pto.mte_gm_ub` | preserve | Rewrite operands only |
| existing `pto.mte_ub_gm` | preserve | Rewrite operands only |
| `func.call` | converted `func.call` | Callee and caller signatures must match |

HIVM and PTO pipe/event attributes are different typed attributes even when
their printed enum names match. Conversion must construct PTO attributes
explicitly; it must not copy source attributes by name or numeric value.

### 4.2 Control-register mapping

`hivm.hir.set_ctrl <enable> at ctrl[<idx>]` updates one bit. PTO
`pto.set_ctrl` writes the complete `i64` control value. Each accepted HIVM op
therefore maps to a read-modify-write sequence:

```text
current = pto.get_ctrl
enable=true  -> updated = current OR  (1 << idx)
enable=false -> updated = current AND NOT(1 << idx)
pto.set_ctrl updated
```

Stage 1 accepts constant indices in `[0, 63]`, which covers bits 60 and 48 in
the artifact. Tests must check the exact masks and verify that consecutive
updates each read the current register state. Replacing these operations with
constants is not valid because the unspecified control bits must be preserved.

### 4.3 Source operation attributes

Do not keep the current all-or-nothing extra-attribute check for accepted AVE
operations. Handle attributes by semantic allowlist:

- `functionType` may be consumed and dropped only when it agrees with the
  already normalized operation and element width. For this artifact that means
  `norm` on normal `f32` loads/stores and `pb32` on `f32` predicates.
- `hivm.is_continuous` may be consumed and dropped only when the memory view is
  proven rank-1 and unit-stride.
- unknown AVE/HIVM attributes still reject the operation.

This preserves strict rejection while accepting metadata produced by the
normal Ascend lowering pipeline.

### 4.4 Memory and pointer representation decision

The preferred Stage 1 target is PTO pointers:

```text
GM kernel memref argument -> !pto.ptr<element, gm>
UB helper memref argument -> !pto.ptr<element, ub>
hivm.hir.pointer_cast(i64) -> pto.castptr i64 -> !pto.ptr<element, ub>
```

This aligns with PTOAS kernel examples and lets the existing MTE and VMI
operations consume the same values. PTO `castptr`, MTE buffer operands, and VMI
load/store all accept PTO pointer types.

This preference must be proven before implementation by constructing the
expected target form for the exact artifact and running PTOAS. If PTOAS
requires a different ABI for the first two special `i8` arguments, record that
explicitly rather than preserving a dynamic memref descriptor by accident.

For the helper's view chain, the expected pointer interpretation is:

```text
extract base + reinterpret offset %iv, size 64, stride 1
  -> pto.addptr %base, %iv
```

The conversion may fold the metadata/view pair only when:

- the base result is the only live metadata result;
- the reinterpret view is rank-1;
- its element type is unchanged;
- its stride is statically one;
- its offset is an element offset accepted by `pto.addptr`;
- size 64 is compatible with the VMI vector user.

Any other metadata/view use remains unsupported in Stage 1. Existing generic
UB-memref support may remain for focused unit tests, but the exact acceptance
path must not produce unrealized memref descriptor casts.

The entry function's zero-offset GM reinterpret casts should become the
original converted GM pointer. The HIVM-to-PTO `memref.memory_space_cast`
operations then become redundant and must be erased. Do not emit an identity
memory-space cast merely to preserve source shape.

### 4.5 Module and function contract

The output must carry a PTOAS compiler-facing vector-kernel contract:

- map the confirmed Ascend target to `pto.target_arch`;
- mark the launched entry with `pto.kernel`;
- place the kernel in a PTOAS-supported vector container, either the
  recommended `pto.section.vector` source form or a normalized module carrying
  `pto.kernel_kind = #pto.kernel_kind<vector>`;
- keep the outlined function as a non-kernel helper if PTOAS accepts the call;
- remove source attributes that PTOAS cannot parse and that no longer affect
  semantics.

The exact container form and target-architecture mapping are validation tasks,
not assumptions. Prefer the least invasive form that the current PTOAS CLI
accepts for this module.

## 5. Ordered Implementation Phases

### Phase 1A: Freeze the expected PTOAS target

1. Create a hand-authored PTOAS version of the exact acceptance artifact.
2. Use PTO pointers for GM and UB values and preserve the existing MTE ops.
3. Represent the helper slice with `pto.addptr` and VMI load/add/store.
4. Add PTO flags, waits, control read-modify-write sequences, and barrier.
5. Select the container, entry marker, target architecture, and treatment of
   the special sync/workspace arguments.
6. Run PTOAS through `--emit-vpto` until this target succeeds without VMI or
   unrealized-cast residue.

Exit criterion: one reviewed target file proves the desired output contract.

Phase 1A result as of 2026-08-17:

- The hand-authored target is
  `Planner/bridge/validation/lowered_vector_add_kernel.expected.pto`.
- It uses module attributes `pto.target_arch = "a5"` and
  `pto.kernel_kind = #pto.kernel_kind<vector>`.
- The entry keeps the source eight-argument shape with five GM memrefs
  converted to PTO GM pointers and the three scalar `i32` arguments preserved.
- The two special `i8` GM arguments are represented as unused
  `!pto.ptr<ui8, gm>` arguments in the target artifact.
- The outlined helper remains a private `func.func` with `no_inline` and PTO UB
  pointer arguments.
- Helper metadata/reinterpret views are represented as `pto.addptr` with the
  loop induction variable as an element offset.
- HIVM `pointer_cast` byte offsets are represented by `pto.castptr`; the
  nonzero `1024` byte UB address is intentionally not rewritten as an element
  offset.
- The file lowers successfully with:

  ```bash
  "$HOME/PTOAS/build/tools/ptoas/ptoas" \
    --pto-arch=a5 \
    --pto-backend=vpto \
    --emit-vpto \
    "$HOME/Planner/bridge/validation/lowered_vector_add_kernel.expected.pto" \
    -o /tmp/lowered_vector_add_kernel.expected.vpto.mlir
  ```

- The emitted VPTO contains `pto.vlds`, `pto.vadd`, `pto.vsts`, MTE,
  control-register, and barrier operations, with no `pto.vmi`, `!pto.vmi`, or
  `unrealized_conversion_cast` residue.

### Phase 1B: Accept normalized AVE attributes

1. Replace the blanket extra-attribute rejection with per-operation
   allowlists.
2. Validate and consume `functionType` for the exact load, predicate, and store
   forms in the artifact.
3. Validate and consume `hivm.is_continuous` on the store.
4. Add negative tests for mismatched and unknown attributes.

Exit criterion: the outlined helper reaches memory/type conversion without
weakening unsupported-attribute rejection.

Phase 1B result as of 2026-08-17:

- `functionType` is consumed by per-operation allowlist for accepted `pge`,
  normal `vload`, and normal `masked_store` forms.
- `hivm.is_continuous` is consumed only on the accepted normal masked-store
  path.
- Unknown and mismatched AVE operation attributes still reject conversion.
- The focused `HIVMAVEToPTOASVMI` test directory passes.
- Running the pass on `lowered_vector_add_kernel.mlir` now proceeds past the
  previous `functionType` failure and first fails on `hivm.hir.set_ctrl` at
  line 26, which is covered by Phase 1E.

### Phase 1C: Convert the exact memory graph to PTO pointers

1. Extend the type converter for the accepted GM and UB memref forms.
2. Convert entry and helper function signatures to the target pointer ABI.
3. Convert HIVM `pointer_cast` to `pto.castptr` for one `i64` address and a
   static rank-1 result.
4. Fold the accepted metadata/reinterpret view chain to `pto.addptr`.
5. Fold zero-offset GM reinterpret casts to their source pointers.
6. Remove redundant `memref.memory_space_cast` operations.
7. Make the already-PTO MTE ops dynamically legal only when all buffer operands
   have legal PTO types.
8. Reject multi-address pointer casts, dynamic UB shapes, non-unit strides,
   element-type changes, and live metadata results not represented in PTO.

Exit criterion: all memory values in the artifact are PTO pointers and there
are no HIVM memory spaces or unrealized casts.

Phase 1C result as of 2026-08-17:

- Accepted GM and UB memrefs now convert to PTO pointer types.
- Signless `i8` GM memrefs convert to `!pto.ptr<ui8, gm>` for the special
  sync/workspace arguments proven in Phase 1A.
- `hivm.hir.pointer_cast` with one `i64` address converts to `pto.castptr`.
- Supported rank-one unit-stride `memref.extract_strided_metadata` and
  `memref.reinterpret_cast` chains fold to source pointers or `pto.addptr`.
- Redundant `memref.memory_space_cast` operations erase when the converted
  source and result pointer types match.
- Existing `pto.mte_gm_ub` and `pto.mte_ub_gm` ops are preserved with remapped
  PTO pointer operands.
- Focused positive and negative lit tests cover pointer memory graph
  conversion, MTE operand legalization, metadata folding, and malformed
  pointer casts.
- The `HIVMAVEToPTOASVMI` lit directory passes with 15 tests.
- Running the pass on `lowered_vector_add_kernel.mlir` now first fails on
  `hivm.hir.set_ctrl` at line 26, which is covered by Phase 1E.

### Phase 1D: Convert calls and structural control flow

1. Add the standard `func.call` type-conversion pattern.
2. Make `func.call` dynamically legal only when its operand/result types match
   the converted callee contract.
3. Keep the existing Func/SCF conversions for the helper loop.
4. Verify symbol resolution and helper visibility after conversion.

Exit criterion: the helper signature, call operands, and SCF body are all
legal under the same type converter.

Phase 1D result as of 2026-08-17:

- Direct `func.call` operations are rewritten with converted operand/result
  types.
- The rewritten helper call preserves `no_inline` and drops the source-only
  `hivm.vector_function` call attribute.
- `func.call` is dynamically legal only when its operand/result types are legal
  and the source-only call marker is gone.
- A focused call-conversion lit test covers the outlined-helper ABI shape.

### Phase 1E: Convert synchronization and control operations

1. Map the four static HIVM flag/wait operations to PTO operations using
   explicit pipe and event enum conversion helpers.
2. Map the `PIPE_ALL` barrier to `pto.barrier`.
3. Lower each HIVM `set_ctrl` to `pto.get_ctrl`, `arith` bit manipulation, and
   `pto.set_ctrl`.
4. Mark only these HIVM operations illegal so any other residual HIVM op still
   fails full conversion.
5. Add positive and negative unit tests for every accepted enum and control-bit
   form.

Exit criterion: no `hivm.hir.*` operation remains in the artifact.

Phase 1E result as of 2026-08-17:

- `hivm.hir.set_ctrl` lowers to `pto.get_ctrl`, `arith.ori`/`arith.andi`, and
  `pto.set_ctrl`.
- Static `PIPE_MTE2 -> PIPE_V` and `PIPE_V -> PIPE_MTE3` flag/wait pairs using
  `EVENT_ID0` lower to PTO `set_flag`/`wait_flag`.
- `hivm.hir.pipe_barrier[<PIPE_ALL>]` lowers to `pto.barrier <PIPE_ALL>`.
- Dynamic event IDs, non-`EVENT_ID0` static events, unsupported pipes, and
  out-of-range control bits remain unsupported.
- The focused `HIVMAVEToPTOASVMI` lit directory passes with 18 tests.
- Running the pass on `lowered_vector_add_kernel.mlir` succeeds and writes
  `/tmp/lowered_vector_add_kernel.phase1e.mlir`.
- The generated artifact has no `ave.hir`, `hivm.hir`, `#hivm.address_space`,
  or `unrealized_conversion_cast` residue. It still contains source-only
  attributes, which are handled by Phase 1F.

### Phase 1F: Adapt metadata and container

1. Map the confirmed Ascend architecture to `pto.target_arch`.
2. Mark `@vector_add_kernel` as the PTO kernel entry.
3. Add the selected vector container contract.
4. Preserve only generic attributes that PTOAS accepts and that remain useful.
5. Remove consumed `hacc.*`, `hivm.*`, `tt.*`, and source DLTI attributes from
   module, function, call, and argument dictionaries.
6. Reject multiple or ambiguous entry functions rather than guessing.

Exit criterion: PTOAS parses the generated text without registering Ascend
dialects.

Phase 1F result as of 2026-08-17:

- Module metadata is replaced with `pto.target_arch = "a5"` and
  `pto.kernel_kind = #pto.kernel_kind<vector>`.
- The original `hacc.entry` function is marked with `pto.kernel`.
- Source-only module, function, and argument attributes from HACC/HIVM/TT/DLTI
  are removed.
- Helper functions preserve `no_inline`.
- A focused metadata/container lit test covers the cleanup behavior.
- Running the pass on `lowered_vector_add_kernel.mlir` succeeds and writes
  `/tmp/lowered_vector_add_kernel.phase1f.mlir`.
- The generated artifact has no `hacc`, `hivm`, `tt.`, `dlti`,
  `unrealized_conversion_cast`, `ave.hir`, `hivm.hir`, or HIVM address-space
  residue.
- PTOAS lowers the generated artifact successfully with:

  ```bash
  "$HOME/PTOAS/build/tools/ptoas/ptoas" \
    --pto-arch=a5 \
    --pto-backend=vpto \
    --emit-vpto \
    /tmp/lowered_vector_add_kernel.phase1f.mlir \
    -o /tmp/lowered_vector_add_kernel.phase1f.vpto.mlir
  ```

- The emitted VPTO has no `pto.vmi`, `!pto.vmi`, or
  `unrealized_conversion_cast` residue.

### Phase 1G: End-to-end acceptance and regression

1. Add the complete kernel as a lit acceptance test under
   `bishengir/test/Conversion/HIVMAVEToPTOASVMI/`.
2. Check the presence of pointer conversion, MTE, flags/waits, control updates,
   barrier, helper call, loop, and VMI load/add/store.
3. Check the absence of `ave.hir`, `hivm.hir`, HIVM address spaces,
   source-only attributes, and `unrealized_conversion_cast`.
4. Run MLIR verification on the converted module.
5. Run the converted artifact manually through PTOAS `--emit-vpto` and record
   the exact command. Add an automated cross-repository test only when the
   PTOAS executable can be discovered reliably.
6. Run the focused bridge lit suite and relevant Ascend conversion tests.

Exit criterion: the exact Stage 1 commands in Section 1 succeed.

Phase 1G result as of 2026-08-17:

- Added the exact-artifact lit regression:
  `bishengir/test/Conversion/HIVMAVEToPTOASVMI/lowered-vector-add-kernel.mlir`.
- The test runs `lowered_vector_add_kernel.mlir` through
  `--convert-hivmave-to-ptoas-vmi`.
- It checks the PTOAS module/kernel contract, helper pointer ABI, entry pointer
  ABI, MTE ops, sync/control ops, helper call, loop, and VMI load/add/store.
- It checks absence of `ave.hir`, `hivm.hir`, HIVM address spaces,
  source-only HACC/HIVM/TT/DLTI attrs, and `unrealized_conversion_cast`.
- The focused `HIVMAVEToPTOASVMI` lit directory passes with 20 tests.
- PTOAS lowering was manually validated in Phase 1F. Automated cross-repo
  PTOAS execution remains deferred until the PTOAS binary can be discovered
  reliably from AscendNPU-IR lit.

## 6. Test Matrix

### Positive tests

- Exact `lowered_vector_add_kernel.mlir` end-to-end conversion.
- AVE `functionType` values `norm` and `pb32` on their expected operations.
- Contiguous `hivm.is_continuous` store.
- GM and UB pointer conversion.
- Zero and nonzero constant UB byte addresses.
- Dynamic loop element offset lowered with `pto.addptr`.
- Converted helper call and `scf.for`.
- MTE2-to-V and V-to-MTE3 static flag/wait pairs.
- Control bits 60 clear/set and bit 48 set.
- `PIPE_ALL` barrier.

### Negative tests

- Mismatched `functionType` or continuity marker on a noncontiguous view.
- Unknown AVE/HIVM operation attribute.
- GM/UB memref element type unsupported by the chosen PTO pointer path.
- Rank greater than one, dynamic UB shape, or non-unit stride.
- Multi-address or malformed HIVM pointer cast.
- Unsupported pipe, dynamic event, or event ID outside the Stage 1 subset.
- Control index outside `[0, 63]`.
- Call whose converted operands do not match the callee signature.
- Residual AVE/HIVM operation or type.
- Ambiguous kernel entry or unknown target-architecture mapping.

### PTOAS checks

The PTOAS handoff is accepted only when:

- the module parses with PTOAS without Ascend dialect registration;
- PTOAS producer-boundary verification succeeds;
- VMI mask/layout assignment succeeds;
- VMI-to-VPTO lowering succeeds;
- emitted VPTO has no `pto.vmi.*`, `!pto.vmi.*`, or
  `unrealized_conversion_cast` residue.

## 7. Assumptions

- The in-tree PTO dialect remains generated and linked with AscendNPU-IR's own
  LLVM/MLIR build; no external PTOAS MLIR library is linked.
- The copied PTO dialect matches the PTOAS CLI used for Stage 1 validation.
- The acceptance artifact is the source of truth for Stage 1 operation and ABI
  coverage.
- The `scf.for` induction variable and AVE load/store indices in the artifact
  are element offsets.
- The vector helper is allowed to remain as a callable helper if PTOAS accepts
  the hand-authored target. Otherwise Stage 1 must add inlining before claiming
  success.
- Existing PTO MTE operations in the artifact have already encoded the desired
  transfer lengths and strides; this pass converts their buffer operands but
  does not reinterpret those numeric operands.

## 8. Open Questions

These must be answered during Phase 1A before the corresponding implementation:

1. Which exact PTO target architecture corresponds to `dav-c310` and
   `Ascend910_9589` in the artifact? Phase 1A proved that `a5` is accepted by
   PTOAS for this target shape, but the Ascend-target-to-PTO-arch mapping still
   needs an explicit source-of-truth confirmation before the pass hard-codes it.
2. Does PTOAS accept the eight-argument kernel ABI after converting all five
   dynamic GM memrefs to PTO pointers, or must sync/workspace and dynamic-shape
   arguments be transformed differently? Phase 1A answer: PTOAS accepts this
   shape when the two special `i8` GM arguments are represented as
   `!pto.ptr<ui8, gm>` and remain unused.
3. Should the output use a `pto.section.vector` source function or a normalized
   `pto.kernel_kind<vector>` module for the current PTOAS CLI? Phase 1A answer:
   the normalized module attribute form is accepted.
4. Does PTOAS accept a normal helper `func.call` from the vector kernel, or must
   the outlined helper be inlined? Phase 1A answer: PTOAS accepts the private
   `no_inline` helper call and lowers the VMI body in that helper.
5. Are the `i64` values passed to HIVM `pointer_cast` byte addresses while
   `pto.addptr` uses element offsets? The artifact uses `0` and `1024`; the
   chosen conversion must preserve those units explicitly. Phase 1A target
   keeps `pointer_cast` values as byte-address `pto.castptr` inputs and uses
   `pto.addptr` only for element offsets derived from memref views.
6. Which source module/function/argument attributes must survive for host ABI
   generation, if any, and what are their PTO equivalents?

Answers belong in the bridge design document and in tests, not only in code.

## 9. Risks

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Pointer offset units are confused | Wrong UB/GM addresses | Prove byte versus element units in Phase 1A and test nonzero offsets |
| Dynamic memref ABI is collapsed incorrectly | PTOAS entry ABI mismatch | Validate a hand-authored pointer ABI before changing the type converter |
| Source attributes are dropped too broadly | Lost layout or ABI semantics | Per-op/per-scope allowlists and negative tests |
| `set_ctrl` is lowered as a constant write | Unrelated control bits are corrupted | Required `get_ctrl` read-modify-write sequence |
| Existing PTO MTE ops become legal with stale HIVM operands | Mixed invalid IR reaches PTOAS | Dynamic legality based on all operand/result types |
| Helper calls are unsupported by PTOAS VMI lowering | End-to-end failure late in the pipeline | Test the target call shape in Phase 1A; inline only if required |
| PTO dialect copy drifts from PTOAS | Parser/verifier incompatibility | Record PTOAS revision and run the exact end-to-end artifact |
| Full conversion reports only the first blocker | Slow diagnosis | Keep focused unit tests per new operation/type family |

## 10. Post-Stage-1 Work

Broader AVE operation coverage, arbitrary DMA/synchronization forms, mixed
Cube/Vector modules, general dynamic memref descriptors, additional data types,
and compiler-driver pipeline integration are deferred until the exact vector-add
kernel passes Stage 1. Subsequent stages should be driven by the next concrete
lowered kernel artifact, not by adding operations speculatively.
