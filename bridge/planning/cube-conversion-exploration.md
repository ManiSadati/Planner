# Cube Conversion Exploration Plan

Last updated: 2026-09-03

Status: both active 64x64 routes pass numerical simulator execution. The
external route preserves and links CCE calls, while the preferred PTODSL route
imports pre-generated `MmadL1` and ND2NZ PTO helpers into the kernel module.
The old direct C++ Cube body has been removed. The 513x513 PTODSL fat object
also passes on A5. F16 B-transpose, BF16 NN, signed INT8 NN, and F32/HF32 NN
now pass the local bridge simulator as explicit PTODSL specializations.

## Goal

Determine how AscendNPU-IR Cube compute and its required DMA/staging sequence
should enter PTOAS. The first fixture is:

```text
bridge/triton-example/cube_dotproduct.py
```

The current deliverable is a narrow, source-backed implementation plus a clear
record of what it does and does not support.

## Active Specialization Work

The next implementation step is no longer hypothetical. Four `64x64x64`
fixtures expose four distinct CCE contracts now implemented as PTO-visible
PTODSL instantiations:

| Contract | PTODSL MMAD helper | PTODSL ND2NZ helper | Fixpipe mapping |
| --- | --- | --- | --- |
| F16/F32 with B transposed | `mmadl1_f16_f32_tb` | reuse `nd2nz_f16_gm_l1` | reuse f32-to-f16; simulator passes |
| BF16/F32 NN | `mmadl1_bf16_f32_nn` | `nd2nz_bf16_gm_l1` | f32-to-bf16; exact BF16 simulator match |
| INT8/INT32 NN | `mmadl1_i8_i32_nn` | `nd2nz_i8_gm_l1` | i32-to-i32; exact INT32 simulator match |
| F32/F32 HF32 NN | `mmadl1_f32_f32_hf32_nn` | `nd2nz_f32_gm_l1` | f32-to-f32; zero-error simulator match |

Each helper is authored in the shared PTODSL Python source and checked in as a
separate pre-generated MLIR resource. C++ selects and imports the matching
resource; it must not duplicate the MMAD body. ND2NZ and Fixpipe remain guarded
direct PTO conversions around that body.

The HF32 early IR currently requires one known correction:
`input_precison = "hf32"` must be `input_precision = "hf32"`. Evidence points
to the installed Ascend Triton adapter as the producer of the typo, while
NPU-IR consistently consumes the correct spelling. This is not a PTODSL
template failure, and the bridge must not silently reinterpret arbitrary
unknown attributes as HF32.

## 2026-09-01 Direction Update

The external-call experiment proved that PTOAS can carry the CCE memref ABI and
link the selected device templates. It remains a valuable compatibility and
numerical reference, but it is not the intended default because PTOAS sees the
Cube implementation only as opaque calls.

The preferred path should resemble PTOAS PTODSL TileLib:

1. intercept the structured region before `convert-hivm-to-std`;
2. preserve shape, layout, init/accumulate, dependency, and tail facts;
3. emit PTO tile operations for PTOAS's in-process Python TileLib, or
   materialize and inline an equivalent PTODSL-generated PTO body;
4. let PTOAS optimize the resulting PTO/VMI/VPTO operations;
5. use the external path and unchanged NPU-IR as references.

The Python source added by commit
`9d97eff1240434e537e45ee9154c65df80208e2e` is now a normalized f16 template
rather than a fixed whole-kernel implementation. It accepts M/K/N up to 64,
caller-owned local buffers, init/accumulate, and NPU-IR event IDs. Python is
used only to generate checked-in MLIR resources. In `ptodsl` mode,
`bishengir-compile` selects a helper by `pto.ptodsl.logical_name`, loads that
resource without running Python, and calls it in place of structured
`hivm.hir.mmadL1`.

## Starting State

- Row softmax and RMSNorm are supported by the current AVE-to-VMI bridge for
  the accepted fixtures, with performance accepted as on par with NPU-IR.
- The vector path therefore moves to maintenance mode. New vector instructions
  can be added when a concrete Cube or future kernel exposes a missing row.
- Simple GM->UB and UB->GM DMA conversion already provides a useful bridge
  pattern, but Cube introduces different memory roles, layouts, pipelines, and
  accumulator behavior.
- The first Cube trace is now complete. The AVE-to-VMI failure is expected and
  is not evidence that PTO lacks Cube support: that vector pass first rejects a
  rank-4 `hivm.hir.pointer_cast`, while the useful Cube path remains in
  structured HIVM operations.

## Frozen Evidence

The decision below is based on:

- fixture: `bridge/testcases/cube_dotproduct/cube_dotproduct.py`, SHA-256
  `92a5552e598962b8f080160b450f6397934602c301747fd560eebca62cae034f`;
- early IR: `bridge/testcases/cube_dotproduct/input.mlir`, SHA-256
  `a3af8a30569d9847be6c5221e8f43d709f11020faae8b8705dac24f3441a907b`;
- NPU-IR revision: `057c6bb32c17bdab84d0fb9c113d1d4a8419f3c3`;
- PTOAS upstream revision: `80ac6d0488a4681fe0a2cd362c463395b03dba94`;
- target: `Ascend910_9589` / `dav-c310`;
- complete pass trace:
  `bridge/testcases/cube_dotproduct/out/after-all.log`.

The generated testcase output is evidence, not a tracked source artifact.

## Observed Structured Region

The last useful structured form is after `hivm-mark-disable-load` and the
sync-block finalization passes, immediately before `convert-hivm-to-std`:

1. Two `hivm.hir.nd2nz` operations move 64x64 f16 matrices from GM ND layout
   to 4x4x16x16 f16 CBUF/L1 buffers.
2. One `hivm.hir.mmadL1` consumes the two L1 buffers, initializes an f32 CC/L0C
   accumulator, and computes `m=k=n=64`.
3. One `hivm.hir.fixpipe` performs NZ-to-ND result movement from f32 CC/L0C to
   f16 GM with `F322F16` conversion.
4. NPU-IR has already assigned ping-pong CBUF/CC addresses and emitted the
   MTE2/MTE1, MTE1/M, M/FIX, and FIX/M event ordering.

`convert-hivm-to-std` replaces those four operations with calls to
`nd2nz_half`, `mma_tile_half_to_float`, and
`fixpipe_nz2nd_float_to_half_4d_to_2d_gm`. The structured layer remains the
right boundary for PTODSL/PTO-native expansion. The call layer is the boundary
for the separate compatibility route because its memrefs retain the runtime
shape, stride, offset, and address-space information consumed by the CCE
functions.

## What The CCE Templates Actually Do

For the current A5/RegBase build, the authoritative device templates are under
`$HOME/AscendNPU-IR/bishengir/lib/Template/lib/RegBase/Cube/`. The build list is
in that directory's `CMakeLists.txt`; the older parallel files under
`Template/lib/Cube`, `Template/lib/DMA`, and `Template/lib/Fixpipe` are not the
selected A5 implementations for this fixture.

- MMAD: `RegBase/Cube/LocalMmad.cpp`, currently a CATLASS-based implementation.
- ND2NZ wrapper/body: `RegBase/Cube/nd2nz.cpp` and
  `RegBase/Cube/compat/DMA/Cbuf/nd2nz.cpp`.
- Fixpipe wrapper/body: `RegBase/Cube/Fixpipe.cpp` and
  `RegBase/Cube/compat/Fixpipe/Fixpipe.cpp`.
- Shared Cube template headers: `RegBase/Cube/include/` and
  `bishengir/lib/Template/include/`.

- `nd2nz_half` calculates the ND and NZ dimensions/strides and issues
  `copy_gm_to_cbuf_multi_nd2nz_b16`. It has a row-loop fallback when the GM
  leading stride exceeds the 16-bit intrinsic range.
- `mma_tile_half_to_float` is more than one MAD. It chooses an L0-sized K
  partition, optionally double-buffers L0A/L0B, moves L1 to L0A and L0B,
  coordinates MTE1 and M events, emits `mad`, handles init versus accumulation,
  and inserts the required barriers.
- `fixpipe_nz2nd_float_to_half_4d_to_2d_gm` configures ND parameters and emits
  `copy_matrix_cc_to_gm` with NZ2ND enabled and f32-to-f16 pre-conversion.

This is why function-name replacement and bare `mmadL1 -> pto.tmatmul` are not
correct general solutions.

## Core Decision

For every NPU-IR Cube or related DMA template used by the fixture, compare the
actual CCE template implementation against PTOAS/PTO semantics.

Classify each row as one of:

| Classification | Meaning | Intended action |
| --- | --- | --- |
| `direct` | One PTO dialect operation preserves the complete source contract. | Emit that PTO operation from a guarded NPU-IR conversion. |
| `PTO composition` | No single operation matches, but a small PTO sequence preserves the contract. | Emit and test the explicit sequence. |
| `template rewrite` | The CCE template hides behavior that must be rebuilt from structured NPU-IR information. | Intercept before the CCE call and rewrite that template lowering in PTO dialect. |
| `unsupported/unknown` | Equivalence cannot yet be shown. | Keep the CCE path and collect more evidence. |

Operation names are not enough to claim `direct`. The comparison must include
dtype, tile role, layout, valid `m/n/k`, transpose, accumulator initialization,
precision modes, padding, address ownership, pipeline assignment, and sync.

## Investigation Stages

1. **Freeze the fixture and revisions**
   Record the exact `cube_dotproduct.py`, active Wilson NPU-IR revision, PTOAS
   upstream/fork revision, target SOC, and compiler options used later.
2. **Trace the real NPU-IR pipeline**
   Starting from the fixture's early IR, record major forms of Cube compute,
   operand DMA/staging, accumulator/result movement, and synchronization before
   they become CCE calls.
3. **Locate selected CCE templates**
   Identify the template declaration, selection logic, call site, and actual
   implementation for every row reached by the fixture.
4. **Inventory PTO targets**
   Check PTOAS/PTO Cube, tile movement, MTE, fixpipe/store, and sync operations
   against the exact source contracts.
5. **Complete the mapping table**
   Assign one of the four classifications, a preferred interception point,
   risks, and source references to each row.
6. **Choose the first implementation slice**
   Select the smallest complete path only after human review. Preserve the CCE
   path behind the default-off PTO conversion switch.
7. **Validate later**
   Use local IR/PTOAS checks and simulator comparison first; use the human-run
   A5 server for authoritative runtime and performance validation.

## Fixture Mapping Table

| Stage | NPU-IR source | PTO mapping | Fixture decision | General-family decision | Main risk |
| --- | --- | --- | --- | --- | --- |
| GM ND -> L1 NZ | `hivm.hir.nd2nz` | Tile route: `pto.tload` into MAT. Low-level route: `pto.mte_gm_l1_frac` in `nd2nz` mode, which expands to `pto.copy_gm_to_cbuf_multi_nd2nz`. | `direct` at low level | `direct` only for guarded dtype/layout/stride forms; retain the large-stride row-loop case | Reproducing ND/NZ shape and 16-bit stride limits |
| L1 -> L0A | internal to `mma_tile_half_to_float` | Tile route: `pto.textract` MAT -> LEFT. Low-level route: `pto.mte_l1_l0a`. | `PTO composition` | part of a template rewrite/composition | Exact A layout, K offsets, transpose, and ping-pong address |
| L1 -> L0B | internal to `mma_tile_half_to_float` | Tile route: `pto.textract` MAT -> RIGHT. Low-level route: `pto.mte_l1_l0b`. | `PTO composition` | part of a template rewrite/composition | Exact B layout, K offsets, transpose, and ping-pong address |
| Cube compute | `hivm.hir.mmadL1`, init=true, 64x64x64 f16/f32 | Tile route: `pto.tmatmul`. Low-level route: `pto.mad`; use accumulating variants when init=false. | `PTO composition`, not a bare one-op replacement | `template rewrite` unless a future PTO op owns L1 staging and K scheduling | K partitioning, init/accumulate, bias, transpose, HF32/I4/MX, unit flags |
| L0C -> GM | `hivm.hir.fixpipe`, NZ2ND, F322F16 | Tile route: `pto.tstore` from ACC. Low-level route: `pto.mte_l0c_gm` or `pto.copy_matrix_cc_to_gm`. | `direct` for this exact mode after verifier proof | guarded direct mappings per destination/quant/relu/layout mode | Packed fixpipe mode, dtype conversion, stride, relu/quant/dual destination |
| Synchronization | NPU-IR set/wait flags and barriers | First route: preserve and translate explicit events around low-level PTO ops. Long-term tile route: let PTOAS insert sync after removing NPU-IR-owned duplicates. | preserve NPU-IR ownership | explicit per route | Double synchronization or an event pair attached to the wrong physical pipe |

## Two Implementation Paths

### Path A: Preserve External CCE Calls

This is the active experiment. Skip the early Cube rewrite, retain the
`nd2nz_half`, `mma_tile_half_to_float`, and fixpipe calls, and preserve their
ranked memref operands through PTOAS. Standard MLIR memref-to-LLVM conversion
should then construct the same descriptor representation used by the normal
NPU-IR path. The resulting PTOAS device object must be linked with the matching
CCE template implementation.

Why explore it first:

- it reuses the complete selected CCE template behavior, including runtime
  sizes and strides, K partitioning, layout handling, fallbacks, precision
  modes, double buffering, and synchronization;
- it can establish broad Cube compatibility without first translating every
  template branch into PTO;
- it provides a close comparison against unchanged NPU-IR and isolates the
  PTOAS carrier/linking problem from template reimplementation.

Risks:

- PTOAS VMI/VPTO and final emission must legally carry standard memrefs without
  collapsing them into raw `!pto.ptr` values;
- HIVM memory spaces must map to LLVM address spaces with a descriptor ABI that
  exactly matches the CCE bitcode;
- the CCE template object and PTOAS device object must use compatible CANN,
  target, data-layout, symbol, and link conventions;
- split MIX kernels require per-function behavior: AIC calls keep memrefs while
  AIV VMI operations use PTO pointers;
- it retains a dependency on CCE template code, so it is a compatibility or
  transitional route rather than a fully open replacement backend.

### Historical Direct C++ Rewrite

This was a validated narrow proof. It intercepted the structured
`nd2nz -> mmadL1 -> fixpipe` region before `convert-hivm-to-std` and emitted PTO
MTE/MAD operations. The strict 64x64 fixture passed PTOAS simulation.

Benefits:

- removes the runtime dependency on CCE templates;
- exposes Cube movement, compute, and sync to PTOAS analysis and optimization;
- can eventually converge on the tile-level PTO abstraction.

Risks:

- every supported CCE template branch must be understood and reproduced;
- layout, stride limits, K partitioning, accumulation, transpose, precision,
  quantization, double buffering, and event ordering create a large semantic
  surface;
- support can become a collection of narrow special cases, and a legal-looking
  PTO sequence can still compute the wrong physical layout.

The result remains useful evidence that the primitive PTO mapping can compute
the fixture correctly. Its hand-written C++ Cube body has now been removed so
the PTODSL source is the only PTO-native implementation. Reintroducing a second
C++ implementation requires an explicit human decision.

## External-Call Implementation Checkpoint

Verified on 2026-08-28 with `bridge/testcases/matmul_64/`:

- `external-calls` mode skips the structured Cube rewrite and enables memref
  preservation in `convert-hivmave-to-ptoas-vmi`;
- the VMI kernel entry uses the normal five GM pointers plus three scalars,
  while `nd2nz_half`, `mma_tile_half_to_float`, and fixpipe retain ranked
  memref declarations and call operands;
- VPTO preserves those call-site memrefs until LLVM ABI preparation;
- both PTOAS LLVM emitters, including the CANN 9.x path selected by the local
  simulator, lower the retained memrefs with standard descriptor structs;
- the pre-CCE LLVM wrapper signatures and `_mlir_ciface_*` calls match the
  unchanged NPU-IR baseline;
- external mode automatically supplies
  `$ASCEND_NPU_IR_ROOT/build/install/lib/meta_op.aic.c310.bc` to PTOAS through
  `PTOAS_AICORE_LL_MODULE`; PTOAS passes it to Cube Bisheng compilation with
  `-cce-link-aicore-ll-module`;
- the resulting fat object links and runs in the A5 operator simulator;
- all 4096 f16 outputs pass comparison with maximum absolute error
  `0.001953125`; the run reports 3647 total ticks.

This proves the compatibility architecture for the observed template set. It
does not yet prove general Cube coverage. The next evidence must include
additional shapes/template branches and a genuine split MIX fixture containing
both AIC and AIV functions. Real A5 hardware validation also remains required.

## Historical First Direct-Path Decision

This decision produced the first passing numerical comparison and remains
historical evidence. It is no longer an active implementation strategy.
The first fixture did not justify adding a new PTOAS Cube instruction or
continuing inside the AVE-to-VMI pass because PTO already contained all
required primitive operations.

The first implementation route was a **low-level PTO composition**
inside `convert-hivm-templates-to-pto`:

1. Match the complete guarded `nd2nz -> mmadL1 -> fixpipe` region, not isolated
   helper calls.
2. Preserve NPU-IR memory addresses, block scheduling, and explicit sync.
3. Emit `pto.mte_gm_l1_frac`, `pto.mte_l1_l0a`, `pto.mte_l1_l0b`, `pto.mad`,
   and `pto.mte_l0c_gm` for the observed ND2NZ, staging, compute, and fixpipe
   behavior.
4. Initially accept only the observed 64x64 f16/f16/f32/f16, no-transpose,
   init=true, NZ2ND case. Reject everything else with a reason and leave the
   normal CCE path available.

This route matches the current bridge architecture, which already emits
low-level PTO MTE operations from HIVM memrefs and preserves NPU-IR sync. It
also avoids introducing PTO tile-buffer/view types while NPU-IR still owns
physical addresses.

This is a conversion recipe that emits PTO MLIR directly, not another CCE
device template and not a new intermediate operation that needs a second
expansion pass.

## Historical Implementation Result

The starter route was implemented in AscendNPU-IR under:

```text
bishengir/lib/Conversion/HIVMTemplatesToPTO/Cube/
  CubePatterns.h
  MatmulRegionToPTO.cpp
```

The matcher was atomic and deliberately strict. It accepted the observed 64x64
f16/f16/f32/f16, init=true, no-transpose, NZ2ND region with the expected
ping-pong addresses and sync operands. Unsupported shapes, layouts, modes, or
  partial regions remained on the normal CCE path.

For the real `cube_dotproduct` fixture it emits:

```text
2 x pto.mte_gm_l1_frac
  -> pto.mte_l1_l0a + pto.mte_l1_l0b
  -> pto.mad
  -> pto.mte_l0c_gm
```

NPU-IR continues to own physical addresses and explicit MTE1/M/FIX event
ordering. The downstream AVE-to-VMI conversion was extended only with the
memory-space, pipe, cast, and PTO-op legality needed to carry this Cube region;
it does not become the Cube pattern owner.

Verification completed on 2026-08-27:

- focused lit coverage passed for existing DMA behavior, strict Cube PTO
  emission, and the Cube-to-VMI continuation;
- the real bridge runner writes
  `bridge/testcases/cube_dotproduct/out/cube_dotproduct.vmi.mlir`;
- unmodified PTOAS lowers that file to
  `bridge/testcases/cube_dotproduct/out/cube_dotproduct.vpto.mlir`;
- VPTO contains `pto.load_cbuf_to_ca`, `pto.load_cbuf_to_cb`, and
  `pto.mad_raw`, with Cube kernel metadata preserved.
- the simulator fixture uses the same row-major 64x64 f16 A/B and f16 output
  contract as the Triton kernel, with an f32-accumulated reference;
- the clean simulator run passes all 4096 outputs with maximum absolute error
  `0.001953125` and reports 3250 total ticks.

That implementation has now been removed. Its result still established an
important physical-layout rule: a
row-major source B requires `pto.mte_l1_l0b {transpose = true}` to construct
PTO's right-tile `nZ` layout. With `false`, the output matched `A @ B.T` rather
than the Triton kernel's `A @ B`. This physical flag is separate from the
source-level `b_transpose` semantic.

## External CCE Compatibility Route

The direct PTO composition is no longer present. A separate
`bridge/testcases/matmul_64/` fixture tests the active compatibility route:
allow `convert-hivm-to-std` to emit CCE template calls, then use
`convert-hivmave-to-ptoas-vmi` only to normalize the surrounding IR.

The runner exposes this as `--bridge-mode external-calls`. It preserves the
three external declarations and four calls, retains ranked memrefs until
standard memref-to-LLVM lowering builds the C-interface descriptors, and links
`meta_op.aic.c310.bc`. The 64x64 simulator result passes all 4096 outputs with
maximum absolute error `0.001953125`. This remains a compatibility/reference
backend because PTOAS cannot optimize inside the external CCE calls.

## Current Source Layout

The bridge integration is separate from both the CCE template implementation
and the existing vector conversion:

```text
bishengir/lib/Conversion/HIVMTemplatesToPTO/
  CMakeLists.txt
  HIVMTemplatesToPTO.cpp           # pass entry and pattern registration
  Cube/
    CubePatterns.h
    MatmulRegionToPTO.cpp          # validation, call adaptation, ND2NZ/Fixpipe
    PTODSLTemplateImporter.cpp     # imports pre-generated PTO helper MLIR
    PTODSLTemplateImporter.h
```

`MatmulRegionToPTO.cpp` must not contain another implementation of the
`MmadL1` staging/MAD body. That implementation is owned by
`nd2nz_mmadl1_64_ptodsl.py`; C++ only imports it, validates/adapts the call, and
converts the separate caller-side operations.

The longer-term route is a **tile-level PTO composition**:

```text
pto.tload(MAT A/B)
  -> pto.textract(LEFT/RIGHT)
  -> pto.tmatmul or pto.tmatmul.acc(ACC)
  -> pto.tstore(GM)
```

PTOAS already demonstrates the required K-loop, L0A/L0B ping-pong, extraction,
sync, and matmul structure in `ptodsl/examples/fa_dn_matmul.py`. Adopting that
route means PTOAS must own tile allocation and synchronization for the region;
it should not be mixed casually with NPU-IR's existing address and event plan.

## Preferred PTODSL Integration Plan

The normalized template boundary is one `MmadL1` microtile with actual M/K/N
values no larger than 64. Whole matmuls such as `q_kt_matmul` remain outer
loops over those calls. The source contract is:

```text
ND2NZ(A/B, actual rows/cols and padded physical extent)
  -> wait caller dependencies
  -> L1-to-L0A/B staging
  -> pto.mad when init, otherwise pto.mad_acc
  -> release caller dependencies
```

The probe's FixPipe writeback is deliberately outside `mmadl1_f16_f32`, just
as the structured NPU-IR operation records that result FixPipe is inserted by
its caller.

Implemented compiler handoff:

1. PTODSL is run explicitly during development to generate the checked-in
   `mmadl1_f16_f32_nn.mlir` and `nd2nz_f16_gm_l1.mlir` instantiations.
2. CMake copies and installs those files under `lib/bishengir/ptodsl/cube/`.
3. The bridge parses both files in the active MLIR context, validates their
   stable symbols and instantiation attributes, and imports them as
   `@__pto_mmadl1_f16_f32_nn` and `@__pto_nd2nz_f16_gm_l1`.
4. C++ adapts NPU-IR pointers, dimensions, strides, init state, and events to
   normal `func.call`s. PTOAS then inlines and lowers both visible bodies.

Normal compilation has no Python subprocess, daemon, or socket dependency.
Python remains the implementation source used for intentional regeneration;
C++ does not duplicate the helpers' PTO instruction sequences.

### 64x64 Coverage Boundary

| Category | Behavior |
| --- | --- |
| Required and working | f16/f32 NN and B-transpose, bf16/f32 NN, signed i8/i32 NN, and f32/f32 HF32 NN pointers and helpers; observed ping-pong addresses; 64x64x64 extents; L1-to-L0A/B staging; physical L0B nZ packing; explicit dependencies; initialization; typed Fixpipe and ND2NZ mappings |
| Present but not numerically proven | `pto.mad_acc` for `init = false`; runtime M/K/N arguments below 64 |
| Not needed by the initial fixture | internal K segmentation; edge padding; A transpose; bias; I4/MX; unit flags; quant/ReLU/remaining Fixpipe modes; ND2NZ large-stride fallback; MIX fallback policy |
| Unmapped or unresolved generally | complete CATLASS scheduling branches; general local-memory capacity and address ownership; PTOAS auto-allocation/auto-sync transition; remaining precision/bias/transpose variants; arbitrary edge-tile zero fill |

No new PTO primitive was required for the observed 64x64 path. Deferred rows
must not be inferred to work merely because their operation names resemble the
implemented sequence.

The compiler integration must choose one owner for local allocation and sync:

- NPU-IR-owned mode passes explicit local addresses and event dependencies to
  a low-level PTODSL body and disables PTOAS auto-allocation/auto-sync there.
- PTOAS-owned mode emits logical tile dependencies and removes the corresponding
  NPU-IR physical event/address plan for that region before TileLib expansion.

The second mode may eventually provide more optimization freedom. The current
first mode is nevertheless a real PTODSL-template path: Python owns the helper
body, while C++ only adapts and imports it. It must not drift back into a
hand-written C++ copy of the CCE template.

Generalization order:

1. keep the passing 64x64 simulator and 513x513 A5 paths green;
2. force a true sub-64 runtime M/K/N helper call, since `matmul_513` pads each
   MMAD microtile to 64 even though its global boundaries are odd;
3. keep the passing BF16 NN, INT8-to-INT32 NN, F32/HF32 NN, and f16
   B-transpose specializations green;
4. add ND2NZ stride fallback, A transpose, bias, I4/MX, unit flags, and remaining FixPipe
   modes from concrete fixtures;
5. validate split MIX ownership and external-CCE fallback for unsupported rows.

## Conversion Boundaries

The PTODSL route uses the existing optional bridge location:

```text
hivm-mark-disable-load
  -> sync-block finalization
  -> convert-hivm-templates-to-pto     # import/adapt PTODSL Cube helper here
  -> convert-hivm-to-std
```

This is late enough that NPU-IR has selected memory placement, layouts, block
scheduling, and sync, but early enough that `nd2nz`, `mmadL1`, and `fixpipe`
still carry structured semantics. The later AVE-to-VMI pass remains responsible
for vector-side operations only.

The external-call experiment deliberately skips that early pass:

```text
convert-hivm-to-std                    # emits selected CCE calls with memrefs
  -> expand-strided-metadata
  -> convert-hivmave-to-ptoas-vmi      # preserves AIC call memrefs
  -> PTOAS VMI/VPTO
  -> standard memref-to-LLVM lowering
  -> link matching CCE template device code
```

## Historical Direct-Path Contract

- this low-level C++ PTO composition has been removed;
- strict 64x64 f16/f16/f32/f16 specialization;
- NPU-IR ownership of memory placement and explicit sync;
- unchanged CCE path as the default/fallback;
- local IR/PTOAS verification and simulator numerical comparison first,
  followed by direct CCE comparison and A5 validation.

## Active Implementation Stages

1. Keep the passing 64x64 simulator and 513x513 A5 PTODSL paths green.
2. Treat PTODSL as the comparison runner's default bridge mode while retaining
   explicit `direct` and `external-calls` modes. This does not enable the bridge
   in ordinary NPU-IR compilation.
3. Prove a helper invocation with a runtime M/K/N value below 64; the 513 case
   proves odd global boundaries and accumulation but still uses padded MMAD
   microtiles.
4. Preserve the four configuration fixtures completed on 2026-09-03. The
   upstream HF32 `input_precison` spelling mismatch remains documented and the
   tracked early IR retains the corrected `input_precision` spelling.
5. Compare each accepted PTODSL configuration with external-call and unchanged
   NPU-IR results before adding its pre-generated helper to the default set.
6. Add a true MIX test and verify ownership, fallback policy, and packaging.

## Related Documents

- `bridge/memory/cube-conversion-status.md`
- `bridge/planning/npuir-to-ptoas-mapping.md`
- `bridge/planning/dma-template-rewrite-plan.md`
- `bridge/memory/dma-template-mapping.md`
- `bridge/designs/ave-to-ptoas-vmi-conversion-design.md`
