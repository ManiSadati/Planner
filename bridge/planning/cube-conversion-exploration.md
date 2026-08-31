# Cube Conversion Exploration Plan

Last updated: 2026-08-28

Status: both first 64x64 routes pass numerical simulator execution. The strict
direct conversion emits PTO operations. The external-call experiment preserves
the CCE calls and memref descriptor ABI through PTOAS, links the matching
NPU-IR template bitcode, and executes successfully.

## Goal

Determine how AscendNPU-IR Cube compute and its required DMA/staging sequence
should enter PTOAS. The first fixture is:

```text
bridge/triton-example/cube_dotproduct.py
```

The current deliverable is a narrow, source-backed implementation plus a clear
record of what it does and does not support.

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
right boundary for a direct PTO rewrite. The call layer is the boundary for the
separate compatibility route because its memrefs retain the runtime shape,
stride, offset, and address-space information consumed by the CCE functions.

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

### Path B: Rewrite Directly To PTO

This is the already validated narrow alternative. Intercept the structured
`nd2nz -> mmadL1 -> fixpipe` region before `convert-hivm-to-std` and emit PTO
MTE/MAD operations. The strict 64x64 fixture passes PTOAS simulation.

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

Path B remains passing evidence and a long-term option. It will not be removed
or changed by the external-call experiment.

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

## Direct-Path Decision

The first fixture does not justify adding a new PTOAS Cube instruction or
continuing inside the AVE-to-VMI pass. PTO already contains all required
primitive operations.

The recommended first implementation route is a **low-level PTO composition**
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

## Implementation Result

The approved starter route is implemented in AscendNPU-IR under:

```text
bishengir/lib/Conversion/HIVMTemplatesToPTO/Cube/
  CubePatterns.h
  MatmulRegionToPTO.cpp
```

The matcher is atomic and deliberately strict. It accepts the observed 64x64
f16/f16/f32/f16, init=true, no-transpose, NZ2ND region with the expected
ping-pong addresses and sync operands. Unsupported shapes, layouts, modes, or
partial regions remain on the normal CCE path.

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

- focused lit coverage passes for existing DMA behavior, strict Cube PTO
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

The numerical run also established an important physical-layout rule: a
row-major source B requires `pto.mte_l1_l0b {transpose = true}` to construct
PTO's right-tile `nZ` layout. With `false`, the output matched `A @ B.T` rather
than the Triton kernel's `A @ B`. This physical flag is separate from the
source-level `b_transpose` semantic.

## Active External CCE Template Experiment

The direct PTO composition remains a passing comparison path. A separate
`bridge/testcases/matmul_64/` fixture tests the active compatibility route:
allow `convert-hivm-to-std` to emit CCE template calls, then use
`convert-hivmave-to-ptoas-vmi` only to normalize the surrounding IR.

The runner exposes this as `--bridge-mode external-calls`. For the copied
64x64 input, the existing prototype preserves the three external declarations
and four calls and marks the output as a Cube module. It currently converts
their memrefs to raw PTO pointers, which is the behavior this experiment must
replace.

The simulator link then fails on exactly these symbols:

```text
_mlir_ciface_nd2nz_half
_mlir_ciface_mma_tile_half_to_float
_mlir_ciface_fixpipe_nz2nd_float_to_half_4d_to_2d_gm
```

This is not only a missing-library problem. The functions in
`meta_op.aic.c310.bc` take pointers to NPU-IR `memref_t` descriptors. The
current PTOAS lowering calls them with raw address-space pointers. The next
experiment must therefore preserve those memrefs rather than inventing
shape-specific adapters:

1. Enable memref preservation only under `--bridge-mode external-calls`.
2. For AIC/Cube functions, map HIVM memory-space attributes but keep memref
   shape, layout, offset, sizes, and strides through VMI and VPTO.
3. For AIV/vector functions, retain the existing `!pto.ptr` conversion at VMI
   uses. In a split MIX module, choose this policy by function core type.
4. Confirm that final memref-to-LLVM lowering creates descriptor allocas and
   `_mlir_ciface_*` calls equivalent to the unchanged NPU-IR output.
5. Integrate the matching CCE template device code into the PTOAS link.
6. Run `matmul_64`, then a true MIX fixture, and compare with both the direct
   PTO result and unchanged NPU-IR baseline.

Do not claim success from VMI/VPTO legality or symbol resolution alone. The
pre-CCE LLVM descriptor ABI and numerical simulator result are both required.

## Proposed Source Layout

Keep the new Cube lowering separate from both the CCE template tree and the
existing vector conversion. Add it as a `Cube/` component of the existing
structured-HIVM conversion:

```text
bishengir/lib/Conversion/HIVMTemplatesToPTO/
  CMakeLists.txt
  HIVMTemplatesToPTO.cpp           # pass entry and pattern registration
  Cube/
    CubePatterns.h
    MatmulRegionToPTO.cpp          # guarded ND2NZ -> MMAD -> fixpipe recipe
    MmadL1ToPTO.cpp                # factor out after semantics are proven
  DMA/
    ND2NZToPTO.cpp                 # factor out guarded reusable DMA later
    FixpipeToPTO.cpp
```

For the first implementation, keep the complete guarded fixture recipe in
`Cube/MatmulRegionToPTO.cpp`. Splitting the three operations too early could
partially rewrite a region and leave an invalid mix of PTO and CCE lowering.
Once each operation has a proven independent contract, reusable ND2NZ and
fixpipe patterns can move into `DMA/`.

The main file should retain pass construction and call a helper such as
`populateHIVMCubeTemplatesToPTOPatterns(patterns)`. Use one pass and the
existing pipeline switch initially, because all these patterns share the same
interception point and memory/sync ownership. A separate
`HIVMCubeTemplatesToPTO` pass is only useful later if Cube needs independent
ordering or enablement.

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

## Conversion Boundaries

The direct PTO rewrite uses the existing optional bridge location:

```text
hivm-mark-disable-load
  -> sync-block finalization
  -> convert-hivm-templates-to-pto     # Cube conversion belongs here
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

## Validated Direct-Path Contract

- low-level PTO composition rather than a new CCE template;
- strict 64x64 f16/f16/f32/f16 specialization;
- NPU-IR ownership of memory placement and explicit sync;
- unchanged CCE path as the default/fallback;
- local IR/PTOAS verification and simulator numerical comparison first,
  followed by direct CCE comparison and A5 validation.

## Active Implementation Stages

1. Preserve the external call memrefs for pure AIC `matmul_64`.
2. Verify VMI, VPTO, and pre-CCE LLVM descriptor structure.
3. Integrate the matching CCE template device definitions and run simulation.
4. Add a true MIX test and verify separate AIC/AIV type policies and packaging.
5. Compare the external-call result with the direct PTO and unchanged NPU-IR
   paths.
6. Request A5 hardware validation from the human after local equivalence.

## Related Documents

- `bridge/memory/cube-conversion-status.md`
- `bridge/planning/npuir-to-ptoas-mapping.md`
- `bridge/planning/dma-template-rewrite-plan.md`
- `bridge/memory/dma-template-mapping.md`
- `bridge/designs/ave-to-ptoas-vmi-conversion-design.md`
