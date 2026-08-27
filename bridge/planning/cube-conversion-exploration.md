# Cube Conversion Exploration Plan

Last updated: 2026-08-27

Status: the strict conversion is implemented and verified through PTOAS
VMI-to-VPTO lowering and numerical simulator execution for the first 64x64
fixture. Direct CCE performance comparison and A5 validation remain pending.

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
`fixpipe_nz2nd_float_to_half_4d_to_2d_gm`. The call layer is useful as a
baseline, but it is too late to be the bridge boundary.

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

## Decision

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

## Conversion Boundary

Use the existing optional bridge location in the NPU-IR driver:

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

## Approved Starter Contract

- low-level PTO composition rather than a new CCE template;
- strict 64x64 f16/f16/f32/f16 specialization;
- NPU-IR ownership of memory placement and explicit sync;
- unchanged CCE path as the default/fallback;
- local IR/PTOAS verification and simulator numerical comparison first,
  followed by direct CCE comparison and A5 validation.

## Remaining Implementation Stages

1. Run the unchanged CCE simulator path with the same fixture and compare
   trace/ticks with the passing bridge path.
2. Request A5 hardware validation from the human.
3. Preserve the B physical-layout regression check while adding new modes.
4. Generalize one dimension at a time: K partitioning, accumulation, transpose,
   bias, precision modes, then additional fixpipe modes.

## Related Documents

- `bridge/memory/cube-conversion-status.md`
- `bridge/planning/npuir-to-ptoas-mapping.md`
- `bridge/planning/dma-template-rewrite-plan.md`
- `bridge/memory/dma-template-mapping.md`
- `bridge/designs/ave-to-ptoas-vmi-conversion-design.md`
