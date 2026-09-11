# Flash-Attention PTODSL Simulator Link Plan

Last updated: 2026-09-10

## Status

The current `flash_atten` testcase produces complete VMI and VPTO modules in
the default PTODSL bridge mode. Both `hivm.hir.mmadL1` operations and their
ND2NZ inputs use imported PTO helpers. Fixpipe and all three previously
unresolved vector-side DMA helpers now lower to native PTO operations.

The VMI contains two `pto.mte_ub_ub`, one padded dynamic
`pto.mte_gm_ub`, two `pto.mte_ub_l1`, and two `pto.mte_l0c_ub`
operations. It contains no calls or declarations for
`copy_ubuf_to_ubuf_1d_float`, `load_gm_to_ubuf_2d_half`, or
`copy_ubuf_to_cbuf_2d_half`.

PTOAS lowers the complete VMI to VPTO. Compiling the complete MIX container,
rather than separately compiling its AIV and AIC VPTO modules, now produces a
valid fat object with one public `flash_atten_kernel` symbol. The active logical
blocks then stall in simulation and do not produce `output.bin`; inactive
blocks exit normally. The current blocker is the converted Cube/Vector sync
contract, not packaging or the three DMA mappings.

NPU-IR's vendored PTO dialect says `pto.sync.set` / `pto.sync.wait` lower to
A5 intra-block synchronization. Current PTOAS changed those operations to FFTS
cross-core synchronization and added separate `pto.set_intra_block` /
`pto.wait_intra_block` operations. The generated pre-CCE LLVM confirms that the
bridge syncs currently become `llvm.hivm.SET.CROSS.CORE` and
`llvm.hivm.WAIT.FLAG.DEV.PIPE.*`.

A temporary direct substitution to the named intra-block operations changed
the stall but did not fix it: AIV0 and AIV1 occupy physical semaphore ranges
`0-15` and `16-31`. A blanket duplication using `id` and `id + 16` also failed,
because not every dependency involves both vector subblocks. The production
translation must retain which producer and consumer subblocks participate.

## Current Conversion Boundary

The generated VMI contains the expected Cube-side sequence:

```text
__pto_nd2nz_f16_gm_l1
__pto_mmadl1_f16_f32_nn
pto.mte_l0c_ub ..., dst_mode(0), nz2nd
```

This confirms that:

- each MMAD A/B/C pointer comes from its real memref through
  `pto.tile_buf_addr`, rather than a reconstructed fixed address;
- the first MMAD's empty `sync_related_args` become disabled (`-1`) optional
  helper events;
- the second MMAD preserves its explicit event values;
- caller-side block and pipeline synchronization remains around both calls;
- each accumulator-to-UB transfer preserves its real L0C and UB pointers,
  64x64 extent, row stride, NZ2ND mode, and single-destination contract.

The observed DMA contracts are deliberately narrow:

- f32 contiguous `memref<64>` UB-to-UB becomes `pto.mte_ub_ub` with one
  burst of eight 32-byte blocks;
- the observed f16 64x64 GM-to-UB implicit-transpose load imports a
  pre-generated PTODSL helper. It performs a contiguous `pto.mte_gm_ub`, then
  transposes the UB tile in place through pairwise
  `pto.vgather2`/`pto.vscatter` swaps. A strided `pto.mte_gm_ub`
  alone does not implement this transpose;
- f16 UB `4x1024 [1040,1]` to contiguous L1 becomes `pto.mte_ub_l1` with
  four bursts of 64 blocks and a one-block source gap.

L0A/L0B scratch allocation remains a separate policy risk.

## Historical Link Failure

The failing log is:

```text
bridge/testcases/flash_atten/out/build/bridge-sim.log
```

The earlier partial conversion reported five unresolved `_mlir_ciface_*`
symbols:

```text
copy_ubuf_to_ubuf_1d_float
load_gm_to_ubuf_2d_half
copy_ubuf_to_cbuf_2d_half
mma_tile_half_to_float
fixpipe_nz2nd_float_to_float_4d_to_2d_ubuf
```

MMAD and Fixpipe were removed from that list by the PTODSL MMAD helper and
direct PTO Fixpipe mapping. The three DMA declarations were then removed by
structured HIVM-to-PTO conversion before `convert-hivm-to-std`. This history
is useful when comparing an older output directory, but it is no longer the
current blocker.

## Current Work

The three structured DMA mappings and focused conversion tests are complete.
`emit-vmi` and `emit-vpto` pass for `flash_atten`; `matmul_64` still links,
runs, and compares successfully after the change. Next work is:

1. Update or isolate the stale vendored PTO synchronization contract used by
   the NPU-IR bridge.
2. Map each NPU-IR synchronization mode to the current PTO operation family.
3. Preserve AIV0/AIV1 participation when converting logical flag IDs to the
   physical intra-block semaphore IDs expected by PTOAS.
4. Rerun the complete fat-object, simulator, and numerical-comparison flow.
5. Generalize each DMA mapping only when another real fixture proves a new
   shape, layout, datatype, padding, or stride contract.

The first correctness specialization is complete for `qk_matmul`: its 64x64
f16 implicit-transpose load lowers without the legacy
`load_gm_to_ubuf_2d_half` declaration and reaches VPTO. The helper is generated
offline from PTODSL and loaded as MLIR during compilation. Other shapes,
datatypes, padding modes, and layouts deliberately fail with an explicit
unsupported-contract diagnostic. Native NDDMA-like PTO support remains future
work because the in-place vector transpose is expected to be slower.

The qk simulator fixture now uses identity Q matrices and a deterministic,
non-symmetric K matrix, with an FP32 `Q @ K^T` reference converted to FP16.
Every produced value in output columns 0-15 matches that reference exactly.
Columns 16-63 remain zero. This validates the implicit-transpose helper for the
currently consumed slice and separates the remaining correctness issue from
the load conversion. Trace that issue through UB-to-L1 NZ packing, MMAD layout,
and Fixpipe before changing the load helper again.

## Compatibility Alternative

The existing `external-calls` bridge mode can provide the NPU-IR AICore
bitcode that defines legacy C-interface helpers. That route remains useful as
a compatibility and behavior reference, but it is not the intended PTODSL
solution because it preserves opaque CCE helper dependencies.

Do not silently add the external bitcode fallback to PTODSL mode. Doing so
would make a nominally PTO-native artifact depend on legacy helper
implementations and would hide incomplete conversion.

## Acceptance Criteria

The PTODSL flash-attention simulator milestone is complete when:

- the generated VMI and VPTO contain no unresolved legacy CCE helper calls
  (complete for the three DMA helpers covered here);
- all replacement operations preserve source memory, layout, synchronization,
  and mixed-kernel section semantics;
- PTOAS emits and links the mixed fat object without NPU-IR AICore bitcode;
- the simulator launches both required MIX components successfully;
- numerical comparison passes against the testcase reference;
- focused negative tests reject unsupported shapes, types, layouts, or modes
  instead of leaving unresolved external declarations.

## Open Questions

- Does NPU-IR's `INTRA_BLOCK_SYNCHRONIZATION` require one vector subblock or a
  collective pair at each generated dependency site?
- Which existing NPU-IR control-flow guards identify AIV0-only, AIV1-only, and
  both-AIV dependencies strongly enough for deterministic conversion?
- Should NPU-IR vendor the current named PTO sync operations, or should the
  bridge first introduce a local compatibility operation and lower it later?
- Can L0A/L0B scratch eventually be passed from an explicit allocator instead
  of being selected by the adapter's provisional two-bank policy?
