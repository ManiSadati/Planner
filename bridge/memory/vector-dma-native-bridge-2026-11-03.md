# Vector DMA runtime tests: native baseline and bridge failures

Updated: 2026-11-03. Recorded simulator runs: 2026-11-02; integer lowering
investigation: 2026-11-03. This publication adds tests and evidence, not fixes.

## Scope and result

The retained suite has **225 explicit configurations across 34 native instances**.
All 225 have exact-instance selection and exact numerical/guard passes through
native NPU-IR. This combines 217 unchanged earlier passes with eight validated
replacement configurations; it is not a fresh 225-case native rerun after those
replacements. The identical retained matrix was run through the PTODSL bridge
with `FALLBACK=1`, using up to 64 workers with private caches/logs/profiles.

| Flow/outcome | Configurations |
| --- | ---: |
| Native: exact target and numerics pass | 225 |
| Bridge: exact native target fallback and numerics pass | 113 |
| Bridge: exact PTODSL target/body and numerics pass | 4 |
| Bridge: conversion fails | 105 |
| Bridge: numerical mismatch after PTODSL selection | 3 |

These are three repair categories, with two distinct symptoms in the DMA
replacement category. Each failing configuration is classified by its observed
first failure; subsequent failures could be exposed after repairs.

| Category | Count | Representative failing configuration |
| --- | ---: | --- |
| F16 DMA replacement / argument adapter | 9: 6 ABI + 3 numerical | `load/2d/f16/small-odd-t4x16` |
| I64 native helper temporary not normalized into VMI | 40 | `load/1d/i64/aligned-t128` |
| I8/I16 AVE store datatype conversion missing | 59 | `load/1d/i16/aligned-t128` |

The actual AVE operation is `ave.hir.masked_store`, with a distribution such as
`NORM_B16`; there is no `ave.hir.masked_norm` operation in these diagnostics.

## Tests and complete case ledger

NPU-IR branch: `mani/vec-dma-test`, based on freshly fetched
`wilsoncxfeng/master` at `a0876a76082e604016350a1825286ed945093039`.
Only the DMA tests and their README are added; no compiler, template, submodule
or shared installation changes. The original `mani/update-fix-oct` checkout is
preserved. Publication commit IDs are recorded in the companion
[publication record](vector-dma-2026-11-03/publication.json).

Test directory, relative to the NPU-IR repository:
`bishengir/test/Integration/PTOASBridge/template_runtime/a5/Vector/DMA/native-instances/`.
It contains six pytest modules (load/store, each 1D/2D/3D), shared
`dma_test_utils.py`, shared `dma_ir_utils.py`, and a README.

| Direction/rank | Datatypes | Configurations per instance | Total |
| --- | --- | ---: | ---: |
| Load 1D | BF16, F16, F32, I16, I32, I64, I8 | 8 | 56 |
| Load 2D | Same seven | 8 | 56 |
| Load 3D | I32, I64 | 8 | 16 |
| Store 1D | Same seven | 6 | 42 |
| Store 2D | Same seven | 5 | 35 |
| Store 3D | F32, I32, I64, I8 | 5 | 20 |

The [225-case ledger](vector-dma-2026-11-03/cases.csv) records every instance,
configuration, native/bridge outcome, selected symbol, reported dependency
fallback, error, log/evidence path, and existing reproduction command.
The [108-failure ledger](vector-dma-2026-11-03/failures.csv) is the failure-only
view; [per-instance counts](vector-dma-2026-11-03/instance-summary.csv) and
[failure groups](vector-dma-2026-11-03/failure-groups.json) preserve exact names.
An empty selected-symbol field on a failed compilation is not a selection pass;
the exact native calls before VMI are documented in the captured integer IR.
An empty dependency list on a failed compilation is not proof of no dependencies.

Physical UB padding outside the logical Triton value remains unclaimed.
The owner excluded 11 configurations selecting out-of-scope `stride_store_*`.
Eight short strided-source-tail cases originally hit a native zero-burst defect;
their explicit replacements retain useful stride/offset/tail coverage as
documented in the test README. The original unsupported contract and its logs
remain preserved, rather than being silently counted as supported.

## 1. F16 DMA replacement and adapter: nine failures

**Single runtime reproducer:**
`test_load_gm_to_ubuf_2d.py::test_load_native_instance[f16-small-odd-t4x16]`.
Exact native instance: `load_gm_to_ubuf_2d_half`.
Logical shape `(3,13)`, tile `(4,16)`, source strides `(23,1)`, destination
strides `(29,1)`, zero additional offsets, one round/program, four warps and
one stage. Allocations also include the utility's guard prefix. The ordinary
masked load uses `other=0` and makes the logical padded lanes observable.

Native passes. Bridge selects **`__pto_load_gm_to_ubuf_2d_half`**, materializes
its PTO body, and executes. The unchanged reference reports **nine wrong logical
padding elements, zero wrong valid-data elements, and zero wrong guards**.
The other two numerical failures have 15 and 21 wrong logical padding elements.
Unrelated output transport falls back to `store_ubuf_to_gm_1d_half`.

The captured [materialized body](vector-dma-2026-11-03/f16-selected-body.mlir)
uses two-byte `pto.mte_gm_ub` bursts/loops but does not consume its padding
mode/value/left operands. Its Python source,
`bishengir/lib/Template/lib/RegBase/Vector/load_gm_to_ub_ptodsl.py`, explicitly
implements the NDDMA branch and retains those arguments without consuming them.
That is a concrete contract gap to investigate. The runtime evidence establishes
incorrect padding with this selected replacement; it does not independently
isolate the template from adapter, buffer-initialization or dependency effects.

**Potential fix:** establish the applicable native branches, preserve existing
UB initialization and reproduce the native padding behavior in the helper or its
existing adapter. Check sizes/strides/offsets and event ordering against the
native path. Do not remove tail assertions or weaken exact comparisons.

The same category also has **six compile-time ABI failures**. A useful secondary
reproducer is
`test_load_gm_to_ubuf_2d.py::test_load_native_instance[f16-multi-tail-t8x32]`:
shape `(19,73)`, tile `(8,32)`, strides `(91,1)` / `(97,1)`.
The selected helper exposes raw PTO pointers plus scalar descriptor fields, but
the surviving native call has two dynamic strided memrefs and native scalar
arguments. The diagnostic says the instance must provide its argument adapter.
Four load cases fail directly; two store cases fail in this load dependency.
**Potential fix:** extend the existing per-instance adapter to the native-valid
descriptor/subview branches, preserving pitches, offsets, sizes and scalar
operands. Do not implement a new ABI architecture as a side effect.

Logs under `$HOME/tmp/dma-bridge-fallback-225-20261102-verified/`:

- Runtime example: `full/0065/run.log`, `full/0065/evidence/`, saved
  `numerical-failure.pt`, `bridge-selection.json`, and `numerical-analysis.json`.
- ABI example: `full/0066/run.log` and `full/0066/evidence/`.
- All three saved-tensor analyses: `numerical-analysis.json` at that root;
  a portable descriptive copy is [included here](vector-dma-2026-11-03/numerical-analysis.json).

## 2. I64 temporary/helper graph: 40 failures

**Single reproducer:**
`test_load_gm_to_ubuf_1d.py::test_load_native_instance[i64-aligned-t128]`.
Exact native instance: `load_gm_to_ubuf_1d_int64_t`.
Shape/tile `(128,)`, unit source/destination strides, no additional offsets,
one round/program, four warps and one stage. Physical GM offsets include the
32-element guard prefix. Native passes exact selection and numerical comparison.

Shared AVE-to-Standard lowering transforms the conditional zero-fill into:

```mlir
%storage = llvm.alloca ... !llvm.struct<"vector_2xvl_s64", ...>
call @_mlir_ciface_vdups_int64_t(%storage, %zero, %predicate)
scf.for ... {
  func.call @masked_store_NORM_B64_int64_t_rank1(..., %storage)
}
```

The allocation is function-local output storage for the CCE broadcast helper:
two 256-byte vector halves, preserving low/high I64 bits. It is not a DMA tensor
allocation. These helpers are defined in the matching native A5 bitcode.
Native compilation handles the graph; the bridge fails at
`failed to legalize operation 'llvm.alloca'` before PTOAS/testcase execution.

The existing native-I64 graph normalizer in
`bishengir/lib/Conversion/HIVMAVEToPTOASVMI/HIVMAVEToPTOASVMI.cpp` recognizes
`vbr` but not this `vdups` spelling, and requires allocation users in the same
block. This case broadcasts outside a loop and stores inside it. Adding the
helper spelling alone therefore does not cover the observed graph.

**Potential fix:** normalize this bounded helper graph to I64 VMI values,
including `_mlir_ciface_vdups_int64_t`, the consuming
`masked_store_NORM_B64_int64_t_rank1`, and the observed predicate casts. Preserve
dominance and loop reuse, I64 bit representation and mask semantics. Do not
simply legalize arbitrary `llvm.alloca` while leaving pointer helper semantics
unresolved. A focused VMI conversion patch is a plausible approach; it has not
been implemented or validated by this test work. The owner assigns this repair
to a teammate.

Evidence: `full/0040/run.log` / `full/0040/evidence/` under the verified bridge
root; full diagnostic `investigate-i64-alloca/compiler.stderr.log`.
The [captured pre-VMI module](vector-dma-2026-11-03/i64-before-vmi.mlir) preserves
the helper graph. Additional native compile-only confirmation is in
`native-i64-confirmation-20261103/`; the original simulator numerical baseline
is `$HOME/tmp/dma-native-236-20261102/full/0040/evidence/`.

## 3. I8/I16 AVE ordinary-store datatype support: 59 failures

**Single reproducer:**
`test_load_gm_to_ubuf_1d.py::test_load_native_instance[i16-aligned-t128]`.
Exact native instance: `load_gm_to_ubuf_1d_int16_t`.
Same `(128,)` shape/tile, unit strides, guard offsets and launch configuration
as the I64 example. Native passes selection and numerical comparison.

Compiler-generated UB zero-fill contains:

```mlir
ave.hir.masked_store <NORM_B16> ... vector<128xi16>
```

The example's predicate enables all lanes. The existing VMI masked-store
converter accepts normal F16/BF16 `NORM_B16` and F32/I32 `NORM_B32` stores;
`isNormalStorePatternForType` rejects I16 `NORM_B16` and I8 `NORM_B8` before
reaching its all-active-mask fast path. Predicate-byte storage has a separate
branch and does not establish ordinary I8 data-store support.

**Potential fix:** add the width-matched ordinary I8/I16 store mappings and
necessary scoped memory/vector-type handling. I8 is already admitted by the
general VMI data-type helper; I16 is not, though narrowly scoped I16 broadcast
support exists. Avoid globally enabling unsupported I16 arithmetic. Validate
predicate lane granularity, signedness/bit preservation, pointer element offsets
and downstream PTOAS lowering before calling the extension complete. The native
passes show these are valid configurations; no intrinsic datatype defect has
been established.

This is surrounding AVE UB initialization, not an arbitrary predicate argument
to the DMA template. `FALLBACK=1` can preserve native template dependencies but
does not supply a converter for the remaining AVE operation.

Evidence: `full/0024/run.log` / `full/0024/evidence/` under the verified root;
full diagnostic `investigate-i16-pipeline-20261103/compiler.stderr.log`.
The [captured failed-pass module](vector-dma-2026-11-03/i16-vmi-failed.mlir)
contains the rejected store and exact native DMA calls. It is a failed-pass
snapshot, not proof of successful VMI conversion.

## Reproduce and verify after fixes

The case ledgers retain original, per-case commands on this server. For the
three primary examples and the secondary ABI example:

```bash
bash "$HOME/tmp/dma-bridge-fallback-225-20261102-verified/run.sh" 65 reproduce-f16-padding
bash "$HOME/tmp/dma-bridge-fallback-225-20261102-verified/run.sh" 40 reproduce-i64-fill
bash "$HOME/tmp/dma-bridge-fallback-225-20261102-verified/run.sh" 24 reproduce-i16-fill
bash "$HOME/tmp/dma-bridge-fallback-225-20261102-verified/run.sh" 66 reproduce-f16-abi
```

Each command requires a fresh output label, uses the original checkout paths,
and selects one exact pytest node under `msprof op simulator`. These diagnostic
runners are preserved local evidence, not new installed tooling or committed
test infrastructure. To run the new worktree directly, select the listed node
from its test directory using Python/pytest under the same simulator profile;
give parallel jobs separate caches, temporaries, profiles and evidence paths.
Check that the compiler/build actually includes the intended repair first.

Next-agent scope: repair the agreed converter or existing DMA adapter/helper
files, not the testcase references or shared installation. Start with these
bounded reproductions, then rerun every affected retained case and the whole
225-case bridge matrix before claiming resolution. Require unchanged exact
numerics/guards, actual padding-operand provenance where applicable, exact
target selection, and materialized PTO bodies for replacements. Report unrelated
dependency fallback separately. New datatype/normalizer support also needs
focused compiler regression checks. Native baseline assertions stay intact.

## Provenance and limits

- NPU-IR source and actual compiler: `a0876a76082e604016350a1825286ed945093039`,
  using `$HOME/AscendNPU-IR/build/bin/bishengir-compile` through private wrappers.
  The activation-default installed `3bf5d9898` compiler was not used for these runs.
- Ascend LLVM: `95c4693447af2473be62bdee19522cbe596f9e96`.
- PTOAS: `5ce40f710f7f83b8d1fb2d8c36647bb811236a49` (CLI 0.69);
  PTO LLVM: `fa2fd1f75d742fec91ecf40d98c9c7961647ec42`.
- Existing `$HOME/.venv/npuir-sim-3.6-candidate/bin/python`; compilation uses
  CANN 9.2 child wrappers, Torch/simulator runtime uses CANN 9.1.0-beta.3.
- Simulator `Ascend950PR_9589`, A5 `dav-c310`; bridge mode `ptodsl`, `FALLBACK=1`.
  Native AIC/AIV fallback bitcode comes from the matching build's
  `build/lib/meta_op.{aic,aiv}.c310.O2.bc`, rather than activation's older bundles.
- [Compact provenance and hashes](vector-dma-2026-11-03/provenance.json) and
  [recorded final audit](vector-dma-2026-11-03/final-audit.json) accompany the report.
  Actual A5 options/flow/matching fallback modules were checked for all 225 bridge
  attempts; all 551 protected source/tool hashes matched at the recorded audit.
- Authoritative full bridge logs: `$HOME/tmp/dma-bridge-fallback-225-20261102-verified/`.
  Native logs: `$HOME/tmp/dma-native-236-20261102/` plus
  `$HOME/tmp/dma-stride-replacements-20261102/`. Large logs/tensors remain local.
- Exclude `$HOME/tmp/dma-bridge-fallback-225-20261102/`: that superseded diagnostic
  run had a wrapper capability-probe error and omitted an A5 compiler option.
  Its results are not used in these counts.
- No new simulator runs for this publication, no physical-A5 or performance
  validation, and no claim that 113 native-fallback passes validate replacements.
