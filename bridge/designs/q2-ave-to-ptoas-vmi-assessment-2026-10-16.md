# Q2 AVE-to-PTOAS VMI assessment

Assessment date: 2026-10-16. Expanded with before/after examples and reassessed against upstream master on 2026-10-16. Analysis only; no compiler, kernel, template, or pipeline implementation was changed.

## Summary

**Observed:** Several Q2 operations need more than a direct instruction replacement. The bridge sometimes computes extra vector elements, changes how values are held, or expands one operation into several. Some of these workarounds preserve the required values; others cover only a restricted case. There are also compilation failures after the bridge has already produced valid VMI. The examples below show each difference step by step.

**Upstream reassessment:** F06 (arithmetic masks discarded during lowering) is **fixed upstream**, but remains present in the PTOAS submodule pinned by AscendNPU-IR. Upstream also has select/compare mask-granularity fixes and stateful stream fusion that the submodule lacks. These are not proof that the exact mixed-width select or singleton continuous-store Q2 paths are resolved. Latest master also changed the VMI memory interface, so replacing the submodule without adapting the bridge is not a validated upgrade. See [upstream commits and submodule gaps](#upstream-commits-and-submodule-gaps).

**Recommended priorities:**

1. Bring in upstream arithmetic mask preservation (F06), with its matching layout/granularity changes; protect partial conversions (F01) and validate continuous stores (F05). Retain rejection outside proven consumer graphs.
2. Reproduce the nested vector-scope failure on latest upstream before proposing a PTOAS fix; fix bridge barrier lifetime handling independently. The former blocks backward layernorm D1200; the latter is a separately recorded blocker in recovered dqkwg.
3. Complete packed BF16 store-carrier handling in the assessed main checkout. Its F16 counterpart exists, but the BF16 form actually present in dqkwg remains rejected.
4. Optimize mixed-width compare/select and B16 broadcast/widening paths after correctness and full-kernel compilation are established.
5. Re-run the exact Q2 corpus with a pinned compiler/PTOAS pair. Historical failure counts and simulator successes are not current end-to-end validation.

**Observed historical performance:** Two forward layernorm cases were measured slower through PTOAS: 442.245 → 489.96 μs (+10.79%) for [2-4096-1200-silu], and 4.3 → 4.9 μs (+13.95%) for [2-2-64-silu]. These are whole-kernel observations from [OLD:23](/home/m00967009/Workspace/Planner/bridge/planning/ptoas_failure_categorization.md:23), not measurements attributing cost to any finding below or to the latest source.

**Scope discipline:** Predicate-memory PB8/PB16/PB32 is a real bridge-wide gap, but the available Q2 evidence does not establish the packed-predicate spill/reload path for an exact Q2 specialization. It is therefore an open question, not a confirmed Q2 inventory item. Similarly, signed vector-scalar min/max and the specific F32-to-F16 row-split fixpipe tracker example are not imported into the AVE findings merely because they appear in the tracker.

**Fusion follow-up (2026-10-19):** [How the bridge can enable later PTOAS fusion](bridge-ptoas-fusion-opportunities-2026-10-19.md) audits actual pass matchers and pipeline gates, including interface migration, broadcast-load combining, layouts, masks/reductions, streams, fused arithmetic, and the separate tile-fusion lifecycle. It distinguishes existing support from proposed bridge changes and unmeasured performance effects.

## Read this first: what the examples mean

The central question is: **does the bridge preserve what the kernel asks the NPU to do?** There are three common outcomes:

- **Same answer, extra steps:** the bridge can calculate the right values, but needs additional conversions, broadcasts, or memory operations.
- **Same answer only in a checked case:** a workaround is safe because the kernel uses only a particular result, such as the first vector element. Other uses must be rejected.
- **Cannot compile or may change behavior:** either a necessary operation is missing from the representation, or a later lowering step loses information.

Start with [F01: converting one active element](#f01), [F04: reduction result shapes](#f04), [F08: splitting and widening](#f08), and [F16: mixed-width selection](#f16). These explain the main representation differences with before/after code. Then read [F05: continuous stores](#f05), [F09: the layernorm compile failure](#f09), and [F10: rejected BF16 packing](#f10) for the important blockers.

### A small glossary

| Term in the report | Meaning in the examples |
|---|---|
| Lane | One element of a vector. Lane 0 is its first element. |
| vector<64xf32> | NPU-IR's type for 64 F32 elements. |
| !pto.vmi.vreg<64xf32> | PTOAS VMI's type for the same number of logical elements. It does not necessarily mean one physical register. |
| Mask / predicate | One true/false choice per lane, saying which elements participate. VL1 enables the first lane; VL32 enables the first 32. |
| Broadcast | Repeat one value across many lanes. |
| B16 / B32 | 16-bit / 32-bit element or predicate granularity. F16 and BF16 values are 16 bits; F32 values are 32 bits. |
| Sparse B16 arrangement | Logical 16-bit values occupy spaced positions in a physical register, for example the even positions. Other positions are not additional useful logical values. |
| UB / GM | On-chip Unified Buffer / global memory. |
| memref | A memory view carrying shape, offset, stride, and memory-space information. |
| !pto.ptr | A typed memory address. Shape and stride must be supplied elsewhere if needed. |
| bitcast | Reinterpret the same bits as another type; it does not numerically convert the value. |
| VMI / VPTO | PTOAS logical vector operations / later physical register and instruction operations. |
| Lowering | Replacing a higher-level operation with operations closer to machine instructions. |
| Vector scope | A region where vector-register values may be created and used under the compiler's lifetime rules. |

### How to read the code

Each finding now starts with its purpose, NPU-IR input, PTOAS output, and a plain-language explanation. The longer source audit follows under **Source evidence and limits**.

The snippets are **excerpts**, not standalone programs. Variable names are made descriptive and unrelated attributes/operations may be omitted. An adapted snippet shows an observed mapping from source/tests, not necessarily an exact full-kernel dump. Pseudocode, abbreviated index lists, proposed output, and conditional paths are labeled explicitly. In particular, a failed conversion has no successful current output; its proposed replacement must not be read as implemented support.

~~~text
NPU-IR / AVE              Bridge output              Later PTOAS lowering
ave.hir.*          ->    pto.vmi.*             ->    physical pto.* operations

Some structured CCE-template operations instead become PTODSL/PTO helpers.
~~~

A change from one operation to several logical operations is evidence of a more complicated mapping. It is not automatically the same number of additional machine instructions: later optimization can remove some of them.

## Scope and method

### Sources and revisions

The assessment uses local source and saved artifacts. The tracker was a starting index, not an authority over newer code.

| Project | Examined baseline |
|---|---|
| Planner | HEAD 4f0bd6e898acd54589a9e72b591fb85cba4b3a87 plus existing local modifications, including the tracker and comparison runner |
| AscendNPU-IR | HEAD 7ad85035117bf6a18bc14e3a8cd53ba1fcd2e747; primary implementation assessed |
| AscendNPU-IR-layernorm | HEAD fc6212ae6a763b36d9057f503d2f98ca24d6a153; validation report and corresponding artifacts |
| PTOAS, historical probe checkout | HEAD a3c33b9372617eb06786fb139935e5be797010d7; existing local build used for original probes |
| PTOAS, actual AscendNPU-IR submodule | third-party/ptoas at 3c1c98fcc8ec0c49b8bededc9a0c7599c6a71a02; remote soyu-wilson/PTOAS, main branch |
| PTOAS, canonical upstream master | [7a32cb83ef927dab4b0c4aa928093acacea24abc](https://github.com/hw-native-sys/PTOAS/commit/7a32cb83ef927dab4b0c4aa928093acacea24abc), fetched 2026-10-16; separate source copy under /tmp/ptoas-q2-upstream-20261016 |
| Q2TritonKernel | HEAD ebf78b205a87ecc5cd935a214bd2ef0fc176dcd9 at /home/x84315859/Q2TritonKernel; source read without changing its Git configuration |

A revision alone does not identify existing dirty files or the provenance of every prebuilt binary. Fresh bridge probes used the already available /tmp/layernorm-tools/bishengir-opt, and FileCheck from AscendNPU-IR/build/bin. Fresh downstream probes used PTOAS/build/tools/ptoas/ptoas. These are explicitly distinguished from source inspection and earlier recorded experiments.

**Baseline distinction (O):** The old local checkout and the actual submodule have byte-identical VMI operation definitions, unified-to-legacy lowering, VMI-to-VPTO lowering, mask-granularity assignment, VMI verifiers, and vector-scope inference files examined here. This supports source-level comparison for those paths; it does not establish the provenance of an existing binary. Original local links and original probe results below describe that older baseline. New upstream links pin the exact master revision above. Commit dates alone do not establish whether a fix exists in the submodule: this fork lacks some upstream changes whose dates precede its own HEAD.

Citation labels identify file families; every linked citation opens its concrete absolute path and starting line. B = HIVMAVEToPTOASVMI.cpp; T = HIVMTemplatesToPTO.cpp; A = HIVMAVEOps.td; N = native HIVMAVEToAVEIntrin.cpp; U = VMILowerUnifiedToLegacy.cpp; P = VMIToVPTO.cpp; S = PTOInferVPTOVecScope.cpp; V/I = VMI operation definitions/verifiers; C = VPTOCANN900LLVMEmitter.cpp; R = saved raw dqkwg AVE log. LN/DQ/CS/CV/SD refer to the cited Q2 Python sources. Line intervals following a link are inclusive reading ranges; the link points to their first line.

### Kernel coverage

The historical Q2 corpus contains 140 parameterized cases from 17 pytest functions. Its classification is preserved in [OLD:45](/home/m00967009/Workspace/Planner/bridge/planning/ptoas_failure_categorization.md:45) and its underlying Logs/results.csv and Logs/results.json. Examined families:

- Layernorm gated forward/backward, including the D64, D128, D50 tail, and D1200 paths: Q2 fused_norm_gate.py, saved source-derived IR/VMI/LLVM under Logs/layernorm_compile_oct15/artifacts, and the two full compiler fixtures layernorm-gated-fwd-128.mlir and layernorm-gated-bwd-64.mlir.
- chunk_bwd_kernel_dqkwg: Q2 chunk_o.py, full older AVE dump in Logs/raw_bwd_dqkwg_sep29.log, recovered representation in /tmp/chunk-dqkwg-recovered.mlir, and focused bridge regressions. Durable citations below use the saved raw log.
- chunk_local_cumsum scalar/vector paths: Q2 cumsum.py, reduced Q2 failure logs, native CCE and PTODSL scan implementations.
- causal_conv1d backward initial-state/varlen paths: Q2 causal_conv1d_bwd.py and reduced legalization failures.
- Structured-stage context from chunk_bwd_dv_local, gated_delta_rule forward h/backward dhu, chunk_fwd_o, scaled_dot_kkt, recompute_w_u, solve_tril/mxr fixed and varlen tests; prepare_wy_repr_bwd is classified separately because its recorded BF16 exponential rejection precedes the bridge.
- Q2 sdpa.py and related attention/cross-entropy source occurrences were inspected to check tracker applicability. Their exact generated AVE/VPTO was not available in the reviewed evidence; source-level applicability is labeled conditional below.

This is an initial evidence-backed assessment of those paths, not an exhaustive certification of every kernel in the Q2 repository or every autotuning specialization. Tests can invoke preprocessing/forward kernels before the named backward kernel: the failing function/location, rather than the pytest name alone, determines attribution.

### Two stages and validation boundaries

Stage 1 converts structured HIVM/CCE-template behavior to PTODSL/PTO sequences. Stage 2 converts AVE operations and their surrounding pointer, synchronization, and function contracts to VMI. The active pipeline orders template conversion before standard/metadata lowering and AVE-to-VMI conversion ([PIPE:294](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Tools/bishengir-compile/regbase/PassPipeline.cpp:294)–313). Some Stage 1 helpers already contain physical PTO operations; they are not evidence that AVE instruction lowering should bypass VMI.

PTOAS subsequently assigns logical layouts, physicalizes VMI, and infers VPTO vector scopes; see [MAN:122](/home/m00967009/Workspace/PTOAS/docs/designs/vmi-implementation-manual.md:122)–147 and the actual lowering code cited below. Parser/verifier acceptance, VMI conversion, VPTO emission, LLVM compilation, simulator correctness, hardware correctness, and performance are separate milestones.

Labels used throughout:

- **O / Observed:** directly present in source, test, saved artifact, or a fresh command result.
- **I / Inference:** a semantic/performance conclusion drawn from that evidence.
- **Q / Unresolved:** requires exact IR, execution, or measurement.

## Upstream commits and submodule gaps

### Which PTOAS is being compared?

**O:** “Current submodule” means AscendNPU-IR/third-party/ptoas at `3c1c98fcc8ec0c49b8bededc9a0c7599c6a71a02`. “Upstream” means canonical `hw-native-sys/PTOAS` master at `7a32cb83ef927dab4b0c4aa928093acacea24abc`, rechecked with `git ls-remote` on 2026-10-16. Neither checkout nor AscendNPU-IR’s gitlink was updated.

**O:** The entries below are capabilities implemented by upstream commits but absent from the pinned submodule’s examined code. They are a Q2-relevant list, not a complete upstream release changelog. Some upstream commits are older than the submodule HEAD date: the submodule is a fork, so a newer date does not imply it includes those changes.

### Fixes implemented upstream but absent from the submodule

| ID / affected finding | Upstream commit | What was wrong in the submodule | What upstream implements | Q2 status and remaining limit | Likely performance effect / comparison needed |
|---|---|---|---|---|---|
| U01 / F06, dqkwg H/VL1 arithmetic and tails | [2503c259](https://github.com/hw-native-sys/PTOAS/commit/2503c259e0c40da604eeff8ba5e952ebdb858bfe), 2026-09-14, `fix(vmi): preserve masks in direct unified VPTO lowering` | Unified masked binary/unary operations are rewritten to legacy operations without the predicate. | Direct unified-to-VPTO arithmetic lowering carries the mask separately for each physical result part. Matching mask/layout handling is included. | **O: lowering defect fixed upstream.** Still present in pinned submodule. **Q:** full Q2 numerical/FP-status equivalence remains unmeasured. F01 masked conversions are separate and remain unresolved. | **I:** avoids computing inactive lanes as all-active; does not require a generic extra select for each add. No speedup measured. Compare emitted arithmetic/predicate counts, register use, and masked-tail kernel timings. |
| U02a / F16, mixed-width select | [3522b219](https://github.com/hw-native-sys/PTOAS/commit/3522b219a2260b57e4bcad660567569c0646ffdc), 2026-09-21, `fix(vmi): assign mask granularity for unified vsel` | Unified vsel is absent from result-width mask-use constraints; a legitimate F16 use may receive the default B32 mask and fail verification. | Select uses request B8/B16/B32 according to data width; cheap masks can be regenerated and comparison masks receive ensure_mask_granularity at the use. | **O: granularity-assignment defect fixed upstream.** The test includes a B32 comparison feeding F16 selection. **Q:** this is not an end-to-end test of Q2’s signed-I32, 64-lane sparse physical path; do not yet remove its widening workaround. | **I:** may permit replacing two data extensions and a truncation with predicate conversion. Physical mask conversion can itself cost instructions. Compare exact Q2 F16 selection with/without bridge widening, including register pressure and machine instructions. |
| U02b / F16 and Q2 comparison predicates | [981abdd9](https://github.com/hw-native-sys/PTOAS/commit/981abdd90561ff697cab95799ba9e90068181867), 2026-10-08, `fix(vmi): constrain unified vcmp/vcmps mask granularity; pin small-VL vsel/vcmp regressions (#1376 #1381)` | Direct granularity assignment does not constrain unified compare result/seed masks by source element width. | Adds source-width constraints for compare result and seed; includes small-vector select/compare regressions. | **O: direct-pass defect fixed upstream.** The commit explicitly says the tested complete upstream pipeline already handled its original small-VL cases; this is not evidence that every Q2 failure was fixed by this commit alone. Exact F16 bridge replacement remains Q. | **I:** primarily compilation robustness; no independent speedup follows from fixing standalone assignment. Measure only after testing the actual replacement sequence. |
| U03 / F05 and eligible S02 memory streams | [5eb87c21](https://github.com/hw-native-sys/PTOAS/commit/5eb87c21ab9479d834e66968f63f0b1def292764), 2026-08-29, `feat(vmi): legalize physical accesses and fuse stateful streams` | No stateful-stream fusion pass exists; no automatic cross-iteration alignment-state fusion is available through that pass. | Legalizes physical memory accesses and fuses eligible adjacent/loop streams, carrying base and alignment state through loops and placing the final flush after them. Enabled in the upstream pipeline. | **O: capability implemented upstream, absent in submodule.** Regression covers contiguous 64-F32 chunks, static loops, and refusal of unsafe cases. **Q:** exact singleton ONEPT_B32 dqkwg sequence is not established as eligible. F05 remains partially addressed, not closed. | **I:** eligible streams can share initialization/flush rather than repeat them, reducing overhead. Compare init_align/vstas counts and loop-carried state for exact dqkwg and D50 copy before timing; no Q2 benefit is measured. |

### U01: the outdated finding corrected in this report

The old report said “PTOAS drops arithmetic masks.” The precise replacement is: **the pinned submodule drops them; latest upstream preserves explicit arithmetic masks.**

~~~text
NPU-IR:       ave.hir.vadd(a, b, first32)
Bridge VMI:   pto.vmi.vadd(a, b, first32)

Submodule:    maskless legacy add -> physical add with all-active predicate
Upstream:     unified masked add -> physical add using first32
~~~

**O:** [Unified masked lowering](https://github.com/hw-native-sys/PTOAS/blob/7a32cb83ef927dab4b0c4aa928093acacea24abc/lib/PTO/Transforms/VMIToVPTO/VMIToVPTOUnifiedPatternInternals.cpp#L57) resolves each physical part’s predicate and passes it into the operation builder. [Masked-compute regression](https://github.com/hw-native-sys/PTOAS/blob/7a32cb83ef927dab4b0c4aa928093acacea24abc/test/lit/vmi_new/vmi_to_vpto_unified_masked_compute.pto#L9) checks `PAT_ALLF` reaching `pto.vadd`, and a partial predicate reaching `pto.vshr`. [Lane-stride mask regression](https://github.com/hw-native-sys/PTOAS/blob/7a32cb83ef927dab4b0c4aa928093acacea24abc/test/lit/vmi_new/vmi_to_vpto_unified_lane_stride_mask.pto#L1) also covers layout/lane-stride-dependent masks. **O:** the old local binary successfully lowered the all-off test input but failed its upstream FileCheck because the required all-off predicate was missing. **Q:** the latest binary was not completed locally, so upstream success here is source/test-contract evidence, not a fresh local pass or device run. `pmode="merge"` is rejected upstream; do not interpret this as arbitrary merge behavior.

### U02: convert the predicate before widening the data

Upstream’s `vsel_compare_f16` regression expresses the relevant alternative, adapted below:

~~~mlir
// B32 comparison produces a B32 predicate.
%cmp = pto.vmi.cmpf "olt", %lhs, %rhs
    : !pto.vmi.vreg<128xf32>, !pto.vmi.vreg<128xf32>
      -> !pto.vmi.mask<128xb32>

// Upstream granularity assignment inserts this use-site conversion.
%mask16 = pto.vmi.ensure_mask_granularity %cmp
    : !pto.vmi.mask<128xb32> -> !pto.vmi.mask<128xb16>
%r = pto.vmi.vsel %mask16, %a, %b
    : !pto.vmi.mask<128xb16>, !pto.vmi.vreg<128xf16>,
      !pto.vmi.vreg<128xf16> -> !pto.vmi.vreg<128xf16>
~~~

**O:** [Select granularity regression](https://github.com/hw-native-sys/PTOAS/blob/7a32cb83ef927dab4b0c4aa928093acacea24abc/test/lit/vmi_new/vmi_mask_granularity_assignment_vsel.pto#L111) lines 111–130 checks this transformation; [Select use constraint](https://github.com/hw-native-sys/PTOAS/blob/7a32cb83ef927dab4b0c4aa928093acacea24abc/lib/PTO/Transforms/VMI/VMIMaskGranularityAssignment.cpp#L253) registers unified select’s result-width constraint. [Comparison result constraint](https://github.com/hw-native-sys/PTOAS/blob/7a32cb83ef927dab4b0c4aa928093acacea24abc/lib/PTO/Transforms/VMI/VMIMaskGranularityAssignment.cpp#L244) also constrains unified comparison result masks. **I:** this is cleaner than widening both data values just to use a B32 predicate. **Q:** the example uses 128 F16 lanes and a floating compare; Q2 F16 uses signed I32 comparison and 64 F16 lanes with layout constraints. Re-run that exact chain through layout assignment and physical emission before deleting the bridge workaround. The old binary fails the upstream vsel assignment test at the F16 mask-width mismatch; this confirms the old defect but does not measure the upstream physical alternative.

### U03: streams can now carry state, with eligibility limits

Upstream’s `fuse_store_loop` regression writes **64 F32 values per iteration**, at offsets `1 + i*64`. The physical fusion is schematically:

~~~text
Before eligible-stream fusion:          After eligible-stream fusion:
for i in 0..4:                         state = init_align()
    state = init_align()               base = dst + 1
    state, base = vstus(chunk[i], ...)  for i in 0..4:
    vstas(state, base)                     state, base = vstus(chunk[i], ...)
                                       vstas(state, base)
~~~

**O:** [Loop stream fusion](https://github.com/hw-native-sys/PTOAS/blob/7a32cb83ef927dab4b0c4aa928093acacea24abc/lib/PTO/Transforms/VPTO/VPTOStatefulStreamFusion.cpp#L689) introduces loop-carried state and emits vstas after the new loop. [Stream-fusion regression](https://github.com/hw-native-sys/PTOAS/blob/7a32cb83ef927dab4b0c4aa928093acacea24abc/test/lit/vmi_new/vmi_to_vpto_stateful_stream_fusion.pto#L58) lines 58–68 supplies the positive loop; later cases keep dynamic-trip, discontinuous, aliased, or side-effect-separated streams unfused. [Upstream pipeline](https://github.com/hw-native-sys/PTOAS/blob/7a32cb83ef927dab4b0c4aa928093acacea24abc/tools/ptoas/ptoas_pipeline.cpp#L996) enables the pass after VMI-to-VPTO. **Q:** the Q2 bridge currently emits one-point stores, not the full-chunk unaligned stream in this test. The pass fuses already recognized stateful physical streams; its presence alone does not prove it will transform singleton `vsts` operations into such streams. First compare the exact dqkwg physical output, final flush, and addresses with native NPU-IR.

### Upstream interface change the bridge must account for

**O:** [2729bb55](https://github.com/hw-native-sys/PTOAS/commit/2729bb556cb0841d589408ebbd6e28fa99126192) (2026-10-13, `refactor(vmi): remove load/store dist_mode`) removed `dist_mode` from the declared VMI memory interface. [Current memory-operation definition](https://github.com/hw-native-sys/PTOAS/blob/7a32cb83ef927dab4b0c4aa928093acacea24abc/include/PTO/IR/VMIOps.td#L1427) now describes broadcast as a one-value vload followed by vbrc. The bridge examples using `dist_mode="brc"` describe the pinned interface; they are not evidence of equivalent latest-master behavior. Merely deleting an attribute is not a validated migration: it can change which addresses or values are read.

~~~mlir
// Pinned bridge form, adapted B16 broadcast example:
%x = pto.vmi.vload %src[%off] {dist_mode = "brc"}
    : !pto.ptr<bf16, ub> -> !pto.vmi.vreg<128xbf16>

// Upstream documented logical representation, proposed bridge replacement:
%one = pto.vmi.vload %src[%off]
    : !pto.ptr<bf16, ub> -> !pto.vmi.vreg<1xbf16>
%x = pto.vmi.vbrc %one
    : !pto.vmi.vreg<1xbf16> -> !pto.vmi.vreg<128xbf16>
~~~

**I/Q:** the new representation describes one memory read plus logical replication, but exact physical layout and synchronization for F02 need probing. For F03, the plain singleton vload alone already produces vector `BRC_B32` in the old binary; do not list that older capability as a new upstream fix. Also audit store operand/interface changes when upgrading the paired toolchain.

### Status of the other findings on latest master

**O:** [Current vcvt definition](https://github.com/hw-native-sys/PTOAS/blob/7a32cb83ef927dab4b0c4aa928093acacea24abc/include/PTO/IR/VMIOps.td#L1384) and [Current extf definition](https://github.com/hw-native-sys/PTOAS/blob/7a32cb83ef927dab4b0c4aa928093acacea24abc/include/PTO/IR/VMIOps.td#L361) still have no conversion-mask operand, so upstream arithmetic fixes do not close F01. [Shuffle source-lane mapping](https://github.com/hw-native-sys/PTOAS/blob/7a32cb83ef927dab4b0c4aa928093acacea24abc/lib/PTO/Transforms/VMIToVPTO/VMIToVPTOMaskInternals.cpp#L275) still rejects padding in the shuffle path with “requires full physical result chunks”; this supports retaining F08’s widening-first workaround, without asserting that every B16 alternative fails. F04 still uses a one-value reduction representation: [the current reduction verifier](https://github.com/hw-native-sys/PTOAS/blob/7a32cb83ef927dab4b0c4aa928093acacea24abc/lib/PTO/IR/VMI/VMIHelpers.cpp#L1074) requires a one-lane result for full reductions. Bridge-side F10/F11/F13/F15/F17 behavior is unchanged by inspecting a different PTOAS checkout; no upstream commit was established that removes those bridge constraints.

**Q:** F09’s old scope reproducer was not rerun on a completed latest build. Its nested segment lookup is still visible in [Nested-use segment collection](https://github.com/hw-native-sys/PTOAS/blob/7a32cb83ef927dab4b0c4aa928093acacea24abc/lib/PTO/Transforms/VPTO/PTOInferVPTOVecScope.cpp#L480), but that alone cannot establish a current full-pipeline failure. F12/F14 still need paired backend/cache/synchronization validation. S01/S02 still need full-kernel correctness and timing. The main findings table deliberately retains these limits instead of treating all old failures as current upstream failures.

### Verification performed for this reassessment

- **O:** Fetched canonical master and history into a separate /tmp checkout; rechecked the remote tip; inspected the actual AscendNPU-IR gitlink and the old/new source implementations. No tracked compiler source or submodule pointer was changed.
- **O:** Located the four fix/feature commits above in upstream history and checked their patches and latest corresponding source/tests. “Missing from submodule” is based on missing behavior/code, rather than operation-name differences or commit dates.
- **O:** Old-binary comparison: upstream unified-masked-compute input lowers with exit 0 but fails new FileCheck (all-off predicate missing); unified-vsel granularity assignment exits 1 (F16/B32 mismatch); stream-fusion invocation exits 1 (unknown pass); short-vector-load input exits 0 and emits vector BRC_B32 for scalar_f32. Its full upstream FileCheck exits 1 on other block-load expectations (PAT_VL1 versus old PAT_VL8/full-load output); only the scalar_f32 result supports the F03 correction. These are compiler probes, not Q2 device tests.
- **O:** A separate latest-master pto-test-opt build was configured under /tmp and compilation was started, then interrupted before completion. No fresh latest-master binary/test success is claimed. Upstream regression expectations and commit-reported test runs are source evidence only.
- **Q:** No latest-master Q2 LLVM/full-kernel/device/simulator timing was obtained. Quantification requires a bridge adapted to the new interface, identical Q2 specializations and compiler flags, and native/pinned/upstream output comparison. Report instruction counts, register use, synchronization and memory operations alongside median device latency.

## Findings table

Read each row from left to right: the AVE pattern, what native AscendNPU-IR does, and what the bridge emits for PTOAS. The new native-lowering column distinguishes observed code (O) from native details that remain unresolved (Q). Each row is a distinct contract or failure mode. Detailed evidence follows under the matching ID. “No additional issue established” does not assert that every specialization works. The final column states the upstream reassessment separately from the current bridge/submodule behavior; it is based on source and upstream regression contracts, not a fresh full-Q2 run.

| Q2 kernel/pattern | AVE operation or abstraction | Native AscendNPU-IR behavior / lowering | Current bridge mapping | Current workaround | PTOAS lowering issue | Semantic match | Performance impact | Recommended approach | Evidence | Upstream master reassessment |
|---|---|---|---|---|---|---|---|---|---|---|
| F01: cumsum input promotion; dqkwg scalar promotion/narrowing | Masked vextf / vtruncf, part_even | O: AVE supplies a mask and part_even selector. Native widening/narrowing passes that mask to vcvt and retains narrowing rounding/saturation. [native code](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Conversion/HIVMAVEToAVEIntrin/HIVMAVEToAVEIntrin.cpp:2117), [native code](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Conversion/HIVMAVEToAVEIntrin/HIVMAVEToAVEIntrin.cpp:2301). | Unmasked extf or vcvt | ALL accepted; VL1 only for checked lane-zero stores, with one multiply allowed for widening | VMI conversions have no mask operand | Exact active values under the guard; other partial forms absent/rejected | I: inactive lanes also converted; not necessarily extra instructions | Add masked conversion semantics; retain guarded rejection meanwhile | [F01](#f01) | O: conversion operations still have no mask; guard remains necessary (see upstream section). |
| F02: dqkwg B16 scalar broadcast used as 64 lanes | vload BRC_B16 | O: Read one UB value and broadcast it on the vector pipeline. Native vlds uses BRC_B16 and a full 128-lane physical B16 register, even for a 64-lane AVE result. [native code](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Conversion/HIVMAVEToAVEIntrin/HIVMAVEToAVEIntrin.cpp:875). | 128-lane brc load + 64-lane lane-zero splat shuffle | Native-width load retains vector pipeline | Direct 64-lane sparse consumer layout cannot be materialized; ordinary partial B16 projection fails | I: exact logical values and address for checked singleton UB forms | I: possible splat/layout cost; one memory read, no scalar load | Make B16 broadcast producers layout-flexible | [F02](#f02) | O: old dist_mode memory interface removed; same bridge output is not a validated upstream input. B16 sparse path still needs an exact probe. |
| F03: dqkwg singleton B32 broadcast staging | vload BRC_B32, one-lane result | O: Native widens the one-lane logical result to a physical vector register and emits vlds with BRC_B32; it remains a vector load, rather than a scalar load. [native code](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Conversion/HIVMAVEToAVEIntrin/HIVMAVEToAVEIntrin.cpp:875). | load_scalar + VMI broadcast | Scalar path only for singleton; multi-lane stays VMI brc load | Restriction of the old dist_mode=brc spelling; plain one-lane vload is an available alternative | Values match; pipeline behavior changes and synchronization must match | I: scalar/vector dependency and possible added broadcast latency | Use/probe existing plain singleton vload before adding compiler support | [F03](#f03) | O: plain one-lane vload has a vector BRC lowering; already available in old binary too. Corrected below; bridge still scalarizes. |
| F04: layernorm statistics and dqkwg gradient reductions | reduction ADD; full-width AVE result with value in lane zero | O: Native ADD reduction emits vcadd-family lowering; AVE retains the vector result type but promises the reduced value at lane 0. No full-width broadcast is promised. [native code](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Conversion/HIVMAVEToAVEIntrin/HIVMAVEToAVEIntrin.cpp:1725). | reduce_addf to one lane + broadcast to original width | Reconstruct result shape; ADD carries reassoc | PTOAS requires one-lane result and permits pairwise reduction | Defined lane-zero behavior matches; bitwise sum order not established | I: avoidable broadcasts if only lane zero is observed | Track defined lanes and eliminate shape-only round trips | [F04](#f04) | O: one-lane reduction representation remains; shape restoration still a bridge concern. |
| F05: dqkwg row reduction stores | continuous ONEPT_B32 | O: Native creates alignment state, emits vstus.post stores, then vstas; continuous-store hoisting carries state/base through the loop and finishes after it. [native code](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Conversion/HIVMAVEToAVEIntrin/HIVMAVEToAVEIntrin.cpp:1119). | singleton vstore → one-point vsts | Stateless store; dependency barriers where needed | Pinned submodule lacks stream fusion; upstream has physical stream fusion, with singleton applicability unresolved | Value-level equivalence unresolved for complete stream; instruction parity absent | I: different store stream/ordering; sign and size unknown | Stateful logical VMI stream lowered to existing VPTO operations | [F05](#f05) | O: stateful stream fusion added (U03); Q: exact singleton ONEPT loop coverage unproven. |
| F06: dqkwg H/VL1 arithmetic and other Q2 tails | Partial arithmetic predicates | O: Native arithmetic lowering passes the AVE predicate to the selected vector intrinsic. H/VL1 therefore still governs the physical computation. [native code](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Conversion/HIVMAVEToAVEIntrin/HIVMAVEToAVEIntrin.cpp:166). | Mask carried into unified arithmetic | No general observability guard on ordinary arithmetic | Pinned submodule discards masks; upstream directly lowers masked arithmetic (U01) | Upstream preserves explicit predicate; pinned baseline is conditional: safe only if inactive lanes cannot affect observed results | I: extra inactive computation; repair may add selects or masks | Integrate upstream mask-preserving lowering with related passes; retain separate F01 guards | [F06](#f06) | O: FIXED upstream by 2503c259; still missing in submodule (U01). |
| F07: dqkwg reduction/scalar wrappers | Singleton VMI ↔ scalar representational casts | O: AVE represents the same useful value through singleton/vector/scalar wrappers. Q: the exact final native instruction sequence for this recovered round trip was not isolated; do not assume a physical scalar extraction. | Fold proven round trips; retain VMI broadcasts | Reject genuine escaping scalar consumers | No general VMI-to-scalar lane extraction; recovered ordering also exposes scope limitations | Exact for checked round trips; no general extraction support | I: avoids spill/reload; broadcasts can remain | Explicit extraction only for real scalar uses; simplify wrappers | [F07](#f07) | Q: no general extraction solution established; round-trip folding remains bridge-side. |
| F08: backward layernorm D64/D1200 | SPARSE self-vintlv + part_even widening | O: Native emits physical interleave, producing the sparse B16 arrangement, followed by part_even masked vcvt to F32. The two operations jointly recover the input halves. [native code](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Conversion/HIVMAVEToAVEIntrin/HIVMAVEToAVEIntrin.cpp:2604), [native code](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Conversion/HIVMAVEToAVEIntrin/HIVMAVEToAVEIntrin.cpp:2117). | Widen dense 128 B16 lanes first; forward two full B32 halves | Whole checked producer/consumer chain rewritten | Partial-result B16 shuffle requires full physical chunks | I: exact logical lanes under checked chain; general sparse carriers rejected | I: avoids physical interleave; two logical shuffles may forward without instructions | Keep widening-first; add safe partial B16 shuffle lowering | [F08](#f08) | O: full-physical-result-chunk shuffle restriction remains; widening-first workaround retained. |
| F09: backward layernorm [2-4096-1200-silu] | Constant vector lifetime across nested scope boundaries | O: AVE can express a zero-vector broadcast and native code emits vector broadcast intrinsics. The reported mask-sharing/scope failure occurs later in PTOAS. Q: the exact native D1200 constant placement was not compared. | VMI constant accepted | No implemented workaround for this blocker; existing manual placement experiment only | Constant introduces pset; CSE shares predicate into nested boundary loop; rematerialization fails | Compilation blocked, not a demonstrated numerical mismatch | No runtime result; manual alternative may rematerialize for 32 iterations | Fix nested-segment discovery and safe immutable rematerialization in PTOAS | [F09](#f09) | Q: old reproducer not rerun on latest; nested segment lookup remains source-visible. Do not call confirmed upstream failure. |
| F10: dqkwg BF16 output packing | part_even truncation → width-wrapper/bitcast → PK_B32 store | O: Native supports F32→BF16 vcvt and BF16 vsts with the source store distribution; it accepts physical packing wrappers. Unaligned stores have a separate vstus/vstas path. [native code](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Conversion/HIVMAVEToAVEIntrin/HIVMAVEToAVEIntrin.cpp:2301), [native code](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Conversion/HIVMAVEToAVEIntrin/HIVMAVEToAVEIntrin.cpp:1045). | F16 carrier collapses to dense VMI store; BF16 analog not accepted in assessed source | F16-only recognizers; none found for BF16 in this checkout | No downstream BF16 conclusion possible before bridge rejection | Absent for observed BF16 carrier; direct logical packing can preserve values | Compilation blocker; I: proper folding needs no new memory traffic | Generalize carrier recognition with bit/lane/mask checks | [F10](#f10) | O: bridge recognizers unchanged; upstream changes cannot repair rejection before PTOAS. |
| F11: causal-convolution views; dqkwg GM/UB addressing | Memref descriptor → pointer-only ABI | O: Native memory lowering uses memref descriptors and getStridedElementPtr to form addresses; offsets/strides remain available when calculating the element pointer. [native code](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Conversion/HIVMAVEToAVEIntrin/HIVMAVEToAVEIntrin.cpp:887). | addptr/castptr with static layout linearization | Restrict metadata uses; lower structured strided accesses before erasure | Pointer-only boundary cannot recover erased sizes/strides/offsets | Exact only under normalized-base and retained-offset contract; rejected Q2 views need fresh rerun | I: address math usually cheap; fallback paths can serialize work | Specify/verify ABI invariant or preserve descriptors explicitly | [F11](#f11) | O: bridge ABI contract unchanged; no general descriptor recovery established. |
| F12: dqkwg marked singleton BF16 GM load | disableDCache on floating memref.load | O: Native preparation marks shared-buffer loads with disableDCache when writes are found, including zero-valued markers. Q: its final native BF16 device-load intrinsic was not isolated here. [marker source](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Dialect/HIVM/Transforms/MarkDisableLoad.cpp:135). | Integer-pointer ld_dev + scalar bitcast | Equal-width i16/i32 payload | Device loads are integer-only; backend/entry/version restrictions | I: bit/address exact for guarded form; cache-bypass retained | I: carrier casts likely free; cache policy may affect latency | Floating device load/cache-policy contract; retain proper backend rejection | [F12](#f12) | Q: no new floating/cache-contract solution established; retain guarded integer carrier. |
| F13: dqkwg strided g/dg transfers | Rank-one non-unit-stride GM gather/scatter | O: The CCE route has synchronized scalar fallbacks for these strided/tail cases; A5 GM writes use st_dev to avoid cached-line interference. It is not necessarily a DMA-only native path. [native fallback](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Template/include/RegBase/DMAUtils.h:394). | Synchronized scalar loops; store uses integer st_dev | Capture length/stride before pointer collapse; event 7 synchronization | No descriptor-aware fallback through ordinary VMI; device-store backend constraints | I: exact logical elements for guarded forms; general layouts outside contract | I: per-element scalar memory/address work and pipe handshakes; native can use same fallback | Match CCE fallback first; use DMA only with proven alignment/tails/cache safety | [F13](#f13) | O: bridge scalar fallback unchanged; no measured upstream improvement established. |
| F14: dqkwg mixed Cube/Vector synchronization | sync_block_set/wait, core-specific A5 semaphore banks | O: HIVM sync_block_set/wait carries core kind, producer/consumer pipes, event ID, and mode. Q: the complete paired native physical sync sequence was not independently captured; the bridge expansion is shown in F14. | Named intra-block ops; Cube emits flag and flag+16; inter-block uses sync.set/wait with mode | Architecture-specific expansion | Old tracker describes stale mode/core loss; remaining generic range/base cases not proven in Q2 | Observed intra-block form improved; full runtime handshake unvalidated here | I: extra Cube semaphore issue per logical sync reflects two banks | Pin/validate exact paired lowering; do not apply stale FFTS claims to intra-block IDs | [F14](#f14) | Q: paired runtime validation still missing; do not reinstate stale tracker claims. |
| F15: Q2 SDPA negative-infinity initializer, conditional scalar staging | F32 -inf used by online maximum | O: The Q2 source requests actual negative infinity. Q: its exact native scalar/vector staging and final constant materialization are unresolved; no native -FLT_MAX substitution is established. | Any directly stored scalar -inf is rewritten to -FLT_MAX | Broad post-conversion constant replacement | Root cause of historical simulator failure not isolated to PTOAS vs simulator | Non-IEEE substitution; exact Q2 trigger unresolved without generated scalar-store IR | Primarily correctness, not a speed optimization | Preserve exact constant; otherwise constrain to proven initializer with exceptional-value tests | [F15](#f15) | O: bridge constant rewrite unchanged; exact Q2 trigger still unresolved. |
| F16: dqkwg signed I32 comparison selecting F16 | B32 predicate driving B16 vsel | O: Native signed comparison retains its predicate; native select emits VselInstrOp for F16/BF16 values using the physical mask. It does not use the bridge's B16→B32 integer-data expansion. [native code](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Conversion/HIVMAVEToAVEIntrin/HIVMAVEToAVEIntrin.cpp:1572), [native code](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Conversion/HIVMAVEToAVEIntrin/HIVMAVEToAVEIntrin.cpp:1615). | Bitcast both data operands to i16, zero-extend to i32, select, truncate, bitcast back | Select exact floating bits through B32 carriers | Avoids sparse B16 comparison/select mask-layout path | I: bit exact for checked equal-lane form; original compare/seed retained | I: two extensions + one truncation; extra registers/layout work | Native predicate granularity conversion and B16 select lowering | [F16](#f16) | O: vsel and compare granularity fixes added (U02); exact sparse mixed-width physical path unvalidated. |
| F17: recovered dqkwg store/load dependencies | Live loaded/computed vectors across inserted VST_VLD barrier | O: AVE allows a loaded vector to have later users. Q: the exact native register lifetime and barrier schedule of the recovered function were not compared. The failure shown here comes from opt-in bridge finalization. | Insert barrier; rematerialize safe values after it | No safe workaround for non-pure crossing value found | Bridge lifetime repair rejects before PTOAS; separate from F09 | Fails safely; moving/reloading arbitrary values could change meaning | I: barriers/duplication could cost; current blocked path has no latency | Coordinate hazard placement and legal register lifetimes, preserving memory order | [F17](#f17) | O: rejection is in bridge finalization; upstream alone cannot resolve it. |
| S01: chunk_local_cumsum vector path | Stage 1 CCE 2-D F32 dim-0 scan | O: Native ordinary floating dim-0 CCE scan copies input and runs a Sklansky tree using vector loads, adds and stores. This is a template composition, not one AVE scan instruction. [native scan](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Template/lib/RegBase/Vector/Cumsum.cpp:215). | PTODSL Sklansky scan helper now exists | Masked gather/scatter rows, reverse indexing, explicit stage barriers | Not a missing VMI scan instruction; exact Q2 rerun still needed | I: matching scan tree intent; runtime/bitwise comparison not established | I: copy + repeated UB accesses + stage barriers; native has similar scan stages | Retain template route; compare logical tails, reverse, aliasing, emitted memory cost | [S01](#s01) | O: template support remains present; no fresh upstream full-kernel result. |
| S02: layernorm D50 UB copy | Stage 1 2-D F32 UB copy | O: Native uses a CCE Copy2D helper that inspects both descriptors and dispatches by strides/shape. Q: the exact aligned/scalar-tail branch selected for Q2 D50 was not isolated. [native copy](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Template/lib/RegBase/Vector/Copy2D.cpp:560). | VMI row/chunk gather + masked store loops | Protect 50-column tail without touching padding | Tail memory safety constrains full-register loads | I: logical values/offsets preserved for static guarded shapes | I: vector/address work can exceed aligned DMA; native unaligned path must be compared | Add aligned DMA fast path only where legal; keep safe tails | [S02](#s02) | O: stream fusion can improve eligible unaligned copies (U03); exact D50 path unmeasured. |

## Detailed evidence and semantic limits

### F01

**Convert only the first value, then store that value.**

**What the kernel needs:** dqkwg has a BF16 value that it needs to use as F32. Although it is held in a 64-lane vector, this particular calculation uses only lane 0.

**NPU-IR before conversion — adapted from the recovered dqkwg test:**

~~~mlir
// VL1 means: enable lane 0; disable lanes 1 through 63.
%first = ave.hir.pge <VL1> : vector<64xi1>

// Convert the enabled BF16 lane to F32.
%wide = ave.hir.vextf %x, <part_even>, %first
    : vector<64xbf16>, vector<64xf32>, vector<64xi1>

// Write only one F32 value.
ave.hir.masked_store <ONEPT_B32> %dst[%zero], %first, %wide
    : memref<1xf32, #hivm.address_space<ub>>,
      vector<64xi1>, vector<64xf32>
~~~

**PTOAS VMI after bridge conversion — adapted from the checked output:**

~~~mlir
// extf has no mask argument: it converts all 64 logical lanes.
%wide = pto.vmi.extf %x
    : !pto.vmi.vreg<64xbf16> -> !pto.vmi.vreg<64xf32>

// [[0]] selects only lane 0.
%one = pto.vmi.shuffle %wide[[0]]
    : !pto.vmi.vreg<64xf32> -> !pto.vmi.vreg<1xf32>

// The other 63 results are never written.
pto.vmi.vstore %one, %dst[%zero]
    : !pto.vmi.vreg<1xf32>, !pto.ptr<f32, ub>
~~~

**Why this can work:** the bridge computes more lanes, but it writes the same lane 0. Think of converting 64 numbers and then keeping only the first one. The unused results do not change the stored value.

**Why support is still incomplete:** this reasoning fails if another operation uses the other 63 results. The bridge therefore checks the consumers and rejects broader uses. The partial-mask cumsum failure is a different use graph; the lane-zero workaround does not automatically fix it.

The reverse conversion has the same limitation:

~~~mlir
// NPU-IR: narrow only the first active F32 lane.
%narrow = ave.hir.vtruncf %x, <rint>, false, <part_even>, %first
    : vector<64xf32>, vector<64xbf16>, vector<64xi1>

// PTOAS VMI: narrow all lanes, retaining rounding and saturation settings.
%narrow = pto.vmi.vcvt %x {rounding = "R", saturate = "NOSAT"}
    : !pto.vmi.vreg<64xf32> -> !pto.vmi.vreg<64xbf16>
// The bridge then selects lane 0 and stores only that BF16 value.
~~~

**Cost:** more lanes are evaluated, but that does not necessarily mean more machine instructions: the hardware instruction already operates on a vector. A timing or instruction-count comparison is needed.

Example sources: [preflight-vl1-extf-one-point-store.mlir](/home/m00967009/Workspace/AscendNPU-IR/bishengir/test/Conversion/HIVMAVEToPTOASVMI/preflight-vl1-extf-one-point-store.mlir:16) and [preflight-vl1-truncf-one-point-store.mlir](/home/m00967009/Workspace/AscendNPU-IR/bishengir/test/Conversion/HIVMAVEToPTOASVMI/preflight-vl1-truncf-one-point-store.mlir:11).

**Source evidence and limits (optional detail).**

**O — Q2 use:** cumsum.py loads and promotes to F32 at [CS:62](/home/x84315859/Q2TritonKernel/src/kernels/fla/ops/utils/cumsum.py:62) and [CS:119](/home/x84315859/Q2TritonKernel/src/kernels/fla/ops/utils/cumsum.py:119). The saved 1-1024-32-False failure reports vextf requiring ALL at decompressed lines 280–285 of Logs/collected/00051_rf_test_chunk_local_cumsum_perf.py_test_chunk_local_cumsum_perf_1-1024-32-False_c64e4b8c6d_failure.log.gz. The log does not identify the precise predicate, so it must not be relabeled VL1. Dqkwg has explicit VL1 narrowing and widening at [R:1059](/home/m00967009/Workspace/Logs/raw_bwd_dqkwg_sep29.log:1059)–1062 and [R:1095](/home/m00967009/Workspace/Logs/raw_bwd_dqkwg_sep29.log:1095)–1109, including the widening→multiply→one-point-store chain.

**O — contract and conversion:** AVE has an explicit mask and physical part selector ([A:204](/home/m00967009/Workspace/AscendNPU-IR/bishengir/include/bishengir/Dialect/HIVMAVE/IR/HIVMAVEOps.td:204), [A:255](/home/m00967009/Workspace/AscendNPU-IR/bishengir/include/bishengir/Dialect/HIVMAVE/IR/HIVMAVEOps.td:255)); native widening passes that mask to vcvt ([N:2117](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Conversion/HIVMAVEToAVEIntrin/HIVMAVEToAVEIntrin.cpp:2117)–2134). VMI extf and vcvt have no mask operand ([V:569](/home/m00967009/Workspace/PTOAS/include/PTO/IR/VMIOps.td:569), [V:1379](/home/m00967009/Workspace/PTOAS/include/PTO/IR/VMIOps.td:1379)). Bridge use-graph guards are [B:1239](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Conversion/HIVMAVEToPTOASVMI/HIVMAVEToPTOASVMI.cpp:1239)–1294; conversions are [B:3875](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Conversion/HIVMAVEToPTOASVMI/HIVMAVEToPTOASVMI.cpp:3875)–3941. VL1 widening permits only matching ONEPT_B32 stores directly or through one matching multiply; narrowing permits only matching ONEPT_B16 stores. Wider partial-mask cumsum consumers do not satisfy this guard. No general workaround was found.

**I/Q:** widening computes the active F16/BF16 value exactly in F32; the one-point store hides other lanes. Narrowing retains rounding/saturation attributes, but exceptional inactive-lane/FP-status behavior is not established. General masked results need explicit inactive-lane semantics, not merely a dropped predicate.

Tests: preflight-vl1-extf-one-point-store.mlir, preflight-reject-vl1-extf.mlir, preflight-vl1-truncf-one-point-store.mlir, preflight-reject-vl1-truncf.mlir in AscendNPU-IR/bishengir/test/Conversion/HIVMAVEToPTOASVMI/. Fresh positive RUN/FileCheck checks passed.

### F02

**Read one BF16 value and repeat it 64 times.**

**What the kernel needs:** one scalar gate value must be available to every lane of a vector calculation. If memory contains 2.5, the result should look like [2.5, 2.5, …, 2.5].

**NPU-IR before conversion — adapted excerpt:**

~~~mlir
%x = ave.hir.vload <BRC_B16> %src[%zero]
    {functionType = #ave.func_dist_type<pk>}
    : memref<1xbf16, #hivm.address_space<ub>>
      into vector<64xbf16>
~~~

**PTOAS VMI after the bridge — abbreviated index list:**

~~~mlir
// First ask for 128 copies, the native B16 register width.
%full = pto.vmi.vload %src[%zero] {dist_mode = "brc"}
    : !pto.ptr<bf16, ub> -> !pto.vmi.vreg<128xbf16>

// Then make the requested 64-lane result by repeating lane 0.
// The actual shuffle contains 64 zero indices; "..." abbreviates them here.
%x = pto.vmi.shuffle %full[[0, 0, 0, ..., 0]]
    : !pto.vmi.vreg<128xbf16> -> !pto.vmi.vreg<64xbf16>
~~~

**Why the bridge takes this detour:** the straightforward 64-lane load runs into a PTOAS layout limitation when its result is widened to F32. The 128-lane load followed by a lane-zero repeat avoids that limitation.

**Why the values match:** every source lane already contains the same number. Selecting lane 0 repeatedly gives exactly the required 64 copies.

**Cost:** the shuffle may require extra vector work. This does not read 128 separate memory locations: the broadcast load still reads one scalar address. The exact physical instruction cost is not measured.

Example source: [preflight-brc-b16-load.mlir](/home/m00967009/Workspace/AscendNPU-IR/bishengir/test/Conversion/HIVMAVEToPTOASVMI/preflight-brc-b16-load.mlir:3).

**Source evidence and limits (optional detail).**

**O:** Dqkwg BRC_B16 broadcasts occur at [R:1096](/home/m00967009/Workspace/Logs/raw_bwd_dqkwg_sep29.log:1096) and in the later vector helper; Q2 loads a single last-gate value at [DQ:282](/home/x84315859/Q2TritonKernel/src/kernels/fla/ops/common/chunk_o.py:282)–295. Bridge [B:3612](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Conversion/HIVMAVEToPTOASVMI/HIVMAVEToPTOASVMI.cpp:3612)–3672 accepts matching F16/BF16 singleton UB sources and only 1/64/128 result lanes. The 64-lane form becomes a native 128-lane broadcast load and a shuffle with 64 zero indices. Original linearized element offset is retained.

The existing tracker documents direct-64 sparse-layout materialization failure and failed first-64 B16 projection. PTOAS physical shuffle validation rejects padding in a result chunk at [P:2560](/home/m00967009/Workspace/PTOAS/lib/PTO/Transforms/VMIToVPTO.cpp:2560)–2600; this is a physicalization limitation, not proof that the logical shuffle is invalid. Unified brc becomes group_broadcast_load at [U:572](/home/m00967009/Workspace/PTOAS/lib/PTO/Transforms/VMILowerUnifiedToLegacy.cpp:572).

**I:** all loaded lanes hold the same bits, so lane-zero replication is value exact. One native register does not mean 128 memory reads: this remains a scalar-address broadcast load. Any added cost is shuffle/materialization and register dependency, not a second memory transfer. Inspect final VPTO/LLVM before counting it as an extra executed instruction.

Test: preflight-brc-b16-load.mlir, including bf16_dqkwg and offset-bearing F16 forms; fresh documented RUN/FileCheck passed.

### F03

**A one-element vector load becomes a scalar load plus a vector broadcast.**

**NPU-IR before conversion:**

~~~mlir
%x = ave.hir.vload <BRC_B32> %src[]
    : memref<f32, #hivm.address_space<ub>>
      into vector<1xf32>
~~~

**PTOAS VMI after conversion:**

~~~mlir
%s = pto.load_scalar %src[%zero] : <f32, ub> -> f32
%x = pto.vmi.broadcast %s : f32 -> !pto.vmi.vreg<1xf32>
~~~

**What changed:** NPU-IR requests a vector load. The workaround reads on the scalar pipeline and then puts the value into a vector representation.

**Why that matters:** a synchronization event that waits for vector work does not automatically wait for scalar work. The number can be right while the timing of the read is wrong unless synchronization covers the new pipeline.

For comparison, the multi-lane case stays a vector load:

~~~mlir
// NPU-IR result: vector<64xf32>
// PTOAS VMI result: 64 repeated F32 values, still a vector-pipeline load.
%x = pto.vmi.vload %src[%zero] {dist_mode = "brc"}
    : !pto.ptr<f32, ub> -> !pto.vmi.vreg<64xf32>
~~~

**Correction after reassessment (O):** the scalar workaround exists in the bridge, but a vector-pipeline alternative already works in the older PTOAS binary. Running upstream test `vmi_to_vpto_short_vector_load.pto`, function `scalar_f32`, through the old lower/layout/VPTO path produced `pto.vlds ... {dist = "BRC_B32"}` with exit 0. The original restriction concerns the old explicit `dist_mode = "brc"` form, not every one-lane vector load. This is not an upstream-only fix.

~~~mlir
// Available logical alternative; adapted from scalar_f32.
%x = pto.vmi.vload %src[%zero]
    : !pto.ptr<f32, ub> -> !pto.vmi.vreg<1xf32>
// Observed old physical output: pto.vlds ... {dist = "BRC_B32"}.
~~~

**Recommendation (I):** prefer this existing vector-load form for the singleton bridge case after testing the complete Q2 consumer and synchronization graph. The probe does not establish every B16 sparse fan-out case.

**Status:** the bridge still uses its scalar workaround. Whether each complete Q2 use has the required scalar/vector synchronization still needs validation. No race was demonstrated for every singleton use.

Upstream singleton alternative: [short-vector-load test, scalar_f32](https://github.com/hw-native-sys/PTOAS/blob/7a32cb83ef927dab4b0c4aa928093acacea24abc/test/lit/vmi_new/vmi_to_vpto_short_vector_load.pto#L49).

Example source: [preflight-brc-load.mlir](/home/m00967009/Workspace/AscendNPU-IR/bishengir/test/Conversion/HIVMAVEToPTOASVMI/preflight-brc-load.mlir:18).

**Source evidence and limits (optional detail).**

**O:** Singleton BRC_B32 appears at [R:889](/home/m00967009/Workspace/Logs/raw_bwd_dqkwg_sep29.log:889) and [R:921](/home/m00967009/Workspace/Logs/raw_bwd_dqkwg_sep29.log:921). Bridge [B:3679](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Conversion/HIVMAVEToPTOASVMI/HIVMAVEToPTOASVMI.cpp:3679) onward distinguishes singleton load_scalar+broadcast from multi-lane VMI brc. The original design records why scalarizing multi-lane loads broke vector-pipe synchronization in Planner/bridge/designs/ave-to-ptoas-vmi-conversion-design.md, Decision 16.

**I/Q:** matching values does not prove matching pipeline execution. A singleton scalar read needs synchronization appropriate to PIPE_S, while the native operation is a vector load. The exact dqkwg scalar-load ordering remains a full-kernel validation requirement. No general claim of a race in every singleton use follows from this.

Tests: preflight-brc-load.mlir, and the recovered singleton_scalar_round_trip function in preflight-singleton-scalar-round-trip.mlir. The latter explicitly separates cast conversion from scope finalization.

### F04

**A reduction returns one useful value, but NPU-IR gives it a 64-lane type.**

**What the kernel needs:** add the input values to compute a mean, variance, or gradient sum.

**NPU-IR before conversion:**

~~~mlir
%sum = ave.hir.reduction <add>, %x, %all
    : vector<64xf32>, vector<64xi1> -> vector<64xf32>
~~~

The important detail is the result type: it still says 64 values. The AVE definition says the reduced result is placed in the first position. It does not promise that all 64 positions contain the sum.

**PTOAS VMI after conversion:**

~~~mlir
// PTOAS represents the reduction result as one logical value.
%one = pto.vmi.reduce_addf %x, %all {reassoc}
    : !pto.vmi.vreg<64xf32>, !pto.vmi.mask<64xpred>
      -> !pto.vmi.vreg<1xf32>

// The bridge restores the 64-lane type expected by the surrounding graph.
%sum = pto.vmi.broadcast %one
    : !pto.vmi.vreg<1xf32> -> !pto.vmi.vreg<64xf32>
~~~

**Small illustration, using four lanes instead of 64:**

~~~text
Input:                     [1, 2, 3, 4]
NPU-IR defined result:     [10, ?, ?, ?]    (? = not promised by this contract)
PTOAS reduction result:    [10]
Bridge broadcast result:   [10, 10, 10, 10]
~~~

**Why the observed use can work:** if the next operation only stores lane 0, both paths store 10.

**Possible unnecessary work:** the bridge may repeat the sum across 64 lanes and immediately select lane 0 again. That repetition is needed for the current type mapping, but could disappear if the compiler tracked the useful one-lane result through its consumers.

**Numerical limit:** changing the order of floating-point additions can change the last few bits. The reassoc attribute allows such reordering; it does not prove bit-for-bit equivalence to native NPU-IR.

**Source evidence and limits (optional detail).**

**O:** Q2 layernorm computes means/variances at [LN:72](/home/x84315859/Q2TritonKernel/src/kernels/fla/modules/fused_norm_gate.py:72)–83 and backward row sums at [LN:292](/home/x84315859/Q2TritonKernel/src/kernels/fla/modules/fused_norm_gate.py:292)–296; dqkwg reduces products at [DQ:251](/home/x84315859/Q2TritonKernel/src/kernels/fla/ops/common/chunk_o.py:251), [DQ:296](/home/x84315859/Q2TritonKernel/src/kernels/fla/ops/common/chunk_o.py:296), [DQ:308](/home/x84315859/Q2TritonKernel/src/kernels/fla/ops/common/chunk_o.py:308). AVE dump: [R:1416](/home/m00967009/Workspace/Logs/raw_bwd_dqkwg_sep29.log:1416)–1425. The AVE definition requires equal source/result shapes but says the result is at the first position ([A:627](/home/m00967009/Workspace/AscendNPU-IR/bishengir/include/bishengir/Dialect/HIVMAVE/IR/HIVMAVEOps.td:627)–650). It does **not** promise a full-width splat.

Bridge [B:3450](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Conversion/HIVMAVEToPTOASVMI/HIVMAVEToPTOASVMI.cpp:3450)–3511 emits one-lane reduce_addf/reduce_maxf and broadcasts when needed for the source type. PTOAS [I:1793](/home/m00967009/Workspace/PTOAS/lib/PTO/IR/VMI.cpp:1793) onward requires reassoc and a one-lane F16/F32 floating result; physical ADD reduction is [P:9732](/home/m00967009/Workspace/PTOAS/lib/PTO/Transforms/VMIToVPTO.cpp:9732)–9816.

**I/Q:** broadcasting reconstructs the carrier width and preserves the defined reduced lane; it is not evidence that native reduction defined every other lane identically. For observed store/projection paths this is sufficient. Addition order, NaNs, signed zero, and cross-register combine behavior require differential tests; reassoc is permission, not bitwise parity.

Recommended: keep reduced values singleton until an actual broadcast consumer requires expansion. Compare vcadd/combine/broadcast counts with the native LLVM.

### F05

**A stream of one-value stores loses its stream state.**

**What the kernel needs:** store one reduction result per row while iterating through rows. Native NPU-IR can treat these writes as one continuous stream.

**NPU-IR before conversion — adapted dqkwg store inside its row loop:**

~~~mlir
ave.hir.masked_store <ONEPT_B32> %row_dst[], %first, %value
    {hivm.is_continuous}
    : memref<f32, strided<[], offset: ?>, #hivm.address_space<ub>>,
      vector<1xi1>, vector<1xf32>
~~~

**Native NPU-IR instruction lowering — pseudocode, not literal MLIR:**

~~~text
state = init_align()
base = output_address

for each row:
    state, base = vstus.post(row_result, base, state)

vstas(state, base)   // Finish the stream after the loop.
~~~

**Current bridge/PTOAS path:**

~~~mlir
// VMI inside the row loop: an independent one-value store.
pto.vmi.vstore %value, %row_dst[%zero]
    : !pto.vmi.vreg<1xf32>, !pto.ptr<f32, ub>
~~~

The later physical operation is a one-point store, schematically:

~~~text
for each row:
    pto.vsts(row_result, row_address, dist = "1PT_B32")
~~~

**What is missing:** there is no state passed from one store to the next and no equivalent end-of-stream operation in the VMI representation.

**Why a barrier does not fix this:** “wait until writes are visible” and “finish this buffered store stream” are different actions. A memory barrier does the former; vstas belongs to the latter.

**Upstream update (O):** [stateful stream fusion](https://github.com/hw-native-sys/PTOAS/blob/7a32cb83ef927dab4b0c4aa928093acacea24abc/lib/PTO/Transforms/VPTO/VPTOStatefulStreamFusion.cpp#L689) can carry physical alignment/base state through eligible constant-trip loops. It is enabled after VMI-to-VPTO in [the upstream pipeline](https://github.com/hw-native-sys/PTOAS/blob/7a32cb83ef927dab4b0c4aa928093acacea24abc/tools/ptoas/ptoas_pipeline.cpp#L996). Its tests cover 64-F32 contiguous chunks; they do not prove the bridge’s singleton ONEPT_B32 stores become eligible streams. See U03 before treating F05 as resolved.

**Status:** this remains a confirmed bridge/native instruction-sequence difference; complete upstream closure is unresolved. Its complete numerical and performance effect in dqkwg has not been established. A state-carrying VMI store would let PTOAS reproduce the native sequence.

**Source evidence and limits (optional detail).**

**O:** The continuous-store gap is in Q2 dqkwg itself, not only the Planner FlashAttention fixture: [R:1419](/home/m00967009/Workspace/Logs/raw_bwd_dqkwg_sep29.log:1419) and [R:1425](/home/m00967009/Workspace/Logs/raw_bwd_dqkwg_sep29.log:1425) carry both hivm.is_continuous and unaligned UB attributes. Native [N:1119](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Conversion/HIVMAVEToAVEIntrin/HIVMAVEToAVEIntrin.cpp:1119)–1139 creates init alignment, vstus.post, and vstas, with continuous markers; native hoisting runs at [PIPE:315](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Tools/bishengir-compile/regbase/PassPipeline.cpp:315)–316. The detailed existing design is Planner/bridge/designs/continuous-one-point-store-compatibility.md.

Bridge [B:4332](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Conversion/HIVMAVEToPTOASVMI/HIVMAVEToPTOASVMI.cpp:4332)–4368 accepts the marker but emits a projected singleton vstore. VMI stores carry no updated alignment/base state; low-level VPTO already has init_align/vstus/vstas. A VST_VLD barrier orders accesses but does not finish an unaligned stream.

**Q:** exact addresses, flush behavior, and numerical effects for the full dqkwg stream have not been executed in this assessment. Do not upgrade historical FlashAttention finite-input simulator success to proof of stream equivalence. A logical stateful VMI sequence is cleaner than a bridge-only template or physical instruction injection at the AVE boundary.

### F06

**Corrected finding: the submodule loses arithmetic masks; upstream preserves them.**

**Status (O):** upstream commit [2503c259](https://github.com/hw-native-sys/PTOAS/commit/2503c259e0c40da604eeff8ba5e952ebdb858bfe) fixes this lowering defect. The code snippets below distinguish the old submodule path from the new upstream path. This corrects the earlier report’s blanket statement that PTOAS drops arithmetic masks.

**NPU-IR before conversion — representative 32-active-lane form:**

~~~mlir
%first32 = ave.hir.pge <H>
    {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
%result = ave.hir.vadd %a, %b, %first32
    : vector<64xf32>, vector<64xi1>
~~~

**Bridge output: the mask is still present.**

~~~mlir
%first32 = pto.vmi.pge "PAT_VL32" : !pto.vmi.mask<64xpred>
%result = pto.vmi.vadd %a, %b, %first32
    : !pto.vmi.vreg<64xf32>, !pto.vmi.vreg<64xf32>,
      !pto.vmi.mask<64xpred> -> !pto.vmi.vreg<64xf32>
~~~

**Pinned-submodule lowering: the older internal add has no mask.**

~~~mlir
%result = pto.vmi.addf %a, %b
    : !pto.vmi.vreg<64xf32>, !pto.vmi.vreg<64xf32>
      -> !pto.vmi.vreg<64xf32>
~~~

The final snippet shows the legacy operation form; the key observation is the missing %first32 operand.

**Upstream master lowering — schematic VPTO, based on the regression checks:**

~~~text
p = physical_predicate_for(first32)
result = pto.vadd(a, b, p)  // The supplied predicate reaches the instruction.
~~~

The unified arithmetic operation now bypasses the maskless legacy add. [The lowering helper](https://github.com/hw-native-sys/PTOAS/blob/7a32cb83ef927dab4b0c4aa928093acacea24abc/lib/PTO/Transforms/VMIToVPTO/VMIToVPTOUnifiedPatternInternals.cpp#L57) resolves the mask per physical part and passes it into the emitted operation. [The regression](https://github.com/hw-native-sys/PTOAS/blob/7a32cb83ef927dab4b0c4aa928093acacea24abc/test/lit/vmi_new/vmi_to_vpto_unified_masked_compute.pto#L66) checks an all-off add and a partial-mask shift; it explicitly checks that the arithmetic uses the generated predicate. Upstream accepts zero predication mode; this is not general merge-mode support. F01 conversion operations still lack mask operands and are a separate gap.

**When the old submodule behavior is harmless:** if the result is written with the same first-32 mask, the other results never reach memory.

**When it can matter:** if a later reduction, shuffle, or broader store observes the other lanes, doing arithmetic there can change the result. The existing conversion does not prove those lanes are always unused.

**Status:** mask loss is observed in the pinned source and fixed in upstream source/regressions. An old-binary run of the upstream all-off regression parsed and lowered successfully but failed its FileCheck because the required all-off predicate was absent. A wrong result for every Q2 tail is not established, and no latest-master Q2 runtime result is claimed. Integrating the upstream fix must include its matching mask/layout changes, rather than merely deleting the old helper.

**Source evidence and limits (optional detail).**

**O:** Dqkwg uses H for 32 active F32 lanes at [R:1427](/home/m00967009/Workspace/Logs/raw_bwd_dqkwg_sep29.log:1427)–1434 and VL1 multiplication at [R:1107](/home/m00967009/Workspace/Logs/raw_bwd_dqkwg_sep29.log:1107)–1109. Current bridge supports H: [B:628](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Conversion/HIVMAVEToPTOASVMI/HIVMAVEToPTOASVMI.cpp:628)–639 maps half the logical width to PAT_VL<n>; [B:1193](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Conversion/HIVMAVEToPTOASVMI/HIVMAVEToPTOASVMI.cpp:1193) accepts direct ALL/VL*/H. Native utility computes H from predicate granularity ([UTIL:462](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Dialect/HIVMAVE/Utils/Utils.cpp:462)–468), and the attribute describes the lowest half ([ATTR:175](/home/m00967009/Workspace/AscendNPU-IR/bishengir/include/bishengir/Dialect/HIVMAVE/IR/HIVMAVEAttrs.td:175)). The observed 64/PB32 and 128/PB16 forms agree; arbitrary differently sized H carriers need a separate check.

Bridge arithmetic preserves the mask ([B:3768](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Conversion/HIVMAVEToPTOASVMI/HIVMAVEToPTOASVMI.cpp:3768)–3787). The pinned PTOAS implementation explicitly discards it in lowerBinaryIgnoringMask and lowerMaskedUnary ([U:233](/home/m00967009/Workspace/PTOAS/lib/PTO/Transforms/VMILowerUnifiedToLegacy.cpp:233)–276), including vadd dispatch at [U:1435](/home/m00967009/Workspace/PTOAS/lib/PTO/Transforms/VMILowerUnifiedToLegacy.cpp:1435). In contrast, the VMI ISA documents zero/merge inactive-lane semantics at PTOAS/docs/isa/vmi-isa/03-eltwise-compute.md:19–48.

**I:** same-mask final stores can hide inactive arithmetic, including the guarded F01 path. Reductions, shuffles, broader-mask consumers, or cross-lane operations can expose it. The ordinary arithmetic conversion does not establish general non-observability. Repair should preserve masks or reject unproven graphs. Merge mode is rejected by these PTOAS helpers; it is not silently implemented by them.

### F07

**A value goes from vector to scalar and straight back to vector.**

**NPU-IR before conversion — shortened version of the recovered wrapper chain:**

~~~mlir
// %loaded contains one F32 value in a one-element vector.
%scalar = builtin.unrealized_conversion_cast %loaded
    : vector<1xf32> to f32
%one = builtin.unrealized_conversion_cast %scalar
    : f32 to vector<1xf32>
%full = builtin.unrealized_conversion_cast %one
    : vector<1xf32> to vector<64xf32>
~~~

**PTOAS VMI after the checked simplification:**

~~~mlir
// Keep the original vector value. No real scalar extraction is needed.
%full = pto.vmi.broadcast %loaded
    : !pto.vmi.vreg<1xf32> -> !pto.vmi.vreg<64xf32>
~~~

**What changed:** two changes of representation disappeared. A scalar operation never used the value between those casts, so there is no reason to move the value into a scalar register.

**Why this is a workaround:** PTOAS VMI has no general “extract this lane into a scalar” operation. Removing a round trip avoids needing one.

**Limit:** this does not work if %scalar is actually used by scalar arithmetic, a call, or a return. Those uses still need a real extraction contract.

**Cost:** this simplification avoids introducing a memory store/reload just to get a scalar. It does not by itself fix the separate vector-scope failures.

Example source: [preflight-singleton-scalar-round-trip.mlir](/home/m00967009/Workspace/AscendNPU-IR/bishengir/test/Conversion/HIVMAVEToPTOASVMI/preflight-singleton-scalar-round-trip.mlir:52).

**Source evidence and limits (optional detail).**

**O:** [R:889](/home/m00967009/Workspace/Logs/raw_bwd_dqkwg_sep29.log:889)–934 contains singleton-vector/scalar/wider-vector wrappers. Bridge preprocessing and scalar-like conversion retain VMI rather than extracting an actual scalar; [B:3167](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Conversion/HIVMAVEToPTOASVMI/HIVMAVEToPTOASVMI.cpp:3167)–3245 and the dedicated tests constrain escaping consumers. preflight-singleton-scalar-round-trip.mlir:19–36 reproduces the ordering; its first RUN checks conversion only, while independent fanouts use explicit finalization. preflight-reject-singleton-scalar-extraction.mlir covers genuine scalar consumers.

**I:** eliminating wrappers is correct for a graph that immediately returns to a same-type vector/broadcast. It avoids a needless UB spill and scalar reload. It cannot implement scalar arithmetic/calls/returns merely by changing types. The tracker’s fanout LLVM success is recorded validation of those fanouts, not complete dqkwg support.

### F08

**Split 128 half-precision values into two groups and widen them.**

**What the layernorm kernel needs:** turn 128 F16 values into 128 F32 values, used as two groups of 64.

**NPU-IR before conversion — adapted from the sparse-interleave test:**

~~~mlir
// This is a physical arrangement step, not just an ordinary logical zip.
%lo, %hi = ave.hir.vintlv %packed, %packed
    {layout_change = #ave<layout_change SPARSE>}
    : vector<128xf16>, vector<64xf16>

%a = ave.hir.vextf %lo, <part_even>, %all
    : vector<64xf16>, vector<64xf32>, vector<64xi1>
%b = ave.hir.vextf %hi, <part_even>, %all
    : vector<64xf16>, vector<64xf32>, vector<64xi1>
~~~

**PTOAS VMI after the bridge — index ranges abbreviated:**

~~~mlir
// Widen all 128 logical values first.
%wide = pto.vmi.extf %packed
    : !pto.vmi.vreg<128xf16> -> !pto.vmi.vreg<128xf32>

// Then take complete 64-lane F32 halves.
%a = pto.vmi.shuffle %wide[[0, 1, ..., 63]]
    : !pto.vmi.vreg<128xf32> -> !pto.vmi.vreg<64xf32>
%b = pto.vmi.shuffle %wide[[64, 65, ..., 127]]
    : !pto.vmi.vreg<128xf32> -> !pto.vmi.vreg<64xf32>
~~~

**Why the order changes:** PTOAS has trouble producing a half-full physical B16 register with a general shuffle. A 64-lane F32 result fills a physical register, so splitting after widening avoids that limitation.

**Small illustration — logical values, not physical register contents:**

~~~text
Input F16 values:      [a, b, c, d]
Required F32 groups:   [float(a), float(b)] and [float(c), float(d)]

Bridge:
  widen first:        [float(a), float(b), float(c), float(d)]
  split second:       [float(a), float(b)] and [float(c), float(d)]
~~~

**Status:** the checked split/widen chain compiles through PTOAS to LLVM. Arbitrary sparse-vector users are still rejected. The two logical shuffles may become register forwarding rather than executed shuffle instructions.

Example source: [preflight-sparse-interleave-extf.mlir](/home/m00967009/Workspace/AscendNPU-IR/bishengir/test/Conversion/HIVMAVEToPTOASVMI/preflight-sparse-interleave-extf.mlir:9).

**Source evidence and limits (optional detail).**

**O:** The two backward layernorm shape groups historically failed vintlv; newer validation identifies the actual packed sparse split. Q2 backward promotions are [LN:253](/home/x84315859/Q2TritonKernel/src/kernels/fla/modules/fused_norm_gate.py:253)–264 and [LN:380](/home/x84315859/Q2TritonKernel/src/kernels/fla/modules/fused_norm_gate.py:380)–389. AveLoopOptimize creates the self-interleave at [LOOP:371](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Dialect/HIVMAVE/Transforms/AveLoopOptimize.cpp:371)–389. The bridge checks 128 B16 input, two 64-lane halves, self operands, and exclusively ALL/part_even F32-widening consumers ([B:3945](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Conversion/HIVMAVEToPTOASVMI/HIVMAVEToPTOASVMI.cpp:3945)–4004).

It emits one logical 128-lane extf, then shuffles lanes 0–63 and 64–127 as full B32 chunks. A direct logical interleave of the sparse carriers would change the lane interpretation. General full-register interleave is separately mapped at [B:4008](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Conversion/HIVMAVEToPTOASVMI/HIVMAVEToPTOASVMI.cpp:4008)–4028.

**O/I:** PTOAS rejects ordinary partial B16 shuffle chunks ([P:2596](/home/m00967009/Workspace/PTOAS/lib/PTO/Transforms/VMIToVPTO.cpp:2596)–2600), whereas the B32 halves can forward complete physical chunks. The fresh saved split_widen_f16/bf16 probe compiled to LLVM. This avoids the native representation’s interleave; it does not imply that one logical extf becomes one physical conversion. Escaping sparse halves remain unsupported.

Tests: preflight-sparse-interleave-extf.mlir and preflight-reject-interleave.mlir. Saved full-kernel validation is [LP:24](/home/m00967009/Workspace/Logs/layernorm_compile_oct15/README.md:24)–69.

### F09

**A simple zero vector compiles, until PTOAS shares its hidden mask across a loop boundary.**

**NPU-IR idea — representative broadcast, not an extracted exact D1200 line:**

~~~mlir
%zero_scalar = arith.constant 0.0 : f32
%all = ave.hir.pge <ALL> : vector<64xi1>
%zeros = ave.hir.broadcast %zero_scalar, %all
    : f32, vector<64xi1> -> vector<64xf32>
~~~

**VMI from the saved failing D1200 artifact:**

~~~mlir
%zeros = "pto.vmi.constant"()
    <{value = dense<0.0> : tensor<64xf32>}>
    : () -> !pto.vmi.vreg<64xf32>
~~~

**PTOAS physical lowering introduces these operations:**

~~~mlir
%scalar = arith.constant 0.0 : f32
%mask = pto.pset_b32 "PAT_ALL" : !pto.mask<b32>
%zeros = pto.vdup %scalar, %mask
    : f32, !pto.mask<b32> -> !pto.vreg<64xf32>
~~~

**Where compilation gets stuck — pseudocode showing the relationship:**

~~~text
mask = all_active_mask()
store(zeros, output)

for each chunk:
    barrier()             // Splits vector scopes.
    use_shared_mask(mask) // PTOAS cannot place a legal copy here.
~~~

**What this means:** there is nothing wrong with the number zero. PTOAS introduced a mask, reused it in a later nested loop, and then could not place that mask inside the required vector scopes. A vector scope is a region where vector-register values are allowed to live.

**Status:** both the minimal reproducer and the complete D1200 VMI fail. Moving the zero constant into its fill loop made the saved manual experiment compile, but that workaround is not implemented and is not a general fix.

**Cost of that possible workaround:** generating zero inside a loop can repeat work. Fixing mask placement in PTOAS is preferable to repeatedly moving constants in the bridge.

**Source evidence and limits (optional detail).**

**O:** Exact affected case: layer_norm_gated_bwd[2-4096-1200-silu], layer_norm_gated_bwd_kernel1, all tested warp choices 2/4/8/16. VMI succeeds; PTOAS scope inference fails. The saved original is Logs/layernorm_compile_oct15/artifacts/bwd-T8192-D1200-silu-BT1-W4-specialized.vmi.mlir:171 and :706.

A VMI splat constant becomes scalar constant + pset_b32 ALL + vdup ([P:5541](/home/m00967009/Workspace/PTOAS/lib/PTO/Transforms/VMIToVPTO.cpp:5541)–5591). CSE shares the introduced predicate with a later nested loop containing a boundary. computeLogicalScopeAnchors skips boundaries and rematerialization cannot find an ancestor-loop anchor ([S:439](/home/m00967009/Workspace/PTOAS/lib/PTO/Transforms/PTOInferVPTOVecScope.cpp:439)–491). See [LR:79](/home/m00967009/Workspace/AscendNPU-IR-layernorm/docs/layernorm-bridge-compilation.md:79) onward for the source-to-LLVM explanation.

**Fresh result:** both the minimal saved reproducer and the complete original VMI fail with “pto.pset_b32 … VPTO vector-scope data cannot have external users … !pto.mask<b32>”. The already saved manual constant-in-fill-loop variant compiles to LLVM. No workaround was implemented here, and existing documentation explicitly says it was deferred.

**I:** rematerializing safe immutable dependencies in each actual nested consumer segment is the general repair. Reloading or cloning arbitrary non-pure values is unsafe. The narrow experiment may repeat materialization in 32 fill iterations; no runtime cost was measured.

### F10

**F16 packed stores are recognized; the matching BF16 pattern is still rejected.**

**NPU-IR before conversion — the BF16 pattern actually present in dqkwg:**

~~~mlir
%narrow = ave.hir.vtruncf %x, <rint>, false, <part_even>, %all
    : vector<64xf32>, vector<64xbf16>, vector<64xi1>

// Physical packing wrappers around the narrowed values:
%expanded = builtin.unrealized_conversion_cast %narrow
    : vector<64xbf16> to vector<128xbf16>
%carrier = llvm.bitcast %expanded
    : vector<128xbf16> to vector<64xf32>
ave.hir.masked_store <PK_B32> %dst[%zero], %all, %carrier
    {functionType = #ave.func_dist_type<pk>, hivm.is_continuous}
    : memref<64xbf16, #hivm.address_space<ub>>,
      vector<64xi1>, vector<64xf32>
~~~

**What the bridge already does for the F16 version:**

~~~mlir
%narrow = pto.vmi.vcvt %x {rounding = "R", saturate = "NOSAT"}
    : !pto.vmi.vreg<64xf32> -> !pto.vmi.vreg<64xf16>
pto.vmi.store %narrow, %dst[%zero]
    : !pto.vmi.vreg<64xf16>, !pto.ptr<f16, ub>
~~~

**What happens for BF16 today:** the assessed source does not accept the analogous wrappers. The temporary BF16 probe fails at the vector<64xbf16> to vector<128xbf16> cast. There is no successful current BF16 PTOAS output to show for this exact path.

**Why the F32 carrier is confusing:** the F32 type here carries packed bits; it is not asking to convert BF16 numbers back to F32. A correct mapping should store the narrowed BF16 values directly after proving the wrappers only express packing.

**Proposed shape, not current emitted support:**

~~~mlir
// Proposed analogous BF16 result after checked wrapper removal:
pto.vmi.store %narrow_bf16, %dst[%zero]
    : !pto.vmi.vreg<64xbf16>, !pto.ptr<bf16, ub>
~~~

Example source for the working F16 analog: [preflight-packed-store.mlir](/home/m00967009/Workspace/AscendNPU-IR/bishengir/test/Conversion/HIVMAVEToPTOASVMI/preflight-packed-store.mlir:12).

**Source evidence and limits (optional detail).**

**O:** The full dqkwg dump has BF16 part_even truncation → vector<64xbf16>-to-vector<128xbf16> wrapper → F32 bitcast → PK_B32 store ([R:1347](/home/m00967009/Workspace/Logs/raw_bwd_dqkwg_sep29.log:1347)–1352, also :1363). These are physical sparse-packing carriers, not a request to numerically convert BF16 to F32.

The assessed bridge recognizer matchPackedF16StoreSource requires F16 ([B:1305](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Conversion/HIVMAVEToPTOASVMI/HIVMAVEToPTOASVMI.cpp:1305) onward), and PK_B32 handling requires a UB F16 base ([B:4277](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Conversion/HIVMAVEToPTOASVMI/HIVMAVEToPTOASVMI.cpp:4277)–4323). Its F16 regression is preflight-packed-store.mlir. The analogous BF16 path has no accepted mapping in this checkout.

**Fresh result:** a temporary BF16 substitution of that regression fails legalization at the widening unrealized_conversion_cast; no source file was edited. The failure agrees with the F16-only source gates. The tracker mentions a separately scoped packed-BF16 fix in a temporary combined compiler; that is not present support in the assessed main source.

**I:** fold only a proven physical packing wrapper and store the dense BF16 logical result. Otherwise a syntactically similar interleave/bitcast may represent real lane manipulation. No PTOAS BF16-store failure can be inferred from this bridge-level rejection.

### F11

**A memory view knows its length and stride; a pointer knows only an address.**

**NPU-IR example — adapted from the strided-view regression:**

~~~mlir
%view = memref.reinterpret_cast %src
    to offset: [%offset], sizes: [64], strides: [32]
    : memref<?xbf16, #hivm.address_space<gm>>
      to memref<64xbf16, strided<[32], offset: ?>,
                #hivm.address_space<gm>>
~~~

This view means: 64 logical elements; adjacent logical elements are 32 BF16 elements apart.

**Address-only part of the PTOAS representation:**

~~~mlir
%base = pto.addptr %src, %offset
    : !pto.ptr<bf16, gm> -> !pto.ptr<bf16, gm>
~~~

**Information comparison:**

~~~text
NPU-IR view:   address + offset + length 64 + stride 32
PTOAS pointer: address after adding offset

To read logical element i:
  needed address = base + i * 32
  wrong shortcut = base + i
~~~

**Why this can fail:** if the bridge discards the stride before converting the access, a later load cannot recover the correct address. Some cases are therefore rejected. Others are converted early to loops that explicitly multiply by the stride; F13 shows those loops.

**Another assumption:** a pointer argument must already point at the expected logical base. If it instead points at the original allocation before a descriptor offset, using offset zero can access the wrong location.

**Status:** this is an address/ABI problem around the vector functions. It does not mean PTOAS lacks BF16 addition or multiplication.

**Source evidence and limits (optional detail).**

**O:** Q2 causal convolution constructs strided block pointers and explicit initial-state addresses at [CV:96](/home/x84315859/Q2TritonKernel/src/kernels/fla/modules/causal_conv1d_bwd.py:96)–121. Saved first failure is memref.reinterpret_cast at decompressed lines 319–324 of Logs/collected/00001_bwd_perf.py_test_causal_conv1d_bwd_with_initial_state_perf_B2-T65535-D128-silu_ba18728572_failure.log.gz. This is a structural/ABI failure, not a missing AVE arithmetic instruction.

Bridge [B:2690](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Conversion/HIVMAVEToPTOASVMI/HIVMAVEToPTOASVMI.cpp:2690)–2745 reconstructs static metadata and supplies zero offset/unused dynamic sizes under its restricted pointer contract. [B:2818](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Conversion/HIVMAVEToPTOASVMI/HIVMAVEToPTOASVMI.cpp:2818)–2908 allows static-stride/rank-zero/singleton and explicit-address exceptions, and emits element-based addptr offsets. Used dynamic sizes and descriptor calls that cannot be reconstructed are rejected. Byte views require byte offset scaling before a typed pointer cast, separately handled at [B:2749](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Conversion/HIVMAVEToPTOASVMI/HIVMAVEToPTOASVMI.cpp:2749).

**I/Q:** the pointer argument must already denote the normalized logical base. A raw descriptor’s allocated/aligned base and offset cannot be recovered from a single pointer without an ABI convention. Exact causal-convolution failure operands are absent from reduced logs, so current support of those ten original shapes remains unresolved.

External CCE-call mode preserves memrefs via bridge-private entry casts rather than establishing a public PTOAS descriptor ABI. It is an alternate compatibility route, not evidence that pointer erasure supports arbitrary descriptors. Tests include preflight-reject-unsupported-memref-metadata.mlir, preflight-reject-strided-gm-descriptor-call.mlir, and preflight-reject-noncontiguous-reinterpret-load.mlir.

### F12

**Read a BF16 value without using the ordinary cached scalar-load path.**

**NPU-IR before conversion — adapted from the recovered load:**

~~~mlir
%value = memref.load %src[%i] {disableDCache = 0 : i32}
    : memref<1xbf16, strided<[1], offset: ?>,
              #hivm.address_space<gm>>
~~~

The presence of disableDCache requests the special load behavior; the zero payload does not turn it off.

**PTOAS after conversion:**

~~~mlir
// Change the pointer's element type; keep the same address.
%bits_ptr = pto.castptr %src
    : !pto.ptr<bf16, gm> -> !pto.ptr<i16, gm>

// Read exactly the same 16 bits with the non-cached device operation.
%bits = pto.ld_dev %bits_ptr[%i] : !pto.ptr<i16, gm> -> i16

// Interpret those bits as BF16 again.
%value = arith.bitcast %bits : i16 to bf16
~~~

**Why integers appear:** PTOAS ld_dev accepts integer payloads. The bridge therefore transports the floating-point bits in an equal-sized integer.

**What does not happen:** the BF16 number is not rounded or numerically converted into an integer. For example, the bits for 1.0 stay the bits for 1.0.

**Status and cost:** this workaround preserves the bits and cache-bypass request in the checked form. It requires the supported A5/backend path. The type changes may compile away; the device load's cache behavior is the meaningful performance difference.

Example source: [preflight-gm-scalar-load.mlir](/home/m00967009/Workspace/AscendNPU-IR/bishengir/test/Conversion/HIVMAVEToPTOASVMI/preflight-gm-scalar-load.mlir:14).

**Source evidence and limits (optional detail).**

**O:** Q2 loads b_g_last at [DQ:284](/home/x84315859/Q2TritonKernel/src/kernels/fla/ops/common/chunk_o.py:284); the generated singleton BF16 GM load carries disableDCache = 0 at [R:1708](/home/m00967009/Workspace/Logs/raw_bwd_dqkwg_sep29.log:1708). Presence, including zero payload, triggers the bridge’s non-cached path ([B:4506](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Conversion/HIVMAVEToPTOASVMI/HIVMAVEToPTOASVMI.cpp:4506)–4564). It casts the floating pointer to equal-width integer pointer, issues ld_dev at the original element index, and bitcasts bits back.

PTOAS official CANN 9.0.0 lowering accepts an integer result and forms the appropriate element-scaled GEP and LD.DEV.u16/u32.GM intrinsic ([C:10918](/home/m00967009/Workspace/PTOAS/lib/PTO/Transforms/VPTOCANN900LLVMEmitter.cpp:10918)–10955). The tracker records A5/ordinary-entry restrictions and explicit rejection on beta.1. Unmarked floating loads use load_scalar; marked helper loads/nontemporal forms remain restricted.

**I/Q:** equal-width bitcasts preserve NaN payloads, signed zero, and address scaling. They need not add numeric conversion instructions. Actual cache-bypass latency and end-to-end read ordering need a hardware comparison; reverting to a cached load is not a valid performance workaround. Test: preflight-gm-scalar-load.mlir and its rejection companion.

### F13

**A strided transfer becomes a loop that copies one value at a time.**

**NPU-IR before conversion — shortened form with view types explained by comments:**

~~~mlir
// %gm_view: n BF16 elements, stride 32 in GM.
// %ub_view: n BF16 elements, stride 1 in UB.
hivm.hir.load
    ins(%gm_view : memref<?xbf16, strided<[32], offset: ?>,
                          #hivm.address_space<gm>>)
    outs(%ub_view : memref<?xbf16, strided<[1], offset: ?>,
                           #hivm.address_space<ub>>)
~~~

**PTOAS gather fallback — adapted excerpt; base offsets were applied earlier:**

~~~mlir
pto.set_flag[<PIPE_MTE2>, <PIPE_S>, <EVENT_ID7>]
pto.wait_flag[<PIPE_MTE2>, <PIPE_S>, <EVENT_ID7>]
scf.for %i = %zero to %n step %one {
  %gm_index = arith.muli %i, %stride32 : index
  %value = pto.load_scalar %gm_base[%gm_index] : <bf16, gm> -> bf16
  pto.store_scalar %value, %ub_base[%i] : <bf16, ub>, bf16
}
pto.set_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID7>]
pto.wait_flag[<PIPE_S>, <PIPE_MTE2>, <EVENT_ID7>]
~~~

**The reverse transfer uses a similar loop, with a device store:**

~~~mlir
// Synchronization before and after the loop is omitted here.
scf.for %i = %zero to %n step %one {
  %value = pto.load_scalar %ub_base[%i] : <bf16, ub> -> bf16
  %bits = arith.bitcast %value : bf16 to i16
  %gm_index = arith.muli %i, %stride32 : index
  pto.st_dev %bits, %gm_bits_base[%gm_index] : !pto.ptr<i16, gm>, i16
}
~~~

**Small address example:**

~~~text
GM elements read:  gm[0], gm[32], gm[64], gm[96]
UB elements written: ub[0], ub[1], ub[2], ub[3]
~~~

**Why the synchronization is present:** MTE2/MTE3 are transfer pipelines and S is the scalar pipeline. The flags make the pipelines wait for the work that must finish first.

**Performance concern:** copying n values now visibly involves n scalar reads/writes plus loop/address work. However, native CCE can choose the same fallback for this layout. Compare against that native branch before calling it a regression.

Example sources: [preflight-strided-gm-gather.mlir](/home/m00967009/Workspace/AscendNPU-IR/bishengir/test/Conversion/HIVMAVEToPTOASVMI/preflight-strided-gm-gather.mlir:19) and [preflight-strided-gm-store.mlir](/home/m00967009/Workspace/AscendNPU-IR/bishengir/test/Conversion/HIVMAVEToPTOASVMI/preflight-strided-gm-store.mlir:63).

**Source evidence and limits (optional detail).**

**O:** Q2 g and dg have stride HV rather than unit stride ([DQ:282](/home/x84315859/Q2TritonKernel/src/kernels/fla/ops/common/chunk_o.py:282) and [DQ:310](/home/x84315859/Q2TritonKernel/src/kernels/fla/ops/common/chunk_o.py:310)–316); cumsum also constructs head-interleaved scalar block pointers ([CS:59](/home/x84315859/Q2TritonKernel/src/kernels/fla/ops/utils/cumsum.py:59)–69). The bridge captures runtime count and positive static stride before erasing descriptors.

Gather: [T:379](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Conversion/HIVMTemplatesToPTO/HIVMTemplatesToPTO.cpp:379)–467 emits an MTE2→S event-7 handshake, scalar GM load/UB store loop, and S→MTE2 handshake, rejecting incompatible padding/cache policy. Scatter: [T:759](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Conversion/HIVMTemplatesToPTO/HIVMTemplatesToPTO.cpp:759)–821 emits MTE3→S, scalar UB load, floating-bit-to-integer cast, st_dev at i*stride, and S→MTE3. Source and destination base offsets remain independent. Atomic or unsupported layouts are not generalized.

**O/I:** integer st_dev and element GEP lowering are [C:10982](/home/m00967009/Workspace/PTOAS/lib/PTO/Transforms/VPTOCANN900LLVMEmitter.cpp:10982)–11018. This preserves the native fallback’s non-cached writes, avoiding assumptions about independent cores writing separate elements of one cached line. Cost is O(length) scalar memory operations, multiplies/addressing, loop overhead, and pipe handshakes. Because the CCE path can use the same scalar fallback, this is not automatically a bridge regression versus native.

Tests: preflight-strided-gm-gather.mlir, preflight-strided-gm-store.mlir. Compare the exact native branch, tails, and event schedule before proposing DMA.

### F14

**One Cube synchronization request addresses two Vector semaphore banks.**

**NPU-IR before conversion — example from the sync regression:**

~~~mlir
hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 2
hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
~~~

**Current PTOAS representation:**

~~~mlir
pto.set_intra_block <PIPE_FIX>, 2
pto.set_intra_block <PIPE_FIX>, 18
pto.wait_intra_block <PIPE_MTE1>, 2
pto.wait_intra_block <PIPE_MTE1>, 18
~~~

**Why 18 appears:** 18 = 2 + 16. The A5 Cube-side mapping explicitly covers two Vector semaphore banks.

For a Vector-side request the bridge emits only the corresponding single bank operation; the source core kind therefore matters.

**Why this entry updates the old tracker:** the older description said the bridge lost the core kind and synchronization mode. The assessed implementation now has this specific intra-block mapping. That old blanket description is no longer accurate.

**Remaining limit:** passing this conversion test does not demonstrate that a complete kernel's handshakes are correctly paired at runtime. Other synchronization modes have different contracts and need their own checks.

Example source: [preflight-sync-block.mlir](/home/m00967009/Workspace/AscendNPU-IR/bishengir/test/Conversion/HIVMAVEToPTOASVMI/preflight-sync-block.mlir:32).

**Source evidence and limits (optional detail).**

**O:** Dqkwg’s saved mixed module contains Cube/Vector sync pairs, for example the source beginning near [R:568](/home/m00967009/Workspace/Logs/raw_bwd_dqkwg_sep29.log:568) and the device portions in the full dump. Current conversion is [B:2535](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Conversion/HIVMAVEToPTOASVMI/HIVMAVEToPTOASVMI.cpp:2535)–2671. Unlike the old tracker row, A5 intra-block sync uses set_intra_block/wait_intra_block, validates logical static IDs 0–15, retains exact core kind, and emits a second Cube operation at ID+16. Non-intra-block sync.set and sync.wait both receive mode. The set rejects an FFTS base for the intra-block form.

PTOAS sync.set/wait restrict A5 FFTS IDs to 0–15 ([SY:4036](/home/m00967009/Workspace/PTOAS/lib/PTO/IR/PTO.cpp:4036), [SY:4193](/home/m00967009/Workspace/PTOAS/lib/PTO/IR/PTO.cpp:4193)); those are not the same contract as the physical intra-block semaphore-bank IDs. The generic bridge helper still accepts static IDs through 31 before the operation-specific path narrows them.

**Q:** other mode/base/pipe forms, dynamic ranges, and general target portability remain unproven. They should not be declared failures in the observed Q2 default intra-block form solely from the old FFTS row. Fresh preflight-sync-block.mlir RUN/FileCheck passed, including Cube bank expansion. Runtime deadlock/order validation is still necessary.

The in-tree PTO snapshot and separate PTOAS binary should be tested as a pair; local verification alone is insufficient. Mixed AIC/AIV sections and HOST sidecars are packaging contracts, not additional AVE instructions.

### F15

**Negative infinity can be changed into a very large negative finite number.**

**Q2 source — this occurrence is real:**

~~~python
m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
~~~

**Possible NPU-IR scalar staging — illustrative, not captured for this exact Q2 SDPA specialization:**

~~~mlir
%minus_inf = arith.constant 0xFF800000 : f32
memref.store %minus_inf, %tmp[]
    : memref<f32, #hivm.address_space<ub>>
~~~

**If that direct scalar-store pattern reaches the bridge, the result changes as follows:**

~~~mlir
// Before the bridge's constant rewrite:
%minus_inf = arith.constant 0xFF800000 : f32
pto.store_scalar %minus_inf, %tmp[%zero] : <f32, ub>, f32

// After the rewrite: -FLT_MAX is finite.
%finite_min = arith.constant 0xFF7FFFFF : f32 // -FLT_MAX
pto.store_scalar %finite_min, %tmp[%zero] : <f32, ub>, f32
~~~

**Why the distinction matters — mathematical illustration:**

~~~text
max(-infinity, -infinity) = -infinity
max(-FLT_MAX,  -infinity) = -FLT_MAX

The values are different before any later subtraction or exponential.
~~~

**Status:** the broad scalar-store rewrite is present in bridge source. The exact Q2 SDPA initializer has not been shown to take that scalar-store path. Its presence in Triton source alone does not prove it is rewritten.

**What to fix:** establish the exact generated path, then preserve infinity through lowering. This is primarily a correctness concern; no speed benefit is established.

**Source evidence and limits (optional detail).**

**O — source applicability, conditional lowering:** Q2 _sdpa_infer_kernel initializes m_i to -inf at [SD:264](/home/x84315859/Q2TritonKernel/src/kernels/sdpa.py:264). The bridge unconditionally scans direct F32 constants stored with pto.store_scalar and replaces negative infinity with negative maximum finite F32 ([B:5195](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Conversion/HIVMAVEToPTOASVMI/HIVMAVEToPTOASVMI.cpp:5195)–5219; invoked at [B:5422](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Conversion/HIVMAVEToPTOASVMI/HIVMAVEToPTOASVMI.cpp:5422)). The tracker’s validated failure was a softmax simulator path.

**Q:** no exact Q2 SDPA generated IR in this assessment proves that this initializer becomes a direct scalar store; it may remain vector-valued. Thus this is a conditional in-scope abstraction risk, not a demonstrated current SDPA failure.

**I:** max(-FLT_MAX, -inf) differs from max(-inf, -inf); subsequent subtraction/exp may turn an exceptional row into different values. NaN behavior also needs testing. No timing benefit justifies this semantic substitution. Correct lowering should preserve the constant bits; locating the historical defect requires comparing VMI, VPTO, LLVM constant materialization, and simulator/hardware behavior.

### F16

**A 32-bit comparison selects 16-bit values, so the bridge widens the value bits.**

**NPU-IR before conversion — adapted from the exact dqkwg helper:**

~~~mlir
// Compare signed 32-bit indices.
%condition = ave.hir.vcmp <LT> %indices, %limit, %all
    : vector<64xi32>, vector<64xi1> -> vector<64xi1>

// Choose one of two F16 values for every lane.
%chosen = ave.hir.vsel %condition, %yes, %no
    : vector<64xi1>, vector<64xf16>
~~~

**PTOAS VMI after conversion — comparison shown as an existing %condition:**

~~~mlir
// Transport each F16 bit pattern in a 32-bit integer lane.
%yes16 = pto.vmi.bitcast %yes
    : !pto.vmi.vreg<64xf16> -> !pto.vmi.vreg<64xi16>
%yes32 = pto.vmi.extui %yes16
    : !pto.vmi.vreg<64xi16> -> !pto.vmi.vreg<64xi32>
%no16 = pto.vmi.bitcast %no
    : !pto.vmi.vreg<64xf16> -> !pto.vmi.vreg<64xi16>
%no32 = pto.vmi.extui %no16
    : !pto.vmi.vreg<64xi16> -> !pto.vmi.vreg<64xi32>

// Select using B32 value lanes, matching the comparison's granularity.
%chosen32 = pto.vmi.vsel %condition, %yes32, %no32
    : !pto.vmi.mask<64xpred>, !pto.vmi.vreg<64xi32>,
      !pto.vmi.vreg<64xi32> -> !pto.vmi.vreg<64xi32>

// Recover the selected F16 bits.
%chosen16 = pto.vmi.trunci %chosen32 {saturate = "NOSAT"}
    : !pto.vmi.vreg<64xi32> -> !pto.vmi.vreg<64xi16>
%chosen = pto.vmi.bitcast %chosen16
    : !pto.vmi.vreg<64xi16> -> !pto.vmi.vreg<64xf16>
~~~

**Bit example:**

~~~text
F16 bits for 1.0:        0x3C00
After zero extension:   0x00003C00
After selection/narrow: 0x3C00
~~~

**Why this preserves values:** the bridge selects the original bits. It does not convert 1.0 into the integer 1 and back, and it does not narrow the I32 comparison.

**Why it may be slower:** two extensions and one truncation surround the select, and the values use wider intermediate registers. Some operations can fold later, so this is a likely cost mechanism rather than a measured instruction penalty.

Example source: [preflight-vsel-dqkwg.mlir](/home/m00967009/Workspace/AscendNPU-IR/bishengir/test/Conversion/HIVMAVEToPTOASVMI/preflight-vsel-dqkwg.mlir:12).

**Source evidence and limits (optional detail).**

**O:** Q2 dqkwg uses comparisons and selections to form temporal/causal masks and apply them ([DQ:276](/home/x84315859/Q2TritonKernel/src/kernels/fla/ops/common/chunk_o.py:276)–301). The exact outlined mixed-width helper is [R:1082](/home/m00967009/Workspace/Logs/raw_bwd_dqkwg_sep29.log:1082)–1091, copied in preflight-vsel-dqkwg.mlir:12–28. It compares I32 lanes and selects F16 values.

Bridge [B:3340](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Conversion/HIVMAVEToPTOASVMI/HIVMAVEToPTOASVMI.cpp:3340)–3405 preserves signed comparison and seed but selects via zero-extended B16 bit patterns in B32 carriers, then NOSAT truncates and bitcasts back. This avoids changing comparison precision or floating values. preflight-vsel-b32-to-b16-mask.mlir also covers unsigned/partial seed and floating compare forms.

**I/Q:** there are two logical integer extensions and one truncation beyond the select; bitcasts may be representation-only. PTOAS may fold particular carriers, so these are not a measured three-instruction penalty. Physical extui/trunci constraints are in [P:14218](/home/m00967009/Workspace/PTOAS/lib/PTO/Transforms/VMIToVPTO.cpp:14218) onward. Proper granularity/layout conversion would allow the predicate to drive native B16 selection without widening both data streams.

The fresh exact dqkwg helper conversion/FileCheck passed. Its full-kernel execution is blocked elsewhere; no NaN-payload or signed-zero runtime test was run.

### F17

**Adding a necessary memory barrier can make a live vector impossible to place.**

There is no complete successful PTOAS output for this failure. The following is **pseudocode explaining the dependency**, rather than a copied dqkwg fragment.

**NPU-IR computation that needs its original loaded value:**

~~~text
old = vector_load(buffer_A)
vector_store(new_value, buffer_B)
next = vector_load(buffer_B)
answer = old + next
~~~

**What bridge finalization needs to arrange:**

~~~text
old = vector_load(buffer_A)
vector_store(new_value, buffer_B)

VST_VLD_barrier()  // Ensure the store happens before the later load.

next = vector_load(buffer_B)
answer = old + next
~~~

**Where the difficulty is:** old was created before the barrier but is needed after it. The vector-scope rules may prevent carrying it across that boundary.

**Why simply reloading is unsafe:** buffer_A might have changed or might alias buffer_B. Reading it again after the store could produce a different value from old. A constant can safely be recreated; an arbitrary memory read cannot.

**Current outcome:** the opt-in bridge finalizer rejects the crossing non-pure value with:

~~~text
cannot place VST_VLD barrier without carrying a non-pure VMI value across it
~~~

**Status:** this is a recorded dqkwg finalization blocker. The automatic pipeline currently skips this finalizer. It is separate from F09, where PTOAS itself introduces and shares a predicate.

**Source evidence and limits (optional detail).**

**O:** The tracker’s host/device packaging status records that a temporary compiler combining host extraction, Cube-pointer, and packed-BF16 changes gets recovered dqkwg to the VST_VLD diagnostic at original recovered line 932 (durable original operation [R:1379](/home/m00967009/Workspace/Logs/raw_bwd_dqkwg_sep29.log:1379)). This is a recorded experiment, not a fresh full-module pass result from the assessed compiler.

The current bridge finalizer inserts dependency barriers and attempts rematerialization; [B:5100](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Conversion/HIVMAVEToPTOASVMI/HIVMAVEToPTOASVMI.cpp:5100)–5124 rejects “cannot place VST_VLD barrier without carrying a non-pure VMI value across it.” Its call sequence is at [B:5444](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Conversion/HIVMAVEToPTOASVMI/HIVMAVEToPTOASVMI.cpp:5444)–5452. This is an opt-in bridge-finalization failure, not a freshly established default-pipeline failure; both finalizers are currently omitted from automatic compilation. It precedes and differs from PTOAS nested physical predicate inference.

**I:** F17 and F09 need separate regressions. Removing the barrier can violate UB visibility; reloading a previously loaded value after an intervening write can change values. Repair needs dependence-aware scope/lifetime handling, not a blanket “clone vector producers” rule. No implementation workaround is established.

### S01

**A cumulative sum is implemented by a template containing many operations.**

**What the kernel needs — small illustration along rows:**

~~~text
Input rows:       Output rows:
[1, 10]           [1, 10]
[2, 20]           [3, 30]
[3, 30]           [6, 60]
~~~

**Structured NPU-IR before template conversion — adapted from the regression:**

~~~mlir
hivm.hir.vcumsum
    ins(%src : memref<32x64xf32, strided<[80, 1]>,
                     #hivm.address_space<ub>>)
    outs(%dst : memref<32x64xf32, strided<[96, 1]>,
                      #hivm.address_space<ub>>)
    cum_dims = [0] reverse = false
~~~

**Bridge/PTOAS representation — shortened call:**

~~~mlir
// Pointer extraction and constant definitions are omitted.
// Arguments preserve rows, columns, independent row strides, and direction.
func.call @__pto_cumsum_2d_f32_dim0(
    %src_ptr, %dst_ptr, %rows32, %cols64,
    %src_stride80, %dst_stride96, %forward)
    : (!pto.ptr<f32, ub>, !pto.ptr<f32, ub>,
       i64, i64, i64, i64, i1) -> ()
~~~

**Inside the PTODSL helper — pseudocode summarizing its actual stages:**

~~~text
copy logical input elements to output

for stage_width = 1, 2, 4, 8, ...:
    left = pto.vgather2(group_left_total, indices, mask)
    right = pto.vgather2(destination_row, indices, mask)
    result = pto.vadd(left, right, mask)
    pto.vscatter(result, destination_row, indices, mask)
    pto.pipe_barrier(V)
~~~

**What this means:** there is no need for one PTOAS instruction named cumsum. Existing vector reads, additions, writes, and barriers can implement the scan.

**Current support:** this PTODSL helper now exists, so the old “missing cumsum helper” classification is outdated for its accepted layouts. The complete original Q2 cases still need rerunning.

**Cost:** multiple scan stages read and write UB repeatedly. Native CCE also implements scan stages, so compare the actual sequences rather than assuming the template expansion is uniquely expensive.

**Source evidence and limits (optional detail).**

**O:** Older reports list _mlir_ciface_cumsum_2d_float_dim0 as missing for three Q2 shapes, but the assessed source implements it: matching/import at [T:316](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Conversion/HIVMTemplatesToPTO/HIVMTemplatesToPTO.cpp:316)–372 and a generated PTODSL template in [SCAN:25](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Template/lib/RegBase/Vector/cumsum_2d_f32_dim0_ptodsl.py:25)–111. It copies logical rows, executes Sklansky stages using masked vgather2/vscatter and vadd, and adds a vector pipe barrier between stages. Reverse mode uses reversed physical row indices.

Native ordinary floating dim-0 scan also copies then uses a Sklansky tree ([CSC:215](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Template/lib/RegBase/Vector/Cumsum.cpp:215)–220; implementation [CSC:25](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Template/lib/RegBase/Vector/Cumsum.cpp:25)–122). The separately named compensated scan must not be conflated with the ordinary helper. Supported bridge shapes require matching rank-two F32 UB layouts, static columns and non-overlapping static row strides.

**I/Q:** the scan is a composition, not a missing primitive VMI instruction. Gather/scatter and per-stage UB traffic may cost more than the native contiguous load/store paths, but native has its own copy and scan traffic. Cancellation/tolerance, forward/reverse tails, exact aliasing and independent strides need full-kernel testing. cumsum-ptodsl.mlir checks selection and imported operations; it does not establish measured Q2 correctness/performance.

### S02

**Copy 50 values without accidentally touching 14 extra vector lanes.**

**Structured NPU-IR before conversion — the actual regression shape:**

~~~mlir
hivm.hir.copy
    ins(%src : memref<2x50xf32, strided<[50, 1], offset: ?>,
                     #hivm.address_space<ub>>)
    outs(%dst : memref<2x50xf32, strided<[64, 1], offset: ?>,
                      #hivm.address_space<ub>>)
~~~

The source rows contain 50 values each. Destination rows are spaced 64 elements apart, leaving 14 padding positions after each logical row.

**PTOAS VMI representation — pseudocode with actual operation names:**

~~~text
for row in 0 .. 1:
    src_row = src_base + row * 50
    dst_row = dst_base + row * 64
    mask = pto.vmi.create_mask(50)       // 50 active lanes out of 64
    indices = pto.vmi.iota(0)           // [0, 1, ..., 63]
    values = pto.vmi.vgather(src_row, indices, mask)
    pto.vmi.masked_store(values, dst_row, offset=0, mask=mask)
~~~

**Why the mask is necessary:** reading all 64 source values would run into the next row or beyond the allocation. Writing all 64 destination lanes would overwrite padding that the logical copy does not own.

**Why the bridge uses vector operations:** they handle these exact tails without assuming a full aligned DMA transfer is legal.

**Possible cost:** address calculations, a mask, and vector reads/writes for each chunk. An aligned DMA fast path could help compatible shapes, but cannot simply replace this 50-value tail with a 64-value transfer.

**Source evidence and limits (optional detail).**

**O:** Historical D50 layernorm failures link unresolved 2-D half load and FP32 UB copy helpers ([OLD:95](/home/m00967009/Workspace/Planner/bridge/planning/ptoas_failure_categorization.md:95) onward). Current half-load matching accepts the observed static-stride/compatible-shape configurations ([T:263](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Conversion/HIVMTemplatesToPTO/HIVMTemplatesToPTO.cpp:263)–278). Current FP32 copy lowers static rank-two rows with independent row strides and 64-lane chunks to VMI gather plus masked store ([T:619](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Conversion/HIVMTemplatesToPTO/HIVMTemplatesToPTO.cpp:619)–683).

copy-ubuf-to-ubuf-2d-f32.mlir:3–16 explicitly tests a tight 50-column source and padded destination, so reading all 64 lanes is invalid. The newer source-derived D50 layernorm configurations reach LLVM ([LP:39](/home/m00967009/Workspace/Logs/layernorm_compile_oct15/README.md:39)–69), but were not executed.

**I:** this is a supported fallback with potential efficiency cost, not a missing PTOAS copy instruction. The aligned rank-one branch already uses mte_ub_ub ([T:686](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Conversion/HIVMTemplatesToPTO/HIVMTemplatesToPTO.cpp:686)–707). A rank-two DMA fast path should be selected only when row alignment, stride units, exact tails, and synchronization are proven equivalent.

## Tracker reconciliation and covered controls

The tracker remains useful, but its rows span different dates and validation levels.

| Tracker terminology | This assessment |
|---|---|
| Signed vsmins/vsmaxs | No occurrence established in reviewed Q2 generated paths; excluded from confirmed inventory. Signed comparison in F16 is different. Current constant min/max code accepts supported direct masks, broader than the tracker’s all-active-only text. |
| reduction ADD/MAX | F04 confirms ADD in Q2; source SDPA MAX is present, but exact Q2 MAX AVE lowering was not captured here. |
| BRC_B32/BRC_B16 singleton; B16 64-lane | F02/F03. Preserve pipeline distinctions. |
| Continuous ONEPT_B32 | F05 additionally establishes dqkwg occurrence. |
| Predicate-valued PB8/PB16/PB32 memory | Real compatibility issue; Q2 packed-memory occurrence unresolved. Ordinary predicates and masked numeric stores do not prove predicate memory. |
| VL1 vextf | F01; also track VL1 vtruncf and cumsum non-ALL promotion separately by consumer contract. |
| Partial arithmetic masks | F06; current H support supersedes older design-note rejection. Mask loss remains in pinned submodule but is fixed upstream by U01. |
| Stored F32 -inf | F15, with exact Q2 scalar-store trigger unproven. |
| sync_block_set/wait | F14; named A5 intra-block support supersedes old blanket core/mode-loss description. |
| Vendored snapshot, external CCE ABI, pointer collapse | F11/F14 and pinned compatibility validation. No independent runtime penalty is assigned to merely vendoring the dialect. |
| GM scalar cache bypass; strided GM stores | F12/F13; backend/version and cache semantics are part of the contract. |
| Singleton wrappers | F07; cast elimination is not scope completion. |
| Target metadata / host-device separation | Packaging context; current hard-coded A5 and host sidecar handoff are not additional AVE instruction inventory. |
| Row-split F32-to-F16 fixpipe example | Not established for the exact Q2 cases reviewed. Historical Q2 BF16/F32 Cube-stage blockers are real but distinct. |
| Layernorm sparse interleave/widening | F08. |
| Layernorm nested-scope rematerialization | F09, reproduced on old local baseline; latest upstream execution remains unresolved. |

**O — finalizer status matters:** the existing predicate-memory note says the comparison runner automatically repairs byte carriers after PTOAS. Current runner [RUN:745](/home/m00967009/Workspace/Planner/bridge/tools/run_comparison_flow.sh:745)–764 copies raw VPTO unchanged, and compiler [PIPE:305](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Tools/bishengir-compile/regbase/PassPipeline.cpp:305)–313 keeps bridge finalizers opt-in. The earlier flash_atten_64x64 simulator result in Planner/bridge/designs/flash-attention-bridge-summary.md depended on explicit physical predicate repair. It does not validate raw VPTO today. Confirm Q2 predicate spilling before adding it to this report’s findings; if present, default-path correctness becomes a high priority.

**O — covered rather than missing:** ordinary full-mask arithmetic, direct floating vmax, NORM loads/stores, ALL/observed H prefixes, fused vexpdif, and full-register interleave have representations. Dqkwg vexpdif is in [R:1263](/home/m00967009/Workspace/Logs/raw_bwd_dqkwg_sep29.log:1263), mapped by [B:3841](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Conversion/HIVMAVEToPTOASVMI/HIVMAVEToPTOASVMI.cpp:3841)–3870 to VMI vexpdif and lowered directly by [P:9149](/home/m00967009/Workspace/PTOAS/lib/PTO/Transforms/VMIToVPTO.cpp:9149)–9224. Do not replace it with subtract+exp merely because a primitive name appears unfamiliar.

The historical layernorm vadd attribute failure is also resolved in current source: [B:3774](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Conversion/HIVMAVEToPTOASVMI/HIVMAVEToPTOASVMI.cpp:3774) permits the unit reductionOp marker on ALL-masked vadd; preflight-vadd-reduction-marker.mlir checks that the loop remains elementwise accumulation rather than horizontal reduction. Fresh RUN/FileCheck passed. It adds no intrinsic operation by itself.

Stage 1’s historical Cube failures must not be counted as VMI vector gaps: current Cube/PTODSLTemplateImporter.cpp already lists BF16 transposed-B, FP32 IEEE, BF16 fixpipe variants, and ND2NZ templates. Exact result/init/dimension/layout matching still needs a corpus rerun. A generic diagnostic mentioning “result, bias, unit flag, A-transpose, or I4” does not prove I4 occurs. The ten BF16 tl.exp frontend failures and native backward BT64 UB overflows occur before the bridge.

## Performance assessment and measurements needed

**O:** The only cited Q2 timing deltas are the two historical forward layernorm measurements above. Newer validation reports 20 selected forward/backward specializations reaching VMI, 19 reaching LLVM; a wider 170-candidate sweep records 145 LLVM successes, four D1200 scope failures, and 21 native UB-capacity failures ([LP:24](/home/m00967009/Workspace/Logs/layernorm_compile_oct15/README.md:24)–69). Those are compilation counts, not runtime or accuracy results.

**I:** likely costs should be attributed as follows:

| Mechanism | Affected findings | Required comparison |
|---|---|---|
| Extra data conversions/register pressure | F02, F16 | Same exact helper before/after lowering improvement; count physical instructions, vregs, masks and spills, then time many iterations with identical data/layouts |
| Broadcast or projection that exists only to restore AVE carrier types | F03, F04, F07 | Native vs bridge LLVM/assembly; determine which logical ops fold and which issue on vector/scalar pipelines |
| Different stateful-store/flush strategy | F05 | Native vstus.post/vstas stream vs bridge vsts: store issue count, UB transaction footprint, barriers, final data and aliasing |
| Per-element scalar memory work and handshakes | F12, F13 | Native branch actually selected for identical stride/tail/cache policy; report scalar instructions, GM/UB transactions and pipeline stalls |
| Masked-tail and scan UB traffic | S01, S02 | Forward/reverse scan sizes, contiguous/padded rows and D50 tails; count gather/scatter vs vlds/vsts/DMA and stage barriers |
| Rematerialization and synchronization | F09, F17 | Compare a legal repaired build with native; count pset/vdup repetitions and barriers; do not assign a runtime penalty to a non-compiling build |
| Predication repair | F01, F06 | Compare native masked instructions vs safe select/zero expansion; test inactive-lane escape graphs and instruction count separately |

Use fixed Q2 shapes and exact BT/warp/autotuning choices, same hardware/core count/compiler options/cache policies, warmups, repeated paired runs, and distribution/error bars. Measure both device kernel time and end-to-end launch time; the 4–5 μs case is especially sensitive to overhead/noise. Compare numerical outputs before timing, including tails, nonzero offsets, NaNs/infinities/signed zero where the kernel contract allows them. Simulator success alone cannot quantify hardware pipeline overlap or cache latency. Avoid summing estimated per-finding penalties: optimizations and register pressure interact.

## Prioritized recommendations

| Priority | Action | Correctness risk | Performance/breadth rationale |
|---|---|---|---|
| P0 | Integrate upstream F06 mask-preserving lowering with its dependencies; keep enforcing partial-conversion guards, F01 | High when inactive lanes escape; reject outside proof | Shared across promotions, tails and reductions; first establish the exact cumsum mask |
| P0 | Probe upstream stream fusion on exact singleton stores; add logical state only if needed, F05 | Unresolved stream/flush correctness | Confirmed dqkwg, plus existing attention evidence; potential store/sync overhead |
| P0 conditional | Establish Q2 predicate-memory and scalar -inf lowering, then protect those default paths | High if those paths occur; F15 changes IEEE values | Source attention patterns exist, but exact generated path is missing; do not implement speculative support inventories |
| P1 | Reproduce F09 on latest upstream, then fix nested-scope rematerialization if still needed | Confirmed pinned-baseline compilation blocker; latest status unresolved | Four D1200 warp choices; preferred general fix over loop-local constant workaround |
| P1 | Complete BF16 packed-store carriers and bridge barrier lifetime handling, F10/F17 | Compilation blockers; unsafe “fixes” could change values/order | Exposes the next dqkwg stage; validate each separately, not only reduced functions |
| P1 | Revalidate descriptor, cache and sync contracts on pinned paired builds, F11–F14 | Addressing/cache/order errors can be severe | Causal convolution, gate vectors and mixed Cube/Vector paths; record full failing IR |
| P2 | Re-probe F16 after upstream mask fixes before retaining its widening workaround; optimize F02/F08 layouts | Lower after correct guarded support | Select has concrete conversion expansion; widening-first may already be efficient |
| P2 | Eliminate redundant reduction/broadcast wrappers and compare scan/copy fallback cost, F03/F04/F07/S01/S02 | Preserve defined-lane and tail semantics | Layernorm broadly affected; historical slowdown is measured but causality is unknown |

For representation changes, prefer: (1) a semantic VMI operation with PTOAS-native physical lowering; (2) a guarded composition of existing logical VMI operations; (3) a narrowly documented compatibility fallback. Preserve Stage 1’s actual CCE-template boundary. Do not introduce bridge-only physical backends for instruction-lowered AVE behavior.

## Open questions

1. What exact non-ALL predicate and inactive-lane consumers occur in each of the seven cumsum promotion failures? Reduced diagnostics cannot answer this; capture full current AVE.
2. Which exact Q2 attention variants spill predicates into packed i1 UB with PB8/PB16/PB32? A tl.where or comparison alone does not establish this.
3. Does Q2 SDPA’s -inf initializer become a direct scalar store and trigger F15? Does exact infinity survive PTOAS LLVM and hardware when the rewrite is removed?
4. Are continuous dqkwg stores equivalent across unaligned offsets, short/tail streams, aliasing, and store/load boundaries?
5. What are the exact inactive-lane and FP-status requirements of the AVE conversion/arithmetic paths? The lane-zero proof covers observed values, not a general exception contract.
6. What becomes the next first failure in the complete current dqkwg module after each independently proposed fix? The temporary combined-compiler experiment is not current main end-to-end evidence.
7. Do the ten original causal-convolution specializations now satisfy the descriptor/address contract? Capture full current views and used metadata before generalizing.
8. Do all observed mixed sync pairs address the right semaphore banks after PTOAS lowering, with correct producer/consumer pipe order at runtime? Static tests do not establish deadlock freedom.
9. Are current native and bridge reduction trees/exceptional values equivalent for the relevant tolerances? A reassoc attribute does not prove numerical equivalence.
10. How much of the historical layernorm slowdown remains on these revisions, and which physical sequences account for it? No per-issue timing is available.
11. Are every prebuilt compiler and generated PTODSL resource exactly reproducible from the recorded source? Add binary/resource hashes and paired toolchain CI.
12. How should Q2 FP8 attention and other repository kernels beyond the saved FLA corpus be captured? Their source alone is insufficient to assert specific unsupported AVE instructions.

## Checks performed for this assessment

**Original-baseline checks (not latest-master test results):**

- Ran the documented positive RUN/FileCheck commands for seven files under AscendNPU-IR/bishengir/test/Conversion/HIVMAVEToPTOASVMI: preflight-sparse-interleave-extf.mlir, preflight-vl1-extf-one-point-store.mlir, preflight-vl1-truncf-one-point-store.mlir, preflight-vsel-dqkwg.mlir, preflight-vadd-reduction-marker.mlir, preflight-sync-block.mlir, preflight-brc-b16-load.mlir. All conversion and FileCheck exits were zero.
- Compiled Logs/layernorm_compile_oct15/layernorm-sparse.vmi.mlir through the local PTOAS LLVM path: exit 0.
- Compiled the saved minimal nested-mask reproducer and complete original backward D1200 VMI: both exit 1 with the predicate scope-escape diagnostic.
- Compiled the existing manually modified D1200 VMI: exit 0. This is experimental artifact validation, not an implemented fix.
- Converted a temporary BF16 analog of the F16 packed-store test: exit 1 at the width-wrapper cast, confirming F10.
- Read the relevant native AVE definitions/lowering, bridge code/tests, PTODSL/CCE implementations, PTOAS layout/physicalization/scope code, compatibility designs, historical corpus classification, and newer source-derived layernorm report.

PTOAS downstream command shape:

~~~sh
PTOAS/build/tools/ptoas/ptoas INPUT.vmi.mlir \
  --pto-arch=a5 --pto-backend=vpto --cann-output-version=9.0.0 \
  --emit-vpto-llvm-ir -o /tmp/q2-assessment-OUTPUT.ll
~~~

For the original assessment, only this Markdown report was added to the project and probe artifacts were written under /tmp. The later upstream reassessment fetched a separate source/history copy and attempted a temporary compiler build; its separate validation status is recorded below. No compiler source fix, submodule update, commit, NPU/simulator benchmark, or destructive command was performed.


## Explanation update

This documentation update records the upstream/submodule commit comparison, corrects F06’s upstream status and F03’s existing alternative, and adds a glossary, a suggested reading order, before/after examples, and a native AscendNPU-IR behavior/lowering table column for all 19 findings. It preserves the original source evidence and validation results. Adapted IR was compared with the cited regression inputs, expected outputs, saved conversion output, or conversion code. Pseudocode and proposed/conditional output are explicitly labeled. No compiler or kernel implementation was changed, and no new performance result is claimed.
