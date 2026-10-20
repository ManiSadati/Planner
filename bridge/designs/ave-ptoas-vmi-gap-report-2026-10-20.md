# AVE–PTOAS VMI gap report — 2026-10-20

Audience: PTOAS project owners and the VMI team. Entry IDs G01–G14 match the categorized gap list; E01–E14 identify the evidence cited for each entry. Requests and proposed acceptance checks are recommendations, not completed implementation or validation.

## Executive summary

The strongest Q2-backed representation requests are **masked floating conversion** and **predicate-valued memory**. Captured cumsum widening rejects because its partial mask is outside the bridge's accepted subset, while VMI conversion operations have no mask operand. Called Q2 solve helpers store and reload predicates as data, but VMI memory operations use numeric vector data/results. These need explicit semantic contracts, not merely more operation-name mappings. See G01/E01 and G03/E03.

There are also correctness and integration issues. A focused probe reproduces inconsistent vector/scalar normalization for min/max; explicitly typed signed/unsigned operations already work, including runtime scalars. Arithmetic-mask loss is a **pinned-version defect fixed upstream**, and the saved D1200 scope failure is a **historical downstream blocker requiring current-pair reproduction**. These findings must not become blanket claims that upstream PTOAS lacks arithmetic masks, signed min/max or broadcasts. See G09/E09, G02/E02 and G10/E10.

Continuous stores and B16 layouts need exact-pattern validation before removing bridge workarounds. Reduction carrier reconstruction is an optimization opportunity. Four additional entries concern bridge restrictions and use of existing target capabilities, rather than demonstrated missing PTOAS instructions. No per-gap runtime penalty, frequency or speedup is established. See G04–G08 and G11–G14.

## Scope, sources and validation boundaries

The report reconciles the four requested documents and their referenced bridge source, VMI definitions/lowering, regression inputs, captured Q2 AVE/VMI and saved diagnostics. The validated inventory and categorized list determine the retained findings; the original assessment and tracker provide history and conflicting claims, not automatic additions to the gap count.

**Evidence E00:** [Validated inventory](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-evidence-inventory-2026-10-20.md:1); [categorized list with code examples](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-categorized-gaps-2026-10-20.md:1); [original assessment](/home/m00967009/Workspace/Planner/bridge/designs/q2-ave-to-ptoas-vmi-assessment-2026-10-16.md:1); [compatibility tracker](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-compatibility-tracker.md:1).

| Evidence baseline | Meaning |
|---|---|
| AscendNPU-IR `8b8f40f88e99`; actual PTOAS submodule `3c1c98fcc8ec`; Q2 `ebf78b205a87` | Inventory source/capture baseline, including existing local state recorded in its provenance |
| Upstream `ec8d1810b6c1`; earlier assessment comparison `7a32cb83ef92` | Retained source comparisons; not fresh execution of the latest upstream pipeline |
| G09 upstream source `739480c309c2`, checked 2026-10-20 | Scalar-normalization follow-up; local compiler probes are separate from upstream source inspection |
| Historical local PTOAS checkout `a3c33b937261` | Original probe context; source correspondence does not establish the provenance of every prebuilt binary |

The inventory's [capture provenance](/home/m00967009/Workspace/Logs/q2_ave_requirements_oct19/provenance.json:1) and the assessment's [baseline distinctions](/home/m00967009/Workspace/Planner/bridge/designs/q2-ave-to-ptoas-vmi-assessment-2026-10-16.md:77) retain the details. Successful parsing, AVE-to-VMI conversion, VPTO emission, LLVM emission and device correctness are separate milestones. Unexecuted FileCheck expectations are source evidence, not test passes. This report adds no compiler/device test results and makes no new upstream-tip claim.

## Categories, ownership and review order

| Category | Entries | PTOAS-facing decision |
|---|---|---|
| Masking and tail behavior | G01, G02 | Add conversion-mask semantics; integrate existing arithmetic-mask preservation |
| Operation and semantic coverage | G03 | Represent predicates as packed memory data |
| Stateful memory streams | G04 | Validate singleton stream fusion; add logical state if required |
| Layout and data movement | G05–G07 | Handle partial B16 results, broadcast-producer layouts and mixed-width predicate use |
| Reduction and broadcast abstractions | G08 | Avoid unnecessary carrier reconstruction where consumers permit |
| Type and conversion support | G09 | Fix scalar/vector normalization; consider a clearer signedness interface |
| Lowering and codegen correctness | G10 | Reproduce and repair nested predicate-scope placement if needed |
| Bridge integration with existing PTOAS capabilities | G11–G14 | Agree target contracts/regressions and separate bridge implementation work |

Stateful streams are separate from ordinary data movement because alignment/base state and completion span stores or iterations. Bridge integration is separate because those entries do not establish missing target instructions. Related B16 entries remain distinct: shuffle-result materialization, broadcast-producer adaptation and predicate conversion require different implementations. G10's PTOAS-introduced predicate failure also differs from G13's bridge-finalizer failure.

For Q2 enablement, review recorded failures first: G01, G10 and bridge-owned G11. G02–G08 and G12 have captured Q2 requirements/workarounds without a demonstrated full-kernel failure caused by every individual issue. G09 has a reproduced focused defect but no exact Q2 occurrence; G13 is conditional and G14 has focused bridge-test relevance. These are applicability tiers, not severity scores; evidence does not rank items within a tier.

## Confirmed and likely PTOAS-facing gaps

### G01 — Preserve masks during floating conversion

**Category / status:** Masking and tail behavior; confirmed representation limit and recorded Q2 conversion rejection.

**Example and limitation:** Q2 cumsum uses `ave.hir.vextf` on 64 F16 lanes with H/PB32, followed by a same-mask store. VMI `extf` / `vcvt` have no conversion-mask operand. The bridge accepts ALL or narrowly checked VL1 one-point-store graphs, so this broader partial conversion rejects.

```mlir
// Captured AVE excerpt; surrounding definitions/attributes omitted.
%wide = ave.hir.vextf %x, <part_even>, %half
    : vector<64xf16>, vector<64xf32>, vector<64xi1>
// Closest declared VMI form; NOT successful conversion of that rejected graph.
%wide = pto.vmi.extf %x
    : !pto.vmi.vreg<64xf16> -> !pto.vmi.vreg<64xf32>
```

The VMI form has no operand for `%half`.

**Requested implementation:** Add masked widening/narrowing with explicit inactive-lane semantics; preserve active values, `part_even` lane mapping, rounding and saturation. Acceptance should cover the captured H-mask graph and existing VL1 guards through physical lowering, with inactive-lane consumers checked explicitly.

**Workaround / impact:** Extra-lane conversion is accepted only under the lane-zero consumer proof; other graphs reject. No runtime conversion overhead is quantified. **Confidence:** high for the contract gap and rejection; general inactive-lane/FP-status equivalence remains unvalidated.

**Evidence E01:** [Inventory F01](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-evidence-inventory-2026-10-20.md:12); [captured H-mask graph](/home/m00967009/Workspace/Logs/q2_ave_requirements_oct19/captures/cumsum_scalar_False/before-bridge.mlir:33); [compiler rejection](/home/m00967009/Workspace/Logs/q2_ave_requirements_oct19/captures/cumsum_scalar_False/compiler.txt:2357); [VMI extf operands](/home/m00967009/Workspace/AscendNPU-IR/third-party/ptoas/include/PTO/IR/VMIOps.td:569); [vcvt operands](/home/m00967009/Workspace/AscendNPU-IR/third-party/ptoas/include/PTO/IR/VMIOps.td:1413).

### G02 — Retain arithmetic predicates through lowering

**Category / status:** Masking and tail behavior; confirmed pinned-version defect, fixed upstream by `2503c259`.

**Example and limitation:** Captured dqkwg uses H-masked `ave.hir.vexp`. The pinned unified-to-legacy binary/unary helpers build maskless operations even when the emitted unified VMI carries a predicate.

**Requested implementation:** Integrate upstream mask-preserving direct VPTO lowering with its matching mask/layout handling into the supported bridge/PTOAS pair. Acceptance should show partial and all-off predicates reaching the physical arithmetic, including consumers that can observe inactive results.

**Workaround / impact:** A same-mask final store can hide inactive results; it does not prove safety for every consumer. **Confidence:** high for source mask loss and upstream correction; Q2 runtime corruption and per-gap timing are not established. No new upstream arithmetic-mask feature is requested.

**Evidence E02:** [Inventory F06](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-evidence-inventory-2026-10-20.md:15); [dqkwg H-mask example](/home/m00967009/Workspace/Logs/q2_ave_requirements_oct19/captures/reused_dqkwg_H2_B4/before-bridge.mlir:983); [pinned helpers](/home/m00967009/Workspace/AscendNPU-IR/third-party/ptoas/lib/PTO/Transforms/VMILowerUnifiedToLegacy.cpp:233); [upstream mask-preserving helper](/tmp/ptoas-research-20261019/lib/PTO/Transforms/VMIToVPTO/VMIToVPTOUnifiedPatternInternals.cpp:57).

### G03 — Load and store predicates as data

**Category / status:** Operation and semantic coverage; confirmed representation gap and captured Q2 PB32 occurrence.

**Example and limitation:** Called Q2 solve helpers spill/reload `vector<64xi1>` through PB32 UB memory. VMI memory operations accept numeric vector data/results; their controlling masks do not represent the predicate being stored.

```mlir
// Simplified AVE excerpt; #ub denotes the HIVM UB address space.
ave.hir.masked_store <NORM_B8> %dst[%c0], %active, %predicate
    {functionType = #ave.func_dist_type<pb32>}
    : memref<8xi1, #map, #ub>, vector<64xi1>, vector<64xi1>
```

`%predicate` is the stored data; `%active` independently controls the write.

**Requested implementation:** Add first-class predicate-valued UB load/store, with PB granularity and explicit packed-address units, lowering through the existing physical predicate-memory operations. Start with the captured PB32 forms; validate tails, controlling masks and addresses before broadening the contract.

**Workaround / impact:** Byte carriers and pointer markers need explicit physical repair. The inspected default compiler/comparison flows omit that repair, so raw carriers are not complete packed-memory support. **Confidence:** high for the logical gap and captured occurrence; the full PB-width/address matrix and end-to-end correctness remain unvalidated. This is a logical VMI request, not missing hardware-level predicate instructions.

**Evidence E03:** [Inventory T02](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-evidence-inventory-2026-10-20.md:13); [solve reload](/home/m00967009/Workspace/Logs/q2_ave_requirements_oct19/captures/solve/before-bridge.mlir:1194), [store](/home/m00967009/Workspace/Logs/q2_ave_requirements_oct19/captures/solve/before-bridge.mlir:1515); [VMI data-versus-mask typing](/home/m00967009/Workspace/AscendNPU-IR/third-party/ptoas/include/PTO/IR/VMIOps.td:758); [physical repair expectation](/home/m00967009/Workspace/AscendNPU-IR/bishengir/test/Conversion/HIVMAVEToPTOASVMI/finalize-vpto-predicate-memory.mlir:5); [opt-in finalizers](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Tools/bishengir-compile/regbase/PassPipeline.cpp:315); [comparison flow](/home/m00967009/Workspace/Planner/bridge/tools/run_comparison_flow.sh:723).

### G04 — Preserve continuous one-point store streams

**Category / status:** Stateful memory streams; confirmed native/bridge sequence difference, conditional need for a new representation.

**Example and limitation:** Dqkwg has `masked_store <ONEPT_B32>` with `hivm.is_continuous`. Native lowering produces updated alignment/base state and completion; the bridge emits independent singleton `pto.vmi.vstore` operations without those results.

**Requested implementation:** Run the exact singleton graph through supported upstream stream fusion and compare state, addresses and final completion with native lowering. If fusion cannot recover the required sequence, introduce logical stateful-store operations/results and lower them to initialization, post-increment writes and completion. A memory barrier alone does not represent stream completion.

**Workaround / impact:** Ordinary one-point writes are represented; native continuous-stream equivalence is unproven. **Confidence:** high for the sequence difference, conditional for new upstream work. Existing upstream tests cover 64-F32 chunks, not closure of this singleton case; no numerical failure or timing penalty is established.

**Evidence E04:** [Inventory F05](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-evidence-inventory-2026-10-20.md:14); [dqkwg continuous stores](/home/m00967009/Workspace/Logs/q2_ave_requirements_oct19/captures/reused_dqkwg_H2_B4/before-bridge.mlir:1376); [native state/completion](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Conversion/HIVMAVEToAVEIntrin/HIVMAVEToAVEIntrin.cpp:1119); [bridge singleton builder](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Conversion/HIVMAVEToPTOASVMI/HIVMAVEToPTOASVMI.cpp:4481); [upstream stream example](/tmp/ptoas-research-20261019/test/lit/vmi_new/vmi_to_vpto_stateful_stream_fusion.pto:58).

### G05 — Materialize partial B16 shuffle results

**Category / status:** Layout and data movement; confirmed materializer restriction with a supported special-case composition.

**Example and limitation:** Backward D1200 contains sparse self-`ave.hir.vintlv` from 128 F16 lanes into two 64-lane halves, consumed by `vextf`. General physical shuffle materialization rejects partial result chunks; the bridge accepts only checked widening consumers of those sparse halves.

**Requested implementation:** Support partial B16 shuffle-result materialization with correct logical/physical lane mapping and explicit padding treatment. Preserve `part_even` interpretation; regress both F16/BF16 and consumers beyond the existing widening-only proof before generalizing.

**Workaround / impact:** Widen the full input first, then select complete B32 halves. Other sparse-half consumers reject. **Confidence:** high for the restriction; the checked composition already exists, and logical shuffles need not become executed shuffle instructions. No measured performance gap is established.

**Evidence E05:** [Inventory F08](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-evidence-inventory-2026-10-20.md:16); [D1200 AVE](/home/m00967009/Workspace/Logs/q2_ave_requirements_oct19/captures/norm_bwd_D1200/before-bridge.mlir:237); [materializer rejection](/home/m00967009/Workspace/AscendNPU-IR/third-party/ptoas/lib/PTO/Transforms/VMIToVPTO.cpp:2596); [bridge testcase](/home/m00967009/Workspace/AscendNPU-IR/bishengir/test/Conversion/HIVMAVEToPTOASVMI/preflight-sparse-interleave-extf.mlir:3); [retained upstream restriction](/tmp/ptoas-research-20261019/lib/PTO/Transforms/VMIToVPTO/VMIToVPTOMaskInternals.cpp:289).

### G06 — Adapt B16 broadcast loads to widening layouts

**Category / status:** Layout and data movement; confirmed pinned workaround, upstream replacement path unvalidated.

**Example and limitation:** Dqkwg broadcasts one BF16 value with `vload <BRC_B16>` into 64 lanes, then widens it. The pinned broadcast producer cannot directly satisfy the sparse consumer layout.

**Requested implementation:** Allow the logical 64-lane broadcast producer to materialize the widening consumer's layout while retaining scalar-address vector broadcast behavior. Validate the exact graph on the supported memory interface; upstream removal of `dist_mode` requires a deliberate bridge migration.

**Workaround / impact:** A 128-lane broadcast followed by a 64-lane lane-zero splat preserves values and still reads one scalar address. **Confidence:** high for the pinned expansion; neither an equivalent current-upstream failure nor an extra executed instruction cost is established.

**Evidence E06:** [Inventory F02](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-evidence-inventory-2026-10-20.md:17); [dqkwg graph](/home/m00967009/Workspace/Logs/q2_ave_requirements_oct19/captures/reused_dqkwg_H2_B4/before-bridge.mlir:951); [bridge expansion](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Conversion/HIVMAVEToPTOASVMI/HIVMAVEToPTOASVMI.cpp:3778); [bridge testcase](/home/m00967009/Workspace/AscendNPU-IR/bishengir/test/Conversion/HIVMAVEToPTOASVMI/preflight-brc-b16-load.mlir:3).

### G07 — Select B16 values with a B32 comparison predicate

**Category / status:** Layout and data movement; confirmed bridge expansion, likely validation/extension work rather than established general missing support.

**Example and limitation:** Dqkwg compares signed I32 indices with `ave.hir.vcmp <LT>`, then selects F16 data using `vsel`. The bridge avoids the sparse B16 mask/select path by selecting widened data bit carriers.

**Requested implementation:** Validate and, if necessary, extend predicate-granularity/layout adaptation for this exact 64-lane graph so B32 comparison predicates drive B16 selection without widening both data streams. Preserve signed comparison, logical lane correspondence and selected bits.

**Workaround / impact:** Bitcast to I16, zero-extend both operands to I32, select, truncate with `NOSAT`, and bitcast back. **Confidence:** high for the expansion, medium for additional upstream implementation need: related mask-conversion support already exists. Physical cost and full-kernel execution remain unmeasured.

**Evidence E07:** [Inventory F16](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-evidence-inventory-2026-10-20.md:19); [dqkwg compare/select](/home/m00967009/Workspace/Logs/q2_ave_requirements_oct19/captures/reused_dqkwg_H2_B4/before-bridge.mlir:938); [bridge lowering](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Conversion/HIVMAVEToPTOASVMI/HIVMAVEToPTOASVMI.cpp:3487); [upstream predicate-conversion regression](/tmp/ptoas-research-20261019/test/lit/vmi_new/vmi_mask_granularity_assignment_vsel.pto:111).

### G08 — Avoid unnecessary reduction carrier reconstruction

**Category / status:** Reduction and broadcast abstractions; confirmed shape difference, optimization opportunity.

**Example and limitation:** AVE ADD reduction retains a 64-lane type but defines the reduced value at the first position. Saved D1200 VMI has a singleton `pto.vmi.reduce_addf` followed by `pto.vmi.broadcast` to restore the carrier width. Floating VMI reduction verifiers require a one-lane result.

**Requested implementation:** Keep reduced values singleton through compatible consumers; eliminate carrier-only broadcasts when legal and materialize a broadcast when genuinely needed. Tests must distinguish AVE's defined first lane from an assumed full-width splat.

**Workaround / impact:** Reduction plus broadcast supplies the expected type. **Confidence:** high for the mapping; missing reduction execution, unnecessary broadcast in every graph and a runtime penalty are not established. ADD is captured; exact Q2 MAX lowering remains unconfirmed.

**Evidence E08:** [Inventory F04](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-evidence-inventory-2026-10-20.md:18); [D1200 output](/home/m00967009/Workspace/Logs/q2_ave_requirements_oct19/captures/norm_bwd_D1200/converted.vmi.mlir:603); [AVE result contract](/home/m00967009/Workspace/AscendNPU-IR/bishengir/include/bishengir/Dialect/HIVMAVE/IR/HIVMAVEOps.td:627); [VMI result checks](/home/m00967009/Workspace/AscendNPU-IR/third-party/ptoas/lib/PTO/IR/VMI.cpp:1793).

### G09 — Normalize scalar min/max signedness consistently

**Category / status:** Type and conversion support; reproduced local normalization error, matching upstream source behavior at `739480c309c2`.

**Example and limitation:** AVE `vsmins` / `vsmaxs` specify signed ordering for signless I32 operands. The bridge's `signed_min_dynamic_scalar` testcase rejects a runtime scalar at its constant-only gate. Separately, a focused VMI probe reproduces the target normalizer changing the vector to `ui32` while leaving the scalar `i32`, which then fails exact type agreement.

```text
Before normalization: vector<i32>  + scalar i32
After normalization:  vector<ui32> + scalar i32  → verifier error
Supported signed form: vector<si32> + scalar si32
```

These are schematic type summaries; operation identity or an explicit signedness contract must retain AVE's signed intent.

**Requested implementation:** Make vector/scalar normalization consistent for constant and runtime scalars. Separately evaluate signed/unsigned scalar min/max operation names or a semantic attribute closer to AVE and `llvm.hivm.vmins.s.x.v64i32`, `llvm.hivm.vmins.u.x.v64i32`, `llvm.hivm.vmaxs.s.x.v64i32` and `llvm.hivm.vmaxs.u.x.v64i32`; this is an optional interface improvement, not a missing intrinsic. Acceptance should cover signed/unsigned ordering, runtime scalars and matching operand types through LLVM emission.

**Workaround / impact:** Explicit signed vector/scalar carriers and a result bitcast preserve semantics. Runtime signed/unsigned operands successfully reach LLVM in the local probe, and same-width casts disappear; extra NPU conversion instructions are not demonstrated. The bridge constant-only gate remains a separate limitation. **Confidence:** high for the defect and supported typed alternative; no exact Q2 occurrence established.

**Evidence E09:** [Inventory T01](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-evidence-inventory-2026-10-20.md:11); [bridge dynamic-scalar testcase](/home/m00967009/Workspace/AscendNPU-IR/bishengir/test/Conversion/HIVMAVEToPTOASVMI/unsupported-signed-vector-scalar-dynamic.mlir:4); [upstream normalizer](/tmp/ptoas-signed-minmax-20261020/lib/PTO/Transforms/VMI/VMINormalizeSignlessIntToUnsigned.cpp:119); [upstream scalar verifier](/tmp/ptoas-signed-minmax-20261020/lib/PTO/IR/VMI/VMIArith.cpp:542); [local mismatch](/tmp/ptoas-signed-minmax-20261020/signless-normalization.log:1); [signed runtime LLVM output](/tmp/ptoas-signed-minmax-20261020/signedness-casts.ll:25), [unsigned output](/tmp/ptoas-signed-minmax-20261020/signedness-casts.ll:69); [separate LLVM intrinsic forms](/home/m00967009/Workspace/AscendNPU-IR/third-party/llvm-project/llvm/include/llvm/IR/IntrinsicsHIVM.td:769).

### G10 — Place generated predicates inside legal nested scopes

**Category / status:** Lowering and codegen correctness; confirmed historical downstream compile failure, current-upstream persistence unverified.

**Example and limitation:** Saved backward layernorm D1200 VMI contains a zero constant. PTOAS physicalization introduces a mask used by `pto.vdup`; scope inference fails on an escaping `!pto.mask<b32>` in the recorded nested-loop graph.

**Requested implementation:** Reproduce the exact saved case on the supported bridge/PTOAS pair, then repair nested consumer-segment discovery/rematerialization if still needed. Add a downstream regression that completes VPTO and LLVM emission without moving unsafe values or losing memory order.

**Workaround / impact:** A manual constant-placement variant is recorded as compiling, but it is not a general implemented fix. The original saved path cannot finish VPTO emission. **Confidence:** high for the historical failure; no latest-upstream failure or runtime penalty is asserted. October 19 `--emit-ptoas-vmi` success stops before this downstream stage.

**Evidence E10:** [Inventory F09](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-evidence-inventory-2026-10-20.md:20); [saved constant](/home/m00967009/Workspace/Logs/layernorm_compile_oct15/artifacts/bwd-T8192-D1200-silu-BT1-W4-specialized.vmi.mlir:171); [scope diagnostic](/home/m00967009/Workspace/Logs/layernorm_compile_oct15/artifacts/bwd-T8192-D1200-silu-BT1-W4-specialized.ptoas.log:1); [scope-anchor lookup](/home/m00967009/Workspace/AscendNPU-IR/third-party/ptoas/lib/PTO/Transforms/PTOInferVPTOVecScope.cpp:439); [October 19 emission boundary](/home/m00967009/Workspace/Logs/q2_ave_requirements_oct19/captures/norm_bwd_D1200/command.json:6).

## Bridge integration entries — no missing PTOAS instruction established

### G11 — Lower runtime integer metadata loads

**Category / status:** Bridge integration; confirmed Q2 bridge rejection.

**Example and limitation:** Solve-varlen uses `memref.load` for singleton I32 GM sequence metadata; a MoE capture records the same class of scalar-load rejection. The bridge admits singleton floating types. PTO's scalar-load verifier itself has no floating-only gate, so this does not establish missing integer loads in VMI/PTOAS.

**Requested implementation:** Bridge owners should map the captured I32 metadata forms through the supported scalar-load path. PTOAS should document and regression-test that existing type/address contract; a new VMI load operation is not justified by this evidence.

**Workaround / impact:** No accepted workaround is recorded for these forms; conversion rejects. **Confidence:** high for the bridge failure, with no target capability defect demonstrated.

**Evidence E11:** [Inventory scalar loads](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-evidence-inventory-2026-10-20.md:28); [Q2 source](/home/m00967009/Workspace/Q2TritonKernel/src/kernels/fla/ops/utils/solve_tril.py:150); [captured I32 load](/home/m00967009/Workspace/Logs/q2_ave_requirements_oct19/captures/solve_varlen/before-bridge.mlir:301); [solve rejection](/home/m00967009/Workspace/Logs/q2_ave_requirements_oct19/captures/solve_varlen/compiler.txt:35873); [MoE rejection](/home/m00967009/Workspace/Logs/q2_ave_requirements_oct19/captures/moe_wgrad_tail/compiler.txt:2729); [PTO verifier](/home/m00967009/Workspace/AscendNPU-IR/third-party/ptoas/lib/PTO/IR/PTO.cpp:9467).

### G12 — Recognize BF16 packed-store wrappers

**Category / status:** Bridge integration; confirmed source restriction and captured Q2 pattern.

**Example and limitation:** Dqkwg narrows with `vtruncf` to BF16, wraps 64 lanes as 128, bitcasts to an F32 carrier and uses `masked_store <PK_B32>`. The bridge recognizer/destination gate accept only F16; this is not proof that PTOAS cannot store BF16.

**Requested implementation:** Bridge owners should recognize only proven BF16 packing wrappers and store the dense logical values. PTOAS should provide a regression for that dense BF16 store contract; no new store instruction is established as necessary.

**Workaround / impact:** The F16 analog works; the inspected BF16 bridge mapping is absent. A separate modified compiler is not current support. **Confidence:** high for source gates and Q2 occurrence; the inventory records an analog rejection, not a new full-kernel run for this report.

**Evidence E12:** [Inventory F10](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-evidence-inventory-2026-10-20.md:26); [dqkwg chain](/home/m00967009/Workspace/Logs/q2_ave_requirements_oct19/captures/reused_dqkwg_H2_B4/before-bridge.mlir:899); [F16-only recognizer](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Conversion/HIVMAVEToPTOASVMI/HIVMAVEToPTOASVMI.cpp:1264); [store gates](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Conversion/HIVMAVEToPTOASVMI/HIVMAVEToPTOASVMI.cpp:4404).

### G13 — Preserve live values when placing memory barriers

**Category / status:** Bridge integration; source-confirmed rejection, conditional Q2 experiment.

**Example and limitation:** The recovered-dqkwg experiment reaches the opt-in bridge finalizer's inability to carry/rematerialize a non-pure VMI value across an inserted `VST_VLD` barrier. This is a bridge rejection, separate from G10's PTOAS-generated predicate failure.

**Requested implementation:** Jointly specify/regression-test legal value lifetimes across the barrier. Bridge owners should place barriers and preserve dependencies using that contract; request additional target lifetime support only after reproducing the exact crossing-value case.

**Workaround / impact:** No safe general workaround is recorded; finalization rejects. **Confidence:** high for the rejection code, conditional for Q2 applicability because the experiment combined temporary fixes. Default compilation omits this finalizer, so a default-path failure is not established.

**Evidence E13:** [Inventory F17 and experiment qualification](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-evidence-inventory-2026-10-20.md:29); [bridge diagnostic](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Conversion/HIVMAVEToPTOASVMI/HIVMAVEToPTOASVMI.cpp:5255); [opt-in policy](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Tools/bishengir-compile/regbase/PassPipeline.cpp:316).

### G14 — Materialize a singleton vector for real scalar consumers

**Category / status:** Bridge integration; confirmed focused-test restriction, Q2 demand unverified.

**Example and limitation:** The bridge's scalar-return/arithmetic rejection cases cast `vector<1xf32>` to `f32`, then return it or use `arith.addf`. Immediate vector/scalar/vector round trips fold, but genuine scalar consumers reject. The inventory does not establish absence of every possible VMI-to-scalar sequence.

**Requested implementation:** PTOAS should specify/test a supported lane-to-scalar materialization contract for those cases; add an operation only if existing sequences are insufficient. Bridge owners should lower genuine consumers through that contract.

**Workaround / impact:** Wrapper elimination handles only graphs returning to matching broadcasts. **Confidence:** high for the bridge restriction; no demonstrated Q2 genuine-extraction need, default-kernel blocker or runtime penalty is asserted.

**Evidence E14:** [Inventory F07](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-evidence-inventory-2026-10-20.md:27); [current scalar-consumer testcases](/home/m00967009/Workspace/AscendNPU-IR/bishengir/test/Conversion/HIVMAVEToPTOASVMI/preflight-reject-singleton-scalar-extraction.mlir:9); [fold restriction](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Conversion/HIVMAVEToPTOASVMI/HIVMAVEToPTOASVMI.cpp:4900).

## Reconciliation with the original assessment and tracker

The documents describe different dates and validation levels. The following corrections govern this report rather than being counted as new missing features:

| Earlier wording | Reconciled conclusion |
|---|---|
| Cumsum partial-conversion predicate unknown | The October 19 captured example establishes H/PB32; G01 does not generalize it to every historical cumsum failure. |
| Exact Q2 predicate spill/reload unconfirmed | Called solve helpers now establish PB32 predicate-valued memory, G03; they do not validate every PB width or attention specialization. |
| Signed min/max effectively constant/ALL-only | G09 distinguishes the PTOAS normalization bug, existing runtime typed support, supported direct partial masks and the bridge's remaining constant-only gate. |
| PTOAS drops arithmetic masks generally | G02 is pinned-version-specific and fixed upstream. |
| Singleton BRC_B32 has no vector alternative; sync fields generally lost; cumsum helper missing | Existing singleton vector-load alternatives, newer named intra-block sync handling and a cumsum helper are recorded; those blanket gaps are excluded. |
| Copied PTO snapshot / row-split single-transfer path | The assessment records direct submodule integration and a retained native row-split helper/alternative two-transfer resource; neither stale description supports a new AVE instruction gap. |

References: [inventory corrections/exclusions](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-evidence-inventory-2026-10-20.md:31), [assessment reconciliation](/home/m00967009/Workspace/Planner/bridge/designs/q2-ave-to-ptoas-vmi-assessment-2026-10-16.md:1463), [assessment signed-mask update](/home/m00967009/Workspace/Planner/bridge/designs/q2-ave-to-ptoas-vmi-assessment-2026-10-16.md:1307). The newer validated inventory takes precedence where it resolves an older uncertainty.

## Unverified items and assessment limits

- FP8, atomics, volatile behavior and other source-only requirements are not capability gaps without an exact target limitation. Frontend/Stage-1 capture blockers are not AVE-to-VMI support verdicts.
- Q2 SDPA's exact scalar-store trigger for the `-inf` rewrite, and the historical PTOAS-versus-simulator root cause, remain unresolved. No general inability to represent infinity is claimed.
- General descriptor/external-call ABI closure and converted row-split fixpipe equivalence need separate validation. Target/core packaging is a surrounding module contract, not another AVE vector instruction.
- Current upstream closure remains unproven for the exact singleton stream, sparse mixed-width select, B16 broadcast migration and historical D1200 scope failure. Inspecting related upstream tests does not execute the Q2 path.
- Capture coverage is selected, not an exhaustive shape/dtype/autotuning certification. Static presence is not a count of dynamic execution or independent failing kernels.
- No per-gap device timing or full numerical-equivalence matrix exists. Logical conversions/shuffles may fold; G09's tested same-width casts do fold. Historical whole-kernel timings in the assessment are not attributable to these gaps and are not used as slide impact figures.
- Some focused evidence lives under `/tmp`; preserve those probes or pin equivalent upstream permalinks before sharing the documents outside this workspace.

These limits follow the [inventory exclusions](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-evidence-inventory-2026-10-20.md:31), [capture scope](/home/m00967009/Workspace/Logs/q2_ave_requirements_oct19/README.md:126) and [assessment's measurement boundaries](/home/m00967009/Workspace/Planner/bridge/designs/q2-ave-to-ptoas-vmi-assessment-2026-10-16.md:1492).

## Requested owner decisions and acceptance

1. **VMI semantics:** assign owners for G01 conversion masks and G03 predicate memory; agree operand/result, inactive-lane and address-unit contracts before bridge workarounds are removed.
2. **Normalization and correctness:** fix G09's scalar/vector mismatch, integrate G02's existing upstream fix, and reproduce G10 on an explicitly paired toolchain before deciding its repair.
3. **Exact-pattern validation:** close G04–G07 with captured Q2 graphs; use G08 as an optimization candidate after correctness, not as a claimed speedup.
4. **Bridge ownership:** track G11–G14 separately, with PTOAS contract/regression support and bridge-side changes where the target capability already exists.

For each accepted change, record producer source, actual submodule checkout, generated resources and consuming compiler provenance. Proposed acceptance is paired AVE-to-VMI and downstream regression success, followed by numerical/device validation for the changed semantics; performance claims require subsequent paired measurement. No change is considered closed merely because parsing succeeds or a related upstream feature exists.
