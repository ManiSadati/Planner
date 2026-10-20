# PTOAS fusion: technical reference

Start with [the simple guide and examples](bridge-ptoas-fusion-opportunities-2026-10-19.md). This reference keeps the detailed source audit and pass conditions.

Audit date: 2026-10-19. Analysis and documentation only; no compiler, kernel, template, pipeline, or submodule implementation was changed.

Companion to [the Q2 AVE-to-VMI compatibility assessment](q2-ave-to-ptoas-vmi-assessment-2026-10-16.md). This audit examines actual optimization matchers and pipeline entry conditions, rather than assuming that fewer VMI operations means faster machine code.

## Main conclusions

1. **Make bridge output recognizable to the existing VMI/VPTO passes first.** Preserve direct SSA producer/consumer chains, canonical predicates, supported logical load/store forms, and useful memory proofs. Several optimizations already run without tile-fusion flags.
2. **Version compatibility is a prerequisite.** The current bridge emits old `pto.load_scalar` and `vload {dist_mode="brc"}` forms. The audited upstream matcher expects `pto.load` or a singleton logical VMI load; the upstream memory interface has removed `dist_mode`. An existing upstream optimization is not evidence the old bridge output automatically uses it.
3. **Tile fusion is a different pipeline.** Plain AVE-to-VMI output does not become tile-fusion input by setting `--enable-op-fusion` or attaching group attributes. The planner recognizes supported tile computations, establishes iteration domains, and constructs real `pto.fusion_region` interfaces before later loop and memory-elision passes run.
4. **Do not discard fusion already performed by NPU-IR.** The bridge already preserves fused `vexpdif`. A concrete conditional opportunity is mapping an already fused AVE `vmula` to VMI `vmula`; no such bridge conversion was found.
5. **No Q2 performance gain is measured here.** The evidence establishes matcher conditions and checked-in instruction-shape expectations. Exact Q2 triggering, emitted instruction counts, numerical results, and hardware timings need paired validation.

## Scope, revisions, and evidence labels

| Project/source | Examined revision |
|---|---|
| AscendNPU-IR bridge and preceding AVE passes | `8b8f40f88e993ac9fde1d13c1084ed53e1d1694c` |
| AscendNPU-IR recorded PTOAS gitlink | `e4469bc4533ee4d1075ee561991ba24c7860fabe` |
| Actual checked-out PTOAS submodule | `3c1c98fcc8ec0c49b8bededc9a0c7599c6a71a02`; differs from gitlink |
| Canonical upstream PTOAS master, verified earlier on 2026-10-19 | `ec8d1810b6c107b391a88305a14bde4c334cb21c`; source copy `/tmp/ptoas-research-20261019` |
| Planner, containing report and comparison runner | `332b19b02092fcc1aabbb714d49f392552a211c9` plus pre-existing local modifications |

**O — observed:** directly inspected code, operation definitions, pipeline wiring, or checked-in regression expectations. **I — inference/proposal:** recommended bridge change or expected performance effect. **Q — unresolved:** needs an exact input, fresh compilation, numerical comparison, or measurement.

Upstream links pin the audited commit. Local NPU-IR links refer to the recorded checkout and line numbers; they may move with later edits. A test citation means its source and FileCheck contract were inspected, not that a fresh run passed. Code examples below are adapted excerpts or explicitly schematic; they are not complete standalone kernels.

The Q2 connections are F02/F03 broadcast loads, F04 reductions, F05 continuous stores, F07 wrappers, F08 logical versus physical packing, F16 predicate/data-width handling, and relevant Stage 1 templates. FMA and tile-native examples are architectural opportunities; occurrence of every such pattern in generated Q2 AVE has not been established. They are not additional confirmed Q2 compatibility failures.

## Which passes actually run?

### Route A: existing vector bridge output

~~~text
AVE vector functions / imported low-level helper bodies
    -> bridge emits VMI and low-level PTO operations
    -> eligible direct helper calls are inlined
    -> unified VMI normalization + canonicalization/CSE
    -> vmi-pre-assignment-combine
    -> mask granularity and layout assignment
    -> layout rematerialization, folding, sinking
    -> VMI-to-VPTO
    -> vpto-stateful-stream-fusion
    -> physical conversion/mask cleanup, scope inference, LICM/CSE
    -> vpto-combine-reductions
    -> scheduler, when enabled; backend validation and emission
~~~

**O:** [Direct helper inlining](https://github.com/hw-native-sys/PTOAS/blob/ec8d1810b6c107b391a88305a14bde4c334cb21c/tools/ptoas/ptoas_pipeline.cpp#L948), [VMI semantic pipeline](https://github.com/hw-native-sys/PTOAS/blob/ec8d1810b6c107b391a88305a14bde4c334cb21c/tools/ptoas/ptoas_pipeline.cpp#L971), and [VPTO emission preparation](https://github.com/hw-native-sys/PTOAS/blob/ec8d1810b6c107b391a88305a14bde4c334cb21c/tools/ptoas/ptoas_pipeline.cpp#L713) establish this order. These combines do not require `--enable-op-fusion`. The physical stream pass follows VMI-to-VPTO in the semantic pipeline. Pure VMI/VPTO input without unexpanded tile ops takes [Pure VMI/VPTO fast-path selection](https://github.com/hw-native-sys/PTOAS/blob/ec8d1810b6c107b391a88305a14bde4c334cb21c/tools/ptoas/ptoas_pipeline.cpp#L1729) and bypasses the tile mainline.

**I:** keep legal computation visible across private helpers where inlining is appropriate. PTOAS honors `no_inline`; opaque external/native CCE calls do not provide an inspectable VMI arithmetic chain. Do not remove a deliberate ABI or synchronization boundary merely to expose a match.

### Route B: separate tile-native fusion lifecycle

~~~text
Supported tile compute operations + tile shapes / destinations
    -> FusionPlan -> OpScheduling -> FusionRegionGen
    -> lower/expand tile operations and inline operation-library bodies
    -> LowLevelLoopFusion -> canonicalize/CSE
    -> FusionPredicateElision -> FusionLoadStoreElision
    -> optional VexpdifFusion
    -> optional innermost unroll, cleanup, and repeated elision/fusion
    -> FlattenFusionRegion
    -> remaining VMI/VPTO lowering and emission preparation
~~~

**O:** [Fusion option defaults](https://github.com/hw-native-sys/PTOAS/blob/ec8d1810b6c107b391a88305a14bde4c334cb21c/tools/ptoas/ptoas.cpp#L290) defaults tile op fusion off. [A5 fusion configuration](https://github.com/hw-native-sys/PTOAS/blob/ec8d1810b6c107b391a88305a14bde4c334cb21c/tools/ptoas/ptoas_pipeline.cpp#L1138) requires A5 and level2/level3; the VPTO backend gets region generation, while EmitC uses last-use annotations. [Tile expansion and fusion gate](https://github.com/hw-native-sys/PTOAS/blob/ec8d1810b6c107b391a88305a14bde4c334cb21c/tools/ptoas/ptoas_pipeline.cpp#L813) and [Post-tile fusion lifecycle](https://github.com/hw-native-sys/PTOAS/blob/ec8d1810b6c107b391a88305a14bde4c334cb21c/tools/ptoas/ptoas_pipeline.cpp#L782) place the post-lowering lifecycle on the tile-expansion path. Vexpdif fusion defaults on, but that does not override its enclosing pipeline/region requirements. The current [Comparison runner](/home/m00967009/Workspace/Planner/bridge/tools/run_comparison_flow.sh:738) does not pass `--enable-op-fusion`; PTODSL mode adds level3 to preserve explicit addresses, which alone does not enable fusion.

**O:** NPU-IR's [NPU-IR bridge/native split](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Tools/bishengir-compile/regbase/PassPipeline.cpp:299) converts templates/standard operations, then AVE-to-VMI, and returns before native intrinsic lowering and continuous-store hoisting. Selected PTODSL template conversion can emit VMI/scalar operations, as [Bridge structured template lowering example](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Conversion/HIVMTemplatesToPTO/HIVMTemplatesToPTO.cpp:694) shows. Selecting PTODSL templates is therefore not proof that supported tile compute reaches FusionPlan.

## Findings and bridge actions

| ID / optimization | Available to ordinary bridge VMI path? | What the bridge can do | Main limit / status |
|---|---|---|---|
| C01: singleton/scalar UB load + broadcast | Yes, after compatible input legalization | Emit supported direct load/broadcast producers; retain one-use, type, UB and dependence information | Old `load_scalar` differs from upstream `load`; F02's exact 64-lane BF16 widening path remains unvalidated |
| C02: grouped loads and grouped broadcast fusion | Yes, when actual group semantics are represented | Preserve row/group identity and stride; emit grouped logical operations where proven | Current bridge does not construct group-load/group-slot/group-broadcast producers; metadata alone cannot trigger this matcher |
| C03: fold/sink layout materialization | Yes | Preserve logical lane meaning and layout freedom; avoid gratuitous pack/unpack or escaping wrappers | Support tables, mask/data compatibility, and fanout constrain folding |
| C04: combine reduction trees | Yes | Keep semantic reductions and recognizable equivalent masks; establish numerical permission first | Different runtime masks do not match; original outside users can retain old reductions; not a general F04 broadcast eliminator |
| C05: simplify physical predicates | Yes | Preserve honest ALL/VL/H/prefix forms and shared predicates | It does not prove arbitrary runtime predicates equal or relax masks |
| C06: physical stateful stream fusion | Yes, if logical accesses become recognized streams | Keep analyzable SCF/address chains; preserve continuity/memory facts for a real consumer | Current matcher needs physical state/base chains, constant loop conditions, and safe effects; one-point vsts is not a stream |
| C07: sub + exp -> expdif | Tile-fusion region path, not general raw VMI | Preserve already fused AVE expdif; for later discovery, provide a supported integration | F32, identical SSA mask, one-use subtraction, real fusion region |
| C08: fused multiply-add | Explicit VMI vmula lowers; separate mul/add discovery not found | Map a semantically valid already fused AVE vmula | Bridge conversion absent in inspected registration; contraction legality must be established |
| C09: tile planning and low-level loop fusion | No automatic entry from plain VMI | Preserve structured HIVM computations as supported tile ops before AVE physicalization, if pursuing this architecture | Real domain, destination and region contracts required; flags or fabricated attrs are insufficient |
| C10: fusion-local predicate and load/store elision | Requires valid fusion-region lifecycle | Preserve identical iteration/access relationships and all observable outputs | Physical patterns and observable frontier checks; cannot safely hide escaping stores |
| C11: fusion-guided unrolling | Requires enabled tile fusion and optional unroll path | Retain static/divisible bounds and valid planner-selected factors | No cleanup-tail generation here; increased register pressure may outweigh benefits |

### C01: expose the supported load/broadcast composition

**O — implementation:** [Singleton VMI load/broadcast combine](https://github.com/hw-native-sys/PTOAS/blob/ec8d1810b6c107b391a88305a14bde4c334cb21c/lib/PTO/Transforms/VMI/VMIPreAssignmentCombine.cpp#L156) and [Scalar UB load/broadcast combine](https://github.com/hw-native-sys/PTOAS/blob/ec8d1810b6c107b391a88305a14bde4c334cb21c/lib/PTO/Transforms/VMI/VMIPreAssignmentCombine.cpp#L204) turn a direct singleton logical load or scalar UB load followed by broadcast into an internal `group_broadcast_load` with one group and zero group stride.

For both forms, [Load movement guard](https://github.com/hw-native-sys/PTOAS/blob/ec8d1810b6c107b391a88305a14bde4c334cb21c/lib/PTO/Transforms/VMI/VMIPreAssignmentCombine.cpp#L112) requires the load and broadcast to be in the same block. Intervening operations must be pure or have known default-resource effects. Reads are permitted; a write must name a memory value that alias analysis proves disjoint from the loaded source. Unknown/full-region effects, allocation/free-like effects, unknown write destinations, and possible aliasing block the move.

Additional conditions:

- Singleton VMI form: exactly one logical loaded element; load result has one use; broadcast result is a VMI vector.
- Scalar PTO form: matcher uses **PTOLoadOp**, not an arbitrary scalar memory op; source is UB-backed; pointer/memref element, loaded scalar, and result element agree; element width is 8, 16, or 32 bits; load result has one use.

**O — current bridge mismatch:** [Bridge singleton/scalar load mapping](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Conversion/HIVMAVEToPTOASVMI/HIVMAVEToPTOASVMI.cpp:3771) emits `pto::LoadScalarOp`. [Checked-out submodule scalar-load definition](/home/m00967009/Workspace/AscendNPU-IR/third-party/ptoas/include/PTO/IR/PTOOps.td:185) spells that operation `pto.load_scalar`. [Upstream scalar-load definition](https://github.com/hw-native-sys/PTOAS/blob/ec8d1810b6c107b391a88305a14bde4c334cb21c/include/PTO/IR/VPTOOps.td#L171) defines upstream `PTOLoadOp` as `pto.load`, the producer [Scalar UB load/broadcast combine](https://github.com/hw-native-sys/PTOAS/blob/ec8d1810b6c107b391a88305a14bde4c334cb21c/lib/PTO/Transforms/VMI/VMIPreAssignmentCombine.cpp#L204) matches. No `LoadScalarOp` definition/use was found in the audited upstream include/lib source. Also [Upstream logical memory interface](https://github.com/hw-native-sys/PTOAS/blob/ec8d1810b6c107b391a88305a14bde4c334cb21c/include/PTO/IR/VMIOps.td#L1427) no longer declares load/store `dist_mode`; [Legacy distribution-mode rejection test](https://github.com/hw-native-sys/PTOAS/blob/ec8d1810b6c107b391a88305a14bde4c334cb21c/test/lit/vmi_new/vmi_dist_mode_removed_invalid.pto#L9) explicitly tests legacy-input rejection. The old bridge is not a drop-in producer for these new interfaces.

**I — proposed bridge shape after interface migration:**

~~~mlir
// Read one logical BF16 value on the vector-load path.
%one = pto.vmi.vload %src[%off]
    : !pto.ptr<bf16, ub> -> !pto.vmi.vreg<1xbf16>
%x = pto.vmi.vbrc %one
    : !pto.vmi.vreg<1xbf16> -> !pto.vmi.vreg<64xbf16>
~~~

This needs no explicit bridge shuffle. **Q:** compile its exact `extf<64xf32>` consumer before retiring F02's 128-lane-plus-splat workaround. [Scalar load/broadcast regression](https://github.com/hw-native-sys/PTOAS/blob/ec8d1810b6c107b391a88305a14bde4c334cb21c/test/lit/vmi_new/vmi_pre_assignment_combine_scalar_ub_load_broadcast.pto#L13) proves expected BRC_B32/B16/B8 output for its own F32/F16/I8 cases, not this BF16 sparse-widening chain. F03's one-lane result can use the singleton vector load without an additional logical replication operation.

**O:** [Bridge B16 broadcast workaround](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Conversion/HIVMAVEToPTOASVMI/HIVMAVEToPTOASVMI.cpp:3778) already emits broadcast-load intent for multi-lane B16 forms; they are not currently scalar-load/broadcast pairs awaiting this combine. The B32 multi-lane branch likewise already requests a broadcast load. Do not count both the old direct broadcast lowering and the proposed fusion as separate savings.

**I — bridge help:** preserve direct producer/consumer SSA, avoid representational casts between them where safely foldable, keep constant/index arithmetic visible, and translate allocation identity/alias facts to a consumer PTOAS actually uses. Do not duplicate a real scalar load with other uses just to meet the one-use condition; that may change ordering or increase traffic.

**Evidence/tests:** [Scalar load/broadcast regression](https://github.com/hw-native-sys/PTOAS/blob/ec8d1810b6c107b391a88305a14bde4c334cb21c/test/lit/vmi_new/vmi_pre_assignment_combine_scalar_ub_load_broadcast.pto#L13) checks no scalar load or vdup remains in the positive physical output. [Scalar/VMI broadcast guard regression](https://github.com/hw-native-sys/PTOAS/blob/ec8d1810b6c107b391a88305a14bde4c334cb21c/test/lit/vmi_new/vmi_pre_assignment_combine_scalar_ub_load_broadcast_guards.pto#L12) retains multiple-use, interfering-store, and GM cases. [No-alias intervening-store cases](https://github.com/hw-native-sys/PTOAS/blob/ec8d1810b6c107b391a88305a14bde4c334cb21c/test/lit/vmi_new/vmi_pre_assignment_combine_scalar_ub_load_broadcast_guards.pto#L47) permits movement across a store into a distinct allocation.

**Performance inference:** an eligible scalar load plus scalar-to-vector duplication becomes one vector broadcast-load instruction per matched physical form, reducing scalar/vector transfer and an intermediate value. A singleton VMI load may already lower to physical BRC; deleting a logical broadcast is not automatically another removed machine instruction. No exact Q2 gain is measured.

### C02: expose groups and adjacent rows as operations, not vague metadata

**O:** [Grouped load canonicalization and broadcast fusion](https://github.com/hw-native-sys/PTOAS/blob/ec8d1810b6c107b391a88305a14bde4c334cb21c/lib/PTO/Transforms/VMI/VMIPreAssignmentCombine.cpp#L40) contains two different transformations:

1. `group_load -> load` when groups are positive, lane count divides into groups, and **constant row stride equals lanes per group**. This replaces the producer in place; one-use is not required.
2. `group_slot_load -> group_broadcast` becomes `group_broadcast_load` when the direct slot-load result has one use, group counts agree, and the broadcast result is a VMI vector.

The second matcher does **not** contain the explicit same-block/intervening-effect/alias guard used by C01. This audit does not infer a broader permission to reorder grouped reads across writes. Preserve naturally adjacent, dependence-safe compositions; do not manufacture unsafe placement merely to trigger the matcher.

**Adapted upstream logical shape:**

~~~mlir
// One scalar per group, then broadcast within each group.
%slots = pto.vmi.vload %src[%off], %row_stride {group = 8}
    : !pto.ptr<f32, ub> -> !pto.vmi.vreg<8xf32>
%expanded = pto.vmi.vbrc %slots {group = 8}
    : !pto.vmi.vreg<8xf32> -> !pto.vmi.vreg<128xf32>
~~~

**O:** no VMI group-load/group-slot/group-broadcast construction was found in the current bridge. Merely adding a `group` hint to existing unrelated flat operations cannot satisfy the direct-defining-op matcher. **I:** earlier NPU-IR structured shapes and row relationships can support an actual grouped representation; the bridge must translate the relationship into legal VMI operations and preserve row stride and lane order.

**Evidence/tests:** [Contiguous grouped-load regression](https://github.com/hw-native-sys/PTOAS/blob/ec8d1810b6c107b391a88305a14bde4c334cb21c/test/lit/vmi_new/vmi_pre_assignment_combine_contiguous_group_load.pto#L13) covers packed adjacent rows; [Noncontiguous grouped-load regression](https://github.com/hw-native-sys/PTOAS/blob/ec8d1810b6c107b391a88305a14bde4c334cb21c/test/lit/vmi_new/vmi_pre_assignment_combine_noncontiguous_group_load.pto#L12) retains gapped/dynamic grouped loads. [Grouped-slot broadcast instruction-selection regression](https://github.com/hw-native-sys/PTOAS/blob/ec8d1810b6c107b391a88305a14bde4c334cb21c/test/lit/vmi_new/vmi_pre_assignment_combine_group_slot_broadcast_load.pto#L71) includes a supported packet case with one physical E2B_B32 load forwarded across four physical results, and a shape requiring a longer fallback. Not every grouped composition becomes one instruction.

**Performance inference:** eligible packet loads can replace repeated BRC loads and selector sequences. Contiguous canonicalization exposes ordinary memory/layout lowering. Q2 occurrence of these exact grouped forms and benefit versus native packing are unresolved.

### C03: let PTOAS fold layout work into loads/stores or move it after arithmetic

These are representation optimizations, not loop/kernel fusion.

**O:** [Load/store layout folding](https://github.com/hw-native-sys/PTOAS/blob/ec8d1810b6c107b391a88305a14bde4c334cb21c/lib/PTO/Transforms/VMI/VMILayoutFold.cpp#L81) can absorb layout conversion into a logical load when every load use is a supported ensure of the same target type. It also cancels inverse layout ensures. [Store-compatible layout folding](https://github.com/hw-native-sys/PTOAS/blob/ec8d1810b6c107b391a88305a14bde4c334cb21c/lib/PTO/Transforms/VMI/VMILayoutFold.cpp#L148) can remove an ensure-to-contiguous before a store if that store supports the producer's existing layout. Masked-store folding needs compatible data **and mask** transformations; it cannot discard a mask rearrangement independently.

**O:** [Materialization sinking conditions](https://github.com/hw-native-sys/PTOAS/blob/ec8d1810b6c107b391a88305a14bde4c334cb21c/lib/PTO/Transforms/VMI/VMILayoutSinkMaterialization.cpp#L247) can transform this schematic pattern:

~~~text
ensure(a, layout B) + ensure(b, layout B)
                 -> arithmetic in B

into:
arithmetic(a, b, original layout A)
                 -> one ensure(result, B)
~~~

All relevant vector operands must have matching source types/layout transitions; the operation must be supported. Governing masks must have compatible ensures or be rematerializable supported mask roots. Arbitrary direct predicates can block it. Fanout may keep the old ensures alive, so operand count alone does not establish savings.

**O — bridge position:** [Bridge logical VMI type conversion](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Conversion/HIVMAVEToPTOASVMI/HIVMAVEToPTOASVMI.cpp:265) initially leaves VMI layouts unspecified. [Bridge sparse split/widen normalization](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Conversion/HIVMAVEToPTOASVMI/HIVMAVEToPTOASVMI.cpp:4081) already interprets AVE's physical sparse split as a logical dense widening plus full F32 halves. This is useful normalization, not a reason to force every VMI value back into AVE's physical arrangement. [NPU-IR layout analysis/removal order](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Dialect/HIVMAVE/Pipelines/HIVMAVEPipelines.cpp:67) shows where source layout analysis and physical normalization happen before removal of layout types.

**I — bridge help:** preserve logical lane mapping/packing provenance, collapse checked carrier-only wrappers, and keep genuine reshuffles explicit. Allow later layout inference where no semantic constraint fixes placement. Hard-coding a workaround's physical layout can remove otherwise legal load/store folding choices.

**Evidence/tests:** [Store-fold instruction regression](https://github.com/hw-native-sys/PTOAS/blob/ec8d1810b6c107b391a88305a14bde4c334cb21c/test/lit/vmi_new/vmi_layout_fold_store.pto#L37) checks no standalone vintlv and direct vstsx2 stores. [Masked-store folding regression](https://github.com/hw-native-sys/PTOAS/blob/ec8d1810b6c107b391a88305a14bde4c334cb21c/test/lit/vmi_new/vmi_layout_fold_masked_store.pto#L13) checks coordinated data/mask folding. [Materialization-sinking regression](https://github.com/hw-native-sys/PTOAS/blob/ec8d1810b6c107b391a88305a14bde4c334cb21c/test/lit/vmi_new/vmi_layout_sink_materialization_binary.pto#L12) includes positive binary/select/compare cases and incompatible materializations that remain unchanged.

**Performance inference:** fewer executed rearrangements or shared result materialization; possibly fewer temporary values. This does not guarantee fewer registers under all fanout/scheduling choices.

### C04: preserve reduction structure and recognizable mask equivalence

**O:** [Reduction-tree matching and reconstruction](https://github.com/hw-native-sys/PTOAS/blob/ec8d1810b6c107b391a88305a14bde4c334cb21c/lib/PTO/Transforms/VPTO/VPTOCombineReductions.cpp#L104) recognizes at least two reduction leaves in a compatible physical combine tree. Matched families are add/vcadd, add/vcgadd, max/vcmax, max/vcgmax, min/vcmin, and min/vcgmin. Leaf source physical vector types must agree. At most one ordinary init/base value is accepted; a reduction from a different family is not a valid init. [Widening i16 grouped-reduction exclusion](https://github.com/hw-native-sys/PTOAS/blob/ec8d1810b6c107b391a88305a14bde4c334cb21c/lib/PTO/Transforms/VPTO/VPTOCombineReductions.cpp#L164) rejects native i16 integer vcgadd because its sums widen despite the input-shaped register type.

**Schematic physical sum example:**

~~~text
Before: r0 = vcadd(a, m); r1 = vcadd(b, m); out = vadd(r0, r1, VL1)
After:  ab = vadd(a, b, m); out = vcadd(ab, m)
~~~

**O:** [Reduction mask equivalence](https://github.com/hw-native-sys/PTOAS/blob/ec8d1810b6c107b391a88305a14bde4c334cb21c/lib/PTO/Transforms/VPTO/VPTOCombineReductions.cpp#L37) accepts mask equivalence only for the same SSA value, or same type and same pattern from the same supported pset/pge operation kind. Equal-looking runtime expressions are insufficient. A pset ALL and pge ALL are not considered equivalent by this helper unless other simplification unifies them. The root combine mask may differ from the reduction mask (VL1 vs ALL); nested combines must match the root combine mask.

**O:** there is no one-use check for the reduction leaves. If old reductions have outside users they can remain, so the rewrite does not guarantee their instructions disappear. The physical pass does not inspect a high-level reassoc/mathmode attribute; it relies on legal physical input contracts.

**O — bridge position:** [Bridge reduction and carrier restoration](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Conversion/HIVMAVEToPTOASVMI/HIVMAVEToPTOASVMI.cpp:3602) emits singleton reductions and then broadcasts when restoring AVE's vector result type. ADD currently supplies a unit reassoc attribute. **I:** retain semantic reductions, original predicates and permitted reassociation, and keep independent lanes grouped as a larger logical reduction when justified. Do not manually expand the reduction into obscure arithmetic or memory traffic before PTOAS can recognize it. **This pass is not evidence that F04's result-shape broadcast is eliminated.** That remains a separate consumer/canonicalization question.

**Evidence/tests:** [Equal versus unknown chunk-mask regression](https://github.com/hw-native-sys/PTOAS/blob/ec8d1810b6c107b391a88305a14bde4c334cb21c/test/lit/vmi_new/vmi_to_vpto_reduce_addf_multichunk.pto#L43) explicitly contrasts unknown per-chunk masks (two vcadd plus vadd) with recognized equal masks (vadd plus one vcadd). [Physical reduction-tree regression](https://github.com/hw-native-sys/PTOAS/blob/ec8d1810b6c107b391a88305a14bde4c334cb21c/test/lit/vpto/vpto_combine_reductions.pto#L12) covers init/grouped trees and different-mask negatives.

**Performance inference:** for a dead-old-leaf k-chunk tree, k reductions become one while k-1 elementwise combines remain. The two-leaf regression removes one reduction instruction. Floating-point addition order changes require source permission and numerical validation; this is not a free strict-IEEE rewrite.

### C05: preserve simple predicates so later combines can see equality

**O:** [Physical predicate simplification](https://github.com/hw-native-sys/PTOAS/blob/ec8d1810b6c107b391a88305a14bde4c334cb21c/lib/PTO/Transforms/VPTO/VPTOMaskSimplify.cpp#L108) and [Prefix predicate simplification](https://github.com/hw-native-sys/PTOAS/blob/ec8d1810b6c107b391a88305a14bde4c334cb21c/lib/PTO/Transforms/VPTO/VPTOMaskSimplify.cpp#L389) fold all-true select, redundant predicate interleave/deinterleave, boolean identities, and supported static-prefix combinations. They require correct predicate type/granularity and known pattern structure. A deinterleave is simplified to prefix predicates only if its result actually remains prefix-shaped; [Prefix deinterleave positive/negative regression](https://github.com/hw-native-sys/PTOAS/blob/ec8d1810b6c107b391a88305a14bde4c334cb21c/test/lit/vpto/vpto_mask_simplify_pdintlv_prefix.pto#L44) contains both positive and retained cases.

**I — bridge help:** retain truthful ALL/VL/H/prefix intent, reuse an actually equal predicate, and eliminate only proven boolean/vector/scalar wrapper round trips. Avoid turning a known mask into memory-loaded bytes and compares unless the semantics require materialization. [Bridge direct predicate conversion](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Conversion/HIVMAVEToPTOASVMI/HIVMAVEToPTOASVMI.cpp:2362) illustrates the existing direct-predicate route. Never replace a partial predicate by ALL to make a matcher succeed; compare seed and inactive-lane behavior remain part of the contract.

**Performance inference:** fewer predicate operations and potentially fewer all-true selects; subsequent CSE can expose identical masks for C04. **Ordering limit:** global VPTO mask simplification happens during emission preparation, after the earlier tile-local expdif lifecycle. It must not be credited with automatically enabling an expdif match that already ran earlier.

### C06: make continuity provable, but recognize the physical stream prerequisite

**O:** [Physical store-stream recognizer](https://github.com/hw-native-sys/PTOAS/blob/ec8d1810b6c107b391a88305a14bde4c334cb21c/lib/PTO/Transforms/VPTO/VPTOStatefulStreamFusion.cpp#L91) recognizes a physical chain:

~~~text
init_align -> vstus(state, base) -> ... -> vstas(final_state, final_base, 0)
~~~

State/base links must match, per-operation advances must be constant and sum without overflow, and intermediate alignment/base results have constrained one-use chains. Loads have a corresponding vldas/vldus chain. Adjacent fusion requires equal pointer types and a proven address difference equal to the prior stream's advance; only allowed pure setup may intervene.

For loop hoisting, [Loop continuity test](https://github.com/hw-native-sys/PTOAS/blob/ec8d1810b6c107b391a88305a14bde4c334cb21c/lib/PTO/Transforms/VPTO/VPTOStatefulStreamFusion.cpp#L499) requires constant bounds/positive step, at least two iterations, and address coefficient times loop step exactly equal to stream advance. [Stream alias/effect checks](https://github.com/hw-native-sys/PTOAS/blob/ec8d1810b6c107b391a88305a14bde4c334cb21c/lib/PTO/Transforms/VPTO/VPTOStatefulStreamFusion.cpp#L596) uses pointer roots and static root addresses, not a general range-aware alias analysis. Unknown effects, uncertain roots, or other stateful operations block fusion. [Loop-carried stream-state rewrite](https://github.com/hw-native-sys/PTOAS/blob/ec8d1810b6c107b391a88305a14bde4c334cb21c/lib/PTO/Transforms/VPTO/VPTOStatefulStreamFusion.cpp#L687) carries base/alignment state through the loop and places final vstas after it.

**Schematic eligible output:**

~~~text
state = init_align()
for each contiguous full chunk:
    state, base = vstus(chunk, state, base)
vstas(state, base)
~~~

**O — bridge position:** [Bridge allowed continuity/unaligned attributes](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Conversion/HIVMAVEToPTOASVMI/HIVMAVEToPTOASVMI.cpp:943) accepts existing `hivm.is_continuous` and unaligned markers; [Bridge logical store emission](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Conversion/HIVMAVEToPTOASVMI/HIVMAVEToPTOASVMI.cpp:4525) emits logical store operations without a stream-state contract. The native path has [Native continuity consumer](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Dialect/HIVMAVE/Transforms/HoistVstas.cpp:79), but [NPU-IR bridge/native split](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Tools/bishengir-compile/regbase/PassPipeline.cpp:299) returns before that pass. Upstream stream fusion does not read `hivm.is_continuous`; it reconstructs eligible physical streams.

**I — bridge help today:** retain SCF loops and clear addptr/index arithmetic, avoid unnecessary helper/effect boundaries, and generate correct logical memory operations that PTOAS can lower into streams. Earlier alignment, extent, alias-range and continuity proofs could help through a supported consumer. **Additional PTOAS work:** dynamic-loop continuity or general disjoint-range knowledge is not currently enabled simply by attaching a marker.

**F05 limit (O/Q):** one-point physical `vsts` is not recognized by this stream parser. [Stateful-stream expected physical output](https://github.com/hw-native-sys/PTOAS/blob/ec8d1810b6c107b391a88305a14bde4c334cb21c/test/lit/vmi_new/vmi_to_vpto_stateful_stream_fusion.pto#L136) lines 170-176 explicitly retain eight group point stores outside the continuous stream. Thus the existence of stream fusion does not close the singleton continuous ONEPT case. Reproduce the exact dqkwg store sequence and flush behavior first.

**Evidence/tests:** [Stateful-stream regression input](https://github.com/hw-native-sys/PTOAS/blob/ec8d1810b6c107b391a88305a14bde4c334cb21c/test/lit/vmi_new/vmi_to_vpto_stateful_stream_fusion.pto#L58) includes constant-loop positives and dynamic-trip, wrong-advance, one-iteration, and aliasing negatives. [Stateful-stream expected physical output](https://github.com/hw-native-sys/PTOAS/blob/ec8d1810b6c107b391a88305a14bde4c334cb21c/test/lit/vmi_new/vmi_to_vpto_stateful_stream_fusion.pto#L136) shows three eligible adjacent stores sharing one init and one final flush, and preserves side-effect/different-root boundaries.

**Performance inference:** remove repeated stream initialization/flush and some address handling. Hoisting adds loop-carried state and can increase live registers; compare both instruction count and pressure.

### C07: keep already fused expdif; do not assume raw sub/exp is re-fused

**O:** [Physical exp-difference fusion](https://github.com/hw-native-sys/PTOAS/blob/ec8d1810b6c107b391a88305a14bde4c334cb21c/lib/PTO/Transforms/TileFusion/PTOVexpdifFusion.cpp#L30) matches physical f32 vsub directly feeding vexp, with identical full physical vector types, **the same SSA mask**, one use of the subtraction, and vexp nested in a real fusion region. It emits VexpdifOp with the selected ODD part. This pure-SSA matcher does not impose the memory-motion guard from C01.

~~~text
vsub(a, b, mask) -> vexp(..., mask)  ->  vexpdif(a, b, mask)
~~~

**O — bridge position:** [Bridge fused AVE exp-difference conversion](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Conversion/HIVMAVEToPTOASVMI/HIVMAVEToPTOASVMI.cpp:3968) already converts AVE VFExpdifOp directly into VMI vexpdif, tested by [Bridge fused softmax-arithmetic regression](/home/m00967009/Workspace/AscendNPU-IR/bishengir/test/Conversion/HIVMAVEToPTOASVMI/preflight-softmax-arithmetic.mlir:5). Keep that operation intact. Native [Native AVE arithmetic fusion](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Dialect/HIVMAVE/Transforms/CombineAVEOPs.cpp:39) can recognize sub/exp before the bridge. A separately bridged VMI vsub/vexp chain does not automatically enter [Physical exp-difference fusion](https://github.com/hw-native-sys/PTOAS/blob/ec8d1810b6c107b391a88305a14bde4c334cb21c/lib/PTO/Transforms/TileFusion/PTOVexpdifFusion.cpp#L30): the default raw-VMI route has no tile fusion region, and the pass checks the region explicitly even if invoked directly.

**I — additional integration options, ranked:** (1) preserve existing semantically valid fused AVE operations; (2) add a suitably verified semantic VMI combine or extend supported pipeline wiring in PTOAS; (3) introduce a genuine lower-level fusion-region contract for bridge output. Simply setting the CLI flag or adding an arbitrary wrapper is insufficient.

**Evidence/tests:** [Exp-difference mask regression](https://github.com/hw-native-sys/PTOAS/blob/ec8d1810b6c107b391a88305a14bde4c334cb21c/test/lit/tile_fusion/pto_vexpdif_fusion_mask.pto#L19) checks same-mask and mask-mismatch behavior; [Exp-difference multiple-use negative regression](https://github.com/hw-native-sys/PTOAS/blob/ec8d1810b6c107b391a88305a14bde4c334cb21c/test/lit/tile_fusion/pto_vexpdif_fusion_multiuse.pto#L48) retains multiple-use subtraction; [Exp-difference F16 negative regression](https://github.com/hw-native-sys/PTOAS/blob/ec8d1810b6c107b391a88305a14bde4c334cb21c/test/lit/tile_fusion/pto_vexpdif_fusion_f16.pto#L41) retains the unsupported F16 pattern. The matcher does not consult a source numerical-permission attribute; preserve the source fusion contract and test exceptional values when extending it.

**Performance inference:** two physical arithmetic operations become one per eligible chunk; an intermediate value is removed. Q2 dqkwg already contains fused expdif, so preserving it avoids a regression rather than proving a newly missed fusion.

### C08: explicitly map existing AVE fused multiply-add

**O:** [VMI fused multiply-add contract](https://github.com/hw-native-sys/PTOAS/blob/ec8d1810b6c107b391a88305a14bde4c334cb21c/include/PTO/IR/VMIOps.td#L1782) and [Explicit fused multiply-add lowering](https://github.com/hw-native-sys/PTOAS/blob/ec8d1810b6c107b391a88305a14bde4c334cb21c/lib/PTO/Transforms/VMIToVPTO/VMIToVPTOPatternInternals7.cpp#L659) support explicit `vmula acc,lhs,rhs` with optional governing mask. A search of the audited transforms found direct vmula lowering but no general separate-multiply-plus-add formation pass. [Explicit-vmula layout regression](https://github.com/hw-native-sys/PTOAS/blob/ec8d1810b6c107b391a88305a14bde4c334cb21c/test/lit/vmi_new/vmi_layout_assignment_fma.pto#L9) starts with explicit vmula: it tests layout, not discovery of an FMA.

Native [Native AVE arithmetic fusion](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Dialect/HIVMAVE/Transforms/CombineAVEOPs.cpp:39) and [Native fusion gating](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Dialect/HIVMAVE/Transforms/CombineAVEOPs.cpp:120) can form AVE VFMula under the Ascend950 FMA option. [Bridge arithmetic conversion registration](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Conversion/HIVMAVEToPTOASVMI/HIVMAVEToPTOASVMI.cpp:4720) handles separate add/mul and fused expdif but contains no VFMula conversion. Therefore enabling an earlier fusion is not sufficient if the bridge cannot translate its result.

**I — concrete bridge opportunity:**

~~~text
Already authorized AVE vmula(acc, a, b, mask)
    -> VMI vmula(acc, a, b, mask)
    -> physical vmula
~~~

Translate operand order, types, masks, and fused numerical semantics explicitly. Do not silently replace a strict separately rounded mul/add pair by FMA. Native fusion pattern implementation is not, by itself, proof that arbitrary source masks or numerical policies permit contraction.

**Performance inference:** potentially one multiply-add instruction instead of separate multiply/add per chunk. **Q:** no exact generated Q2 VFMula occurrence or measured opportunity was established here. This is a conditional architectural finding, not an additional confirmed in-scope Q2 failure.

### C09: tile loop fusion requires a supported structured entry point

**O:** [Tile-fusion recognized computation and operands](https://github.com/hw-native-sys/PTOAS/blob/ec8d1810b6c107b391a88305a14bde4c334cb21c/lib/PTO/Transforms/TileFusion/FusionOpSemantics.cpp#L18) recognizes supported tile elementwise/scalar operations, exp, row broadcasts, and row/column reductions, with TileBuf inputs/outputs or DPS destinations. [Unknown-operation classification](https://github.com/hw-native-sys/PTOAS/blob/ec8d1810b6c107b391a88305a14bde4c334cb21c/lib/PTO/Transforms/TileFusion/FusionOpSemantics.cpp#L99) classifies unknown operations, including ordinary VMI ops, as boundaries rather than tile computations.

[Tile-fusion candidate/domain/cost checks](https://github.com/hw-native-sys/PTOAS/blob/ec8d1810b6c107b391a88305a14bde4c334cb21c/lib/PTO/Transforms/TileFusion/PTOFusionPlan.cpp#L386) uses a conservative DAG-greedy strategy. Candidate groups need proven compatible iteration domains, dataflow connections, no disallowed intervening boundaries, and positive estimated benefit. It can schedule connected branches together; initial adjacency is not universally required. It penalizes excessive live tiles/VF parameters. Dynamic shape expressions need equivalence proof, not just similar-looking syntax.

[Fusion-region interface construction](https://github.com/hw-native-sys/PTOAS/blob/ec8d1810b6c107b391a88305a14bde4c334cb21c/lib/PTO/Transforms/TileFusion/PTOFusionRegionGen.cpp#L318) constructs real inputs/outputs and yields. After expansion, [Post-lowering loop fusion](https://github.com/hw-native-sys/PTOAS/blob/ec8d1810b6c107b391a88305a14bde4c334cb21c/lib/PTO/Transforms/TileFusion/PTOLowLevelLoopFusion.cpp#L257) requires compatible loop depths, equivalent bounds/steps, identical loop attributes, supported leaf bodies and legal effects/dependencies. [Loop-fusion region traversal](https://github.com/hw-native-sys/PTOAS/blob/ec8d1810b6c107b391a88305a14bde4c334cb21c/lib/PTO/Transforms/TileFusion/PTOLowLevelLoopFusion.cpp#L960) only walks fusion regions and skips operation-library helper definitions.

**I — bridge help:** if tile fusion is an objective, preserve supported structured HIVM computation **before** reducing it to AVE register instructions, and translate valid shape, physical shape, group/reduction axis, destination identity and escaping outputs to tile-native PTO. Let PTOAS planning/scheduling establish legal groups.

This is a larger frontend architecture choice, not a local AVE opcode mapping. Existing CCE/PTODSL template calls may still expand to low-level VMI; their presence does not prove eligible tile computation exists. A future explicitly verified low-level bridge-region integration is another option, but requires both an interface and pipeline work.

**Do not** attach planner-output `pto.fusion.group_id/order` to arbitrary VMI ops or invent empty-yield regions. These attributes do not prove domain/dependence legality, and later store deletion relies on the true observable frontier.

**Evidence/tests:** [Full tile-fusion lifecycle regression](https://github.com/hw-native-sys/PTOAS/blob/ec8d1810b6c107b391a88305a14bde4c334cb21c/test/lit/tile_fusion/op_fusion_backend_lifecycle_level3.pto#L49) starts from three connected tile adds. [Unproven dynamic-domain negative regression](https://github.com/hw-native-sys/PTOAS/blob/ec8d1810b6c107b391a88305a14bde4c334cb21c/test/lit/tile_fusion/fusion_plan_dynamic_shape_negative.pto#L9) retains unproven dynamic-domain work; [Reduction output-domain negative regression](https://github.com/hw-native-sys/PTOAS/blob/ec8d1810b6c107b391a88305a14bde4c334cb21c/test/lit/tile_fusion/fusion_plan_reduce_output_domain_negative.pto#L9) rejects a reduction-output-domain mismatch; [Call-boundary negative regression](https://github.com/hw-native-sys/PTOAS/blob/ec8d1810b6c107b391a88305a14bde4c334cb21c/test/lit/tile_fusion/op_scheduling_negative_call_boundary.pto#L9) preserves a call boundary.

**Performance inference:** eligible fused loops can avoid repeated loop/predicate/address overhead and create opportunities for intermediate-memory elimination. Different reduction domains, dependencies, or pressure can make fusion unprofitable or illegal.

### C10: tile-local predicate and memory elision need exact observable outputs

**O:** [Fusion-local predicate elision](https://github.com/hw-native-sys/PTOAS/blob/ec8d1810b6c107b391a88305a14bde4c334cb21c/lib/PTO/Transforms/TileFusion/PTOFusionPredicateElision.cpp#L170) removes redundant physical plt_b8/b16/b32 operations inside a fusion region when a dominating prior candidate has equivalent scalar input and the same width. It replaces both mask and scalar-out results; it is not a general VMI predicate analysis.

[Fusion-local load/store matching](https://github.com/hw-native-sys/PTOAS/blob/ec8d1810b6c107b391a88305a14bde4c334cb21c/lib/PTO/Transforms/TileFusion/PTOFusionLoadStoreElision.cpp#L257) forwards physical vlds from matching vsts when base/offset/mask relationships agree and load consumers expose a compatible mask. Unsupported effects invalidate tracking. It can remove overwritten or dead internal stores, but [Observable output frontier handling](https://github.com/hw-native-sys/PTOAS/blob/ec8d1810b6c107b391a88305a14bde4c334cb21c/lib/PTO/Transforms/TileFusion/PTOFusionLoadStoreElision.cpp#L437) retains observable materialization for yielded outputs or later uses. Region body/loop structure must satisfy the supported straight-line contracts.

**Schematic eligible effect:**

~~~text
Stage 1: x = compute(...); store x -> temporary
Stage 2: y = load temporary; consume y

After legal region-local forwarding: consume x
~~~

**I — bridge help if integrating with tile fusion:** preserve exact shared addressing and predicates, internal-buffer identity, and every escaping value or memory effect in the region interface. Provide actual read/write effect models and allocation/range/lifetime facts where understood. An opaque CCE call or unknown effect legitimately prevents elimination. Do not delete barriers or hide a store to make a temporary appear internal.

**Evidence/test:** [Full tile-fusion lifecycle regression](https://github.com/hw-native-sys/PTOAS/blob/ec8d1810b6c107b391a88305a14bde4c334cb21c/test/lit/tile_fusion/op_fusion_backend_lifecycle_level3.pto#L49) checks the lifecycle in stages: lines 63-75 retain the fused loop's three predicate calculations; lines 80-94 remove redundant predicates; lines 99-109 remove intermediate stores/reloads while preserving the output store; lines 111-128 flatten the wrapper.

**Performance inference:** fewer UB writes/reads and predicate instructions. This can remove intermediate UB traffic, but the existing plain-VMI bridge has no automatic entry into it. Its actual significance requires measurement.

### C11: planner-directed unrolling is optional and requires divisible static loops

**O:** [Fusion-region unroll implementation](https://github.com/hw-native-sys/PTOAS/blob/ec8d1810b6c107b391a88305a14bde4c334cb21c/lib/PTO/Transforms/TileFusion/PTOUnrollAfterLoopFusion.cpp#L74) consumes region row/column factors (column preferred), targets only the innermost loop, requires a constant positive trip count divisible by a factor greater than one, and does not construct a cleanup tail. It resets the consumed factor. Outer unrolling is deliberately avoided because it can disrupt later load/store elision.

**I — bridge help:** preserve loop bounds/steps and equivalent valid-shape expressions for supported structured paths. Let the PTOAS planner/cost model select factors; do not copy an unrelated NPU-IR unroll choice into planner-owned attributes without a compatible contract. Without a genuine fusion region and enabled unroll path there is no benefit. [A5 fusion configuration](https://github.com/hw-native-sys/PTOAS/blob/ec8d1810b6c107b391a88305a14bde4c334cb21c/tools/ptoas/ptoas_pipeline.cpp#L1138) warns when costmodel-generated factors have no enabled consumer.

**Performance inference:** possible loop/address overhead reduction and more instruction scheduling freedom, balanced against live-register growth and code size. Lower instruction count per iteration alone does not prove lower kernel latency.

## Facts worth preserving before the bridge

| Fact from earlier NPU-IR passes | Existing consumer / missing consumer work | Fusion helped |
|---|---|---|
| Logical lane map, split/pack provenance, group and reduction axis | Translate into supported logical VMI grouping/layout semantics; do not copy physical AVE layouts blindly | C02/C03; preserves existing F08 benefit |
| Allocation identity, view offset/range, accessible extent, lifetime | C01 already uses AliasAnalysis; physical-read proofs need extent/alignment; stream alias helper would need broader range support | C01/C06; future C10 |
| Effective address alignment, offset divisibility, analyzable bounds | [PTOAS physical-read proof inputs](https://github.com/hw-native-sys/PTOAS/blob/ec8d1810b6c107b391a88305a14bde4c334cb21c/lib/PTO/Transforms/VMIToVPTO/VMIToVPTOMemoryInternals.cpp#L17) and memory selection already consume supported SSA/memref proofs; custom attrs require explicit support | Select stream/block forms before C06; avoid unnecessary setup |
| Equal predicate/prefix facts and active-lane domain | Materialize canonical predicates and shared SSA; current helpers do not consume arbitrary equivalence metadata | C04/C05; C07 requires exact shared SSA earlier in tile lifecycle |
| Continuous-stream proof plus iteration advance/trip/tail contract | Native has [Native continuity consumer](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Dialect/HIVMAVE/Transforms/HoistVstas.cpp:79); current upstream C06 reconstructs only supported physical/loop forms | C06; not sufficient for point-store F05 |
| Numerical reassociation/contraction/special-value permission | Establish legality before emitting VMI reduction/FMA; do not assume physical combines inspect absent flags | C04/C07/C08 |
| Tile valid shapes, physical shapes, destination/write identities, escaping outputs | Actual tile planner and FusionRegionGen input contracts | C09/C10/C11; requires structured entry point |

[NPU-IR vector alias annotation](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Dialect/HIVM/Transforms/AnnotateVFAlias.cpp:204) demonstrates an existing upstream alias-annotation producer, but its execution in the normal bridge path was not established. Exporting a fact requires a proof valid at its use: reused arena buffers need ranges and lifetimes rather than unconditional global noalias. [Numerical and load-safety policy options](https://github.com/hw-native-sys/PTOAS/blob/ec8d1810b6c107b391a88305a14bde4c334cb21c/include/PTO/Transforms/Passes.td#L1345) also makes clear that numerical fast paths and physical load safety are distinct policies; a fusion hint cannot override either contract.

## Recommended implementation order (no implementation performed)

| Priority | Concrete action | Required validation before claiming benefit |
|---|---|---|
| P0 | Pin and migrate the bridge/PTOAS operation interfaces, including old scalar load and memory distribution syntax | Parser/verifier and bridge regression corpus on the same built revisions |
| P1 | Express vector scalar-address broadcast with singleton logical load plus broadcast where needed; use singleton load alone for F03 | Exact F02 64-lane BF16->F32 chain, F03 ordering, offsets, masks, multiuse and intervening-write negatives; compare BRC/scalar/vdup/shuffle counts |
| P1 | Preserve shared predicates, direct reductions, legal mathematical permissions, and pure SSA helpers | Equal/unequal mask tree cases; emitted vcadd/vcgadd counts; numerical and exceptional-value comparison |
| P1 | Retain logical lane/packing intent and memory proofs for existing layout/stream machinery | F08 and packed-store tests; no new memory overread; static-loop stream setup/flush counts, alias and tail negatives |
| P1 conditional | Add a verified mapping for already fused AVE vmula | Source occurrence, operand order, active/inactive lanes, rounding/contracted arithmetic, explicit vmula emission |
| P2 | Define and consume richer group, alias-range and dynamic-continuity contracts | Positive cases previously blocked by the precise guard, paired with overlap/noncontiguous/dynamic negatives |
| P2 architectural | Evaluate a tile-native structured entry before AVE lowering, or a separately verified low-level region interface | Real planner/region lifecycle, shape/domain negatives, escaping-output preservation, end-to-end Q2 correctness and latency |

Cleanest division of responsibility: earlier NPU-IR passes prove facts while information exists; the bridge translates semantic operations and supported facts; PTOAS owns layout choice, combination, scheduling and target instruction selection. Bridge-side normalization is justified when it recovers logical meaning from AVE physical carriers, as F08 already does.

## Reproduction and measurement plan

1. Build and record a paired NPU-IR/PTOAS toolchain; record gitlinks, actual checkouts, binary hashes, backend, A5/CANN version, fusion flags and numerical/load-safety policies. A source revision does not establish a prebuilt binary's provenance.
2. Start with the cited tests' own RUN lines. Positive output alone is insufficient: run multiple-use, mask-mismatch, aliasing-store, unknown-effect, noncontiguous, dynamic-bound and escaping-output negatives for the relevant pass.
3. For vector bridge output, dump before/after pre-assignment combining, assigned/folded layouts, VMI-to-VPTO/stream fusion, and final mask/reduction optimization. Check direct-defining-op and same-block conditions at the stage the matcher runs.
4. For tile experiments, dump planned group/order, generated regions, expanded loops, predicate elision, memory elision and flattening. Confirm the input reached the tile mainline and regions have correct outputs. A fusion flag with no eligible tile computation is not evidence of fusion.
5. Run exact Q2 specializations used in the companion assessment, then compare native/bridge numerical outputs. Time paired runs on the same hardware with warmup and repetitions. Report kernel latency with emitted operation counts and register/spill evidence; do not sum hypothetical per-pass savings.

Expected isolated comparisons:

| Case | Compare in final physical output |
|---|---|
| Scalar UB load/broadcast | scalar load + vdup versus vector BRC; check extra consumers do not disappear |
| Grouped slot broadcast | E2B packet versus repeated BRC/selector fallback for the exact shape |
| Widening + arithmetic + store | standalone ensure materialization/interleave versus direct supported store distribution |
| Multi-chunk reduction | old reduction leaves versus source combines plus one reduction; check external users |
| Contiguous stream loop | initialization/flush per iteration versus loop-carried state and one final flush |
| Fused expdif/vmula | surviving physical operation count per chunk and numerical policy |
| Tile temporary chain | actual UB reads/writes and predicate instructions removed while final output remains |

## Verification status and open questions

**O:** pass definitions, actual matchers, CLI defaults, pipeline gates, bridge producers, native preceding passes and listed regression source/contracts were inspected. The report distinguishes support code from fresh execution. Documentation verification checked 85 unique links for target existence and line bounds, all 11 finding sections, balanced code fences/tables, whitespace, and the companion-report crosslink. Separate source reviews checked matcher conditions and pipeline gating; identified citation/example corrections were applied.

**Not established:** a fresh latest-master test run, complete current Q2 compile/runtime triggering matrix, device/simulator correctness or per-fusion timing. No build or benchmark was run for this audit. No compiler source fix, submodule update or commit was made.

Open questions to resolve with concrete inputs:

- Does the proposed BF16 singleton-load/broadcast/widen chain eliminate F02's old layout failure on the paired latest build?
- Which Q2 reductions expose a multi-chunk combine tree after the actual bridge's result-shape restoration and later canonicalization?
- Does exact dqkwg F05 lower to point stores or any eligible stateful stream on the migrated pipeline, and what native flush contract must be preserved?
- Which structured Q2 helpers can remain supported tile computations, rather than opaque calls or already lowered VMI, without undoing NPU-IR tiling/synchronization decisions?
- Which memory/range facts are already recoverable from SSA and which need a new jointly defined bridge/PTOAS interface?
- Does an actual generated in-scope AVE vmula exist under an authorized numerical policy, rather than only in an optional native fusion pass?
- Are any apparent instruction savings offset by register pressure, repeated rematerialization, spills, or lost overlap?
