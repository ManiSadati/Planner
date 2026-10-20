# AVE–PTOAS VMI: slide outline — 2026-10-20

Seven main slides, including an overview table, followed by two optional appendix groups. The audience is PTOAS project owners; links are review references and can move into presentation notes. Scope is source and compile evidence on the report's named revisions, not a device-performance claim. [scope / E00](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:17).

Code examples are short, adapted excerpts from the cited sources, not standalone programs; SSA names and unrelated attributes are simplified. Paired forms show separate stages, and rejected IR is labeled. `#ub` / `#gm` abbreviate HIVM memory-space attributes; `#map` is the source layout. Ellipses occur only in explicitly abbreviated shuffle indices.

## Slide 1 — Add masks to floating conversion

**On slide**

- **Example:** Q2 cumsum's H-masked `ave.hir.vextf` rejects; VMI `extf` / `vcvt` have no conversion-mask operand. [G01](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:47) · [E01](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:68).
- **PTOAS request:** Add masked widening/narrowing with defined inactive-lane behavior; preserve rounding and lane placement. [G01](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:47) · [E01](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:68).
- **Acceptance:** The captured H-mask graph lowers correctly alongside the existing guarded VL1 cases. [G01](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:47) · [E01](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:68).

**Code example — captured AVE and closest declared VMI form**

```mlir
%half = ave.hir.pge <H> {functionType = #ave.func_dist_type<pb32>}
    : vector<64xi1>
%wide = ave.hir.vextf %x, <part_even>, %half
    : vector<64xf16>, vector<64xf32>, vector<64xi1>
// Existing unmasked VMI form; NOT successful conversion of the rejected input.
%wide = pto.vmi.extf %x
    : !pto.vmi.vreg<64xf16> -> !pto.vmi.vreg<64xf32>
```

AVE carries `%half`; the available VMI form has no conversion-mask operand. [G01](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:47) · [E01](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:68) [Captured AVE](/home/m00967009/Workspace/Logs/q2_ave_requirements_oct19/captures/cumsum_scalar_False/before-bridge.mlir:33).

**Speaker notes**

The example is a captured 64-lane F16-to-F32 graph with a same-mask store. Existing ALL and narrowly proven VL1 paths are supported; this is not a claim that all conversion is unavailable. Arithmetic-mask preservation is a separate issue. Numerical/FP-status validation and any timing attribution remain outstanding. [G01](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:47) · [E01](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:68).

## Slide 2 — Treat predicates as memory data

**On slide**

- **Example:** Called Q2 solve helpers spill/reload PB32 predicates in UB; VMI memory data/results are numeric vectors. [G03](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:82) · [E03](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:101).
- **PTOAS request:** Add predicate-valued load/store with PB granularity and explicit packed-address units, lowered to existing physical predicate-memory instructions. [G03](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:82) · [E03](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:101).
- **Acceptance:** Validate the captured PB32 store/reload and its controlling mask before generalizing widths and addresses. [G03](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:82) · [E03](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:101).

**Code example — predicate data in a called Q2 solve helper**

```mlir
ave.hir.masked_store <NORM_B8> %dst[%c0], %active, %predicate
    {functionType = #ave.func_dist_type<pb32>}
    : memref<8xi1, #map, #ub>, vector<64xi1>, vector<64xi1>
```

`%predicate` is the value being stored, while `%active` independently controls the write; VMI needs to represent both roles. [G03](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:82) · [E03](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:101) [Captured store](/home/m00967009/Workspace/Logs/q2_ave_requirements_oct19/captures/solve/before-bridge.mlir:1515).

**Speaker notes**

A mask that controls a numeric store is different from a mask stored as data. Physical support already exists. The byte-carrier workaround needs explicit repair, which the inspected default flows omit; its raw output is not proof of packed-memory equivalence. Other PB widths, tails and indexed addresses need validation. [G03](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:82) · [E03](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:101).

## Slide 3 — Support explicit signed `vmins` / `vmaxs`

**Category:** Operation and semantic coverage — explicit signedness interface.

**On slide**

- **AVE behavior:** `ave.hir.vsmins` / `ave.hir.vsmaxs` specify signed ordering on `i32` vector/scalar operands. Current VMI expresses signed ordering through matching `si32` operands, requiring signedness casts in the bridge IR. [G09](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:163) · [E09](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:181).
- **PTOAS request:** Add explicit signed forms of `pto.vmi.vmins` / `pto.vmi.vmaxs`, or a signedness attribute, accepting signless `i32` vector/scalar operands and results. Support constant and runtime scalars; lower to the signed `llvm.hivm.vmins.s.x.v64i32` / `llvm.hivm.vmaxs.s.x.v64i32` intrinsics. [G09](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:163) · [E09](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:181).
- **Acceptance:** Preserve signed ordering and masks without bridge-added signedness casts; keep unsigned behavior distinct. The demonstrated benefit is simpler IR; no NPU instruction saving is established. [G09](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:163) · [E09](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:181).

**Code example — AVE signed minimum and current VMI workaround**

```mlir
// AVE: the operation specifies signed ordering, including for a runtime scalar.
%min = ave.hir.vsmins %v, %s, %m
    : vector<64xi32>, i32, vector<64xi1>
// Current VMI: signed ordering comes from both operand types.
%sv = pto.vmi.bitcast %v : !pto.vmi.vreg<64xi32> -> !pto.vmi.vreg<64xsi32>
%ss = builtin.unrealized_conversion_cast %s : i32 to si32
%r = pto.vmi.vmins %sv, %ss, %pm
    : !pto.vmi.vreg<64xsi32>, si32, !pto.vmi.mask<64xpred>
      -> !pto.vmi.vreg<64xsi32>
%min = pto.vmi.bitcast %r : !pto.vmi.vreg<64xsi32> -> !pto.vmi.vreg<64xi32>
```

An explicit signed VMI form would carry AVE's intent directly and remove these carrier casts; `vmaxs` uses the same signedness pattern. [G09](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:163) · [E09](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:181) [AVE testcase](/home/m00967009/Workspace/AscendNPU-IR/bishengir/test/Conversion/HIVMAVEToPTOASVMI/unsupported-signed-vector-scalar-dynamic.mlir:4), [typed VMI probe](/tmp/ptoas-signed-minmax-20261020/signedness-casts.pto:5).

**Side note — normalization bug:** A separate focused probe normalizes the vector from `i32` to `ui32` but leaves the scalar `i32`, causing a verifier error. Normalize both operands consistently while preserving the intended signedness. [G09](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:163) · [E09](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:181) [Actual rejection](/tmp/ptoas-signed-minmax-20261020/signless-normalization.log:1).

**Speaker notes**

The main request is an operation-level signedness interface; signed/unsigned arithmetic already works with explicit `si32` / `ui32` vector and scalar types, including runtime scalars. The tested same-width casts disappear from LLVM. The bridge's constant-only gate remains a separate restriction. The normalization defect has matching upstream source at `739480c309c2`, but the probe used the local compiler, not a freshly built upstream compiler. Exact Q2 `vsmins` / `vsmaxs` occurrence is unconfirmed; relevance here is the focused bridge testcase. [G09](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:163) · [E09](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:181).

## Slide 4 — Integrate the mask fix; recheck scope placement

**On slide**

- **Pinned arithmetic-mask loss:** Integrate the existing upstream mask-preserving lowering and matching layout handling; verify predicates reach physical arithmetic. [G02](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:70) · [E02](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:80).
- **Historical D1200 scope failure:** Reproduce on the supported pair; if still present, repair nested predicate placement/rematerialization and add a downstream regression. [G10](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:183) · [E10](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:193).
- **Boundary:** Upstream fixes arithmetic masks; current-upstream closure of the saved scope failure is unverified. [G02](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:70) · [E02](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:80); [G10](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:183) · [E10](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:193).

**Code example — masked arithmetic from dqkwg**

```mlir
%y = ave.hir.vexp %x, %half
    : vector<64xf32>, vector<64xi1>
```

The captured H predicate must survive lowering; the pinned unary helper builds its legacy operation without that predicate. [G02](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:70) · [E02](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:80) [Captured operation](/home/m00967009/Workspace/Logs/q2_ave_requirements_oct19/captures/reused_dqkwg_H2_B4/before-bridge.mlir:983).

**Code example — zero constant in the saved failing D1200 case**

```mlir
%zeros = "pto.vmi.constant"()
    <{value = dense<0.0> : tensor<64xf32>}>
    : () -> !pto.vmi.vreg<64xf32>
```

Physical lowering introduces a mask for this constant, and the saved diagnostic reports that `!pto.mask<b32>` escaping its vector scope. [G10](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:183) · [E10](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:193) [Saved VMI](/home/m00967009/Workspace/Logs/layernorm_compile_oct15/artifacts/bwd-T8192-D1200-silu-BT1-W4-specialized.vmi.mlir:171), [diagnostic](/home/m00967009/Workspace/Logs/layernorm_compile_oct15/artifacts/bwd-T8192-D1200-silu-BT1-W4-specialized.ptoas.log:1).

**Speaker notes**

The arithmetic correction is upstream commit `2503c259`; it is an integration request, not a new missing upstream feature. The saved D1200 diagnostic concerns an escaping `!pto.mask<b32>` introduced during constant lowering. VMI emission success stops before this failure stage. Do not equate it with the separate opt-in bridge barrier-finalizer issue in the appendix. [G02](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:70) · [E02](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:80); [G10](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:183) · [E10](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:193); [G13](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:221) · [E13](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:231).

## Slide 5 — Preserve continuous-store intent in VMI

**Category:** Stateful memory streams — lowering and fusion coverage (G04 / F05).

**On slide**

- **AVE / NPU-IR:** Two continuous `ave.hir.masked_store <ONEPT_B32>` operations write one F32 each per row. In the focused native test, NPU-IR initializes two stream states before the loop, carries them across rows, and completes both streams after the loop. [G04 / E04](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:103) · [Q2 stores](/home/m00967009/Workspace/Logs/q2_ave_requirements_oct19/captures/reused_dqkwg_H2_B4/before-bridge.mlir:1370), [native output](/tmp/ptoas-onept-check-20261020/native-after-hoist.mlir:10).
- **VMI / PTOAS today:** `pto.vmi.vstore` is a unified store interface, not a request for streaming. Its fixed singleton-selection rule chooses a group store, which lowers to independent `1PT_B32` writes; the tested pipeline does not recover the native two-stream pattern. [G04 follow-up: upstream output](/tmp/ptoas-onept-check-20261020/singleton-semantic-pipeline.mlir:24) · [Selection rule](https://github.com/hw-native-sys/PTOAS/blob/739480c309c265feee7703cdde42fe89da60b1d9/lib/PTO/Transforms/VMI/VMILowerUnifiedToLegacy.cpp#L476), [point-store preference](https://github.com/hw-native-sys/PTOAS/blob/739480c309c265feee7703cdde42fe89da60b1d9/lib/PTO/Transforms/VMIToVPTO/VMIToVPTOPatternInternals3.cpp#L259).
- **PTOAS request — interface and lowering:** Add a documented way to express continuous-stream intent on unified `pto.vmi.vstore`, such as an attribute. Preserve that intent through singleton/group-store lowering and select `vstus`/`vstas` streaming when eligible. Define eligibility and safe fallback behavior. [G04 / E04](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:103) · [Observed singleton selection](https://github.com/hw-native-sys/PTOAS/blob/739480c309c265feee7703cdde42fe89da60b1d9/lib/PTO/Transforms/VMI/VMILowerUnifiedToLegacy.cpp#L476).
- **PTOAS request — loop fusion:** Extend fusion to multiple independent output streams in one loop when aliasing, ordering and intervening memory effects permit. The current guard rejects a loop containing another stateful stream. [G04 / E04](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:103) · [Fusion guard](https://github.com/hw-native-sys/PTOAS/blob/739480c309c265feee7703cdde42fe89da60b1d9/lib/PTO/Transforms/VPTO/VPTOStatefulStreamFusion.cpp#L582).

**Acceptance:** For an eligible marked two-output, 32-row loop, initialize both streams before the loop, carry their state across iterations, and complete both after the loop. Preserve the original addresses and values, and add regressions for unsafe fusion cases. [Native reference result](/tmp/ptoas-onept-check-20261020/native-after-hoist.mlir:10), [current upstream comparison](/tmp/ptoas-onept-check-20261020/singleton-semantic-pipeline.mlir:24).

**AVE example — one value per output per row**

```mlir
// Inside the row loop; each value and mask has one lane.
ave.hir.masked_store <ONEPT_B32> %out0[%row], %first, %value0
    {hivm.is_continuous}
    : memref<32xf32, #ub>, vector<1xi1>, vector<1xf32>
ave.hir.masked_store <ONEPT_B32> %out1[%row], %first, %value1
    {hivm.is_continuous}
    : memref<32xf32, #ub>, vector<1xi1>, vector<1xf32>
```

Each output array receives consecutive four-byte values; `hivm.is_continuous` identifies the stream across rows. The probe simplifies the captured rank-zero destination views to indexed arrays. [Q2 stores](/home/m00967009/Workspace/Logs/q2_ave_requirements_oct19/captures/reused_dqkwg_H2_B4/before-bridge.mlir:1373), [native probe](/tmp/ptoas-onept-check-20261020/native-two-onept.mlir:11).

**NPU-IR native lowering — simplified view of the tested result**

```text
state0 = init_align()
state1 = init_align()
for row = 0..31:
    state0, end0 = vstus.post(value0[row], &out0[row], 4 bytes, state0)
    state1, end1 = vstus.post(value1[row], &out1[row], 4 bytes, state1)
vstas(state0, end0)
vstas(state1, end1)
```

Each output keeps its own alignment state between iterations; both final `vstas` completions are outside the loop. This is schematic notation for the emitted intrinsics, not executable syntax. [Native output](/tmp/ptoas-onept-check-20261020/native-after-hoist.mlir:10), [completion operations](/tmp/ptoas-onept-check-20261020/native-after-hoist.mlir:29).

**PTOAS VMI lowering — simplified excerpt of the tested result**

```mlir
// Reductions and construction of the first-lane predicate are omitted.
scf.for %row = %c0 to %c32 step %c1 {
  pto.vsts %sum0, %out0[%row], %first {dist = "1PT_B32"}
      : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
  pto.vsts %sum1, %out1[%row], %first {dist = "1PT_B32"}
      : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
}
```

Each `1PT_B32` writes one F32 value independently; this result has no alignment state carried between iterations and no final stream-completion operations. [VMI probe input](/tmp/ptoas-onept-check-20261020/singleton-loops.pto:28), [upstream physical output](/tmp/ptoas-onept-check-20261020/singleton-semantic-pipeline.mlir:24).

**Speaker notes**

This October 20 follow-up refines the report's earlier conditional G04 finding. The native check ran the local NPU-IR compiler with `--convert-hivmave-to-ave-intrin --hoist-vstas` for Ascend950. The VMI check used a fresh build of unmodified upstream `739480c309c2`, including its semantic pipeline through stateful stream fusion; three existing upstream regression controls passed. These are focused two-output examples, not complete Q2-kernel executions. [Native commands and results](/tmp/ptoas-onept-check-20261020/native-two-onept-results.json:1), [upstream compiler provenance and checks](/tmp/ptoas-onept-check-20261020/results.json:1).

The current selection is deterministic: the requested change is explicit control over store strategy and documented eligibility, not a fix for random lowering. Ordinary singleton writes and physical stateful operations already exist in PTOAS. Standalone `1PT_B32` writes do not require a final `vstas`; they do not create an alignment-buffer stream. The gap is preserving continuous-store intent and covering safe multi-stream loops. A new physical instruction or a new state-carrying VMI operation is not demonstrated necessary. Device correctness and any performance saving remain to validate.

## Slide 6 — Handle full-width AVE reduction results

**Category:** Reduction and broadcast abstractions — G08 / F04.

**On slide**

- **Example and limitation:** Q2 backward layernorm D1200 uses full-width `ave.hir.reduction <add>` results. VMI floating reductions require a singleton result, so the bridge adds `pto.vmi.broadcast` to restore the AVE carrier width. [G08](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:151) · [E08](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:161).
- **PTOAS request:** Keep reduced values singleton through compatible consumers; eliminate broadcasts used only to reconstruct the result carrier, and materialize a broadcast when consumers require it. [G08](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:151) · [E08](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:161).
- **Acceptance:** Preserve AVE's defined first result position and masks; test consumers that need only that position separately from consumers that need replication. [G08](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:151) · [E08](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:161).

**Code example — captured AVE and generated VMI**

```mlir
// AVE retains the vector width; the reduced value is defined at position zero.
%sum = ave.hir.reduction <add>, %x, %all
    : vector<64xf32>, vector<64xi1> -> vector<64xf32>
// Bridge VMI reconstructs the surrounding carrier width.
%one = pto.vmi.reduce_addf %x, %all {reassoc}
    : !pto.vmi.vreg<64xf32>, !pto.vmi.mask<64xpred>
      -> !pto.vmi.vreg<1xf32>
%sum = pto.vmi.broadcast %one
    : !pto.vmi.vreg<1xf32> -> !pto.vmi.vreg<64xf32>
```

The reduction is supported; the extra logical broadcast reconstructs its result carrier, and whether it can be removed depends on the consumers. [G08](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:151) · [E08](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:161) [Captured AVE](/home/m00967009/Workspace/Logs/q2_ave_requirements_oct19/captures/norm_bwd_D1200/before-bridge.mlir:368), [generated VMI](/home/m00967009/Workspace/Logs/q2_ave_requirements_oct19/captures/norm_bwd_D1200/converted.vmi.mlir:603).

**Speaker notes**

AVE's full-width result does not promise a full-width splat. The result-shape difference is confirmed; an unnecessary broadcast in every graph, an extra machine instruction and a runtime penalty are not established. ADD is captured in Q2. The tracker also lists MAX, but its exact Q2 lowering remains unvalidated. This is an abstraction/optimization request, not a claim that PTOAS cannot execute reductions. [G08](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:151) · [E08](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:161).

## Slide 7 — PTOAS VMI requests at a glance

**On slide**

| Category | Topic / detail slide | Requested PTOAS capability | Current evidence / status |
|---|---|---|---|
| Masking and tails | Masked floating conversion — S1, [G01](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:47) | Add conversion masks with defined inactive lanes, rounding and lane placement. | Direct mask operand absent; captured H-mask conversion rejects. [E01](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:68) |
| Operation semantics | Predicate memory — S2, [G03](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:82) | Add predicate-valued load/store with PB granularity and explicit packed-address units. | Q2 PB32 spill/reload established; direct VMI memory payloads are numeric. [E03](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:101) |
| Operation semantics | Signed scalar min/max — S3, [G09](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:163) | Add explicit signed `vmins` / `vmaxs` semantics for signless operands, including runtime scalars. | Typed signed support exists; interface improvement requested. Normalization bug is a separate fix. [E09](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:181) |
| Lowering correctness | Arithmetic masks — S4, [G02](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:70) | Integrate upstream mask-preserving lowering and matching layout handling. | Pinned lowering drops masks; upstream fix exists. [E02](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:80) |
| Lowering correctness | Nested predicate scopes — S4, [G10](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:183) | Reproduce on the supported pair; repair predicate placement/rematerialization if still needed. | Historical D1200 failure; current-upstream persistence unverified. [E10](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:193) |
| Stateful memory streams | Continuous singleton stores — S5, [G04](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:103) | Express and preserve stream intent on `pto.vmi.vstore`; fuse multiple independent streams safely. | Stream primitives exist; tested singleton dispatch emits point stores, and multi-stream fusion is restricted. [Upstream follow-up](/tmp/ptoas-onept-check-20261020/results.json:1), [fusion guard](https://github.com/hw-native-sys/PTOAS/blob/739480c309c265feee7703cdde42fe89da60b1d9/lib/PTO/Transforms/VPTO/VPTOStatefulStreamFusion.cpp#L582) |
| Reduction / broadcast | Reduction result carriers — S6, [G08](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:151) | Retain singleton results through compatible consumers; remove carrier-only broadcasts where legal. | Reduction works; result-shape optimization. ADD captured; exact Q2 MAX unvalidated. [E08](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:161) |
| Layout and data movement | Partial B16 shuffle; broadcast layout; mixed-width predicates — Appendix A, G05–G07 | Support partial B16 results; adapt 64-lane B16 broadcast layouts; adapt B32 predicates for B16 selection. | Existing compositions cover checked graphs; exact upstream closure varies. [E05](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:125), [E06](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:137), [E07](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:149) |
| Bridge integration / contracts | Integer loads; BF16 wrappers; barrier lifetimes; scalar consumers — Appendix B, G11–G14 | Document/test scalar-load, BF16-store, barrier-lifetime and lane-to-scalar contracts; coordinate bridge fixes. | No new target instruction established; barrier case conditional, genuine Q2 scalar extraction unconfirmed. [E11](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:207), [E12](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:219), [E13](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:231), [E14](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:243) |

**Speaker notes**

The table covers all 14 report entries. Seven main requests remain separate; the two appendix rows summarize three layout items and four bridge-integration items without merging their implementation requests. Use the detail slides for the code and exact evidence boundaries. Signed min/max follows Slide 3's revised operation-semantics category; its normalization defect remains a side note. G04 uses the fresh upstream follow-up recorded on Slide 5 rather than the report's earlier conditional status.

Rows distinguish new interface/coverage requests, an existing upstream fix to integrate, conditional correctness work and optimization opportunities. They are not a severity or frequency ranking. No per-gap Q2 performance improvement has been measured; logical IR expansions and differing store strategies do not by themselves establish a runtime penalty. [Assessment limits](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:260).

## Appendix A — Layout and optimization review

These are captured mappings or proposed optimization work, not a quantified performance case.

| Entry | Example and specific PTOAS request | Evidence boundary |
|---|---|---|
| G05 — Partial B16 results | D1200 sparse `vintlv` → `vextf`: support partial B16 shuffle-result materialization with correct lane mapping/padding treatment. [G05](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:115) · [E05](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:125) | Widen-first composition already works for checked consumers; general materialization still restricts partial chunks. |
| G06 — Broadcast producer layout | Dqkwg `vload <BRC_B16>` → widening: allow the 64-lane producer to satisfy the consumer layout while retaining scalar-address broadcast behavior. [G06](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:127) · [E06](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:137) | Pinned workaround confirmed; validate the migrated upstream memory interface before claiming a remaining upstream defect. |
| G07 — Predicate granularity | Dqkwg I32 compare → F16 `vsel`: validate/extend B32-to-B16 predicate adaptation to avoid widening both data streams. [G07](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:139) · [E07](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:149) | Related upstream support exists; exact 64-lane sparse path remains unvalidated. |

**Code examples for the optional appendix**

**G05 — sparse split and widening workaround**

```mlir
// AVE: split before widening.
%lo, %hi = ave.hir.vintlv %x, %x
    {layout_change = #ave<layout_change SPARSE>}
    : vector<128xf16>, vector<64xf16>
%a = ave.hir.vextf %lo, <part_even>, %all
    : vector<64xf16>, vector<64xf32>, vector<64xi1>
// Bridge VMI: widen before selecting a full B32 half.
%wide = pto.vmi.extf %x
    : !pto.vmi.vreg<128xf16> -> !pto.vmi.vreg<128xf32>
%a = pto.vmi.shuffle %wide[[0, 1, ..., 63]] // index list abbreviated
    : !pto.vmi.vreg<128xf32> -> !pto.vmi.vreg<64xf32>
```

Widening first avoids the unsupported partial physical B16 shuffle result for this checked consumer graph. [G05](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:115) · [E05](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:125) [Bridge testcase](/home/m00967009/Workspace/AscendNPU-IR/bishengir/test/Conversion/HIVMAVEToPTOASVMI/preflight-sparse-interleave-extf.mlir:3).

**G06 — broadcast-layout workaround on the pinned interface**

```mlir
%full = pto.vmi.vload %src[%c0] {dist_mode = "brc"}
    : !pto.ptr<bf16, ub> -> !pto.vmi.vreg<128xbf16>
%x = pto.vmi.shuffle %full[[0, 0, ..., 0]] // 64 zero indices, abbreviated
    : !pto.vmi.vreg<128xbf16> -> !pto.vmi.vreg<64xbf16>
```

A 128-lane scalar-address broadcast and lane-zero replication supply the required 64-lane value without fixing the producer directly to the problematic consumer layout. [G06](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:127) · [E06](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:137) [Expected bridge output](/home/m00967009/Workspace/AscendNPU-IR/bishengir/test/Conversion/HIVMAVEToPTOASVMI/preflight-brc-b16-load.mlir:3).

**G07 — mixed-width select workaround**

```mlir
// %cmp comes from the I32 comparison; floating-data bitcasts are omitted.
%yes32 = pto.vmi.extui %yes16
    : !pto.vmi.vreg<64xi16> -> !pto.vmi.vreg<64xi32>
%no32 = pto.vmi.extui %no16
    : !pto.vmi.vreg<64xi16> -> !pto.vmi.vreg<64xi32>
%chosen32 = pto.vmi.vsel %cmp, %yes32, %no32
    : !pto.vmi.mask<64xpred>, !pto.vmi.vreg<64xi32>,
      !pto.vmi.vreg<64xi32> -> !pto.vmi.vreg<64xi32>
%chosen16 = pto.vmi.trunci %chosen32 {saturate = "NOSAT"}
    : !pto.vmi.vreg<64xi32> -> !pto.vmi.vreg<64xi16>
```

The current mapping widens both data bit carriers instead of adapting only the predicate for B16 selection. [G07](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:139) · [E07](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:149) [Exact dqkwg regression expectations](/home/m00967009/Workspace/AscendNPU-IR/bishengir/test/Conversion/HIVMAVEToPTOASVMI/preflight-vsel-dqkwg.mlir:5).

**Speaker notes**

Keep these requests separate: result-chunk materialization, broadcast-producer adaptation and predicate conversion require different implementations. Logical shuffle/conversion counts do not prove machine-instruction costs. [G05](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:115) · [E05](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:125); [G06](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:127) · [E06](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:137); [G07](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:139) · [E07](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:149).

## Appendix B — Bridge-owned and conditional work

These entries do not establish missing PTOAS instructions; target requests concern an explicit contract or regression support.

| Entry | Ownership and requested action | Evidence boundary |
|---|---|---|
| G11 — Integer metadata loads | Bridge: lower captured I32 GM `memref.load`. PTOAS: document/test the existing scalar-load contract. [G11](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:197) · [E11](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:207) | Recorded Q2 bridge rejection; no target integer-load defect established. |
| G12 — BF16 store wrappers | Bridge: recognize proven BF16 `PK_B32` carriers. PTOAS: regress dense BF16 stores after wrapper removal. [G12](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:209) · [E12](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:219) | Captured Q2 pattern and F16-only bridge gates; not evidence that PTOAS cannot store BF16. |
| G13 — Barrier lifetimes | Jointly specify/test live-value handling across `VST_VLD`; bridge places barriers without unsafe rematerialization. [G13](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:221) · [E13](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:231) | Conditional recovered-dqkwg experiment with temporary fixes; finalizer is opt-in. |
| G14 — Real scalar consumers | PTOAS: specify/test lane-to-scalar materialization, adding an operation only if needed. Bridge: lower real scalar consumers through it. [G14](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:233) · [E14](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:243) | Focused scalar-return/arithmetic tests; genuine Q2 extraction demand unconfirmed. |

**Code examples for the optional appendix**

**G11 — runtime integer metadata load**

```mlir
%sequence = memref.load %view[%c0]
    : memref<1xi32, strided<[1], offset: ?>, #gm>
```

This Q2 integer load meets the captured metadata requirement but is rejected by the bridge's floating-only scalar-load gate. [G11](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:197) · [E11](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:207) [Captured load](/home/m00967009/Workspace/Logs/q2_ave_requirements_oct19/captures/solve_varlen/before-bridge.mlir:301).

**G12 — BF16 packed-store wrappers**

```mlir
%expanded = builtin.unrealized_conversion_cast %narrow
    : vector<64xbf16> to vector<128xbf16>
%carrier = llvm.bitcast %expanded : vector<128xbf16> to vector<64xf32>
ave.hir.masked_store <PK_B32> %dst[%c0], %all, %carrier
    : memref<64xbf16, #map, #ub>, vector<64xi1>, vector<64xf32>
```

The F32 carrier contains packed BF16 bits, while the current bridge recognizer accepts only the F16 analog. [G12](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:209) · [E12](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:219) [Captured chain](/home/m00967009/Workspace/Logs/q2_ave_requirements_oct19/captures/reused_dqkwg_H2_B4/before-bridge.mlir:899).

**G13 — bridge barrier-finalizer rejection**

```cpp
// Excerpt from rematerializing values after the inserted barrier.
auto rematerialized = rematerializeVMIValueAfterBarrier(
    value, barrier, builder, mapping);
if (failed(rematerialized))
  return barrier.emitError(
      "cannot place VST_VLD barrier without carrying a non-pure VMI "
      "value across it");
```

The opt-in bridge finalizer rejects this lifetime case before establishing any new PTOAS instruction defect. [G13](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:221) · [E13](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:231) [Current rejection code](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Conversion/HIVMAVEToPTOASVMI/HIVMAVEToPTOASVMI.cpp:5255).

**G14 — genuine scalar consumer**

```mlir
%scalar = builtin.unrealized_conversion_cast %value
    : vector<1xf32> to f32
%sum = arith.addf %scalar, %scalar : f32
return %sum : f32
```

The scalar arithmetic cannot be handled by folding the value back into a vector broadcast. [G14](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:233) · [E14](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:243) [Current rejection testcase](/home/m00967009/Workspace/AscendNPU-IR/bishengir/test/Conversion/HIVMAVEToPTOASVMI/preflight-reject-singleton-scalar-extraction.mlir:15).

**Speaker notes**

G11 is a recorded Q2 blocker owned by bridge conversion. G13 must stay separate from the PTOAS-introduced predicate failure on Slide 4. Source-only requirements, unresolved infinity/ABI cases and historical whole-kernel timings are excluded from the main claims. Use the report's limitations before expanding the scope. [G11](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:197) · [E11](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:207); [G13](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:221) · [E13](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:231); [scope / E00](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:17).

## Review handoff

Agree owners and semantic contracts for the main requests; use captured Q2 graphs to evaluate the conditional work. The report's acceptance section distinguishes proposed checks from completed results. All gap claims above cite a report entry Gxx and its labeled evidence Exx; the report links those references to source/test artifacts.

