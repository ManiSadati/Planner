# AVE–PTOAS VMI: slide outline — 2026-10-20

Nine numbered slides, ending with a complete summary table. Two optional appendix groups follow Slide 6; device-object integration and interface compatibility follow them, before the final summary. The audience is PTOAS project owners; links are review references and can move into presentation notes. Scope includes the original report, the October 21 Q2 re-audit and separately cited PR #1660; no device-performance result is claimed. [Original scope / E00](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:17), [Q2 re-audit](/home/m00967009/Workspace/Planner/bridge/designs/q2-ptoas-vmi-additional-gaps-and-priorities-2026-10-21.md:5).

Code examples are short, adapted excerpts from the cited sources, not standalone programs; SSA names and unrelated attributes are simplified. Paired forms show separate stages, and rejected IR is labeled. `#ub` / `#gm` abbreviate HIVM memory-space attributes; `#map` is the source layout. Ellipses occur only in explicitly abbreviated shuffle indices.

## Slide 1 — Add masks to floating conversion

**Category:** New Operations Request.

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

**Category:** New Operations Request.

**On slide**

- **Example:** Called Q2 solve helpers spill/reload PB32 predicates in UB; VMI memory data/results are numeric vectors. [G03](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:82) · [E03](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:101).
- **PTOAS request:** Add predicate-valued load/store with PB granularity and explicit packed-address units, lowered to existing physical predicate-memory instructions. [G03](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:82) · [E03](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:101).
- **Acceptance:** Validate the captured PB32 store/reload and its controlling mask before generalizing widths and addresses. [G03](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:82) · [E03](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:101).

**Code comparison — numeric VMI store versus AVE predicate store**

```mlir
// Supported numeric stores: equivalent contiguous, full-vector examples.
pto.vmi.store %values, %dst[%c0]
    : !pto.vmi.vreg<64xf32>, !pto.ptr<f32, ub>
// Unified spelling; shown as an alternative, not a second write.
pto.vmi.vstore %values, %dst[%c0]
    : !pto.vmi.vreg<64xf32>, !pto.ptr<f32, ub>
```

Both store numeric data; the unified `vstore` also supports optional masks and other access modes. [Legacy store definition](/home/m00967009/Workspace/PTOAS/include/PTO/IR/VMIOps.td:732) · [Unified store definition](/home/m00967009/Workspace/PTOAS/include/PTO/IR/VMIOps.td:1461) · [Contiguous lowering testcase](/home/m00967009/Workspace/PTOAS/test/lit/vmi_new/vmi_to_vpto_load_store_contiguous.pto:14).

```llvm
; Numeric F32 normal-store path, after resolving %dst[%c0].
; %active is an all-active physical predicate for this example.
call void @llvm.hivm.vstsx1.v64f32(
    <64 x float> %values, ptr addrspace(6) %dst,
    i32 0, i32 2, i32 0, <256 x i1> %active)
```

The numeric vector is the stored payload; the last predicate operand controls which lanes write. This intrinsic call is adapted from saved Q2 bridge LLVM output; it does not describe every VMI store mode. [Emitted call](/home/m00967009/Workspace/Planner/bridge/testcases/row_softmax_16x256/row_softmax_16x256.ptoas.ll:140) · [Intrinsic signature](/home/m00967009/Workspace/AscendNPU-IR/third-party/llvm-project/llvm/include/llvm/IR/IntrinsicsHIVM.td:1155).

```mlir
// Q2 KKT: simplified names; #ub denotes the UB address space.
%all = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>}
    : vector<64xi1>
ave.hir.masked_store <NORM_B8> %dst[%c0], %all, %predicate
    {functionType = #ave.func_dist_type<pb32>}
    : memref<64xi1, #map, #ub>, vector<64xi1>, vector<64xi1>
```

Here the predicate itself is the payload, and `%all` enables the whole write. [Captured Q2 store](/home/m00967009/Workspace/Logs/q2_ave_requirements_oct19/captures/reused_kkt_H2_B1/before-bridge.mlir:456).

```llvm
; Ascend950 native PB32 path; address calculation omitted.
%init = call <32 x i8> @llvm.hivm.init.vector.align.data()
%packed = call { <32 x i8>, ptr addrspace(6) } @llvm.hivm.pstu.b32(
    <256 x i1> %predicate, ptr addrspace(6) %dst, <32 x i8> %init)
%state = extractvalue { <32 x i8>, ptr addrspace(6) } %packed, 0
%end = extractvalue { <32 x i8>, ptr addrspace(6) } %packed, 1
call void @llvm.hivm.vstas(
    <32 x i8> %state, ptr addrspace(6) %end, i32 0, i32 0)
```

`pstu.b32` packs PB32 predicate data for memory, returns updated alignment/pointer state, and `vstas` completes the store. This simplified LLVM sequence follows native lowering and intrinsic declarations; it is not a newly captured end-to-end compile. [Native PB32 dispatch](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Conversion/HIVMAVEToAVEIntrin/HIVMAVEToAVEIntrin.cpp:1197) · [Sequence builder](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Conversion/HIVMAVEToAVEIntrin/HIVMAVEToAVEIntrin.cpp:777) · [PSTU signature](/home/m00967009/Workspace/AscendNPU-IR/third-party/llvm-project/llvm/include/llvm/IR/IntrinsicsHIVM.td:1416) · [VSTAS signature](/home/m00967009/Workspace/AscendNPU-IR/third-party/llvm-project/llvm/include/llvm/IR/IntrinsicsHIVM.td:1361).

**Speaker notes**

A mask that controls a numeric store is different from a mask stored as data. The physical `<256 x i1>` predicate register represents the logical 64-lane PB32 predicate here. `pstu.b32` has no separate controlling-mask operand: the all-active KKT example does not establish general partial-mask/tail semantics. Physical support already exists; VMI needs a predicate-data memory interface that selects it correctly. Other PB widths, tails and indexed addresses need validation. [G03](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:82) · [E03](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:101).

## Slide 3 — Support explicit signed `vmins` / `vmaxs`

**Category:** Feature Request — explicit signedness interface; the normalization side note is a Lowering Issue.

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

The main request is an operation-level signedness interface; signed/unsigned arithmetic already works with explicit `si32` / `ui32` vector and scalar types, including runtime scalars. The earlier local LLVM probe removes same-width casts. The October 21 re-audit establishes a Q2 Mamba DQKV occurrence with constant scalars and an H mask; its extracted helper converts to signed VMI. A fresh upstream build reproduces the normalization mismatch and accepts independently typed signed runtime scalars. Unchanged bridge output separately needs the interface migration on Slide 8. **Priority: Low** for the captured form because a typed workaround exists; no runtime cast penalty is established. [G09 / re-audit](/home/m00967009/Workspace/Planner/bridge/designs/q2-ptoas-vmi-additional-gaps-and-priorities-2026-10-21.md:25), [Q2 operations](/home/m00967009/Workspace/Logs/q2_vmi_gap_reaudit_oct21/captures/mamba_volatile/before-bridge.mlir:942), [fresh signed lowering](/home/m00967009/Workspace/Logs/q2_vmi_gap_reaudit_oct21/typed-signed.mlir:2).

## Slide 4 — Integrate the mask fix; recheck scope placement

**Category:** Lowering Issue.

**On slide**

- **Pinned arithmetic-mask loss:** Integrate the existing upstream mask-preserving lowering and matching layout handling; verify predicates reach physical arithmetic. [G02](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:70) · [E02](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:80).
- **Reproduced scope failure:** Fresh upstream still rejects the D1200-derived focused loop because a generated `!pto.mask<b32>` escapes its vector scope. Repair predicate placement/rematerialization and add a downstream regression. **Priority: High for the focused defect.** [G10 / re-audit](/home/m00967009/Workspace/Planner/bridge/designs/q2-ptoas-vmi-additional-gaps-and-priorities-2026-10-21.md:19), [fresh diagnostic](/home/m00967009/Workspace/Logs/q2_vmi_gap_reaudit_oct21/nested-scope.stderr:1).
- **Boundary:** Fresh upstream preserves arithmetic masks. Complete Q2 D1200 replay stops earlier at the interface mismatch on Slide 8, so the focused scope failure does not establish that the full migrated kernel still fails there. [Mask result](/home/m00967009/Workspace/Logs/q2_vmi_gap_reaudit_oct21/partial-arithmetic.mlir:2), [Q2 replay diagnostic](/home/m00967009/Workspace/Logs/q2_vmi_gap_reaudit_oct21/q2-d1200.stderr:1).

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

**Category:** Feature Request — continuous-store intent and eligible stream fusion (G04 / F05).

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

**VMI input — the higher-level code before PTOAS lowering**

```mlir
// Each reduction produces one logical F32 lane; reductions omitted.
scf.for %row = %c0 to %c32 step %c1 {
  pto.vmi.vstore %one0, %out0[%row]
      : !pto.vmi.vreg<1xf32>, !pto.ptr<f32, ub>
  pto.vmi.vstore %one1, %out1[%row]
      : !pto.vmi.vreg<1xf32>, !pto.ptr<f32, ub>
}
```

These two unified VMI stores express consecutive one-value writes to separate outputs, but carry no explicit continuous-stream intent comparable to AVE's `hivm.is_continuous`. This is the simplified input to the tested lowering below, with destination names changed for consistency. [VMI probe: reductions and stores](/tmp/ptoas-onept-check-20261020/singleton-loops.pto:32) · [G04 / E04](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:103).

**PTOAS physical lowering — simplified excerpt of the tested result**

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

**Category:** Performance Issue — reduction carrier reconstruction (G08 / F04).

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

AVE's full-width result does not promise a full-width splat. The October 21 paired loop probe confirms extra VPTO work for a wide carrier: an in-loop `pto.vdup {position = "LOWEST"}` and a final projection broadcast disappear with a singleton accumulator. **Priority: Medium for demonstrated generated-work overhead; elapsed-time degradation and device equivalence remain unmeasured.** This is specific to lane-zero-only consumers, not every reduction graph. ADD is captured in Q2; exact Q2 MAX lowering remains unvalidated. [G08 / re-audit](/home/m00967009/Workspace/Planner/bridge/designs/q2-ptoas-vmi-additional-gaps-and-priorities-2026-10-21.md:20), [paired output](/home/m00967009/Workspace/Logs/q2_vmi_gap_reaudit_oct21/reduction-loop-pair.mlir:11).

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

## Slide 7 — Device object output for Triton-Ascend integration

**Category:** Feature Request — implemented in PR, pending upstream merge at the October 21 check.

**On slide**

- **Integration need:** Our end-to-end Triton-kernel execution path through Triton-Ascend needed a device object file. PTOAS previously exposed fat-object emission for this VPTO output path. The Triton-Ascend requirement is project context supplied by the integration owner; the before/after output paths are visible in the [PR implementation](https://github.com/hw-native-sys/PTOAS/pull/1660/files).
- **Work added:** [PR #1660 — `feat: add an option to emit device object file`](https://github.com/hw-native-sys/PTOAS/pull/1660) adds `--emit-device-object` to write the final linked AICore device object directly to `-o`, without host-stub/fat-object packaging. Existing default fat-object output remains available. [Driver selection](https://github.com/hw-native-sys/PTOAS/blob/a3c33b9372617eb06786fb139935e5be797010d7/tools/ptoas/driver.cpp#L1122) · [Device emission](https://github.com/hw-native-sys/PTOAS/blob/a3c33b9372617eb06786fb139935e5be797010d7/tools/ptoas/ObjectEmission.cpp#L1204).
- **PTOAS next step:** Review and merge the opt-in output mode, then validate the emitted object in the Triton-Ascend execution path. **Status checked October 21: PR open, not merged.** [PR status](https://github.com/hw-native-sys/PTOAS/pull/1660).

**Usage — select the output required by the integration**

```bash
# Existing default: fat object, including host-side packaging.
ptoas kernel.pto --pto-backend=vpto --pto-arch=a5 -o kernel.fat.o

# Added by PR #1660: final linked AICore device object.
ptoas kernel.pto --pto-backend=vpto --pto-arch=a5 \
  --emit-device-object -o kernel.device.o
```

The flag changes the output artifact; changing the `.o` filename alone does not select device-object emission. [CLI option](https://github.com/hw-native-sys/PTOAS/blob/a3c33b9372617eb06786fb139935e5be797010d7/tools/ptoas/ptoas.cpp#L641) · [Output dispatch](https://github.com/hw-native-sys/PTOAS/blob/a3c33b9372617eb06786fb139935e5be797010d7/tools/ptoas/driver.cpp#L1122).

**Speaker notes**

At inspected PR head `a3c33b9372617eb06786fb139935e5be797010d7`, device emission compiles the available cube/vector LLVM modules and links their device objects through the AICore linker path. Fat-object emission additionally packages device code with a host stub. The new mode requires the VPTO backend and an explicit output file; it rejects mixed-backend modules and debug-IR output flags. [Emission implementation](https://github.com/hw-native-sys/PTOAS/blob/a3c33b9372617eb06786fb139935e5be797010d7/tools/ptoas/ObjectEmission.cpp#L1204) · [Mode restrictions](https://github.com/hw-native-sys/PTOAS/blob/a3c33b9372617eb06786fb139935e5be797010d7/tools/ptoas/driver.cpp#L1334).

This slide records integration work, separate from the VMI instruction gaps. Source and PR status were checked for this update; a successful end-to-end Triton-Ascend device run was not independently verified here.

## Slide 8 — N01: Compatibility with the current VMI/PTO interface

**Category:** Lowering Issue — bridge/PTOAS interface compatibility.

**On slide**

- **Observed:** Current upstream rejects Q2 D1200's `pto.load_scalar` and the freshly converted Mamba helper's `pto.vmi.vload {dist_mode = "brc"}`. Scalar and broadcast loads still have supported forms. [N01](/home/m00967009/Workspace/Planner/bridge/designs/q2-ptoas-vmi-additional-gaps-and-priorities-2026-10-21.md:29).
- **PTOAS / bridge request:** Publish a supported revision pair and migration contract. Update bridge emission or provide an explicit compatibility conversion for the changed scalar-memory and unified-memory interfaces; preserve addresses, types and broadcast behavior. [N01 requested action](/home/m00967009/Workspace/Planner/bridge/designs/q2-ptoas-vmi-additional-gaps-and-priorities-2026-10-21.md:35).
- **Priority: Low while retaining a compatible pair; blocking for unchanged bridge output on current upstream.** Pinning avoids this mismatch while migration is completed; it does not fix the other gaps. [Workaround and boundary](/home/m00967009/Workspace/Planner/bridge/designs/q2-ptoas-vmi-additional-gaps-and-priorities-2026-10-21.md:46).

**Actual rejection messages — separate Q2 replays**

```text
error: custom op 'pto.load_scalar' is unknown
error: 'pto.vmi.vload' op dist_mode is no longer supported
```

The existing D1200 output and newly converted Mamba helper fail at the IR interface before complete downstream validation. [D1200 diagnostic](/home/m00967009/Workspace/Logs/q2_vmi_gap_reaudit_oct21/q2-d1200.stderr:1), [Mamba diagnostic](/home/m00967009/Workspace/Logs/q2_vmi_gap_reaudit_oct21/mamba-signed.stderr:1).

**Current broadcast form — tested VMI composition**

```mlir
%one = pto.vmi.vload %src[%c0]
    : !pto.ptr<bf16, ub> -> !pto.vmi.vreg<1xbf16>
%many = pto.vmi.vbrc %one
    : !pto.vmi.vreg<1xbf16> -> !pto.vmi.vreg<64xbf16>
```

This form lowers to a scalar-address `BRC_B16` load; current scalar-memory syntax uses `pto.load`. [Tested input](/home/m00967009/Workspace/Logs/q2_vmi_gap_reaudit_oct21/probes.pto:9), [lowering](/home/m00967009/Workspace/Logs/q2_vmi_gap_reaudit_oct21/mixed-select-and-brc.mlir:18), [scalar definition](/tmp/ptoas-q2-audit-upstream-20261021/include/PTO/IR/VPTOOps.td:171).

**Speaker notes**

The checked upstream revision is `37f5f25e3b5b433e7cc760d148fea83dddef9934` (October 21). This is a compatibility/migration request; no additional missing VMI instruction was established. Acceptance is successful conversion and downstream replay of both captured graphs with the selected compiler pair, followed by device validation. The focused broadcast probe does not establish complete migration or whole-kernel correctness. [N01 evidence and limits](/home/m00967009/Workspace/Planner/bridge/designs/q2-ptoas-vmi-additional-gaps-and-priorities-2026-10-21.md:33).

## Slide 9 — All issues and priorities at a glance

**On slide**

| Category | Priority | Issue / detail slide | Requested action | Evidence / status |
|---|---|---|---|---|
| New Operations Request | **High** | G01 — Masked floating conversion · S1 | Add masked widening/narrowing with defined inactive lanes and packing. | Direct mask form rejects; broader workaround unvalidated. [G01 audit](/home/m00967009/Workspace/Planner/bridge/designs/q2-ptoas-vmi-additional-gaps-and-priorities-2026-10-21.md:17) |
| New Operations Request | **High** | G03 — Predicate memory · S2 | Add predicate-valued load/store and packed-buffer/address semantics. | Q2 spills and Mamba carrier failure; byte route incomplete. [G03 audit](/home/m00967009/Workspace/Planner/bridge/designs/q2-ptoas-vmi-additional-gaps-and-priorities-2026-10-21.md:18) |
| Lowering Issue | **High — focused** | G10 — Nested predicate scopes · S4 | Rematerialize generated predicates inside consuming scopes. | Fresh focused failure; full Q2 replay first hits N01. [G10 audit](/home/m00967009/Workspace/Planner/bridge/designs/q2-ptoas-vmi-additional-gaps-and-priorities-2026-10-21.md:19) |
| Lowering Issue | **High — bridge** | G11 — Integer metadata loads · Appendix B | Bridge: lower I32 GM loads; PTOAS: document/test the scalar-load contract. | Existing target capability; default bridge mapping rejects. [G11 / E11](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:197), [priority](/home/m00967009/Workspace/Planner/bridge/designs/q2-ptoas-vmi-additional-gaps-and-priorities-2026-10-21.md:50) |
| Lowering Issue | **High — bridge** | G12 — BF16 store wrappers · Appendix B | Bridge: recognize BF16 carriers; PTOAS: regress the resulting stores. | BF16 storage exists; wrapper recognition blocks conversion. [G12 / E12](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:209), [priority](/home/m00967009/Workspace/Planner/bridge/designs/q2-ptoas-vmi-additional-gaps-and-priorities-2026-10-21.md:50) |
| Lowering Issue | **Low — integration** | G02 — Arithmetic masks · S4 | Integrate existing mask-preserving lowering with a compatible bridge. | Fixed upstream; fresh partial arithmetic retains its mask. [Current result](/home/m00967009/Workspace/Logs/q2_vmi_gap_reaudit_oct21/partial-arithmetic.mlir:2) |
| Lowering Issue | **Low** | G05 — Partial B16 results · Appendix A | Extend partial-result materialization for demonstrated additional consumers. | Widen-first composition works for checked Q2 graphs. [G05 audit](/home/m00967009/Workspace/Planner/bridge/designs/q2-ptoas-vmi-additional-gaps-and-priorities-2026-10-21.md:22) |
| Lowering Issue | **Low** | G06 — B16 broadcast layout · Appendix A | Adapt bridge producers to the supported broadcast/widen composition. | Current composition lowers; old spelling needs N01 migration. [G06 audit](/home/m00967009/Workspace/Planner/bridge/designs/q2-ptoas-vmi-additional-gaps-and-priorities-2026-10-21.md:23) |
| Lowering Issue | **Low** | G07 — Mixed-width selection · Appendix A | Validate exact sparse predicate/layout adaptation before simplifying. | Both tested routes lower; faster replacement unproven. [G07 audit](/home/m00967009/Workspace/Planner/bridge/designs/q2-ptoas-vmi-additional-gaps-and-priorities-2026-10-21.md:24) |
| Lowering Issue | **Low — typed workaround** | G09 — Scalar normalization side note · S3 | Normalize vector/scalar types consistently, preserving signedness. | Mismatch reproduced; matching `si32` operands work. [G09 audit](/home/m00967009/Workspace/Planner/bridge/designs/q2-ptoas-vmi-additional-gaps-and-priorities-2026-10-21.md:25) |
| Lowering Issue | **Low — compatible pair** | N01 — VMI/PTO interface migration · S8 | Publish a revision contract and migrate old memory forms or provide an adapter. | Unchanged bridge/current upstream combination blocks; pinning avoids mismatch. [N01](/home/m00967009/Workspace/Planner/bridge/designs/q2-ptoas-vmi-additional-gaps-and-priorities-2026-10-21.md:29) |
| Lowering Issue | **Unranked — conditional** | G13 — Barrier lifetimes · Appendix B | Agree/test live-value handling; fix bridge barrier placement where needed. | Opt-in experiment; default Q2 defect unestablished. [G13 / E13](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:221) |
| Performance Issue | **Medium*** | G08 — Reduction carriers · S6 | Keep lane-zero-only accumulators singleton; remove carrier-only broadcasts. | Paired lowering shows removable `vdup`; timing unmeasured. [G08 audit](/home/m00967009/Workspace/Planner/bridge/designs/q2-ptoas-vmi-additional-gaps-and-priorities-2026-10-21.md:20) |
| Feature Request | **Low** | G04 — Continuous-store intent/fusion · S5 | Express stream intent on `vstore`; fuse eligible independent streams safely. | Point stores work; stream parity differs and speed impact is unmeasured. [G04 audit](/home/m00967009/Workspace/Planner/bridge/designs/q2-ptoas-vmi-additional-gaps-and-priorities-2026-10-21.md:21) |
| Feature Request | **Low** | G09 — Explicit signed `vmins` / `vmaxs` · S3 | Add signed forms or an attribute accepting signless vector/scalar carriers. | Typed signed support exists; Q2 Mamba occurrence confirmed. [G09 audit](/home/m00967009/Workspace/Planner/bridge/designs/q2-ptoas-vmi-additional-gaps-and-priorities-2026-10-21.md:25) |
| Feature Request | **Unranked — Q2 need pending** | G14 — Genuine scalar consumers · Appendix B | Specify/test lane-to-scalar materialization; add an operation only if needed. | Focused tests exist; exact Q2 demand remains unconfirmed. [G14 / E14](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-gap-report-2026-10-20.md:233) |
| Feature Request | **Low — merge follow-up** | PR #1660 — Device object output · S7 | Review/merge `--emit-device-object` and validate Triton-Ascend integration. | Implemented in PR; open at the October 21 check. [PR #1660](https://github.com/hw-native-sys/PTOAS/pull/1660) |

**Priority key:** High = no validated general workaround; Medium* = extra generated work demonstrated, timing unmeasured; Low = bounded workaround or integration follow-up is adequate for now. Conditional/unconfirmed Q2 items remain unranked. [Audit priority rule and caveat](/home/m00967009/Workspace/Planner/bridge/designs/q2-ptoas-vmi-additional-gaps-and-priorities-2026-10-21.md:11).

**Speaker notes**

Coverage is G01–G14, N01 and PR #1660, including both appendices. G09 has separate rows for the signedness interface and normalization fix because their implementation requests and categories differ. New Operations Request covers new masked-operation forms or predicate-data interfaces; Lowering Issue covers existing-capability mapping/correctness and interface migration; Performance Issue covers demonstrated extra generated work; Feature Request covers supported-interface extensions and integration features. Categories describe the requested work, not proof that an underlying physical instruction is missing.

Priorities follow the Q2 re-audit where assigned. Low for G02 and PR #1660 denotes follow-up on an existing implementation, not a new unimplemented capability. G11/G12 remain bridge-owned High items. G08's Medium tier does not establish elapsed-time degradation; device equivalence and timing remain to validate. N01 becomes blocking if the incompatible compiler combination is required. Unranked G13/G14 have insufficient applicability evidence for a High/Medium/Low assignment. [Priority boundaries](/home/m00967009/Workspace/Planner/bridge/designs/q2-ptoas-vmi-additional-gaps-and-priorities-2026-10-21.md:27), [bridge/conditional entries](/home/m00967009/Workspace/Planner/bridge/designs/q2-ptoas-vmi-additional-gaps-and-priorities-2026-10-21.md:48).

## Review handoff

Agree owners and semantic contracts for the main requests; use captured Q2 graphs to evaluate the conditional work. Original Gxx/Exx references lead to the report and its source/test artifacts; updated priorities/status and N01 lead to the October 21 re-audit and fresh compiler outputs. Device-object work cites PR #1660 directly. The final summary includes every main-slide and appendix issue while distinguishing implementation ownership, resolved work and unverified cases.

