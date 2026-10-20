# Categorized AVE–PTOAS VMI gaps for review — 2026-10-20

The sole findings source is the [evidence inventory](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-evidence-inventory-2026-10-20.md:1). References below point to its entries and underlying source/test artifacts; no other list, inventory or report supplies findings. This document proposes review actions, not slides or implemented fixes.

Revision and validation limits follow the inventory: AscendNPU-IR `8b8f40f88e99`, actual PTOAS submodule `3c1c98fcc8ec`, Q2 `ebf78b205a87`; upstream source comparisons use `ec8d1810b6c1`, except T01 at `739480c309c2`. Saved compiler output and unexecuted regression expectations remain distinct. No new compiler or device tests were run for this categorization.

Code examples are simplified excerpts, not standalone programs: SSA names are shortened, surrounding definitions/locations are omitted, and `#ub` / `#gm` abbreviate HIVM UB / GM address-space attributes. Paired before/after excerpts are separate representations. Abbreviated index lists, omitted type clauses and forms that are not successful conversions are labeled explicitly.

## Review priority and consolidation

| Tier | Demonstrated relevance | Entries |
|---|---|---|
| A — review recorded Q2 blockers first | Concrete Q2 compiler rejection or downstream failure is recorded. Ownership may be PTOAS or bridge. | G01, G10, G11 |
| B — review captured Q2 requirements and workarounds | Q2 captures show the operation, representation difference or restricted mapping; a full-kernel failure caused by that specific issue is not established. | G02–G08, G12 |
| C — focused or conditional relevance | Supported by focused tests/probes or a conditional experiment; exact Q2 demand/default-path failure is unestablished. | G09, G13, G14 |

These tiers rank evidence of applicability, not severity, frequency or runtime cost. The inventory does not establish an ordering within a tier or measured per-entry performance impact.

Paired operations sharing one request remain consolidated: widening/narrowing, predicate spill/reload, and signed min/max. No further duplicate capability requests were identified among the 14 entries. Broadcast-producer adaptation, partial-register shuffle materialization and mixed-width selection stay separate because their required lowerings differ. PTOAS predicate-scope inference and bridge barrier placement also remain separate.

## Masking and tail behavior

### G01 — Preserve masks during floating conversion

- **Category / priority:** Masking and tail behavior; A. Inventory F01.
- **AVE behavior and example:** Convert only active F16/BF16 lanes to F32, or narrow active F32 lanes with `ave.hir.vextf` / `vtruncf`, `part_even`. Q2 cumsum has an H/PB32 mask, widening and a same-mask store; its conversion rejects.
- **PTOAS limitation:** VMI `extf` / `vcvt` have no conversion-mask operand. The bridge accepts ALL and narrowly checked VL1 one-point-store consumers.
- **Implementation request:** Add masked widening/narrowing semantics and lowering, specifying inactive-lane behavior while preserving active values, rounding, saturation and lane placement. Cover the captured H-mask graph as well as VL1.
- **Workaround / consequence:** Convert extra lanes only under the checked lane-zero consumer guard; broader partial forms remain rejected. Arithmetic-mask fixes do not close this gap.
- **Evidence:** [Inventory F01](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-evidence-inventory-2026-10-20.md:12); [Q2 example, lines 33–44](/home/m00967009/Workspace/Logs/q2_ave_requirements_oct19/captures/cumsum_scalar_False/before-bridge.mlir:33); [rejection](/home/m00967009/Workspace/Logs/q2_ave_requirements_oct19/captures/cumsum_scalar_False/compiler.txt:2357); [VMI conversion operands](/home/m00967009/Workspace/AscendNPU-IR/third-party/ptoas/include/PTO/IR/VMIOps.td:1413).
- **Confidence / caveat:** High for missing mask representation and recorded rejection. General inactive-lane equivalence and runtime cost remain unvalidated.

**AVE example — captured cumsum widening (G01).**

```mlir
%half = ave.hir.pge <H> {functionType = #ave.func_dist_type<pb32>}
    : vector<64xi1>
%wide = ave.hir.vextf %x, <part_even>, %half
    {functionType = #ave.func_dist_type<even>}
    : vector<64xf16>, vector<64xf32>, vector<64xi1>
```

The conversion explicitly requests only the H-mask lanes, and this captured graph is rejected by the bridge. [Source, lines 33–39](/home/m00967009/Workspace/Logs/q2_ave_requirements_oct19/captures/cumsum_scalar_False/before-bridge.mlir:33).

**Closest declared VMI form — not emitted support for that rejected input (G01).**

```mlir
%wide = pto.vmi.extf %x
    : !pto.vmi.vreg<64xf16> -> !pto.vmi.vreg<64xf32>
```

This VMI operation has no place to carry `%half`, so using it requires a separate proof that evaluating extra lanes preserves the required behavior. [Operation definition](/home/m00967009/Workspace/AscendNPU-IR/third-party/ptoas/include/PTO/IR/VMIOps.td:569).

### G02 — Retain arithmetic predicates through lowering

- **Category / priority:** Masking and tail behavior; B. Inventory F06.
- **AVE behavior and example:** Calculate only selected lanes. Captured dqkwg uses H-masked `ave.hir.vexp` followed by a same-mask store.
- **PTOAS limitation:** The pinned unified-to-legacy binary/unary path discards arithmetic masks.
- **Implementation request:** Integrate upstream mask-preserving direct VPTO lowering, including matching mask/layout handling, into the supported compiler pair. Retain regressions for partial and all-off predicates; preserve supplied predicates through physical lowering.
- **Workaround / consequence:** A matching final store can hide inactive results; that is not a general proof for other consumers.
- **Evidence:** [Inventory F06](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-evidence-inventory-2026-10-20.md:15); [dqkwg example](/home/m00967009/Workspace/Logs/q2_ave_requirements_oct19/captures/reused_dqkwg_H2_B4/before-bridge.mlir:983); [pinned mask-discarding helpers](/home/m00967009/Workspace/AscendNPU-IR/third-party/ptoas/lib/PTO/Transforms/VMILowerUnifiedToLegacy.cpp:233); [upstream alternative](/tmp/ptoas-research-20261019/lib/PTO/Transforms/VMIToVPTO/VMIToVPTOUnifiedPatternInternals.cpp:57).
- **Confidence / caveat:** High for the pinned defect. Fixed upstream by `2503c259`; this is an integration request, not a newly missing upstream feature. Q2 runtime corruption is not demonstrated.

**Pinned PTOAS unary lowering — C++ excerpt (G02).**

```cpp
// Inside lowerMaskedUnary, after the merge-mode rejection.
Value source = op.getSource();
Value raw = createLegacy(loc, resultType, source);
op.getResult().replaceAllUsesWith(raw);
op->erase();
```

The legacy builder receives the source but no predicate, demonstrating mask loss independently of whether a later masked store hides it. [Pinned implementation](/home/m00967009/Workspace/AscendNPU-IR/third-party/ptoas/lib/PTO/Transforms/VMILowerUnifiedToLegacy.cpp:263).

## Operation and semantic coverage

### G03 — Load and store predicates as data

- **Category / priority:** Operation and semantic coverage; B. Inventory T02.
- **AVE behavior and example:** Spill and reload true/false values themselves. Called Q2 solve helpers use PB32 `vload <NORM>` and `masked_store <NORM_B8>` on UB `i1` memory.
- **PTOAS limitation:** VMI memory data/results are numeric vregs; a controlling mask is supported, but mask-valued memory data is not represented directly.
- **Implementation request:** Add first-class predicate-valued UB load/store with explicit PB granularity and packed-address semantics; lower through the existing physical predicate-memory operations. Validate the captured PB32 form before generalizing widths and addresses.
- **Workaround / consequence:** Numeric byte carriers require explicit physical repair. The inspected default flows omit that repair, so raw carriers do not establish complete packed-memory support.
- **Evidence:** [Inventory T02](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-evidence-inventory-2026-10-20.md:13); [solve reload](/home/m00967009/Workspace/Logs/q2_ave_requirements_oct19/captures/solve/before-bridge.mlir:1194), [store](/home/m00967009/Workspace/Logs/q2_ave_requirements_oct19/captures/solve/before-bridge.mlir:1515); [memory typing](/home/m00967009/Workspace/AscendNPU-IR/third-party/ptoas/include/PTO/IR/VMIOps.td:758); [physical repair testcase](/home/m00967009/Workspace/AscendNPU-IR/bishengir/test/Conversion/HIVMAVEToPTOASVMI/finalize-vpto-predicate-memory.mlir:5).
- **Confidence / caveat:** High for the representation gap and Q2 PB32 occurrence. Other PB widths, tails and indexed addresses are not fully validated.

**AVE predicate store — current bridge testcase (G03).**

```mlir
ave.hir.masked_store <NORM_B8> %dst[%c0], %all, %predicate
    {functionType = #ave.func_dist_type<pb32>}
    : memref<64xi1, strided<[1]>, #ub>,
      vector<64xi1>, vector<64xi1>
```

Here `%predicate` is the data being stored, while `%all` controls the store. [Testcase](/home/m00967009/Workspace/AscendNPU-IR/bishengir/test/Conversion/HIVMAVEToPTOASVMI/preflight-predicate-memory.mlir:27).

**VMI byte-carrier workaround — adapted regression expectation (G03).**

```mlir
// Definitions of zero/one vectors and pointer markers are omitted.
%values = pto.vmi.vsel %predicate, %ones, %zeros
    : !pto.vmi.mask<64xpred>, !pto.vmi.vreg<64xi32>,
      !pto.vmi.vreg<64xi32> -> !pto.vmi.vreg<64xi32>
%bytes = pto.vmi.trunci %values {saturate = "SAT"}
    : !pto.vmi.vreg<64xi32> -> !pto.vmi.vreg<64xui8>
pto.vmi.store %bytes, %byte_ptr[%c0]
    : !pto.vmi.vreg<64xui8>, !pto.ptr<ui8, ub>
```

The mask becomes numeric zero/one bytes, which require the explicit repair pass before representing native packed predicate memory. [Expected bridge output](/home/m00967009/Workspace/AscendNPU-IR/bishengir/test/Conversion/HIVMAVEToPTOASVMI/preflight-predicate-memory.mlir:9).

**Physical PB32 repair — adapted finalizer expectation (G03).**

```mlir
// Pointer/address repair is omitted; pstu/vstas type clauses are omitted.
%align = pto.init_align : !pto.align
%align_out, %base_out = pto.pstu %align, %predicate_b32, %typed_ptr
pto.vstas %align_out, %base_out, %c0
```

The explicit finalizer restores existing physical predicate-store operations; this sequence is not first-class VMI predicate-memory support. [Finalizer expectation](/home/m00967009/Workspace/AscendNPU-IR/bishengir/test/Conversion/HIVMAVEToPTOASVMI/finalize-vpto-predicate-memory.mlir:5).

## Stateful memory streams

This category separates cross-iteration alignment/base state and stream completion from ordinary addressing or register layout.

### G04 — Preserve continuous one-point store streams

- **Category / priority:** Stateful memory streams; B. Inventory F05.
- **AVE behavior and example:** Write successive row results as one continuous stream. Dqkwg has `masked_store <ONEPT_B32>` with `hivm.is_continuous`.
- **PTOAS limitation:** The bridge's singleton `vstore` representation carries no native alignment/base state or stream completion across iterations.
- **Implementation request:** First validate upstream stream fusion on the exact singleton pattern. If it does not recover the native sequence, add a logical stateful-store representation and lowering to initialization, post-increment stores and final completion.
- **Workaround / consequence:** Independent one-point stores represent ordinary writes but do not preserve the native stream sequence.
- **Evidence:** [Inventory F05](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-evidence-inventory-2026-10-20.md:14); [dqkwg stores](/home/m00967009/Workspace/Logs/q2_ave_requirements_oct19/captures/reused_dqkwg_H2_B4/before-bridge.mlir:1376); [native sequence](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Conversion/HIVMAVEToAVEIntrin/HIVMAVEToAVEIntrin.cpp:1119); [upstream stream testcase](/tmp/ptoas-research-20261019/test/lit/vmi_new/vmi_to_vpto_stateful_stream_fusion.pto:58).
- **Confidence / caveat:** High for the sequence difference. Upstream's 64-F32 stream example does not establish singleton eligibility; numerical or performance effects remain unproven.

**AVE continuous singleton store — dqkwg excerpt (G04).**

```mlir
ave.hir.masked_store <ONEPT_B32> %row_dst[], %first, %value
    {functionType = #ave.func_dist_type<norm>, hivm.is_continuous}
    : memref<f32, strided<[], offset: ?>, #ub>,
      vector<1xi1>, vector<1xf32>
```

The marker asks native lowering to retain continuous-store handling even though each iteration contributes one value. [Captured store; unaligned-access attribute omitted](/home/m00967009/Workspace/Logs/q2_ave_requirements_oct19/captures/reused_dqkwg_H2_B4/before-bridge.mlir:1376).

**Current bridge VMI mapping — adapted from the conversion builder (G04).**

```mlir
pto.vmi.vstore %value, %row_dst[%c0]
    : !pto.vmi.vreg<1xf32>, !pto.ptr<f32, ub>
```

The independent singleton store returns neither an updated alignment state nor an updated base pointer. [Bridge builder](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Conversion/HIVMAVEToPTOASVMI/HIVMAVEToPTOASVMI.cpp:4481).

**Native lowering — C++ excerpt (G04).**

```cpp
auto result = hivm_regbaseintrins::buildVstusPostOp(
    data, dataPtr, elemSize, alignData->getResult(0), rewriter);
Value alignOut = rewriter.create<LLVM::ExtractValueOp>(
    loc, result->getResult(0), 0);
Value baseOut = rewriter.create<LLVM::ExtractValueOp>(
    loc, result->getResult(0), 1);
hivm_regbaseintrins::buildVstasOp(alignOut, baseOut, rewriter);
```

Native lowering explicitly obtains updated alignment/base values and uses them for completion, exposing the state absent from the singleton VMI store. [Native implementation](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Conversion/HIVMAVEToAVEIntrin/HIVMAVEToAVEIntrin.cpp:1129).

## Layout and data movement

### G05 — Materialize partial B16 shuffle results

- **Category / priority:** Layout and data movement; B. Inventory F08.
- **AVE behavior and example:** Split a packed half-precision vector into sparse halves, then widen useful lanes. Q2 backward D1200 contains sparse `ave.hir.vintlv` feeding `vextf`.
- **PTOAS limitation:** General shuffle materialization rejects partial physical result chunks; the bridge supports only checked widening consumers of these sparse halves.
- **Implementation request:** Support partial B16 shuffle-result materialization with correct logical-to-physical lane mapping and defined padding treatment, preserving `part_even` interpretation.
- **Workaround / consequence:** Widen first and select complete B32 halves. Other sparse-half consumers reject; logical shuffles do not establish executed instruction overhead.
- **Evidence:** [Inventory F08](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-evidence-inventory-2026-10-20.md:16); [D1200 example](/home/m00967009/Workspace/Logs/q2_ave_requirements_oct19/captures/norm_bwd_D1200/before-bridge.mlir:237); [physical restriction](/home/m00967009/Workspace/AscendNPU-IR/third-party/ptoas/lib/PTO/Transforms/VMIToVPTO.cpp:2596); [F16/BF16 bridge testcase](/home/m00967009/Workspace/AscendNPU-IR/bishengir/test/Conversion/HIVMAVEToPTOASVMI/preflight-sparse-interleave-extf.mlir:3).
- **Confidence / caveat:** High for the restricted mapping and materializer rejection. The checked split/widen composition already exists; general interleave support is not absent.

**Paired sparse split/widen excerpts — bridge testcase (G05).**

```mlir
// AVE: split first, then widen a sparse half; unrelated attributes omitted.
%lo, %hi = ave.hir.vintlv %x, %x
    {layout_change = #ave<layout_change SPARSE>}
    : vector<128xf16>, vector<64xf16>
%a = ave.hir.vextf %lo, <part_even>, %all
    : vector<64xf16>, vector<64xf32>, vector<64xi1>

// Bridge VMI: widen first, then select a full B32 half.
%wide = pto.vmi.extf %x
    : !pto.vmi.vreg<128xf16> -> !pto.vmi.vreg<128xf32>
%a = pto.vmi.shuffle %wide[[0, 1, ..., 63]] // index list abbreviated
    : !pto.vmi.vreg<128xf32> -> !pto.vmi.vreg<64xf32>
```

Reordering the operations avoids materializing a partial physical B16 shuffle result; the upper half uses the corresponding indices 64–127. [AVE input and expected mapping](/home/m00967009/Workspace/AscendNPU-IR/bishengir/test/Conversion/HIVMAVEToPTOASVMI/preflight-sparse-interleave-extf.mlir:3).

### G06 — Adapt B16 broadcast loads to widening layouts

- **Category / priority:** Layout and data movement; B. Inventory F02.
- **AVE behavior and example:** Read one BF16 scalar and repeat it across 64 lanes before widening. Captured dqkwg uses `vload <BRC_B16>` in this graph.
- **PTOAS limitation:** The pinned broadcast producer cannot directly materialize the sparse layout required by widening consumers.
- **Implementation request:** Allow the 64-lane B16 broadcast producer to satisfy the consumer layout while retaining scalar-address vector broadcast behavior. Validate the representation on the supported upstream memory interface.
- **Workaround / consequence:** Load a 128-lane broadcast register and splat lane zero into 64 lanes. This still reads one scalar address; extra executed shuffle cost is unmeasured.
- **Evidence:** [Inventory F02](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-evidence-inventory-2026-10-20.md:17); [dqkwg example](/home/m00967009/Workspace/Logs/q2_ave_requirements_oct19/captures/reused_dqkwg_H2_B4/before-bridge.mlir:951); [bridge expansion](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Conversion/HIVMAVEToPTOASVMI/HIVMAVEToPTOASVMI.cpp:3778); [bridge testcase](/home/m00967009/Workspace/AscendNPU-IR/bishengir/test/Conversion/HIVMAVEToPTOASVMI/preflight-brc-b16-load.mlir:3).
- **Confidence / caveat:** High for the pinned workaround. Upstream removed `dist_mode`; an equivalent current-upstream limitation is not established.

**Paired B16 broadcast excerpts — pinned interface (G06).**

```mlir
// AVE: broadcast one BF16 value into 64 logical lanes; attributes omitted.
%x = ave.hir.vload <BRC_B16> %src[%c0]
    : memref<1xbf16, #ub> into vector<64xbf16>

// Bridge VMI: native-width broadcast, then lane-zero replication.
%full = pto.vmi.vload %src[%c0] {dist_mode = "brc"}
    : !pto.ptr<bf16, ub> -> !pto.vmi.vreg<128xbf16>
%x = pto.vmi.shuffle %full[[0, 0, ..., 0]] // 64 zero indices, abbreviated
    : !pto.vmi.vreg<128xbf16> -> !pto.vmi.vreg<64xbf16>
```

The workaround preserves one scalar-address broadcast while adapting its result to the widening consumer's layout. [Bridge testcase and expected output](/home/m00967009/Workspace/AscendNPU-IR/bishengir/test/Conversion/HIVMAVEToPTOASVMI/preflight-brc-b16-load.mlir:3).

### G07 — Select B16 values with a B32 comparison predicate

- **Category / priority:** Layout and data movement, including predicate granularity; B. Inventory F16.
- **AVE behavior and example:** Compare signed I32 indices, then select F16 values using the resulting predicate. The exact dqkwg helper contains `ave.hir.vcmp <LT>` followed by `vsel`.
- **PTOAS limitation:** Current bridge lowering avoids the sparse B16 comparison/select path by widening the data bit carriers.
- **Implementation request:** Validate and, where necessary, extend predicate-granularity/layout adaptation so the exact 64-lane I32 comparison can drive B16 selection directly, preserving lane correspondence and signed comparison semantics.
- **Workaround / consequence:** Zero-extend both B16 bit carriers to I32, select, then truncate. This adds logical conversions; final instruction cost is unmeasured.
- **Evidence:** [Inventory F16](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-evidence-inventory-2026-10-20.md:19); [dqkwg example](/home/m00967009/Workspace/Logs/q2_ave_requirements_oct19/captures/reused_dqkwg_H2_B4/before-bridge.mlir:938); [bridge expansion](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Conversion/HIVMAVEToPTOASVMI/HIVMAVEToPTOASVMI.cpp:3487); [upstream predicate-conversion testcase](/tmp/ptoas-research-20261019/test/lit/vmi_new/vmi_mask_granularity_assignment_vsel.pto:111).
- **Confidence / caveat:** High for the workaround; medium for remaining upstream implementation need. Upstream has related granularity support, but the exact Q2 path remains unvalidated.

**Paired mixed-width selection excerpts — dqkwg testcase (G07).**

```mlir
// AVE: compare I32 indices, select F16 values.
%cmp = ave.hir.vcmp <LT> %indices, %limit, %all
    : vector<64xi32>, vector<64xi1> -> vector<64xi1>
%chosen = ave.hir.vsel %cmp, %yes, %no : vector<64xi1>, vector<64xf16>

// Bridge VMI core: bitcasts to/from i16 are omitted.
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

The bridge widens both data bit patterns to match the comparison's B32 path instead of adapting only the predicate for B16 selection. [Expected conversion and AVE input](/home/m00967009/Workspace/AscendNPU-IR/bishengir/test/Conversion/HIVMAVEToPTOASVMI/preflight-vsel-dqkwg.mlir:3).

## Reduction and broadcast abstractions

### G08 — Avoid reduction carrier reconstruction when unnecessary

- **Category / priority:** Reduction and broadcast abstractions; B. Inventory F04.
- **AVE behavior and example:** Reduce a vector while retaining its vector-shaped carrier; only the first reduced position is defined. Q2 D1200 output contains a singleton ADD reduction followed by broadcast.
- **PTOAS limitation:** Floating reductions require a one-lane result; the bridge broadcasts to restore AVE's carrier width. This is a representation/optimization opportunity, not missing reduction execution.
- **Implementation request:** Preserve singleton reduced values through compatible consumers and eliminate carrier-only broadcasts where legal; materialize a broadcast when the consumer actually requires it. Test preservation of the defined first lane.
- **Workaround / consequence:** Singleton reduction plus broadcast reconstructs the surrounding type. A runtime penalty is not established.
- **Evidence:** [Inventory F04](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-evidence-inventory-2026-10-20.md:18); [D1200 example, lines 603–604](/home/m00967009/Workspace/Logs/q2_ave_requirements_oct19/captures/norm_bwd_D1200/converted.vmi.mlir:603); [AVE result contract](/home/m00967009/Workspace/AscendNPU-IR/bishengir/include/bishengir/Dialect/HIVMAVE/IR/HIVMAVEOps.td:627); [VMI result checks](/home/m00967009/Workspace/AscendNPU-IR/third-party/ptoas/lib/PTO/IR/VMI.cpp:1793).
- **Confidence / caveat:** High for the shape difference. AVE does not define a full-width splat; ADD is captured, while exact Q2 MAX lowering is unconfirmed. Evidence does not establish optimization urgency.

**AVE reduction — dqkwg excerpt (G08).**

```mlir
%sum = ave.hir.reduction <add>, %x, %all
    : vector<64xf32>, vector<64xi1> -> vector<64xf32>
```

The result retains a 64-lane carrier even though the AVE contract defines the reduced value at its first position. [Captured reduction](/home/m00967009/Workspace/Logs/q2_ave_requirements_oct19/captures/reused_dqkwg_H2_B4/before-bridge.mlir:1373).

**VMI reduction and carrier reconstruction — D1200 output (G08).**

```mlir
%one = pto.vmi.reduce_addf %x, %all {reassoc}
    : !pto.vmi.vreg<64xf32>, !pto.vmi.mask<64xpred>
      -> !pto.vmi.vreg<1xf32>
%sum = pto.vmi.broadcast %one
    : !pto.vmi.vreg<1xf32> -> !pto.vmi.vreg<64xf32>
```

The bridge reconstructs a full-width carrier from VMI's singleton result, so optimization must check whether downstream consumers actually require that broadcast. [Saved VMI output](/home/m00967009/Workspace/Logs/q2_ave_requirements_oct19/captures/norm_bwd_D1200/converted.vmi.mlir:603).

## Type and conversion support

### G09 — Normalize scalar min/max signedness consistently

- **Category / priority:** Type and conversion support; C. Inventory T01.
- **AVE behavior and example:** Apply signed scalar min/max to signless I32 operands using `ave.hir.vsmins` / `vsmaxs`. The current bridge testcase `signed_min_dynamic_scalar` supplies a runtime I32 scalar and expects rejection at the constant-only gate.
- **PTOAS limitation:** Signless normalization changes the vector to `ui32` but leaves the scalar `i32`, causing a type mismatch. Preserving signed behavior needs explicit signed carriers.
- **Implementation request:** Normalize vector/scalar operands consistently, including runtime scalars. Separately consider signed/unsigned scalar min/max operation names or a semantic attribute to carry AVE-like intent without manual wrappers; lower to the existing `.s` / `.u` intrinsics.
- **Workaround / consequence:** Explicit `si32` vector/scalar carriers and a result bitcast; the bridge still restricts scalars to constants. Same-width casts disappear in the tested LLVM path, so extra runtime conversion cost is not established.
- **Evidence:** [Inventory T01](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-evidence-inventory-2026-10-20.md:11); [current bridge testcase](/home/m00967009/Workspace/AscendNPU-IR/bishengir/test/Conversion/HIVMAVEToPTOASVMI/unsupported-signed-vector-scalar-dynamic.mlir:4); [upstream normalization](/tmp/ptoas-signed-minmax-20261020/lib/PTO/Transforms/VMI/VMINormalizeSignlessIntToUnsigned.cpp:119); [local mismatch](/tmp/ptoas-signed-minmax-20261020/signless-normalization.log:1); [runtime-scalar LLVM result](/tmp/ptoas-signed-minmax-20261020/signedness-casts.ll:25).
- **Confidence / caveat:** High for the normalization defect. Typed signed/unsigned operations already work; separate operation names are an optional interface proposal. Exact Q2 occurrence and relative priority are unestablished beyond focused bridge relevance.

**AVE signed min with a runtime scalar — current rejection testcase (G09).**

```mlir
// %scalar is an i32 function argument, not a constant.
%all = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>}
    : vector<64xi1>
%r = ave.hir.vsmins %x, %scalar, %all
    : vector<64xi32>, i32, vector<64xi1>
```

Signed ordering is encoded by `vsmins`, although both integer operand types are signless `i32`. [Bridge testcase](/home/m00967009/Workspace/AscendNPU-IR/bishengir/test/Conversion/HIVMAVEToPTOASVMI/unsupported-signed-vector-scalar-dynamic.mlir:4).

**Upstream normalization — C++ excerpt (G09).**

```cpp
for (OpOperand &use : op->getOpOperands()) {
  Value v = use.get();
  auto vreg = dyn_cast<pto::VMIVRegType>(v.getType());
  if (!vreg || !hasSignlessIntElement(vreg))
    continue;
  use.set(insertNormalizeCastBefore(op, v, rewriter));
}
```

Only vector-register operands enter normalization, leaving the scalar `i32` unchanged when the vector becomes `ui32`. [Upstream source at `739480c309c2`](/tmp/ptoas-signed-minmax-20261020/lib/PTO/Transforms/VMI/VMINormalizeSignlessIntToUnsigned.cpp:119).

**Explicit signed VMI carriers — local runtime-scalar probe (G09).**

```mlir
%signed = pto.vmi.bitcast %x
    : !pto.vmi.vreg<64xi32> -> !pto.vmi.vreg<64xsi32>
%ss = builtin.unrealized_conversion_cast %scalar : i32 to si32
%r = pto.vmi.vmins %signed, %ss, %all
    : !pto.vmi.vreg<64xsi32>, si32, !pto.vmi.mask<64xpred>
      -> !pto.vmi.vreg<64xsi32>
%bits = pto.vmi.bitcast %r
    : !pto.vmi.vreg<64xsi32> -> !pto.vmi.vreg<64xi32>
```

The explicit carriers preserve signed semantics for a runtime scalar, and these same-width casts disappear in the recorded local LLVM lowering. [Probe input](/tmp/ptoas-signed-minmax-20261020/signedness-casts.pto:2), [LLVM result](/tmp/ptoas-signed-minmax-20261020/signedness-casts.ll:25).

## Lowering and codegen correctness

### G10 — Place generated predicates inside legal nested scopes

- **Category / priority:** Lowering and codegen correctness; A. Inventory F09.
- **AVE behavior and example:** Use a constant/broadcast across nested vector computation. Saved backward D1200 VMI reaches PTOAS, then fails on an escaping `!pto.mask<b32>` introduced during lowering.
- **PTOAS limitation:** Vector-scope inference cannot place the shared generated predicate legally in the recorded nested-loop case.
- **Implementation request:** Reproduce on the supported upstream pair, then fix nested-consumer scope-anchor discovery/rematerialization if still needed. Add the saved nested-loop reproducer as a downstream regression while preserving legal value lifetimes.
- **Workaround / consequence:** The inventory records a successful manually adjusted constant-placement variant, without a general implemented repair. The original saved path cannot complete VPTO emission.
- **Evidence:** [Inventory F09](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-evidence-inventory-2026-10-20.md:20); [D1200 diagnostic](/home/m00967009/Workspace/Logs/layernorm_compile_oct15/artifacts/bwd-T8192-D1200-silu-BT1-W4-specialized.ptoas.log:1); [scope-anchor lookup](/home/m00967009/Workspace/AscendNPU-IR/third-party/ptoas/lib/PTO/Transforms/PTOInferVPTOVecScope.cpp:439).
- **Confidence / caveat:** High for the historical downstream failure; current-upstream persistence is unvalidated. October 19 VMI emission success does not establish downstream success.

**Saved D1200 VMI constant (G10).**

```mlir
%zeros = "pto.vmi.constant"()
    <{value = dense<0.0> : tensor<64xf32>}>
    : () -> !pto.vmi.vreg<64xf32>
```

The original VMI zero constant has no explicit predicate operand. [Saved failing VMI](/home/m00967009/Workspace/Logs/layernorm_compile_oct15/artifacts/bwd-T8192-D1200-silu-BT1-W4-specialized.vmi.mlir:171).

**Physical expansion — adapted from constant lowering (G10).**

```mlir
%zero = arith.constant 0.0 : f32
%all = pto.pset_b32 "PAT_ALL" : !pto.mask<b32>
%zeros = pto.vdup %zero, %all
    : f32, !pto.mask<b32> -> !pto.vreg<64xf32>
```

PTOAS introduces a physical mask for the constant, which the saved failure identifies as an escaping scope value. [Constant lowering](/home/m00967009/Workspace/AscendNPU-IR/third-party/ptoas/lib/PTO/Transforms/VMIToVPTO.cpp:5566), [recorded diagnostic](/home/m00967009/Workspace/Logs/layernorm_compile_oct15/artifacts/bwd-T8192-D1200-silu-BT1-W4-specialized.ptoas.log:1).

**Scope-anchor lookup — C++ excerpt (G10).**

```cpp
// Earlier code has checked that ancestor exists.
auto anchorIt = logicalScopeAnchors.find(ancestor);
if (anchorIt == logicalScopeAnchors.end())
  return failure();
usesBySegment[anchorIt->second].push_back(&use);
```

Rematerialization requires a known consumer-segment anchor and fails when that lookup cannot place the escaping value. [Scope inference](/home/m00967009/Workspace/AscendNPU-IR/third-party/ptoas/lib/PTO/Transforms/PTOInferVPTOVecScope.cpp:484).

## Bridge integration with existing PTOAS capabilities

This category retains confirmed bridge restrictions without asserting missing PTOAS instructions. The requests distinguish target contract/regression work from bridge implementation; evidence does not justify inventing a new VMI operation for these cases.

### G11 — Lower runtime integer metadata loads

- **Category / priority:** Bridge integration with existing PTOAS capabilities; A.
- **AVE behavior and example:** Read integer runtime metadata surrounding AVE computation. Q2 solve-varlen loads I32 sequence metadata from GM; MoE has a recorded scalar-load rejection too. These are `memref.load` operations, not AVE arithmetic instructions.
- **PTOAS limitation:** No absent integer scalar-load capability is established. The bridge's scalar-load pattern admits singleton floating types and rejects these integer loads.
- **Implementation request:** PTOAS: lock down the existing integer scalar-load type/address contract with regressions. Bridge: lower the captured I32 metadata forms through the supported scalar-load path; no new VMI load operation is requested on this evidence.
- **Workaround / consequence:** No accepted workaround is recorded for these captured forms; bridge conversion rejects.
- **Evidence:** [Inventory scalar loads](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-evidence-inventory-2026-10-20.md:28); [Q2 source example](/home/m00967009/Workspace/Q2TritonKernel/src/kernels/fla/ops/utils/solve_tril.py:150); [captured I32 loads](/home/m00967009/Workspace/Logs/q2_ave_requirements_oct19/captures/solve_varlen/before-bridge.mlir:301); [rejection](/home/m00967009/Workspace/Logs/q2_ave_requirements_oct19/captures/solve_varlen/compiler.txt:35873); [PTO scalar verifier](/home/m00967009/Workspace/AscendNPU-IR/third-party/ptoas/lib/PTO/IR/PTO.cpp:9467).
- **Confidence / caveat:** High for the bridge rejection; no target capability defect is demonstrated. This priority concerns enabling the captured Q2 path.

**Runtime I32 metadata load — Q2 solve-varlen excerpt (G11).**

```mlir
%sequence = memref.load %view[%c0]
    : memref<1xi32, strided<[1], offset: ?>, #gm>
```

This captured integer load is rejected by the bridge's floating-only scalar-load gate, although the PTO scalar verifier has no corresponding floating-only restriction. [Captured input](/home/m00967009/Workspace/Logs/q2_ave_requirements_oct19/captures/solve_varlen/before-bridge.mlir:301), [bridge gate](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Conversion/HIVMAVEToPTOASVMI/HIVMAVEToPTOASVMI.cpp:4639).

### G12 — Recognize BF16 packed-store wrappers

- **Category / priority:** Bridge integration with existing PTOAS capabilities; B. Inventory F10.
- **AVE behavior and example:** Narrow to BF16, carry packed bits through width wrappers/F32 bitcast, then `masked_store <PK_B32>`. Captured dqkwg contains this exact chain.
- **PTOAS limitation:** No inability to store BF16 is established. The bridge's packed-source recognizer and destination gate accept only F16.
- **Implementation request:** PTOAS: provide regression coverage for the dense BF16 logical-store form used after carrier removal. Bridge: recognize and fold the proven BF16 packing wrappers and permit the matching BF16 destination.
- **Workaround / consequence:** The F16 analog is supported; the BF16 mapping is absent from the inspected bridge. A separately modified compiler does not establish current support.
- **Evidence:** [Inventory F10](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-evidence-inventory-2026-10-20.md:26); [dqkwg chain, lines 899–908](/home/m00967009/Workspace/Logs/q2_ave_requirements_oct19/captures/reused_dqkwg_H2_B4/before-bridge.mlir:899); [F16 recognizer](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Conversion/HIVMAVEToPTOASVMI/HIVMAVEToPTOASVMI.cpp:1264); [store gates](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Conversion/HIVMAVEToPTOASVMI/HIVMAVEToPTOASVMI.cpp:4404).
- **Confidence / caveat:** High for the source restriction and Q2 pattern. The inventory's analog rejection is recorded evidence; this categorization did not reproduce a full-kernel failure.

**BF16 packed-store carrier — dqkwg excerpt (G12).**

```mlir
// %narrow is the BF16 result of vtruncf; attributes/views are omitted.
%expanded = builtin.unrealized_conversion_cast %narrow
    : vector<64xbf16> to vector<128xbf16>
%carrier = llvm.bitcast %expanded : vector<128xbf16> to vector<64xf32>
ave.hir.masked_store <PK_B32> %dst[%c0], %all, %carrier
    : memref<64xbf16, #map, #ub>, vector<64xi1>, vector<64xf32>
```

The F32 value carries packed BF16 bits, but the bridge's F16-only wrapper recognizer rejects the analogous BF16 mapping. [Captured chain](/home/m00967009/Workspace/Logs/q2_ave_requirements_oct19/captures/reused_dqkwg_H2_B4/before-bridge.mlir:899), [recognizer gate](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Conversion/HIVMAVEToPTOASVMI/HIVMAVEToPTOASVMI.cpp:1270).

### G13 — Preserve live values when placing memory barriers

- **Category / priority:** Bridge integration with existing PTOAS capabilities; C. Inventory F17.
- **AVE behavior and example:** Keep a loaded/computed vector live while ordering a store before a later load. The recorded recovered-dqkwg experiment reaches the bridge's `VST_VLD` barrier-placement rejection.
- **PTOAS limitation:** A new target defect is not established: rejection occurs in opt-in bridge finalization when a non-pure value cannot be safely rematerialized.
- **Implementation request:** PTOAS and bridge: specify and regression-test the legal lifetime/scope contract across `VST_VLD`. Bridge: place barriers and preserve dependencies using that contract. Identify any additional target lifetime support only after reproducing the exact crossing-value case.
- **Workaround / consequence:** No safe general workaround is recorded. The finalizer rejects this case and is omitted from the default pipeline.
- **Evidence:** [Inventory F17](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-evidence-inventory-2026-10-20.md:29); [exact bridge diagnostic](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Conversion/HIVMAVEToPTOASVMI/HIVMAVEToPTOASVMI.cpp:5255); [opt-in finalizer policy](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Tools/bishengir-compile/regbase/PassPipeline.cpp:316).
- **Confidence / caveat:** High for the rejection code; conditional for Q2 applicability. The experiment combined temporary fixes; no default-path full-kernel failure is established. This is distinct from G10.

### G14 — Materialize a singleton vector for real scalar consumers

- **Category / priority:** Bridge integration with existing PTOAS capabilities; C. Inventory F07.
- **AVE behavior and example:** A singleton-vector wrapper supplies a genuine scalar return or scalar arithmetic. Current bridge rejection testcases include `scalar-return.mlir` and `scalar-arithmetic.mlir`.
- **PTOAS limitation:** The inventory establishes bridge rejection, not absence of every possible VMI-to-scalar sequence. Wrapper folding works only when scalar uses return to matching broadcasts.
- **Implementation request:** PTOAS: specify and test a supported lane-to-scalar materialization path for these consumers; add an extraction operation only if an existing sequence is insufficient. Bridge: map genuine scalar consumers through that contract.
- **Workaround / consequence:** Immediate vector/scalar/vector round trips fold; real scalar arithmetic/returns remain rejected.
- **Evidence:** [Inventory F07](/home/m00967009/Workspace/Planner/bridge/designs/ave-ptoas-vmi-evidence-inventory-2026-10-20.md:27); [current scalar-consumer testcases](/home/m00967009/Workspace/AscendNPU-IR/bishengir/test/Conversion/HIVMAVEToPTOASVMI/preflight-reject-singleton-scalar-extraction.mlir:9); [fold restriction](/home/m00967009/Workspace/AscendNPU-IR/bishengir/lib/Conversion/HIVMAVEToPTOASVMI/HIVMAVEToPTOASVMI.cpp:4900).
- **Confidence / caveat:** High for the bridge restriction; exact Q2 demand for genuine extraction is unestablished. Evidence supports focused contract review, not a demonstrated Q2 blocker.

**Genuine scalar consumer — current bridge rejection testcase (G14).**

```mlir
%scalar = builtin.unrealized_conversion_cast %value
    : vector<1xf32> to f32
%sum = arith.addf %scalar, %scalar : f32
return %sum : f32
```

A real scalar addition cannot be handled by folding a wrapper back into a vector broadcast. [Scalar-arithmetic testcase](/home/m00967009/Workspace/AscendNPU-IR/bishengir/test/Conversion/HIVMAVEToPTOASVMI/preflight-reject-singleton-scalar-extraction.mlir:15).

The inventory's excluded or unresolved cases remain excluded. No operation is classified as unsupported merely because it is absent from tests, and no performance numbers are inferred from logical operation counts.
