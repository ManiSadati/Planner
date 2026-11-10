# AVE to PTOAS VMI Compatibility Tracker

Last updated: 2026-10-29

This file tracks AVE instruction compatibility gaps encountered so far while
lowering AscendNPU-IR AVE to PTOAS VMI. Keep entries short; detailed analysis
belongs in separate design notes.

This is also a standing input to Explorer. A PTOAS branch, issue, PR, or
upstream commit matching one of the watch triggers below is bridge-relevant
even when it does not mention AscendNPU-IR. The last broad vector-kernel
comparison baseline is `hw-native-sys/PTOAS:main` at `fc8db5e`. This review also
compared the AscendNPU-IR PTO snapshot and bridge at `57175b6c`, including the
2026-09-02 `sync_block_set`/`sync_block_wait` conversion under review, with
PTOAS at `bdcb319d6`. That comparison is evidence of interface drift, not a
replacement for end-to-end regression. Bridge workarounds remain active until
the matching conversion and PTOAS regression both pass.

| AVE instruction or contract | Compatibility gap summary | PTOAS change to watch | Current bridge handling | Fix later |
| --- | --- | --- | --- | --- |
| Signed integer `ave.hir.vsmins` / `ave.hir.vsmaxs` | Direct PTOAS VMI vector-scalar min/max is not safe for signless AVE `i32`: PTOAS normalizes the vector carrier to unsigned while leaving the scalar signless, and unsigned ordering would change signed min/max behavior. See `signed-vector-scalar-min-max-compatibility.md`. | Changes to `VMINormalizeSignlessIntToUnsigned`, VMI integer vector-scalar min/max typing, signed scalar normalization, or dynamic/partially masked scalar forms. | Emit explicitly signed vector/scalar VMI for the constant-scalar, all-active-mask form; reject dynamic scalars and partial masks. | Generalize dynamic scalars and partial masks only after a PTOAS-compatible signed scalar contract is verified. |
| `ave.hir.reduction <ADD>` / `<MAX>` | AVE reductions can keep the original vector width, while PTOAS VMI floating reductions return a one-lane vector. | Changes to unified `vcadd`/`vcmax`, legacy `reduce_*`, reduction result types, layout assignment, or reduction-combine passes that can preserve or reconstruct a wider result. | Emit the PTOAS reduction and broadcast back to the AVE result width when needed. | Revisit if PTOAS gains direct support for AVE-style reduction result shapes. |
| H-masked `ave.hir.vextf` (`PB32`, `part_even`) | Native widening accepts a predicate; logical VMI `extf` has no mask. Dropping the mask can convert inactive exceptional values and cannot generally preserve the AVE contract. See the [mapping ledger](ave-vmi-non-one-to-one-mappings.md#h-masked-b16-widening-to-same-mask-stores--2026-10-28). | Native masked B16-to-F32 VMI conversion with explicit part/layout, inactive-lane and FP-effect semantics, plus lowering regressions. | Preserve existing ALL/VL1 paths. For the guarded 64-lane UNPK_B16 H/PB32 graph whose only result uses are same-mask NORM_B32 F32 UB stores, select positive zero into inactive source lanes before unmasked `extf`; retain the original store predicate and addresses. | Keep other masks, producers, parts and consumer graphs rejected until separately proved. Replace the composition only after native masked conversion preserves this exact contract. Narrowing and general masked-conversion support remain separate work. |
| `ave.hir.vload <BRC_B16>` / `<BRC_B32>` and singleton load/broadcast round trips | Current PTOAS combines an ordinary one-lane VMI load/broadcast pair into a group broadcast load with stride zero. When a memref-derived pointer lacks an alignment proof, its slots=1 fallback rejects that stride. | Changes to VMI pre-assignment load/broadcast combining, pointer alignment analysis, and group-slot-load fallback handling of zero stride. | Emit a one-lane `pto.vmi.vload` and broadcast for wider results. In memref-preserving mode, mark broadcasts of loaded lanes as `pto.vmi.vbrc {group = 1}` after scalar-cast cleanup; retain the original load, address, shared users, result type, and B16 projection. One-lane results remain loads. | Retire the one-group workaround after PTOAS fixes the zero-stride fallback and PTODSL row-softmax/RMSNorm plus shared-load regressions pass. |
| `ave.hir.masked_store <ONEPT_B32>` | AVE has both ordinary one-point stores and continuous one-point streams. Native NPU-IR lowers a continuous stream to loop-carried `vstus.post.f32` state followed by one `vstas`, but PTOAS VMI has no alignment/base state that can represent this sequence. This occurs in the 64x64 FlashAttention max and sum reductions. See `continuous-one-point-store-compatibility.md`. | Changes to VMI scalar/one-point stores, singleton vectors, rank-zero UB destinations, alignment state, post-increment stores, or VMI lowering to VPTO `init_align`/`vstus`/`vstas`. | Emit singleton `pto.vmi.vstore`, which PTOAS lowers to `pto.vsts {dist = "1PT_B32"}`. Insert store/load memory barriers where required by the current bridge dependency handling. This covers the ordinary one-point form but does not preserve the native continuous-store instruction sequence. | Add a state-carrying VMI representation that lowers to the existing VPTO `init_align`/`vstus`/`vstas` operations. Do not introduce a bridge-only template or bypass VMI by emitting physical VPTO operations directly. |
| Predicate-valued UB load/store (`PB8`, `PB16`, `PB32`) | AVE can store a predicate itself in packed UB memory and reload it with bit-addressed offsets. PTOAS VMI masks can control numeric memory operations, but VMI has no operation whose stored value or loaded result is a mask. This occurs in masked FlashAttention. See `predicate-memory-store-compatibility.md` and the bring-up context in `flash-attention-bridge-summary.md`. | VMI predicate-memory load/store operations, mask-valued memory operands/results, PB8/PB16/PB32 granularity, bit addressing, or VMI lowering to VPTO `psts`/`pstu`/`vstas`/`plds`. | Deferred at the human's request on 2026-10-29; recent predicate compiler/tests/runner changes were rolled back. The baseline zero/one byte carrier and experimental physical finalizer remain historical/opt-in mechanisms; default compilation omits the finalizers and native packed-memory equivalence is not established. No direct-physical exception or PTOAS mask-boundary work is authorized. | Add first-class predicate-memory operations to PTOAS VMI and lower them normally to the existing VPTO operations. Then remove the byte carrier, pointer marker, and bridge-owned post-PTOAS finalizer. Do not replace native instruction lowering with a bridge-only PTODSL template. |
| Partial `pge <VL*>` masks on AVE arithmetic | PTOAS accepts masks on unified VMI arithmetic, but its unified-to-legacy expansion currently discards masks for binary operations and most unary operations. The bridge does not enforce the narrow condition under which this is harmless. | Changes to unified VMI arithmetic expansion, inactive-lane/passthrough semantics, `pmode`, or native masked VPTO arithmetic. | Preserve the AVE mask in emitted VMI. This is only known safe when inactive lanes are never observed, such as a straight-line tail whose final store uses the same mask. | Either make PTOAS preserve inactive-lane semantics or add a proven bridge expansion. Until then, reject partial-mask arithmetic unless use analysis proves inactive lanes cannot escape. |
| Stored scalar `f32 -inf` | The validated softmax simulator path produced invalid results when the exact negative-infinity initializer was stored through `pto.store_scalar`. Replacing it changes IEEE behavior for `-inf` and NaN inputs. | Changes to scalar constant/store lowering, VPTO/LLVM constant materialization, and simulator handling of floating infinities. | Rewrite any `f32 -inf` that directly feeds `pto.store_scalar` to `-FLT_MAX`. The implementation is broader than the intended reduction-initializer use. | Remove the rewrite after PTOAS carries `-inf` correctly. Before then, constrain it to a proven max-reduction initializer and add exceptional-value tests. |
| `hivm.hir.sync_block_set` / `sync_block_wait` | The source carries core kind, producer and consumer pipes, sync mode, and an optional FFTS base operand. The vendored PTO `sync.wait` cannot carry `ffts_mode`, and neither target op carries the FFTS base. Current PTOAS also restricts A5 static FFTS IDs to `[0, 15]`, while the vendored verifier and bridge allow `[0, 31]`. | Changes to `SyncSetOp`/`SyncWaitOp`, named `set_cross_block`/`wait_cross_block` and `set_intra_block`/`wait_intra_block` ops, FFTS event ranges, and A5 sync lowering. | Emit `pto.sync.set` on the source `tpipe` and `pto.sync.wait` on the source consumer `pipe`; carry the numeric mode only on set; ignore core kind, the opposite pipe, and `ffts_base_addr`; accept static IDs through 31. IDs used by the current smoke test reach PTOAS VPTO, but the general mapping is not proven. | Refresh the PTO snapshot, preserve matching set/wait mode, enforce the target-specific ID range, and map to the named operation when appropriate. Reject any source FFTS base or core/pipe form whose semantics cannot be represented and verified. |
| Vendored PTO dialect snapshot | AscendNPU-IR compiles against a copied PTO dialect while the emitted text is consumed by a separately evolving PTOAS. The reviewed trees already differ in sync operations/verifiers and VMI memory operands, so local verification can accept IR that current PTOAS rejects or interprets differently. | Any PTOAS IR, parser, verifier, builder, normalization, or pipeline change; currently especially sync mode/event changes and removal of VMI memory `repeat_stride`. | Keep using the in-tree snapshot so AscendNPU-IR and its MLIR version remain build-independent. Compatibility is checked later by invoking PTOAS on emitted text. | Record an exact PTOAS source revision for each snapshot, refresh headers and implementations as one unit, and add a CI test that sends representative bridge output through that exact PTOAS revision. |
| Memref descriptor to `!pto.ptr` collapse | Dynamic memref offsets, sizes, and strides are not representable in the pointer-only VMI ABI. The bridge treats converted arguments as normalized logical base pointers and may synthesize zero for unavailable metadata. This is correct only if the preceding NPU-IR pipeline has already incorporated the descriptor offset and unused metadata stays unused. | PTO pointer/view ABI, descriptor-bearing PTO types, dynamic-shape materialization, or VMI memory addressing changes. | Accept a narrow rank-one/unit-stride path, reconstruct explicit `pto.addptr` offsets where available, and reject several used dynamic-size/noncontiguous forms. | State and verify the normalized-base invariant at the pass boundary, reject any used metadata that cannot be reconstructed, or adopt a descriptor-preserving PTO ABI. Add nonzero-offset and dynamic-layout end-to-end tests. |
| External CCE-call memref ABI | External Cube templates require ranked memrefs while the PTO kernel boundary is pointer-based. The resulting pointer-to-memref `unrealized_conversion_cast` and `pto.preserve_external_call_memrefs` marker are bridge-private conventions, not a general PTOAS VMI contract. | PTOAS external-call ABI, CCE template integration, memref materialization, and unknown/residual cast handling. | In external-call mode, preserve PTO-address-space memrefs internally, change kernel arguments back to PTO pointers, and leave controlled pointer-to-memref casts at entry. | Replace bridge-private residual casts with an explicit supported ABI/materialization operation, or retire this compatibility route once the PTO-native Cube path covers the required kernels. |
| PTO module target/kernel metadata | PTOAS `pto.kernel_kind` has only `cube` and `vector`; mixed VPTO is represented structurally and normalized into one child module per kind. The bridge still hard-codes `pto.target_arch = "a5"`. | PTO target/device-spec schema, function core-kind rules, section splitting, and mixed Cube/Vector module packaging. | Preserve `hivm.func_core_type` through conversion. Pure modules receive one module-level kind. Mixed modules omit the outer kind and wrap every defined AIC/AIV function in `pto.section.cube`/`pto.section.vector`; PTOAS then creates the canonical child modules. Reject unsplit `MIX`, ambiguous, or unclassified definitions. | Derive the target from the confirmed source device specification. Revisit the single-block/void-function restriction if a mixed kernel reaches the bridge with a more general CFG or result ABI. |
| Row-split f32-to-f16 L0C-to-UB fixpipe | AVE represents the FP32-to-FP16 accumulator transfer as `hivm.hir.fixpipe` with `pre_quant = F322F16`, `dual_dst_mode = ROW_SPLIT`, source shape `4x4x16x16xf32`, and per-subblock UB destination shape `32x64xf16`. PTOAS `pto.mte_l0c_ub` dual-destination modes reject explicit `pre_quant` operands. | Changes to `pto.mte_l0c_ub` split-mode verification, FP32-to-FP16 backend intrinsic selection, or accumulator-store type rules. | Emit `pto.mte_l0c_ub` with `dst_mode(split_m)`, `nz2nd`, logical `m = n = 64`, f32 L0C source, f16 UB destination, and no explicit `pre_quant`. PTOAS chooses the f32-to-f16 intrinsic from the destination pointer type. | Generalize only after proving non-64 tile sizes, column split, pre-ReLU, channel split, C0 tail padding, and other quant modes. |
| Normal F32-to-BF16 L0C-to-CBUF Fixpipe | Native NPU-IR registers `fixpipe_normal_float_to_bfloat16_t_4d_to_4d_cbuf`, but PTODSL generation fails because the shared PTOAS `nz2nz` verifier requires an F32 destination and rejects BF16 L1. See [the issue note](fixpipe-normal-f32-bf16-cbuf-compatibility.md). | Changes to structured accumulator-store `nz2nz` verification, `pto.mte_l0c_l1` destination typing, and F32-to-BF16 intrinsic selection and lowering. | The attempted PTODSL instance is blocked at generation; runtime parity is unverified. | Admit the precise supported F32 L0C-to-BF16 L1 normal-layout conversion and validate generation, target lowering, numerical conversion, supported controls, and existing F32 CBUF regressions before enabling the instance. |
| Masked UB stores through `arith.select` pointers | PTOAS pointer-remainder analysis cannot trace `arith.select`, so VMI-to-VPTO lowering rejects masked stores even when every selected buffer and access offset preserves 32-byte alignment. Observed in F32 Fixpipe vector consumers; see [the issue note](selected-pointer-masked-store-alignment-compatibility.md). | Changes to `PTOAddressAnalysis.cpp` pointer-remainder analysis through selections, including nested selections, and masked-store alignment checks. | NPU-IR bridge finalization branches eligible masked stores on the selected UB buffer, preserving value, mask, offset, and attributes while exposing directly traceable aligned destinations. This can duplicate code; performance impact was not measured. | Selection-aware pointer alignment remains unsupported in the recorded PTOAS analysis. The bridge workaround remains necessary for the affected stores until equivalent alignment handling is available and validated. |

The scalar-load broadcast workaround was compiled on `master` at
`fe4bde154` plus the working fix against candidate PTOAS `62df5d41` plus its
local ABI fixes. PTODSL row-softmax and RMSNorm reach device objects with no
undefined symbols. This is compilation evidence, not numerical or hardware
validation; see the [validation report](../../../Logs/run_reports/ave-loaded-broadcast-fix-e9igul_6/report.md).

## Review Priority

1. Provide a PTOAS VMI representation for predicate-valued memory, including
   PB width and bit-address semantics, so the post-PTOAS carrier finalizer can
   be removed.
2. Provide or confirm a PTOAS VMI representation for continuous ONEPT store
   state before treating the FlashAttention max and sum reduction stores as an
   exact native-to-bridge mapping.
3. Resolve the sync snapshot/range/mode mismatch and silent source-field loss
   before treating FlashAttention block synchronization as complete.
4. Prevent partial arithmetic masks from reaching a PTOAS path that discards
   them unless inactive lanes are proven unobservable.
5. Narrow and ultimately remove the `-inf` to `-FLT_MAX` rewrite before claiming
   IEEE exceptional-value coverage.
6. Automate PTO dialect snapshot provenance and bridge-output compatibility
   testing; local AscendNPU-IR verification is not sufficient.
7. Generalize pointer ABI and target metadata only when kernels outside the
   currently proven A5 normalized-base contract require it.

## Tracking Rules

- Explorer should distinguish proposed support, merged support, and
  bridge-validated support. A merged PTOAS change does not automatically retire
  the NPU-IR workaround.
- Mention the affected tracker row in the daily report and link the exact PTOAS
  issue, PR, branch, or commit.
- Promote an exact semantic match to `Investigate`. Use `Watch` when only a
  related pass or contract changed.
- Remove or simplify bridge handling only after the corresponding AscendNPU-IR
  conversion test and PTOAS lowering test pass on the updated baseline.

The PTOAS-side overview is `PTOAS/design/lowering-pipeline.md`; detailed bridge
decisions remain in `bridge/designs/ave-to-ptoas-vmi-conversion-design.md`.

## Q2 I32-to-F32 rounding gap — 2026-10-29

The sampling capture `entry_9bdd01c8__top_p_sample_kernel` needs
`ave.hir.vsitofp` I32-to-F32 with VL1 and TRUNC rounding. VMI `sitofp` has
neither mask nor rounding operands; current PTOAS
`VMIToVPTOPatternInternals6b.cpp::lowerSameWidth` hardcodes `R` (nearest-even).
Unified `vcvt` also forbids rounding for this type pair and dispatches to that
same operation. This is a representation gap, not permission to discard TRUNC.
The current bridge supports only ALL/PB32, 64-lane I32-to-F32 with RINT, using
a signedness bitcast plus sitofp. Watch upstream explicit integer-to-float
rounding/mask support; a proper extension must retain rounding through layout
and physical lowering. Partial masks need a separate inactive-lane proof.
No PTOAS changes were made. Evidence: NPU-IR baseline `ed872278c`, standalone
PTOAS `40f3000a6`, [item 8 report](../../../Logs/run_reports/q2-item08-vsitofp-20261029/report.txt).

## Gather index signedness

The VMI vgather surface accepts signless I16 offsets, but current B16
physical gather contract checks require unsigned I16. The bridge emits an
explicit same-width unsigned bitcast (also for I32) to preserve native
unsigned element-index semantics. Watch VMI unified/legacy index signedness
normalization and gather contract checks. This is no numeric widening.
[Evidence](../../../Logs/run_reports/q2-indexed-gather-20261030/report.txt).

## Native I64 logical representation — 2026-10-30

Current VMI has no direct I64 vector-register representation for the native
`vector_2xvl_s64` ABI. The approved bridge workaround recognizes closed
cast/broadcast/add/NORM_B64-store graphs and composes paired I32 values,
carry propagation, explicit inactive zeroing, and interleaved I32 stores.
Masks are restricted to observed ALL/VL4 PB32 forms. This does not add general
I64 arithmetic, escaped stack state, or new predicate-memory support.
Watch for logical multiword/I64 representation and memory lowering that preserve
both words, carry, masks, and element addressing. Existing descriptor-store
alignment limitations remain deferred. No PTOAS change was made.
[Evidence and limits](../../../Logs/run_reports/q2-i64-helper-20261030/report.md).

## I16 boolean carriers consumed under PB32 — 2026-10-30

VMI already has integer comparison operations; this bridge composition adapts
the native carrier/mask contract rather than adding a missing I16 instruction.
For the guarded top-p zero/one carrier graph, widen logical I16 values to I32
and compare under the original VL2/PB32 seed. F32 consumers then keep the same
predicate granularity and inactive lanes. Do not generalize to arbitrary I16
vectors, B16 predicate users, ordering comparisons or ALL-seed reconstruction.
Watch mixed-width mask materialization that preserves the original partial
seed and native observable lane projection. No PTOAS changes were made.
[Evidence and limits](../../../Logs/run_reports/q2-i16-boolean-cmp-20261030/report.md).

## FP8 native carrier compaction — 2026-10-30

Current PTOAS accepts the typed FP8 logical strided store. The bridge now folds
the guarded native two-self-deinterleave compaction graph, because VMI narrowing
already produces dense logical FP8 values. Preserve explicit VL64 byte masking
and stride conversion (2080 elements = 65 blocks); native ALL could expose
padded bytes. This is a carrier/mask composition, not a missing FP8 store
instruction. General sparse carriers and arbitrary FP8 deinterleave forms stay
unsupported. Watch packing-layout contracts and exact short-vector store tails.
[Evidence and limits](../../../Logs/run_reports/q2-fp8-compaction-20261030/report.md).

## Guarded masked BF16 widening: VL16 — 2026-10-30

The existing zero-before-unmasked-extf composition now accepts BF16 VL16/PB32
UNPK_B16 load graphs whose results reach only matching-mask stores through
the existing bounded lane-wise graph. This extends a bridge guard, not general
VMI masked conversion support. F16 VL16, escapes and changed masks remain
rejected. Continue watching explicit masked VMI widening with compatible
part/layout, inactive-lane and FP-effect semantics; no PTOAS changes were made.
[Evidence and limits](../../../Logs/run_reports/q2-bf16-vl16-widening-20261030/report.md).

## Short BF16 gather/interleave/widen carrier — 2026-10-30

The bridge folds the guarded Delta native self-interleave before part_even
widening of 64 BF16 gather values. Logical VMI can express the dense gather
and widening directly; a logical short interleave would split after 32 original
values and change lane order. PTOAS owns physical layout adaptation. This is
not general short-vector interleave support or a newly missing VMI instruction.
Keep unused-high, single-widening, exact-mask/type/use guards. No PTOAS changes.
[Evidence and limits](../../../Logs/run_reports/q2-delta-carrier-fold-20261030/report.md).

## Full-register BRC_B16/NORM numeric loads — 2026-10-30

The existing scalar-load plus native-width broadcast composition now accepts
NORM for 128-lane numeric B16 results. This fixes a bridge attribute guard,
not a missing target instruction. Keep singleton UB/type/index checks and
existing PK forms; shortened NORM carriers remain unsupported. No PTOAS change.
[Evidence and limits](../../../Logs/run_reports/q2-brc-b16-norm-20261030/report.md).

## TRUNC SI32-to-F32 observable only through F16 — 2026-10-30

VMI sitofp still lacks a mask/rounding contract and lowers SI32-to-F32 using
nearest. A new guarded bridge composition handles only VL1/PB32 TRUNC values
whose sole F32 use is RINT/NOSAT F16 narrowing to matching singleton point
stores. Zero inactive integers, use nearest, retain F16 conversion/store.
Final half values agree for all I32 inputs: F32-exact magnitudes agree first;
larger magnitudes overflow F16 in either path. Intermediate F32 can differ,
so general TRUNC support, BF16 consumers and escaping F32 uses stay rejected.
Continue watching explicit masked/rounded signed conversion in VMI/physical
lowering. No PTOAS changes. Target numerical/FP-status tests remain unrun.
[Evidence and limits](../../../Logs/run_reports/q2-trunc-half-chain-20261030/report.md).

## Scalar I32 left-shift bridge coverage — 2026-10-30

VMI already provides the required scalar left shift. A new bounded bridge
pattern maps 64-lane signless I32 `vshls` with constant sign/count 0–31 and a
matching direct PGE/PB32 mask through signed/unsigned bitviews to existing
`pto.vmi.vshls`. This is direct instruction coverage, not a missing VMI
capability. No PTOAS change. Committed as `b7729ce6b` on NPU-IR base
`08dfca88c`; 154 conversion tests pass and signed/unsigned memory probes emit
LLVM in both modes. The original Mamba capture now stops at H/PB32 RINT
`vsitofp`; whole-capture replay stays 21/62 (NPU 20/61). Masked sitofp needs
its own inactive-lane proof before any composition is implemented.
[Evidence and limits](../../../Logs/run_reports/q2-left-shift-20261030/report.md).

## Masked H/PB32 RINT signed I32-to-F32 — 2026-10-30

Current VMI sitofp still lacks an explicit predicate. The approved guarded
bridge composition selects I32 zero for inactive H lanes before signed nearest
conversion, then retains a closed lane-wise F32 graph ending in same-mask
numeric UB stores. Active conversion bits and RINT match directly; inactive
zeros convert exactly and cannot affect stored output. ALL/RINT and the
separate VL1/TRUNC-to-F16 workaround stay unchanged. Returns, calls, reductions,
mask changes and unrelated rounding/types remain rejected. No PTOAS change;
proper masked/rounded VMI conversion remains the upstream solution.

Committed as `abcacb918` on NPU-IR base `b7729ce6b`; 156 tests pass, memory
probes emit LLVM in both modes, and the original Mamba fused-2 VF emits LLVM in
pointer mode. Descriptor VF emission retains the existing store-alignment
rejection. Full Mamba now stops at short F16 self-interleave; replay remains
21/62 (NPU 20/61). Target numerical/FP-status validation was not performed.
[Evidence and limits](../../../Logs/run_reports/q2-masked-rint-sitofp-20261030/report.md).

## Dense unaligned F16 short carrier fold — 2026-10-30

The existing logical VMI load/extf surface can represent the proved active
values after a guarded carrier fold. Ascend950 unaligned-marked UNPK_B16 uses
dense vldus.post in native lowering; self-interleave's low result plus part_even
widening recovers the dense source values. The new bounded composition handles
only the two-value VL2/PB32 F16 UB graph, selects inactive F16 zero before
unmasked extf, and retains matching F32 stores. Ordinary aligned unpack is a
different carrier and stays excluded. This is a native/logical layout adapter,
not a new VMI interleave instruction. No PTOAS changes.

Committed as `fcad33231` on NPU-IR base `abcacb918`; 158 tests pass.
Aligned/offset memory probes emit LLVM in both modes; original Mamba VFs 4/9
convert in both modes. Unaligned/dynamic stores retain the proven-alignment
rejection. Full Mamba now exposes numeric I32 ONEPT_B32 on a rank-two singleton
view. Capture replay remains 21/62 (NPU 20/61); target FP status is untested.
[Evidence and limits](../../../Logs/run_reports/q2-f16-load-carrier-fold-20261030/report.md).

## I32 ONEPT_B32 numeric stores — 2026-10-30

Existing VMI shuffle/lane-zero projection plus one-element vstore represents
the required I32 write. A guarded bridge extension reuses the F32 point-store
composition for matching signless I32 UB data, 1/64 lanes and direct matching
VL1/PB32 predicates. Keep NORM/unaligned/continuous attribute checks and existing
address normalization; reject other masks/types/carriers. No floating bitcasts,
predicate-memory changes, alignment bypass or new target instruction.

Committed as `70f1b643e` on NPU-IR base `fcad33231`; 160 conversion tests
pass and five memory probes emit LLVM in both modes, with one-word stores and
constant/dynamic offsets applied once. Original Mamba VF 6 emits LLVM in pointer
mode; descriptor mode retains a memref.reinterpret_cast elimination blocker.
Full Mamba now exposes a separate F16 PK_B32 direct-carrier store. Replay stays
21/62 (NPU 20/61); no numerical/hardware execution was performed.
[Evidence and limits](../../../Logs/run_reports/q2-i32-onepoint-store-20261030/report.md).

## Dense F16 unaligned PK_B32 prefix carrier — 2026-10-30

The existing VMI narrowing/store operations represent this closed native
carrier graph after a guarded fold. On Ascend950 an unaligned-marked PK_B32
store uses VSTUS with a byte count: VL2/F16 writes four dense bytes. Keep
64-lane ALL/PB32 RINT/NOSAT part_even narrowing, remove only its closed
self-deinterleave carrier, and store the first two logical F16 values under
VL2. Ordinary aligned PK_B32 has a different source projection and stays
excluded. This uses the existing bridge graph-preparation phase; no Q2 pass
or PTOAS change is needed. Global masked/packed-memory support is not implied.

Committed as `6fc233981` on NPU-IR base `70f1b643e`; 162 tests pass.
Aligned/offset probes emit LLVM in both modes, preserving all-lane conversion
and two-F16 write footprint. Original Mamba VF 7 converts in both modes;
unaligned/dynamic destinations retain the LLVM alignment rejection. Full
Mamba now reaches memref.get_global for its 320-I32 GM constant table. Replay
remains 21/62 (NPU 20/61); no target numerical execution was performed.
[Evidence and limits](../../../Logs/run_reports/q2-f16-packed-prefix-20261030/report.md).

## Constant I32 GM table: downstream capability probe — 2026-10-30

The next Mamba rejection is memref.get_global for its private constant
320-I32 GM table. Five standalone compile-only probes pass with current PTOAS:
LLVM global/address/load, LLVM address adapted to PTO GM load, standard memref
global/get_global/load with numeric memory space 1, exact Q2 table payload and
a pointer-load control. Emitted LLVM retains all 1280 initializer bytes,
private constant linkage, 32-byte alignment and GM address space 1. Dynamic GM
loads reach observable UB stores without native C-interface helpers.

This tested capability needs no PTOAS implementation change. Bridge global
legalization/address adaptation remains unimplemented; numeric-space versus
LLVM-global representation must preserve the original GM→UB helper ABI in
both bridge modes. No DMA, object/link, loader, hardware or numerical check was
performed, and no Q2 pass count changed. Compiler worktrees remain clean.
[Probe evidence and provenance](../../../Logs/run_reports/q2-global-table-probe-20261030/report.md).


## Constant I32 GM table: bridge implementation — 2026-10-30

The successful downstream global probe is now used by a bounded bridge
adapter: private initialized dense rank-one identity I32 HIVM GM globals
become LLVM constant arrays; each address uses addressof → i64 ptrtoint →
existing pointer-cast lowering. Preserve symbol/initializer/alignment and
original views. Reject mutable/public/external/uninitialized/other storage
and unknown definition/access attributes. This repairs a bridge legalization
gap using existing target support, rather than adding a VMI instruction.

Committed as `c10c70934` on NPU-IR base `6fc233981`; 165 conversion tests
pass. Small/exact-table scalar and native-copy probes emit LLVM in both modes.
The 320-word initializer is bit-exact. Preserved-mode bridge calls retain
memref descriptors, but current standalone PTOAS emits pointer-valued native
wrapper parameters and leaves the external `_mlir_ciface_*` dependency;
linking/native ABI correctness is not established. Any upstream ABI repair is
a separate task, not an implicit part of global support. Full preserved-mode
Mamba bridge conversion now passes; downstream LLVM emission still rejects
masked-store alignment. Capture replay improves 21/62 → 22/62 (NPU 21/61),
with no regressions/timeouts. No object/link/runtime claim or deferred-policy
change. Numeric-space/LLVM globals remain outside the adapter.
[Evidence, commands and limits](../../../Logs/run_reports/q2-readonly-i32-global-20261030/report.md).


## Numeric I32-to-I8 packed-store graph — probe, 2026-11-02

Committed as `0032a7750` on NPU-IR base `a0876a760`: a guarded graph adapter
uses existing VMI trunci NOSAT and a 64-lane UI8 store for the observed
ALL/PB32 SI2UI PP0 / aligned PK4_B32 closed carrier. No target instruction or
PTOAS implementation change is needed. The source unaligned marker remains
excluded because native lowering writes256 bytes. Preserve source ranges,
views, exact64-byte writes, byte bits and native descriptor types; reject
other masks/parts/types/layouts/escapes/attributes. Preserved numeric I8 UB
address carriers retain I8 while memory consumers use UI8 byte pointers.

170 conversion tests pass. Sixteen memory probes emit LLVM across both bridge
modes and both PTOAS builds; original VF pointer mode emits helper-free LLVM.
Preserved VF still hits a separate alignment rejection. Full Mamba bridge
conversion now passes in preserved mode, improving the fresh old/new capture
replay 22/62 →23/62 (NPU22/61), no regressions/timeouts. Full-kernel pinned LLVM
still rejects a dynamic external-call pointer bridge; standalone rejects a
surviving reinterpret view. Those native-call/view lowerings and direct pinned
memref-load support remain separate from the now-supported numeric byte
conversion. No object/link/runtime or numerical claim and no policy bypass.
[Exact guards, evidence and remaining blockers](../../../Logs/run_reports/q2-i32-i8-packed-store-20261102/report.md).


## Direct aligned F16 VL2 widening — 2026-11-02

The bridge now admits the observed two-element aligned UNPK_B16 → part_even
F16/F32 widening → VL2/PB32 NORM_B32 store graph, reusing existing select-zero
and unmasked VMI extf. It is a bounded extension to existing mask coverage;
no new VMI instruction or PTOAS implementation is required. The previous
unaligned/interleave VL2 path is distinct; other masks/consumers and PLT63
stay rejected by this new subcase.

Implemented in NPU-IR commit `94b01f69d` (base `dc304dd99`);
172 conversion tests pass,
12 memory probes emit LLVM across both modes/builds, and both original Top-P
VFs emit LLVM in pointer mode. Preserved VF LLVM retains an unproven-store-
alignment rejection. Replay remains 23/62 without regressions because full
sampling/filtering captures expose later I32 rank-zero-cast and masked
narrowing gaps. No object/link/runtime or numerical claim. General masked
conversion remains bounded by its producer/consumer proof and the unmasked
VMI interface; no global mask legality relaxation or alignment bypass.
[Contracts, native controls and stage results](../../../Logs/run_reports/q2-f16-vl2-extf-20261102/report.md).


## Scoped dynamic indirect-load descriptor workaround — 2026-11-02

Pinned PTOAS still rejects dynamic pointer carriers feeding generic descriptor
consumers; the check is unchanged. The approved bridge adapter handles only
indirect_load_nonvolatile_1d_half_int64_t's GM F16 source, whose native source
and installed bitcode read only aligned pointer. An explicit base-only sentinel
view satisfies existing metadata-base/reinterpret validation while retaining
the call ABI and all real UB descriptors. It is not a valid generic replacement
for unknown extents or a new predicate/memory instruction.

Both affected saved frontend recipes produce pinned ELF device objects;197
conversion tests and the LLVM descriptor-field audit pass. Unknown helpers,
wrong signatures and shared unknown users still reject. Standalone retains
emission-stage reinterpret-cast validation failure. Native-contract changes
require adapter revalidation; other dynamic descriptor consumers need genuine
metadata or separately verified buffer-only contracts. Compiler source/tests committed as `882396b41`; Planner notes remain local; no PTOAS source change or full-sweep/runtime claim.
[Detailed evidence](../../../Logs/run_reports/q2-indirect-buffer-adapter-20261102/report.md).
