# Triton-Ascend 3.6 Filtered Template Effort Subgroups

Recorded: 2026-10-23. This is a filtered planning edition of
[the original 191-symbol effort breakdown](native-template-effort-subgroups-2026-10-23.md).
It removes 16 historical non-A5/unresolved identities from the active estimate:
**175 retained symbols in 63 subgroups**. The original census and effort document
are preserved unchanged.

## Filter And Evidence

The original census recorded calls demanded by lowered IR, not successful
native linking or numerical support. "Not catalogued" did not itself mean
"unsupported by native NPU-IR."

For this edition, the 16 excluded symbols satisfy both checks:

1. No exact definition in any of the frozen campaign's AIC, AIV, mixed-AIC
   or mixed-AIV **C310/A5** bitcode bundles.
2. No occurrence in the successful checkpoints of the completed Triton-Ascend
   3.6 Q2 campaign: 28,751 configurations across 33 retained function paths,
   22,684 complete outcomes, 6,067 compile failures, and 42 observed symbols.

These 16 happen to equal the historical CSV's `catalogued=false` set, but the
filter is backed by definition/capture checks, not that flag alone. Fourteen
broadcast/transpose/addition helpers exist in **C220** bundles; they are not
globally nonexistent native functions. The two F16-accumulator symbols were
absent from all eight audited C310/C220 bundles. See the exclusion ledger below.

The project owner reports that the legacy calls are no longer generated with
3.6. The bounded Q2 evidence corroborates non-observation in its retained scope;
it is not a universal proof of absence across Q3, Triton-Ascend or all NPU-IR
tests. Failed/excluded configurations are unknown, not zero dependencies.
Conversely, all other historical requirements remain in the estimate even if
this narrower Q2 campaign did not exercise them.

This is therefore **old scope minus identified legacy exclusions**, not a new
complete 3.6 census or a native-supported/PTODSL-supported list. Presence of a
bitcode definition does not prove a successful final link, supported controls,
runtime correctness or bridge replacement.

## Reuse And Effort Legend

The first implementation in a subgroup pays for the native algorithm, PTO/PTODSL
expression, adapter and test harness. Later datatype/rank/transpose siblings
should be cheaper once that contract is genuinely established. Groups are
engineering judgments, not file-layout changes or guaranteed mechanical ports.

The effort column is **first / follow-on incremental effort**, not hours:

| Rating | Meaning |
| --- | --- |
| L | Mostly typed/rank wrapper, registration, adapter reuse and focused verification. |
| M | A distinct layout, mode, descriptor path or numerical contract needs work. |
| H | Algorithm/state-machine work, new test infrastructure or a difficult backend boundary. |
| R | Resolve feasibility/provenance before estimating implementation. |

Single-member groups estimate possible future siblings. Low follow-on effort
does not waive the complete agreed instance-validation matrix. Existing partial
implementations reduce only the work actually completed and verified.

## Active Summary

| Census category | Retained symbols | Subgroups | Main reuse boundary |
| --- | ---: | ---: | --- |
| MmadL1 | 15 | 4 | F16/F32, BF16/F32, F32/F32, INT8/I32 arithmetic and packing. |
| Fixpipe | 7 | 5 | Single/dual, F32/I32, UB/GM/L1 and layout contracts. |
| Nd2Nz | 4 | 3 | 16-bit, 32-bit and byte packing. |
| Cube setup | 4 | 3 | Element-width-specific fill and padding. |
| Cube copy | 1 | 1 | GM-to-L1 movement. |
| Vector DMA | 52 | 13 | Direction, rank, B64 padding and UB/L1 transfer. |
| Vector Integer64 | 46 | 15 | Shared B64 representation, then operation-specific behavior. |
| Vector collective | 9 | 7 | Scan dimension/type, product, reversal and sorting. |
| Vector math | 3 | 2 | Integer division/remainder versus floating division. |
| SIMT indexing | 1 | 1 | SIMT execution and indirect addressing. |
| Support: printing | 25 | 5 | Formatting width, rank and core/memory-space ABI. |
| Support: cube debug | 4 | 2 | Unsuffixed versus Cube-qualified runtime identity. |
| Support: vector debug | 2 | 1 | Vector-qualified runtime identity. |
| Support: synchronization | 2 | 1 | One lock/unlock protocol and resource contract. |
| **Active total** | **175** | **63** | **Planning buckets, not independent greenfield algorithms.** |

Original subgroup IDs are retained for cross-reference; M5, F6 and O1-O5 are
intentionally absent. Every retained symbol appears exactly once below.

## MmadL1

One shared foundation covers native instance selection, descriptor adaptation,
L0 capacity planning, shared L0A/L0B scratch state, internal K slicing, and
paired producer/consumer synchronization. Do not budget a new selector for each
datatype. The physical L0C accumulator selection is not the L0A/L0B bank counter.

| ID: subgroup; effort | Exact symbols | First-instance hurdle | Follow-on work and decisive tests |
| --- | --- | --- | --- |
| M1: F16 inputs, F32 accumulator; H / L-M | `mma_tile_half_to_float`, `mma_tile_half_to_float_ta`, `mma_tile_half_to_float_tb`, `mma_tile_half_to_float_ta_tb` | Establish the complete native staging, transpose, initialization/accumulation and synchronization contract, not just a 64x64 MAD. | Reuse the body for TA/TB/TA+TB, but verify each operand's packing, non-square tiles, tails, caller/internal K loops, flags and shared scratch. |
| M2: BF16 inputs, F32 accumulator; M / L-M | `mma_tile_bfloat16_t_to_float`, `mma_tile_bfloat16_t_to_float_ta`, `mma_tile_bfloat16_t_to_float_tb`, `mma_tile_bfloat16_t_to_float_ta_tb` | Reuse 16-bit staging from M1; establish BF16 instruction typing and numerical reference. Do not treat a bit reinterpretation as numerical equivalence. | Transpose machinery and harness should transfer after BF16 is proved. Include BF16-sensitive inputs and the same controls as M1. |
| M3: F32 inputs, F32 accumulator; H / M | `mma_tile_float_to_float`, `mma_tile_float_to_float_ta`, `mma_tile_float_to_float_tb`, `mma_tile_float_to_float_ta_tb` | Different element width, staging/capacity constraints and precision handling. Verify native IEEE semantics and actual selected symbol. | Reuse F32 packing across transpose variants; rerun split-K/full-bank and accuracy cases. HF32 is a control/precision extension outside these observed symbol names, not an extra counted census member. |
| M4: signed INT8 inputs, I32 accumulator; H / M | `mma_tile_int8_t_to_int32_t`, `mma_tile_int8_t_to_int32_t_ta`, `mma_tile_int8_t_to_int32_t_tb` | Byte packing, integer accumulation, segmented transfers and an I32 Fixpipe dependency make this a real boundary from M1/M2. | Reuse integer packing for transpose variants, not float assumptions. Test negative/high-magnitude inputs, byte-layout tails, varied blocks and native-supported accumulation. |

TA/TB is expected to be cheaper **after that subgroup's physical layout is
correct**, not automatically cheap across F16, F32 and INT8. Existing bodies
also do not imply completed unit-flag, scratch-reuse or full-matrix validation.
Current control migrations and runtime campaigns are maintained separately in
Planner's bridge memory. Reconcile their latest evidence before subtracting
completed work; this filtered classification does not recertify template support.

## Fixpipe

Shared foundation: L0C layout/strides, valid M/N, output pitch, transfer controls
and native unit-flag 0/2/3 behavior. A dual transfer is not just renaming a single
one: participant/destination selection and synchronization need their own proof.

| ID: subgroup; effort | Exact symbols | First-instance hurdle | Follow-on work and decisive tests |
| --- | --- | --- | --- |
| F1: single F32 NZ2ND, UB/GM; M / M | `fixpipe_nz2nd_float_to_float_4d_to_2d_ubuf`, `fixpipe_nz2nd_float_to_float_4d_to_2d_gm` | Correct NZ-to-ND shape/pitch extraction and flags; UB and GM use different destination/address-space contracts. | Reuse unpacking and test data, but GM needs its own transfer and completion tests. Include tails, strides, offsets and flag sequences. |
| F2: single I32 NZ2ND, UB/GM; M / M | `fixpipe_nz2nd_int32_t_to_int32_t_4d_to_2d_ubuf`, `fixpipe_nz2nd_int32_t_to_int32_t_4d_to_2d_gm` | Preserve integer bit patterns and relevant native conversion controls instead of borrowing float typing. | Share with F1 only after instruction-mode parity is established; check negatives, extreme values, tails and both destinations. |
| F3: dual F32 NZ2ND to UB; H / M | `fixpipe_nz2nd_dual_float_to_float_4d_to_2d_ubuf` | Native dual/subblock split, destination bases, row/channel behavior and synchronization. | Reuse F1 arithmetic/layout, but prove uneven partitions, participating vector cores and flags. |
| F4: dual I32 NZ2ND to UB; M / M | `fixpipe_nz2nd_dual_int32_t_to_int32_t_4d_to_2d_ubuf` | Combine F2's integer contract with F3's dual destination protocol. | Should be cheaper once both exist; still test integer edge values and asymmetric split tails. |
| F5: normal F32 L0C-to-L1; H / M | `fixpipe_normal_float_to_float_4d_to_4d_cbuf` | No NZ2ND unpack: preserve the 4D destination layout and Cube-to-Cube dependency chain. | Validate by a subsequent consumer that reads the L1 result, including offsets and producer/consumer completion. |

Quantization, ReLU, scale, channel splitting and runtime unit-flag conditions
can add work **within** these symbols. This list does not add unobserved
conversion symbols such as F32-to-INT8 to the 191-symbol scope.

## Nd2Nz

All four share the native GM ND descriptor to L1 NZ layout conversion. Group by
packing width, not floating versus integer naming alone.

| ID: subgroup; effort | Exact symbols | First-instance hurdle | Follow-on work and decisive tests |
| --- | --- | --- | --- |
| N1: 16-bit ND2NZ; M / L | `nd2nz_half`, `nd2nz_bfloat16_t` | Source strides, NZ block packing, zero padding, offsets and transfer-field limits. | BF16 should mostly reuse the 16-bit movement body. Test bit-preserving patterns and a consuming matmul, not only contiguous squares. |
| N2: 32-bit ND2NZ; M / L | `nd2nz_float` | Recalculate element/byte stride units and physical packing instead of inheriting 16-bit constants. | Reuse N1 descriptor and tiling harness; test skinny/odd shapes and offsets. |
| N3: byte ND2NZ; M / L | `nd2nz_int8_t` | Byte-granularity alignment and packed inner dimensions. | Reuse common movement planning; validate with integer consumers, negative values and boundary tails. |

## Cube Setup And Copy

Fill helpers need nonzero-pattern tests: an all-zero test can hide datatype or
partial-width mistakes. These are separate from a numerical matmul algorithm.

| ID: subgroup; effort | Exact symbols | First-instance hurdle | Follow-on work and decisive tests |
| --- | --- | --- | --- |
| S1: 16-bit L1 fill; M / L | `set_l1_2d_half`, `set_l1_2d_bfloat16_t` | Typed fill value encoding, padded rows and layout. | Reuse width-specific implementation; check nonzero fills and partial valid rows. |
| S2: 32-bit L1 fill; L-M / L | `set_l1_2d_float` | Width-specific fill/stride encoding. | Reuse S1 harness, with exact F32 patterns. |
| S3: byte L1 fill; M / L | `set_l1_2d_int8_t` | Byte replication, signed values and short tails. | Reuse descriptor handling; verify surrounding memory is unchanged. |
| C1: 1D GM-to-L1 copy; M / L | `load_gm_to_cbuf_1d_float` | Correct MTE path, address spaces, offsets, byte counts and completion. | Reuse movement utilities, but not ND2NZ layout logic. Test nonzero offsets and nontrivial lengths. |

## Vector DMA

Shared foundation: effective pointers plus descriptor sizes/strides; byte versus
element units; padding/cache/atomic controls where applicable; MTE limits;
alignment fast paths and native scalar/strided fallback paths. A contiguous-only
implementation is not full parity for the same native symbol.

Most non-B64 datatype variants should be cheap after the rank/direction engine
works. INT64 **loads** are split out because native Copy1D/2D/3D include special
B64 padding handling; a raw byte copy does not establish that contract.

| ID: subgroup; effort | Exact symbols | First-instance hurdle | Follow-on work and decisive tests |
| --- | --- | --- | --- |
| D1: 1D GM-to-UB, up to 32-bit elements; M / L | `load_gm_to_ubuf_1d_bfloat16_t`, `load_gm_to_ubuf_1d_float`, `load_gm_to_ubuf_1d_half`, `load_gm_to_ubuf_1d_int16_t`, `load_gm_to_ubuf_1d_int32_t`, `load_gm_to_ubuf_1d_int8_t` | Native contiguous, strided, unaligned and padding branches. | Same-width float/integer variants are mostly typed movement, not new arithmetic. Test each width and nonzero padding, not only zero fills. |
| D2: 1D B64 GM-to-UB; M / L | `load_gm_to_ubuf_1d_int64_t` | Native nonzero 64-bit padding repair and its vector/MTE dependencies. | Reuse D1; test distinct high/low 32-bit padding halves, tails and strides. |
| D3: 2D GM-to-UB, up to 32-bit elements; H / L | `load_gm_to_ubuf_2d_bfloat16_t`, `load_gm_to_ubuf_2d_float`, `load_gm_to_ubuf_2d_half`, `load_gm_to_ubuf_2d_int16_t`, `load_gm_to_ubuf_2d_int32_t`, `load_gm_to_ubuf_2d_int8_t` | Independent row/column strides and native promotion/flattening or scalar fallback paths. | Reuse rank/direction engine; prove non-contiguous inner dimensions, offsets, odd rows and padding. |
| D4: 2D B64 GM-to-UB; M / L | `load_gm_to_ubuf_2d_int64_t` | D3 plus B64 padding semantics and width-sensitive fallback. | Reuse D2's padding tests across a nontrivial row pitch. |
| D5: 3D I32 GM-to-UB; H / L | `load_gm_to_ubuf_3d_int32_t` | Three-axis descriptor scheduling, contiguous-axis selection and fallback loops. | Reuse lower-rank cores; test each stride/axis branch instead of only contiguous flattening. |
| D6: 3D B64 GM-to-UB; M / L | `load_gm_to_ubuf_3d_int64_t` | D5 plus full-width padding/transfer behavior. | Reuse D4/D5; include unequal dimensions, padded planes and distinct high bits. |
| D7: 1D UB-to-GM store; M / L | `store_ubuf_to_gm_1d_bfloat16_t`, `store_ubuf_to_gm_1d_float`, `store_ubuf_to_gm_1d_half`, `store_ubuf_to_gm_1d_int16_t`, `store_ubuf_to_gm_1d_int32_t`, `store_ubuf_to_gm_1d_int64_t`, `store_ubuf_to_gm_1d_int8_t` | Store-tail safety, offsets/strides and applicable native atomic/cache state cleanup. | Share movement engine across widths. Test sentinels around writes; atomic modes, if exercised, need separate proof. |
| D8: 2D UB-to-GM store; H / L | `store_ubuf_to_gm_2d_bfloat16_t`, `store_ubuf_to_gm_2d_float`, `store_ubuf_to_gm_2d_half`, `store_ubuf_to_gm_2d_int16_t`, `store_ubuf_to_gm_2d_int32_t`, `store_ubuf_to_gm_2d_int64_t`, `store_ubuf_to_gm_2d_int8_t` | Independent destination/source strides, partial rows, native reshaping and scalar fallback. | Reuse D3/D7; prove strided write placement and untouched gaps. |
| D9: 3D UB-to-GM store; H / L | `store_ubuf_to_gm_3d_float`, `store_ubuf_to_gm_3d_int32_t`, `store_ubuf_to_gm_3d_int64_t`, `store_ubuf_to_gm_3d_int8_t` | Plane scheduling and native axis/fallback decisions. | Reuse D5/D8 and datatype dispatch; verify unequal plane pitches and tails. |
| D10: 1D UB-to-UB copy; M / L | `copy_ubuf_to_ubuf_1d_bfloat16_t`, `copy_ubuf_to_ubuf_1d_float`, `copy_ubuf_to_ubuf_1d_half`, `copy_ubuf_to_ubuf_1d_int16_t`, `copy_ubuf_to_ubuf_1d_int32_t`, `copy_ubuf_to_ubuf_1d_int64_t`, `copy_ubuf_to_ubuf_1d_int8_t` | UB-local transfer path, mask/tail safety and alignment; do not assume GM DMA instructions apply. | Same byte-width machinery should transfer; include B64 high bits and native-defined alias behavior. |
| D11: 2D UB-to-UB copy; M / L | `copy_ubuf_to_ubuf_2d_bfloat16_t`, `copy_ubuf_to_ubuf_2d_bool`, `copy_ubuf_to_ubuf_2d_float`, `copy_ubuf_to_ubuf_2d_half`, `copy_ubuf_to_ubuf_2d_int32_t`, `copy_ubuf_to_ubuf_2d_int8_t` | Row/inner-stride cases, scalar fallback and width/mask mapping. Native bool delegates through an I8 view rather than requiring a new numerical algorithm. | Reuse D10 rank handling; test bool bit preservation, gaps and asymmetric strides. |
| D12: 1D UB-to-L1 copy; M / L | `copy_ubuf_to_cbuf_1d_bfloat16_t`, `copy_ubuf_to_cbuf_1d_float`, `copy_ubuf_to_cbuf_1d_half`, `copy_ubuf_to_cbuf_1d_int8_t` | UB-to-L1 instruction/layout and vector-to-Cube completion, distinct from UB-local copy. | Mostly width-specific wrappers once proved; validate through an actual Cube consumer. |
| D13: 2D UB-to-L1 copy; M / L | `copy_ubuf_to_cbuf_2d_float` | Multidimensional source/destination strides plus D12's cross-engine dependency. | Reuse D12 and 2D descriptor utilities; test non-contiguous rows and a downstream matmul. |

## Vector Integer64

First resolve one shared 64-bit VMI/VPTO representation, lane/mask layout and
load/store ABI against the pinned backend. Native uses `vector_2xvl_s64` and
specialized helpers; it is not evidence that every required PTO operation
already supports the same representation. If a primitive is missing, record
the required backend work or an explicitly allowed native fallback. Do not
assume all 46 are scalar decompositions or all 46 are direct instruction maps.

| ID: subgroup; effort | Exact symbols | First-instance hurdle | Follow-on work and decisive tests |
| --- | --- | --- | --- |
| I1: aligned normal B64 memory; H / L | `vload_NORM_int64_t_rank1`, `masked_store_NORM_B64_int64_t_rank1` | Correct 64-bit lane/mask and pointer layout; this unlocks most later I-groups. | Load/store are separate direction checks. Use high-bit patterns, partial predicates and untouched-lane sentinels. |
| I2: unaligned normal B64 memory; H / M | `vload_NORM_int64_t_unalign_rank1`, `vload_NORM_int64_t_unalign_rank2`, `masked_store_NORM_B64_int64_t_unalign_rank1` | Unaligned address handling and rank-dependent descriptor/index behavior. | Reuse I1 representation, not its alignment assumptions. Test all relevant misalignment classes and tails. |
| I3: B64 broadcast loads; M / L | `vload_BRC_B64_int64_t_rank0`, `vload_BRC_B64_int64_t_rank1`, `vload_BRC_B64_int64_t_rank3` | Scalar/broadcast addressing and replication of both halves of a 64-bit value. | Rank wrappers should be cheap after the descriptor contract is proved; test singleton and offset sources. |
| I4: one-point B64 stores; M / L | `masked_store_ONEPT_B64_int64_t_rank0`, `masked_store_ONEPT_B64_int64_t_rank1` | Predicate-controlled single-element store, not a normal full vector store. | Share I1 mask logic; prove exactly one output changes for each active/inactive case. |
| I5: integer width casts; M / L | `cast_int32_t_to_int64_t`, `cast_uint32_t_to_int64_t`, `cast_int64_t_to_int32_t` | Signed versus zero extension and native narrowing behavior. | Mostly typed wrappers after lane placement is known; test sign boundaries and high-bit truncation. |
| I6: float/I64 casts; H / M | `cast_float_to_int64_t`, `cast_int64_t_to_float` | Rounding, saturation/range and precision loss; native accepts rounding controls not encoded by symbol. | Shared conversion harness, but each direction needs numerical edge cases and the native special-value contract. |
| I7: lane arithmetic; M-H / L-M | `vabs_int64_t`, `vadd_int64_t`, `vadds_int64_t`, `vsub_int64_t`, `vmul_int64_t`, `vmuls_int64_t` | Establish native-width arithmetic and overflow semantics with the pinned backend; multiplication may require additional work even after addition. | Reuse vector/vector-scalar loop and predicate handling. Test carry/high bits, negatives, minimum integer and low-product behavior as natively defined. |
| I8: bitwise B64; M / L | `vand_int64_t`, `vor_int64_t`, `vxor_int64_t` | Preserve both halves and masked-off values. | These should be among the cheapest siblings after I1; test high/low halves independently. |
| I9: scalar-count shifts; M / L | `vshls_int64_t`, `vshrs_int64_t` | Cross-half behavior and native signed-right-shift/count semantics. | Share layout and scalar dispatch; test counts around 0/31/32/63 and sign bits, plus native-defined out-of-range cases only. |
| I10: compare/min/max/select; H / L-M | `vcmp_eq_int64_t`, `vcmp_ge_int64_t`, `vcmp_gt_int64_t`, `vcmp_le_int64_t`, `vcmp_lt_int64_t`, `vcmp_ne_int64_t`, `vcmp_ugt_uint64_t`, `vmax_int64_t`, `vmin_int64_t`, `vsel_int64_t` | Predicate representation and signed/unsigned ordering; full-width select must agree with comparison masks. | Reuse comparison core for predicates and min/max, but unsigned high-bit ordering is an explicit subcase, not a name-only substitution. |
| I11: duplicate/interleave; M / L-M | `vbr_int64_t`, `vdups_int64_t`, `vintlv_int64_t`, `vdintlv_int64_t` | Lane permutation across the native multi-register value; positions/modes may be arguments. | Reuse lane machinery; test alternating high/low patterns and inverse interleave/deinterleave where contracts permit. |
| I12: indexed vector gather; H / M | `vgather_int64_t_rank2` | Index units, lane ordering, masks and descriptor shape; distinct from the SIMT gather below. | Reuse I1 memory and I11 lane tests, then irregular/permuted/repeated indices within native legality. |
| I13: horizontal reductions; H / M | `vcadd_int64_t`, `vcmax_int64_t`, `vcmin_int64_t` | Cross-lane/multi-register reduction, neutral elements, predicate tails and result ABI. | Share reduction harness; addition and signed min/max still require their own arithmetic checks. |
| I14: signed I64 division; R-H / M | `vdiv_int64_t` | Exact signed quotient behavior and backend instruction/algorithm availability. It is not low-cost merely because I7 exists. | Test large magnitudes and sign combinations; match native-defined exceptional behavior rather than inventing it. |
| I15: I64 index generation; M / L | `vci_int64_t` | Full-width starting value, lane increment and control semantics. | Reuse lane foundation; test starting values crossing the low/high-half boundary. |

## Vector Collectives And Math

Scan/product/sort need algorithm-specific reference tests. A shape change can
cross an internal algorithm threshold without changing the instance name.

| ID: subgroup; effort | Exact symbols | First-instance hurdle | Follow-on work and decisive tests |
| --- | --- | --- | --- |
| V1: 1D F32 cumulative sum; H / L | `cumsum_1d_float_dim0` | Prefix-scan scheduling, native scan direction/options, masks and numerical ordering. | Shared scan harness, with long vectors and thresholds; native floating order may preclude unjustified zero tolerance. |
| V2: 2D F32 cumulative sum; H / M | `cumsum_2d_float_dim0` | Cross-row scan and physical row pitch; not necessarily a rank-only wrapper around V1. | Reuse scan primitives and existing 2D PTODSL work; test unequal dimensions, reverse cases if supported and row tails. |
| V3: I32 cumulative sums; M / L-M | `cumsum_1d_int32_t_dim0`, `cumsum_2d_int32_t_dim0` | Integer overflow/accumulation semantics combined with V1/V2 schedules. | Reuse rank harnesses after I32 arithmetic is proved; test negatives, long accumulation and defined overflow behavior. |
| V4: I8 2D cumulative sum; M-H / L | `cumsum_2d_int8_t_dim0` | Native widening/narrowing, byte packing and intermediate precision. | Reuse V2/V3 scheduling, but validate truncation/overflow at the same stages as native. |
| V5: 2D F32 cumulative product; H / M | `cumprod_2d_float_dim0` | Multiplicative prefix algorithm and rounding/underflow, not merely replacing a scan opcode. | Reuse scan layout utilities; test zeros, signs, long axes and numerical tolerance against native. |
| V6: 1D reversal; M / L | `flip_1d_int32_t`, `flip_1d_int8_t` | Correct lane/block reversal and partial tails. | Width variants should share indexing; test lengths around vector boundaries and odd sizes. |
| V7: 2D F32 sort; H / M | `sort_2d_float` | Sort schedule, scratch, ordering/key-index contract and native special-value/tie behavior. | Requires its own harness and algorithm-threshold cases; do not count as a cheap elementwise op. |
| A1: signed I32 division/remainder; H / M | `vdiv_int32_t`, `vmod_int32_t` | Exact signed arithmetic and division/remainder relationship; establish backend capability and native exceptional cases. | Share quotient/remainder machinery where correct; test both signs, large quotients and remainder sign. |
| A2: F32 division accuracy path; H / M | `vdivfhp_float` | Reproduce native accuracy/refinement and special-value handling, not assume one bare floating divide suffices. | Separate accuracy harness with near-zero/large values and justified error criteria. |

## SIMT Indexing

| ID: subgroup; effort | Exact symbols | First-instance hurdle | Follow-on work and decisive tests |
| --- | --- | --- | --- |
| T1: BF16 data/I32-index 2D SIMT gather; R-H / M | `gather_simt_2d_bfloat16_t_int32_t` | Establish exact selected implementation, SIMT execution/launch ABI, indirect memory semantics and PTOAS capability or approved native dependency path. SIMD VMI gather alone is not proof of equivalence. | Once the execution foundation works, dtype/rank dispatch can transfer; test gather axis, negative-index normalization, out-of-range behavior and independent strides per native contract. |

This is one observed symbol, not evidence that SIMT is only one instance for
the complete target suites. Resolve the execution-path feasibility early so it
does not become an end-of-project discovery.

## Printing, Debug And Synchronization

These are real dependencies, but not necessarily compute-template rewrites.
Decide whether the launch permits their documented native runtime/fallback
implementation. Count that integration/testing cost; do not silently drop the
calls or claim a fully PTO-native product while relying on them.

| ID: subgroup; effort | Exact symbols | First-instance hurdle | Follow-on work and decisive tests |
| --- | --- | --- | --- |
| P1: 1D UB printing, up to 32-bit types; M / L | `print_1d_bfloat16_t_ubuf`, `print_1d_float_ubuf`, `print_1d_half_ubuf`, `print_1d_int16_t_ubuf`, `print_1d_int32_t_ubuf`, `print_1d_int8_t_ubuf` | Runtime output ABI, value formatting and descriptor traversal. | Reuse sink/harness; validate content and integer sign/float formatting, not a tensor allclose. |
| P2: higher-rank UB printing, up to 32-bit types; M / L | `print_2d_bfloat16_t_ubuf.vector`, `print_2d_float_ubuf.vector`, `print_2d_int16_t_ubuf`, `print_2d_int32_t_ubuf`, `print_2d_int32_t_ubuf.vector`, `print_2d_int8_t_ubuf`, `print_2d_int8_t_ubuf.vector`, `print_3d_float_ubuf`, `print_3d_int32_t_ubuf`, `print_3d_int8_t_ubuf`, `print_5d_int32_t_ubuf` | Rank/stride traversal and exact qualified ABI identity. | Share formatting from P1; rank wrappers should be cheap but preserve `.vector` and unsuffixed symbols distinctly. |
| P3: I64 UB printing; M / L | `print_1d_int64_t_ubuf`, `print_2d_int64_t_ubuf`, `print_3d_int64_t_ubuf`, `print_5d_int64_t_ubuf` | Full-width value formatting and descriptor traversal without truncation. | Reuse P1/P2 output infrastructure; test values outside I32 range and rank-dependent ordering. |
| P4: Cube-side GM printing; H / L | `print_2d_float_gm.cube`, `print_2d_int32_t_gm.cube` | Cube/core-qualified runtime and GM address space, not UB vector formatting alone. | Reuse sink if ABI permits; prove output and runtime support on the intended core path. |
| P5: scalar GM printing; M / L | `print_scalar_bool_gm`, `print_scalar_int32_t_gm` | Scalar argument/data access and output encoding. | Mostly typed wrappers once runtime is established; test false/true and signed integer values. |
| B1: unsuffixed debug lifecycle; M / L | `init_debug`, `finish_debug` | Debug buffer/workspace lifecycle, link dependencies and ordering. | Treat init/finish as a pair; verify repeated kernels and output completion. |
| B2: Cube-qualified debug lifecycle; M / L | `init_debug.cube`, `finish_debug.cube` | Preserve exact core-specific identity and resource ownership. | Reuse B1 only after ABI compatibility is checked; pair with P4. |
| B3: Vector-qualified debug lifecycle; M / L | `init_debug.vector`, `finish_debug.vector` | Vector participant/runtime ownership and exact qualified identity. | Reuse B1 sink/protocol; test mixed-core isolation, not just final symbol resolution. |
| L1: cross-block lock protocol; R-H / L | `sync_block_lock`, `sync_block_unlock` | GM lock initialization, participants, memory ordering, progress and launcher metadata. Existing bridge launcher guards can block these kernels before template replacement matters. | One protocol owns both operations; multi-block/repeated-launch tests are mandatory. Never substitute an ordinary local pipeline barrier. |

## Cross-Cutting Work Not Visible In Symbol Counts

| Work package | Why it changes the estimate | Budgeting rule |
| --- | --- | --- |
| Unit flags and shared scratch | A symbol can serve off/check/update modes, internal K slices and repeated/mixed calls. These are not new instance names. | Implement shared foundations once, then budget per-instance ABI migration, regeneration and contract tests. Producer and Fixpipe consumer semantics differ. |
| Memref descriptors/adapters | Sizes, physical pitches, offsets, casts/subviews and runtime expressions can differ for the same symbol. | Keep the approved per-template adapter approach. Account for a new family adapter once and any distinct rank/layout paths; do not assume symbol selection performs adaptation. |
| Alignment, non-contiguity and padding | One DMA symbol includes a fast path and more complex native paths. Nonzero B64 padding is a concrete example. | First-instance completion includes those exercised contracts; otherwise label the restriction rather than charging every later symbol for rediscovery. |
| Numerical controls | Precision, rounding, saturation, signedness and accumulation need tests independent of shape. | Share a reference harness, but not an unverified tolerance or float-to-integer behavior. |
| Resources and exact-symbol resolution | Each offline instance MLIR needs the correct export, installation/registry entry, adapter and active integration pattern. | Reuse the native-selected identity with `__pto_` prefix. Missing implementation may fall back only under policy; present incompatible implementations must fail. |
| Backend capability | SIMT, I64, special math or synchronization may need PTOAS/runtime work, not another PTODSL wrapper. | Time-box an early capability probe per foundation; treat unresolved dependencies as launch risks rather than cheap remaining symbols. |
| Runtime evidence | A native symbol in a compile-only census does not prove final linking, legal runtime behavior or replacement parity. | Separate native baseline, bridge target selection/body, numerical result and physical-A5 validation. Keep native failures outside PTODSL implementation claims. |
| Support dependency policy | Printing, debug and locks can prevent a suite from running even if compute is complete. | Explicitly choose a native dependency strategy or replacement scope; do not remove behavior to manufacture a pass. |

## Estimation And Delivery Process

Estimate **remaining work**, not 175 times the cost of the first MmadL1:

```text
engineering effort =
    remaining shared foundations
  + first missing contract per subgroup
  + typed/rank/transpose sibling implementations
  + per-instance native and replacement validation
  + integration, regression, performance and physical-A5 release gates
  + newly discovered suite/control requirements
```

For each subgroup, choose a representative that exercises its distinctive
boundary, not merely the easiest all-zero square. Review native CCE code, create
tests and prove the native symbol plus numerical baseline first. Use legal
small/odd/boundary/larger tensors, varied hardware tile configurations and grids;
add relevant control/stride/precision sequences rather than an arbitrary Cartesian
product. Coordinate a native-fallback baseline with the tester before new
implementation, as required by the current
[development policy](../designs/ptodsl-template-development-and-fallback.md).

Then implement the PTODSL body and existing-style adapter, render/import the
instance offline, add focused compiler checks, and run the **whole agreed
native-supported matrix for that instance** through the bridge. Require the
exact PTODSL target and materialized body; dependency fallback is not target
coverage. The approved 3.6 [source/setup record](../memory/triton-ascend-3.6-shared-a5-2026-10-22.md)
is the environment baseline. Do not repair native compiler/templates or change
shared environments just to make this planning exercise look complete.

Measure seed-versus-sibling implementation/review effort and native/bridge
matrix wall time separately. Reuse those observations to replace L/M/H with
real ranges. The earlier census's 4h30m/8h14m wall times are compile-only campaign
measurements, **not per-instance simulator validation estimates**. This document
does not have enough comparable measurements to justify a product-launch date.

Calendar time follows the critical path and available people/tools. Template
and test authors can prepare independent groups concurrently; shared generators,
adapters and build/install prefixes still need coordinated ownership. Simulator
worker limits, test queueing, review and access to physical A5 hardware can
dominate the schedule even when a sibling implementation is short.

### Suggested Early Gates

1. Reconcile the latest MmadL1/Fixpipe implementation and runtime results before
   crediting completed work; do not overwrite another agent's in-progress status.
2. Seed high-frequency Cube/ND2NZ and 1D/2D GM-UB families, but probe I64, SIMT
   and cross-block-lock feasibility early in parallel with that work.
3. Validate cheap siblings after a seed is genuinely complete; use the paired
   test/implementation process rather than postponing all runtime checks.
4. Expand the census to the actual Q3 and Triton-Ascend launch targets. Preserve
   failures and unknowns; do not extrapolate this filtered scope into full-suite closure.
5. Agree whether native dependency fallback is acceptable for launch and what
   numerical, performance, repeated-launch and physical-A5 gates define release.

## Filter Provenance And Scope Growth

The filter reused existing artifacts under
`$HOME/template-census/triton36/q2-full/run-20261022/`:

- `summary.json`: completed campaign scope and failure counts.
- `observed-template-counts.csv`: exact 42-symbol successful-checkpoint set.
- `bitcode-symbol-audit/audit.json`: snapshot bundle hashes and reader methods.
- `bitcode-symbol-audit/meta_op.{aic,aiv}.c310.bc.definitions.json`: exact externally visible A5 definitions.
- `bitcode-symbol-audit/meta_op.mix.{aic,aiv}.c310.bc.nm.txt`: qualified mixed-core definitions.
- `bitcode-symbol-audit/*c220.bc.nm.txt`: legacy architecture comparison.

Matching preserves `.cube` and `.vector`; strip only the
`_mlir_ciface_` prefix. Search all four target bundles before declaring a
definition missing. The campaign used the approved 3.6 candidate and frozen
compiler/resources; its source/build provenance is recorded separately in
`provenance.json`. The existing audit explicitly notes historical local C310
template fixes: this is not an untouched-upstream support claim.

That 3.6 Q2 campaign also observed eight symbols outside the old 191 list:
`copy_ubuf_to_cbuf_2d_bfloat16_t`,
`cumsum_2d_float_dim0_comp`, `cumsum_2d_float_dim1`,
`cumsum_3d_float_dim1`,
`fixpipe_normal_float_to_bfloat16_t_4d_to_4d_cbuf`,
`fixpipe_nz2dn_dual_float_to_float_4d_to_2d_ubuf`,
`fixpipe_nz2nd_float_to_bfloat16_t_4d_to_2d_gm`, and
`indirect_load_nonvolatile_1d_float_int64_t`.
They are **not silently added to this subtractive edition**. A reconciled union
would contain at least 183 symbols before broader suite discovery, not 175.
Classify those separately before treating this as a complete launch budget.

## Excluded Historical Identities

Excluded means removed from this active estimate, not globally unsupported.
All 16 are absent from the audited A5 definitions and retained 3.6 Q2 captures.

| Exact excluded symbol | Historical group | Definition comparison |
| --- | --- | --- |
| `mma_tile_half_to_half` | M5 | Missing in audited C310 and C220 bundles. |
| `fixpipe_nz2nd_half_to_half_4d_to_2d_gm` | F6 | Missing in audited C310 and C220 bundles. |
| `broadcast_scalar_bfloat16_t_to_1d` | O1 | C220 definition exists; no C310 definition. |
| `broadcast_scalar_float_to_1d` | O1 | C220 definition exists; no C310 definition. |
| `broadcast_scalar_float_to_2d` | O2 | C220 definition exists; no C310 definition. |
| `broadcast_scalar_half_to_1d` | O1 | C220 definition exists; no C310 definition. |
| `broadcast_scalar_int32_t_to_1d` | O1 | C220 definition exists; no C310 definition. |
| `broadcast_scalar_int8_t_to_1d` | O1 | C220 definition exists; no C310 definition. |
| `transpose_3d_without_last_axis_bfloat16_t` | O3 | C220 definition exists; no C310 definition. |
| `transpose_3d_without_last_axis_float` | O3 | C220 definition exists; no C310 definition. |
| `transpose_3d_without_last_axis_half` | O3 | C220 definition exists; no C310 definition. |
| `transpose_3d_without_last_axis_int8_t` | O3 | C220 definition exists; no C310 definition. |
| `vadd_1d_float` | O4 | C220 definition exists; no C310 definition. |
| `vadd_1d_int32_t` | O5 | C220 definition exists; no C310 definition. |
| `vadd_2d_float` | O4 | C220 definition exists; no C310 definition. |
| `vadd_3d_float` | O4 | C220 definition exists; no C310 definition. |

The newer native runtime record also reports that the former
`vadd_1d_int32_t` unresolved-symbol failure did not reproduce in the restored
INT8 cases under 3.6. That is evidence for those configurations, not a new A5
definition of this old helper.

## Original Subgroup Source Basis And Limits

Source inspection used `$HOME/AscendNPU-IR` at
`dced14b0604ebcc5a87357af9f3c785f715131b7` plus in-progress coworker changes,
and its pinned PTOAS `d36100d03255e5e8a858c497a4277708d8b889c9`. This is an
effort analysis against that checkout, not a review of all dirty changes or a
fresh upstream/fork survey. The current Explorer summary has authentication
and certificate failures, so it cannot establish absence of recent backend changes.

Relevant native paths, relative to `$HOME/AscendNPU-IR`:

- `bishengir/lib/Template/lib/RegBase/Cube/LocalMmad.cpp`: MmadL1 dispatch, packing, shared state and flags.
- `bishengir/lib/Template/lib/RegBase/Cube/Fixpipe.cpp` and `bishengir/lib/Template/include/RegBase/`: Fixpipe registration and shared definitions.
- `bishengir/lib/Template/lib/RegBase/Cube/nd2nz.cpp`, `set2d.cpp`, `Copy1D.cpp` and `compat/DMA/Cbuf/`: Cube movement/fill and reused native implementations.
- `bishengir/lib/Template/lib/RegBase/Vector/Copy1D.cpp`, `Copy2D.cpp`, `Copy3D.cpp`: movement, descriptor branches and B64 padding.
- `bishengir/lib/Template/lib/RegBase/Vector/VecOpI64.cpp` and `bishengir/lib/Template/include/RegBase/`: B64 intrinsic wrappers, modes, casts and memory contracts.
- `bishengir/lib/Template/lib/RegBase/Vector/Cumsum.cpp`, `Cumprod.cpp`, `Flip1D.cpp`, `Sort2D.cpp`, `Vdiv.cpp`: distinct algorithm families.
- `bishengir/lib/Template/lib/RegBase/Vector/SIMTGatherUBToUB.cpp` and `SIMTGatherOutToUB.cpp`: SIMT capability/contract investigation starting points; confirm the actual census symbol's definition.
- `bishengir/lib/Template/lib/Debug/Debug.cpp` and `bishengir/lib/Template/lib/Synchronization/SyncBlockLock/SyncBlockLock.cpp`: support runtime and lock protocol.

The boundaries and effort ratings are engineering judgment, not measured
throughput. No new compilation, simulation, implementation, support-status
update, commit or push was performed for this classification.
