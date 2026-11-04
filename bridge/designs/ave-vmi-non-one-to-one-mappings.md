# AVE contracts without a one-to-one VMI mapping

Last updated: 2026-10-30

Maintain this ledger whenever an investigation or implementation encounters a
non-one-to-one mapping. Record supported compositions as well as unresolved
contracts. Link existing detailed notes rather than duplicating their analysis.
This ledger does not classify every composition as a missing VMI capability.
Historical entries require revision-specific revalidation before changing code.

Each entry records source contract/cases, target sequence, why direct mapping
is insufficient, semantic guards, implementation status, revision/tool evidence,
validation stages and remaining limits. Add actual target gaps separately to
the [compatibility tracker](ave-ptoas-vmi-compatibility-tracker.md).

## BF16 packed-store carrier folding — 2026-10-28

**Source and affected cases:** Q2 `chunk_bwd_dqkwg` helper 25 and
`chunk_scaled_dot_kkt_fwd` consumer helper 5 contain:

```text
vtruncf: F32[N] -> BF16[N], part_even
  -> unrealized_conversion_cast: BF16[N] -> BF16[2N]
  -> llvm.bitcast: BF16[2N] -> F32[N]
  -> masked_store <PK_B32> into BF16 UB memory
```

**Mapping:** This graph uses a many-to-one carrier fold. Preserve the existing
`pto.vmi.vcvt` producing logical BF16[N], discard the representation wrappers,
and store that logical value with `pto.vmi.store` for ALL or `pto.vmi.vstore`
for an accepted partial predicate. AVE physical register carriers cannot be
translated independently as ordinary logical widening/bitcasting: doing so
would change which values the store observes. No additional target packing
instruction is established necessary.

**Semantic guards:** Match the exact N/2N/N lane relationship and matching
F16/BF16 source and expanded element types. Require a UB destination with the
same element type as the recovered logical source. Preserve conversion
rounding/saturation, logical lane order, destination offsets/strides, accepted
store masks and the semantic write footprint. The tested BF16 cases use
64 values and R/NOSAT conversion; malformed widths and mixed types reject.

**Status and evidence:** Implemented as an unstaged/uncommitted extension of
the F16 path on AscendNPU-IR `master`, base `c9d9b0e51`. See the
[converter](../../../AscendNPU-IR/bishengir/lib/Conversion/HIVMAVEToPTOASVMI/HIVMAVEToPTOASVMI.cpp),
[positive regressions](../../../AscendNPU-IR/bishengir/test/Conversion/HIVMAVEToPTOASVMI/preflight-packed-store.mlir),
[rejection regressions](../../../AscendNPU-IR/bishengir/test/Conversion/HIVMAVEToPTOASVMI/preflight-reject-packed-store.mlir),
and [validation report and provenance](../../../Logs/run_reports/q2-bf16-packed-store-2026-10-28/report.md).

Both captured helpers fail before the change and pass bridge conversion after
it, in pointer and preserved-memref modes. Offset/mask regressions pass; the
F16 control has identical VMI output. Reduced full BF16 stores lower to VPTO
with the retained October 21 PTOAS test driver at `37f5f25e3b5b433e7cc760d148fea83dddef9934`.
Validation uses a private partial NPU-IR pass rebuild with retained headers and
libraries, not a complete current production compiler build.

**Remaining limit:** A reduced partial store to a dynamic address without proven
alignment converts to VMI but the tested PTOAS lowering rejects it to protect
the write footprint. This is a downstream alignment requirement, distinct from
the BF16 carrier-folding fix. No arbitrary unaligned partial-store support,
full-kernel linking, predicate-buffer correctness, numerical equivalence,
simulator performance or A5 hardware validation is established by these checks.

## Scalar I32 GM metadata uses PTO scalar memory — 2026-10-28

**Source and affected cases:** `memref.load` of singleton, unit-stride I32 GM
views with dynamic offsets in Q2 variable-length solve/MXR and MoE
weight-gradient captures. These scalar reads surround AVE computation; they are
not vector-register loads. MoE also contains I64 metadata and local stack
buffers, which remain outside this I32 extension.

**Mapping and reason:** Use `pto.load` returning I32 for ordinary reads, or
`pto.ld_dev` for a presence-based `disableDCache` marker on an ordinary AICore
entry function. Each read maps directly to an existing PTO scalar operation,
rather than a one-to-one VMI vector instruction. Do not introduce a singleton
vector load plus scalar extraction. Existing memref/view normalization supplies
PTO pointer arithmetic in element units; preserved helper descriptors retain
aligned pointer and descriptor offset separately from the access index. No
missing target scalar-load instruction is established by this bridge omission.

**Semantic guards:** Rank-one GM memref, static size one, static unit stride,
and supported floating or I32 element type. Keep all 32 integer bits, dynamic
offsets and indices; no floating bitcasts are needed for I32. A zero-valued
`disableDCache` attribute still selects bypass. Reject nontemporal policy and
noncached helper reads, unsupported integer widths including I64, non-singleton
or nonunit-stride views, and UB integer scalar accesses. The noncached I32 case
is a regression control, not an established occurrence in the retained Q2
metadata captures.

**Status and evidence:** Implemented, unstaged/uncommitted, on
`fix/remaining-ave-to-vmi-supports-q2`, base `09f07af13`. See
[positive regressions](../../../AscendNPU-IR/bishengir/test/Conversion/HIVMAVEToPTOASVMI/preflight-gm-i32-scalar-load.mlir),
[guard regressions](../../../AscendNPU-IR/bishengir/test/Conversion/HIVMAVEToPTOASVMI/preflight-reject-gm-scalar-load.mlir),
and the [validation report](../../../Logs/run_reports/q2-i32-gm-scalar-load-2026-10-28/report.md)
for commands, tool provenance and exact failures. The private partial rebuild
passes 94 of 97 AVE conversion tests; all three remaining failures also occur
in the baseline harness. Reduced pointer-mode cached and bypass loads reach
LLVM; cached reads emit `load i32` and bypass uses `llvm.hivm.LD.DEV.u32.GM`.

**Preserved-mode limit:** Bridge conversion passes for the reduced I32
helper descriptors. Preserved entry arguments use the existing native-memref
reconstruction cast. The retained standalone PTOAS first requires metadata
expansion for the view probe, then fails GM memref memory-space conversion to
LLVM. An unchanged F32 descriptor control reproduces that latter failure.
Do not claim preserved-mode LLVM/link closure from the bridge pass alone, or
turn off native-memref preservation as a workaround. No numerical, object/link,
simulator or hardware checks were performed.

### Follow-on contracts exposed by the Q2 metadata replay

The unchanged October captures still do not complete conversion. These are
observations with the retained validation pair, not a claim about every current
upstream implementation:

- **Preserved solve/MXR predicate buffers:** a byte-carrier pointer cannot
  materialize `memref<16x256x1xi1, #pto.address_space<vec>>`. There is no
  established direct vector-VMI memory/descriptor mapping for this PB32 buffer.
  Keep this separate from scalar metadata loads; see the existing
  [predicate-memory contract](predicate-memory-store-compatibility.md).
- **Pointer-mode solve/MXR synchronization:** conversion emits
  `pto.set_intra_block` / `pto.wait_intra_block` with `PIPE_S`, event 15; the
  retained PTO verifier rejects that pipe. This is a non-vector synchronization
  contract whose mapping remains unresolved in the replay. Investigate the
  native producer/consumer protocol before choosing a different operation or
  pipe; do not substitute a legal pipe just to obtain compiler acceptance.
- **MoE I64 memory:** preserved-mode replay still rejects a singleton I64 GM
  load; pointer mode stops at an I64 reinterpret cast. Local I64 stack memory
  is also present. Existing scalar PTO capability is not proof of a complete
  bridge pointer/type/ABI mapping for these forms. They remain separate scope.

Exact operations and first diagnostics are retained in
[replay results](../../../Logs/run_reports/q2-i32-gm-scalar-load-2026-10-28/replay-results.json).

## H-masked B16 widening to same-mask stores — 2026-10-28

**Source and cases:** The forward and reverse Q2 local-cumsum captures contain
`UNPK_B16` loads into 64-lane F16 carriers, `vextf part_even` under a direct
H/PB32 mask, then `NORM_B32` stores using that same SSA mask. Only logical lanes
0–31 reach memory. Native widening takes the predicate; VMI `extf` has no mask.
Unmasked conversion followed by a masked store alone does not prevent extra
conversion of inactive exceptional inputs.

**Composition:** Preserve the original load and its address/distribution. Emit
a separately materialized logical 32-active-lane predicate for B16 selection,
a positive-zero B16 vector, and `vsel(mask, original_source, zero)`. Widen that
sanitized value with VMI `extf`. Keep the original B32 store mask and all store
addresses. Active B16 bits remain unchanged; inactive original NaNs never reach
the conversion. Inactive results remain unobserved. Other users of the original
source retain the original source, including a tested shared-source return.

**Guards:** Exact rank-one 64-lane F16/BF16→F32, `part_even`, direct H with
explicit PB32 function type, and an `UNPK_B16` producer with a matching-element,
rank-one unit-stride UB base. Require nonempty result uses, all of which consume
this result as the payload of same-SSA-mask `NORM_B32` stores into rank-one
unit-stride F32 UB bases. Reject full-vector returns/extra result consumers,
different masks, other widths/parts, arbitrary producers, non-UB destinations,
nonunit strides and unsupported attributes. Existing ALL/VL1 widening and all
narrowing conversions are unchanged.

**Status:** Implemented, unstaged/uncommitted, on
`fix/remaining-ave-to-vmi-supports-q2`, base `fd56e7f0d`. See
[positive regressions](../../../AscendNPU-IR/bishengir/test/Conversion/HIVMAVEToPTOASVMI/preflight-h-masked-extf-store.mlir),
[rejection regressions](../../../AscendNPU-IR/bishengir/test/Conversion/HIVMAVEToPTOASVMI/preflight-reject-h-masked-extf-store.mlir),
and [evidence and tool manifest](../../../Logs/run_reports/q2-h-masked-widening-2026-10-28/report.md).
Both original cumsum modules and their extracted widening helpers now pass
bridge conversion in pointer and preserved-memref modes; all eight corresponding
baseline checks reject the H widening. The private partial rebuild passes 96 of
99 AVE conversion tests, with the same three unrelated baseline failures.

**Downstream evidence:** Both exact pointer-mode widening helpers reach LLVM.
Aligned F16 and derived BF16 reduced compositions pass VPTO and LLVM emission.
FileCheck confirms bit-preserving select feeds the conversion intrinsic and the
conversion result feeds the store. The exact helper uses post-increment source
and destination steps of 64 and 128 bytes respectively, matching the original
32-element F16/F32 rows. The logical select and store predicates can acquire
different physical granularities; do not force them to share a physical mask.
These are compiler checks using retained, revision-qualified tools, not device
numerical/FP-status or performance validation.

**Remaining work and target gap:** General masked widening/narrowing and other
consumer graphs remain unsupported. Native masked VMI conversion would avoid
this bridge composition if it preserves the same lane and inactive-FP contract;
track it in the [compatibility tracker](ave-ptoas-vmi-compatibility-tracker.md).
The select/zero/predicate sequence adds generated work; no timing impact was
measured. Full captured modules still retain native
`load_gm_to_ubuf_1d_half`, `cumsum_2d_float_dim0` and
`store_ubuf_to_gm_1d_float` declarations/calls. Their template/descriptor ABI
route has no demonstrated one-to-one VMI replacement in this replay. Do not
count VMI acceptance as PTO-native scan integration, final-link success or
full-kernel numerical correctness; follow the existing
[template/fallback policy](ptodsl-template-development-and-fallback.md).

## Predicate memory — deferred and rolled back 2026-10-29

The human deferred this work and requested restoration of the pre-investigation
compiler and runner. The BF16, scalar I32 GM and H-masked widening support above
remains. Predicate-specific code, regressions and runner changes were removed;
there is no authorized direct-physical exception or PTOAS boundary extension.
The baseline byte-carrier path still does not establish native packed-memory
semantics. Existing experimental finalizers remain outside the default flow.
Historical findings are retained in [Logs](../../../Logs/run_reports/q2-predicate-inline-boundary-2026-10-28/report.md);
see [rollback evidence](../../../Logs/run_reports/q2-predicate-rollback-2026-10-29/report.md).

## Cached scalar I64 GM metadata — 2026-10-29

**Source:** singleton unit-stride I64 GM metadata reads in the captured MoE
`_padded_copy_wgrad` kernel. **Mapping:** existing scalar `pto.load`, preserving
64-bit values and element-scaled view/access offsets; this is a scalar PTO
operation, not a VMI vector conversion. No missing PTOAS instruction or early
physical vector/mask exception is established.

**Guards:** I64 pointer admission is GM-only. Scalar reads require a static
singleton, unit-stride rank-one view and cached access. UB scalar I64,
non-singleton/nonunit-stride accesses, nontemporal and `disableDCache` remain
rejected. Stack-memory patterns and predicate-memory deferral are unchanged.

**Evidence:** base `8320f451e`, pinned PTOAS `5ce40f710`; private current
converter rebuild, 97/100 conversion regressions pass with the same three
baseline failures. Reduced pointer-mode loads reach LLVM as `load i64` with
I64 GEPs and no bitcasts. Both pointer and preserved replays of the unchanged
MoE module now pass bridge conversion; full-module target/link and numerical
validation remain untested. See the [report and manifest](../../../Logs/run_reports/q2-i64-gm-scalar-load-2026-10-29/report.md).

## Masked numeric prefix stores need alignment proof — next assessment 2026-10-29

At NPU-IR `e275b5649`, preserved norm D128 and convolution backward captures
convert to VMI but retained pinned PTOAS rejects their masked F32 stores for
missing destination-alignment proof. The first examples use VL16/VL32 into
16/32-element UB memrefs. Logical store conversion alone does not establish
physical write-footprint legality; no full-write replacement is valid merely
because the vector carrier has 64 lanes. Recommended next work is tracing
allocation/callsite alignment through memref-derived addresses; keep unknown
forms rejected, all vector lowering in PTOAS, and predicate memory deferred.
No implementation is selected or performed. See the [fresh assessment](../../../Logs/run_reports/q2-next-support-2026-10-29/report.md).

### Alignment proof interface finding — 2026-10-29

The pinned PTOAS remainder analysis does not recognize standard
`memref.assume_alignment` or aligned-pointer extraction from preserved memrefs.
A fresh reduced masked F32 store still rejects with an explicit 32-byte proof.
Inlining probes also leave full norm/convolution modules blocked. Proposed
bridge proof propagation therefore needs a small PTOAS address-analysis
extension to consume it; that new source scope is pending explicit approval.
No compiler edit or physical-lowering workaround has been made. See the
[concrete proposed patch and probes](../../../Logs/run_reports/q2-masked-store-alignment-2026-10-29/report.md).

### Alignment work deferred by the human — 2026-10-29

No PTOAS change is authorized. The human deferred bridge-side alignment work
as well. The problem, probe results, alternatives, unmeasured costs and required
guards are retained in the [alignment note](masked-numeric-store-alignment.md).
The earlier pending PTOAS scope request is closed without approval. No compiler
alignment implementation was added.

## Mamba native dim1 cumsum dependency — assessed 2026-10-29

The unchanged captured Mamba module now reaches LLVM, but still calls the
external `_mlir_ciface_cumsum_2d_float_dim1` implementation. This structured
scan/helper ABI has no demonstrated one-to-one VMI replacement in the replay.
A dim0 PTODSL resource/import exists; that is not dim1 support. Next investigate
actual fallback link inputs and native semantics before choosing a PTODSL
replacement. No final-link failure, missing VMI instruction or new compiler
implementation is claimed. See the [fresh full-capture assessment](../../../Logs/run_reports/q2-next-after-alignment-2026-10-29/report.md).

## Dual pointer/descriptor views — 2026-10-29

**New human authorization:** retain explicit PTO pointers for VMI consumers and
memref descriptors for native calls. This supersedes the earlier blanket
deferral only for this bridge-side representation change; PTOAS edits,
predicate-memory work and finalizers remain outside scope.

**Composition:** a pointer-backed buffer can have both a direct PTO pointer
and a native-compatible memref view. Recover the aligned base separately from
the descriptor offset through casts, metadata bases, selects, reinterpret
views, subviews and supported GM byte views. Reinterpret offsets replace the
source offset; subview offsets compose using source strides; GM byte-view shifts
stay in bytes. Numeric operations/masks remain VMI, with physical lowering in
PTOAS. Native declarations and call descriptor ABIs are unchanged.

**Internal helper boundary:** private vector helpers and compiler-owned
outlined VFs with direct symbol uses may accept pointers for static identity-layout
arguments, reconstructing descriptors only if consumers need them. Every
recorded caller must supply a base proven 32-byte aligned; a misaligned or
unknown caller retains the existing descriptor ABI. Nonempty argument attributes, normalized pointer/descriptor element-type
mismatches, unmarked public/kernel/native interfaces, address-taken symbols and dynamic descriptor
specialization are excluded. Unknown local pointer provenance also retains
descriptor extraction; this is not universal alignment support.

**Validation:** current reduced MmadL1 F32 row consumer passes bridge, VPTO and
LLVM emission; shared-native and outlined-native examples pass VPTO; the
misaligned helper remains rejected. Captured Q2 norm-backward D128 and convolution-backward initial still fail
VPTO on unproven descriptor-backed helper arguments. Public helper signatures
remain unchanged; internalizing generated helpers requires a separate decision.
The conversion suite passes 98/101 with the same three baseline failures.
No full current MmadL1 pytest, object/link, numerical or runtime claim.
See the [evidence](../../../Logs/run_reports/bridge-dual-memory-views-2026-10-29/report.md)
and [implementation plan](../planning/plans/2026-10-29-dual-memory-views.md).


### Compiler-owned outlined VF follow-up — 2026-10-29

The human explicitly requested lowering the defined outlined VF and its
callers rather than retaining the private-only restriction. Before conversion
drops call annotations, recognize compiler ownership from
hivm.vector_function on the definition and every direct call. A temporary
bridge marker carries this information through conversion and is removed
before output. Source public visibility no longer blocks these compiler-owned
VFs; a generated-looking name alone is insufficient. Unmarked or mixed
marked/unmarked public uses, declarations, C interfaces and kernel entry points
remain protected. Every argument still requires proven aligned callers and a
reconstructible static descriptor.

The captured actual f16-nn nn-disabled and small-b16 inputs now pass the current
converter and pinned PTOAS device-binary emission. The retained norm D128
capture also passes VPTO; convolution initial remains blocked later on another
unproven destination. The converter suite remains 98/101 with the unchanged
three baseline failures. No kernel runs or whole-matrix claim. Source captures
use retained frontend/template tools; this is not a full latest frontend build.
See the [follow-up evidence](../../../Logs/run_reports/bridge-outlined-memory-views-2026-10-29/report.md).


### Strided template consumer regression follow-up — 2026-10-29

The current full source-built converter already handles both supported
consumer forms: a native external call retaining its memref descriptor, and
PTO/VMI operations after template implementation lowering/inlining. A shared
stride-32 GM view can serve both forms while preserving its dynamic element
offset, length and UB destination offset. This does not make a bare pointer
carry strides: the inlined template body explicitly computes `i * 32`.

The two strided gather/store regressions had incomplete PTODSL-mode RUN
pipelines. Without `convert-hivm-to-std` and
`resolve-hivm-ptodsl-template-calls`, an unavailable resource left raw
`hivm.hir.load/store` at the AVE boundary. Correct the tests to model native
fallback and preserve its descriptors; retain the separate direct-mode
checks. Do not bypass PTODSL resource selection or relax the nonunit-stride
consumer guard to accommodate an incomplete pipeline.

Added `preflight-strided-gm-consumers.mlir` covers external-only,
inlined-PTO-only and shared-view cases. Its explicit private PTO helper is
an already-lowered boundary fixture, not a newly integrated PTODSL template.
No compiler/PTOAS changes or new finalizer use are involved. Current full-tool
validation passes all 102 AVE conversion tests. A broader 137-test selection
passes 133; four unchanged template/standard tests still fail independently.
This is IR regression validation, not simulator or numerical validation.
See the [commands and results](../../../Logs/run_reports/bridge-consumer-regressions-2026-10-29/report.md).


## Masked narrowing to same-mask packed stores — 2026-10-29

**Source contract:** Q2 delta state, output, setup and WY backward captures
contain 64-lane F32-to-F16/BF16 `vtruncf part_even`, `rint`, `sat=false`
under PB32 H/VL16 masks, followed by the B16 carrier cast/bitcast and PK_B32
UB stores. Logical VMI `vcvt` has no predicate operand.

**Composition:** select positive zero into inactive F32 lanes, convert with
VMI vcvt, then reuse the existing packing fold and original store mask.
A separate select predicate avoids coupling F32 and B16 mask granularity.
Active values retain rounding; inactive exceptional inputs cannot cause extra
conversion effects, and inactive result lanes cannot escape to other users.

**Guards:** exact 64-lane types, PB32 H/VL16, rint/NOSAT/part_even; all users
must follow the packing cast/bitcast into same-mask PK_B32 stores to matching
rank-one unit-stride B16 UB memory. Other graphs remain rejected.

**Implementation:** NPU-IR commit `3d35622a7`; no PTOAS modification or finalizer.
Private rebuilt tools pass 106 conversion tests and four aligned-address
F16/BF16 H/VL16 probes through current PTOAS VPTO. All four Q2 first narrowing
blockers are removed: fwd_o converts; two cases expose pointer layouts and WY
exposes widening. Unknown-alignment probes still hit PTOAS's store-alignment
guard. No object/link, numerical, simulator or hardware result is claimed.
See [commands, replays and hashes](../../../Logs/run_reports/q2-item04-masked-narrowing-20261029/report.txt).


## Masked widening through lane-wise store graphs — 2026-10-29

**Source:** WY and gating/sigmoid kernels widen 64-lane F16/BF16 `part_even`
values under PB32 H/VL1/VL4 masks, then use arithmetic before numeric stores.
Logical VMI extf has no predicate. The producer is a rank-one unit-stride
UNPK_B16 load, or BRC_B16 under VL1; other producers remain rejected.

**Composition:** select +0 into inactive B16 lanes with an independent logical
predicate before unmasked widening. Follow every use through a bounded,
same-block lane-wise graph. Arithmetic/compare governing masks and terminal
numeric-store masks must be the original SSA mask. Lane-wise selects are
permitted; no cross-lane, call/region/return escape is admitted. Known packing
carriers may follow same-mask narrowing, which retains its independent guards.
The original predicate and addresses remain on stores; inactive exceptional
input values cannot introduce extra conversion effects.

**Evidence:** NPU-IR `a9d76f894`, private rebuilt tools, 108 passing conversion
tests and seven aligned-address F16/BF16 memory/branch probes reaching current
standalone PTOAS VPTO. Both fused sigmoid source/test variants now convert;
WY exposes a cast and gating exposes VL4 narrowing. Predicate-memory work,
finalizers and unproven alignment stay deferred. No object/link, numerical,
simulator or hardware result is claimed. See [commands and manifest](../../../Logs/run_reports/q2-item05-masked-widening-20261029/report.txt).


## Scalar SSBUF synchronization counters — 2026-10-29

**Source:** four Q2 attention captures reject rank-zero I32 SSBUF pointers.
The earlier worklist called this a Cube layout gap; exact failed-op notes and
an accepting reduced CC-pointer probe correct that classification.

**Mapping:** PTO pointer spaces have no SSBUF equivalent. Preserve native
scalar address space 11 with LLVM inttoptr/load/store rather than assigning
SSBUF addresses to UB/GM. Rank-zero identity-layout I32 alone is admitted;
accesses have no indices, stores have no extra attributes, and loads admit
only the observed DCache-visited integer marker. Unknown/nontemporal policies
remain rejected. Stores and conservatively all supported loads are volatile,
so polling reads remain observable and cannot be hoisted away. This is scalar
native LLVM transport, not a vector physical-lowering exception.

**Evidence:** NPU-IR `b18c48b80`; private rebuilt tools pass 110 conversion
tests. Pointer and descriptor-preserving polling probes reach current PTOAS
LLVM IR retaining addrspace(11) and volatile accesses. All four source cases
move to the separate PIPE_S intra-block sync verifier rejection. No pipe was
changed to obtain acceptance. No complete-kernel link, numerical, simulator
or hardware success is claimed; shared install remains unchanged. See
[commands, diagnostics and hashes](../../../Logs/run_reports/q2-item06-cube-pointers-20261029/report.txt).


## Constant PLT prefixes with unused updated extent — 2026-10-29

**Source:** two causal-convolution captures use a constant seven-lane PB16
`ave.hir.plt` predicate for NORM_B16 stores; the second result is unused.

**Composition:** emit logical `pto.vmi.create_mask(count)` and a dead
`count - capacity` replacement for the unused updated extent. Admit only
nonnegative constants within one register, with PB8/256, PB16/128 or PB32/64
lanes (or the equivalent inferred width), and known attributes. Store masks
retain their original logical width and addresses. Reject dynamic/oversized
counts, live updated extents, mismatched distribution and unknown attributes.
AVE TableGen documents subtraction by full register capacity, while current
PTOAS PLT documents/lowers a clamped carry. Live-carry equivalence is unresolved;
this bounded mask-only mapping makes no claim about that contract and does not
substitute VMI PLT carry semantics. No PTOAS change or finalizer is involved.

**Evidence:** NPU-IR commit `ed872278c` on baseline `b18c48b80`, private rebuilt
tools, 112 passing conversion tests. Aligned 0-, 7- and 128-lane F16 store
probes reach current standalone PTOAS LLVM emission. Both original captures
advance past PLT and stop at the separate I16 VCI blocker; neither complete
kernel compiles yet. No object/link, numerical, simulator or hardware result
is claimed; shared install remains unchanged. See
[commands, diagnostics and manifest](../../../Logs/run_reports/q2-item07-plt-20261029/report.txt).


## Signed I32-to-F32 nearest conversion — 2026-10-29

**Source:** the Q2 fused penalty/temperature capture uses 64 I32 lanes,
ALL/PB32, and rint. Native AVE selects signed S322F32 with rounding mode 0.

**Composition:** a bit-preserving VMI bitcast changes the signless I32 carrier
to explicit SI32, followed by `pto.vmi.sitofp`. No lane reordering or numeric
change occurs in the bitcast. All lanes are active and widths agree. Current
PTOAS physical sitofp hardcodes nearest-even, so the bridge rejects every
other rounding mode, partial/dynamic masks, other widths and extra attributes.
The sampling capture uses VL1/TRUNC; it remains unsupported. General masked
conversion and explicit rounding require a separately verified contract.

**Evidence:** baseline `ed872278c`, private rebuilt tools, 114 conversion tests
pass. The penalty/temperature capture now passes the bridge; sampling remains
rejected with the precise guard diagnostic. An aligned I32-load/F32-store probe
reaches LLVM with `llvm.hivm.vcvtif.s322f32.x(..., i32 0)` and no external
CCE helper. No object/link/runtime result claimed. Shared tools unchanged. See
[commands and manifest](../../../Logs/run_reports/q2-item08-vsitofp-20261029/report.txt).


## Dense BF16 packing after two narrowings — 2026-10-29

**Source:** recompute_w_u_fwd has two all-active F32[64]-to-BF16[64]
part_even conversions followed by DENSE vdintlv to BF16[128]; high is unused.
Native low vdintlv concatenates the sparse EVEN lanes of the first and second
register. Current PTOAS paired BF16 narrowing documents the same lane order.

**Composition:** retain both logical narrowings (including rounding/saturation),
channel_merge the two dense 64-lane results, then shuffle even channel indices
0,2,...126 followed by odd indices1,3,...127. Low is [A0..A63,B0..B63].
Replacing the unused high result with the same value has no observable effect;
a live high result is rejected. Reject unknown producers, partial narrowing
masks, other shapes/types and extra attributes. DENSE is consumed only for
this proven carrier; generic semantic deinterleave remains separate.

**Evidence:** baseline NPU-IR 2f2f7cb81, private tools,120/120 conversion tests.
The original capture now converts; an aligned load/narrow/pack/store probe
emits current PTOAS LLVM. Parser-sensitive test inputs use fully qualified
rounding attrs. No object/link or numerical result claimed; shared install
unchanged. [Commands/manifest](../../../Logs/run_reports/q2-item12-dense-packing-20261029/report.txt).


## FP8 narrowing and unresolved sparse compaction — 2026-10-29

**Supported:** all-active64-lane F32-to-f8E4M3FN RINT/NOSAT part_even
narrowing maps to logical VMI vcvt preserving type, rounding and saturation.
Current PTOAS lowers it to the F322F8E4M3 intrinsic. Only this observed form
is admitted; SAT/other rounding, partial masks and other widths stay rejected.

**Unresolved composition:** the attention capture follows narrowing with two
self-vdintlv carriers to compact one-in-four physical FP8 lanes. Logical VMI
already represents64 dense values; replaying physical compaction as semantic
logical deinterleave would corrupt lane order. Those consumers remain rejected,
including the subsequent wide-mask strided FP8 store. No general FP8-memory
coverage follows from accepting the isolated conversion.

**Evidence:** baseline87862b7b7, private rebuilt tools,122 passing conversion
tests; original now rejects the first FP8 vdintlv. A derived aligned VMI
load/convert/store probe emits LLVM with RINT/NOSAT and no CCE helper. This
is downstream conversion evidence, not bridge FP8-store or kernel success.
[Commands/manifest](../../../Logs/run_reports/q2-item13-fp8-narrowing-20261029/report.txt).


## Dense B16 one-point stores — 2026-10-29

**Source:** bwd_dtrap_ddt uses ONEPT_B16 with a dense128-lane F16 value,
VL1/PB16 and functionType=NORM. The initial worklist label suggested B32;
the exact failed-op note confirms B16. Native ONEPT_B16 stores element zero.

**Composition:** admit NORM only for a full128-lane B16 value and128-lane
VL1/PB16 mask, then use the existing logical lane-zero shuffle and singleton
VMI vstore. Type bits, base and element offset remain unchanged. Existing
PK acceptance is retained; shorter NORM carriers and continuous NORM streams
remain rejected. This does not solve state-carrying continuous one-point ABI.

**Evidence:** baseline d7c9d9644, private tools,126 conversion tests pass.
F16/BF16 aligned load/one-point-store probes emit native one-point stores
(dist4) through PTOAS LLVM. Original advances to a numeric BRC_B16 load with NORM metadata;
that newly exposed distribution contract remains unsupported. Predicate-memory
work stays independently deferred. [Commands/manifest](../../../Logs/run_reports/q2-item15-dense-onepoint-20261029/report.txt).


## Scalar right shift sign operand — 2026-10-29

**Source:** Mamba exponent/bit extraction uses 64-lane I32 vshrs,
H/PB32,constant shifts23/16 and sign=false. The initial signed-shift label
is corrected: false selects native unsigned zero-filling; true selects signed
arithmetic shift. VMI selects this behavior through its integer type.

**Composition:** bitcast the signless source to explicit UI32/SI32 according
to a constant sign operand, materialize the same in-range count as I16,
masked VMI vshrs, then bitcast back to the original I32 carrier. Preserve
the direct64-lane PB32 mask exactly; reject dynamic sign/count, counts outside
0..31, unknown attrs and other widths. No arithmetic conversion occurs in
the bitcasts and narrowing the proven count to I16 loses no bits.

**Evidence:** baseline255a9ad5b,private tools,128 passing conversion tests.
Aligned masked load/shift/store probes emit vshrs.u.x forfalse (23 and0)
and vshrs.s.x fortrue (16). Original now rejects vgather; no kernel success
claimed. [Commands/manifest](../../../Logs/run_reports/q2-item16-right-shift-20261029/report.txt).


## VL4 packed narrowing extension

Implementation: `8a34456df`.

The existing select-zero/narrow/packed-store composition now admits PB32 VL4
for64-lane F32-to-F16/BF16 RINT/NOSAT part_even conversion. All packing,
same-mask store, layout and no-escape guards remain.128 conversion tests pass;
both aligned F16/BF16 probes reach PTOAS LLVM; original gating capture converts.
See [evidence](../../../Logs/run_reports/q2-vl4-packed-narrowing-20261030/report.txt).


## H-masked BF16 rounding round-trip before add/exp stores

Implementation: `9610c76ef`.

Recognize64-lane F32->BF16 RINT/NOSAT part_even under H/PB32,one
same-mask BF16->F32 part_even widening,and one add/exp with only same-mask
NORM_B32 unit-stride F32 UB store consumers in the same block. Reject all
escaping values,other parts/attributes/masks and unsupported consumer graphs.
Composition: zero inactive F32 inputs,logical VMI vcvt,logical extf,then
original masked arithmetic/store. Retain both conversions to preserve BF16
rounding. Rewrite the proven pair together so acceptance does not depend on
which individual conversion pattern runs first. No predicate-memory lowering.
130 tests pass. Both aligned add/exp probes emit F322BF16 and BF162F32 LLVM
intrinsics. Seven DQKWG captures now reach deferred predicate casts.
[Evidence](../../../Logs/run_reports/q2-masked-narrow-widen-20261030/report.txt).


## I16 vmuls and short-carrier H/PB16 mask

Implementation: `8f2530cf9`.

Admit64/128 I16 vmuls with matching I16 scalar and ALL/H PB16 predicate.
Native H/PB16 activates64 physical lanes even when the logical carrier is64,
so reconstruct PAT_VL64 locally for that consumer.128-lane H and ALL reuse
existing masks; shared floating-point predicates stay unchanged. VMI vmuls
emits native signed I16 multiply with the same16-bit modular product; no
widening/saturation introduced.132 tests pass and derived VMI memory sinks
emit LLVM. Original3 captures advance to gather/broadcast.
[Evidence](../../../Logs/run_reports/q2-i16-vmuls-20261030/report.txt).


## I16 scalar broadcasts

Implementation: `4ef6a884c`.

Admit64/128 I16 carriers to existing typed constant-splat/VMI broadcast
lowering. Masked AVE broadcast must have matching all-active mask lanes;
unmasked scalar broadcast has no mask to drop. Preserve all16 bits for
zero,one,-32768 and dynamic scalars. No integer arithmetic widening.
134 tests pass,including partial/mismatched-mask rejections. Derived VMI
memory sinks emit LLVM; original top-p and two convolution captures now
reject I16 vsel. This is no claim of general I16 memory or select support.
[Evidence](../../../Logs/run_reports/q2-i16-broadcast-20261030/report.txt).


## I16 selection with PB16 and B32 comparison masks

Implementation: `81c65af67`.

64/128 I16 selection accepts matching direct PB16 ALL/H/VL4 masks;short64 H
requires local64-active-lane reconstruction. B32 comparison predicates reuse
the existing B16-bit zero-extension/B32-select/NOSAT-truncation composition.
All16 bits survive,including negative values; comparisons retain original
width and seed.136 conversion tests and derived LLVM probes pass. Top-p
now rejects I16 compare;convolution cases now reject gather.
[Evidence](../../../Logs/run_reports/q2-i16-vsel-20261030/report.txt).


## Native-width UB indexed gather

Implementation: `0d0500e43`.

Compose logical-origin base normalization and scalar-stride addptr with a
bit-preserving unsigned index view and logical VMI vgather(pmode=zero).
Vector offsets count source elements;do not rescale them by memref strides
or byte width. Native hardware applies element-size scaling. Short B16 H
carriers need local64-active-lane masks. Support64 I32/I32 gathers and64/128
F16/BF16 with I16 offsets;reject unsupported layouts,spaces,widths and attrs.
Nonzero/dynamic base offsets need explicit reinterpret casts. Signedness
view is required because current physical gather lowering demands UI16
even though the VMI surface accepts signless I16. No extension/truncation.
138 tests pass. Four memory probes in pointer and descriptor modes emit
LLVM with single base-offset/stride application. Two convolution captures
convert;delta/Mamba now expose vintlv/vshls.
[Evidence](../../../Logs/run_reports/q2-indexed-gather-20261030/report.txt).

## Native I64 helper graphs — implemented 2026-10-30

Two remaining Q2 captures allocate `vector_2xvl_s64` and use native signed
widening, broadcast, add and NORM_B64 store helpers. There is no one-to-one
logical I64 VMI register mapping. Approved workaround: recognize closed helper
graphs and replace them with low/high I32 SSA planes, explicit carry, and data
interleave plus identically duplicated/split PB32 store predicates. Initially
bound masks to observed ALL/VL4; preserve sign extension, zeroing, descriptor
origin and I64-to-I32 element scaling. Reject escaping/unknown graphs. Native
conversion ignores its mask, whereas addition and store preserve theirs.
Implemented in the AscendNPU-IR bridge: validate all pointer uses, private native
declarations, one writer per temporary, same-block write-before-read ordering,
constant allocation count one, and bounded contiguous rank-one UB stores before
rewriting. Widening uses arithmetic shift by 31 for the high word. VMI addc/addcs
propagate low-word carry into the high word; explicit selects preserve native
MODE_ZEROING. An ALL store interleaves both planes into two I32 stores; a VL4
store writes only eight words from the first register. Descriptor origin remains
in I64 units; only the helper access offset is scaled by two for the I32 view.
I64 UB pointer admission is address-only; I64 vector operations stay unsupported.

140 conversion tests pass, including eight unsafe-graph rejections. A host model
of actual emitted VMI passes 327 sign/carry/mask/store checks; this is not target
execution. Both isolated captured I64 graphs reach LLVM in pointer mode.
Pointer-backed probes reach LLVM in both modes with offsets applied once and
without native helper dependencies. Public descriptor-only probes still reject
unproven store alignment; no alignment claim or finalizer was added. Both full
captures advance to separate I8 conversion gaps, so neither full kernel is
reported fixed. No object/link, simulator or hardware validation.
[Implementation evidence](../../../Logs/run_reports/q2-i64-helper-20261030/report.md).
[Approved proposal and native source evidence](../../../Logs/run_reports/q2-i64-helper-proposal-20261030/proposal.md).

## Top-p I16 comparison with PB32 consumers — implemented 2026-10-30

After the I64 commit, `entry_fa23a1c7__top_p_filter_kernel` compares
I16 zero/one broadcast/select carriers with NE under VL2/PB32, then uses the
predicate in an F32 select. A direct I16 comparison would introduce B16
predicate granularity at a B32 consumer. Approved bounded composition: prove
the zero/one broadcast/select carrier graph, zero-extend its logical I16 values to
I32, and compare using the original VL2/PB32 governing mask. Keep unsupported
producers, mask uses and comparisons rejected. The proof concerns I16 samples at
B32 element boundaries, which are the comparison bits observed by the F32
consumer; it does not require assuming that unused adjacent halfwords match.
Keep the original partial seed, not ALL. This graph does not establish general
I16 comparison support. Ranked alternatives were broader typed predicate
conversion support or upstream normalization of the boolean carrier graph.

Implemented with exact 64-lane EQ/NE, VL2/PB32, uniform zero/one ALL/PB32
broadcasts, recursively selected by B32 comparisons under the same seed, and
only F32 select consumers. Depth is bounded; unknown data, constants, layout
attributes, seeds, tails, granularity, ordering and predicate escapes stay
rejected. 142 conversion tests pass, including eight new negative forms.
Host evaluation of actual emitted VMI passes 128 vectors/8192 lane checks;
pointer-backed EQ/NE probes reach LLVM in both modes without native helpers.
The exact captured first VF reaches LLVM in pointer mode; descriptor-only
stores retain the deferred alignment rejection. The full capture now rejects
a separate VL1 F16 widening graph. No full-kernel link or target numerical
validation is claimed.
[Evidence](../../../Logs/run_reports/q2-i16-boolean-cmp-20261030/report.md).

## FP8 native compaction before strided store — implemented 2026-10-30

After `7028dc525`, attention's 64-lane F32-to-f8E4M3FN conversion is
followed by two native self-deinterleave low results, then a 2x32 UB store
under native VL64/PB8, with stride 2080 FP8 elements. VMI narrowing is already
dense; logical deinterleaves would change the payload order. Approved guarded
fold: remove only that exact native compaction chain, retain the dense logical
result, and use typed FP8 VMI strided store with explicit VL64 byte predicate
and 65-block stride. Preserve both 32-byte row writes and base/access offsets.
Require unused high results and closed observed uses; reject general FP8
deinterleave and unknown layouts. Implemented by matching the complete graph
before conversion and replacing only its store/carriers with typed FP8 VMI
store; the original narrowing remains for ordinary attribute validation.
Require exact 2x32 UB shape/strides, zero access indices, supported attributes
and explicit reinterpret-cast provenance for nonzero/dynamic origins.

144 conversion tests pass, including thirteen unsafe-graph rejections.
A symbolic-byte model checks 224 vectors/14336 payload bytes and exact write
footprints; it does not model FP8 quantization or target execution. Pointer-
backed zero, static-offset and dynamic-offset probes emit LLVM in both modes
without native helper calls; the captured VF emits LLVM in pointer mode.
The VF's descriptor route stops at an earlier F32 reinterpret-cast validation
failure. The full attention capture advances to an unsupported FP8 CBUF pointer
cast, so full-kernel compilation remains blocked. No PTOAS changes, alignment
assertions, finalizers or predicate-memory changes.
[Implementation evidence](../../../Logs/run_reports/q2-fp8-compaction-20261030/report.md).
[Proposal and target-acceptance probes](../../../Logs/run_reports/q2-fp8-compaction-proposal-20261030/proposal.md).

## VL16/PB32 masked BF16 widening — implemented 2026-10-30

Next after FP8: four UNPK_B16 BF16-to-F32 part_even sites in
`prepare_wy_repr_bwd_tile16` use VL16/PB32 and same-mask direct/arithmetic
stores. Approved bounded extension of the existing H/VL1/VL4 composition:
zero inactive B16 inputs, use unmasked VMI extf, retain the original arithmetic
and store masks and the bounded consumer proof. Reject predicate changes,
escapes and cross-lane users; do not claim general masked widening. Implemented
only for BF16 VL16; F16 VL16 and other neighboring forms remain rejected.
146 conversion tests pass, including direct/arithmetic positive graphs and
eight rejection forms. Pointer-backed probes emit LLVM in both modes; the four
exact captured VFs emit LLVM in pointer mode. Descriptor-only VF stores retain
the deferred alignment rejection. The complete Q2 capture now reaches a
predicate-memory pointer cast, an explicitly deferred group: no new complete
capture passes (21/62 total, 20/61 NPU). No target execution or numerical claim.
[Implementation evidence](../../../Logs/run_reports/q2-bf16-vl16-widening-20261030/report.md).
[Proposal and inspected capture](../../../Logs/run_reports/q2-masked-bf16-widening-proposal-20261030/proposal.md).

## Delta short BF16 gather/interleave/widen — implemented 2026-10-30

Next after `cb0ba2722`: a 64-lane BF16 H/PB16 gather feeds a short self-vintlv
whose unused high result and ALL/PB32 part_even F32-widening low consumer
identify a native physical carrier operation. Approved bounded fold: widen
the original logical gather directly, preserving all 64 values and existing
gather masks, zeroing and element-offset semantics. Logical short interleave
would incorrectly split after 32 original inputs. Require the observed closed
graph; keep general short interleave rejected. Implemented with a single
widening low consumer, unused high result, closed gather uses, same block,
BF16/I16 types, H/PB16 gather and ALL/PB32 part_even F32 widening. Keep the
widening at its original insertion point and retain existing gather validation.
148 conversion tests pass, including nine unsafe-form rejections. A symbolic
model checks 80 vectors/5120 lane routes. Native-graph memory probes emit LLVM
in both modes; the captured VF emits LLVM in pointer mode. Descriptor-mode VF
emission retains a reinterpret-cast validation failure. The full Delta capture
now reaches deferred PIPE_S verification, so no new full capture passes.
[Implementation evidence](../../../Logs/run_reports/q2-delta-carrier-fold-20261030/report.md).
[Proposal and target probes](../../../Logs/run_reports/q2-delta-interleave-proposal-20261030/proposal.md).

## Full-register numeric BRC_B16/NORM — implemented 2026-10-30

Next after `219930ced`: Mamba's `entry_21ad6498_bwd_dtrap_ddt_kernel` rejects
singleton F16 UB BRC_B16 load to 128 lanes because the bridge accepts PK but
not NORM. Approved bounded extension: accept NORM only for full 128-lane B16
numeric output and reuse the existing one-lane VMI load plus native-width vbrc.
Keep current singleton/type/address/attribute checks and PK forms; reject
shortened NORM carriers. This reuses an existing composition rather than
claiming a one-to-one VMI mapping. Implemented by extending only the accepted
attribute condition for rank-one 128-lane results. Existing PK forms and all
singleton UB/element/index/attribute checks remain. 150 conversion tests pass;
F16/BF16 memory probes with zero/nonzero origins emit LLVM in both modes.
The original affected VF emits LLVM in pointer mode; descriptor-mode stores
retain the deferred alignment rejection. The complete Mamba capture advances
to a separate masked F16 widening graph, so no new full capture passes.
[Implementation evidence](../../../Logs/run_reports/q2-brc-b16-norm-20261030/report.md).
[Proposal](../../../Logs/run_reports/q2-brc-b16-norm-proposal-20261030/proposal.md).

## TRUNC SI32-to-F32 with only F16 point-store consumers — implemented 2026-10-30

Next after `f90e5c0fc`: top-p's masked TRUNC SI32-to-F32 result feeds only
RINT/NOSAT part_even F16 narrowing and matching ONEPT_B16 stores. Approved
bounded equivalence: zero inactive I32 lanes, use nearest VMI sitofp, retain
the half conversion/store. Integers through magnitude 2^24 are F32-exact;
larger magnitudes overflow F16 under either first-stage rounding, so final
half values agree. This does not implement general TRUNC F32 results. Require
the exact VL1/PB32 graph and reject all other F32 users/consumer formats.
CPU checks find zero final-half mismatches over 661782 inputs despite 254290
different intermediate F32 values; desired target probes emit LLVM in both
modes. Implemented with one same-block F16 narrowing consumer and only matching
same-block singleton rank-0/1 UB point stores. Existing general RINT/ALL support
is unchanged; arbitrary TRUNC F32 results remain rejected. 152 conversion tests
pass, including eleven rejection forms. Actual TRUNC-source memory probes emit
LLVM in both modes; the captured VF emits LLVM in pointer mode. Descriptor-mode
VF emission retains a memref-space conversion blocker. The full top-p capture
advances to a separate masked F16 widening graph; no new full capture passes.
CPU value evidence is unchanged (661782 inputs, zero final-half mismatches);
target numerical/FP-status validation remains untested.
[Implementation evidence](../../../Logs/run_reports/q2-trunc-half-chain-20261030/report.md).
[Proposal, proof and probes](../../../Logs/run_reports/q2-trunc-half-chain-proposal-20261030/proposal.md).

## Masked H/PB32 RINT signed I32-to-F32 — 2026-10-30

**Source and case:** Q2 `entry_b84b14f2_mamba3_siso_bwd_kernel_dk_state_post`,
fused-2 outlined VF: 64 I32 lanes to F32, H/PB32, RINT. Its F32 result flows
through same-mask lane-wise arithmetic and selects into an H-masked F32 UB
store. The left-shift fix exposes this as the next conversion rejection.

**Why not one-to-one:** Native SI32-to-F32 has an explicit mask and rounding
operand. Current VMI sitofp is unmasked and its physical lowering uses nearest
rounding. Passing arbitrary inactive I32 values to it can introduce extra
inexact conversions. Directly dropping the predicate is not the chosen mapping.

**Approved bridge composition:** `vsel(mask, I32 source, I32 zero)` → signed
I32 bitview → VMI sitofp (RINT). Preserve every active source bit, signedness,
nearest-even rounding and lane index. Zero converts exactly to positive F32
zero, avoiding additional inactive integer-conversion inexactness. The closed
consumer proof makes the inactive values unobservable at memory sinks.

**Semantic guards:** Only 64 signless I32-to-F32 lanes, direct H/PB32 PGE,
RINT, no part or extra conversion attributes. Follow at most 64 same-block
values through F32 add/subtract/multiply/scalar add/scalar multiply using the
identical mask SSA value, or lane-wise F32 selects. Every path must terminate
in a matching-mask NORM_B32 F32 UB rank-one unit-stride store; reject dead
paths, result escapes, calls/regions, cross-lane operations, changed masks,
other result types and unknown operation attributes. No address/stride,
store-footprint, mask, saturation, layout or synchronization rewrite is made.
ALL/RINT and the separate guarded VL1/TRUNC-to-F16 composition remain distinct.

**Alternatives ranked:** For the current bridge-only scope, the guarded zero
selection plus existing conversion is the smallest justified solution. A
proper masked/rounded VMI conversion is the cleaner long-term target contract
but requires an independently scoped PTOAS change. Unmasked conversion of
arbitrary inactive integers without sanitization is rejected.

**Status:** Committed as `abcacb918` on NPU-IR base `b7729ce6b`.
All 156 conversion tests pass, including 13 negative consumer/contract cases.
Actual-memory probes emit LLVM in both modes with zero → integer-select →
RINT conversion dataflow, retained store masks and typed offsets. The original
fused-2 VF converts in both modes and emits LLVM in pointer mode; descriptor
mode retains a proven-store-alignment rejection. The complete Mamba capture
now stops at a distinct 64-lane F16 self-interleave. No additional complete
capture passes: replay remains 21/62, NPU 20/61, no timeouts. Target numerical/
FP-status execution is not authorized or claimed. Deferred predicate-memory
and PIPE_S work stays untouched.
[Evidence](../../../Logs/run_reports/q2-masked-rint-sitofp-20261030/report.md).

## Ascend950 dense unaligned F16 self-interleave/widen — 2026-10-30

**Source and case:** Mamba `entry_b84b14f2_mamba3_siso_bwd_kernel_dk_state_post`
outlined VFs 4 and 9 load two F16 values with UNPK_B16 and a typed unaligned
marker, self-interleave a 64-lane carrier, discard the high result, then use
part_even F32 widening under VL2/PB32 and a matching two-value F32 store.

**Native proof and guard:** `HIVMAVEToAVEIntrin` on Ascend950 replaces the
unaligned-marked nonbroadcast load with dense `vldus.post` before selecting
load distributions. Physical F16 interleave duplicates each of the first
64 dense halfwords into a 128-halfword low result. Part-even widening selects
positions 0,2,...,126, recovering the original 64 halfwords. Ordinary aligned
UNPK_B16 loads have a different sparse carrier; this fold must not accept them.
No general logical-short-vector interleave equivalence is claimed.

**Approved composition:** Keep the dense logical VMI F16 load; replace the
interleave/widen pair with a fresh logical VL2 input predicate, select positive
F16 zero for inactive input lanes, then unmasked VMI extf. Preserve the original
VL2/PB32 store predicate, addresses/offsets/strides and two-F32 write footprint.
The separate input predicate permits B16 selection without changing B32 users.
Active F16 values, signs and lane order survive unchanged. Inactive signaling
NaNs are not sent to the unmasked conversion. Target FP-status is untested.

**Boundaries:** Ascend950 target and typed unaligned marker; static two-element
unit-stride F16 UB view at zero access index; same block; no other load users;
self-interleave without attrs; unused high and one low widening; part_even,
64 F16/F32 lanes, VL2/PB32; widening used only by matching same-block zero-index
NORM_B32 two-element unit-stride F32 UB stores. Existing load/widen/store attr
validation remains. Reject other targets/ordinary unpack, carrier/result
escapes, mask/part/type changes and different memory footprints.

**Status:** Committed as `fcad33231` on NPU-IR base `abcacb918`.
All 158 conversion tests pass, including 13 negative forms. Native-intrinsic
probes confirm the dense unaligned path versus ordinary sparse unpack; symbolic
carrier checks cover all 65536 halfword payloads (not target FP execution).
Aligned and aligned-offset memory probes emit LLVM in both modes, retaining
typed offsets and compatible VL2 input/store predicates. Original VFs 4 and 9
convert in both modes; unaligned/dynamic stores retain the existing LLVM
alignment rejection. Full Mamba now stops at a separate numeric I32 ONEPT_B32
store on a rank-two singleton view. Replay remains 21/62 (NPU 20/61), no
timeouts and no new full capture passes. Wider full-carrier reconstruction is
the fallback alternative, not part of this change. No PTOAS edits, finalizer
changes, predicate-memory work or blanket alignment assumptions.
[Evidence](../../../Logs/run_reports/q2-f16-load-carrier-fold-20261030/report.md).

## I32 ONEPT_B32 numeric UB stores — 2026-10-30

**Source and case:** Mamba `entry_b84b14f2_mamba3_siso_bwd_kernel_dk_state_post`
outlined VF 6 stores lane zero of a 64-I32 vector through ONEPT_B32, direct
VL1/PB32, into a dynamically offset `1x1xi32` UB view with an unaligned marker.
The existing one-point bridge path accepts F32 but excludes I32.

**Approved composition:** Reuse lane-zero VMI shuffle (omitted for a one-lane
value) plus unmasked one-element VMI vstore. Keep all 32 bits without floating
conversion, the exact destination pointer/index/descriptor offset and one-I32
write footprint. Direct VL1/PB32 makes the selected first lane active. Native
ONEPT_B32 stays on the one-point store distribution even with the unaligned
marker; the continuous native path explicitly writes four bytes without a
predicate. This does not introduce general masked I32 stores or scalar memref
store changes. It is an extension of the existing F32 composition, not a new
VMI instruction or missing target scalar-store capability.

**Guards:** Only matching signless I32 source/base types in UB; 1/64 source
lanes; matching direct VL1/PB32 predicate lane count. Retain existing NORM,
unaligned/continuous marker validation and static-stride address/index
normalization. Reject type/memory-space/mask/width mismatches, dynamic or other
masks, and unsupported attributes/strides. F32 and B16 float proofs remain
unchanged. No predicate-memory or PIPE_S changes and no alignment bypass.

**Status:** Committed as `70f1b643e` on NPU-IR base `fcad33231`.
All 160 conversion tests pass, including 12 new negative forms. Five memory
probes emit LLVM in both modes with one I32 native point store, exact lane-zero
projection and constant/dynamic offsets applied once. Native probes confirm
ONEPT_B32 distribution and the continuous four-byte write. Original VF 6
converts in both modes and emits LLVM in pointer mode; descriptor mode retains
a memref.reinterpret_cast elimination blocker. Full Mamba now stops at a
separate F16 PK_B32 direct-carrier store. Replay remains 21/62 (NPU 20/61),
no timeouts and no additional complete capture passes. Ranked choice remains
reuse of existing VMI point-store lowering before upstream AVE normalization
or a target change. No target numerical/hardware claim.
[Evidence](../../../Logs/run_reports/q2-i32-onepoint-store-20261030/report.md).

## Dense F16 self-deinterleave to unaligned PK_B32 prefix — 2026-10-30

**Source and case:** Mamba outlined VF 7 in
`entry_b84b14f2_mamba3_siso_bwd_kernel_dk_state_post`: 64-F32 ALL/PB32
RINT/NOSAT part_even narrowing, 64-F16 self-vdintlv with unused high result,
then PK_B32 into a two-F16 UB view under VL2/PB32 and the unaligned marker.

**Native proof:** Part-even narrowing occupies even physical halfwords. The
low native self-deinterleave concatenates their 64 converted payloads twice.
On Ascend950, a non-onepoint unaligned store uses VSTUS rather than the stated
PK distribution. getElemSizeByStoreMask computes two F16 values = four bytes,
so only the first two dense payloads are written. Ordinary aligned PK_B32
selects a different carrier projection and is explicitly outside the fold.

**Approved composition:** Retain the original ALL narrowing and its rounding/
saturation; VMI vcvt already has dense logical payloads. Replace the closed
self-vdintlv/store carrier with a logical VL2 masked VMI store of those values.
Keep memory base/view offsets and zero access index, exact two-F16 footprint,
all-lane conversion work and operation order. Do not narrow only two inputs or
zero the other F32 lanes: that would discard conversions present in the source.
The existing bridge graph-preparation phase permits atomic replacement before
individual conversion patterns; no Q2 or standalone pass is added.

**Guards:** Ascend950; typed unaligned PK_B32 marker/PK attrs; rank-one static
two-F16 unit-stride UB view, zero index; direct VL2/PB32 64-lane store mask;
same-block 64-F32-to-64-F16 ALL/PB32 RINT/NOSAT part_even narrowing, only used
by the self-deinterleave; one low-result store use, unused high; no carrier,
rounding, mask, type, footprint or attribute escapes. Normal address conversion
preserves constant/dynamic offsets. Reject aligned stores and other contracts.
No predicate-memory, PIPE_S, finalizer or alignment-policy change.

**Status:** Committed as `6fc233981` on NPU-IR base `70f1b643e`.
All 162 conversion tests pass, including 14 rejection forms. Native probes
confirm the four-byte dense unaligned store and contrasting aligned PK store;
carrier permutation checks cover all 65536 halfword payloads after conversion.
Aligned/offset memory probes emit LLVM in both modes with the full ALL RINT/
NOSAT conversion retained, no deinterleave, and exact typed offsets/footprint.
Original VF 7 converts in both modes; unaligned/dynamic destinations retain the
LLVM alignment rejection. Full Mamba now reaches memref.get_global for its
320-I32 GM constant table; that separate issue is unchanged. Replay remains
21/62 (NPU 20/61), no timeouts or new complete-capture passes. Ranked choice
remains the proved logical fold; explicit carrier reconstruction is the more
complex alternative. No target numerical/FP-status execution was performed.
[Evidence](../../../Logs/run_reports/q2-f16-packed-prefix-20261030/report.md).

## Private initialized I32 GM global address adaptation — 2026-10-30

**Source:** Mamba's private constant `@tbl`, a 320-I32 identity-layout GM array
with 32-byte alignment. memref.get_global feeds a unit-stride view/cast and the
existing native GM→UB helper. The previous downstream probe confirmed exact
1280-byte constant emission, GM addresses and scalar loads through LLVM and
numeric-space memref representations.

**Chosen mapping:** LLVM private constant array with the exact Dense I32
initializer/alignment and GM address space 1; LLVM addressof → i64 ptrtoint →
existing pointer_cast lowering for each original memref result. Pointer mode
obtains a PTO GM pointer. Preserved-memref mode retains the existing descriptor
path, with static array extent, zero offset and unit stride, for native helper
views/calls. This is a data/address/descriptor composition, not a new VMI op.

**Guards and alternatives:** Referenced HIVM globals only; private, constant,
initialized dense signless I32, positive static rank one, identity layout, GM,
natural-or-better optional alignment; no unknown global attributes. Preserve
symbol identity, constant bits, alignment, address space and original access
views. Keep existing native-call ABI policy; do not silently convert a
preserved native helper to bare-pointer ABI. Numeric-space and LLVM globals
remain outside this adaptation. Unsupported referenced forms reject. Retaining
memref globals through broad memory-space/type conversion is the less localized
alternative; no PTOAS change is justified by the successful downstream probe.

**Status:** Committed as `c10c70934` on NPU-IR base `6fc233981`.
All 165 conversion tests pass, including ten unsupported source/access forms.
Small and exact-table scalar/native-copy probes emit LLVM in both modes; exact
1280-byte initializer bits, GM address space and alignment are preserved.
Bridge output retains native-call memrefs/views in preserved mode. Standalone
PTOAS nevertheless emits pointer-valued wrapper parameters in both modes and
leaves `_mlir_ciface_load_gm_to_ubuf_1d_int32_t` external: native ABI/link
correctness is not established. Full captured Mamba passes preserved-mode
bridge conversion, then reaches the separate masked-store proven-alignment
failure; pointer mode retains a dynamic GM reinterpret-cast rejection.
Replay improves to 22/62 (NPU 21/61), no regressions/timeouts. No object/link/
target execution claim. Deferred predicates, PIPE_S, disabled finalizers and
alignment policy remain unchanged.
[Evidence](../../../Logs/run_reports/q2-readonly-i32-global-20261030/report.md).


## I32-to-I8 PP0 / PK4_B32 store — capability probe, 2026-11-02

**Status:** Committed as `0032a7750` on branch
`fix/bridge-i32-i8-packed-store`, base `a0876a760`.
Captured Mamba ddt/dtrap/dinput_states VF 0 now converts in both modes; the full
capture passes the preserved-mode bridge. Observed contract: 64-I32 flags
AND255, ALL/PB32 NOSAT SI2UI PP0/functionType EVEN, 64-I8 result → physical
256-I8 cast → 64-I32 bitcast → aligned PK4_B32 store to 64-I8 unit-stride UB
storage. Native selector 0/distribution 12 select one low byte per I32 lane.

**Composition:** Match the closed same-block graph and emit logical NOSAT
VMI trunci + exact 64-lane UI8 store, retaining range/predicate work and
original view origin. UI8 matches the existing byte-pointer policy and retains
all bits, including 128–255. Preserved numeric rank-one I8 UB addresses retain
I8 on the descriptor carrier; consumers form UI8 byte views. Native descriptor
signatures stay I8, verified with a shared native-call regression. This exact
address representation correction does not change I1 or Cube pointer rules.

**Guards:** Ascend950, fixed64 lanes, direct ALL/PB32 masks, exact range proof,
PP0/SI2UI/NOSAT/EVEN, one-use physical carrier chain and no unknown attrs;
rank-one exact64-I8 unit-stride UB, zero access index, explicit view for a
nonzero/unknown origin. Reject other PP/SAT/UNI/masks, escapes, widths, spaces,
strides/footprints and unaligned-marked source stores. The latter writes
**256 native bytes**, so it cannot be folded into this 64-byte operation.

**Validation:** 170/170 conversion tests pass; 24 negative forms reject. All
16 memory probes emit LLVM (both bridge modes and both PTOAS builds), and
original VF pointer mode emits helper-free LLVM on both builds. Preserved VF
LLVM retains a prior store-alignment rejection. Full-kernel pinned LLVM rejects
a dynamic native-call pointer bridge; standalone rejects a surviving view.
The original full capture now passes preserved-mode bridge conversion: fresh
baseline/current replay improves 22/62 → 23/62, NPU21/61 →22/61, no regressions
or timeouts. Byte projection/emitted-IR checks cover 268 vectors/6432 memory
comparisons, exact64-byte writes and intact surrounding bytes; no hardware/
simulator numerical check. No PTOAS, finalizer, deferred predicate/PIPE_S or
alignment-policy change. Separate broad narrowing/carrier/store patterns remain
a possible later extension, not part of this bounded composition.
[Implementation, commands and limits](../../../Logs/run_reports/q2-i32-i8-packed-store-20261102/report.md).


## Direct aligned F16 VL2 widening — 2026-11-02

**Status:** Implemented in NPU-IR commit `94b01f69d` on
`fix/bridge-f16-vl2-extf` (base `dc304dd99`). This extends the existing guarded masked-widening composition;
it does not replace the earlier unaligned-load/self-interleave VL2 fold.
Original Top-P sample/filter fused_1 VFs use ordinary aligned UNPK_B16 F16
load → part_even widening → matching VL2/PB32 NORM_B32 store, with two active
values and no interleave.

**Composition:** Reuse zero-inactive-F16 selection before unmasked VMI extf,
then keep the original masked F32 store. Native and target use identical
UNPK distribution 14 and conversion part 0; actual LLVM uses B32 VL2 selection
and store masks. Active input bits/order are preserved, inactive logical
conversion inputs become +F16 zero, and each output store writes 8 bytes.
View origins remain applied once in F16/F32 element units.

**Guards:** Ascend950; fixed 64 F16/F32 lanes, direct VL2/PB32, part_even;
one-use aligned UNPK_B16 load from an exact two-F16 unit-stride rank-one UB
view, zero access index, normal PK attrs; only same-block direct matching-mask
NORM_B32 stores to exact two-F32 UB views with zero indices and normal attrs.
Nonzero/unknown origins require explicit reinterpret views. Reject other
producers/escapes, arithmetic/cross-block users, masks/parts/types/spaces,
shapes/strides/indices/attrs, unaligned markers and PLT63. An unaligned load is
dense and changes the observed second halfword, so it remains excluded.

**Validation:** 172/172 conversion tests pass, including 23 rejected forms.
Twelve memory LLVM probes pass in both modes/builds; both original VFs convert
in both modes and emit LLVM in pointer mode on both builds. Preserved original
VF LLVM retains the store-alignment rejection. Symbolic carrier checks cover
all 65,536 halfword payloads in both active positions and 196,608 exact 8-byte write
comparisons; this is not floating-point or target execution. Replay remains
23/62 (NPU 22/61), no regressions: both Top-P captures pass vextf and expose a
singleton/rank-zero I32 cast (sampling) and masked VL2 F32→F16 narrowing
(filtering). Mamba PLT63 remains separate. No target code, finalizer,
predicate-memory/PIPE_S or alignment-policy change.
[Evidence, files and limits](../../../Logs/run_reports/q2-f16-vl2-extf-20261102/report.md).

## Guarded I32 singleton wrappers — 2026-11-02

Extends existing singleton scalar-round-trip handling to signless I32:
rank-zero/singleton wrappers share one VMI lane; scalar uses must all return
through matching broadcasts, and one-point stores use lane-zero projection.
This is a graph-dependent composition, not general vector-to-scalar extraction.
Bits, lane ordering and existing address/store handling are preserved; no
rounding, signedness conversion, scratch memory or synchronization is added.
Scalar arithmetic/returns, mixed scalar users and multi-lane extraction stay
rejected. No predicate-memory or finalizer change.

174 conversion tests pass. Pointer-mode reduced LLVM passes on both PTOAS
builds; preserved-mode LLVM retains raw memref memory-space conversion errors.
Replay remains 23/62 without regressions; sampling advances to unsupported I32
reduction. [Files, provenance and limits](../../../Logs/run_reports/q2-i32-singleton-20261102/report.md).

## Signed I32 minimum reduction — 2026-11-02

Bounded 64-lane I32 MIN under ALL/VL2 PB32 maps to signed bitcast → masked
VMI reduce_mini (one lane) → signless bitcast → result broadcast. The reduction
itself has an existing VMI mapping; the full AVE result representation needs
this composition. Explicit si32 preserves signed ordering through PTOAS;
bitcasts preserve payload bits and the original mask excludes inactive lanes.
Other integer kinds/widths/shapes/masks remain rejected. No PTOAS changes,
finalizers, predicate-memory, scratch or synchronization changes.

176 conversion tests pass. Four capability and four generated LLVM probes pass
on both PTOAS builds and select signed vcmin.s. Capture replay stays 23/62
without regressions; sampling advances to unsupported elementwise vsmin.
No runtime/numerical or object/link claim.
[Evidence and guards](../../../Logs/run_reports/q2-i32-min-reduction-20261102/report.md).

## Signed I32 elementwise minimum — 2026-11-02

AVE vsmin on 64 I32 lanes with direct ALL/VL2 PB32 maps to signed bitcasts of
both operands → masked VMI vmin → signless result bitcast. The minimum has an
existing target operation; representation changes preserve bits and signed
ordering through PTOAS normalization. Original predicates are retained.
Other widths/shapes/masks/distributions, unknown attrs and integer max/unsigned
min remain excluded. No target, allocation, synchronization or finalizer change.

178 conversion tests pass, including seven valid rejection inputs. Four target
capability and four compiler-generated LLVM probes pass on both PTOAS builds;
the signed vmin and store share the same predicate. Replay stays 23/62 without
regressions; sampling advances to I32 ADD reduction. No numerical/runtime claim.
[Evidence and limits](../../../Logs/run_reports/q2-i32-vsmin-20261102/report.md).

## Bounded I32 addition reduction — 2026-11-02

AVE ADD on 64 signless I32 lanes with direct ALL/VL2 PB32 maps to masked VMI
reduce_addi (one lane) followed by broadcast to the AVE shape. The reduction
has an existing target operation; vector result representation needs the
composition. No numeric cast, saturation, widening or FP reassociation flag.
Keep the original predicate and reject other widths/shapes/masks/distributions,
unsigned element types and unknown attrs. No target, scratch, synchronization,
finalizer or predicate-memory change.

180 conversion tests pass, including seven valid rejection inputs. Four target
capability and four generated LLVM probes pass on both PTOAS builds and use
integer vcadd.s.x; reduction and store share the predicate. Replay remains
23/62 without regressions; sampling advances to i64 → scalar I32 UB pointer_cast.
No numerical/runtime or object/link claim.
[Evidence, files and limits](../../../Logs/run_reports/q2-i32-add-reduction-20261102/report.md).

## Scalar I32 UB pointer views — 2026-11-02

Admit rank-zero identity-layout signless I32 UB with one I64 address and no
dynamic sizes. Direct mode uses the existing one-to-one pto.castptr; preserved
mode reuses descriptor materialization. This adds no new non-direct mapping or
PTOAS requirement. Byte addresses, scalar storage and zero access index remain
unchanged; no scratch, synchronization, finalizer or predicate-memory changes.
Other scalar types/spaces/layouts and multiple addresses remain excluded by
this extension, while existing rank-one/SSBUF/Cube forms remain unchanged.

182 tests pass with seven valid rejection inputs. Constant and dynamic origins
emit LLVM in both modes on both PTOAS builds, retaining byte origin once and
one zero-index I32 store. Replay remains 23/62 without regressions; sampling
advances to numeric scalar I32 UB load, which remains unsupported.
[Files, provenance and stage limits](../../../Logs/run_reports/q2-scalar-i32-pointer-20261102/report.md).

## Guarded scalar I32 UB loads — 2026-11-02

Identity-layout rank-zero signless I32 UB load maps directly to PTO scalar load
at offset zero. Preserved mode reuses existing descriptor normalization; this
adds no new target representation workaround. Accept only ordinary access and
optional markDCacheInvalidatePatternVisited=0:i32 bookkeeping. Reject bypass,
nontemporal, unknown attrs, other marker forms/types/ranks/layouts. Preserve byte
origin once; no vector extraction, scratch, volatile or synchronization changes.

184 tests pass, ten valid negatives reject, and marked/plain loads emit LLVM
in both modes on both PTOAS builds with one four-byte read. Capture replay rises
to 24/62 without regressions: sampling converts. Its full preserved capture
emits pinned LLVM but retains six external native-helper dependencies; standalone
fails residual memref.reinterpret_cast validation. No object/link/runtime claim.
[Evidence, files and limits](../../../Logs/run_reports/q2-scalar-i32-load-20261102/report.md).

## F16 VL2 narrowing to packed stores — 2026-11-02

Ascend950 64-F32→64-F16 RINT/NOSAT part_even under direct VL2/PB32, used only
by closed same-block carrier graphs into matching normal PK_B32 stores to exact
two-F16 unit-stride UB views at index zero. Extend existing guarded narrowing:
select positive zero for inactive F32 inputs → unmasked VMI conversion → retain
masked packed store. Active input bits/selectors are preserved; native carrier
PK_B32 and VMI F16 PK_B32 select the same even halfwords and write four bytes.
Preserve view origins once; reject escapes, other types/parts/masks/rounding,
unaligned/unknown attrs, layouts and footprints. No target/finalizer/alignment,
predicate-memory, pass, scratch or synchronization change.

186 tests pass with 15 valid rejected forms. Twelve memory LLVM probes pass in
both modes/builds; original VF pointer LLVM passes, preserved VF alignment stays
rejected. Native/target intrinsic and SSA inspection plus 393,216 symbolic packing
comparisons preserve four-byte footprint, without numerical/FP-status execution.
Replay rises to 25/62 without regressions: filtering converts and emits pinned
LLVM; standalone retains descriptor reinterpret-cast rejection. External native
helpers remain, without object/link/runtime claims.
[Files, proof and limits](../../../Logs/run_reports/q2-f16-vl2-narrow-20261102/report.md).

## Guarded F16 PLT63 widening — 2026-11-02

Extend existing widening composition for Ascend950 constant PLT(63)/PB32 with
unused updated extent. Exact 63-F16 aligned UNPK view → part-even widening →
same-mask F32 lane-wise arithmetic → exact 63-F32 NORM_B32 UB stores, zero indices,
same block and closed graph. Fresh prefix 63 selects zero for inactive F16 lane 63
before unmasked VMI extf; retain masked arithmetic/store and origins once.
Native/target active unpack, bits and conversion part match. Existing PLT carry
rules remain unchanged; reject live carry, different counts/types/views/attrs,
escapes, cross-lane operations and changed masks. No new pass, target, template,
finalizer, predicate-memory, alignment, scratch or synchronization change.

188 tests pass with 22 valid rejected forms; the previous deliberate PLT63
rejection is now a positive regression. Six target capability and 12 memory LLVM
probes pass on both builds; original VF pointer LLVM passes, preserved VF retains
alignment rejection. Twenty LLVM sequences and 589,824 symbolic output/252-byte
footprint comparisons checked, without numerical/FP-status execution.
Replay rises to 26/62 without regressions: Mamba converts and emits pinned LLVM;
standalone retains descriptor reinterpret-cast validation. Four external native
helper dependencies remain; no object/link/runtime claim.
[Files, proof and limits](../../../Logs/run_reports/q2-plt63-extf-20261102/report.md).

## Numeric I8 constant-one four-byte fill — 2026-11-02

Ascend950 ALL/PB32 I8 constant-one broadcast → closed 64/256-byte carrier and
I32 bitcast → VL4/PB32 aligned PK4_B32 store into exact four-I8 unit-stride UB
view, zero index. Native distribution 12 chooses one byte per active I32 word;
uniformity proves equivalence to dense four-byte VMI constant/store. Target
uses PLT4/B8 NORM store; origins remain in byte units exactly once. Reject
variable/other constants, masks, unaligned/unknown attrs, footprints/layouts,
spaces and escapes. The unaligned source writes 16 bytes, so it is not folded.
This composition does not enable general I8 expansion or predicate-memory.
Existing graph preparation only; no new pass, target, ABI, scratch or sync.

190 tests pass with 18 valid rejection forms. Six capability and 12 memory LLVM
probes pass on both builds; original VF pointer LLVM passes, preserved mode
retains F16 alignment rejection. Twenty LLVM sequences and 16,384 byte-layout/
footprint comparisons checked without target execution. Replay rises to 27/62
without regressions. Full sampler LLVM exposes a dynamic GM F16 descriptor
ABI rejection in pinned PTOAS; standalone retains reinterpret-cast validation.
No full-kernel LLVM, object/link/runtime or numerical claim.
[Files, proof and stage results](../../../Logs/run_reports/q2-i8-four-byte-fill-20261102/report.md).


## Preserved logical I1 memref and raw pointer views — 2026-11-02

The user approved the descriptor/view boundary, without reopening deferred
predicate-memory encoding. In preserved-memref mode, static positive
identity-layout I1 UB pointer_cast roots use an element-matched signless-I8
PTO pointer/byte descriptor plus standard memref.view. This composition keeps
original logical I1 sizes, strides and offsets for metadata and native users;
explicit aligned-buffer pointer extraction shares the address with raw byte
VMI consumers. No allocation/copy or predicate-packing rule is introduced.
The carrier extent is descriptor-construction metadata. Invalid direct
UI8-pointer-to-I1-descriptor casts remain rejected. Dynamic/nonidentity roots,
non-i64 addresses, empty/overflowing extents and existing unsupported spaces
remain outside this new composition.

A module-scoped guard rejects logical I1 tile-buffer pointer projection when
these roots/views exist, including inside outlined functions. This preserves
the predicate-memory deferral; it deliberately also rejects unrelated logical
I1 accesses in such a module. The initial unguarded 46/62 acceptance exposed
inherited byte-expanded predicate memory and is not the final supported result.
No finalizer activation, encoding, alignment or PIPE_S policy changes.

Private compiler build and 192/192 conversion tests pass. Both reduced probes
emit pinned LLVM with correct descriptor fields: [32,256,1]/[256,1,1]/offset0,
and selected [32,64]/[256,1]/offset8. Raw VMI uses the original aligned address.
Standalone retains preserved memory-space conversion failure. Final original
capture replay remains 27/62 (26/61 NPU), with no regressions; 20 former cast
failures now reach explicit deferred predicate-memory rejection. No full
frontend, object/link/runtime or numerical claim; observer fixture symbols
remain external. Compiler source and tests committed as `def3b1265`; this ledger remains local.
[Implementation, manifest, commands and limits](../../../Logs/run_reports/q2-i1-dual-views-20261102/report.md).


## Exact constant-prefix numeric stores — 2026-11-02

User approved a shared bounded bridge composition for normal rank-one
unit-stride UB stores: F32 prefixes2/4/8 of64 and F16 prefix7 of128, direct
constant masks and compatible PB granularity, on A5. VMI shuffle selects [0,N),
then dense VMI store writes exactly N elements. Values, order, base address,
descriptor offsets, access indices and other mask uses are preserved. Existing
normal-store attribute checks remain. No asserted alignment, enlarged write,
read-modify-write, new allocation, PTOAS edit, finalizer or predicate support.
Unsupported shapes/layouts/masks/packed forms retain prior lowering/rejection.

194/194 conversion tests pass. Pointer-mode converter output emits pinned and
standalone LLVM with exact8/16/32/14-byte vstus.post/vstas sequences; generic
preserved descriptor argument LLVM probes retain memory-space conversion
failure. Actual saved frontend MLIR replay produces8/11 ELF objects; all11
original first stores advance, with two later F16 VL4 stores and one packed
PB32 F16 narrowing/store still rejected. Full143-case sweep and execution were
not rerun. Compiler/tests committed as `4a35e286e`; Planner documentation remains local.
[Commands, cases, proof and limitations](../../../Logs/run_reports/q2-prefix-store-fix-20261102/report.md).


## I32 scalar-store default policy — 2026-11-02

The existing one-to-one scalar PTO store mapping now accepts explicit
nontemporal=false as well as absence of the default flag. It still rejects true
and unknown attributes; shape, identity layout, UB, value and index checks
remain. No new composition, load policy, ABI, alignment, predicate or PTOAS
change.195 conversion tests and four reduced pinned/standalone LLVM emissions
pass. Three sampler recipes clear their original store guard: reject sampler
produces an ELF object; top-p reaches I32 load attributes; join-probability
reaches the existing dynamic descriptor ABI rejection. No full sweep or runtime
claim. Compiler change committed separately; Planner notes remain local.
[Commands, source/tool manifest and results](../../../Logs/run_reports/q2-i32-store-default-policy-20261102/report.md).


## I32 scalar-load default policy — 2026-11-02

The existing one-to-one rank-zero I32 UB scalar load now accepts explicit
nontemporal=false after rejecting true with the existing typed check. The
zero-i32 cache marker and all other attribute/shape/type/space/index guards
remain strict. No new composition, ABI, alignment, predicate or PTOAS change.
195 conversion tests and four pinned/standalone LLVM emissions pass; the
original top-p saved frontend recipe now produces a pinned PTOAS ELF device
object with FALLBACK=1. No full sweep or runtime/numerical claim. Compiler
change committed separately; Planner notes remain local.
[Exact commands, manifest and stage results](../../../Logs/run_reports/q2-i32-load-default-policy-20261102/report.md).


## F16 prefix4 extension — 2026-11-02

Approved extension of the existing exact dense-prefix composition: F16 prefix4
of128, normal UB store, direct PGE/constant PLT and compatible PB16. One shared
eligibility condition changes; first4 lane projection and dense store preserve
bits/order, original dual views/address/index and8-byte footprint. Existing
unit-stride, A5, layout, type and attribute guards remain. Packed PB32, other
prefixes, alignment assertions, predicate/finalizer and PTOAS changes remain
outside scope.195 tests and four pinned/standalone pointer/descriptor LLVM
checks pass. Both original convolution-update saved frontend recipes now emit
ELF device objects with FALLBACK=1. No full sweep or numerical/runtime claim.
Compiler change committed separately; Planner notes remain local.
[Manifest, commands and results](../../../Logs/run_reports/q2-f16-prefix4-store-20261102/report.md).


## Guarded packed F16 graph: dense2 final store — 2026-11-02

The approved Mamba fix reuses the existing guarded ALL RINT/NOSAT EVEN narrow
and closed self-deinterleave matcher. Only its final memory composition changes:
full64 logical conversion -> first2 projection -> dense F16 store2, exact4 bytes.
The original matcher guards, upstream FP work and dual-view offsets remain;
other packed forms, aligned source, parts/rounding/saturation, widths and escapes
stay excluded. Current native source confirms unaligned PK_B32 uses a byte-count
VSTUS path; target LLVM explicitly packs the sparse result before its four-byte
store. No new pass, PTOAS, ABI, alignment assertion, sync, predicate or finalizer.
195 tests pass, including15 packed negative fixtures; four pinned/standalone
pointer/descriptor LLVM checks pass. Original saved frontend Mamba state-post
recipe now emits an ELF device object with FALLBACK=1. No full sweep or runtime/
numerical claim. Compiler change committed separately; Planner notes stay local.
[Proof, commands and provenance](../../../Logs/run_reports/q2-packed-f16-dense2-20261102/report.md).


## Buffer-only native indirect-load GM adapter — 2026-11-02

User approved a scoped ABI composition for the first operand of the exact
external indirect_load_nonvolatile_1d_half_int64_t signature. After launch ABI
conversion, trace only dynamic identity GM F16 launch carriers through safe
metadata-only casts; use metadata base -> explicit offset0,size0,stride1 sentinel
view -> original argument type. The source extent remains unknown, not invented.
Native source and the actual installed bitcode body read only GM aligned pointer;
loop bounds and all other offsets/strides come from the four unchanged UB
operands. Real explicit-view metadata and shared users are preserved. Unknown
helpers/signatures/roots and arbitrary metadata readers are not adapted. The
native contract must be revalidated if its implementation begins reading GM
metadata. No generic cast legality, launch ABI, native/PTOAS, predicate, alignment,
allocation policy or finalizer changes.

197 tests pass. Pinned LLVM field audit checks all25 call fields for both opaque
and real-sized source views. Three unknown/signature/shared-user probes retain
original pinned rejection. Both original join-probability and Mamba DDT saved
frontend recipes now emit ELF device objects with FALLBACK=1. Standalone retains
its reinterpret-cast validation limitation. No full sweep or runtime/numerical
claim. Compiler source/tests committed as `882396b41`; Planner notes remain local.
[Proof, native binary audit and commands](../../../Logs/run_reports/q2-indirect-buffer-adapter-20261102/report.md).

## Native template stack storage preservation — 2026-11-03

The user selected native-call preservation for caller-owned `llvm.alloca`
storage, analogous to external-call memref preservation. In
`preserve-external-call-memrefs=true` mode, allocations used directly and
exclusively by marked native fallback calls retain their element type, count,
alignment and pointer identity. This is a native ABI boundary, not a VMI
register mapping. Ordinary-call uses, pointer escapes and allocations outside
that mode remain rejected. Existing supported I64 graph-to-VMI normalization
is unchanged; no native template, PTOAS, predicate-memory or finalizer changes.

Implemented on `fix/native-template-alloca` from `e5f16345a`, uncommitted.
The converter and tools were rebuilt privately against existing build libraries;
168 AVE-to-VMI tests pass. Pinned PTOAS LLVM emission retains the aggregate
allocation and both producer/consumer pointer arguments. This does not establish
native linking or numerical correctness. Replaying the captured I64 DMA padding
case now passes allocation legality but still rejects the native predicate cast
from logical PB32 lanes to the physical native-call mask ABI. That separate
boundary remains unresolved. [Evidence and commands](../../../Logs/run_reports/native-template-alloca-2026-11-03/report.txt).

## I16/I8 normal masked padding stores — 2026-11-03

The user approved AVE-to-VMI support for the observed integer padding fills:
`NORM_B16` with 128 I16 lanes/PB16 and `NORM_B8` with 256 I8 lanes/PB8.
Full masks use VMI store only when the whole value fits; direct prefix masks
remain VMI vstore predicates. The VL32 examples write exactly 64/32 UB bytes.
I16 maps directly; I8 uses a same-width I8-to-UI8 value bitcast for VMI's byte
pointer representation. This composition changes no payload bits or lane count.
It is not a missing PTOAS instruction.

Guards require matching widths/granularity, static rank-one unit-stride UB
views, constant in-bounds indices and aligned effective offsets. Unknown masks,
short I16 carriers, strides and unaligned/continuous forms remain rejected.
No general integer arithmetic, predicate-memory or finalizer support is added.
Native byte descriptors and eligible local-helper pointer parameters retain I8;
VMI consumers obtain UI8 views separately. Descriptor recovery uses that same
native type. Existing helper specialization still requires aligned addresses
from every caller; unknown callers keep descriptors and gain no alignment claim.

Implemented uncommitted on `fix/narrow-integer-masked-stores` from `d4a75c4e4`.
171 conversion tests pass. Both original source-derived failing cases now emit
pinned PTOAS LLVM. Six simulator executions pass exact whole-backing-allocation
comparisons: paired native/bridge I16 pad-zero and I8 small-odd cases, then bridge
I16/I8 nonzero left-padding cases. Padding arguments are checked where requested;
DMA calls remain native fallback and padding computation is VMI. Native linking
and execution were exercised, not just profiler completion. Initial private
resource-layout/capture failures are retained separately. This is not the full
225-case matrix or physical A5 validation. Unknown-alignment descriptor-only
micro-probes still retain PTOAS's downstream rejection.
[Commands, provenance and results](../../../Logs/run_reports/narrow-integer-stores-2026-11-03/report.txt).
