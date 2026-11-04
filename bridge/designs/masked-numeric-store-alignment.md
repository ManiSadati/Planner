# Masked numeric-store alignment — partial bridge support

Last updated: 2026-10-29

The human initially deferred this item and prohibited PTOAS source changes.
A subsequent MmadL1 investigation led to explicit approval of dual memory views:
keep pointers for VMI and descriptors for native calls. That bounded bridge
change is implemented and compilation-tested; see the
[mapping entry](ave-vmi-non-one-to-one-mappings.md#dual-pointerdescriptor-views--2026-10-29).
It recovers existing address provenance rather than asserting new alignment.
Unknown cases, PTOAS changes and general alignment work remain deferred.
No new pass, finalizer, address rounding or physical-vector workaround was added.

## Reproduced failure

At NPU-IR `e275b5649`, the retained Q2 norm-backward D128 and convolution-backward
initial captures pass preserved-memref AVE-to-VMI conversion. The retained
PTOAS driver rejects a VL16/VL32 masked F32 UB store because it cannot prove
32-byte destination alignment. VMI carries 64 lanes, but only 16/32 may write;
a full-register store is not an equivalent replacement.

The bridge extracts an integer aligned-pointer field from a preserved memref,
casts it to `pto.ptr` and retains descriptor/access offsets separately. The
pinned PTOAS remainder analysis recognizes limited address expressions but not
that extraction or standard `memref.assume_alignment` evidence. A reduced input
with an explicit 32-byte assume still fails. Keeping `hivm.storage_aligned`
alone is not a verified solution.

## Investigation and alternatives

The first norm destination selects UB addresses 29696 and 134400, both divisible
by 32. This proves that particular base, not every UB address or dynamic view.
Any future implementation must trace call sites, descriptor offsets, access
indices, aliases and all branches before propagating an alignment fact.

- Propagating standard alignment metadata has normally negligible runtime cost,
  but the pinned PTOAS does not consume it here. A scoped address-analysis
  extension was proposed; the human rejected PTOAS modification.
- Helper specialization/inlining could expose actual addresses. Scratch probes
  removing `no_inline` and allowing normal private-helper inlining still leave
  the full modules blocked. Code-size/register-pressure effects were not measured.
- A bridge-only arithmetic representation of already-proven alignment might be
  recognized by PTOAS. It is not implemented or validated. It must preserve the
  address exactly; do not round an unknown address or infer alignment from UB
  space or shape. Extra scalar-instruction cost has not been measured.

Keep vector stores in VMI and physical lowering in PTOAS. Keep unknown or
misaligned cases rejected; do not widen the semantic write footprint to obtain
acceptance. Predicate-memory support is separately deferred.

## Evidence

[Investigation and probes](../../../Logs/run_reports/q2-masked-store-alignment-2026-10-29/report.md)
contain native allocation traces, actual PTOAS commands, alignment-assume and
inlining probes. This is compiler evidence only, not linking, simulator,
hardware, numerical or performance validation.

## Dual-view validation — 2026-10-29

Against base NPU-IR e275b5649 / PTOAS 5ce40f710, the reduced MmadL1 row
consumer now passes VPTO and LLVM emission. Shared/native helper regressions
retain descriptor calls and direct VMI pointers. Internal outlined static
identity-layout arguments are specialized only when every direct caller proves
32-byte base alignment. A mixed aligned/misaligned caller test retains the
descriptor signature and still fails downstream, as required.

The complete retained norm-backward D128 and convolution-backward initial
captures remain blocked on descriptor-backed public helper arguments; do not
claim all numeric stores supported. Public generated helper signatures are
preserved, rather than inferring private visibility from their names. The current MmadL1 pytest matrix has not been run.
See the [fresh commands and results](../../../Logs/run_reports/bridge-dual-memory-views-2026-10-29/report.md).


## Compiler-owned outlined helpers — current follow-up

The human subsequently authorized fixing compiler-generated outlined bodies
and callers even when source-public. Ownership now uses the explicit VF
annotations on both definition and every direct call, rather than public
visibility or a name heuristic. Unmarked public functions and external native
descriptor ABIs remain unchanged. The all-call alignment proof remains
mandatory. The captured actual nn-disabled kernel now emits a device binary,
and norm D128 passes VPTO; convolution initial still has a later unproven
store. See the [source-capture replay and diagnostics](../../../Logs/run_reports/bridge-outlined-memory-views-2026-10-29/report.md).


## Approved constant-prefix composition — 2026-11-02

The human authorized the bounded bridge-only normal-prefix workaround. It
preserves existing dual views and emits a short dense store of exactly the
proven active lanes, so it also handles sub-alignment strides without inventing
an alignment fact. Initial coverage: F32 N=2/4/8 of64 and F16 N=7 of128, direct
constant prefix, rank-one unit-stride UB, A5, compatible lane granularity.

194 conversion tests pass. Of11 fresh full-recompile alignment failures, replay
now produces8 ELF device objects. All11 original first stores are resolved;
two convolution updates expose later F16 VL4 stores and Mamba exposes a packed
PB32 narrowing/store. Those are outside this composition. Unknown/general
alignment, other prefix forms, predicate memory, PTOAS changes and finalizers
remain outside this approval. Generic preserved descriptor LLVM probes still
show an independent memory-space conversion limitation. No runtime/numerical
claim. Compiler/tests committed as `4a35e286e`; Planner notes remain local.
[Exact evidence and remaining cases](../../../Logs/run_reports/q2-prefix-store-fix-20261102/report.md).


## Approved F16 prefix4 extension — 2026-11-02

The existing constant-prefix bridge composition now includes F16 prefix4 of128
with PB16, selecting exactly4 values and writing8 bytes without asserting base
alignment. Both convolution-update recipes previously blocked by this later
store now produce pinned PTOAS ELF device objects.195 conversion tests and
four pointer/descriptor LLVM emissions pass on pinned/standalone tools. Packed
PB32 Mamba narrowing remains a separate contract. No full sweep or execution
claim; compiler change committed separately and Planner notes remain local.
[Exact evidence](../../../Logs/run_reports/q2-f16-prefix4-store-20261102/report.md).


## Guarded packed F16 final store — 2026-11-02

The remaining Mamba state-post graph already matched the guarded packing fold;
its final wide masked store caused alignment rejection. The approved bridge
change projects the first2 logical narrowed values into a dense F16 store2,
without reducing the original full64 conversion work or weakening its matcher.
195 tests and four LLVM checks pass; actual target LLVM packs sparse conversion
payloads and writes4 bytes. Mamba state-post now emits a pinned ELF device object.
No full-suite rerun or numerical claim; other ABI/predicate/pipe gaps remain.
[Exact evidence](../../../Logs/run_reports/q2-packed-f16-dense2-20261102/report.md).
