# Vector DMA combined contract expansion

> Current retained suite: [302-case retention update](vector-dma-contract-retention-2026-11-03.md).
> The 357-case collection and 132-candidate results below describe the historical
> expansion run; owner-directed pruning and Value-state revalidation are separate.

Date: 2026-11-03. Test worktree: `$HOME/worktrees/AscendNPU-IR-vec-dma-test`,
branch `mani/vec-dma-test`, base test commit `ab3f19b409f39375e9ea9a132d6f91a0113f2a9b`.
The retained test changes are published on that branch; see the
[publication record](vector-dma-retained-2026-11-03/publication.json). The earlier published
[225-case baseline](vector-dma-native-bridge-2026-11-03.md) remains separate;
it was not rerun in full during this expansion.

## Scope and final results

Added **132 explicit combined contract candidates** to the six existing
load/store 1D/2D/3D pytest modules, using `dma_contract_utils.py` and the existing
shared template-runtime utilities. Collection is now **357 cases**: 225 original
cases plus 132 new candidates. No runtime-input matrix or special IEEE-value
matrix was added. No compiler, native CCE template, shared installation, or
template implementation was changed by this testing work.

The user explicitly requested **native vector DMA fallback rather than PTODSL
DMA replacement validation**. The bridge used `FALLBACK=1` and a private copy of
the current rebuilt compiler, with nine vector DMA resources excluded only from
that private prefix. No `__pto_load_gm_to_ub*`, `__pto_store_ubuf_to_gm*`, or
`__pto_copy_ubuf_to_*` calls/bodies are permitted by this experiment's assertions.
Unrelated resource files are preserved. Shared resources were not removed.

| New candidates | Count | Final native result |
| --- | ---: | --- |
| Atomic Add/Max/Min plus ordinary-store reset | 45 | 22 contract passes; 23 frontend rejections |
| Padding-mode/cache probes, with combined Value-padding state sequence | 32 | 30 numerical passes with requested-contract gaps; two numerical failures |
| Singleton tensors/axes | 34 | 34 contract passes |
| Permuted 2D/3D layouts | 20 | 20 contract passes |
| Large transfer crossing the native split threshold | 1 | One contract pass |
| **Total** | **132** | **77 contract passes, 30 contract gaps, 23 frontend-unsupported, two numerical failures** |

All **107 cases with passing exact native numerical and guard checks** were
then run through the bridge. This includes the 30 strict argument-coverage
probes, without pretending that their requested argument values were generated.
The 23 frontend rejections and two native numerical failures were not run through
the bridge.

| Bridge outcome, native DMA fallback only | Count |
| --- | ---: |
| Full contract, exact native selection/fallback metadata, exact numerics pass | 47 |
| Exact numerics pass, intended padding/cache arguments still not generated | 18 |
| NPU-IR AVE-to-VMI conversion fails | 39 |
| PTOAS lowering fails | 3 |
| **Attempted** | **107** |

There are **65 exact bridge numerical/guard passes**, of which 47 also satisfy
the intended full contract. There are no bridge numerical mismatches among the
cases that reached execution. This experiment establishes **zero vector DMA
PTODSL replacement passes**, because those replacements were explicitly excluded.

The full collection is not entirely green: unsupported/unreached candidates and
native numerical failures retain their strict assertions. No blanket xfail,
skip, datatype substitution, relaxed tolerance, or arbitrary-callee acceptance
was added to make the totals pass. The companion green-node lists identify the
77 native and 47 bridge contract passes for focused reruns.

## Per-case evidence

The authoritative local evidence root is
`$HOME/tmp/dma-contracts-20261103/`.

- [All 132 cases](vector-dma-contracts-2026-11-03/cases.csv): explicit configuration,
  anticipated native symbol, actual selected symbol where compilation completed,
  actual scalar operand proofs, both flow outcomes, dependency fallback, errors,
  reproduction commands and log/evidence directories.
- [Failures and coverage gaps](vector-dma-contracts-2026-11-03/failures-and-gaps.csv).
- [Per-instance summary](vector-dma-contracts-2026-11-03/instance-summary.json).
- [Bridge failure groups and every affected case name](vector-dma-contracts-2026-11-03/failure-groups.json).
- [Final audit and per-area totals](vector-dma-contracts-2026-11-03/final-audit.json).
- [Native green nodes](vector-dma-contracts-2026-11-03/native-green-nodes.txt)
  and [bridge green nodes](vector-dma-contracts-2026-11-03/bridge-green-nodes.txt).

Native evidence comes from `native-pinned/` plus the corrected 14-case
`native-singleton-corrected/` rerun. Bridge evidence comes from `bridge-pinned/`
plus `bridge-singleton-corrected/`. Earlier diagnostic/pilot outcomes are
preserved but are not extra unique configurations or final passes.

Each runtime attempt retains the actual compiler input/checkpoint, compiler
command/status/log, pytest result, runtime settings/library maps, selected IR and
calls when reached, exact numerical counts, and saved tensors on mismatch.
Bridge attempts also retain actual PTOAS commands/inputs/errors when reached.
Large logs, binaries, profiles and tensors remain local rather than in Planner.

## What the new cases establish

### Atomic stores and reset

Two programs use different inputs to atomically update the same initialized GM
region, then perform ordinary stores into separate nonzero-initialized regions.
Both final data and untouched guards are compared exactly. Actual native call
operands must include the requested atomic kind and a subsequent kind-zero call
for the same exact intended store instance; the reset call alone cannot satisfy
atomic coverage.

All 22 native-generated atomic/reset contracts pass both flows:

- F32 and I32: Add/Max/Min for the tested 1D, 2D and 3D instances: 18 cases.
- F16 and BF16: Add for the tested 1D and 2D instances: four cases.

The 23 remaining candidates fail in Triton semantic checking before NPU-IR:
I8 Add/Max/Min (nine), I16 Add/Max/Min (six), F16 Max/Min (four), BF16 Max/Min
(four). These are frontend reachability limits, not evidence that native CCE
hardware atomic helpers fail. I64 hardware atomic modes were outside the
eligible native store datatype set and were not added.

Representative complete pass: `store/1d/f32/atomic-add-reset`, case67,
`store_ubuf_to_gm_1d_float`. Logs are in `native-pinned/0067/` and
`bridge-pinned/0067/`.

### Padding, eviction and logical state transitions

The Null candidate uses a masked load without an `other` operand and observes
only its valid lanes. In the same kernel, two additional loads from distinct
source windows observe complete logical tiles padded first with -7, then zero.
The FirstElem reachability candidate uses `other=first_value`; that semantic
equivalence is deliberately insufficient unless actual native mode1 appears.

The observed native ABI consistently uses **Value mode2** for these patterns,
including the intended Null/FirstElem probes. The actual eviction operand is
**EvictFirst0**, including calls requested with Triton `evict_last`.
The captured early compiler inputs already lack the requested eviction hint on
the generated memory copy. These are failures to generate the intended contract
through this tested Triton path, not PTODSL DMA implementation failures.

There are 30 native strict coverage gaps with correct outputs; the two remaining
padding candidates expose the I64 numerical defect below. The bridge reproduces
18 of those coverage gaps numerically; 12 are blocked earlier by I8/I16 or I64
vector conversion.

The separate Value(-7)-to-Value(0) logical-padding sequence is verified in
**15 native cases and nine bridge cases**. Its results are retained even where
the broader Null/cache candidate fails its argument assertion.

All padding claims concern **logical Triton positions made observable by the
kernel**. Physical UB padding outside the logical value remains unobserved.
Checking the actual padding arguments does not close that physical-padding gap.

### Native 3D I64 numerical defect

Exact intended instance: `load_gm_to_ubuf_3d_int64_t`.
Cases56/57 are `load/3d/i64/pad-null-evict-last` and
`load/3d/i64/pad-firstelem-evict-first`, with logical shape `(2,3,11)`, tile
`(2,4,16)`, source strides `(113,23,1)` and no additional offset/left padding.
Both actually execute **Value mode2**, not FirstElem mode1.

Each has exactly six wrong positions: the first logical right-padding element
at column11 in each of six valid rows. Valid source data and guards match.

| Intended I64 padding | Observed I64 padding |
| --- | --- |
| `-7` | `-25769803783` |
| `137438954016` (`0x2000000220`) | `2336462209568` (`0x22000000220`) |

Every wrong value repeats the intended scalar's low32 bits in both halves.
Independent [saved-tensor/actual-operand analysis](vector-dma-contracts-2026-11-03/native-i64-padding-analysis.json)
confirms the positions, values and native Value-mode arguments.

This is consistent with missing B64 padding repair in the native 3D path:
`Copy1D.cpp` and `Copy2D.cpp` explicitly invoke `align_pad_for_load_b64_*` after
their contiguous DMA paths; the analogous `Copy3D.cpp` branch returns after its
3D helper without that repair. This is a source-supported diagnosis; no native
compiler/template fix was attempted. The existing baseline's nonzero-I64 probe
used a different left-padding contract and does not contradict this new case.

These two cases are retained as native failures and were not run through the
bridge or counted as FirstElem coverage. A native/template investigator should
verify the 3D B64 correction with these exact reproductions.

### Singleton dimensions, layouts and splitting

All 34 singleton cases pass native validation. Actual evidence shows the tested
2D singleton descriptors retain rank2; the tested 3D middle-singleton-axis
descriptors collapse to rank2. Only that actual anticipated target is credited.
An initial rank1 expectation for 14 rank2 cases was a test-author mistake. Only
the explicit factory expectation was corrected; all 14 were rerun and passed.
The kernel, shapes, reference, and exact-symbol assertion remain unchanged.

All 20 permuted-layout cases pass native validation. Five pass the full bridge
contract; 15 expose the bridge limitations below. Different constant layouts
therefore provide useful coverage without a separate runtime-argument matrix.

The F32 split case passes both flows:
`load/2d/f32/burst-split`, exact `load_gm_to_ubuf_2d_float`, shape `(4093,8)`,
tile `(4096,8)`. Actual captured descriptors prove GM strides `[16,1]`, UB strides
`[8,1]`, no left padding, and 4093 rows. The native F32 aligned burst chunk is
`floor(4095/8)*8 = 4088`, leaving a five-row remainder. The proof is in
`native-pinned/0033/evidence/burst-split-proof.json` and its bridge counterpart.
This establishes the native branch predicate and exact numerics, not an
instruction/performance trace.

The first 4101x8/tile8192x8 diagnostic candidate correctly exceeded UB
(2,097,152 bits requested versus 2,031,616 available). It was explicitly replaced
by the named legal case above; no runtime block-size clamp was introduced.
A literal-descriptor-extent inspector bug found in the intermediate large-case
probe was fixed; the final full-matrix large case passes its complete checks.

## Bridge blockers, with DMA replacements excluded

| Failure group | Cases | Representative |
| --- | ---: | --- |
| AVE ordinary I8/I16 masked-store datatype conversion | 22 | `load/1d/i16/singleton` |
| I64 native helper temporary: `llvm.alloca` legalization | 14 | `load/1d/i64/singleton` |
| AVE `vgather` layout/index-carrier/mask restriction | 3 | `store/2d/f32/permuted-layout` |
| PTOAS masked-store destination alignment proof | 3 | `store/2d/bf16/permuted-layout` |

All 42 have passing native numerical/guard baselines. These are surrounding
vector conversion/PTOAS limitations, not failed vector DMA PTODSL templates.

The three gather cases are `store/2d/f32/permuted-layout`,
`store/3d/f32/permuted-layout`, and `store/3d/i32/permuted-layout`.
The converter rejects the observed `ave.hir.vgather` because it requires native
B16/B32 UB shapes, matching unsigned index carriers and direct PB16/PB32 masks.
Next action: inspect the actual gather contract and extend the corresponding
carrier/layout/mask conversion; preserve the native DMA fallback calls.

The three alignment cases are `store/2d/bf16/permuted-layout`,
`store/2d/f16/permuted-layout`, and `store/2d/i32/permuted-layout`.
PTOAS rejects a VMI masked store whose destination alignment is not proven:
the available unaligned form writes a contiguous low-bit prefix, while the
observed operation requires per-lane write predicates. Next action: establish
the required alignment or lower the valid unaligned masked access without
writing outside its semantic footprint. Do not disable the alignment check.

For I8/I16 and I64, retain the previously reported datatype/helper investigations.
`FALLBACK=1` permits native dependencies but does not automatically legalize
remaining AVE/LLVM operations. Preserving the I64 allocation/helper graph as
native fallback remains a possible separate approach; this work did not patch
the converter or claim that accepting `alloca` alone is sufficient.

## Toolchain and audit

The shared compiler changed while the template developer was rebuilding. A
hash guard stopped the next batch before launch. The owner then explicitly
selected the current rebuilt compiler and later directed the exclusion of
vector DMA PTODSL instances. The authoritative run uses:

- NPU-IR version `a0876a76082e`, build date 2026-11-03, with uncommitted DMA
  developer work. This is not the earlier 2026-11-02 binary despite the same Git
  version string. Exact tested binary/resource hashes are the provenance anchor.
- A private copy at `$HOME/tmp/dma-contracts-20261103/native-dma-prefix/`,
  including matching native c310 bitcode. Resource discovery is executable-relative.
  The [private-prefix manifest](vector-dma-contracts-2026-11-03/private-prefix-manifest.json)
  lists every copied artifact and all nine excluded vector DMA resources.
- Existing `$HOME/.venv/npuir-sim-3.6-candidate/bin/python`; CANN9.2 compiler
  child wrappers and CANN9.1.0-beta.3 parent runtime; simulator `Ascend950PR_9589`.
- Pinned PTOAS `5ce40f710f7f83b8d1fb2d8c36647bb811236a49` / CLI0.69,
  PTO LLVM `fa2fd1f75d742fec91ecf40d98c9c7961647ec42`,
  Ascend LLVM `95c4693447af2473be62bdee19522cbe596f9e96`.
- Bridge mode `ptodsl`, `FALLBACK=1`,
  `TEMPLATE_TEST_DMA_NATIVE_FALLBACK_ONLY=1`; private AIC/AIV fallback bundles.
  Native runs unset the bridge mode and use enable0/fallback0.

The audit verifies all 107 bridge flow/target/fallback settings, actual native
DMA calls at the compiler checkpoint, absence of vector DMA PTODSL calls/bodies,
68 PTOAS inputs without such replacements, and 65 completed exact native-target
selection/fallback-metadata/numerical checks. All 60 protected final files match
their recorded hashes. The original kernel, input patterns, CPU reference prefix
and original pytest functions are AST-unchanged. Thirty-two previous native
padding-operand proofs also replay successfully after the inspector extension.

The launcher diagnostics are retained separately: sandbox-local simulator sockets
were initially blocked; a native retry mistakenly retained bridge mode; an
intermediate private opt launcher routed to the compiler. These setup errors were
corrected and their attempts excluded. No shared installation was patched.

This is simulator functional coverage. Physical A5, performance, physical UB
padding outside the logical tensor, unreachable native contracts and full-resource
DMA replacement integration are not established.

## Handoff

Tests remain in the existing six pytest files under
`bishengir/test/Integration/PTOASBridge/template_runtime/a5/Vector/DMA/native-instances/`.
The README explains candidate versus supported coverage and the explicit
native-fallback-only mode. Keep the supported native configurations, exact target
and scalar checks, full references and guards. Preserve the native failures and
reachability gaps separately. Compiler/PTOAS fixes and implementation of the new
DMA replacements belong to their assigned developers.

After the DMA developer provides a stable full-resource handoff, replacement
validation must restore those resources and require the exact intended `__pto_`
target plus materialized PTO body. The native-fallback passes in this record
cannot substitute for that validation.
