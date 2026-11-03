# Retained vector DMA tests after native-contract triage

Date: 2026-11-03. Current branch/worktree: `mani/vec-dma-test` at
`$HOME/worktrees/AscendNPU-IR-vec-dma-test`. Test publication is recorded in the
[publication manifest](vector-dma-retained-2026-11-03/publication.json).

This supersedes the collection totals in the historical
[132-candidate expansion](vector-dma-contract-expansion-2026-11-03.md).
Its original source snapshots, failures, logs and saved tensors remain intact.
The current collection is **302 cases: 210 retained original cases plus 92
retained expansion cases**.

## Owner-directed pruning and retained coverage

- Removed 23 atomic datatype/mode combinations rejected by Triton's frontend.
  Retained all 22 native-generated atomic/reset contracts, which pass both flows.
- Removed all 16 FirstElem reachability probes: 15 had correct outputs but
  emitted Value mode; one was also a native 3D I64 numerical failure.
- Removed the other native 3D I64 nonzero-padding numerical failure, from the
  intended Null/cache probe. Both native failures remain documented in the
  historical report; no native code or golden result was changed.
- Kept the 15 supported nonzero-to-zero sequences, renamed
  `value-padding-minus7-to-zero`. They now explicitly assert **Value mode2 and
  EvictFirst0**, as well as the actual -7/zero scalar sequence. These are
  Value-padding state tests, not claims that Null/FirstElem/EvictLast were fixed.
- Removed 15 redundant original standalone zero-padding cases replaced by
  those sequences.
- Kept 16 original left-padding3 corner cases, including I64 scalars with
  distinct upper/lower halves. The new sequences have no left padding and do
  not subsume those arguments. Kept the original 3D I64 zero case because that
  instance has no retained nonzero transition.

The kernel is AST-identical to the expansion kernel. Its existing internal
pattern selector is now decoupled from the anticipated native padding mode by
an explicit `value_transition` field. Numerical reference logic is unchanged
under that equivalent selector substitution. The retained sequences use
EvictFirst instead of requesting an unpropagated EvictLast hint. All 15 were
rerun through native and bridge paths after this change.

More operations do not automatically make one testcase a superset of another.
Only the duplicated zero cases were removed; different left-padding and I64
scalar-packing contracts were preserved.

## Validation and evidence

The 92 retained expansion cases have passing native contract/numerical evidence:
77 unchanged cases from the completed expansion matrix plus 15 fresh Value-state
reruns. All 15 fresh native reruns pass. The 210 original retained configurations
have their earlier native baseline evidence. This is **not a fresh full302
native or bridge sweep**.

| Retained expansion flow | Full contract passes | Conversion/lowering blocked |
| --- | ---: | ---: |
| Native, 92 configurations | 92 | 0 |
| Bridge, native DMA fallback only, 92 configurations | 56 | 36 |

The 15 fresh transition bridge reruns yield nine passes and six existing
I8/I16/I64 conversion blockers. The remaining 77 configurations reuse unchanged
completed expansion evidence. Numerical tolerances, exact symbol checks, full
reference outputs and guards remain unchanged.

Bridge settings remain `FALLBACK=1` and
`TEMPLATE_TEST_DMA_NATIVE_FALLBACK_ONLY=1`. The immutable private compiler prefix
and matching A5 fallback bitcode are the ones recorded in the expansion report;
vector DMA PTODSL resources remain excluded. No vector DMA replacement validation
is claimed. The shared compiler/resources/installations are untouched.

- [Every retained expansion case and its evidence](vector-dma-retained-2026-11-03/cases.csv).
- [All 36 bridge failures](vector-dma-retained-2026-11-03/bridge-failures.csv).
- [Grouped failures with every configuration name](vector-dma-retained-2026-11-03/failure-groups.json).
- [Retention/provenance audit](vector-dma-retained-2026-11-03/retention-audit.json).
- [Exactly 15 original zero cases removed](vector-dma-retained-2026-11-03/removed-original-zero-cases.json).
- [Native green expansion nodes](vector-dma-retained-2026-11-03/native-green-nodes.txt)
  and [bridge green expansion nodes](vector-dma-retained-2026-11-03/bridge-green-nodes.txt).

Fresh local logs and current reproduction commands are under
`$HOME/tmp/dma-retained-20261103/`, with the reruns in `state-native/` and
`state-bridge/`. Unchanged cases reference their original expansion attempt
directories. `historical-source/` preserves pre-pruning test sources; historical
removed pytest IDs intentionally no longer collect in the current checkout.

## Why the remaining bridge cases fail, and what to do

Keep these **36 native-passing regression cases**. Their failures are outside
the native DMA template call; allowing native dependencies does not supply
missing conversion of the surrounding vector operations.

| Group | Cases | Cause | Repair scope |
| --- | ---: | --- | --- |
| I8/I16 `ave.hir.masked_store` | 18 | Normal-store conversion rejects the observed integer datatype/distribution combinations used around DMA | Scoped AVE-to-VMI datatype, mask and memory-store conversion |
| I64 `llvm.alloca` | 12 | Native vector-broadcast helpers use temporary pointer storage; the bridge does not legalize/recover the observed helper graph | AVE-to-VMI helper normalization, or verified preservation of allocation/helper ABI through fallback |
| `ave.hir.vgather` layout/index/mask | 3 | Permuted layouts generate a gather contract not accepted by the current carrier/layout/mask checks | Scoped AVE-to-VMI gather conversion |
| PTOAS masked-store alignment | 3 | Per-lane masked writes lack the required proven destination alignment; the available unaligned form only supports a contiguous low-bit prefix | PTOAS alignment propagation or legal masked-store decomposition |

The three gather cases are:

- `store_ubuf_to_gm_2d_float` / `permuted-layout`.
- `store_ubuf_to_gm_3d_float` / `permuted-layout`.
- `store_ubuf_to_gm_3d_int32_t` / `permuted-layout`.

Source inspection narrows the gather restrictions: the converter's current
B32 branch accepts I32 results, excluding the observed F32 results. The 3D I32
case also has a rank3 source view and VL2 predicate outside the accepted forms.
Base-offset/subview handling may need attention after the initial gate; adding
F32 to one predicate alone is not established as a complete fix.

The three alignment cases are:

- `store_ubuf_to_gm_2d_bfloat16_t` / `permuted-layout`.
- `store_ubuf_to_gm_2d_half` / `permuted-layout`.
- `store_ubuf_to_gm_2d_int32_t` / `permuted-layout`.

In the captured BF16 alignment example, the caller uses UB base128 bytes and
rows spaced by 16 BF16 elements (32 bytes). The destination is passed through an
outlined helper's memref argument, then reconstructed by pointer extraction.
This suggests missing alignment provenance across that boundary; it does not
establish that the actual allocation is unaligned. Carrying valid alignment facts
or recognizing a legal store form needs separate compiler/PTOAS validation.

The I8/I16 and I64 cases are listed exhaustively in the companion failure CSV
and JSON. Existing I64 allocation work belongs to its assigned compiler agent.
Do not delete native-valid tests to make bridge totals green, suppress alignment
checks, silently select another instance, or treat these as missing DMA PTODSL
bodies: all vector DMA replacement resources are excluded in this experiment.

When fixes arrive, rerun the failing groups with the same references and exact
native fallback checks. Restore full resources and require exact `__pto_`
selection plus materialized bodies separately when validating DMA replacements.

Physical UB padding beyond the logical Triton tensor and physical A5 hardware
remain outside this simulator validation.
