# Native template census: two 10,000-configuration samples

Recorded: 2026-10-19. Status: both campaigns finished; all 20,000 selected
configurations attempted. **191 exact native symbols were observed.**

This is a native NPU-IR dependency census of Q2TritonKernel and
AscendNPU-IR-Test Python cases on A5 (`Ascend950PR_9589`). Bridge and fallback
were disabled. Shape-only host preparation intercepted original JIT calls;
native compilation retained parse-checked modules after `convert-hivm-to-std`
and `convert-hivmave-to-std`. A private guard stopped the external binary
backend. No kernels, simulator execution, final native linking, numerical
checks, or PTODSL implementations were validated by these campaigns.

The [exact-symbol inventory](native-template-census-symbols-2026-10-19.md) and its
[CSV](native-template-census-symbols-2026-10-19.csv) identify the dependency requirements encountered so far.
They do not establish PTODSL support or replace the focused implementation and
numerical prerequisites in the
[development guide](../designs/ptodsl-template-development-and-fallback.md).

## Unique instances by category

Counts deduplicate exact canonical symbols across all usable captures,
including captures recovered from incomplete entries. Q2 alone contributes
30 unique symbols; the table combines both source suites.

| Category | A | B | Combined unique | Tentative extrapolation |
|---|---:|---:|---:|---|
| MmadL1 | 15 | 14 | 16 | ≈16 |
| Fixpipe | 8 | 7 | 8 | ≈8 |
| Nd2Nz | 4 | 4 | 4 | ≈4 |
| Cube setup | 4 | 4 | 4 | ≈4 |
| Cube copy | 1 | 0 | 1 | Insufficient data |
| Vector DMA | 45 | 51 | 52 | ≈56 |
| Vector Integer64 | 37 | 38 | 46 | ≈64 |
| Vector collective | 6 | 7 | 9 | ≈9–10 |
| Vector math | 3 | 3 | 3 | ≈3 |
| SIMT indexing | 0 | 1 | 1 | Insufficient data |
| Support: printing | 17 | 22 | 25 | ≈29 |
| Support: cube debug | 4 | 4 | 4 | ≈4 |
| Support: vector debug | 2 | 2 | 2 | ≈2 |
| Support: synchronization | 2 | 2 | 2 | ≈2 |
| Other uncatalogued helpers | 13 | 14 | 14 | ≈14 |
| **Total** | **161** | **173** | **191** | **≈220 for sampled paths; unobserved families unknown** |

Two symbols absent from the 2,941-symbol catalog are assigned to their observed
semantic families here: `mma_tile_half_to_half` and
`fixpipe_nz2nd_half_to_half_4d_to_2d_gm`. They remain marked `catalogued=false`
in the CSV. The remaining 14 uncatalogued helpers include transpose,
broadcast, and addition calls. This does not modify the exported catalog or
existing PTODSL implementation/test statuses.

Machine-readable [category counts and estimates](native-template-census-categories-2026-10-19.csv) also include
static call sites, deduplicated test-file/function counts and rare-symbol
frequencies. Symbols shared between A/B count once in the combined column;
categories can share test files, so file totals are not additive.

## What the estimate means

The ≈220 figure is an exploratory **lower-end extrapolation for additional
configurations of the sampled native paths**, not a reliable prediction for a
complete, repaired CI suite. It does not include an estimate for families
never observed, other architectures, all MLIR/lit tests, or paths excluded or
blocked by harness/frontend incompatibilities. Sparse Copy/SIMT categories
are deliberately reported as insufficient data. An unchanged estimate for a
category does not prove all its instances were exercised.

Using unique native compilations as incidence sites, the bias-corrected
[Chao incidence estimator](https://vegandevs.github.io/vegan/reference/specpool.html)
is:

```text
S_est = S_observed + ((N - 1) / N) * Q1 * (Q1 - 1) / (2 * (Q2 + 1))
```

Here N=7,770 unique compilations, Q1=29 symbols seen in one compilation, and
Q2=14 seen in two. The pooled calculation is 218.063; estimating categories
separately gives 216.831 in total, both rounded to about 220. The category
table uses separately calculated, rounded values. This is not a confidence
interval: configurations were deliberately selected and are correlated,
failed paths are systematic, and unobserved feature families cannot be
inferred from these frequencies. It is not an upper bound or a sufficient
implementation budget for unrestricted full CI.

## Campaigns, usable dumps and coverage gaps

| Measure | A | B |
|---|---:|---:|
| Selected / attempted configurations | 10,000 | 10,000 |
| Selected test functions | 1,736 | 1,446 |
| Workers | 16 | 8 |
| Elapsed wall time | 4h 30m | 8h 14m |
| Complete entries | 3,674 | 3,663 |
| Partial entries with usable dumps | 420 | 201 |
| Missing-result entries with recovered usable dumps | 139 | 109 |
| Total entries with usable checkpoints | 4,233 | 3,973 |
| Unique usable native compilations | 3,975 | 3,884 |
| Unique symbols | 161 | 173 |

The samples have zero exact or normalized compiler-relevant configuration
overlap. B uses source-declared alternative shapes, dtypes and tuning choices
from the same frozen sources, with ordinary and corner-case preferences.
290 original functions had no fresh configuration under the selection policy;
the eight exhausted network slots were redistributed among the four major
families. Both use the same toolchain/harness/environment fingerprint. Their
compiled specialization sets overlap by 89; different source configurations
can produce the same native input. The symbol sets share 143; B adds 30.

| Final entry status | A | B |
|---|---:|---:|
| Complete | 3,674 | 3,663 |
| Partial | 420 | 201 |
| Preparation blocked | 2,845 | 2,983 |
| Compilation failed | 803 | 1,031 |
| Missing final result | 1,893 | 1,705 |
| Timeout | 358 | 411 |
| Target not launched | 7 | 6 |

Only 8,206/20,000 configurations yielded at least one usable checkpoint;
all recovered captures are included in the inventory. Other failures did not
yield usable symbol checkpoints. Failures are preserved with worker/compiler
logs; a wrapper status alone does not identify the failing compiler pass.
Missing evidence is not feature absence. A valid zero-call dump, including
some pure SIMT compilations, remains a valid compilation observation.

## Provenance and limits of symbol coverage

The private compiler reports NPU-IR `e9a8145b74223ced608ffbae9d411627562f85ec`,
but was copied from a dirty build. Its SHA-256 is
`015a2c0398eb68484fedbae900f1657322dadd5e5ffe8366d930e960a4e39299`;
the recorded revision is not proof of identity with a clean source checkout.
Other recorded revisions are Ascend LLVM
`8ce9217f83f3376970d10428de4e2d1c15039e08`, pinned PTOAS
`d36100d03255e5e8a858c497a4277708d8b889c9`, and PTO LLVM
`fa2fd1f75d742fec91ecf40d98c9c7961647ec42`.
Test-source revisions are Q2 `ebf78b205a87ecc5cd935a214bd2ef0fc176dcd9`
and AscendNPU-IR-Test `17bae1e6e00eacffa7e7594f966069c0b21869d9`.

Both campaigns shared one private, read-only snapshot of compiler/resources,
Python environment, harness and test sources, with separate writable outputs
and caches. Final toolchain integrity checks passed. An earlier shared-install
campaign stopped when binaries changed; its
results are excluded. Full metadata and artifact hashes are in
[the evidence record](native-template-census-evidence-2026-10-19.json). Large manifests, IR and logs remain outside
Git under the portable artifact locations listed there.

Seeing a native symbol does not prove that all controls/descriptors for that
symbol are supported. A separate source-only comparison used adapter commit
`1ae23721592de8708f72d25bc859ec13070dcb68`, not the campaign compiler revision:
the SDPA capture explicitly enables unit-flag synchronization and supplies
native mode 3 to MmadL1/Fixpipe, while that adapter baseline rejects unit-flag
controls. No exact pre-conversion structured dump was saved, so original
attribute presence and folded conditions cannot be fully reconstructed.

Likewise, no HF32 native specialization was observed despite saved inputs
containing misspelled `input_precison = "hf32"` annotations. Bias/I4 and dynamic
physical MmadL1 descriptor rejection cases were not established by the usable
captures. These gaps limit extrapolation; they do not establish unsupported
native behavior or PTODSL correctness.

## Handoff

Use the exact-symbol list to prioritize contract investigation and explicit
implementation/fallback coverage. Keep compiler failures and unknown controls
separate from unsupported features. Follow the existing per-template adapter
policy and required native/bridge numerical baselines before implementation.
No rerun, new capture, source change, or toolchain rebuild is implied by this
documentation. Reconcile future evidence against the same exact symbol and
its control/descriptor contract before updating implementation triage.
