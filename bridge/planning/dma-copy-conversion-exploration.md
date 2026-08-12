# DMA Copy Conversion Exploration Plan

Last updated: 2026-08-12

## Active Focus

Current implementation focus is DMA conversion using `dma_copy_kernel` as the
first source-backed example.

Role: current focused exploration. For the broader DMA category strategy, use
`bridge/planning/dma-template-rewrite-plan.md`.

Input fixture:

```text
bridge/triton-example/dma_copy_kernel.mlir
```

Generated local replay artifacts are intentionally ignored by git and live
under:

```text
bridge/examples/npuir-early-ir/replay/dma_copy_kernel/
```

The first objective is not to rewrite code. The first objective is to locate
the conversion sweet spot and record the concrete IR syntax at each major
DMA-load/store transition.

## Core Question

Where should the PTOAS bridge intercept DMA?

Candidate boundaries:

| Candidate boundary | What is visible | Main upside | Main risk |
|---|---|---|---|
| Before conversion to HIVM | high-level `memref` / tensor / linalg-style IR | closest to frontend intent | not NPU-specific enough; memory scope/sync/layout decisions may not exist yet |
| Structured HIVM DMA before memory/sync planning | `hivm.hir.load` / `hivm.hir.store` style ops, if present | clean DMA semantics and address spaces | may be too early to preserve final scheduling/sync choices |
| Structured HIVM DMA after memory/sync planning but before `HIVMToStandard` | structured DMA plus more concrete buffers/sync context | likely best first sweet spot | exact pass boundary must be verified in `dma_copy_kernel` logs |
| Template/library-call level after `HIVMToStandard` | calls such as `load_gm_to_ubuf_1d_float` and `store_ubuf_to_gm_1d_float` | easy to match and close to current backend behavior | semantics are encoded in names/templates; harder to recover full shape/layout/flags |
| Inside NPU-IR templates | actual implemented template instruction sequence | preserves tuned backend behavior | still coupled to CCE-template design; may be too late for PTOAS ownership |

Working hypothesis: the first bridge-analysis/export pass should run after
NPU-specific memory/sync facts are present but before structured DMA is lowered
to template/library calls.

## Trace To Build From `dma_copy_kernel`

During exploration, fill this table with actual pass names, source locations,
and syntax snippets from `compile.log`.

| Step | Pass boundary | Load representation | Store representation | What changed | Keep for mapping? |
|---|---|---|---|---|---|
| 0 | A5 dump after `hacc-append-device-spec` | TBD | TBD | frontend/linalg/memref input state | maybe source intent only |
| 1 | first pass where GM->UB movement appears | TBD | TBD | identify first concrete movement syntax | yes |
| 2 | first pass where `hivm.hir.load` / `hivm.hir.store` appears | TBD | TBD | structured HIVM DMA appears | likely yes |
| 3 | after memory planning / buffer sizing | TBD | TBD | buffers/address spaces/sizes become concrete | likely yes |
| 4 | after sync insertion/decomposition | TBD | TBD | pipe/event sync becomes explicit | likely yes |
| 5 | immediately before `HIVMToStandard` | TBD | TBD | last structured-DMA point | likely sweet spot candidate |
| 6 | immediately after `HIVMToStandard` | TBD | TBD | template/library-call boundary | comparison/fallback |
| 7 | after `convert-hivmave-to-ave-intrin` | `func.call @load_gm_to_ubuf_1d_float` | `func.call @store_ubuf_to_gm_1d_float` | vector side is regbase-intrinsic; DMA is helper call | too late as primary source |

For each major transition, record:

- exact pass name from `IR Dump After ...`;
- exact load/store syntax;
- source and destination address spaces;
- dtype/rank/shape/stride facts;
- padding/atomic/layout flags, if visible;
- sync/control facts around the DMA;
- whether the representation is suitable as a PTOAS bridge input.

## NPU-IR Template Investigation

For `dma_copy_kernel`, first template-level names already visible at the late
endpoint are:

```text
load_gm_to_ubuf_1d_float
store_ubuf_to_gm_1d_float
```

Exploration must find:

| Item | Question |
|---|---|
| Template declaration | Where is the template declared and selected? |
| Template implementation | Where is the CCE/template body implemented? |
| Converter call site | Which pass/function rewrites structured DMA into the call? |
| Template parameters | Which operands encode size, dtype, pad value, rank, stride, event, repeat, and alignment? |
| Implemented instruction sequence | What actual low-level DMA/sync/vector instructions are inside the template? |
| Unsupported cases | Which template variants exist for padding, atomic, alignment, rank, dtype, ND2NZ, and UB/GM direction? |

Source areas to inspect first:

```text
$HOME/AscendNPU-IR/bishengir/lib/Template/include/DMA/
$HOME/AscendNPU-IR/bishengir/lib/Template/lib/DMA/
$HOME/AscendNPU-IR/bishengir/lib/Conversion/HIVMToStandard/
$HOME/AscendNPU-IR/bishengir/lib/Dialect/HIVM/IR/LibraryFunctionOpInterface/
$HOME/AscendNPU-IR/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td
```

## PTOAS Mapping Investigation

There are two mapping tracks. Keep both until evidence rules one out.

### Track A: Convert From Template-Call Level

Map the current template/library calls directly to PTOAS-facing movement:

| NPU-IR template call | Likely PTOAS/PTO target | Notes to verify |
|---|---|---|
| `load_gm_to_ubuf_1d_float` | `TLOAD` into Vec tile or low-level VPTO `mte_gm_ub` | verify how PTOAS encodes GM offset, UB allocation, count, dtype, padding/alignment |
| `store_ubuf_to_gm_1d_float` | `TSTORE` from Vec tile or low-level VPTO `mte_ub_gm` | verify store count, stride, and whether sync is explicit or PTOAS-owned |

This track is fast to prototype but may lose important shape/layout/flag
semantics.

### Track B: Rewrite From Structured DMA / Template Internals

Map either structured `hivm.hir.load` / `hivm.hir.store` or the actual
instructions inside the templates to PTOAS concepts:

| NPU-IR concept | Likely PTOAS/PTO target | Notes to verify |
|---|---|---|
| GM -> UB data movement | `TLOAD` to Vec tile, or VPTO `mte_gm_ub` | first DMA row |
| UB -> GM data movement | `TSTORE` from Vec tile, or VPTO `mte_ub_gm` | first DMA row |
| UB clear/fill helper around DMA | VMI/vector op, PTO fill, or keep as helper depending on syntax | only if required by real `dma_copy_kernel` trace |
| Pipe/event sync around DMA | PTO-ISA `TSYNC`, PTOAS sync ops, or explicit level3-style sync | must not mix with PTOAS auto-sync accidentally |

This track is more likely to preserve semantics and should be preferred if the
structured-DMA boundary has enough information.

PTOAS/PTO source areas to inspect first:

```text
$HOME/PTOAS/PTOAS_Markham/include/PTO/IR/VPTOOps.td
$HOME/PTOAS/PTOAS_Markham/include/PTO/IR/VMIOps.td
$HOME/PTOAS/PTOAS_Markham/lib/
$HOME/pto-isa/docs/isa/TLOAD.md
$HOME/pto-isa/docs/isa/TSTORE.md
$HOME/pto-isa/docs/isa/TMOV.md
$HOME/pto-isa/docs/isa/TSYNC.md
```

## Exploration Commands

Regenerate full logs, if needed:

```bash
cd "$HOME/Planner"
bash bridge/tools/replay_npuir_from_device_spec.sh
```

List pass boundaries in the full log:

```bash
grep -n "IR Dump" bridge/examples/npuir-early-ir/replay/dma_copy_kernel/compile.log
```

Find major DMA syntax transitions:

```bash
grep -n "memref.copy\|hivm.hir.load\|hivm.hir.store\|load_gm_to_ubuf\|store_ubuf_to_gm" \
  bridge/examples/npuir-early-ir/replay/dma_copy_kernel/compile.log
```

Find template definitions and call sites in NPU-IR:

```bash
cd "$HOME/AscendNPU-IR"
rg -n "load_gm_to_ubuf_1d_float|store_ubuf_to_gm_1d_float|load_gm_to_ubuf|store_ubuf_to_gm" \
  bishengir
```

Find PTOAS movement candidates:

```bash
cd "$HOME/PTOAS/PTOAS_Markham"
rg -n "mte_gm_ub|mte_ub_gm|TLOAD|TSTORE|TSYNC|gm.*ub|ub.*gm" \
  include lib test tests docs
```

## Output Of The Exploration

Create or update one result note after exploration:

```text
bridge/memory/dma-copy-conversion-trace.md
```

That note should contain:

- the filled pass-by-pass load/store trace table;
- the selected sweet spot and rejected alternatives;
- exact NPU-IR template implementation locations;
- template-call-level PTOAS mapping;
- instruction-level PTOAS mapping;
- first code patch recommendation in `$HOME/AscendNPU-IR`.

No NPU-IR code should be changed until this trace is filled from the real
`dma_copy_kernel` log and source locations.
