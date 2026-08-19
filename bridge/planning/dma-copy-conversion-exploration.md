# DMA Copy Conversion Exploration Plan

Last updated: 2026-08-14

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

The initial exploration objective is complete. The source-backed trace is
recorded in `bridge/memory/dma-copy-conversion-trace.md`.

Current result: AscendNPU-IR now has a standalone
`convert-hivm-templates-to-pto` pass that converts the real `dma_copy_kernel`
load and store after
`hivm-mark-disable-load`. The generated local artifact is
`bridge/examples/npuir-early-ir/replay/dma_copy_kernel/after-convert-hivm-templates-to-pto.mlir`.

Current next step: choose a guarded compiler-pipeline entry for this conversion
and map the surrounding explicit sync before attempting end-to-end PTO lowering.

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

Decision: the first bridge-analysis/export pass should run after
`hivm-mark-disable-load` and before `convert-hivm-to-std`.

Reason: at that boundary, `hivm.hir.load` / `hivm.hir.store` still carry
structured DMA operands; `gm` / `ub` address spaces, UB pointer casts,
double-buffering, dynamic valid length, and explicit MTE/V/MTE sync are visible.
After `convert-hivm-to-std`, this becomes helper calls such as
`load_gm_to_ubuf_1d_float` and `store_ubuf_to_gm_1d_float`.

## Trace To Build From `dma_copy_kernel`

The detailed table is filled in `bridge/memory/dma-copy-conversion-trace.md`.
Summary:

| Step | Pass boundary | Load representation | Store representation | Mapping decision |
|---|---|---|---|---|
| 0 | `AppendTargetDeviceSpec` | `memref.copy` | `bufferization.materialize_in_destination` | source intent only |
| 1 | `ConvertToHIVMOp` | first `hivm.hir.load` | first `hivm.hir.store` | structured but too early |
| 2 | `PlanMemoryRegBase` | `hivm.hir.load` GM->UB | `hivm.hir.store` UB->GM | usable backup boundary |
| 3 | `GraphSyncSolver` / `MarkDisableLoad` | structured DMA plus explicit sync | structured DMA plus explicit sync | selected boundary |
| 4 | `ConvertHIVMToStandard` | `func.call @load_gm_to_ubuf_1d_float` | `func.call @store_ubuf_to_gm_1d_float` | comparison/fallback only |
| 5 | `convert-hivmave-to-ave-intrin` | helper call remains | helper call remains | too late |

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

Current source-backed answer:

- `HIVMDMAOps.td` defines `hivm.hir.load` as `PIPE_MTE2` and
  `hivm.hir.store` as `PIPE_MTE3`.
- `LibraryFunctionOpInterfaceImpl.cpp` constructs copy-like helper names from
  op name, source/destination address spaces, rank, and dtype.
- `Conversion/HIVMToStandard/regbase/HIVMToStandard.cpp` rewrites
  `hivm::LoadOp` and `hivm::StoreOp` with
  `CopyOpToLibraryCallPattern`.
- Current regbase vector templates live in
  `bishengir/lib/Template/lib/RegBase/Vector/Copy1D.cpp` and
  `bishengir/lib/Template/include/RegBase/DMAUtils.h`.
- The C310/A5 contiguous path lowers to `copy_gm_to_ubuf_align_v2` and
  `copy_ubuf_to_gm_align_v2`.

## PTOAS Mapping Investigation

There are two mapping tracks. Keep both in the plan, but use low-level VPTO MTE
ops for the first code patch because this is the most concrete PTOAS surface
for the observed NPU-IR helper behavior.

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

Decision for first slice: structured DMA has enough information. Use
`hivm.hir.load` / `hivm.hir.store` as source, preserve NPU-IR sync, and map the
simple contiguous row to `pto.mte_gm_ub` / `pto.mte_ub_gm`.

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

Result note:

```text
bridge/memory/dma-copy-conversion-trace.md
```

That note contains:

- the filled pass-by-pass load/store trace table;
- the selected sweet spot and rejected alternatives;
- exact NPU-IR template implementation locations;
- template-call-level PTOAS mapping;
- instruction-level PTOAS mapping;
- first code patch recommendation in `$HOME/AscendNPU-IR`.

NPU-IR code can now start, but the first patch should be dry-run/export only.
Do not replace the production CCE-template lowering path until the generated
mapping record and one PTOAS-facing test are reviewed.
