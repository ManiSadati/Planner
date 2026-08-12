# DMA Template Mapping Memory

Last updated: 2026-08-12

## Current Decision

DMA/template rewriting should start from structured HIVM DMA ops after
`hivm-mark-disable-load` and before `convert-hivm-to-std`, not from the final
CCE template/library-call layer and not from the AVE-to-VMI prototype.

Reason: before `convert-hivm-to-std`, NPU-IR still exposes source/destination
memory spaces, dtype, rank, strides, padding, layout conversion, atomic mode,
and sync context. After the conversion, much of that becomes encoded in
library-call names and CCE template behavior.

The `dma_copy_kernel` trace is recorded in
`bridge/memory/dma-copy-conversion-trace.md`.

For the first code patch, prefer low-level VPTO MTE as the concrete PTOAS
target:

- `hivm.hir.load` GM->UB -> `pto.mte_gm_ub`
- `hivm.hir.store` UB->GM -> `pto.mte_ub_gm`
- `hivm.hir.set_flag` / `wait_flag` -> PTOAS explicit sync ops

Keep tile-level `TLOAD` / `TSTORE` as the longer-term cleaner interface, but do
not use it as the first proof of concept unless the tile/view ownership model is
resolved.

## First Practical Target

Start with:

- `hivm.hir.load` GM->UB, contiguous 1D first;
- `hivm.hir.store` UB->GM, contiguous, no atomic.

Emit a strict mapping/export record first, preserving surrounding explicit sync
and rejection reasons. Then generate a minimal PTOAS/VPTO test from it.

Padding nuance from `dma_copy_kernel`: the observed load carries
`pad_mode = <PadValue>` and zero pad value, but left padding is zero and the
tail-fill path is separate. The first pass should record this rather than
rejecting every `PadValue` case blindly. Nonzero padding remains out of scope.

## Working DMA Categories

| Category | Likely PTOAS/PTO target | Risk |
|---|---|---|
| GM->UB normal load | `TLOAD` to Vec tile or `pto.mte_gm_ub` | Low |
| UB->GM normal store | `TSTORE` from Vec tile or `pto.mte_ub_gm` | Low |
| UB->GM atomic store | `TSTORE` atomic or VPTO store atomic sequence | Medium |
| UB->UB copy | `TMOV` Vec->Vec or `pto.mte_ub_ub` | Medium |
| UB->L1 copy | `TMOV` Vec->Mat or `pto.mte_ub_l1` | Medium |
| GM->L1 normal load | `TLOAD` to Mat tile or `pto.mte_gm_l1` | Medium |
| GM->L1 ND2NZ | `TLOAD` Mat ND->NZ or `copy_gm_to_cbuf_multi_nd2nz` | Medium/high |
| L1->UB layout conversion | `TMOV` Mat->Vec or `pto.mte_l1_ub` | Medium/high |
| L1->GM NZ2ND | likely two-step Mat->Vec then Vec->GM unless direct support exists | High |
| L0C fixpipe | `TSTORE`, `TSTORE_FP`, `TMOV` Acc->*, or `mte_l0c_*` | High |
| MX scale load | PTOAS/PTO-ISA MX scale load path, not confirmed | High |
| indexed/gather/scatter memory | VMI gather/scatter or PTO-ISA `MGATHER`/`MSCATTER` | Separate design |

## Review Decisions Still Open

- Exact dry-run output format: JSON, MLIR comments, or bridge dialect/module.
- Exact pass-pipeline insertion point for the new pass in the current
  AscendNPU-IR driver.
- Exact PTOAS lit-test shape for dynamic event ids.
- Whether the second DMA row should be GM->L1 ND2NZ or UB->UB / UB->L1.

## Do Not Forget

- VMI is the right target family for vector semantics, masks, and logical vector
  loads/stores. It is not the primary target for ND2NZ, L1/L0 staging, cube
  operand movement, or fixpipe.
- CCE templates are still useful as reference evidence for legality, alignment,
  padding, and scalar/vector fallback behavior.
- L1->GM NZ2ND and L0C fixpipe need extra caution. Do not assume a clean
  one-op PTOAS mapping without source-backed verification.
- Do not write machine-specific absolute home paths into docs; use `$HOME` or
  repo-relative paths.
