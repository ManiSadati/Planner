# PTOAS Pipeline And Validation Guide

Last updated: 2026-08-07

Purpose: quick working guide for using PTOAS as context for the NPU-IR bridge. This does not replace PTOAS upstream docs.

## Local Checkout Warning

Local path:

```text
$HOME/PTOAS/PTOAS_Markham
```

This checkout is useful for source inspection, local builds, and experiments, but it is not authoritative. Its `origin` is a personal fork and the current branch `mani/fix_ptodsl` may be far behind active upstream/fork design.

Always check upstream/fork activity before making design claims.

## Useful Inspection Commands

From the PTOAS repo:

```bash
git branch --show-current
git remote -v
git status --short
rg -n "ExpandTileOp|VMIToVPTO|PTOInsertSync|PlanMemory" tools lib include docs ptodsl
```

Useful source entry points:

```text
tools/ptoas/ptoas.cpp
include/PTO/Transforms/Passes.td
lib/PTO/Transforms/ExpandTileOp.cpp
lib/PTO/Transforms/VMIToVPTO.cpp
lib/PTO/Transforms/VMILayoutAssignment.cpp
lib/PTO/Transforms/PTOPlanMemory.cpp
lib/PTO/Transforms/PTOPlanMemoryModern.cpp
lib/PTO/Transforms/InsertSync/
```

## Useful PTOAS Commands

Show CLI version:

```bash
ptoas --version
```

Run a simple parse/print:

```bash
ptoas test/lit/pto/empty_func.pto
```

Run auto sync on a PTO file:

```bash
ptoas test/lit/pto/empty_func.pto --enable-insert-sync -o outputfile.cpp
```

Emit VPTO IR for a VMI pipeline case:

```bash
ptoas test/lit/vmi_new/vmi_ptoas_cli_pipeline.pto --pto-arch=a5 --pto-backend=vpto --emit-vpto -o -
```

Dump IR after tile expansion when debugging PTODSL TileLib:

```bash
ptoas --pto-arch=a5 --pto-backend=vpto --emit-vpto \
  --tile-lib-backend=ptodsl \
  --mlir-print-ir-after=pto-expand-tile-op \
  input.pto -o -
```

Use level3 when addresses/sync are intentionally authored by the caller:

```bash
ptoas input.pto --pto-arch=a5 --pto-backend=vpto --pto-level=level3 --emit-vpto -o -
```

## Test Surfaces

Local PTOAS can be built/run on this server. Useful test surfaces:

```text
test/lit/
test/lit/vmi_new/
test/vpto/
test/dsl-st/
ptodsl/tests/
test/tilelang_st/
```

Build-tree test target when the local build is available:

```bash
ninja -C "$PTO_SOURCE_DIR/build" check-pto
```

PTODSL Python tests:

```bash
python3 ptodsl/tests/test_jit_compile.py
python3 ptodsl/tests/test_docs_as_test.py
```

## Validation Reality

Compile-only checks can run locally when the PTOAS build environment is configured.

On-board validation needs NPU/A5 resources. For this project, Codex should propose exact commands or test cases, then the human runs them on the A5 server and returns logs.

## Bridge-Specific Debugging Notes

- Use `--emit-vpto` to inspect whether VMI residuals are gone after `VMIToVPTO`.
- Use `--mlir-print-ir-after=pto-expand-tile-op` to see the boundary between TileOps and VPTO-facing helper IR.
- When testing sync ownership, explicitly record `--pto-level`, `--enable-insert-sync`, `--enable-bufid-sync`, or `--enable-graph-sync-solver`.
- Do not silently mix NPU-IR-authored sync with PTOAS auto-sync. Decide and document ownership per mapping row.
- For pointer/memory mappings, distinguish `pto.castptr` pointer conversion from tile-native `pto.alloc_tile addr` planning.
