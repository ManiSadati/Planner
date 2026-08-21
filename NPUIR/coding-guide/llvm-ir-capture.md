# NPU-IR LLVM IR Capture

Last updated: 2026-08-18

## Purpose

This note records how to capture the temporary LLVM IR that appears after
NPU-IR lowers through `hivmc-a5` and immediately before CCE `bisheng` consumes
it.

This is relevant for bridge work because it shows the exact low-level LLVM/HIVM
intrinsic form that CCE sees, including vector intrinsics and template/helper
calls.

## Key Finding

`bishengir-compile` can generate a temporary `kernel.ll` file during final
device compilation. The file is normally deleted after the compile finishes.

On the current CANN 9.1 beta simulator path:

- `--save-linked-ir` asks `hivmc-a5` to preserve or produce linked LLVM IR;
- `hivmc-a5` emits `kernel.ll` in the output directory used by `-o`;
- CCE `bisheng` is then invoked on that file;
- this CANN beta currently rejects the `-emit-llvm` flag in that path, so the
  command exits nonzero;
- a watcher can still copy `kernel.ll` before cleanup.

For normal successful simulator runs, do not pass `--save-linked-ir`. Use this
only when the goal is to inspect LLVM IR.

## Capture Script

Use the checked-in helper:

```bash
NPUIR/tools/capture_pre_cce_llvm_ir.sh \
  --input "$IN" \
  --out-dir "$OUT" \
  --cann-root "$CANN_ROOT"
```

This does not run the full NPU-IR simulator. It replays a TTAdapter MLIR through
`bishengir-compile --save-linked-ir`, watches the `-o` output directory for
temporary `kernel*.ll` / `*mix*.ll` files, copies them into
`$OUT/ll-dumps`, and stops the compiler after the first successful capture by
default.

Pass `--wait-for-compile` only when you intentionally want to let
`bishengir-compile` continue after capture.

## Manual Vector Add Flow

First generate or locate the Triton `kernel.ttadapter.mlir` dump. The simulator
runner writes it under its configured `TRITON_DUMP_DIR`.

Example input shape:

```bash
IN="$HOME/tmp/npuir-sim-vector-add-wrapper/dump/<hash>/kernel.ttadapter.mlir"
OUT="$HOME/tmp/npuir-vector-add-llvm-ir-capture"

mkdir -p "$OUT/ll-dumps"
chmod 700 "$OUT" "$OUT/ll-dumps"
```

Then run:

```bash
NPUIR/tools/capture_pre_cce_llvm_ir.sh \
  --input "$IN" \
  --out-dir "$OUT" \
  --cann-root /path/to/cann-9.1.0-beta.3
```

Then inspect the copied LLVM IR:

```bash
ls -lh "$OUT/ll-dumps"
rg -n "define|llvm\\.hivm|load_gm_to_ubuf|store_ubuf_to_gm|vadd" \
  "$OUT/ll-dumps"/*
```

## Verified Vector Add Evidence

The captured `kernel.ll` for `vector_add_kernel` contains:

```text
@llvm.hivm.vldsx1.v64f32
@llvm.hivm.vadd.s.x.v64f32
@llvm.hivm.vstsx1.v64f32
@llvm.hivm.SET.FLAG.IMM
@llvm.hivm.WAIT.FLAG.IMM
@llvm.hivm.BARRIER
load_gm_to_ubuf_1d_float
store_ubuf_to_gm_1d_float
```

This confirms that the LLVM IR boundary still exposes both:

- vector-side LLVM/HIVM intrinsics for UB vector compute;
- DMA/template helper calls for GM/UB movement.

For bridge planning, this is a late observation point. It is useful for
checking what CCE sees, but it is not the preferred rewrite point for DMA. The
preferred DMA rewrite point remains the structured HIVM stage after
`hivm-mark-disable-load` and before `convert-hivm-to-std`.

## Caveats

- The capture is timing-sensitive. The helper starts the watcher before
  invoking `bishengir-compile`.
- Watch the `-o` output directory, not only `$TMPDIR`.
- `--save-temps=<dir>` is not reliable in this CANN 9.1 beta path because
  `bishengir-compile` forwards it into `hivmc-a5`, and this `hivmc-a5` rejects
  that option.
- `--save-linked-ir` may intentionally make the compile fail on this CANN beta.
  Treat a captured `.ll` as success for IR inspection even if the command exits
  nonzero.
