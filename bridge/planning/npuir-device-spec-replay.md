# NPU-IR Replay From Device-Spec IR

Last updated: 2026-08-11

Latest replay result:

```text
bridge/memory/npuir-device-spec-replay-results-2026-08-11.md
```

## Current Inputs

The A5-generated MLIR files currently live beside the Triton source fixtures:

```text
bridge/triton-example/*_kernel.mlir
```

These files are dumps right after:

```text
AppendTargetDeviceSpec (hacc-append-device-spec)
```

They already contain `hacc.target` and `dlti.target_system_spec` attributes for
the A5 target, but the function bodies are still high-level
memref/tensor/linalg-style IR. That makes them useful local inputs for replaying
the NPU-IR compiler pipeline after Python/Triton lowering has already happened
on an A5 server.

## Important Source Finding

In the regbase compile pipeline, `hacc-append-device-spec` is added at the start
of `buildBiShengHIRPipeline`, before canonicalization, HFusion, convert-to-HIVM,
and HIVM optimization.

`AppendDeviceSpec` overwrites an existing device spec and emits a warning. So
feeding these dumps back into full `bishengir-compile` will likely rerun the
append pass and overwrite the same target metadata. That is acceptable for a
first replay experiment, but it is not a perfect suffix-only replay.

For exact suffix replay, we may later need either:

- a custom debug pipeline that starts after `hacc-append-device-spec`; or
- a `bishengir-opt` pass pipeline assembled from the regbase builder pieces.

For now, full `bishengir-compile` replay is the fastest way to see whether the
local build can reproduce the later IR dumps. The target endpoint is
`convert-hivmave-to-ave-intrin`; we do not need final LLVM/C/CCE/binary output
for the current bridge work.

## Replay Wrapper

Use:

```text
bash bridge/tools/replay_npuir_from_device_spec.sh
```

Default assumptions:

- input directory: `bridge/triton-example/`
- input files: `*_kernel.mlir`
- output directory: `bridge/examples/npuir-early-ir/replay/`
- compiler: `$HOME/AscendNPU-IR/build/install/bin/bishengir-compile`
- target: `Ascend910_9589`
- compiler flags include `--mlir-disable-threading`
- compiler flags include `--mlir-print-ir-after-all` by default

Useful overrides:

```text
BISHENGIR_COMPILE=$HOME/AscendNPU-IR/build/install/bin/bishengir-compile
NPU_TARGET=Ascend910_9589
AICORE_LIBDEVICE_BC=$HOME/path/to/libdevice.10.bc
OUTPUT_ROOT=bridge/examples/npuir-early-ir/replay
bash bridge/tools/replay_npuir_from_device_spec.sh
```

`AICORE_LIBDEVICE_BC` is optional for early pass-dump work. It may become needed
if the local run reaches downstream CCE/HIVMC-style linking.

Each case gets:

- `source-after-append-device-spec.mlir`
- `command.txt`
- `compile.log`
- `temps/`
- `compile-exit-code.txt`
- `after-convert-hivmave-to-ave-intrin.mlir`, if the replay reaches the target
  pass
- `after-convert-hivmave-to-ave-intrin-dump-count.txt`, recording how many
  matching target-pass dumps appeared in the compiler log

When the target pass is printed more than once, the wrapper writes the last
matching target-pass dump to `after-convert-hivmave-to-ave-intrin.mlir`. The
full raw compiler output remains in `compile.log`.

The wrapper uses:

```text
--mlir-disable-threading
--mlir-print-ir-after-all
```

by default. The full raw compiler output remains in `compile.log`; the wrapper
still extracts the last matching `convert-hivmave-to-ave-intrin` dump into the
small target file.

To reduce log size and print only the target pass, run with:

```text
PRINT_AFTER_ALL=0 bash bridge/tools/replay_npuir_from_device_spec.sh
```

## Expected Local Failure Mode

Even after NPU-IR builds locally, final binary emission may fail if the external
A5 backend pieces are not installed on this server. One known non-A5-server
failure is:

```text
Cannot find hivmc-a5 under $PATH
```

That does not necessarily block bridge work. If
`after-convert-hivmave-to-ave-intrin.mlir` exists, the current replay objective
succeeded even if the compiler later exits nonzero.

The useful artifacts for us are the MLIR dumps before and around:

- conversion to HIVM;
- memory planning;
- sync insertion/decomposition;
- `convert-hivm-to-std`;
- `convert-hivmave-to-std`;
- `convert-hivmave-to-ave-intrin`.

The default wrapper prints all pass dumps into `compile.log` but extracts only
the target endpoint dump into the small `after-convert-hivmave-to-ave-intrin`
file. Use `PRINT_AFTER_ALL=0` only when a smaller target-pass-only log is
needed.

## First Review Targets

Use `dma_copy_kernel.mlir` first. It already contains:

- `memref.reinterpret_cast` from GM-like arguments;
- local `memref.alloc`;
- `memref.copy` into the local buffer;
- `bufferization.materialize_in_destination` back to output.

That should be the cleanest path to later `hivm.hir.load` and
`hivm.hir.store` evidence.

Use `matmul64_kernel.mlir` second for cube/template evidence. It contains
`linalg.matmul` plus explicit copies into local buffers, so it should exercise
the cube/matmul lowering path.

Use `row_softmax_kernel.mlir` and `rmsnorm_kernel.mlir` after that for reduction
and vector-side lowering evidence.
