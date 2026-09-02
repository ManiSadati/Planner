# PTOAS Direct Device Object Emission

## Summary

This change adds a PTOAS CLI mode for emitting a directly loadable AICore device object from the VPTO backend:

```bash
ptoas --pto-backend=vpto --emit-device-object input.pto -o kernel.npubin
```

The output is an AICore ELF executable (`ET_EXEC`, machine `0x1029`) instead of the existing host-side x86-64 fat object that embeds `__aicore_rel_binary`.

## Motivation

Triton-Ascend currently needs a workaround for the PTOAS path:

1. Run PTOAS to produce a host fat object.
2. Extract the embedded `__aicore_rel_binary` section.
3. Run `ld.lld` again in Triton-Ascend to produce the final loadable `npubin`.

That makes Triton-Ascend responsible for object-file surgery and final device linking. The cleaner contract is for PTOAS to emit the final device object directly, so Triton-Ascend can pass the returned bytes to the existing runtime loader without extra linking.

## Changed Files

- `PTOAS/tools/ptoas/ptoas.cpp`
  - Adds the new CLI flag `--emit-device-object`.

- `PTOAS/tools/ptoas/ptoas.h`
  - Exports `mlir::pto::emitDeviceObject` so the driver can inspect it.

- `PTOAS/tools/ptoas/driver.cpp`
  - Rejects `--emit-device-object` unless the effective backend is VPTO.
  - Rejects combining `--emit-device-object` with IR-output/debug-output modes.
  - Rejects mixed-backend module mode for `--emit-device-object`.
  - Dispatches single-backend VPTO object compilation to the new direct device-object emission path.

- `PTOAS/tools/ptoas/ObjectEmission.h`
  - Declares `emitDeviceObjectLLVM`.

- `PTOAS/tools/ptoas/ObjectEmission.cpp`
  - Adds `emitDeviceObjectLLVM`.
  - Reuses the existing cube/vector LLVM-to-object compilation path.
  - Reuses existing VF_SIMT size verification/patching for vector objects.
  - Adds a final device-object link step using:

```bash
ld.lld -m aicorelinux -Ttext 0 <device objects> -o <output> --allow-multiple-definition
```

## Behavior

Existing default behavior is unchanged:

```bash
ptoas --pto-backend=vpto input.pto -o kernel.fatobj.o
```

still emits a host x86-64 relocatable fat object containing `__aicore_rel_binary`.

New behavior:

```bash
ASCEND_HOME_PATH=/home/a84369921/Ascend/cann-9.1.0-beta.3 \
PATH=/home/a84369921/Ascend/cann-9.1.0-beta.3/bin:$PATH \
build/tools/ptoas/ptoas \
  --pto-arch=a5 \
  --pto-backend=vpto \
  --emit-device-object \
  test/vpto/cases/micro-op/binary-vector/vadd/kernel.pto \
  -o /tmp/ptoas-vadd-device.o
```

emits a final AICore executable object suitable for Triton-Ascend's `npubin` loading path.

## Validation

Built:

```bash
cmake --build build --target PTOASCompilerImplementation -j16
cmake --build build --target PTOASPythonPackage -j16
```

Checked CLI registration:

```bash
build/tools/ptoas/ptoas --help | rg "emit-device-object"
```

Checked invalid EmitC use:

```bash
build/tools/ptoas/ptoas \
  --pto-backend=emitc \
  --emit-device-object \
  test/vpto/cases/micro-op/binary-vector/vadd/kernel.pto \
  -o /tmp/ptoas-device-object-invalid.o
```

Expected error:

```text
Error: --emit-device-object requires the VPTO backend.
```

Checked invalid IR-output combination:

```bash
build/tools/ptoas/ptoas \
  --pto-backend=vpto \
  --emit-device-object \
  --emit-vpto \
  test/vpto/cases/micro-op/binary-vector/vadd/kernel.pto \
  -o /tmp/ptoas-invalid-combo.o
```

Expected error:

```text
Error: --emit-device-object cannot be combined with debug IR output flags.
```

Checked emitted object shape:

```bash
file /tmp/ptoas-vadd-device.o
readelf -h /tmp/ptoas-vadd-device.o
readelf -l /tmp/ptoas-vadd-device.o
```

Observed:

```text
ELF 64-bit LSB executable, unknown arch 0x1029
Type: EXEC
Machine: 0x1029
Number of program headers: 2
```

Checked default VPTO fatobj path remains unchanged:

```bash
ASCEND_HOME_PATH=/home/a84369921/Ascend/cann-9.1.0-beta.3 \
PATH=/home/a84369921/Ascend/cann-9.1.0-beta.3/bin:$PATH \
build/tools/ptoas/ptoas \
  --pto-arch=a5 \
  --pto-backend=vpto \
  test/vpto/cases/micro-op/binary-vector/vadd/kernel.pto \
  -o /tmp/ptoas-vadd-fatobj.o
```

Observed:

```text
ELF 64-bit LSB relocatable, x86-64
section: __aicore_rel_binary
```

## Notes For Follow-Up

The next Triton-Ascend change should replace the current extraction/linking workaround with a direct PTOAS invocation using `--emit-device-object`, then return the produced bytes as the `npubin` artifact.
