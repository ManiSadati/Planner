# Triton-Ascend 3.2.2 PTOAS Bridge Patch

This is a narrow backport of Melika's Triton-Ascend bridge work to the older
3.2.2 Python backend. It keeps native NPU-IR compilation as the default and
adds an opt-in path:

```text
Triton Python -> TTIR -> TTAdapter MLIR -> PTOAS VMI -> PTOAS device object
```

No Python subprocess, socket, or change inside a Triton kernel is required.

## Patch Variants

- `installed-site-packages.patch` targets a normal installed 3.2.2 package.
  It changes `compiler.py`, `utils.py`, and `get_ascend_devices.py`.
- `a5-file-set.patch` targets the copied A5 files `compile.py` and `utils.py`.
  The copied `__init__.py` does not need a change for this 3.2.2 port.

The variants preserve version-specific behavior. In particular, the local
3.2.2 file's newer native options are not removed when patching the A5 copy.

## Check And Apply

For an installed venv:

```bash
SITE_PACKAGES="$VIRTUAL_ENV/lib/python3.10/site-packages"
bash apply.sh check installed "$SITE_PACKAGES"
bash apply.sh apply installed "$SITE_PACKAGES"
```

For the copied A5 file set:

```bash
bash apply.sh check a5-files "$HOME/ta-3.2.2-dev"
bash apply.sh apply a5-files "$HOME/ta-3.2.2-dev"
```

After checking the A5 patch, install the two changed files in that venv:

```bash
SITE_PACKAGES="$VIRTUAL_ENV/lib/python3.10/site-packages"
cp "$HOME/ta-3.2.2-dev/compile.py" \
  "$SITE_PACKAGES/triton/backends/ascend/compiler.py"
cp "$HOME/ta-3.2.2-dev/utils.py" \
  "$SITE_PACKAGES/triton/backends/ascend/utils.py"
```

Use `revert` instead of `apply` to remove the exact patch.

## Run Native NPU-IR

Native compilation remains the default:

```bash
unset TRITON_ASCEND_COMPILE_FLOW
python3 kernel.py
```

## Run Through PTOAS

Point Triton at the NPU-IR build containing the bridge pass and at the PTOAS
executable. Supplying these explicitly avoids accidentally using an older
`bishengir-compile` bundled inside the Triton package.

```bash
export TRITON_ASCEND_ARCH=Ascend910_9589
export TRITON_ASCEND_COMPILE_FLOW=ptoas
export TRITON_NPU_COMPILER_PATH="$ASCEND_NPU_IR_ROOT/build/install"
export TRITON_PTOAS_PATH="$(command -v ptoas)"
python3 kernel.py
```

On an A5 machine, use the target spelling expected by that Triton environment.
The bridge accepts `Ascend910_95*` and `Ascend950*` targets.

## Print Compile Commands

Command printing is off by default. Enable it for both native and bridge runs:

```bash
export TRITON_PRINT_COMPILE_COMMAND=1
python3 kernel.py
```

The backend prints the final shell-escaped `bishengir-compile` and `ptoas`
commands immediately before executing them. This replaces the unconditional
`cmdlist` prints in the copied A5 compiler file.

## Current Limits

- The PTOAS flow supports A5 SIMD/vector and mixed AIC/AIV kernels.
- Pure SIMT is rejected explicitly.
- Kernels using `sync_block_lock` are rejected until the PTOAS path provides
  matching launcher resource metadata.
- This patch targets the inspected 3.2.2 files. Run `check` before applying it
  to another wheel or development snapshot.

