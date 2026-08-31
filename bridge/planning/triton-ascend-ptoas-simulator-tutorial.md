# Triton-Ascend NPU-IR-to-PTOAS Simulator Tutorial

This tutorial shows how to install a modified Triton-Ascend checkout into the
NPU-IR simulator virtual environment and run the vector-add testcase through
both compilation flows:

```text
Native flow: Python -> TTIR -> NPU-IR -> npubin -> existing Triton launcher

PTOAS flow:  Python -> TTIR -> TTAdapter -> NPU-IR PTOAS VMI
             -> PTOAS fat object -> extracted AICore object -> npubin
             -> existing Triton launcher
```

The commands use the paths validated in this workspace. Change the path
definitions in the next section if the repositories or CANN are elsewhere.

## 1. Define The Repository And Tool Paths

Run these definitions in every new shell:

```bash
export WORKSPACE_ROOT=/home/m00967009/Workspace
export PLANNER_ROOT="$WORKSPACE_ROOT/Planner"
export TRITON_ASCEND_ROOT="$WORKSPACE_ROOT/triton-ascend"
export ASCEND_NPU_IR_ROOT="$WORKSPACE_ROOT/AscendNPU-IR"
export PTOAS_ROOT="$WORKSPACE_ROOT/PTOAS"
export CANN_ROOT=/home/a84369921/Ascend/cann-9.1.0-beta.3
export NPU_SIM_VENV=/home/m00967009/.venv/npuir-sim-system
```

These variables keep the later commands readable and make it clear which
checkout supplies each compiler.

The expected built tools are:

```text
$ASCEND_NPU_IR_ROOT/build/bin/bishengir-compile
$PTOAS_ROOT/build/tools/ptoas/ptoas
$CANN_ROOT/tools/bisheng_compiler/bin/ld.lld
/usr/bin/objcopy
```

Verify that they exist before installing Triton-Ascend:

```bash
test -x "$ASCEND_NPU_IR_ROOT/build/bin/bishengir-compile"
test -x "$PTOAS_ROOT/build/tools/ptoas/ptoas"
test -x "$CANN_ROOT/tools/bisheng_compiler/bin/ld.lld"
test -x /usr/bin/objcopy
```

Reason: the PTOAS flow is an orchestration layer. It does not build
AscendNPU-IR, PTOAS, `objcopy`, or the AICore linker itself.

## 2. Prepare The Simulator Virtual Environment

Source CANN first, then activate the simulator virtual environment:

```bash
source "$CANN_ROOT/set_env.sh"
source "$NPU_SIM_VENV/bin/activate"

export ASCEND_HOME_PATH="$CANN_ROOT"
export PYTHONNOUSERSITE=1
unset PYTHONPATH
```

The order matters:

- CANN configures its runtime, simulator, compiler, and library paths.
- Activating the virtual environment afterward makes its `python3`, `pip`,
  CMake, and Ninja take precedence.
- `PYTHONNOUSERSITE=1` prevents packages under
  `~/.local/lib/python3.10/site-packages` from being mixed into the simulator
  environment.
- `PYTHONPATH` is unset so an earlier Triton overlay cannot shadow the package
  installed into the virtual environment.

The working Torch pair in this environment is:

```text
torch      2.7.1+cpu
torch_npu  2.7.1.post8
```

Verify it:

```bash
python3 - <<'PY'
import torch
import torch_npu

print("torch:", torch.__version__, torch.__file__)
print("torch_npu:", torch_npu.__version__, torch_npu.__file__)
PY
```

Reason: Torch-NPU contains C++ headers and libraries used when Triton builds
its runtime helper extension. A Torch-NPU release from a different Torch or
CANN generation can fail with missing ACL types before kernel compilation.

## 3. Install Build Tools Into The Virtual Environment

Install the Python build tools without using `--user`:

```bash
python3 -m pip install \
  "cmake>=3.20,<4" \
  ninja \
  wheel \
  pybind11

hash -r
```

Check the selected programs:

```bash
command -v python3
command -v cmake
command -v ninja
cmake --version
```

They should resolve under `$NPU_SIM_VENV/bin`. In particular, do not use
`~/.local/bin/cmake`: it is a Python wrapper tied to the user-site `cmake`
module, which is intentionally hidden by `PYTHONNOUSERSITE=1`.

Reason: Triton-Ascend's setup script invokes `cmake` by name. Installing CMake
inside the virtual environment gives it a self-consistent executable and
Python package while keeping CMake below the project's version-4 boundary.

## 4. Remove An Older Copied Triton Package

Inspect the currently installed Triton-Ascend distribution:

```bash
python3 -m pip show triton-ascend || true
```

If the virtual environment contains an older copied release such as 3.2.0,
remove it before creating the editable installation:

```bash
python3 -m pip uninstall -y triton-ascend
```

Reason: `setup_ascend.py develop` creates an editable link to the source tree,
but it does not necessarily remove an existing physical `site-packages/triton`
directory. Python finds that copied directory before the editable link, making
the installation appear successful while `import triton` still loads the old
release.

## 5. Install The Modified Triton-Ascend Checkout

Before running the setup script, inspect and preserve local source changes:

```bash
cd "$TRITON_ASCEND_ROOT"
git status --short
```

`setup_ascend.py` reapplies Triton-Ascend's patch files before configuring the
build. It checks out a known set of upstream Triton files and can overwrite
uncommitted edits to those patch-managed files. Commit or otherwise preserve
such edits first. Changes under `third_party/ascend/backend`, including this
PTOAS flow implementation, are outside that reset list.

Configure native build paths:

```bash
cd "$TRITON_ASCEND_ROOT"

export LD_LIBRARY_PATH="/home/m00967009/opt/gcc-15/lib64:$CANN_ROOT/x86_64-linux/lib64:${LD_LIBRARY_PATH:-}"
export CCACHE_DIR="$TRITON_ASCEND_ROOT/build/ccache"
export CCACHE_TEMPDIR="$TRITON_ASCEND_ROOT/build/ccache-tmp"
```

Install the checkout in editable mode:

```bash
TRITON_BUILD_PROTON=OFF \
TRITON_APPEND_CMAKE_ARGS="-DTRITON_BUILD_UT=OFF" \
python3 setup_ascend.py develop --no-deps
```

Reasons for these options:

- `setup_ascend.py` applies the Ascend-specific build configuration and uses
  the Ascend-patched LLVM package. Plain `pip install -e .` bypasses this
  setup and can select an incompatible upstream LLVM.
- `develop` installs an editable link, so changes under
  `$TRITON_ASCEND_ROOT/python` are used directly.
- `--no-deps` preserves the known-working Torch and Torch-NPU pair and avoids
  fetching an upstream Triton package.
- disabling Proton and native unit tests reduces build scope for this runtime
  integration test.
- the GCC library path prevents executables linked with the newer compiler
  runtime from loading the older system `libstdc++.so.6`.
- repository-local ccache paths avoid unwritable global cache directories.

Python-only edits become visible immediately. Rerun the command after changing
C++ bindings, TableGen definitions, or native libraries.

## 6. Verify The Editable Installation

Run this check before invoking `msprof`:

```bash
python3 - <<'PY'
import os
import sys
import torch
import torch_npu
import triton
from importlib.metadata import distribution
from triton.backends import backends

print("python:", sys.executable)
print("torch:", torch.__version__, torch.__file__)
print("torch_npu:", torch_npu.__version__, torch_npu.__file__)
print("triton:", triton.__version__, os.path.realpath(triton.__file__))
print("distribution:", distribution("triton-ascend").version)
print("backends:", sorted(backends))
print("active:", {
    name: backend.driver.is_active()
    for name, backend in backends.items()
})
PY
```

Expected important values:

```text
python:       /home/m00967009/.venv/npuir-sim-system/bin/python3
torch:        2.7.1+cpu
torch_npu:    2.7.1.post8
triton:       3.6.0 /home/m00967009/Workspace/triton-ascend/python/triton/__init__.py
distribution: 3.6.0+git...
active:       {'amd': False, 'ascend': True, 'nvidia': False}
```

Also verify that the installed source exposes the new flow:

```bash
python3 - <<'PY'
from triton.backends.ascend import compiler

assert "compile_flow" in compiler.NPUOptions.__dataclass_fields__
assert hasattr(compiler, "ptoas_vmi_to_npubin")
print("PTOAS compile flow is available")
PY
```

Reason: this catches package shadowing and missing backend entry-point metadata
before starting the relatively slow simulator.

## 7. Select The Rebuilt AscendNPU-IR And PTOAS Tools

Set the runtime tool paths after sourcing CANN:

```bash
export PATH="$ASCEND_NPU_IR_ROOT/build/bin:$PATH"
export TRITON_PTOAS_PATH="$PTOAS_ROOT/build/tools/ptoas/ptoas"
export TRITON_OBJCOPY_PATH=/usr/bin/objcopy
```

The AICore linker is found automatically at:

```text
$ASCEND_HOME_PATH/tools/bisheng_compiler/bin/ld.lld
```

It can be overridden when necessary:

```bash
export TRITON_AICORE_LD_PATH="$CANN_ROOT/tools/bisheng_compiler/bin/ld.lld"
```

Verify every selected tool:

```bash
command -v bishengir-compile
bishengir-compile --help 2>&1 | grep -- --emit-ptoas-vmi
"$TRITON_PTOAS_PATH" --help | head
"$TRITON_OBJCOPY_PATH" --version | head -1
"$TRITON_AICORE_LD_PATH" --version | head -1
```

`command -v bishengir-compile` must print:

```text
/home/m00967009/Workspace/AscendNPU-IR/build/bin/bishengir-compile
```

Reason: CANN's `set_env.sh` places its installed compiler on `PATH`. That
compiler does not contain the local `--emit-ptoas-vmi` change. Prepending the
rebuilt AscendNPU-IR directory after sourcing CANN ensures Triton launches the
new binary. With the current Triton helper, `TRITON_NPU_COMPILER_PATH` is only
a fallback when no `bishengir-compile` is already on `PATH`; it is not a true
override.

## 8. Set Common Simulator Controls

Use these controls for both flows:

```bash
export TRITON_ALWAYS_COMPILE=1
export TRITON_SIMULATOR_CLEAN_EXIT=1
export TRITON_DISABLE_FFTS=1
export TRITON_BISHENGIR_DISABLE_LIB_CALL_NOINLINE=1
unset TRITON_COMPILE_ONLY
```

Reasons:

- `TRITON_ALWAYS_COMPILE=1` prevents an old cache entry from hiding compiler
  changes during development.
- `TRITON_SIMULATOR_CLEAN_EXIT=1` lets the testcase exit after reporting its
  result, avoiding a known CANN 9.1 beta Torch-NPU teardown fault.
- FFTS is disabled to match the validated vector-kernel path.
- disabling lib-call no-inline matches the NPU-IR simulator environment used
  by the comparison tooling.
- `TRITON_COMPILE_ONLY` must be unset. When it is true, the launcher prints
  `skip running kernel`; compilation may succeed, but output comparison fails
  because no kernel executed.

`msprof` rejects group- or world-writable application directories. Prepare the
testcase and use a normal creation mask if needed:

```bash
cd "$PLANNER_ROOT"
chmod go-w bridge/testcases/vadd
chmod go-w bridge/testcases/vadd/vector_add.py
umask 022
```

## 9. Run The Native NPU-IR Baseline

The native flow is the default, but select it explicitly for comparison:

```bash
cd "$PLANNER_ROOT"

TRITON_ASCEND_COMPILE_FLOW=npuir \
TRITON_ASCEND_ARCH=Ascend910_9589 \
msprof op simulator \
  --kernel-name=vector_add_kernel \
  --soc-version=Ascend950PR_9589 \
  --core-id=0 \
  --output=bridge/testcases/vadd/out/profile-npuir \
  python3 bridge/testcases/vadd/vector_add.py
```

Reason: this establishes that the Python host code, Torch tensors, native
NPU-IR compiler path, generated launcher, and simulator are working before
introducing PTOAS.

The testcase must print:

```text
allclose: True
```

## 10. Run The NPU-IR-to-PTOAS Flow

Select the new flow and the A5/9599 target:

```bash
cd "$PLANNER_ROOT"

TRITON_ASCEND_COMPILE_FLOW=ptoas \
TRITON_ASCEND_ARCH=Ascend950PR_9599 \
msprof op simulator \
  --kernel-name=vector_add_kernel \
  --soc-version=Ascend950PR_9599 \
  --core-id=0 \
  --output=bridge/testcases/vadd/out/profile-ptoas \
  python3 bridge/testcases/vadd/vector_add.py
```

This command performs the complete integrated path:

1. Triton lowers the Python kernel to TTIR.
2. Triton-Ascend lowers TTIR through TTAdapter and NPU-IR input stages.
3. The rebuilt `bishengir-compile` receives `--emit-ptoas-vmi` and stops after
   the `ConvertHIVMAVEToPTOASVMI` bridge endpoint.
4. PTOAS lowers the VMI module and emits its host fat object.
5. `objcopy` extracts the `__aicore_rel_binary` section.
6. CANN's AICore `ld.lld` links the extracted relocatable object into the
   `npubin` expected by Triton-Ascend.
7. The existing generated Triton host launcher registers and launches that
   `npubin` in the simulator.
8. `vector_add.py` copies the result back and compares it with Torch.

The testcase must again print:

```text
allclose: True
```

## 11. Enable Compiler Dumps When Diagnosing The New Flow

Use separate cache, dump, and profile directories so native and PTOAS artifacts
cannot be confused:

```bash
export TRITON_DEBUG=1
export TRITON_CACHE_DIR="$PLANNER_ROOT/bridge/testcases/vadd/out/cache-ptoas"
export TRITON_DUMP_DIR="$PLANNER_ROOT/bridge/testcases/vadd/out/dump-ptoas"
export ASCEND_PROCESS_LOG_PATH="$PLANNER_ROOT/bridge/testcases/vadd/out/logs-ptoas"

mkdir -p \
  "$TRITON_CACHE_DIR" \
  "$TRITON_DUMP_DIR" \
  "$ASCEND_PROCESS_LOG_PATH"
chmod 700 \
  "$TRITON_CACHE_DIR" \
  "$TRITON_DUMP_DIR" \
  "$ASCEND_PROCESS_LOG_PATH"
```

In debug mode, the PTOAS flow preserves useful intermediates, including the
PTOAS host fat object and extracted AICore relocatable object. The exact cache
subdirectory is content-hash based.

Reason: `msprof` can finish parsing even after its child Python process fails.
Compiler dumps and the testcase's `allclose` result are stronger evidence than
the final `Op profiling finish` line alone.

## 12. Success Criteria

A run passes only when all of the following are true:

- the selected `bishengir-compile` is the rebuilt AscendNPU-IR binary;
- the expected Triton flow is selected;
- the Python child process has no traceback;
- the log does not contain `skip running kernel`;
- the testcase prints `allclose: True`;
- `msprof` reports the profiling task completed successfully;
- the profile output directory contains data for the intended run.

Warnings that the CANN installation owner differs from the current user are
expected on this shared installation and are not, by themselves, a failure.

## 13. Common Failures

### `Unknown command line argument '--emit-ptoas-vmi'`

Cause: CANN's installed `bishengir-compile-a5` was selected instead of the
rebuilt AscendNPU-IR compiler.

Fix:

```bash
export PATH="$ASCEND_NPU_IR_ROOT/build/bin:$PATH"
command -v bishengir-compile
```

Apply this after sourcing CANN.

### `0 active drivers ([]). There should only be one`

Cause: Triton code was exposed without its `triton.backends` entry-point
metadata, or the wrong package installation is active.

Fix: use the editable installation from this tutorial and keep `PYTHONPATH`
unset.

### `import triton` still reports version 3.2.0

Cause: an old copied `site-packages/triton` directory shadows the editable
source link.

Fix:

```bash
python3 -m pip uninstall -y triton-ascend
cd "$TRITON_ASCEND_ROOT"
python3 setup_ascend.py develop --no-deps
```

### `~/.local/bin/cmake` cannot import `cmake`

Cause: the user-site CMake wrapper is running while user-site Python packages
are disabled.

Fix:

```bash
python3 -m pip install "cmake>=3.20,<4" ninja wheel
hash -r
command -v cmake
```

The result must point inside `$NPU_SIM_VENV/bin`.

### Missing `aclmdlRICondHandle` or related ACL types

Cause: an incompatible Torch-NPU package, such as 2.12.0, is being combined
with this CANN installation.

Fix: use `torch 2.7.1+cpu` and `torch_npu 2.7.1.post8` from the simulator
virtual environment, with `PYTHONNOUSERSITE=1` and no user-site `PYTHONPATH`.

### `msprof` finishes but `allclose` is false

Check the log for:

```text
[INFO]: skip running kernel
```

If present, unset compile-only mode:

```bash
unset TRITON_COMPILE_ONLY
```

`msprof` completing its wrapper and parser is not proof that the kernel ran.

## 14. New-Shell Checklist

After the editable installation has been completed once, the normal rerun
sequence is:

```bash
export WORKSPACE_ROOT=/home/m00967009/Workspace
export PLANNER_ROOT="$WORKSPACE_ROOT/Planner"
export ASCEND_NPU_IR_ROOT="$WORKSPACE_ROOT/AscendNPU-IR"
export PTOAS_ROOT="$WORKSPACE_ROOT/PTOAS"
export CANN_ROOT=/home/a84369921/Ascend/cann-9.1.0-beta.3
export NPU_SIM_VENV=/home/m00967009/.venv/npuir-sim-system

source "$CANN_ROOT/set_env.sh"
source "$NPU_SIM_VENV/bin/activate"

export ASCEND_HOME_PATH="$CANN_ROOT"
export PYTHONNOUSERSITE=1
unset PYTHONPATH

export PATH="$ASCEND_NPU_IR_ROOT/build/bin:$PATH"
export TRITON_PTOAS_PATH="$PTOAS_ROOT/build/tools/ptoas/ptoas"
export TRITON_OBJCOPY_PATH=/usr/bin/objcopy
export TRITON_AICORE_LD_PATH="$CANN_ROOT/tools/bisheng_compiler/bin/ld.lld"

export TRITON_ALWAYS_COMPILE=1
export TRITON_SIMULATOR_CLEAN_EXIT=1
export TRITON_DISABLE_FFTS=1
export TRITON_BISHENGIR_DISABLE_LIB_CALL_NOINLINE=1
unset TRITON_COMPILE_ONLY

cd "$PLANNER_ROOT"
```

Then run either the native command from Section 9 or the PTOAS command from
Section 10.
