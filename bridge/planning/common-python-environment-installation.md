# Common Python Environment for the NPU-IR to PTOAS Flow

This guide creates one Python 3.10 Conda environment for testing the current
checkouts of AscendNPU-IR, PTOAS, and Triton-Ascend. It avoids mixing packages
from `~/.local`, other Conda environments, or older virtual environments.

The expected source branches are:

| Project | Branch |
|---|---|
| AscendNPU-IR | `ptoas-end-to-end-infrastructure` |
| PTOAS | `feat/emit-device-obj` |
| Triton-Ascend | `feature/npuir-ptoas-bridge-support` |

The installation order is:

1. Create the common environment and install shared dependencies.
2. Build AscendNPU-IR and expose its compiler executables.
3. Build and install PTOAS into the environment.
4. Build the LLVM revision required by Triton-Ascend.
5. Build and install Triton-Ascend into the environment.

AscendNPU-IR is primarily consumed through its compiler executables. PTOAS and
Triton-Ascend contain Python ABI-specific native extensions, so they must be
built with the same Python interpreter used to run the Triton kernel.

## 1. Create the Conda Environment

Keep the environment outside all three repositories so deleting or rebuilding
one checkout does not remove it. This procedure assumes that `conda` is already
installed and available in `PATH`; it does not require a system Python 3.10 or
root access because Conda installs the interpreter inside the environment.

```bash
export WORKSPACE=/home/m00967009/Workspace
export BRIDGE_VENV=$HOME/.conda/envs/ascend-bridge-py310

source "$(conda info --base)/etc/profile.d/conda.sh"

# Optional clean restart if an older environment exists at this path:
# conda deactivate
# conda env remove --yes --prefix "$BRIDGE_VENV"

conda create --yes --prefix "$BRIDGE_VENV" python=3.10 pip
conda activate "$BRIDGE_VENV"
export PYTHONNOUSERSITE=1
unset PYTHONPATH
unset PYTHONHOME
unset PYTHONUSERBASE

python -m pip install --upgrade pip setuptools wheel
```

Do not install these packages into Conda's `base` environment and do not use
`pip --user`. The dedicated environment and `PYTHONNOUSERSITE=1` prevent
packages from `~/.local` or another environment from being imported.

Install the build and runtime dependencies shared by the projects:

```bash
python -m pip install \
  "cmake>=3.20,<4" "ninja>=1.11.1" \
  "pybind11==2.13.6" "scikit-build-core>=0.12.2,<2" \
  "nanobind>=2.4" \
  numpy==1.26.4 scipy==1.13.1 pandas \
  attrs==24.2.0 decorator==5.1.1 psutil==6.0.0 \
  pyyaml pyelftools pytest pytest-xdist
```

Install the Torch and TorchNPU versions used by the simulator:

```bash
python -m pip install torch==2.7.1
python -m pip install torch-npu==2.7.1.post8 \
  --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi
```

PyTorch may install an upstream `triton` package. Remove it before installing
Triton-Ascend because both distributions provide the same top-level `triton`
Python package:

```bash
python -m pip uninstall -y triton
```

## 2. Configure CANN

Source CANN before activating the Conda environment for the final time. This
keeps the environment's `python`, `cmake`, and `ptoas` commands first in `PATH`.

```bash
export CANN_ROOT=/home/a84369921/Ascend/cann-9.1.0-beta.3
source "$CANN_ROOT/set_env.sh"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$BRIDGE_VENV"
export PYTHONNOUSERSITE=1
unset PYTHONPATH
unset PYTHONHOME
unset PYTHONUSERBASE

export ASCEND_HOME_PATH="$CANN_ROOT"
export BISHENG_INSTALL_PATH="$CANN_ROOT/tools/bishengir/bin"
export PATH="$BRIDGE_VENV/bin:$CANN_ROOT/tools/bisheng_compiler/bin:$PATH"
```

`BISHENG_INSTALL_PATH` determines which `hivmc-a5` executable the locally built
AscendNPU-IR compiler launches.

## 3. Build AscendNPU-IR

The Triton flow needs the compiler from the current standalone AscendNPU-IR
checkout. It does not require the `ascendnpu-ir` Python wheel.

```bash
export ASCEND_NPU_IR_ROOT="$WORKSPACE/AscendNPU-IR"
cd "$ASCEND_NPU_IR_ROOT"

git branch --show-current
git submodule update --init --recursive

./build-tools/build.sh \
  -r \
  -o "$ASCEND_NPU_IR_ROOT/build" \
  --install-prefix "$ASCEND_NPU_IR_ROOT/build/install" \
  --build-type Release \
  --apply-patches \
  -j 32
```

Expose this compiler before the compiler packaged with CANN:

```bash
export TRITON_NPU_COMPILER_PATH="$ASCEND_NPU_IR_ROOT/build/install/bin"
export PATH="$TRITON_NPU_COMPILER_PATH:$PATH"
hash -r

command -v bishengir-compile
bishengir-compile --version
bishengir-compile --help-hidden | grep emit-ptoas-vmi
```

Installing the optional AscendNPU-IR Python wheel is intentionally omitted. It
would add another bundled compiler location without being used by this flow.

## 4. Prepare PTOAS LLVM

PTOAS requires the VPTO LLVM checkout. Do not use AscendNPU-IR's LLVM build for
PTOAS because the two projects have different LLVM source requirements.

```bash
export PTOAS_LLVM_SOURCE_DIR="$WORKSPACE/llvm-project"
export PTOAS_LLVM_BUILD_DIR="$PTOAS_LLVM_SOURCE_DIR/build-shared"
export PTOAS_ROOT="$WORKSPACE/PTOAS"
export PYTHON_BIN="$BRIDGE_VENV/bin/python"
```

An existing VPTO LLVM build can be reused only if its MLIR Python bindings were
built for Python 3.10:

```bash
find "$PTOAS_LLVM_BUILD_DIR/tools/mlir/python_packages" \
  -name '*cpython-310*.so' | head
```

If that command produces no output, rebuild VPTO LLVM with the common Python:

```bash
cmake -G Ninja -S "$PTOAS_LLVM_SOURCE_DIR/llvm" -B "$PTOAS_LLVM_BUILD_DIR" \
  -DLLVM_ENABLE_PROJECTS="mlir;clang" \
  -DBUILD_SHARED_LIBS=ON \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DPython3_EXECUTABLE="$PYTHON_BIN" \
  -DPython_EXECUTABLE="$PYTHON_BIN" \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_TARGETS_TO_BUILD=host

cmake --build "$PTOAS_LLVM_BUILD_DIR" -j32
```

## 5. Build and Install PTOAS

Remove the old PTOAS build directory so CMake cannot reuse another Python ABI
or another environment's `pybind11` location.

```bash
cd "$PTOAS_ROOT"
git branch --show-current
rm -rf build

PYTHON_BIN="$PYTHON_BIN" \
LLVM_BUILD_DIR="$PTOAS_LLVM_BUILD_DIR" \
PTO_BUILD_DIR="$PTOAS_ROOT/build" \
  ./quick_install.sh
```

The quick installer performs an editable install with build isolation disabled.
This makes the `ptoas` command and Python package belong to the common Conda
environment while retaining an incremental CMake build tree.

Configure PTOAS runtime lookup and verify the extension ABI:

```bash
export LD_LIBRARY_PATH="$PTOAS_LLVM_BUILD_DIR/lib:${LD_LIBRARY_PATH:-}"
export TRITON_PTOAS_PATH="$BRIDGE_VENV/bin/ptoas"

command -v ptoas
ptoas --version
python -c "import ptoas._core; print(ptoas._core.__file__)"
```

The printed `_core` filename must contain `cpython-310`. A previously built
`_core.cpython-311-*.so` cannot be loaded by this environment; rebuild PTOAS
rather than copying or renaming that file.

## 6. Prepare Custom LLVM for Triton-Ascend

Triton-Ascend cannot use an arbitrary LLVM revision. Start from the exact
revision recorded by the current Triton-Ascend checkout and apply its Ascend
LLVM patch. This LLVM installation is separate from the VPTO LLVM build used by
PTOAS in Sections 4 and 5.

Define distinct source, build, and installation directories:

```bash
export TRITON_ASCEND_ROOT="$WORKSPACE/triton-ascend"
export TRITON_LLVM_SOURCE_DIR="$WORKSPACE/llvm-project-triton"
export TRITON_LLVM_BUILD_DIR="$WORKSPACE/build-llvm-triton"
export TRITON_LLVM_INSTALL_DIR="$WORKSPACE/install-llvm-triton"
export TRITON_LLVM_REVISION="$(cat "$TRITON_ASCEND_ROOT/cmake/llvm-hash.txt")"
export TRITON_LLVM_PATCH="$(find "$TRITON_ASCEND_ROOT/third_party/ascend/patch" \
  -maxdepth 1 -type f -name "llvm_patch_${TRITON_LLVM_REVISION:0:7}*.patch" \
  -print -quit)"
test -f "$TRITON_LLVM_PATCH"
```

`LLVM_SYSPATH` must eventually point to `TRITON_LLVM_INSTALL_DIR`. It must not
point to the LLVM source directory or CMake build directory.

Clone LLVM, check out Triton-Ascend's required revision, and apply the Ascend
patch. Skip the clone command if the source checkout already exists. Apply the
patch only once to a clean checkout of that revision.

```bash
git clone https://github.com/llvm/llvm-project.git "$TRITON_LLVM_SOURCE_DIR"
git -C "$TRITON_LLVM_SOURCE_DIR" checkout "$TRITON_LLVM_REVISION"
git -C "$TRITON_LLVM_SOURCE_DIR" apply --check "$TRITON_LLVM_PATCH"
git -C "$TRITON_LLVM_SOURCE_DIR" apply "$TRITON_LLVM_PATCH"
```

Build and install LLVM. The installation must contain the LLVM and MLIR
headers, libraries and CMake package files, LLD, and `FileCheck` expected by
Triton-Ascend.

```bash
cmake -G Ninja \
  -S "$TRITON_LLVM_SOURCE_DIR/llvm" \
  -B "$TRITON_LLVM_BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$TRITON_LLVM_INSTALL_DIR" \
  -DLLVM_ENABLE_PROJECTS="clang;lld;mlir" \
  -DLLVM_TARGETS_TO_BUILD="Native;NVPTX;AMDGPU" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_LLD=ON \
  -DLLVM_BUILD_TOOLS=ON \
  -DLLVM_OPTIMIZED_TABLEGEN=ON \
  -DMLIR_ENABLE_BINDINGS_PYTHON=OFF

cmake --build "$TRITON_LLVM_BUILD_DIR" -j32
cmake --install "$TRITON_LLVM_BUILD_DIR"
```

`FileCheck` is built for LLVM's test infrastructure but is not normally part
of the LLVM install target. Triton packages it with its Python module, so copy
it into the custom installation explicitly:

```bash
install -Dm755 \
  "$TRITON_LLVM_BUILD_DIR/bin/FileCheck" \
  "$TRITON_LLVM_INSTALL_DIR/bin/FileCheck"
```

Verify the prefix before starting the much larger Triton build:

```bash
test -f "$TRITON_LLVM_INSTALL_DIR/include/llvm/Config/llvm-config.h"
test -f "$TRITON_LLVM_INSTALL_DIR/lib/cmake/llvm/LLVMConfig.cmake"
test -f "$TRITON_LLVM_INSTALL_DIR/lib/cmake/mlir/MLIRConfig.cmake"
test -f "$TRITON_LLVM_INSTALL_DIR/lib/cmake/lld/LLDConfig.cmake"
test -x "$TRITON_LLVM_INSTALL_DIR/bin/FileCheck"
```

If the custom LLVM already exists, the build and clone commands can be
omitted. It must still be based on `TRITON_LLVM_REVISION`, include the Ascend
patch (plus any intended local changes), and pass these checks.

## 7. Build and Install Triton-Ascend

Triton-Ascend must be installed last so its `triton` Python package is the only
one visible in the environment. This procedure uses the non-editable
`setup_ascend.py install` mode. The resulting `libtriton.so` is installed under
the Conda environment's `site-packages/triton/_C`; it is not expected to appear
under the checkout's `python/triton/_C` directory.

```bash
export TRITON_ASCEND_ROOT="$WORKSPACE/triton-ascend"
cd "$TRITON_ASCEND_ROOT"

git branch --show-current
git submodule update --init --recursive

# Remove both distributions before reinstalling. They share the same
# top-level "triton" package directory.
python -m pip uninstall -y triton triton-ascend triton_ascend
SITE_PACKAGES="$(python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
find "$SITE_PACKAGES" -maxdepth 1 \
  \( -name 'triton' -o -name 'triton*.dist-info' -o -name 'triton*.egg-info' \) \
  -print

# This environment is dedicated to the bridge. Remove only the leftovers
# displayed by the command above.
rm -rf "$SITE_PACKAGES/triton" \
  "$SITE_PACKAGES"/triton*.dist-info \
  "$SITE_PACKAGES"/triton*.egg-info

rm -rf build
rm -f python/triton/_C/libtriton*.so

export MAX_JOBS=32
export LLVM_SYSPATH="$TRITON_LLVM_INSTALL_DIR"
export PATH="$LLVM_SYSPATH/bin:$PATH"
unset PYTHONPATH
unset PYTHONHOME
unset PYTHONUSERBASE
hash -r

TRITON_BUILD_WITH_CCACHE=false \
TRITON_BUILD_WITH_CLANG_LLD=true \
TRITON_BUILD_PROTON=OFF \
TRITON_WHEEL_NAME=triton-ascend \
TRITON_APPEND_CMAKE_ARGS="-DTRITON_BUILD_UT=OFF" \
  python setup_ascend.py install
```

Setting `LLVM_SYSPATH` makes Triton's setup pass the following locations to
CMake automatically:

- `LLVM_INCLUDE_DIRS=$LLVM_SYSPATH/include`
- `LLVM_LIBRARY_DIR=$LLVM_SYSPATH/lib`
- `LLVM_SYSPATH=$LLVM_SYSPATH`

Removing `build` before this command is important when switching LLVM builds;
otherwise CMake can retain paths to Triton's downloaded LLVM package. Putting
`$LLVM_SYSPATH/bin` first also makes the setup wrapper's default
`TRITON_BUILD_WITH_CLANG_LLD=true` use the custom `clang`, `clang++`, and LLD.

Use `setup_ascend.py`, not plain `pip install .` or `pip install -e .`. The
Ascend wrapper adds the Ascend backend and applies its required source patches.
Do not subsequently install the upstream `triton` distribution: both projects
write the same top-level `triton` package, so upstream Triton would overwrite
parts of this installation.

Test the physical installation from outside the repository with no source-tree
entry in `PYTHONPATH`:

```bash
cd /tmp

python - <<'PY'
import triton
from triton._C import libtriton
from triton._C.libtriton.ascend import ir

print("triton:", triton.__file__)
print("libtriton:", libtriton.__file__)
print("ascend ir:", ir)
PY
```

Both printed file paths must be under `$BRIDGE_VENV/lib/python3.10/site-packages`,
not `$TRITON_ASCEND_ROOT/python/triton`, `~/.local`, or another Conda
environment. Confirm the installed extension directly:

```bash
test -f "$BRIDGE_VENV/lib/python3.10/site-packages/triton/_C/libtriton.so"
```

Confirm that the requested prefix was recorded by CMake:

```bash
grep -E '^(LLVM_INCLUDE_DIRS|LLVM_LIBRARY_DIR|LLVM_SYSPATH):' \
  "$TRITON_ASCEND_ROOT"/build/cmake.*-cpython-3.10/CMakeCache.txt
```

All three values must refer to `TRITON_LLVM_INSTALL_DIR`.

## 8. Persistent Runtime Environment

Run the following in every new shell before compiling or simulating kernels:

```bash
export WORKSPACE=/home/m00967009/Workspace
export BRIDGE_VENV=$HOME/.conda/envs/ascend-bridge-py310
export CANN_ROOT=/home/a84369921/Ascend/cann-9.1.0-beta.3
export ASCEND_NPU_IR_ROOT="$WORKSPACE/AscendNPU-IR"
export PTOAS_ROOT="$WORKSPACE/PTOAS"
export PTOAS_LLVM_BUILD_DIR="$WORKSPACE/llvm-project/build-shared"
export TRITON_LLVM_INSTALL_DIR="$WORKSPACE/install-llvm-triton"

source "$CANN_ROOT/set_env.sh"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$BRIDGE_VENV"

export PYTHONNOUSERSITE=1
unset PYTHONPATH
unset PYTHONHOME
unset PYTHONUSERBASE
export ASCEND_HOME_PATH="$CANN_ROOT"
export BISHENG_INSTALL_PATH="$CANN_ROOT/tools/bishengir/bin"
export TRITON_NPU_COMPILER_PATH="$ASCEND_NPU_IR_ROOT/build/install/bin"
export TRITON_PTOAS_PATH="$BRIDGE_VENV/bin/ptoas"
export LLVM_SYSPATH="$TRITON_LLVM_INSTALL_DIR"
export PATH="$BRIDGE_VENV/bin:$TRITON_NPU_COMPILER_PATH:$LLVM_SYSPATH/bin:$CANN_ROOT/tools/bisheng_compiler/bin:$PATH"
export LD_LIBRARY_PATH="$PTOAS_LLVM_BUILD_DIR/lib:${LD_LIBRARY_PATH:-}"
hash -r
```

`LLVM_SYSPATH` is needed while rebuilding Triton-Ascend. Keeping it in the
runtime environment also makes subsequent reinstallations use the same LLVM.
PTOAS continues to load shared libraries from `PTOAS_LLVM_BUILD_DIR`; these two
LLVM locations must not be exchanged.

## 9. Verify the Unified Environment

```bash
which python python3 cmake ptoas bishengir-compile clang clang++ llvm-config
llvm-config --version

python - <<'PY'
import sys
import torch
import torch_npu
import triton
import ptoas._core

print("python:", sys.executable)
print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
print("triton:", triton.__file__)
print("ptoas core:", ptoas._core.__file__)
PY
```

Expected results:

- `python`, `python3`, `cmake`, and `ptoas` resolve under `$BRIDGE_VENV`.
- `bishengir-compile` resolves under the standalone AscendNPU-IR installation.
- `triton.__file__` resolves under the common environment's Python 3.10
  `site-packages`, not the source checkout, `~/.local`, or an upstream Triton
  installation.
- `ptoas._core` has a `cpython-310` filename.
- `clang`, `clang++`, and `llvm-config` resolve under
  `$TRITON_LLVM_INSTALL_DIR/bin`.

## 10. Run the Bridge Flow

Use the Conda environment's absolute Python path when launching through
`msprof`. This prevents `msprof` from resolving another `python3` executable.

From the Planner repository:

```bash
cd "$WORKSPACE/Planner"

TRITON_ASCEND_COMPILE_FLOW=ptoas \
msprof op simulator \
  --kernel-name=vector_add_kernel \
  --soc-version=Ascend950PR_9599 \
  --output=bridge/testcases/vadd/out/profile-ptoas \
  "$BRIDGE_VENV/bin/python" bridge/testcases/vadd/vector_add.py
```

For the native NPU-IR flow, change the compile-flow value:

```bash
TRITON_ASCEND_COMPILE_FLOW=npuir \
msprof op simulator \
  --kernel-name=vector_add_kernel \
  --soc-version=Ascend950PR_9599 \
  --output=bridge/testcases/vadd/out/profile-npuir \
  "$BRIDGE_VENV/bin/python" bridge/testcases/vadd/vector_add.py
```
