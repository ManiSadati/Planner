# Common Python Environment for the NPU-IR to PTOAS Flow

This guide creates one Python 3.11 Conda environment for testing the current
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
4. Build and install Triton-Ascend into the environment.

AscendNPU-IR is primarily consumed through its compiler executables. PTOAS and
Triton-Ascend contain Python ABI-specific native extensions, so they must be
built with the same Python interpreter used to run the Triton kernel.

## 1. Create the Conda Environment

Keep the environment outside all three repositories so deleting or rebuilding
one checkout does not remove it. This procedure assumes that `conda` is already
installed and available in `PATH`; it does not require a system Python 3.11 or
root access because Conda installs the interpreter inside the environment.

```bash
export WORKSPACE=/home/m00967009/Workspace
export BRIDGE_VENV=$HOME/.conda/envs/ascend-bridge-py311

source "$(conda info --base)/etc/profile.d/conda.sh"
conda create --yes --prefix "$BRIDGE_VENV" python=3.11 pip
conda activate "$BRIDGE_VENV"
export PYTHONNOUSERSITE=1

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
export LLVM_SOURCE_DIR="$WORKSPACE/llvm-project"
export LLVM_BUILD_DIR="$LLVM_SOURCE_DIR/build-shared"
export PTOAS_ROOT="$WORKSPACE/PTOAS"
export PYTHON_BIN="$BRIDGE_VENV/bin/python"
```

An existing VPTO LLVM build can be reused only if its MLIR Python bindings were
built for Python 3.11:

```bash
find "$LLVM_BUILD_DIR/tools/mlir/python_packages" \
  -name '*cpython-311*.so' | head
```

If that command produces no output, rebuild VPTO LLVM with the common Python:

```bash
cmake -G Ninja -S "$LLVM_SOURCE_DIR/llvm" -B "$LLVM_BUILD_DIR" \
  -DLLVM_ENABLE_PROJECTS="mlir;clang" \
  -DBUILD_SHARED_LIBS=ON \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DPython3_EXECUTABLE="$PYTHON_BIN" \
  -DPython_EXECUTABLE="$PYTHON_BIN" \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_TARGETS_TO_BUILD=host

cmake --build "$LLVM_BUILD_DIR" -j32
```

## 5. Build and Install PTOAS

Remove the old PTOAS build directory so CMake cannot reuse another Python ABI
or another environment's `pybind11` location.

```bash
cd "$PTOAS_ROOT"
git branch --show-current
rm -rf build

PYTHON_BIN="$PYTHON_BIN" \
LLVM_BUILD_DIR="$LLVM_BUILD_DIR" \
PTO_BUILD_DIR="$PTOAS_ROOT/build" \
  ./quick_install.sh
```

The quick installer performs an editable install with build isolation disabled.
This makes the `ptoas` command and Python package belong to the common Conda
environment while retaining an incremental CMake build tree.

Configure PTOAS runtime lookup and verify the extension ABI:

```bash
export LD_LIBRARY_PATH="$LLVM_BUILD_DIR/lib:${LD_LIBRARY_PATH:-}"
export TRITON_PTOAS_PATH="$BRIDGE_VENV/bin/ptoas"

command -v ptoas
ptoas --version
python -c "import ptoas._core; print(ptoas._core.__file__)"
```

The printed `_core` filename must contain `cpython-311`.

## 6. Build and Install Triton-Ascend

Triton-Ascend must be installed last so its `triton` Python package is the only
one visible in the environment.

```bash
export TRITON_ASCEND_ROOT="$WORKSPACE/triton-ascend"
cd "$TRITON_ASCEND_ROOT"

git branch --show-current
git submodule update --init --recursive
rm -rf build

export MAX_JOBS=32
hash -r

python setup_ascend.py develop --no-deps
```

Use `setup_ascend.py`, not plain `pip install -e .`. The Ascend setup wrapper
registers the Ascend backend. `--no-deps` prevents its dependency metadata from
installing a second upstream Triton distribution over the local checkout.

## 7. Persistent Runtime Environment

Run the following in every new shell before compiling or simulating kernels:

```bash
export WORKSPACE=/home/m00967009/Workspace
export BRIDGE_VENV=$HOME/.conda/envs/ascend-bridge-py311
export CANN_ROOT=/home/a84369921/Ascend/cann-9.1.0-beta.3
export ASCEND_NPU_IR_ROOT="$WORKSPACE/AscendNPU-IR"
export PTOAS_ROOT="$WORKSPACE/PTOAS"
export LLVM_BUILD_DIR="$WORKSPACE/llvm-project/build-shared"

source "$CANN_ROOT/set_env.sh"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$BRIDGE_VENV"

export PYTHONNOUSERSITE=1
export ASCEND_HOME_PATH="$CANN_ROOT"
export BISHENG_INSTALL_PATH="$CANN_ROOT/tools/bishengir/bin"
export TRITON_NPU_COMPILER_PATH="$ASCEND_NPU_IR_ROOT/build/install/bin"
export TRITON_PTOAS_PATH="$BRIDGE_VENV/bin/ptoas"
export PATH="$BRIDGE_VENV/bin:$TRITON_NPU_COMPILER_PATH:$CANN_ROOT/tools/bisheng_compiler/bin:$PATH"
export LD_LIBRARY_PATH="$LLVM_BUILD_DIR/lib:${LD_LIBRARY_PATH:-}"
hash -r
```

## 8. Verify the Unified Environment

```bash
which python python3 cmake ptoas bishengir-compile

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
- `triton.__file__` resolves to the current Triton-Ascend checkout, not
  `~/.local` and not an upstream Triton installation.
- `ptoas._core` has a `cpython-311` filename.

## 9. Run the Bridge Flow

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
