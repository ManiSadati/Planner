# NPU-IR A5 Simulator Workflow

Last updated: 2026-08-19

## Status

The Codex-accessible server can compile and execute selected Triton/NPU-IR
kernels with the CANN operator simulator. This is useful for fast functional
checks, IR capture, and rough tick/cycle comparison, but it does not replace
validation or performance measurement on a real A5 machine.

Verified small vector-add result:

```text
soc selection: Ascend950PR_9589
core: core0.veccore0
max error: 0.0
allclose: True
simulator running time: 0.97 us
msprof status: Profiling running finished. All task success.
```

Verified large vector-add result:

```text
kernel: vector_add_large_kernel
shape: (1000, 2000)
numel: 2000000
block_cols: 2048
max error: 0.0
allclose: True
terminal tick counter: 54710
core0.veccore0 summed cycles: 196042
core0.veccore1 summed cycles: 211889
```

The simulator currently maps that SOC selection to its installed
`Ascend950pr_9599_*` model configuration files. This is CANN behavior, not a
Planner alias.

## Version Set

The working Python environment is `$HOME/.venv/npuir-sim-system` and uses:

```text
system Python: 3.10
PyTorch: 2.7.1+cpu
TorchNPU: 2.7.1.post8
Triton Ascend: 3.2.0
CANN: 9.1 beta 3
NPU-IR: 1.2.0 from local Wilson-fork-based development branch
```

TorchNPU `2.7.1.post8` is the CANN 9.1-compatible release in the
[official TorchNPU compatibility matrix](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.md).
The unqualified `2.7.1` package targets an older CANN release and crashed during
allocation in this environment.

## Set Up Python

The setup is reproducible with:

```bash
cd "$HOME/Planner"
bash bridge/tools/setup_npuir_simulator_env.sh
```

The setup applies a narrow compatibility patch to Triton Ascend 3.2.0:

- remove a runtime enum that is absent from the CANN 9.1 beta headers;
- allow the local NPU-IR compiler to disable `hacc.noinline` emission;
- propagate `TRITON_DISABLE_FFTS` through the compiler path used for A5.

The patch lives beside the setup script and will need review when Triton Ascend
or CANN is upgraded.

## Install NPU-IR Templates

Simulator compilation needs the NPU-IR template bitcode, not only
`bishengir-compile` and `bishengir-opt`. For an existing build cache, `-t` alone
does not rerun CMake. Reconfigure the existing build explicitly:

```bash
cd "$HOME/AscendNPU-IR"

cmake \
  -S third-party/llvm-project/llvm \
  -B build \
  -DBISHENGIR_BUILD_TEMPLATE=ON \
  -DBISHENG_COMPILER_PATH="$CANN_ROOT/tools/bisheng_compiler/bin"

cmake --build build \
  --target install-bishengir-publish-products \
  -j 96
```

Verify these A5 files exist:

```text
$HOME/AscendNPU-IR/build/install/lib/meta_op.aic.c310.bc
$HOME/AscendNPU-IR/build/install/lib/meta_op.aiv.c310.bc
```

For a fresh build directory, pass both `-t` and
`--bisheng-compiler="$CANN_ROOT/tools/bisheng_compiler/bin"` to
`build-tools/build.sh`.

## Run Vector Add

Point `CANN_ROOT` at the shared or private CANN 9.1 installation:

```bash
cd "$HOME/Planner"
CANN_ROOT=/path/to/cann-9.1.0-beta.3 \
  bash bridge/tools/run_vector_add_simulator.sh
```

Each run writes its Triton cache, IR dumps, CANN logs, and `msprof` profile under
`$HOME/tmp/npuir-simulator/` by default.

For the large vector-add case, use:

```bash
cd "$HOME/Planner"
source bridge/tools/source_vector_add_large_simulator_env.sh

msprof op simulator \
  --kernel-name=vector_add_large_kernel \
  --soc-version=Ascend950PR_9589 \
  --core-id=0 \
  --output="$OUT/profile" \
  python3 "$HOME/Planner/bridge/triton-example/vector_add_large.py"
```

Or run the wrapper:

```bash
cd "$HOME/Planner"
bridge/tools/run_vector_add_large_simulator.sh
```

For the later LLVM/HIVM IR handed from `hivmc-a5` into CCE `bisheng`, see
`bridge/memory/npuir-llvm-ir-capture.md`. That file is normally temporary and
needs to be copied while the compiler is running.

## Compatibility Details

`TRITON_DISABLE_FFTS=1` is required for this simulator path. Without it, the
launcher requests the FFTS C2C control address and CANN returns error `207000`
(`feature not supported`).

The CANN 9.1 beta `hivmc-a5` binary also rejects the `hacc.noinline` LLVM
attribute produced by the newer local NPU-IR default. The simulator runner sets
`TRITON_BISHENGIR_DISABLE_LIB_CALL_NOINLINE=1`, which adds
`--enable-lib-call-no-inline=false` to compilation.

After a correct kernel result, normal TorchNPU interpreter teardown can exit
with signal 11 under this simulator. `TRITON_SIMULATOR_CLEAN_EXIT=1` activates
an opt-in clean exit in `vector_add.py` after output synchronization and result
checking. Normal hardware runs do not use this path.
