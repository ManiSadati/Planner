# NPU-IR and PTOAS Comparison Flows

`bridge/tools/run_comparison_flow.sh` compares two end-to-end Triton-Ascend
compiler paths with the same Python testcase and generated Triton host launcher:

```text
npu-sim:
  Triton Python -> TTIR -> NPU-IR -> npubin -> generated host launcher -> simulator

bridge-sim:
  Triton Python -> TTIR -> NPU-IR -> PTOAS VMI -> PTOAS -> npubin
                -> generated host launcher -> simulator
```

The script also provides compiler-only commands for inspecting PTOAS VMI and
VPTO MLIR. These inspection commands are not part of the simulation launch.

## Prerequisites

The following components must already be built:

- AscendNPU-IR with `bishengir-compile --emit-ptoas-vmi` support.
- PTOAS with an executable `ptoas` binary.
- Triton-Ascend with the `TRITON_ASCEND_COMPILE_FLOW=ptoas` changes.
- A simulator-compatible Python environment containing Torch, Torch-NPU, and
  the modified Triton-Ascend installation.

Set the three repository/install roots:

```bash
cd /home/m00967009/Workspace/Planner

export ASCEND_NPU_IR_ROOT=/home/m00967009/Workspace/AscendNPU-IR
export PTOAS_ROOT=/home/m00967009/Workspace/PTOAS
export CANN_ROOT=/home/a84369921/Ascend/cann-9.1.0-beta.3
export NPU_SIM_VENV=/home/m00967009/.venv/npuir-sim-system
```

The script sources `$CANN_ROOT/set_env.sh`, activates `$NPU_SIM_VENV`, disables
user-site package shadowing with `PYTHONNOUSERSITE=1`, and verifies that the
environment can import Torch, Torch-NPU, and the Triton Ascend backend. It
preserves the CANN `PYTHONPATH` configured by `set_env.sh`, which the simulator
needs during `aclInit`. If `NPU_SIM_VENV` is unset, it defaults to:

```text
$HOME/.venv/npuir-sim-system
```

Compiler selection is flow-specific:

- `npu-sim` uses CANN's `tools/bishengir/bin/bishengir-compile`, which matches
  the installed `hivmc-a5` used by the native 9599 code-generation path.
- `bridge-sim`, `emit-vmi`, and `print-ir` use the rebuilt AscendNPU-IR
  compiler, which contains the local `--emit-ptoas-vmi` change.

Using the rebuilt compiler with CANN's older `hivmc-a5` in the native flow can
fail when `hivmc-a5` encounters newer attributes such as `hacc.noinline`.

Verify the selected local tools before a long simulator run:

```bash
"$ASCEND_NPU_IR_ROOT/build/bin/bishengir-compile" --help 2>&1 \
  | grep -- --emit-ptoas-vmi
"$PTOAS_ROOT/build/tools/ptoas/ptoas" --help | head
```

## Testcase Layout

A simulator testcase needs exactly one Python file containing one
`@triton.jit` kernel:

```text
bridge/testcases/<case>/
  <kernel>.py
```

The script infers both the Python filename and kernel function name. The Python
program remains responsible for creating tensors, launching the kernel, and
comparing its output with Torch.

Compiler-only inspection also needs a checked-in or otherwise prepared input.
The preferred filename is:

```text
bridge/testcases/<case>/compile-input.mlir
```

If `compile-input.mlir` does not exist, the script falls back to `input.mlir`.
The script does not generate either file. End-to-end simulation starts directly
from the Python testcase, while `emit-vmi` uses this explicit, replayable MLIR
input.

## Command Interface

```bash
bridge/tools/run_comparison_flow.sh [--clean-build] \
  [--print-ir-after-all] <option> <testcase>
```

Available options:

| Option | Purpose |
| --- | --- |
| `npu-sim` | Run the Python testcase through native NPU-IR. |
| `bridge-sim` | Run the same Python testcase through NPU-IR and PTOAS. |
| `emit-vmi` | Compile the testcase MLIR directly to PTOAS VMI MLIR. |
| `emit-vpto` | Lower the VMI MLIR to VPTO MLIR for inspection. |
| `print-ir` | Run `emit-vmi` and print IR after every NPU-IR pass. |

`--clean-build` removes the testcase's build directories before running. It
does not remove the testcase source files.

## Native NPU-IR Simulation

Run the baseline:

```bash
bridge/tools/run_comparison_flow.sh npu-sim vadd
```

The script runs the equivalent of:

```bash
TRITON_ASCEND_COMPILE_FLOW=npuir \
TRITON_ASCEND_ARCH=Ascend950PR_9599 \
msprof op simulator \
  --kernel-name=vector_add_kernel \
  --soc-version=Ascend950PR_9599 \
  --core-id=0 \
  --output=bridge/testcases/vadd/out/build/npu-python/profile \
  python3 bridge/testcases/vadd/vector_add.py
```

Important outputs:

```text
bridge/testcases/vadd/out/build/npu-python/msprof.log
bridge/testcases/vadd/out/build/npu-python/command.txt
bridge/testcases/vadd/out/build/npu-python/cache/
bridge/testcases/vadd/out/build/npu-python/dump/
bridge/testcases/vadd/out/build/npu-python/profile/
```

For `vadd`, success requires `allclose: True` in `msprof.log` and no Python
traceback. An `msprof` report directory by itself does not prove correctness.

## NPU-IR-to-PTOAS Simulation

Run the new flow with the same Python host program:

```bash
bridge/tools/run_comparison_flow.sh bridge-sim vadd
```

The script sets:

```text
TRITON_ASCEND_COMPILE_FLOW=ptoas
TRITON_ASCEND_ARCH=Ascend950PR_9599
TRITON_PTOAS_PATH=<resolved PTOAS binary>
```

It then launches the Python testcase with `msprof op simulator` for
`Ascend950PR_9599`. Triton-Ascend performs VMI emission, PTOAS compilation,
npubin construction, and launch through its existing generated host code.
There is no separate CMake/C++ host fixture in this flow.

Important outputs:

```text
bridge/testcases/vadd/out/build/ptoas-python/msprof.log
bridge/testcases/vadd/out/build/ptoas-python/command.txt
bridge/testcases/vadd/out/build/ptoas-python/cache/
bridge/testcases/vadd/out/build/ptoas-python/dump/
bridge/testcases/vadd/out/build/ptoas-python/profile/
```

The expected correctness result is again `allclose: True`.

## Emit PTOAS VMI Directly

To inspect only the NPU-IR bridge endpoint:

```bash
bridge/tools/run_comparison_flow.sh emit-vmi vadd
```

This invokes the rebuilt compiler with:

```text
bishengir-compile bridge/testcases/vadd/compile-input.mlir \
  <NPU-IR pipeline flags> \
  --emit-ptoas-vmi \
  -o bridge/testcases/vadd/out/vadd.vmi.mlir
```

The VMI file is now a normal compiler output. The script no longer requests a
single pass diagnostic and extracts MLIR text from the compiler log.

Outputs:

```text
bridge/testcases/vadd/out/vadd.vmi.mlir
bridge/testcases/vadd/out/build/emit-vmi.log
bridge/testcases/vadd/out/build/emit-vmi.command.txt
bridge/testcases/vadd/out/build/temps-vmi/
```

## Print IR After Every Pass

Use either spelling:

```bash
bridge/tools/run_comparison_flow.sh print-ir vadd
```

```bash
bridge/tools/run_comparison_flow.sh --print-ir-after-all emit-vmi vadd
```

Both add `--mlir-print-ir-after-all` to the direct `bishengir-compile` command
while still writing the final VMI module normally with `--emit-ptoas-vmi`.

Outputs:

```text
bridge/testcases/vadd/out/after-all.log
bridge/testcases/vadd/out/vadd.vmi.mlir
```

This can produce a large log. Use it to inspect transformations around a
specific pass without relying on the log as the source of the final VMI file.

## Emit VPTO For Inspection

Run:

```bash
bridge/tools/run_comparison_flow.sh emit-vpto vadd
```

If `out/vadd.vmi.mlir` does not exist, the script first runs `emit-vmi`. It then
invokes PTOAS with `--pto-backend=vpto --emit-vpto`.

Outputs:

```text
bridge/testcases/vadd/out/vadd.vpto.mlir
bridge/testcases/vadd/out/build/emit-vpto.log
bridge/testcases/vadd/out/build/emit-vpto.command.txt
```

This command is for inspecting PTOAS lowering. `bridge-sim` does not consume
this VPTO file; it exercises Triton-Ascend's complete integrated PTOAS flow.

## Common Comparison Sequence

For a clean comparison of both executable paths:

```bash
bridge/tools/run_comparison_flow.sh --clean-build npu-sim vadd
bridge/tools/run_comparison_flow.sh bridge-sim vadd
```

For compiler inspection in addition to simulation:

```bash
bridge/tools/run_comparison_flow.sh print-ir vadd
bridge/tools/run_comparison_flow.sh emit-vpto vadd
```

The compatibility wrapper can run that full sequence:

```bash
bridge/tools/run_npuir_ptoas_bridge_tests.sh --all vadd
```

## Troubleshooting

### The compiler rejects `--emit-ptoas-vmi`

Check `out/build/emit-vmi.command.txt`. Its first argument must resolve to the
rebuilt AscendNPU-IR compiler, not the compiler under the CANN installation.

```bash
"$ASCEND_NPU_IR_ROOT/build/bin/bishengir-compile" --help 2>&1 \
  | grep -- --emit-ptoas-vmi
```

### Triton reports zero active drivers

The Python environment does not see the installed Triton-Ascend backend. Run
the script from the simulator virtual environment in which the modified
Triton-Ascend package was installed, and keep `PYTHONNOUSERSITE=1` so a second
user-site Triton package cannot shadow it.

### The simulator rejects testcase permissions

`msprof` rejects group- or world-writable application directories. Fix the
testcase permissions before running:

```bash
chmod go-w bridge/testcases/vadd
chmod go-w bridge/testcases/vadd/vector_add.py
```

### The simulator creates a report but the test does not pass

Inspect the corresponding `msprof.log`. A successful run must contain the
Python comparison result and must not contain `skip running kernel`, a
traceback, or an application exception before comparison.
