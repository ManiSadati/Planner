# NPU-IR / PTOAS Comparison Flows

This is the reusable runbook for comparing baseline AscendNPU-IR against the
NPU-IR-to-PTOAS bridge path. The default example is `vadd`, but the same scripts
work for other kernels when you provide a testcase name, Triton kernel name, and
Python file.

Use a normal shell for `msprof op simulator`. Inside Codex, `msprof` must run
outside the sandbox; otherwise local simulator socket/process setup can fail and
TorchNPU may report `aclInit` / empty SOC errors.

## Setup Once

Default `vadd` comparison:

```bash
cd "$HOME/Planner"
source bridge/tools/source_comparison_env.sh \
  --cann-root /path/to/cann-9.1.0-beta.3
```

Another kernel:

```bash
cd "$HOME/Planner"
source bridge/tools/source_comparison_env.sh \
  --cann-root /path/to/cann-9.1.0-beta.3 \
  --testcase <bridge-testcase-name> \
  --kernel-name <triton-kernel-function-name> \
  --python-file <repo-relative-python-file>
```

The setup script exports the shared comparison variables, creates `OUT_ROOT`,
and prints the active configuration. Main variables:

```text
TESTCASE
KERNEL_NAME
PY_FILE
CANN_ROOT
SOC_VERSION
CORE_ID
PTOAS_SIM_SOC_VERSION
OUT_ROOT
```

Default output root:

```text
$HOME/tmp/npuir-ptoas-comparison/<testcase>-<timestamp>
```

Record exact repo versions before comparing:

```bash
bridge/tools/run_comparison_flow.sh record-versions
```

Output:

```text
$OUT_ROOT/versions.md
```

This matters because PTOAS VMI, VPTO address analysis, and sync behavior are
actively moving upstream.

## Flow Commands

Run the cheap IR-focused sequence:

```bash
bridge/tools/run_comparison_flow.sh all-ir
```

This runs:

```text
record-versions
early-ir
baseline-ir
bridge-ir
bridge-lower
```

Run individual flows:

```bash
bridge/tools/run_comparison_flow.sh early-ir
bridge/tools/run_comparison_flow.sh baseline-sim
bridge/tools/run_comparison_flow.sh baseline-ir
bridge/tools/run_comparison_flow.sh bridge-ir
bridge/tools/run_comparison_flow.sh bridge-lower
bridge/tools/run_comparison_flow.sh ptoas-lower
bridge/tools/run_comparison_flow.sh bridge-sim
```

## Flow Meaning

| Flow | Purpose | Main output |
|---|---|---|
| `record-versions` | Save exact repo branch/commit state | `$OUT_ROOT/versions.md` |
| `early-ir` | Run Triton Python through `msprof` compile path and collect TTIR/TTAdapter MLIR | `$OUT_ROOT/early-ir/` |
| `baseline-sim` | Full baseline Triton -> NPU-IR -> CANN operator simulator run | `$OUT_ROOT/baseline-npuir-sim/` |
| `baseline-ir` | Compile early IR without bridge passes and dump `convert-hivmave-to-ave-intrin` | `$OUT_ROOT/baseline-npuir-ir/compile.log` |
| `bridge-ir` | Compile early IR with bridge passes and dump `convert-hivmave-to-ptoas-vmi` | `$OUT_ROOT/bridge-ptoas-vmi/compile.log` |
| `bridge-lower` | Run checked-in testcase through NPU-IR bridge, PTOAS VPTO, and PTOAS LLVM IR | `$OUT_ROOT/bridge-runner/$TESTCASE/` |
| `ptoas-lower` | Re-run PTOAS VPTO / LLVM lowering from the bridge-produced VMI | `$OUT_ROOT/ptoas-only/` |
| `bridge-sim` | Run the bridge-produced PTOAS path with the testcase PTOAS simulator fixture | `$OUT_ROOT/bridge-sim/$TESTCASE/` |

## Expected `vadd` Result

For the default `vadd` baseline simulator flow, expect:

```text
max error: 0.0
allclose: True
```

For the bridge VMI flow, expect operations such as:

```text
pto.vmi.load
pto.vmi.vadd
pto.vmi.store
pto.mte_gm_ub
pto.mte_ub_gm
```

## Simulator Notes

`baseline-sim` uses:

```text
Triton Python
  -> Triton/NPU-IR lowering
  -> NPU-IR backend
  -> CANN operator simulator
  -> Torch correctness check
```

`bridge-sim` uses:

```text
NPU-IR compile input
  -> bridge-produced PTO/VMI
  -> PTOAS VPTO
  -> PTOAS testcase host fixture
  -> CANN simulator
```

These are not automatically equivalent. For a new kernel, `bridge-sim` is only
comparable after the PTOAS host fixture matches the Triton host behavior:

```text
input shape
data initialization
launch grid / core count
output verification
timing metric source
```

## Adding A New Kernel

Start with the cheap path:

1. Add or point to a Python/Triton file with host-side correctness checking.
2. Source `bridge/tools/source_comparison_env.sh` with the new `--testcase`,
   `--kernel-name`, and `--python-file`.
3. Run `bridge/tools/run_comparison_flow.sh all-ir`.
4. Inspect baseline AVE IR, bridge PTOAS VMI, VPTO MLIR, and VPTO LLVM IR.

Only add runtime comparison after the host side is equivalent:

1. Add `bridge/testcases/<testcase>/compile-input.mlir` or `compile_input.mlir`.
2. Add a PTOAS simulator fixture:

```text
bridge/testcases/<testcase>/
  run_sim.sh
  CMakeLists.txt
  main.cpp
  launch.cpp
  gen_data.py
  compare.py
```

3. Run `bridge/tools/run_comparison_flow.sh baseline-sim`.
4. Run `bridge/tools/run_comparison_flow.sh bridge-sim`.
5. Compare correctness first, then cycles/ticks/runtime.

## A5 Hardware Runtime

This server is simulator-only. Authoritative runtime still needs the A5 server.
When running there, capture:

```text
NPU-IR commit
PTOAS commit
Planner testcase commit
kernel name
shape/input parameters
SOC/device
full command
stdout/stderr
runtime numbers
correctness output
```

If the A5 server can generate early MLIR but local replay is easier, place the
early dumps under the Planner examples area and run the compile/lowering flows
locally.

