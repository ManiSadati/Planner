# NPU-IR / PTOAS Comparison Flows

This document is the reusable runbook for comparing baseline AscendNPU-IR
against the NPU-IR-to-PTOAS bridge path. The examples use `vadd`, but the
variables are written so the same flow can be reused for any Triton kernel or
bridge testcase.

Use a normal shell for `msprof op simulator`. Inside Codex, `msprof` must run
outside the sandbox; otherwise local simulator socket/process setup can fail and
TorchNPU may report `aclInit` / empty SOC errors.

## Comparison Inputs

For the default `vadd` example:

```bash
cd "$HOME/Planner"

export CANN_ROOT=/path/to/cann-9.1.0-beta.3
export SOC_VERSION=Ascend950PR_9589
export CORE_ID=0

export TESTCASE=vadd
export KERNEL_NAME=vector_add_kernel
export PY_FILE=bridge/triton-example/vector_add.py

export RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
export OUT_ROOT="$HOME/tmp/npuir-ptoas-comparison/${TESTCASE}-${RUN_ID}"
mkdir -p "$OUT_ROOT"
```

For another kernel, change:

```bash
export TESTCASE=<bridge-testcase-name>
export KERNEL_NAME=<triton-kernel-function-name>
export PY_FILE=<repo-relative-python-file>
```

Record exact source versions with every comparison:

```bash
git -C "$HOME/AscendNPU-IR" rev-parse --abbrev-ref HEAD
git -C "$HOME/AscendNPU-IR" rev-parse HEAD
git -C "$HOME/PTOAS/PTOAS_Markham" rev-parse --abbrev-ref HEAD
git -C "$HOME/PTOAS/PTOAS_Markham" rev-parse HEAD
git -C "$HOME/Planner" rev-parse HEAD
```

This matters because PTOAS VMI, VPTO address analysis, and sync behavior are
actively moving upstream.

## Flow 1: Early IR Dump From Triton

Purpose: generate or refresh early NPU-IR/TTAdapter MLIR from the Python
Triton source.

```bash
cd "$HOME/Planner"
export CANN_ROOT=/path/to/cann-9.1.0-beta.3
export SOC_VERSION=Ascend950PR_9589
export CORE_ID=0

EARLY_OUT="$OUT_ROOT/early-ir"

NPUIR/tools/dump_early_ir_from_triton.sh \
  "$KERNEL_NAME" \
  "$PY_FILE" \
  "$EARLY_OUT"
```

Pick the TTAdapter MLIR for later compile-only flows:

```bash
export EARLY_IR="$(
  find "$EARLY_OUT/early-ir" -name '*kernel.ttadapter.mlir' | sort | tail -1
)"
echo "$EARLY_IR"
```

Important outputs:

```text
$EARLY_OUT/msprof.log
$EARLY_OUT/early-ir-manifest.txt
$EARLY_OUT/early-ir/*kernel.ttadapter.mlir
$EARLY_OUT/early-ir/*kernel.ttir.mlir
```

## Flow 2: Full Baseline NPU-IR Simulator

Purpose: run the original Triton Python through baseline NPU-IR and the CANN
operator simulator. This gives the baseline correctness result and simulator
profile.

```bash
cd "$HOME/Planner"
export CANN_ROOT=/path/to/cann-9.1.0-beta.3
source NPUIR/tools/source_npuir_simulator_env.sh

BASELINE_SIM_OUT="$OUT_ROOT/baseline-npuir-sim"
mkdir -p "$BASELINE_SIM_OUT"/cache "$BASELINE_SIM_OUT"/dump \
  "$BASELINE_SIM_OUT"/logs "$BASELINE_SIM_OUT"/profile
chmod 700 "$BASELINE_SIM_OUT" "$BASELINE_SIM_OUT"/cache \
  "$BASELINE_SIM_OUT"/dump "$BASELINE_SIM_OUT"/logs \
  "$BASELINE_SIM_OUT"/profile

export TRITON_CACHE_DIR="$BASELINE_SIM_OUT/cache"
export TRITON_DUMP_DIR="$BASELINE_SIM_OUT/dump"
export ASCEND_PROCESS_LOG_PATH="$BASELINE_SIM_OUT/logs"
unset BISHENGIR_ENABLE_PTOAS_BRIDGE

msprof op simulator \
  --kernel-name="$KERNEL_NAME" \
  --soc-version="$SOC_VERSION" \
  --core-id="$CORE_ID" \
  --output="$BASELINE_SIM_OUT/profile" \
  python3 "$PLANNER_DIR/$PY_FILE" \
  >"$BASELINE_SIM_OUT/msprof.stdout.log" \
  2>"$BASELINE_SIM_OUT/msprof.stderr.log"
```

Expected for `vadd`:

```text
max error: 0.0
allclose: True
```

Important outputs:

```text
$BASELINE_SIM_OUT/msprof.stdout.log
$BASELINE_SIM_OUT/msprof.stderr.log
$BASELINE_SIM_OUT/dump/
$BASELINE_SIM_OUT/profile/
$BASELINE_SIM_OUT/logs/
```

The profile directory contains simulator timing tables and traces. Use the same
SOC, core id, input shape, and kernel launch shape when comparing against the
bridge path.

## Flow 3: Baseline NPU-IR Compile-Only IR Dump

Purpose: inspect the baseline NPU-IR lowering around the AVE intrinsic boundary
without enabling the PTOAS bridge passes.

```bash
cd "$HOME/Planner"
unset BISHENGIR_ENABLE_PTOAS_BRIDGE

BASELINE_IR_OUT="$OUT_ROOT/baseline-npuir-ir"
mkdir -p "$BASELINE_IR_OUT"

"$HOME/AscendNPU-IR/build/install/bin/bishengir-compile" "$EARLY_IR" \
  --target=Ascend910_9589 \
  --enable-auto-multi-buffer=true \
  --enable-auto-bind-sub-block=true \
  --disable-ffts \
  --limit-auto-multi-buffer-of-local-buffer=no-limit \
  --enable-auto-blockify-loop \
  --enable-hfusion-compile=true \
  --enable-hivm-compile=true \
  --enable-triton-kernel-compile=true \
  --mlir-disable-threading \
  --mlir-print-stacktrace-on-diagnostic \
  --enable-vf-merge-level=1 \
  --mlir-print-ir-after=convert-hivmave-to-ave-intrin \
  --save-temps="$BASELINE_IR_OUT/temps" \
  -o "$BASELINE_IR_OUT/$KERNEL_NAME" \
  >"$BASELINE_IR_OUT/compile.log" \
  2>&1
```

Useful output:

```text
$BASELINE_IR_OUT/compile.log
$BASELINE_IR_OUT/temps/
```

On this server, the command may still fail after the requested IR dump if the
closed backend tool such as `hivmc-a5` is unavailable. That is acceptable for
IR comparison if the requested dump was printed.

## Flow 4: Bridge Compile To PTOAS VMI

Purpose: run NPU-IR with the bridge passes enabled and capture the PTOAS VMI
boundary.

```bash
cd "$HOME/Planner"
export BISHENGIR_ENABLE_PTOAS_BRIDGE=1

BRIDGE_IR_OUT="$OUT_ROOT/bridge-ptoas-vmi"
mkdir -p "$BRIDGE_IR_OUT"

"$HOME/AscendNPU-IR/build/install/bin/bishengir-compile" "$EARLY_IR" \
  --target=Ascend910_9589 \
  --enable-auto-multi-buffer=true \
  --enable-auto-bind-sub-block=true \
  --disable-ffts \
  --limit-auto-multi-buffer-of-local-buffer=no-limit \
  --enable-auto-blockify-loop \
  --enable-hfusion-compile=true \
  --enable-hivm-compile=true \
  --enable-triton-kernel-compile=true \
  --mlir-disable-threading \
  --mlir-print-stacktrace-on-diagnostic \
  --enable-vf-merge-level=1 \
  --mlir-print-ir-after=convert-hivmave-to-ptoas-vmi \
  --save-temps="$BRIDGE_IR_OUT/temps" \
  -o "$BRIDGE_IR_OUT/$KERNEL_NAME" \
  >"$BRIDGE_IR_OUT/compile.log" \
  2>&1
```

For checked-in bridge testcases, prefer the runner because it extracts the last
successful VMI dump automatically:

```bash
cd "$HOME/Planner"
export BISHENGIR_ENABLE_PTOAS_BRIDGE=1

OUTPUT_ROOT="$OUT_ROOT/bridge-runner" \
bridge/tools/run_npuir_ptoas_bridge_tests.sh \
  --from-bishengir-compile \
  --clean \
  --emit-vpto \
  --emit-llvmir \
  "$TESTCASE"
```

Important runner outputs:

```text
$OUT_ROOT/bridge-runner/$TESTCASE/$TESTCASE.vmi.mlir
$OUT_ROOT/bridge-runner/$TESTCASE/$TESTCASE.vpto.mlir
$OUT_ROOT/bridge-runner/$TESTCASE/$TESTCASE.vpto.ll
$OUT_ROOT/bridge-runner/$TESTCASE/npuir.log
$OUT_ROOT/bridge-runner/$TESTCASE/npuir-command.txt
$OUT_ROOT/bridge-runner/$TESTCASE/ptoas-vpto-command.txt
$OUT_ROOT/bridge-runner/$TESTCASE/ptoas-llvmir-command.txt
```

For `vadd`, the VMI output should include vector and movement operations such as:

```text
pto.vmi.load
pto.vmi.vadd
pto.vmi.store
pto.mte_gm_ub
pto.mte_ub_gm
```

## Flow 5: PTOAS Lowering Only

Purpose: rerun PTOAS lowering manually from a bridge-produced VMI file.

```bash
cd "$HOME/Planner"
source "$HOME/.bashrc"
activate_ptoas >/dev/null

VMI_MLIR="$OUT_ROOT/bridge-runner/$TESTCASE/$TESTCASE.vmi.mlir"
VPTO_MLIR="$OUT_ROOT/ptoas-only/$TESTCASE.vpto.mlir"
VPTO_LL="$OUT_ROOT/ptoas-only/$TESTCASE.vpto.ll"
mkdir -p "$(dirname "$VPTO_MLIR")"

ptoas \
  --pto-arch=a5 \
  --pto-backend=vpto \
  --emit-vpto \
  "$VMI_MLIR" \
  -o "$VPTO_MLIR"

ptoas \
  --pto-arch=a5 \
  --pto-backend=vpto \
  --emit-vpto-llvm-ir \
  "$VMI_MLIR" \
  -o "$VPTO_LL"
```

Use `activate_ptoas` or an equivalent PTOAS environment. Calling a raw PTOAS
binary without the matching Python/runtime environment can fail on Python module
imports.

## Flow 6: NPU-IR-to-PTOAS Simulator

Purpose: run the converted PTOAS path in the CANN simulator with the testcase's
PTOAS host fixture.

This requires a bridge testcase fixture:

```text
bridge/testcases/<testcase>/
  run_sim.sh
  CMakeLists.txt
  main.cpp
  launch.cpp
  gen_data.py
  compare.py
```

If a new kernel does not have this fixture yet, use flows 1-5 first; runtime
comparison is not equivalent until the host fixture matches the Triton host
behavior.

```bash
cd "$HOME/Planner"
source "$HOME/.bashrc"
activate_ptoas >/dev/null

export BISHENGIR_ENABLE_PTOAS_BRIDGE=1
export ASCEND_HOME_PATH="$CANN_ROOT"
export SOC_VERSION=Ascend950PR_9599
export SIM_LIB_DIR="$ASCEND_HOME_PATH/tools/simulator/$SOC_VERSION/lib"
export BUILD_JOBS=16

OUTPUT_ROOT="$OUT_ROOT/bridge-sim" \
PTOAS_BIN="$(command -v ptoas)" \
BISHENGIR_COMPILE="$HOME/AscendNPU-IR/build/install/bin/bishengir-compile" \
bridge/tools/run_npuir_ptoas_bridge_tests.sh \
  --from-bishengir-compile \
  --clean \
  --run-simulator \
  "$TESTCASE"
```

Important outputs:

```text
$OUT_ROOT/bridge-sim/$TESTCASE/sim.log
$OUT_ROOT/bridge-sim/$TESTCASE/sim-command.txt
$OUT_ROOT/bridge-sim/$TESTCASE/$TESTCASE.vpto.mlir
```

Keep this separate from `msprof op simulator` on the original Triton Python.
The baseline path uses the Triton/NPU-IR host path; the bridge simulator path
uses the PTOAS testcase host fixture.

## Flow 7: A5 Hardware Runtime

Purpose: get authoritative runtime and correctness on real hardware. This
cannot be completed on the Codex-accessible simulator-only server.

Run the same source versions and input shapes on the A5 server. Capture:

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
early dumps under the Planner examples area and use flows 3-5 locally.

## Minimum Comparison Checklist

For each kernel, collect:

| Item | Baseline NPU-IR | Bridge / PTOAS |
|---|---|---|
| Source commit | `AscendNPU-IR` commit | `AscendNPU-IR` + `PTOAS` commits |
| Input source | Triton Python and shape | same Triton semantics or equivalent fixture |
| Early IR | TTAdapter MLIR | same TTAdapter MLIR when possible |
| Functional result | Torch check / simulator output | PTOAS fixture check |
| IR checkpoint | AVE intrinsic dump | PTOAS VMI, VPTO, VPTO LLVM IR |
| Simulator target | same `SOC_VERSION`, `CORE_ID` | same or explicitly documented |
| Timing metric | simulator profile cycles/ticks/runtime | same metric source |
| Hardware runtime | A5 result when available | A5 result when available |

Do not compare baseline Python/Triton simulator runtime against a PTOAS fixture
runtime unless the input shape, data initialization, launch grid, core count,
and verification path are known to be equivalent.

## Adding A New Kernel

Start with the cheap path:

1. Add or point to a Python/Triton file with host-side correctness checking.
2. Set `KERNEL_NAME`, `PY_FILE`, and `TESTCASE`.
3. Run flow 1 to capture early IR.
4. Run flows 3 and 4 to compare baseline AVE IR against bridge PTOAS VMI.
5. Run flow 5 to check PTOAS can lower the VMI to VPTO / LLVM IR.

Only add runtime comparison after the host side is equivalent:

1. Add `bridge/testcases/<testcase>/compile-input.mlir` or `compile_input.mlir`.
2. Add a PTOAS simulator fixture matching the Triton host behavior.
3. Run flow 2 for baseline simulator results.
4. Run flow 6 for bridge simulator results.
5. Compare correctness first, then cycles/ticks/runtime.

