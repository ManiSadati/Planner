# Vector Add End-to-End Workflow

This document uses `bridge/triton-example/vector_add.py` as the smallest
end-to-end testcase for comparing:

- baseline NPU-IR from Triton Python to AVE intrinsics / simulator runtime;
- NPU-IR with the PTOAS bridge passes enabled;
- PTOAS lowering from bridge-produced PTO/VMI to VPTO and VPTO LLVM IR;
- PTOAS simulator execution of the bridge-produced VPTO.

The commands below keep environment setup in existing scripts where possible,
but keep the important compiler and simulator invocations visible.

## Inputs And Outputs

Important inputs:

```text
bridge/triton-example/vector_add.py
bridge/testcases/vadd/compile-input.mlir
bridge/testcases/vadd/input.mlir
```

Important generated output roots:

```text
$HOME/tmp/npuir-early-ir/
bridge/out/npuir-ptoas-bridge/vadd/
```

`bridge/triton-example/vector_add.py` is the Python/Triton source. It launches
`vector_add_kernel`, adds two 256-element `float32` vectors, and checks the
result with Torch.

`bridge/testcases/vadd/compile-input.mlir` is the checked-in NPU-IR compile
input used by the bridge runner. Use it when you want a stable repo-local input
instead of regenerating early IR from Python.

## 1. Set Up The NPU-IR Simulator Environment

Use the shared environment script. It activates the NPU-IR simulator venv,
sources CANN, and puts the local NPU-IR build on `PATH`.

```bash
cd "$HOME/Planner"
export CANN_ROOT=/path/to/cann-9.1.0-beta.3
source NPUIR/tools/source_npuir_simulator_env.sh
```

The script expects:

```text
$HOME/AscendNPU-IR/build/install/bin/bishengir-compile
$HOME/.venv/npuir-sim-system/
```

## 2. Generate Early IR From `vector_add.py`

The practical way to get the early Triton/NPU-IR dump on Bluezone is to invoke
the NPU-IR simulator compile path and collect the dumps. The simulator may run
the kernel too, but the purpose of this step is IR capture.

```bash
cd "$HOME/Planner"
NPUIR/tools/dump_early_ir_from_triton.sh \
  vector_add_kernel \
  bridge/triton-example/vector_add.py
```

The script prints an output root like:

```text
$HOME/tmp/npuir-early-ir/vector_add_kernel-YYYYMMDDTHHMMSSZ
```

Useful files:

```text
$OUT/command.txt
$OUT/msprof.log
$OUT/early-ir-manifest.txt
$OUT/early-ir/*kernel.ttir.mlir
$OUT/early-ir/*kernel.ttadapter.mlir
```

For later manual commands:

```bash
EARLY_OUT="$HOME/tmp/npuir-early-ir/vector_add_kernel-YYYYMMDDTHHMMSSZ"
EARLY_IR="$(find "$EARLY_OUT/early-ir" -name '*kernel.ttadapter.mlir' | sort | tail -1)"
```

## 3. Compile With The PTOAS Bridge Passes Enabled

The bridge passes are default-off. Enable them explicitly:

```bash
export BISHENGIR_ENABLE_PTOAS_BRIDGE=1
```

Manual `bishengir-compile` command:

```bash
cd "$HOME/Planner"
BRIDGE_OUT="$HOME/tmp/npuir-bridge-vadd"
mkdir -p "$BRIDGE_OUT"

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
  --save-temps="$BRIDGE_OUT/temps" \
  -o "$BRIDGE_OUT/vector_add_kernel" \
  >"$BRIDGE_OUT/npuir-bridge.log" \
  2>&1
```

Expected bridge behavior:

- `convert-hivm-templates-to-pto` runs before `convert-hivm-to-std`;
- `convert-hivmave-to-ptoas-vmi` runs before `convert-hivmave-to-ave-intrin`;
- the log should contain an IR dump after `convert-hivmave-to-ptoas-vmi`;
- later compiler stages may fail while the bridge dump is still useful.

For routine work, prefer the existing bridge runner because it extracts the
last successful `convert-hivmave-to-ptoas-vmi` dump automatically:

```bash
cd "$HOME/Planner"
export BISHENGIR_ENABLE_PTOAS_BRIDGE=1
bridge/tools/run_npuir_ptoas_bridge_tests.sh \
  --from-bishengir-compile \
  --clean \
  --emit-vpto \
  --emit-llvmir \
  vadd
```

That writes:

```text
bridge/out/npuir-ptoas-bridge/vadd/vadd.vmi.mlir
bridge/out/npuir-ptoas-bridge/vadd/vadd.vpto.mlir
bridge/out/npuir-ptoas-bridge/vadd/vadd.vpto.ll
bridge/out/npuir-ptoas-bridge/vadd/npuir.log
bridge/out/npuir-ptoas-bridge/vadd/npuir-command.txt
```

## 4. Lower The Bridge Output With PTOAS

If you already have a bridge-produced VMI file:

```bash
VMI_MLIR="$HOME/Planner/bridge/out/npuir-ptoas-bridge/vadd/vadd.vmi.mlir"
VPTO_MLIR="$HOME/Planner/bridge/out/npuir-ptoas-bridge/vadd/vadd.vpto.mlir"
VPTO_LL="$HOME/Planner/bridge/out/npuir-ptoas-bridge/vadd/vadd.vpto.ll"
PTOAS_BIN="$HOME/PTOAS/PTOAS_Markham/build/tools/ptoas/ptoas"
```

Emit VPTO MLIR:

```bash
"$PTOAS_BIN" \
  --pto-arch=a5 \
  --pto-backend=vpto \
  --emit-vpto \
  "$VMI_MLIR" \
  -o "$VPTO_MLIR"
```

Emit VPTO LLVM IR:

```bash
"$PTOAS_BIN" \
  --pto-arch=a5 \
  --pto-backend=vpto \
  --emit-vpto-llvm-ir \
  "$VMI_MLIR" \
  -o "$VPTO_LL"
```

The bridge runner in step 3 runs both PTOAS commands for you when
`--emit-vpto --emit-llvmir` is used.

## 5. Run The NPU-IR-to-PTOAS Path In Simulator

This is the bridge runtime simulation path:

```text
NPU-IR compile input
  -> bridge-produced PTO/VMI
  -> PTOAS VPTO
  -> PTOAS testcase host fixture
  -> CANN simulator
```

Use the bridge runner:

```bash
cd "$HOME/Planner"
export BISHENGIR_ENABLE_PTOAS_BRIDGE=1
export ASCEND_HOME_PATH=/path/to/cann-9.1.0-beta.3
export SOC_VERSION=Ascend950PR_9599
export SIM_LIB_DIR="$ASCEND_HOME_PATH/tools/simulator/$SOC_VERSION/lib"
export BUILD_JOBS=16

bridge/tools/run_npuir_ptoas_bridge_tests.sh \
  --from-bishengir-compile \
  --clean \
  --run-simulator \
  vadd
```

The underlying simulator fixture command, after the runner has copied the
testcase into `bridge/out/.../sim/`, is:

```bash
cd "$HOME/Planner/bridge/out/npuir-ptoas-bridge/vadd/sim"
PTOAS_BIN="$HOME/PTOAS/PTOAS_Markham/build/tools/ptoas/ptoas" \
KERNEL_MLIR="$HOME/Planner/bridge/out/npuir-ptoas-bridge/vadd/vadd.vpto.mlir" \
BUILD_DIR="$HOME/Planner/bridge/out/npuir-ptoas-bridge/vadd/sim/build" \
RUN_DIR="$HOME/Planner/bridge/out/npuir-ptoas-bridge/vadd/sim/build/run" \
bash run_sim.sh
```

The bridge runner passes the generated VPTO file to the simulator fixture through
`KERNEL_MLIR`; users normally only select the testcase name.

Useful outputs:

```text
bridge/out/npuir-ptoas-bridge/vadd/sim.log
bridge/out/npuir-ptoas-bridge/vadd/sim-command.txt
```

The active comparison wrapper profiles the PTOAS-generated executable with
`msprof op simulator --application ...`, while still using the vadd PTOAS host
fixture for data generation and correctness checking.

## 6. Compile Baseline NPU-IR To AVE Intrinsics Without Bridge Passes

Unset the bridge flag to keep baseline NPU-IR behavior:

```bash
unset BISHENGIR_ENABLE_PTOAS_BRIDGE
```

Manual `bishengir-compile` command:

```bash
cd "$HOME/Planner"
BASELINE_OUT="$HOME/tmp/npuir-baseline-vadd"
mkdir -p "$BASELINE_OUT"

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
  --save-temps="$BASELINE_OUT/temps" \
  -o "$BASELINE_OUT/vector_add_kernel" \
  >"$BASELINE_OUT/npuir-baseline.log" \
  2>&1
```

For checked-in early IR fixtures, the existing replay helper does the same
style of pass-dump capture:

```bash
cd "$HOME/Planner"
unset BISHENGIR_ENABLE_PTOAS_BRIDGE
TARGET_PASS=convert-hivmave-to-ave-intrin \
INPUT_DIR=bridge/triton-example \
OUTPUT_ROOT="$HOME/tmp/npuir-baseline-replay" \
NPUIR/tools/replay_npuir_from_device_spec.sh
```

Useful output:

```text
$BASELINE_OUT/npuir-baseline.log
$HOME/tmp/npuir-baseline-replay/vector_add_kernel/after-convert-hivmave-to-ave-intrin.mlir
```

## 7. Run Full Baseline NPU-IR Simulation From Triton Python

This is the full baseline path:

```text
vector_add.py
  -> Triton/NPU-IR lowering
  -> NPU-IR backend
  -> CANN operator simulator
  -> Torch correctness check
```

Use the environment script first:

```bash
cd "$HOME/Planner"
export CANN_ROOT=/path/to/cann-9.1.0-beta.3
source NPUIR/tools/source_npuir_simulator_env.sh
```

Manual `msprof op simulator` command:

```bash
OUT="$HOME/tmp/npuir-simulator/vector-add-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUT"/cache "$OUT"/dump "$OUT"/logs "$OUT"/profile
chmod 700 "$OUT" "$OUT"/cache "$OUT"/dump "$OUT"/logs "$OUT"/profile

export TRITON_CACHE_DIR="$OUT/cache"
export TRITON_DUMP_DIR="$OUT/dump"
export ASCEND_PROCESS_LOG_PATH="$OUT/logs"
unset BISHENGIR_ENABLE_PTOAS_BRIDGE

msprof op simulator \
  --kernel-name=vector_add_kernel \
  --soc-version=Ascend950PR_9589 \
  --core-id=0 \
  --output="$OUT/profile" \
  python3 "$HOME/Planner/bridge/triton-example/vector_add.py"
```

The one-shot wrapper for this exact baseline case is:

```bash
cd "$HOME/Planner"
export CANN_ROOT=/path/to/cann-9.1.0-beta.3
NPUIR/tools/run_vector_add_simulator.sh
```

Expected correctness output from the Python file:

```text
max error: 0.0
allclose: True
```

Useful outputs:

```text
$OUT/dump/
$OUT/profile/
$OUT/logs/
```

## Comparison Matrix

| Goal | Bridge flag | Main command | Main output |
|---|---:|---|---|
| Early IR dump from Python | off by default | `NPUIR/tools/dump_early_ir_from_triton.sh vector_add_kernel bridge/triton-example/vector_add.py` | `$HOME/tmp/npuir-early-ir/.../early-ir/` |
| NPU-IR bridge compile | `1` | `bishengir-compile ... --mlir-print-ir-after=convert-hivmave-to-ptoas-vmi` | PTO/VMI dump in compile log |
| PTOAS VPTO / LLVM IR | n/a | `ptoas --emit-vpto`, `ptoas --emit-vpto-llvm-ir` | `.vpto.mlir`, `.vpto.ll` |
| PTOAS simulator for bridge path | `1` during bridge compile | `bridge/tools/run_comparison_flow.sh bridge-sim` | `$OUT_ROOT/bridge-sim/vadd/sim.log`, `$OUT_ROOT/bridge-sim/vadd/profile/` |
| Baseline AVE intrinsic dump | unset | `bishengir-compile ... --mlir-print-ir-after=convert-hivmave-to-ave-intrin` | AVE intrinsic dump in compile log |
| Full baseline NPU-IR simulator | unset | `msprof op simulator ... python3 vector_add.py` | Torch correctness, profile, ticks/logs |

## Current Gaps

- Manual extraction of a successful MLIR dump from a `bishengir-compile` log is
  intentionally avoided in this document. Use
  `bridge/tools/run_npuir_ptoas_bridge_tests.sh` or
  `NPUIR/tools/replay_npuir_from_device_spec.sh` when extraction matters.
- The bridge simulator fixture is currently vadd-specific. New kernels need
  equivalent host fixtures before runtime numbers are comparable.
- Simulator ticks/cycles are useful for rough comparison only. Final runtime
  claims still need the A5 hardware server.
