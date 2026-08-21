# Comparison Flows

This is the human runbook for comparing baseline NPU-IR with the
NPU-IR-to-PTOAS path on the Codex-accessible Bluezone server.

The concrete example is:

```text
Python file:    bridge/triton-example/vector_add.py
Triton kernel:  vector_add_kernel
Bridge case:    vadd
```

For another kernel, use the same commands but change the setup values. For
example, for `bridge/triton-example/vector_add_large.py`:

```bash
source bridge/tools/source_comparison_env.sh \
  --cann-root /path/to/Ascend/cann-9.1.0-beta.3 \
  --testcase vadd_large \
  --kernel-name vector_add_large_kernel \
  --python-file bridge/triton-example/vector_add_large.py
```

For active NPU-IR bridge development, the comparison wrapper prefers
`$HOME/AscendNPU-IR/build/bin/bishengir-compile` when it exists. That build-tree
binary is usually newer than `$HOME/AscendNPU-IR/build/install/bin`.

Important: run `msprof op simulator` from a normal Bluezone shell. Inside
Codex, the simulator must run outside the sandbox; otherwise socket setup can
fail and TorchNPU may report `aclInit` or empty-SOC errors.

Before running any section below, do this once:

```bash
cd "$HOME/Planner"

source bridge/tools/source_comparison_env.sh \
  --cann-root /path/to/cann-9.1.0-beta.3 \
  --testcase vadd \
  --kernel-name vector_add_kernel \
  --python-file bridge/triton-example/vector_add.py

bridge/tools/run_comparison_flow.sh record-versions
```

The script prints the output directory. It will look like:

```text
$HOME/tmp/npuir-ptoas-comparison/vadd-YYYYMMDDTHHMMSSZ
```

The rest of this document refers to that directory as:

```text
$OUT_ROOT
```

## 1. Run End-To-End NPU-IR On Simulator On Codex Server / Bluezone

This is the baseline NPU-IR runtime path. It starts from the Triton Python file,
compiles through NPU-IR, runs with the CANN operator simulator, and checks the
result in Python.

Flow:

```text
vector_add.py
  -> Triton/NPU-IR lowering
  -> NPU-IR backend
  -> CANN operator simulator
  -> Python correctness check
```

Run:

```bash
bridge/tools/run_comparison_flow.sh baseline-sim
```

Main outputs:

```text
$OUT_ROOT/baseline-npuir-sim/msprof.stdout.log
$OUT_ROOT/baseline-npuir-sim/msprof.stderr.log
$OUT_ROOT/baseline-npuir-sim/dump/
$OUT_ROOT/baseline-npuir-sim/profile/
$OUT_ROOT/baseline-npuir-sim/logs/
```

For `vector_add.py`, expected correctness output (in $OUT_ROOT/baseline-npuir-sim/msprof.stdout.log):

```text
max error: 0.0
allclose: True
```

Use this as the baseline functional result and baseline simulator profile. When
you compare numbers later, keep the same SOC, core id, input shape, and launch
shape.

## 2. Generate Initial IR Dump From Triton Kernel In NPU-IR On Codex Server / Bluezone

This uses the simulator compile path to generate early IR dumps from the Triton
kernel. The most important artifact is the TTAdapter MLIR. That is the input we
reuse for compile-only experiments.

For `vector_add.py`, run:

```bash
bridge/tools/run_comparison_flow.sh early-ir
```

By default this does **not** wait for the full simulator run to finish. It stops
after the first `kernel.ttadapter.mlir` dump appears, because section 3 only
needs that initial compiler input.

If you want the simulator to keep running to the end while also collecting early
IR, run:

```bash
EARLY_IR_STOP_AFTER_DUMP=0 bridge/tools/run_comparison_flow.sh early-ir
```

Main outputs:

```text
$OUT_ROOT/early-ir/msprof.log
$OUT_ROOT/early-ir/early-ir-manifest.txt
$OUT_ROOT/early-ir/vadd.ttadapter.mlir
$OUT_ROOT/early-ir/early-ir/*kernel.ttadapter.mlir
$OUT_ROOT/early-ir/early-ir/*kernel.ttir.mlir
```

Example TTAdapter output path:

```text
$OUT_ROOT/early-ir/vadd.ttadapter.mlir
```

The nested `early-ir/early-ir/*kernel.ttadapter.mlir` files are the raw dumps.
The comparison wrapper copies the latest one to
`$OUT_ROOT/early-ir/vadd.ttadapter.mlir`, and the later compile flows use that
stable path automatically.

Use this section when you have a new Triton kernel and need the initial MLIR
that NPU-IR sees after Triton lowering.

### 2.1. Capture Baseline Pre-CCE LLVM IR

This does not run the full simulator. It replays the TTAdapter MLIR from
section 2 through `bishengir-compile --save-linked-ir`, copies the temporary
`kernel*.ll` / `*mix*.ll` files, and stops the compiler after the first
successful capture by default.

Run:

```bash
bridge/tools/run_comparison_flow.sh baseline-llvm-ir
```

Main outputs:

```text
$OUT_ROOT/baseline-pre-cce-llvm-ir/compile.log
$OUT_ROOT/baseline-pre-cce-llvm-ir/command.txt
$OUT_ROOT/baseline-pre-cce-llvm-ir/ll-dumps/
```

Use this when you want to inspect the LLVM/HIVM boundary that CCE would see,
without waiting for an end-to-end simulator run.

## 3. Perform `bishengir-compile` In Bluezone To Get The Full PTOAS Dialect

This starts from the TTAdapter MLIR from section 2 and runs the NPU-IR compiler
with the PTOAS bridge passes enabled.

The important output is the MLIR file that PTOAS can consume. For the current
bridge this is named:

```text
$OUT_ROOT/bridge-ptoas-vmi/vadd.vmi.mlir
```

Run:

```bash
bridge/tools/run_comparison_flow.sh bridge-ir
```

For `vadd`, the generated PTOAS-input MLIR should contain operations like:

```text
pto.vmi.load
pto.vmi.vadd
pto.vmi.store
pto.mte_gm_ub
pto.mte_ub_gm
```

Main outputs:

```text
$OUT_ROOT/bridge-ptoas-vmi/vadd.vmi.mlir
$OUT_ROOT/bridge-ptoas-vmi/compile.log
$OUT_ROOT/bridge-ptoas-vmi/compile-after-all.log
$OUT_ROOT/bridge-ptoas-vmi/command.txt
$OUT_ROOT/bridge-ptoas-vmi/command-after-all.txt
$OUT_ROOT/bridge-ptoas-vmi/after-convert-hivmave-to-ptoas-vmi-dump-count.txt
```

`compile.log` is the targeted dump after `convert-hivmave-to-ptoas-vmi`; the
script extracts `$OUT_ROOT/bridge-ptoas-vmi/vadd.vmi.mlir` from that file.
`compile-after-all.log` is the full `--mlir-print-ir-after-all` log for
debugging the surrounding NPU-IR passes.

If you also want PTOAS VPTO and VPTO LLVM IR immediately after the bridge
compile, run section 4.2 next.

There is also an older testcase-runner path:

```bash
bridge/tools/run_comparison_flow.sh bridge-lower
```

That path uses the checked-in `bridge/testcases/vadd/compile-input.mlir`
instead of the section 2 TTAdapter dump. Use it only when you intentionally want
the checked-in fixture input.

Baseline NPU-IR is still default-off for the bridge. The script enables
`BISHENGIR_ENABLE_PTOAS_BRIDGE=1` only for the bridge flows.

## 4. Use The PTOAS MLIR Generated From Section 3

The input for this section is:

```text
$OUT_ROOT/bridge-ptoas-vmi/vadd.vmi.mlir
```

For another testcase, replace `vadd` with that testcase name. For example:

```text
$OUT_ROOT/bridge-ptoas-vmi/vadd_large.vmi.mlir
```

### 4.1. How To Write The Host Code

The PTOAS simulator path needs host code. That host code must match the Triton
Python testcase, otherwise runtime numbers are not comparable.

For `vector_add.py`, the host fixture is:

```text
bridge/testcases/vadd/
```

A new testcase should use the same structure:

```text
bridge/testcases/<new_case>/
  compile-input.mlir
  run_sim.sh
  CMakeLists.txt
  main.cpp
  launch.cpp
  gen_data.py
  compare.py
```

What each file should do:

```text
compile-input.mlir
  Stable NPU-IR input for the bridge compiler flow.

gen_data.py
  Creates the same input tensors as the Triton Python file.
  For vector_add.py, this means the same two float32 vectors.

main.cpp
  Allocates host/device buffers, copies inputs, launches the kernel, and copies
  the output back.

launch.cpp
  Contains the low-level launch wrapper for the PTOAS-generated kernel.

compare.py
  Checks the output against the same reference used by the Triton Python test.

run_sim.sh
  Builds the host fixture and runs it through the CANN simulator.
```

For a new kernel, do not trust runtime comparison until these match:

```text
input shape
input dtype
input values
kernel arguments
launch grid / core count
output shape
correctness check
```

Example: for `vector_add_large.py`, the host fixture must use the same large
shape and the same vector-add reference. If the Python file uses a
`1000 x 2000` logical shape, the PTOAS fixture must allocate and compare that
same logical shape.

### 4.2. How To Run The PTOAS Flow

Before running the PTOAS lowering or PTOAS simulator fixture, activate the PTOAS
environment:

```bash
activate_ptoas
cd "$HOME/Planner"
```

This is needed because `ptoas` depends on its built Python extension and shared
library paths. If this is not active, `ptoas-lower` can fail with `ptoas not
found` or a Python `_core` import error.

The comparison wrapper sources `$CANN_ROOT/set_env.sh` and adds
`$CANN_ROOT/tools/bisheng_compiler/bin` to `PATH` for PTOAS lowering and PTOAS
simulator flows. Without that CANN setup, PTOAS may print:

```text
VPTO LLVM emission failed: unable to find 'bisheng' in PATH
VPTO LLVM emission: falling back to configured default target attributes
```

or the PTOAS simulator executable may fail to load CANN simulator libraries.

First, lower the PTOAS-input MLIR from section 3 to VPTO and VPTO LLVM IR:

```bash
bridge/tools/run_comparison_flow.sh ptoas-lower
```

For `vadd`, outputs:

```text
$OUT_ROOT/ptoas-only/vadd.vpto.mlir
$OUT_ROOT/ptoas-only/vadd.vpto.ll
```

Then run the PTOAS simulator fixture through `msprof op simulator`:

```bash
bridge/tools/run_comparison_flow.sh bridge-sim
```

The wrapper still uses the PTOAS host fixture for data generation and
correctness checking, but the generated PTOAS executable itself is launched as:

```text
msprof op simulator --application <ptoas-host-executable>
```

For larger fixtures this can take noticeably longer than the small `vadd`
example. For example, `vadd_large` uses a `1000 x 2000` logical shape and a
64-core PTOAS launch, so the simulator has much more work than the original
small vector-add fixture.

For `vadd`, outputs:

```text
$OUT_ROOT/bridge-sim/vadd/sim.log
$OUT_ROOT/bridge-sim/vadd/sim-command.txt
$OUT_ROOT/bridge-sim/vadd/profile/
$OUT_ROOT/ptoas-only/vadd/vadd.vpto.mlir is passed through KERNEL_MLIR
```

For `vadd_large`, replace `vadd` with `vadd_large`:

```text
$OUT_ROOT/bridge-sim/vadd_large/sim.log
```

Useful checks:

```bash
tail -f "$OUT_ROOT/bridge-sim/$TESTCASE/sim.log"
grep -E "Total tick|compare passed|compare failed|Model stopped|ERROR|error" \
  "$OUT_ROOT/bridge-sim/$TESTCASE/sim.log"
```

If `$OUT_ROOT/ptoas-only/vadd.vpto.mlir` exists, the simulator flow uses that
generated VPTO file. That means the normal section order is:

```text
section 2 early-ir
  -> section 3 bridge-ir creates vadd.vmi.mlir
  -> section 4.2 ptoas-lower creates vadd.vpto.mlir
  -> section 4.2 bridge-sim runs that generated vadd.vpto.mlir
```

Compare correctness first. Only compare timing after the host fixture in 4.1 is
known to be equivalent to the Triton Python host path.

For final performance, repeat the comparison on the A5 hardware server and
record:

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
