# Bridge

`bridge/` contains the working documentation, validation artifacts, testcases,
and tools for converting AscendNPU-IR vector IR into PTOAS VMI and lowering it
through the PTOAS pipeline.

- `designs/`: design decisions and implementation constraints for the bridge.
- `memory/`: durable facts, repository findings, risks, and workflow notes.
- `planning/`: active plans, mapping tables, staged work, and exploration outputs.
- `testcases/`: representative NPU-IR inputs and PTOAS simulator fixtures.
- `tools/`: scripts that run NPU-IR, extract VMI, and invoke PTOAS.
- `validation/`: hand-authored expected outputs and validation oracles.

The primary scope is the vector-side bridge. Cube operations are outside the
current scope unless a planning document explicitly expands it.

## Current Status

- Explorer bot is installed as a user systemd timer and runs daily at 7:00am Eastern.
- Latest configured-scope PTOAS report: `explorer/reports/README.md`.
- Latest big-change report: `explorer/reports/daily/2026-08-11.md`.
- Durable state summary: `bridge/memory/project-state.md`.
- A5/early-IR workflow memory: `bridge/memory/a5-ir-workflow.md`.
- Codex-server NPU-IR build/replay memory: `bridge/memory/npuir-codex-server-build.md`.
- A5 full install/runtime note: `bridge/memory/npu_ir_installation.md`.
- Triton source fixtures: `bridge/triton-example/`.
- A5-generated early IR dump folder: `bridge/examples/npuir-early-ir/`.
- Planning overview and roadmap: `bridge/planning/README.md`.
- NPU-IR device-spec replay plan: `bridge/planning/npuir-device-spec-replay.md`.
- Latest NPU-IR replay result: `bridge/memory/npuir-device-spec-replay-results-2026-08-11.md`.
- Active DMA-copy conversion exploration plan: `bridge/planning/dma-copy-conversion-exploration.md`.
- DMA template mapping memory: `bridge/memory/dma-template-mapping.md`.
- Upstream/fork watch list: `bridge/memory/upstream-watch.md`.
- Current mapping draft: `bridge/planning/npuir-to-ptoas-mapping.md`.
- DMA rewrite plan: `bridge/planning/dma-template-rewrite-plan.md`.

## NPU-IR to PTOAS Test Runner

`bridge/tools/run_npuir_ptoas_bridge_tests.sh` runs bridge testcases through
AscendNPU-IR and PTOAS:

```text
AVE/NPU-IR MLIR -> HIVMAVEToPTOASVMI -> PTOAS VMI -> VPTO or VPTO LLVM IR
```

The script assumes the repositories are siblings under the same workspace:

```text
workspace/
  AscendNPU-IR/
  PTOAS/
  Planner/
```

By default it uses:

- `AscendNPU-IR/build/bin/bishengir-opt`
- `AscendNPU-IR/build/bin/bishengir-compile`
- `PTOAS/build/tools/ptoas/ptoas`

Override these paths with the corresponding command-line options or environment
variables described below.

### Input modes

The default mode runs `input.mlir` with `bishengir-opt` and explicitly invokes
the bridge pass:

```bash
cd /home/m00967009/Workspace/Planner
bridge/tools/run_npuir_ptoas_bridge_tests.sh --emit-vpto vadd
```

Use `--from-bishengir-compile` to start from a higher-level
`compile-input.mlir`:

```bash
bridge/tools/run_npuir_ptoas_bridge_tests.sh \
  --from-bishengir-compile \
  --emit-vpto \
  vadd
```

In this mode the script passes
`--mlir-print-ir-after=convert-hivmave-to-ptoas-vmi` to
`bishengir-compile`, extracts the last successful dump for that pass, and uses
that dump as the PTOAS input. Dumps marked `Failed` are ignored. This makes a
successful bridge dump available even when a later compiler stage fails.

### Emission and simulator modes

Emit VPTO MLIR:

```bash
bridge/tools/run_npuir_ptoas_bridge_tests.sh --emit-vpto vadd
```

Emit VPTO LLVM IR:

```bash
bridge/tools/run_npuir_ptoas_bridge_tests.sh --emit-llvmir vadd
```

Run the testcase simulator. This automatically enables VPTO emission:

```bash
bridge/tools/run_npuir_ptoas_bridge_tests.sh \
  --from-bishengir-compile \
  --clean \
  --run-simulator \
  vadd
```

Run all available outputs and the simulator:

```bash
bridge/tools/run_npuir_ptoas_bridge_tests.sh --all vadd
```

Simulator mode copies the testcase into the output directory, replaces its
`lowered_vector_add_kernel_vpto.mlir` with the newly generated VPTO file, and
runs the testcase's `run_sim.sh`. The simulator fixture requires a sourced CANN
environment with `ASCEND_HOME_PATH` set, a usable `bisheng` compiler, and the
matching simulator libraries. The fixture-specific environment variables are:

```bash
export ASCEND_HOME_PATH=/path/to/cann
export SOC_VERSION=Ascend950PR_9599
export SIM_LIB_DIR="$ASCEND_HOME_PATH/tools/simulator/$SOC_VERSION/lib"
export BUILD_JOBS=16
```

### Options

| Option | Meaning |
|---|---|
| `--from-bishengir-compile` | Use `compile-input.mlir` and extract the VMI pass dump from `bishengir-compile`. |
| `--emit-vpto` | Ask PTOAS to emit VPTO MLIR. |
| `--emit-llvmir` | Ask PTOAS to emit VPTO LLVM IR. |
| `--run-simulator`, `--sim` | Run the testcase simulator after VPTO emission. |
| `--all` | Enable VPTO, LLVM IR, and simulator execution. |
| `--clean`, `--clean-build` | Remove the selected testcase output directory before running. |
| `--testcase-root DIR` | Override the testcase root. |
| `--output-root DIR` | Override the output root. |
| `--bishengir-opt PATH` | Override the `bishengir-opt` executable. |
| `--bishengir-compile PATH` | Override the `bishengir-compile` executable. |
| `--ptoas PATH` | Override the PTOAS executable. |
| `--npu-target TARGET` | Set the AscendNPU-IR compile target; default `Ascend910_9589`. |
| `--pto-arch ARCH` | Set the PTOAS architecture; default `a5`. |
| `--pto-backend BACKEND` | Set the PTOAS backend; default `vpto`. |
| `-h`, `--help` | Print command help. |

If no emission or execution option is specified, the script defaults to
`--emit-vpto`. If no testcase name is specified, it runs every direct child of
the testcase root containing `input.mlir` (or `compile-input.mlir` in compile
mode). Multiple testcase names may be supplied.

The equivalent environment overrides are `TESTCASE_ROOT`, `OUTPUT_ROOT`,
`BISHENGIR_OPT`, `BISHENGIR_COMPILE`, `PTOAS_BIN`, `NPU_TARGET`, `PTO_ARCH`,
`PTO_BACKEND`, and `EXTRA_BISHENGIR_COMPILE_FLAGS`. Terminal status messages
use colors when connected to a terminal. Set `NO_COLOR=1` to disable ANSI
colors or `FORCE_COLOR=1` to force them when output is redirected.

### Testcase layout

A direct-input testcase minimally contains:

```text
bridge/testcases/<name>/
  input.mlir
```

A compile-driver testcase contains:

```text
bridge/testcases/<name>/
  compile-input.mlir
```

To support simulator mode, it also needs a simulator fixture such as
`run_sim.sh`, `CMakeLists.txt`, host-side launch code, and data/verification
scripts. The current `vadd` fixture is based on the PTOAS lowered-vector-add
test and is the reference layout.

### Outputs

The default output directory is:

```text
bridge/out/npuir-ptoas-bridge/<name>/
```

Important files include:

- `<name>.vmi.mlir`: VMI produced by the bridge pass or extracted from the
  compile-driver log.
- `<name>.vpto.mlir`: VPTO emitted by PTOAS.
- `<name>.vpto.ll`: VPTO LLVM IR emitted by PTOAS.
- `npuir.log`: AscendNPU-IR output and diagnostics.
- `ptoas-vpto.log` / `ptoas-llvmir.log`: PTOAS diagnostics.
- `sim.log`: simulator and numerical-check output.
- `npuir-command.txt`, `ptoas-*-command.txt`, `sim-command.txt`: reproducible
  commands used for each stage.
- `temps/`: compiler temporary files when compile-driver mode is used.

Generated output under `bridge/out/` is ignored by Git. Use `--clean` when a
fresh testcase simulator build or output directory is needed.

The configured PTOAS watcher scope is current as of 2026-08-11. GitHub
direct-fork/fork-of-fork discovery is implemented for configured GitHub repos.
GitCode issue/PR tracking is still pending.
