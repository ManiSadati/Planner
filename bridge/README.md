# Bridge

`bridge/` contains the working documentation, validation artifacts, testcases,
comparison structure, and cross-repo tools for the AscendNPU-IR-to-PTOAS path.

- `designs/`: design decisions and implementation constraints for the bridge.
- `memory/`: durable facts, repository findings, risks, and workflow notes.
- `planning/`: active plans, mapping tables, staged work, and exploration outputs.
- `testcases/`: representative NPU-IR inputs and PTOAS simulator fixtures.
- `tools/`: cross-repo scripts that run NPU-IR, extract VMI, invoke PTOAS, or
  compare the two paths.
- `validation/`: hand-authored expected outputs and validation oracles.

Repo-specific docs or scripts should live under `NPUIR/`, `PTOAS/`, or
`PTO-ISA/` instead of `bridge/`. `bridge/` should keep material that needs both
sides or exists to compare them.

The current practical scope is comparison infrastructure plus the first vector
and DMA bridge rows. Cube/template work remains planning-only until a mapping
or template-rewrite strategy is selected.

## Current Status

- Explorer bot is installed as a user systemd timer and runs daily at 7:00am Eastern.
- Latest configured-scope PTOAS report: `explorer/reports/README.md`.
- Latest big-change report: `explorer/reports/daily/2026-08-20.md`.
- Current priority: clean up the comparison structure so baseline NPU-IR and
  NPU-IR-to-PTOAS runs have comparable commands, targets, core counts,
  tick/cycle interpretation, and host behavior.
- Durable state summary: `bridge/memory/project-state.md`.
- A5/early-IR workflow memory: `NPUIR/coding-guide/a5-ir-workflow.md`.
- Codex-server NPU-IR build/replay memory: `NPUIR/coding-guide/codex-server-build.md`.
- Codex-server CANN operator simulator workflow: `NPUIR/coding-guide/simulator-workflow.md`.
- NPU-IR LLVM IR capture workflow: `NPUIR/coding-guide/llvm-ir-capture.md`.
- A5 full install/runtime note: `NPUIR/coding-guide/a5-installation.md`.
- Triton source fixtures: `bridge/triton-example/`.
- Generic baseline-vs-bridge comparison runbook:
  `bridge/comparison-flows.md`.
- Simple bridge runner: `bridge/tools/run_comparison_flow.sh`.
- Vector-add end-to-end workflow:
  `bridge/testcases/vadd/README.md`.
- A5-generated early IR dump folder: `bridge/examples/npuir-early-ir/`.
- Planning overview and roadmap: `bridge/planning/README.md`.
- NPU-IR device-spec replay plan: `NPUIR/coding-guide/device-spec-replay.md`.
- Latest NPU-IR replay result: `NPUIR/coding-guide/device-spec-replay-results-2026-08-11.md`.
- Active DMA-copy conversion exploration plan: `bridge/planning/dma-copy-conversion-exploration.md`.
- DMA template mapping memory: `bridge/memory/dma-template-mapping.md`.
- Upstream/fork watch list: `bridge/memory/upstream-watch.md`.
- Current mapping draft: `bridge/planning/npuir-to-ptoas-mapping.md`.
- DMA rewrite plan: `bridge/planning/dma-template-rewrite-plan.md`.

## Simulator And Hardware Reality

This server can run selected Triton/NPU-IR workloads through the CANN operator
simulator. The small and large vector-add kernels have both run successfully
through the local NPU-IR simulator path. These runs are useful for functional
checks, IR capture, and rough tick/cycle comparison.

Real hardware runtime and authoritative performance validation still require
the A5 server. When comparing NPU-IR and NPU-IR-to-PTOAS, keep simulator
numbers separate from A5 runtime numbers.

## NPU-IR to PTOAS Tool

`bridge/tools/run_comparison_flow.sh` is the simple bridge runner. It takes one
option and one testcase name. Add `--clean-build` when you want to remove the
testcase build directories before running:

```bash
cd "$HOME/Planner"

export ASCEND_NPU_IR_ROOT=/path/to/AscendNPU-IR
export CANN_ROOT=/path/to/CANN
export PTOAS_ROOT=/path/to/PTOAS

bridge/tools/run_comparison_flow.sh early-ir vadd
bridge/tools/run_comparison_flow.sh emit-vpto vadd
bridge/tools/run_comparison_flow.sh --clean-build bridge-sim vadd
```

The bridge defaults to `direct`, which now rewrites supported non-Cube DMA
templates only. Use `ptodsl` for the preferred PTO-visible Cube path:

```bash
bridge/tools/run_comparison_flow.sh \
  --bridge-mode ptodsl emit-vpto cube_dotproduct
bridge/tools/run_comparison_flow.sh \
  --bridge-mode ptodsl bridge-sim cube_dotproduct
```

Use `external-calls` for the Cube compatibility/reference route. It skips the
early structured Cube conversion, preserves selected CCE template calls and
their memrefs, and runs the later VMI conversion around them:

```bash
bridge/tools/run_comparison_flow.sh \
  --bridge-mode external-calls emit-vmi matmul_64
bridge/tools/run_comparison_flow.sh \
  --bridge-mode external-calls emit-vpto matmul_64
bridge/tools/run_comparison_flow.sh \
  --bridge-mode external-calls bridge-sim matmul_64
```

The `matmul_64` external-call path now passes end to end. Public kernel inputs
remain raw PTO pointers, CCE calls retain ranked memrefs, standard MLIR lowering
creates the C-interface descriptors, and external mode links NPU-IR's installed
`meta_op.aic.c310.bc` automatically. The simulator run passes all 4096 outputs
with maximum absolute error `0.001953125` and reports 3647 total ticks. No
shape-specific descriptor adapter is used.

The older `bridge/tools/run_npuir_ptoas_bridge_tests.sh` script is now only a
small wrapper around the same implementation.

### Options

| Option | Meaning |
|---|---|
| `early-ir` | Run the Triton Python testcase through the NPU-IR compile path and write `input.mlir` in the testcase directory. |
| `print-all` | Run `bishengir-compile --mlir-print-ir-after-all` from `input.mlir` and save the full log. |
| `npu-sim` | Run the Triton Python testcase end to end through the NPU-IR simulator. |
| `emit-vmi` | Run `bishengir-compile` with the PTOAS bridge enabled and extract PTOAS VMI MLIR. |
| `emit-vpto` | Run PTOAS on the VMI MLIR and emit VPTO MLIR. |
| `bridge-sim` | Run `input.mlir -> VMI -> VPTO -> run_sim.sh` for end-to-end PTOAS simulator testing. |

`--clean-build` can be passed before or after the option. It removes
`bridge/testcases/<name>/out/build/` and the legacy
`bridge/testcases/<name>/build/` directory, then rebuilds the selected flow.
`--bridge-mode direct` is the default and rewrites supported non-Cube HIVM DMA
templates. `--bridge-mode ptodsl` imports the Python-owned `MmadL1` PTO helper
and converts its separate ND2NZ/Fixpipe caller operations. `--bridge-mode
external-calls` skips the early Cube conversion and runs
`convert-hivmave-to-ptoas-vmi` after `convert-hivm-to-std` has emitted the CCE
calls.

### Testcase Layout

The testcase lives directly under `bridge/testcases/`:

```text
bridge/testcases/<name>/
  <one Python file containing one @triton.jit kernel>
  input.mlir
```

`early-ir` creates `input.mlir`. The other compile options use it. `bridge-sim`
requires the PTOAS simulator fixture files: `CMakeLists.txt`, `main.cpp`,
`launch.cpp`, `gen_data.py`, and `compare.py`. If those files exist and
`run_sim.sh` is missing, the script creates a simple `run_sim.sh` automatically.
If any fixture file is missing, `bridge-sim` stops and asks you to generate the
missing files first.

The script infers the Python file by finding the only `.py` file in the testcase
directory that contains `@triton.jit`. It infers the kernel name from the
function immediately after that decorator.

### Outputs

All generated files stay inside the testcase:

```text
bridge/testcases/<name>/input.mlir
bridge/testcases/<name>/out/<name>.vmi.mlir
bridge/testcases/<name>/out/<name>.vpto.mlir
bridge/testcases/<name>/out/after-all.log
bridge/testcases/<name>/out/build/
```

`out/build/` contains command files, compiler temporary files, simulator logs,
profiles, and fixture build products.

The configured PTOAS watcher scope is current as of 2026-08-19. GitHub
direct-fork/fork-of-fork discovery is implemented for configured GitHub repos.
GitCode issue/PR tracking is still pending.
