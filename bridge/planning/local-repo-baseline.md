# Local Repo Baseline

Last updated: 2026-08-07

Scope: Stage 1 of `bridge/planning/initial-exploration-plan.md`.

Status: historical baseline. Do not treat branch names or remotes in this file
as current without re-checking the local repos.

This baseline only records local repo state and where relevant source/docs appear to live. It does not resolve the NPU-IR-to-PTOAS mapping table yet.

## Summary

- PTOAS local repo is rich in VMI/VPTO/PTODSL design docs and has the clearest next PTOAS investigation path.
- PTOAS local repo must not be treated as source of truth: its `origin` remote is a personal fork, and `mani/fix_ptodsl` may be behind the active upstream/fork ecosystem.
- AscendNPU-IR local repo has explicit conversion directories for `HIVMToStandard` and `HIVMAVEToAVEIntrin`, plus docs already calling those out as important.
- PTO-ISA local repo is a tile programming library / virtual ISA surface, with docs and tests for tile instructions, events, sync, CPU simulation, NPU ST, and communication.
- All three local repos have uncommitted or untracked state. Treat existing changes as user-owned.

## Repo State

| Repo | Local path | Current branch | HEAD | Working tree | Remotes |
| --- | --- | --- | --- | --- | --- |
| PTOAS | `$HOME/PTOAS/PTOAS_Markham` | `mani/fix_ptodsl` tracking `origin/mani/fix_ptodsl`, ahead 770 | `b7208d14f` | modified `lib/TileOps/a5/tload.py`; untracked `build-clang21/`, `build-llvm21/`, `install-llvm21/`, `mani_log/` | `origin`, `upstream`, `zhendong` |
| AscendNPU-IR | `$HOME/AscendNPU-IR` | `mani/fuse-explore` tracking `origin/mani/fuse-explore` | `4254b5de` | clean | `origin` only, pointing to `git@gitcode.com:manisadati/AscendNPU-IR.git` |
| PTO-ISA | `$HOME/pto-isa` | `master` tracking `origin/master` | `896d8ec69` | modified `include/pto/npu/a5/TBinOp.hpp` | `origin`, pointing to `git@gitcode.com:cann/pto-isa.git` |

## PTOAS Baseline

### Primary Docs

- `README.md` / `README_en.md`: build, install, CLI, VPTO backend examples.
- `docs/build_with_installed_llvm.md`: installed LLVM21 VPTO build path.
- `docs/no_npu_compile_only_guide_zh.md`: compile-only validation flow for generated C++.
- `docs/vpto-spec.md`: VPTO source and backend semantics.
- `docs/ptoas-tile-fusion-design.md`: current tile fusion and VPTO backend boundary design.
- `docs/PTO_IR_manual.md`: PTO IR manual.
- `docs/isa/vmi-isa/`: VMI ISA references.
- `docs/isa/tile-op/`: tile op references.
- `docs/isa/micro-isa/`: micro instruction references.
- `docs/designs/`: many design docs, including VMI layout, tile fusion, memory, sync, and A5 work.
- `ptodsl/README.md`: PTODSL package, TileLib daemon/default expansion path, tests.
- `ptodsl/docs/user_guide/`: PTODSL user guide, including data movement, compute, predicates/masks, sync, and VMI.
- `ptodsl/docs/developer_guide/`: TileLib template authoring/debugging.

### Relevant Source Directories

- `include/PTO/IR`, `lib/PTO/IR`: PTO dialect / IR implementation.
- `include/PTO/Transforms`, `lib/PTO/Transforms`: compiler passes.
- `include/PTO/Transforms/GraphSyncSolver`, `lib/PTO/Transforms/GraphSyncSolver`: sync solver path.
- `include/PTO/Transforms/InsertSync`, `lib/PTO/Transforms/InsertSync`: sync insertion.
- `include/PTO/Transforms/TileFusion`, `lib/PTO/Transforms/TileFusion`: tile fusion.
- `lib/TileOps/a5`: A5 tile op template implementations.
- `ptodsl/ptodsl/tilelib`: PTODSL TileLib implementation.
- `tools/ptoas`: main CLI/tool driver.
- `tools/ptobc`: PTO bytecode tool.
- `test/lit`, `test/vpto`, `test/dsl-st`, `ptodsl/tests`: key regression surfaces.

### Build/Test Entry Points

- Source install: `./quick_install.sh` with `PTO_BUILD_DIR`.
- Build tree test: `ninja -C "$PTO_SOURCE_DIR/build" check-pto`.
- CLI examples:
  - `ptoas test/lit/pto/empty_func.pto`
  - `ptoas test/lit/pto/empty_func.pto --enable-insert-sync -o outputfile.cpp`
  - `ptoas test/lit/pto/empty_func.pto --pto-level=level3 -o outputfile.cpp`
  - `ptoas test/lit/vmi_new/vmi_ptoas_cli_pipeline.pto --pto-arch=a5 --pto-backend=vpto --emit-vpto -o -`
- PTODSL tests:
  - `python3 ptodsl/tests/test_jit_compile.py`
  - `python3 ptodsl/tests/test_docs_as_test.py`
  - `scripts/sim_dsl.sh test/dsl-st`
- NPU compile-only docs point to `test/npu_validation/scripts/generate_testcase.py` and `run_remote_npu_validation.sh`.

### Important PTOAS Leads For Stage 2

- Use this local checkout for available docs/builds, but cross-check design conclusions against upstream PTOAS and active fork networks, including forks-of-forks, branches, issues, and PRs.
- `docs/ptoas-tile-fusion-design.md` says the VPTO backend boundary is `ExpandTileOp`.
- The same design doc says old `View2Memref` / `PTOToA5VM` mainline was removed.
- `README.md` notes the VPTO backend enables a VMI to VPTO semantic pipeline and public function signatures should not expose `!pto.vmi.*`.
- `ptodsl/README.md` says PTOAS uses the PTODSL TileLib daemon by default for VPTO tile-op expansion.

## Testing Reality

- PTOAS can be built and run on this server.
- AscendNPU-IR can be edited on this server, but full A5 validation may require a separate A5 machine.
- For A5-dependent work, the expected loop is: Codex edits locally, the human runs on the A5 server, and the human returns logs/results for the next pass.

## AscendNPU-IR Baseline

### Primary Docs

- `README.md`: overview, repository layout, docs/build links.
- `docs_for_ai/NPU_IR_OVERVIEW.md`: existing local AI-oriented overview.
- `docs_for_ai/VFFUSION_AND_VECTORIZATION.md`: existing local AI-oriented vectorization/fusion context.
- `docs/source/en/introduction/architecture.md`: architecture doc.
- `docs/source/en/introduction/quick_start/installing_guide.md`: build/install/test commands.
- `docs/source/en/user_guide/compile_option.md`: `bishengir-compile` options.
- `docs/source/en/user_guide/debug_option.md`: debug options and HIVM snippets.
- `docs/source/en/developer_guide/dialects/HIVMDialect.md`: `hivm.hir.*` op definitions.
- `docs/source/en/developer_guide/passes/HIVMPasses.md`: HIVM pass docs.
- `docs/source/en/developer_guide/features/CV/CVOptimization.md`: cube/vector optimization notes.
- `docs/source/en/developer_guide/features/CVPipeline/CVPipelining.md`: cube/vector pipeline notes.
- `docs/source/en/developer_guide/conversion/framework_interface.md`: framework-to-HFusion/HIVM examples and commands.
- `docs/source/en/developer_guide/conversion/triton_interface.md`: Triton interface, including sync block use.

### Relevant Source Directories

- `bishengir/include/bishengir/Dialect/HIVM`, `bishengir/lib/Dialect/HIVM`: HIVM dialect and transforms.
- `bishengir/include/bishengir/Dialect/HIVMAVE`, `bishengir/lib/Dialect/HIVMAVE`: HIVMAVE dialect and transforms.
- `bishengir/lib/Conversion/HIVMToStandard`: candidate earlier interception point.
- `bishengir/lib/Conversion/HIVMAVEToAVEIntrin`: likely replacement boundary from human overview.
- `bishengir/lib/Conversion/HIVMAVEToStandard`: adjacent lowering path.
- `bishengir/lib/Conversion/VectorToHIVMAVE`: vector-to-HIVMAVE conversion; creates `pge` patterns.
- `bishengir/lib/Tools/bishengir-compile/regbase/PassPipeline.cpp`: regbase pass pipeline; observed includes and adds `HIVMToStandard` then `HIVMAVEToAVEIntrin`.
- `bishengir/test/bishengir-compile`, `bishengir/test/Dialect/HIVM`, `bishengir/test/Transforms`, `bishengir/test/Integration/HIVM`: likely smallest source-backed examples.

### Build/Test Entry Points

- First build: `./build-tools/build.sh -o ./build --build-type Release --apply-patches`.
- Subsequent build: `./build-tools/build.sh -o ./build --build-type Release`.
- Fast build: `./build-tools/build.sh -o ./build --fast-build`.
- Build with tests: `./build-tools/build.sh -o ./build --build-type Debug --apply-patches --build-test`.
- Test target: `cmake --build . --target "check-mlir;check-bishengir"`.
- Direct lit: `./bin/llvm-lit ../bishengir/test`.
- Pass-level tests generally document their own `bishengir-opt`/`FileCheck` RUN lines.

### Important NPU-IR Leads For Stage 3

- `docs_for_ai/NPU_IR_OVERVIEW.md` already calls out `HIVMToStandard` and `HIVMAVEToAVEIntrin`.
- `docs_for_ai/VFFUSION_AND_VECTORIZATION.md` references:
  - `bishengir/lib/Conversion/HIVMToStandard/HIVMToStandard.cpp`
  - `bishengir/lib/Conversion/HIVMToStandard/regbase/HIVMToStandard.cpp`
  - `bishengir/lib/Conversion/HIVMAVEToAVEIntrin/HIVMAVEToAVEIntrin.cpp`
- `docs/source/en/developer_guide/dialects/HIVMDialect.md` documents the exact ops needed later:
  - `hivm.hir.load`
  - `hivm.hir.store`
  - `hivm.hir.mmadL1`
  - `hivm.hir.nd2nz`
  - `hivm.hir.set_flag`
  - `hivm.hir.sync_block_set`
- `docs/source/en/developer_guide/features/DFX/DFX.md` contains snippets where `mmadL1`, `set_flag`, and `sync_block_set` survive into `ConvertHIVMToStandard`.
- `bishengir/lib/Tools/bishengir-compile/regbase/PassPipeline.cpp` is a key pipeline source for confirming exact pass order.

## PTO-ISA Baseline

### Local Instruction File

- `AGENTS.md` exists in `$HOME/pto-isa`; read it before doing coding work inside that repo.

### Primary Docs

- `README.md`: PTO Tile Library overview, target users, roadmap, and quick test commands.
- `docs/coding/README.md`: developer documentation index.
- `docs/coding/ProgrammingModel.md`: programming model.
- `docs/coding/Tile.md`: tile type/shape/mask/data organization.
- `docs/coding/Event.md`: event synchronization.
- `docs/coding/compilation-process.md`: developer-visible compilation flow.
- `docs/coding/cpu_sim.md`: CPU simulation.
- `docs/coding/operator-fusion.md`: public caveats around fusion.
- `docs/isa/`: instruction reference docs.
- `docs/isa/comm/README.md`: communication extension.
- `tests/README.md`: test entry points.
- `tests/script/README.md`: build/run scripts.

### Relevant Source Directories

- `include/pto/npu`: NPU tile instruction surface.
- `include/pto/npu/a2a3`, `include/pto/npu/a5`, `include/pto/npu/kirin9030`, `include/pto/npu/kirinX90`: architecture-specific implementations.
- `include/pto/common`: common tile/event/architecture support.
- `include/pto/comm`: communication primitives.
- `include/pto/cpu`: CPU simulation path.
- `kernels/manual`: manual kernels, including A2/A3 and A5 examples.
- `tests/cpu`, `tests/npu`, `tests/costmodel`: CPU, NPU, and cost model test surfaces.
- `agents/skills`: repo-provided coding/debug/performance/ISA skills for PTO-ISA work.

### Build/Test Entry Points

- Full CPU simulator: `python3 tests/run_cpu.py --clean --verbose`.
- Demos:
  - `python3 tests/run_cpu.py --demo gemm --verbose`
  - `python3 tests/run_cpu.py --demo flash_attn --verbose`
- Single ST testcase: `python3 tests/script/run_st.py -r [sim|npu] -v [a3|a5] -t [TEST_CASE] -g [GTEST_FILTER_CASE]`.
- One-click scripts:
  - `./tests/run_st.sh`
  - `./tests/run_cpu_tests.sh`
  - `./tests/run_comm_test.sh`
  - `./tests/run_costmodel_tests.sh`
- Build/run combined: `./build.sh --run_all --a3 --sim`.
- Wheel build: `python3 -m build --wheel`.

### Important PTO-ISA Leads For Stage 4

- `README.md` describes PTO as a virtual ISA for tile-oriented programming.
- `docs/coding/ProgrammingModel.md` says the compiler/runtime may choose memory placement and insert required synchronization in the productive/auto style.
- `docs/coding/Event.md` should be used to compare `TSYNC`/event abstractions with NPU-IR `set_flag`/`sync_block_*`.
- `docs/coding/compilation-process.md` intentionally frames internal compiler stages as implementation details unless documented elsewhere.
- `docs/coding/operator-fusion.md` cautions against treating undocumented automatic fusion as a stable public guarantee.

## Immediate Next Reads

For Stage 2, start with:

1. `$HOME/PTOAS/PTOAS_Markham/docs/ptoas-tile-fusion-design.md`
2. `$HOME/PTOAS/PTOAS_Markham/docs/vpto-spec.md`
3. `$HOME/PTOAS/PTOAS_Markham/ptodsl/README.md`
4. `$HOME/PTOAS/PTOAS_Markham/ptodsl/docs/user_guide/14-vmi-virtual-instruction-set.md`
5. `$HOME/PTOAS/PTOAS_Markham/docs/isa/vmi-isa/00-architecture-overview.md`

For Stage 3, start with:

1. `$HOME/AscendNPU-IR/docs_for_ai/NPU_IR_OVERVIEW.md`
2. `$HOME/AscendNPU-IR/docs_for_ai/VFFUSION_AND_VECTORIZATION.md`
3. `$HOME/AscendNPU-IR/bishengir/lib/Tools/bishengir-compile/regbase/PassPipeline.cpp`
4. `$HOME/AscendNPU-IR/bishengir/lib/Conversion/HIVMToStandard/`
5. `$HOME/AscendNPU-IR/bishengir/lib/Conversion/HIVMAVEToAVEIntrin/`
6. `$HOME/AscendNPU-IR/docs/source/en/developer_guide/dialects/HIVMDialect.md`

For Stage 4, start with:

1. `$HOME/pto-isa/AGENTS.md`
2. `$HOME/pto-isa/README.md`
3. `$HOME/pto-isa/docs/coding/ProgrammingModel.md`
4. `$HOME/pto-isa/docs/coding/Event.md`
5. `$HOME/pto-isa/docs/coding/compilation-process.md`
6. `$HOME/pto-isa/docs/isa/`
