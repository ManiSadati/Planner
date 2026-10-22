# Triton-Ascend 3.6 Shared A5 Compilation Handoff

Recorded: 2026-10-22. Scope: the user-approved minimal driver refactor,
isolated installation, and bounded native/PTOAS simulator checks. This record
supersedes the pending driver follow-up in the
[MmadL1 unit-flag record](mmadl1-unit-flags-2026-10-21.md), not its template
coverage or remaining numerical gates.

## Decisions And Implementation

The user requested that native and bridge compilation share the existing native
function and option construction, minimizing the diff and deferring naming
cleanup. Native remains the default. Preserve PTOAS binary generation and
necessary launcher/resource handling. No template algorithms, per-template
adapters, uniform-ABI policy, compiler sources, or submodule pins were changed.

Repository: `$HOME/triton-ascend`, origin
`git@github.com:Melikano/triton-ascend.git`.

| Revision | Meaning |
| --- | --- |
| `e9f9b169f8ffd6dda35a6853135c83d7e66df25f` | Native 3.6 frontend base used to build the candidate |
| `e701011bda3ceb2e94f1b3905084dc2e34b97424` | Melika's original 3.6 bridge; its separate bridge builder omitted native options, including `unit_flag` |
| `461aa8f086c6bf26229ee324b8f4b5d3902e002d` | Published refactor commit, `Share native A5 compilation with PTOAS VMI output` |

Local branch: `feature/shared-a5-native-ptoas`, clean at handoff. Relative to
Melika's bridge, this commit changes only:

- `third_party/ascend/backend/compiler.py`
- `third_party/ascend/unittest/pytest_ut/test_ptoas_compile_flow.py`

With the user's approval, removed the duplicated `linalg_to_ptoas_vmi()` and
`_build_ptoas_vmi_compile_options()`. Both flows now call
`linalg_to_bin_enable_npu_compile_910_95(..., emit_ptoas_vmi=False)` with the
native option block retained. The bridge passes `emit_ptoas_vmi=True` and adds
only `--emit-ptoas-vmi`, `--enable-hivm-compile=true`, and
`--mlir-disable-threading` to those NPU-IR options, with a textual VMI output
destination. Executable selection, metadata preparation, diagnostics and UB
metadata extraction are shared. VMI returns before native object/callback
reading; native object handling and callbacks retain their original behavior.

The separate `ptoas_vmi_to_npubin()` step remains, invoking PTOAS with
`--pto-backend=vpto --pto-arch=a5 --emit-device-object`. Bridge launcher defaults,
lock/resource guards, pure-SIMT rejection, and debug stage wrapping remain.
Both flows share frontend stages through `bcmlir`; the bridge adds `ptovmi`
before its final `npubin` stage.

## Shared Source Baseline

Published on 2026-10-22 after repository write access was granted:
[`Melikano/triton-ascend:feature/shared-a5-native-ptoas`](https://github.com/Melikano/triton-ascend/tree/feature/shared-a5-native-ptoas),
validated at [`461aa8f086c6bf26229ee324b8f4b5d3902e002d`](https://github.com/Melikano/triton-ascend/commit/461aa8f086c6bf26229ee324b8f4b5d3902e002d).
The local branch now tracks the same-named remote branch. The earlier push-access
blocker is resolved; this is a published feature branch, not a claim of merging
into Melika's original bridge branch or upstream.

Colleagues should use a compatible installation from this source baseline and
record the installed revision. A stock 3.6 package or a similarly named venv is
not proof of the patch. The preferred candidate below exists on this server
only; do not copy its account paths into other setups. If the installation is
missing, report the gap and propose an authorized installer task.

For an existing clone of Melika's fork, fetch the branch without switching or
overwriting another worker's checkout:

```bash
git fetch origin feature/shared-a5-native-ptoas
git show --no-patch --oneline 461aa8f086c6bf26229ee324b8f4b5d3902e002d
```

For authorized installation, use an isolated checkout at the recorded commit
and follow that revision's build requirements. This document records a native
wheel plus an ABI-compatible three-file Python overlay, not a portable prebuilt
wheel or an automatic installer. Inspect the installed base/extension before
applying those files; do not overlay 3.6 source onto a 3.2.x installation.
A local commit patch remains at
`$HOME/tmp/triton36-install-20261022/shared-a5-native-ptoas.patch` (refactor only;
the published branch already includes its Melika bridge dependencies).

## Installed State And Rollback

New isolated interpreter: `$HOME/.venv/npuir-sim-3.6-candidate/bin/python`,
Python 3.10.12. Installed distributions: `triton-ascend==3.6.0+gite9f9b169`,
`triton==3.6.0`, `torch==2.7.1+cpu`, `torch_npu==2.7.1.post8`, and
`numpy==1.26.4`. Imported Triton reports 3.6.0; backend Python and `libtriton.so`
resolve inside this candidate, without an editable install or frontend shims.
`pip check` passed.

Built a native wheel first, tested native compilation, then overlaid the exact
branch versions of backend `__init__.py`, `compiler.py`, and `utils.py` after
checking pristine installed bytes against the native base. These three files
include Melika's bridge plus the shared-path refactor. There is no binary/C++
delta from the base requiring an extension rebuild. **The installed wheel
metadata still names the native base; this is a recorded local Python patch,
not a published wheel built from the branch.** Use the patch manifest and file
hashes to identify it, rather than the version string alone.

Build dependencies: `pybind11==3.1.0`, pinned LLVM artifact
`llvm-f6ded0be-4ca23101-ubuntu-x64.tar.gz` (LLVM commit
`f6ded0be897e2878612dd903f7e8bb85448269e5`), and nlohmann/json 3.11.3.
The native frontend's embedded NPU-IR pin is
`aea934a66646e837c54fea11e87db54d42eb3221`. An initial wheel built with pybind11
2.13.6 failed import with duplicate `PADDING_OPTION` registration; rebuilding
with 3.1.0 resolved this without source changes.

All task artifacts are under `$HOME/tmp/triton36-install-20261022/`:

- `verification.json`: results, actual imported paths, emitted arguments and
  before/after comparisons.
- `native-installed.json`, `patched-installed.json`, and
  `installed-bridge-patch.json`: installed provenance and exact patch hashes.
- `tool-manifest-before.json` and `llvm-artifact.json`: tested tool hashes/pins.
- `build-native.sh`, `build-native.log`, and `build-native-pybind2.log`:
  reproducible build settings and the earlier build attempt.
- `wheels/triton_ascend-3.6.0+gite9f9b169-cp310-cp310-linux_x86_64.whl`:
  pristine native wheel; `native-backend-backup/`: original three Python files.
- `apply-bridge.py`, `inspect_candidate.py`, and `installed-requirements.txt`:
  patch application, import inspection and package snapshot.

The original `$HOME/.venv/npuir-sim-3.2.2` was preserved; its imported Triton
version is 3.2.0 despite `triton-ascend==3.2.2`. Its recorded backend/extension
hashes were unchanged after the task. The shell's default Python remains the
Miniconda Python 3.13.13, which has no Triton. No default interpreter or global
shell configuration was changed. Normal rollback is selecting the original
interpreter. Candidate-only rollback can restore the backed-up files or
reinstall the saved native wheel, which removes the candidate's bridge patch.

## Preferred Local Use

Project-owner decision: use this candidate for new local native and bridge tests.
Its shared A5 compiler path avoids the previous partial option-builder copy,
including the missing unit-flag forwarding. Keep kernels backend-neutral.
This preference applies to this server; do not assume this venv exists elsewhere.

Use `$HOME/.venv/npuir-sim-3.6-candidate/bin/python` explicitly. Configure the
existing CANN installation through `$CANN_ROOT/set_env.sh`, and resolve tools as
follows; this does not install or update anything:

```bash
source "$CANN_ROOT/set_env.sh"
source "$HOME/.venv/npuir-sim-3.6-candidate/bin/activate"
export ASCEND_NPU_IR_ROOT="$HOME/AscendNPU-IR"
export TRITON_NPU_COMPILER_PATH="$ASCEND_NPU_IR_ROOT/build/install/bin/bishengir-compile"
export TRITON_PTOAS_PATH="$ASCEND_NPU_IR_ROOT/build-ptoas/install/bin/ptoas"
export PTOAS_AICORE_LL_MODULE="$ASCEND_NPU_IR_ROOT/build/install/lib/meta_op.aic.c310.bc"
export PTOAS_AIVECTOR_LL_MODULE="$ASCEND_NPU_IR_ROOT/build/install/lib/meta_op.aiv.c310.bc"
export PATH="$ASCEND_NPU_IR_ROOT/build/install/bin:$ASCEND_NPU_IR_ROOT/build-ptoas/install/bin:$CANN_ROOT/tools/bishengir/bin:$CANN_ROOT/tools/bisheng_compiler/bin:$CANN_ROOT/tools/profiler/bin:$PATH"
export LD_LIBRARY_PATH="$ASCEND_NPU_IR_ROOT/build-ptoas-llvm/lib:$ASCEND_NPU_IR_ROOT/build-ptoas/lib:${LD_LIBRARY_PATH:-}"
export PYTHONNOUSERSITE=1
export TRITON_BISHENGIR_DISABLE_LIB_CALL_NOINLINE=1
export TRITON_SIMULATOR_CLEAN_EXIT=1
unset PYTHONPATH TRITON_ASCEND_ARCH TRITON_DISABLE_FFTS
```

Then select native `TRITON_ASCEND_COMPILE_FLOW=npuir` with
`BISHENGIR_ENABLE_PTOAS_BRIDGE=0` and `FALLBACK=0`, or bridge
`TRITON_ASCEND_COMPILE_FLOW=ptoas` with `BISHENGIR_ENABLE_PTOAS_BRIDGE=1`,
`BISHENGIR_PTOAS_BRIDGE_MODE=ptodsl` and `FALLBACK=1` for dependency fallback.
Unset the bridge-mode variable for native checks. Use fresh private caches/logs,
the testcase's explicit filters, and `msprof op simulator` with
`--soc-version=Ascend950PR_9589`; verify the compiler target in the actual log.
The adjacent evidence runners record the full per-run setup, including the
existing no-inline compatibility setting and clean simulator exit.

Keep the old venv for rollback. Coordinate and obtain approval before modifying
or reinstalling this shared candidate or replacing shared compiler binaries.
Report a failure instead of silently selecting the old venv or hot-patching tools.
The wheel version alone does not identify the installed Python overlay.

## External Toolchain Provenance

Simulation used the existing external tools, without rebuilding or replacing
them. The source checkout `$HOME/AscendNPU-IR` was on
`mani/mmadl1-scratch-state` at `58f887f7688ba26e17ea02548558113abb702c19`.
Its Ascend LLVM pin was `8ce9217f83f3376970d10428de4e2d1c15039e08`, PTOAS pin
`d36100d03255e5e8a858c497a4277708d8b889c9`, and PTO LLVM pin
`fa2fd1f75d742fec91ecf40d98c9c7961647ec42`. These are separate from the
frontend wheel's LLVM and embedded NPU-IR pin. Textual MLIR crosses toolchains;
the two external LLVM forks are not linked together.

`build/install/bin/bishengir-compile --version` embeds `75525e826a7b`, not the
source checkout HEAD, and reports LLVM `8ce9217f83f3`. Follow-up review matched
its SHA256 exactly to `$HOME/tmp/mmadl1-unit-flags.Rovzfe/installed.sha256`.
That earlier report records building the dirty scratch-state/unit-flag changes
before committing them as `58f887f7`; the older stamp alone is not evidence of
a stale build. This hash match is provenance evidence, not a fresh clean rebuild.
Its SHA256 is
`765cf8da946a15272dceb4a37f789394c783590814102f9bd49331845668fd8c`;
the bytecode reader, PTOAS wrapper and native AIC/AIV fallback bitcode hashes
are in the manifest. Tested shared tool hashes were unchanged after the task.
PTOAS was `$HOME/AscendNPU-IR/build-ptoas/install/bin/ptoas` (reports 0.60).
The runner supplies the existing CANN 9.1.0-beta.3 installation as `$CANN_ROOT`.

## Verification And Limits

All simulator runs used the existing `vector_add_kernel` in
`$HOME/AscendNPU-IR/bishengir/test/Integration/PTOASBridge/testcases/vadd/vector_add.py`:
256 float32 elements, tile 256, grid `(1,)`, profiler core 0. Both compiler and
simulator explicitly selected **`Ascend950PR_9589`**. The 3.6 target comes from
`GPUTarget.arch`; deprecated `TRITON_ASCEND_ARCH` was unset.

| Installed state / flow | Evidence under task root | Result |
| --- | --- | --- |
| Native base, before bridge patch | `native-baseline-ipc/run.log` | allclose true, maximum error 0.0 |
| Patched candidate, native `npuir` | `native-patched/run.log` | allclose true, maximum error 0.0 |
| Patched candidate, bridge `ptoas` | `bridge-patched/run.log` | allclose true, maximum error 0.0; VMI vadd and PTOAS invocation confirmed |

Actual native argument lists matched before/after patching; the bridge list
matched native plus the three intentional options above. The bridge smoke used
`BISHENGIR_PTOAS_BRIDGE_MODE=ptodsl` and `FALLBACK=1`. This establishes bounded
bridge execution, not no-fallback coverage or new PTODSL template support.

`installed-tests.log` records **46 passed** using the actual installed 3.6
extension: 22 focused committed tests and 24 native differential cases against
Melika's original function, comparing complete arguments, result, metadata and
debug output. Command tests cover explicit None/False/True and numeric zero,
including unit-flag forwarding, guards and errors. Subprocess/device behavior
is mocked in these tests. Historical differential test source is at
`$HOME/tmp/triton-shared-a5-review/test_native_baseline.py`.

The initial sandboxed simulator attempt failed `aclInit` with code 507008 and
a child failure despite profiler exit zero (`native-baseline/run.log`). The
three successful runs used approved execution outside the Codex sandbox for
CANN local IPC; child success and numerical output were checked too. The
kernel's existing clean-exit hook was used. No target substitution or kernel
change was introduced.

Follow-up checks with this installed 3.6 candidate completed four additional
simulator runs, all with exact comparisons (`atol=0`, `rtol=0`):

| Test configuration | Native NPU-IR | PTODSL bridge |
| --- | --- | --- |
| F16 NN MmadL1, M/N/K=16/16/16, unit flags enabled, three rounds and two launches | PASS | PASS |
| F32 NZ2ND-to-UB Fixpipe, M/N/K=13/17/9, blocks 16/16/16, grid=2 | PASS | PASS |

Native IR confirmed `mma_tile_half_to_float` and
`fixpipe_nz2nd_float_to_float_4d_to_2d_ubuf`; bridge IR confirmed their exact
`__pto_` calls and materialized helper bodies, not native target fallback.
The unit-flag run forwarded `--enable-hivm-unit-flag-sync=True` in both flows,
with native flag 3 and PTO MAD check-and-set/Fixpipe check-and-clear operations.
All child tests printed PASS; all four profiler processes exited zero.
Evidence and exact runner: `$HOME/tmp/triton36-template-smoke.uoO1B5/REPORT.md`
and `run.sh`; each flow's directory retains logs, selection IR and fresh caches.
The shared compiler and backend hashes were unchanged; no source or install
changes were needed.

No full 3.6 MmadL1/Fixpipe matrix, internal split-K follow-up, full model,
physical A5, performance, or comprehensive template-suite result is claimed.
Preserve the remaining gates in the MmadL1 record and the current per-template
adapter policy.

## Next Agent Brief

Objective: use the preferred local frontend and preserve its recorded
provenance while expanding only assigned coverage. Own integration and
documentation reconciliation; coordinate before touching environments/builds
that another agent owns. Read the manifests and diff the two changed frontend
files at `461aa8f0`. Retain the compiler hash/build-before-commit explanation
instead of silently replacing shared tools. Use the published source baseline
above for colleague installation/handoffs; source publication does not distribute
the locally built wheel or validate a different host's package/extension ABI.

If a fresh smoke is needed after a change, use fresh labels with the recorded
runner (requires CANN local IPC access):

```bash
bash "$HOME/tmp/triton36-install-20261022/run-vadd.sh" npuir sync-native-new
bash "$HOME/tmp/triton36-install-20261022/run-vadd.sh" ptoas sync-bridge-new
```

Recheck imported paths, patch hashes, actual SoC, emitted arguments, child
success and numerical output. Reproducing the pristine native baseline requires
the stored wheel/backup in a separate environment; the runner now selects the
patched candidate. Do not replace the working 3.2.2 environment, modify shared
compiler binaries/pins, implement templates, revive uniform-ABI work, or fold
unrelated pending Planner edits into this handoff. The original installer
handoff was local and uncommitted. The Triton source branch is now published;
the locally built wheel and runtime evidence remain server-local artifacts.
