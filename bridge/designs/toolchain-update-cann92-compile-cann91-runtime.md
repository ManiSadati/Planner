# Updating NPU-IR, PTOAS and Triton Ascend: CANN 9.2 Compilation / 9.1 Runtime

Validated snapshot: 2026-10-28, Linux/Bash, simulator **Ascend950PR_9589**.
This guide collects the working update procedure and the failures encountered
while updating the existing Triton-Ascend 3.6 candidate. The
[installation record](../memory/toolchain-refresh-cann92-2026-10-28.md) contains
the server-specific locations, evidence filenames and rollback state.

The working arrangement uses **CANN 9.2 in compiler subprocesses** and the
previous **CANN 9.1.0-beta.3 for Torch-NPU and the simulator**. No implementation
patch was needed in NPU-IR, PTOAS or Triton. Changes were source/pin updates,
normal build configuration, and local compiler wrappers/environment wiring.
This is the tested version combination; it is not a claim that every CANN,
Torch or hardware combination supports the same split.

On the already updated server, normal activation of
`$HOME/.venv/npuir-sim-3.6-candidate/bin/activate` selects the working profile.
The commands below are for reproducing the update in dedicated locations;
they are not a request to replace that installation.

## 1. Freeze The Revisions And Preserve The Working Setup

| Component | Validated revision / version |
| --- | --- |
| [Wilson NPU-IR master](https://gitcode.com/wilsoncxfeng/AscendNPU-IR/tree/master) | `3bf5d9898e0c45447c81780bc35de7548f5cef99`; compiler 2.0.0 |
| Ascend LLVM submodule | `95c4693447af2473be62bdee19522cbe596f9e96`; LLVM 19.1.7 |
| PTOAS submodule | `5ce40f710f7f83b8d1fb2d8c36647bb811236a49`; CLI 0.69 |
| PTO LLVM submodule | `fa2fd1f75d742fec91ecf40d98c9c7961647ec42` |
| [Shared Triton 3.6 bridge](https://github.com/Melikano/triton-ascend/tree/feature/shared-a5-native-ptoas) | Python overlay `461aa8f086c6bf26229ee324b8f4b5d3902e002d` |
| Candidate packages | Python 3.10.12; Triton 3.6.0; triton-ascend 3.6.0+gite9f9b169; Torch 2.7.1+cpu; torch_npu 2.7.1.post8 |

Record `git status`, remotes, HEAD and submodule revisions before updating.
A branch checkout does not rebuild installed tools. Both LLVM forks can report
19.1.7 while having different APIs: record their commits, not just versions.

Keep an existing working venv/install for rollback. Use a clean dedicated
checkout/worktree and a new install prefix. On the validated server, local
`master` was fast-forwarded in a separate worktree while another agent's active
checkout and dirty PTOAS edits were preserved. Do not switch or reset someone
else's checkout, update their submodules, or rebuild a shared prefix in use.

Use NPU-IR's pinned PTOAS and LLVM sources. NPU-IR embeds PTO dialect code using
Ascend LLVM; standalone PTOAS uses PTO LLVM. Keep textual MLIR as their boundary;
do not link the two LLVM forks into one compiler.

## 2. Choose Paths And Check Access

Set these to the existing toolkits and your intended source/build/environment
locations. These variables are also used by the wrapper example below.

```bash
export CANN_COMPILER_ROOT=/path/to/cann-9.2.0
export CANN_RUNTIME_ROOT=/path/to/cann-9.1.0-beta.3
export NPUIR_SRC="$HOME/worktrees/AscendNPU-IR-update"
export TOOLCHAIN_PREFIX="$HOME/.local/opt/ascend-toolchain-3bf5d9898"
export CANDIDATE_VENV="$HOME/.venv/npuir-sim-3.6-candidate"
export PTOAS_LLVM_BUILD_DIR="$NPUIR_SRC/build-ptoas-llvm"
export HOST_CC=/usr/bin/clang
export HOST_CXX=/usr/bin/clang++
export CCACHE_TEMPDIR="$HOME/tmp/ascend-update-ccache"
mkdir -p "$CCACHE_TEMPDIR"

test -r "$CANN_COMPILER_ROOT/set_env.sh"
test -x "$CANN_COMPILER_ROOT/tools/bisheng_compiler/bin/ccec"
"$CANN_COMPILER_ROOT/tools/bisheng_compiler/bin/ccec" --version
"$CANDIDATE_VENV/bin/python" --version
"$HOST_CC" --version
cmake --version
ninja --version
```

Reading the toolkit's top directory is insufficient: setup scripts, nested
directories, compiler binaries, headers and libraries must be accessible.
Use `namei -l` on the failing path. The owner/admin can grant the intended
account recursive read/traverse and appropriate execute access, for example:

```bash
# Owner/admin command; replace account and toolkit path.
setfacl -R -m u:ACCOUNT:rX /path/to/cann-9.2.0
```

The validated NPU-IR host build used system Clang 14 with LLD; device compilation
used CANN 9.2's Bisheng compiler. A host Clang 15 path is another build choice,
not the Ascend LLVM source pin. Record the actual host compilers used.
This source requires CMake >= 3.28 and Ninja >= 1.12.

## 3. Obtain A Clean Matching Source Tree

For a new dedicated clone, the following freezes the validated snapshot. Use
your existing authenticated Git access. Check that the destination is unused
before cloning; an existing shared checkout needs a coordinated worktree instead.

```bash
test ! -e "$NPUIR_SRC"
mkdir -p "$(dirname "$NPUIR_SRC")"
git clone --no-recurse-submodules \
  git@gitcode.com:wilsoncxfeng/AscendNPU-IR.git "$NPUIR_SRC"
git -C "$NPUIR_SRC" checkout --detach 3bf5d9898e0c45447c81780bc35de7548f5cef99
git -C "$NPUIR_SRC" submodule update --init \
  third-party/llvm-project third-party/ptoas \
  third-party/ptoas-llvm-project third-party/shmem
git -C "$NPUIR_SRC" status --short
git -C "$NPUIR_SRC" submodule status
```

Compare all four commits with the table. If updating to a newer master instead,
record its new dependency set and revalidate it; these results do not establish
coverage for later commits. Do not cherry-pick SET2D or deferred HOST branches
to perform this update.

## 4. Build NPU-IR And Its Ascend LLVM Together

Run compiler setup inside a subshell so the parent runtime stays unchanged.
Use a fresh build directory, or explicitly reconfigure an existing cache as
described below. Lower parallelism to suit the machine; this server used 64
build jobs and two parallel host link jobs.

```bash
(
  set -e
  unset LD_PRELOAD LD_LIBRARY_PATH PYTHONPATH PYTHONHOME
  unset ASCEND_HOME_PATH ASCEND_OPP_PATH ASCEND_AICPU_PATH ASCEND_TOOLKIT_HOME
  export CANN_ROOT="$CANN_COMPILER_ROOT"
  set +u
  source "$CANN_ROOT/set_env.sh"
  export PATH="$CANN_ROOT/tools/bisheng_compiler/bin:$PATH"
  cd "$NPUIR_SRC"
  bash build-tools/build.sh \
    --build "$NPUIR_SRC/build" --build-type Release --jobs 64 \
    --install-prefix "$TOOLCHAIN_PREFIX/npuir" \
    --c-compiler "$HOST_CC" --cxx-compiler "$HOST_CXX" \
    --build-bishengir-template \
    --bisheng-compiler "$CANN_ROOT/tools/bisheng_compiler/bin" \
    --enable-lld \
    --add-cmake-options '-DBISHENGIR_BUILD_HIVMC=ON -DLLVM_PARALLEL_LINK_JOBS=2'
  cmake --build build --parallel 32 --target \
    mlir-translate llvm-dis llvm-link FileCheck count not split-file
  cmake --install build --prefix "$TOOLCHAIN_PREFIX/npuir"
)
```

`build.sh` can skip CMake generation when a cache already exists. Merely passing
a new compiler path on its command line may therefore leave old settings active.
For an existing tree with unchanged host compilers, explicitly reconfigure
inside the CANN 9.2 subshell before building:

```bash
cmake -S "$NPUIR_SRC/third-party/llvm-project/llvm" -B "$NPUIR_SRC/build" \
  -DBISHENG_COMPILER_PATH="$CANN_COMPILER_ROOT/tools/bisheng_compiler/bin" \
  -DBISHENG_COMPILER_EXECUTABLE="$CANN_COMPILER_ROOT/tools/bisheng_compiler/bin/bisheng" \
  -DBISHENGIR_BUILD_HIVMC=ON
```

Use a new build tree when changing host compilers or build ABI. Avoid a rebuild
flag against another worker's existing tree. Keep internal HIVMC enabled:
current native NPU-IR links its implementation even though CANN also ships an
external `hivmc-a5` executable.

Verify installed compiler banners and resources, not just build-tree results:

```bash
"$TOOLCHAIN_PREFIX/npuir/bin/bishengir-compile" --version
"$TOOLCHAIN_PREFIX/npuir/bin/bishengir-opt" --version
"$TOOLCHAIN_PREFIX/npuir/bin/llvm-config" --version
ls "$TOOLCHAIN_PREFIX/npuir/lib/"meta_op.*.c310.O[02].bc
ls "$TOOLCHAIN_PREFIX/npuir/lib/bishengir/ptodsl/"
```

Native bundles now have `.O0.bc` and `.O2.bc` suffixes. For the tested CANN 9.2
compiler path the default selects O2. Install the compiler's matching resources;
do not substitute old bundles just to make an install succeed.

## 5. Build PTOAS With Its Own Pinned LLVM

Use the same Python ABI as the intended Triton environment. PTOAS's pinned
LLVM 19 Python bindings require compatible **pybind11 < 3** headers. Set
`PYBIND11_DIR` to an existing matching CMake package if automatic discovery
cannot find one. Do not copy Python 3.13 extension binaries into a 3.10 venv;
a header package located under another Python's directory is not itself an ABI.

```bash
export PYTHON_BIN="$CANDIDATE_VENV/bin/python"
export PTOAS_SOURCE_DIR="$NPUIR_SRC/third-party/ptoas"
export PTOAS_LLVM_SOURCE_DIR="$NPUIR_SRC/third-party/ptoas-llvm-project"
export PTOAS_BUILD_DIR="$NPUIR_SRC/build-ptoas"
export PTOAS_INSTALL_DIR="$TOOLCHAIN_PREFIX/ptoas"
# If needed: export PYBIND11_DIR=/path/to/pybind11-2.x/share/cmake/pybind11
bash "$NPUIR_SRC/build-tools/build_ptoas.sh" --jobs 32
```

This builds PTO LLVM and PTOAS. When a matching `fa2fd1f7` LLVM SDK and matching
Python ABI are already built, set `PTOAS_LLVM_BUILD_DIR` to that actual build and
use `--ptoas-only`; this is what the validated server did. Do not reconfigure a
shared LLVM build while someone uses it, or replace it with Ascend LLVM because
both version banners say 19.1.7.

## 6. Keep Compiler And Runtime Environments Separate

Do not globally source CANN 9.2 into the Torch-NPU runtime for this tested
package combination. Kernel compiler subprocesses need 9.2, while the parent
Torch-NPU/simulator process needs the previous 9.1 runtime.

The installed server profile uses one shared wrapper with tool-name links.
The following is a portable adaptation of its logic; it expects the variables
from section 2 to be exported. Create the `compiler-bin` directory, then save it as
`$TOOLCHAIN_PREFIX/compiler-bin/compiler-environment`, make it executable, and
link the desired tool names to it. Keep its configuration with your local
environment profile; the server's installed wrapper pins its paths directly.

```bash
mkdir -p "$TOOLCHAIN_PREFIX/compiler-bin"
```

```bash
#!/usr/bin/env bash
set -e
: "${CANN_COMPILER_ROOT:?}" "${TOOLCHAIN_PREFIX:?}"
: "${CANDIDATE_VENV:?}" "${PTOAS_LLVM_BUILD_DIR:?}"
tool_name="${0##*/}"
unset LD_PRELOAD LD_LIBRARY_PATH PYTHONPATH PYTHONHOME
unset ASCEND_HOME_PATH ASCEND_OPP_PATH ASCEND_AICPU_PATH ASCEND_TOOLKIT_HOME CANN_ROOT
export PATH="$CANDIDATE_VENV/bin:/usr/local/bin:/usr/bin:/bin"
export CANN_ROOT="$CANN_COMPILER_ROOT"
source "$CANN_ROOT/set_env.sh"
export PATH="$CANN_ROOT/tools/bisheng_compiler/bin:$CANN_ROOT/tools/bishengir/bin:$CANDIDATE_VENV/bin:$PATH"
if [[ -n "${TRITON_TOOLCHAIN_TRACE_FILE:-}" ]]; then
  printf '%s\t%s\t%s\n' "$tool_name" "$ASCEND_HOME_PATH" "$(command -v bisheng)" >> "$TRITON_TOOLCHAIN_TRACE_FILE"
fi
if [[ "$tool_name" == ptoas ]]; then
  export LD_LIBRARY_PATH="$PTOAS_LLVM_BUILD_DIR/lib:${LD_LIBRARY_PATH:-}"
  export PYTHONPATH="$TOOLCHAIN_PREFIX/ptoas${PYTHONPATH:+:$PYTHONPATH}"
  exec "$CANDIDATE_VENV/bin/python" "$TOOLCHAIN_PREFIX/ptoas/bin/ptoas" "$@"
fi
exec "$TOOLCHAIN_PREFIX/npuir/bin/$tool_name" "$@"
```

Preserve `"$@"`, `FALLBACK`, bridge selection and other kernel controls. Clear
inherited profiler/runtime preloads only in compiler children. Keep each
compiler's LLVM dependencies separate; the tested NPU-IR binaries link their
Ascend LLVM statically, while PTOAS needs its matching PTO LLVM library path.

For a new wrapper directory, create links after saving the wrapper:

```bash
chmod 755 "$TOOLCHAIN_PREFIX/compiler-bin/compiler-environment"
for tool in bishengir-compile bishengir-opt mlir-translate llvm-dis llvm-link llvm-config ptoas; do
  ln -s compiler-environment "$TOOLCHAIN_PREFIX/compiler-bin/$tool"
done
```

Configure runtime paths in your candidate profile or a subshell:

```bash
source "$CANDIDATE_VENV/bin/activate"
export CANN_ROOT="$CANN_RUNTIME_ROOT"
source "$CANN_ROOT/set_env.sh"
export ASCEND_NPU_IR_ROOT="$NPUIR_SRC"
export PATH="$TOOLCHAIN_PREFIX/compiler-bin:$CANN_ROOT/tools/bishengir/bin:$CANN_ROOT/tools/bisheng_compiler/bin:$CANN_ROOT/tools/profiler/bin:$PATH"
export LD_LIBRARY_PATH="$PTOAS_LLVM_BUILD_DIR/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$TOOLCHAIN_PREFIX/ptoas${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
export TRITON_NPU_COMPILER_PATH="$TOOLCHAIN_PREFIX/compiler-bin/bishengir-compile"
export TRITON_PTOAS_PATH="$TOOLCHAIN_PREFIX/compiler-bin/ptoas"
export PTOAS_AICORE_LL_MODULE="$TOOLCHAIN_PREFIX/npuir/lib/meta_op.aic.c310.O2.bc"
export PTOAS_AIVECTOR_LL_MODULE="$TOOLCHAIN_PREFIX/npuir/lib/meta_op.aiv.c310.O2.bc"
export TRITON_BISHENGIR_DISABLE_LIB_CALL_NOINLINE=1
export TRITON_SIMULATOR_CLEAN_EXIT=1
unset TRITON_ASCEND_ARCH TRITON_DISABLE_FFTS
```

These manual exports change the current shell; a subshell confines them. The
installed server profile additionally restores prior environment values on
deactivation. Back up existing activation files before adding a persistent hook.

Triton 3.6 was already correctly installed here, so it was retained. On another
machine, verify the actual interpreter, imported backend, extension and shared
bridge revision using the [3.6 setup record](../memory/triton-ascend-3.6-shared-a5-2026-10-22.md).
Do not overlay 3.6 Python files onto 3.2 binaries or assume a stock 3.6 wheel has
the shared-option patch. Backend lookup may prefer bundled tools over PATH:
inspect the selected executable before claiming the new compiler was tested.

```bash
python - <<'PY'
import sys, triton
from triton._C import libtriton
from triton.backends.ascend import compiler, utils
print(sys.executable, triton.__version__, triton.__file__)
print(compiler.__file__, libtriton.__file__)
print(utils._get_npucompiler_path()[0], utils._get_ptoas_path()[0])
PY
bishengir-compile --version
ptoas --version
```

## 7. Validate Both Flows Before Switching Shared Defaults

Use a new cache namespace for this compiler revision. The installed backend's
hash does not include the external compiler revision; an old cache can conceal
which compiler actually ran. Retain previous caches for rollback.

Start with the existing `vector_add_kernel`: 256 F32 elements, tile 256, grid 1.
For each flow choose a fresh private output directory and set the environment
below. Run native first, then bridge; do not launch a full model/matrix implicitly.

```bash
export FLOW=npuir  # Then repeat with FLOW=ptoas and a new output directory.
mkdir -p "$HOME/tmp"
export RUN_ROOT="$(mktemp -d "$HOME/tmp/ascend-update-$FLOW.XXXXXX")"
mkdir -p "$RUN_ROOT"/{cache,dump,logs,profile,tmp}
chmod 700 "$RUN_ROOT" "$RUN_ROOT"/{cache,dump,logs,profile,tmp}
export TRITON_ASCEND_COMPILE_FLOW="$FLOW"
if [[ "$FLOW" == ptoas ]]; then
  export BISHENGIR_ENABLE_PTOAS_BRIDGE=1 BISHENGIR_PTOAS_BRIDGE_MODE=ptodsl FALLBACK=1
else
  export BISHENGIR_ENABLE_PTOAS_BRIDGE=0
  unset BISHENGIR_PTOAS_BRIDGE_MODE FALLBACK
fi
export TRITON_CACHE_DIR="$RUN_ROOT/cache" TRITON_DUMP_DIR="$RUN_ROOT/dump"
export ASCEND_PROCESS_LOG_PATH="$RUN_ROOT/logs" TMPDIR="$RUN_ROOT/tmp"
export TRITON_TOOLCHAIN_TRACE_FILE="$RUN_ROOT/compiler-environments.tsv"
export TRITON_ALWAYS_COMPILE=1 TRITON_DEBUG=1 TRITON_KERNEL_DUMP=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
ulimit -Sn 65536
msprof op simulator --kernel-name=vector_add_kernel \
  --soc-version=Ascend950PR_9589 --core-id=0 --output="$RUN_ROOT/profile" \
  "$CANDIDATE_VENV/bin/python" \
  "$NPUIR_SRC/bishengir/test/Integration/PTOASBridge/testcases/vadd/vector_add.py" \
  > "$RUN_ROOT/run.log" 2>&1
rg '^max error: 0[.]0$|^allclose: True$' "$RUN_ROOT/run.log"
```

Require **both** exact-result lines, child success and the intended flow/tools;
the last `rg` is an inspection command, not sufficient by itself as an automated
PASS gate. `msprof` returned zero for an earlier child import failure. Inspect
the actual compiler target, bridge VMI/PTOAS invocation, and compiler child
environment. Confirm parent runtime-library mappings come from CANN 9.1.

### Recorded Validation

| Check | Result at the pinned snapshot |
| --- | --- |
| Native and bridge Python/Triton simulator vadd | Both exact, max error 0.0, child/profiler success, requested SoC |
| Shared-option/native differential tests | 46 pass before and after installation; subprocess/device behavior is mocked |
| Actual emitted native/bridge argument lists | Equal except the three intentional VMI emission/threading options |
| Real saved-IR native/VMI/PTOAS device-object replay | Pass; compiler-only evidence |
| PTOAS/MLIR/PTODSL imports with actual candidate extension | Pass |
| Installed activation, tool resolution, runtime maps and deactivation restoration | Pass |
| Five selected NPU-IR lit tests | Four pass; one FileCheck formatting mismatch below |

The bridge smoke used `FALLBACK=1` and linked native load/store dependencies.
It does not establish no-fallback coverage, MmadL1 numerical correctness for
the updated toolchain, a full suite, performance or physical-A5 readiness.
HOST/workspace consumer work remains [deferred](../planning/deferred-work.md#host-callbacks-and-workspace-metadata).

## Errors We Hit And What Resolved Them

| Symptom | Cause / resolution |
| --- | --- |
| Git HEAD looks current, compiler banner is old | Build/install and PATH were not updated together. Check the executable selected by the backend and its banner/hash. |
| Missing `hivm_GET_COREID`, `hivm_SET_COND`, `hivm_vdintlvv2`, `hivm_vcaddv2_v64f32_x` | The old Ascend LLVM pin `8ce9217f` lacked required declarations. The matching `95c46934` pin resolves them. Do not invent IDs or delete translation mappings. |
| `CCE_PRINT_CC` / `CCE_DEBUG_NAME` errors in native debug templates | CANN 9.1 compiler was too old for these sources. Compile templates with 9.2; no template source shim was required. |
| Undefined `runHIVMCCompileA3/A5`, `hivmcCanonicalizePath`, `isVFStackOverflowError` | Internal HIVMC had been disabled in our configuration. Restore upstream default `BISHENGIR_BUILD_HIVMC=ON` and explicitly regenerate an existing CMake cache. |
| Torch-NPU import cannot find `libhccl.so` under 9.2 | This tested Torch-NPU package expects the older runtime library layout. Keep its previous 9.1 runtime and isolate 9.2 compiler children; do not guess library aliases/stubs. |
| PTOAS import cannot find `libMLIRTargetCpp.so.19.1` | Matching PTO LLVM shared-library path is missing. Use that SDK's `lib` directory, not Ascend LLVM's similarly versioned libraries. |
| Wrong Python extension ABI / pybind discovery fails | Set the intended candidate `PYTHON_BIN`; use compatible pybind11 < 3 for this PTO LLVM/PTOAS build. Keep frontend and PTOAS build requirements separate. |
| `ccache` cannot create files under `/run/user/...` | Set a writable task-specific `CCACHE_TEMPDIR`; no compiler source fix is needed. |
| CANN CMake dependency fetch fails with DNS/network errors | PTOAS fetches its build dependency. Use authorized network access or a verified local copy/configuration for that same dependency; do not change the compiler implementation. |
| `llvm-lit` cannot find `count` / `not` | Build the test-support utilities and `FileCheck` before running scoped lit checks. This is runner setup, not the kernel failing. |
| `msprof` refuses a group-writable output directory | Create private output/cache/profile directories and `chmod 700` them. Always inspect the log even when `msprof` exits zero. |
| `aclInit` 507008 only inside the agent sandbox | The simulator may need local IPC unavailable there. Use the approved scoped execution environment; do not replace the working venv to hide the failure. |
| Late `ccec-post-link` step produces no output for minutes | Some template optimization steps are slow and single-threaded. Inspect your process CPU/log state before diagnosing a hang or replacing its tools mid-build. |
| Callback-free replay has no `libkernel.so` | A successful callback-free native compilation can omit it; the installed native backend already guards its presence. Do not apply that assumption to metadata-requiring HOST kernels. |

Optional CMake capability probes can print `Failed` during a successful
configuration. Diagnose the first actual failed command/fatal error rather
than treating every unsuccessful optional probe as a compiler defect.

### Known FileCheck Failure

`bishengir/test/Conversion/HIVMTemplatesToPTO/mmadl1-all-unit-flags-ptodsl.mlir`
expects `disable_gemv :`, while the new printer emits:

```mlir
pto.mad ... disable_gemv {operandSegmentSizes = array<i32: ...>} : ...
```

The unit-flag operations are present. Temporary checks accepting the added
dictionary pass with `FALLBACK=0` and `FALLBACK=1`, including STATE checks.
The repository test was left unchanged: the official result remains **4/5**.
Have the test owner review attribute-tolerant expectations while retaining the
semantic unit-flag checks. This diagnosis does not substitute for numerical
MmadL1 testing and does not justify weakening checks for unrelated failures.

## 8. Keep Rollback And A Reviewable Handoff

Record all four source commits, Python/package/import paths, actual compiler
banners/hashes, CANN compiler/runtime roots, wrapper configuration, resource
hashes, targets, commands and numerical results. Keep logs/build artifacts out
of the Planner commit. Preserve source patches that existed before the update.

Before wiring a candidate, back up activation and backend files and list every
new profile/link/path file. Verify activation/deactivation and a dry rollback
preflight. The validated installation's guarded rollback restores its original
activation and removes only its newly created environment wiring; it retains
compiler prefixes, worktrees, old caches and evidence. Its exact rollback files
and commands are in the [installation record](../memory/toolchain-refresh-cann92-2026-10-28.md).

Do not remove a reused PTO LLVM build as "old build output": the new PTOAS may
still depend on its shared libraries. Coordinate later cleanup separately.
