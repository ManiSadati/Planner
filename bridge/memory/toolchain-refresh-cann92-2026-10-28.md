# Wilson Master / CANN 9.2 Toolchain Refresh

For a portable procedure and error-to-cause index, use the
[update and troubleshooting guide](../designs/toolchain-update-cann92-compile-cann91-runtime.md).
This record describes this server's installation and evidence.

Status: the complete compiler toolchain is installed and wired into the existing
3.6 candidate. Native and bridge vector-add simulator checks pass exactly on
`Ascend950PR_9589`. The owner explicitly selected **CANN 9.2 for compilation and
the previous CANN 9.1 beta for runtime**. The old environment and shared installs
are preserved. No compiler/frontend source patches or source-repository
commits/pushes were made.

## Revisions And Locations

| Component | Staged revision / version |
| --- | --- |
| Wilson NPU-IR `master` | `3bf5d9898e0c45447c81780bc35de7548f5cef99`; compiler reports 2.0.0 |
| Ascend LLVM | `95c4693447af2473be62bdee19522cbe596f9e96`; LLVM 19.1.7 |
| PTOAS | `5ce40f710f7f83b8d1fb2d8c36647bb811236a49`; CLI 0.69 |
| PTO LLVM | `fa2fd1f75d742fec91ecf40d98c9c7961647ec42`; existing pinned build reused |
| Kernel compilation | Owner-provided CANN 9.2.0; `Ascend950PR_9589` |
| Torch-NPU / simulator runtime | Previous CANN 9.1.0-beta.3; `Ascend950PR_9589` |
| Existing candidate | Python 3.10.12; Triton 3.6.0, `triton-ascend==3.6.0+gite9f9b169`, shared A5 Python overlay `461aa8f0` |

Local `master` was fast-forwarded from `a8989093f` and checked out in
`$HOME/worktrees/AscendNPU-IR-master-20261028`. Its source and clean PTOAS/LLVM
worktrees match the master pins. The original checkout remains on another
agent's `mani/vector-dma-template` at `b411e48a7`; its dirty PTOAS backend work
is preserved. The previous master is retained as
`backup/master-before-toolchain-update-20261028`.

Install prefix: `$HOME/.local/opt/ascend-toolchain-3bf5d9898-20261028/`, with
`npuir/` and `ptoas/` subdirectories. LLVM/MLIR tools, headers, libraries and
CMake exports are under `npuir/`. Native O0/O2 template bundles and registry-
matched PTODSL resources are installed. PTOAS uses its own pinned LLVM build;
the two LLVM forks are not linked together.

Evidence root: `$HOME/tmp/toolchain-master-update-20261028/`.
`manifest.json` records tool hashes/banners, all pins, checks, preserved-file
hashes, actual runtime-library paths, compiler traces and rollback state. Actual
CANN paths are recorded there as installation data; use `$CANN_ROOT` in portable
commands.

## Build And Compiler Evidence

- Full NPU-IR/LLVM build and installation completed with CANN 9.2.0:
  `npuir-cann92-build.log`. PTOAS build/install: `ptoas-build.log`.
- The former missing-intrinsic mismatch is resolved by the new Ascend LLVM
  pin. Fresh headers include all four previously missing IDs, and both native
  HIVM translation files compile and link. This supersedes that blocker for
  this new revision pair, without resuming deferred HOST consumer work.
- CANN 9.1 beta failed on current debug-template macros. A temporary diagnostic
  shim was insufficient and was never applied to compiler source. CANN 9.2
  compiles these templates without that shim, including mixed AIC/AIV.
- The initial task build configuration incorrectly disabled internal HIVMC.
  Restoring the current upstream default `BISHENGIR_BUILD_HIVMC=ON` fixed the
  resulting unresolved linker symbols; no implementation change was made.
- Existing installed Triton backend: **46 option-sharing/native differential
  checks pass**. Its compiler file still matches published `461aa8f0` exactly.
  No Triton rebuild/reinstall has been needed for the checks performed so far.
- PTOAS 0.69 CLI, native extension, PTODSL imports and an MLIR parse/verify pass
  with the actual candidate Python and the matching PTO LLVM library path.
- Five focused NPU-IR lit tests give **four passes and one failure**. The
  failure is `mmadl1-all-unit-flags-ptodsl.mlir`: checks expect `disable_gemv :`,
  but the new PTOAS printer emits an `operandSegmentSizes` dictionary before
  the colon. Temporary attribute-tolerant CHECKs pass with `FALLBACK=0` and
  `FALLBACK=1`, including STATE checks. Official source/tests remain unchanged;
  this is still an official test failure, not a five-test PASS.
- `checks/vadd-replay/results.json` records real native device-object emission,
  bridge VMI emission containing `pto.vmi.vadd`, and successful PTOAS device-
  object emission. The unchanged fixture is IR previously produced by this
  Triton 3.6 candidate. Target is `Ascend950PR_9589`; the bridge uses
  `FALLBACK=1` and selectively links native load/store dependencies from the
  newly built CANN 9.2 bundles. This is compiler-only replay, not a fresh
  Python JIT or numerical/runtime result.

Reproduction scripts are `build-npuir.sh`, `build-ptoas.sh`, and
`replay-vadd.py` under the evidence root. Build with the existing CANN 9.2
installation selected by `$CANN_ROOT`. Build jobs were bounded to 64 for
NPU-IR and 32 for PTOAS. No simulator matrix was launched.

## Verified Compiler / Runtime Separation

Using CANN 9.2 for the whole process initially failed before compilation:
`torch_npu==2.7.1.post8` requires `libhccl.so`, which this 9.2 installation
does not provide under that name. The owner explicitly directed compilation
to use 9.2 while running with the previously working CANN 9.1 beta. No HCCL
alias, stub, Torch upgrade or cross-version runtime-library fallback was added.

`compiler-bin/compiler-environment` is one shared launcher with tool-name links.
It preserves compiler arguments and kernel controls, clears inherited runtime
library/preload paths in the child only, sources CANN 9.2, and executes the
matching staged compiler. Only the PTOAS child adds its own LLVM library path.
The parent Torch-NPU/simulator process stays on CANN 9.1. The actual installed
profile's loaded-library map confirms CANN 9.1 for HCCL, runtime and ACL.

Both actual Python/Triton simulator checks pass with cold private caches:

| Flow | Compiler environment | Runtime / simulator | Result |
| --- | --- | --- | --- |
| Native NPU-IR | CANN 9.2, new NPU-IR/LLVM | CANN 9.1 beta, `Ascend950PR_9589` | max error 0.0; allclose True |
| PTOAS bridge | CANN 9.2, new NPU-IR and PTOAS | CANN 9.1 beta, `Ascend950PR_9589` | max error 0.0; allclose True |

Case: the unchanged `vector_add_kernel`, 256 F32 elements, tile 256, grid 1,
profiler core 0; bridge mode `ptodsl`, `FALLBACK=1`. Both profiler and child
success were checked. Evidence is `smoke-npuir-runtime91/` and
`smoke-ptoas-runtime91/`, including `compiler-environments.tsv`. Emitted native
and bridge compiler argument lists match modulo the three intentional VMI
options (`checks/emitted-argument-parity.json`). This is bounded functional
validation, not a matrix, performance or physical-A5 result.

## Candidate Installation And Rollback

Use the existing candidate normally in Bash:

```bash
source "$HOME/.venv/npuir-sim-3.6-candidate/bin/activate"
```

Its activation now selects the compiler wrappers, CANN 9.1 runtime, matching
PTOAS/LLVM paths and native compilation by default. To select the bridge for a
kernel, set `TRITON_ASCEND_COMPILE_FLOW=ptoas`,
`BISHENGIR_ENABLE_PTOAS_BRIDGE=1`, `BISHENGIR_PTOAS_BRIDGE_MODE=ptodsl` and
`FALLBACK=1` as before. Keep simulator target `Ascend950PR_9589`.

The profile restores prior environment settings on deactivation. A fresh cache
namespace (`ascend-3bf5d9898-compile92-runtime91`) prevents reuse across this
external-compiler update; the backend hash currently omits that revision.
Previous caches are retained. Candidate-only changes are activation/profile,
command links and one Python path file; `env-install.json` lists them exactly.

After installation, actual Torch/Torch-NPU/Triton/PTOAS/PTODSL imports, selected
tools, runtime-library mappings, deactivation restoration and **46 option/native
parity checks pass**. Triton's backend source and extension were retained. The
old 3.2.2 environment and shared NPU-IR/PTOAS installs remain intact; five
protected hashes match the pre-update snapshot, while candidate activation is
the recorded intentional sixth-file change.

Rollback files are under the evidence root, especially `env-rollback/activate`,
`activate-before`, `triton-installed-before-*.py` and `env-install.json`.
`rollback-env.py --check` passes without changing files. To restore the candidate
wiring, deactivate first, then run `python3` on `rollback-env.py`. It verifies
file/link integrity, restores the original activation and removes only this
refresh's created profile, path file and links. Compilers, worktrees, caches and
evidence are retained.

## Next Agent

Use this profile and manifest for current local tests. The original manual
tool-path exports in older handoffs select historical installs. Keep the
old environment and shared builds available; do not replace them or edit the
other agent's dirty PTOAS checkout. Host builds used system Clang 14; device
template/binary compilation uses the selected CANN 9.2 compiler.

Hand the printer-expectation failure to the NPU-IR test owner separately; no
compiler behavior or official test was changed to make it pass. Larger kernels,
template matrices, performance and physical A5 need their own validation.
No new HOST consumer or SET2D branch was adopted. HOST integration remains
[deferred](../planning/deferred-work.md#host-callbacks-and-workspace-metadata):
the seven deferred HOST integration cases and two compiler fixtures were not
run, and this refresh does not establish bridge workspace/HOST metadata support.
