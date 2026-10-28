# Triton HOST metadata integration against shared mainline

Owner decision, 2026-10-28: **deferred; do not adopt or install these changes
for now.** See [deferred work](../planning/deferred-work.md#host-callbacks-and-workspace-metadata).
The implementation, evidence and next-owner instructions below are preserved
for a future explicitly approved resumption, not an active assignment.

Recorded: 2026-10-28. **Frontend changes are published; matching-mainline
verification and candidate installation are blocked by an NPU-IR/LLVM dependency
mismatch.** The candidate and old environment were not patched. No NPU kernels
or simulations were performed. This follows the
[mainline handoff](ptoas-mainline-workspace-handoff-2026-10-27.md), not the SET2D
branch's different host-export contract.

## Publication and integration-agent report

At the user's request, the five frontend/test files were committed and pushed to
[Melikano/triton-ascend:feature/ptoas-host-metadata-mainline](https://github.com/Melikano/triton-ascend/tree/feature/ptoas-host-metadata-mainline)
at [3e9b51d0577516ffc8d60d838b61d8ba4b857c6a](https://github.com/Melikano/triton-ascend/commit/3e9b51d0577516ffc8d60d838b61d8ba4b857c6a).
The branch starts from `461aa8f0`; the original `feature/shared-a5-native-ptoas`
branch remains at that base. The new branch tracks its same-named remote and
the source working tree is clean. No PR or merge is claimed. Planner handoff
updates remain local and preserve unrelated pending documentation work.

Integration-agent action: reconcile the mainline NPU-IR source, intended LLVM
dependency and actual installed/build-tree tools before using this branch as an
installed baseline. The pinned LLVM header proof and failing translation targets
below identify the dependency gap. Do not substitute the SET2D branch or discard
native conversion behavior. Once a matching toolchain is available, run the seven
pending integration cases and two mainline compiler fixtures, then coordinate
candidate installation. Runtime/numerical validation remains with the testing
agent. Neither environment was changed by publication.

Pre-publication rerun: `prepublish-focused-tests.log`, **86 passed, 7 skipped**.
The staged and committed file bytes match the recorded tested hashes.

## Source and tool provenance

| Component | Verified state |
| --- | --- |
| Triton-Ascend source | `$HOME/triton-ascend`, `feature/ptoas-host-metadata-mainline`, published `3e9b51d0577516ffc8d60d838b61d8ba4b857c6a`, based on `461aa8f086c6bf26229ee324b8f4b5d3902e002d` |
| Shared NPU-IR mainline | `wilsoncxfeng/master`, remote and local `a8989093f4980299dd145d5321cd9dd1e948a312` |
| Ascend LLVM pin | `8ce9217f83f3376970d10428de4e2d1c15039e08` |
| PTOAS source pin | `5ce40f710f7f83b8d1fb2d8c36647bb811236a49` |
| PTO LLVM pin | `fa2fd1f75d742fec91ecf40d98c9c7961647ec42` |
| Existing installed NPU compiler | `$HOME/AscendNPU-IR/build/install/bin/bishengir-compile`, banner `75525e826a7b`; not a mainline build |
| Existing build-tree NPU compiler | `$HOME/AscendNPU-IR/build/bin/bishengir-compile`, banner `b411e48a7537`; also predates the mainline sidecar contract |
| Candidate Python | `$HOME/.venv/npuir-sim-3.6-candidate/bin/python`, Python 3.10.12, imported Triton 3.6.0 |
| Candidate distributions | `triton==3.6.0`, `triton-ascend==3.6.0+gite9f9b169`; existing `461aa8f0` Python overlay remains |

Mainline was inspected and built directly; no SET2D checkout, cherry-pick or
SET2D host-export option was used. NPU-IR tracked source, compiler sources,
templates and pins were not edited. Existing unrelated PTOAS submodule edits
were preserved. Standalone PTOAS was not invoked or rebuilt for this task.

Evidence root: `$HOME/tmp/triton-host-metadata-mainline-20261028/`.
`verification.json` records source state, exact changed-file/tool hashes and
tool versions. `provenance-before.json` and the final protected-hash comparison
cover the installed/build-tree NPU compilers and optimizers, old-environment
compiler, and candidate compiler. The candidate extension remains
`4d8cc09bc5bddfd907ff79faee6f7ddf5950563fd4e8694d1f43ba0a24b5121e`.

## Implemented frontend behavior

Five Triton paths are owned by this change:

- `third_party/ascend/backend/compiler.py`
- `third_party/ascend/backend/ptoas_host.py` (new)
- `third_party/ascend/unittest/pytest_ut/test_ptoas_compile_flow.py`
- `third_party/ascend/unittest/pytest_ut/test_ptoas_host_metadata.py` (new)
- `third_party/ascend/unittest/pytest_ut/fixtures/ptoas_host_metadata.mlir` (new)

The existing shared A5 builder/options remain. Only after successful device
compilation and a present VMI output does the bridge check the matching
optimizer's mainline separation capability and look for the exact automatic
`<device-output>.host.mlir` sidecar. A known HOST-bearing input missing its
sidecar fails. Callback-free successful compilation can retain zero defaults.
Sidecars from failed device compilations are never consumed.

The host helper resolves sibling `bishengir-opt` and `mlir-translate`, obtains a
native CPU layout/triple from `bisheng` or `clang` (explicit override:
`TRITON_PTOAS_HOST_CC`), lowers HOST MLIR through standard CPU passes, translates
to LLVM IR, and links `libkernel.so` with `-shared -fPIC -Wl,-z,defs`. This is
host compilation, not another device compilation. Indices are explicitly
64-bit, and CPU codegen receives the native CPU triple/layout. Host helpers and
symbol bindings are retained; callback results are evaluated, never parsed out
of constants or hardcoded.

Host command sequence (paths are per-compilation temporary files):

```sh
bishengir-opt kernel.ptovmi.mlir.host.mlir \
  --set-llvm-module-datalayout='data-layout="<native CPU layout>"' \
  --expand-strided-metadata --lower-affine --convert-scf-to-cf --arith-expand \
  --convert-math-to-llvm --finalize-memref-to-llvm=index-bitwidth=64 \
  --convert-index-to-llvm=index-bitwidth=64 --convert-arith-to-llvm \
  --convert-cf-to-llvm --convert-func-to-llvm=index-bitwidth=64 \
  --reconcile-unrealized-casts -o kernel.host.llvm.mlir
mlir-translate --mlir-to-llvmir --allow-unregistered-dialect \
  kernel.host.llvm.mlir -o kernel.host.ll
"$TRITON_PTOAS_HOST_CC" -shared -fPIC -Wl,-z,defs kernel.host.ll -o libkernel.so
```

The helper obtains the native layout/triple with the CPU compiler's
`-x c -S -emit-llvm -o - -` on empty input, and replaces inherited LLVM target
lines before the final command. Tool/lowering/link errors propagate.

The existing native callback reader is shared for workspace size, task type,
lock layout and lock initialization. Native calls retain their original default
`c_int64` behavior. Bridge callback definitions must have zero arguments,
exported C calling convention and a validated return ABI: `i64` for workspace
and lock metadata; native task-type callbacks may return `i8` (as emitted by
mainline `InsertInferTaskTypeFunc.cpp`) or compatible `i64`. The bridge uses the
corresponding ctypes type. Unsupported signatures, missing symbols/libraries,
tool failures and negative workspace/task values propagate as errors.

Existing early launcher restrictions remain. Nonzero callback-derived
`sync_block_lock_layout` also fails before returning VMI. Native binary versus
bridge VMI return types, launcher argument placement, and workspace-size-times-
grid allocation are preserved. Only metadata values enter the normal kernel
cache, not temporary paths. The new backend module is covered by Triton's
existing backend-source cache versioning.

## Focused verification

`final-focused-tests.log`: **86 passed, 7 skipped** using the candidate's actual
3.6 extension and source backend loaded by `run-source-tests.py`; frontend/native
extension imports were not stubbed. This comprises 22 existing flow/option tests,
24 historical native differential cases, and 40 HOST metadata checks.

Real CPU libraries cover workspace returns 0, 16, 64 and a value above 32 bits;
non-native bound symbols; helper calls; affine/rounded arithmetic; native `i8`
task type; CPU index ABI despite device layout attributes; invalid signatures;
lowering/link/library failures; and unsupported lock metadata. Cache checks use
real FileCacheManager/CompiledKernel serialization and actual `triton.compile()`
cache hits, with NPU stages/launch plumbing and unrelated frontend/CANN version
tokens mocked. Launcher checks generate actual C++ allocation/argument code
without compiling or launching an NPU kernel.

Final positive CPU tests use `cpu-test-tools/bin/bishengir-opt`, an explicitly
test-only adapter to the existing LLVM 19 `mlir-opt --allow-unregistered-dialect`,
and its matching core `mlir-translate`, plus CANN's CPU `bisheng`. They invoke no
SET2D or NPU-specific lowering pass. This adapter is **not** a matching NPU-IR
compiler and cannot satisfy the production export-capability check.

Reproduction: with `CANN_ROOT` or `TRITON_TEST_HOST_CC` set, run
`bash "$HOME/tmp/triton-host-metadata-mainline-20261028/run-focused-checks.sh" cpu`.
The `mainline` mode instead requires a verified `TRITON_TEST_MAINLINE_TOOL_DIR`.
The runner retains real host modules/libraries/cache files under `pytest-runs/`.

The seven real-mainline integration cases remain skipped because no verified
mainline tool prefix was produced. They cover automatic sidecars and the shared
backend for workspace 0/16/64, plus callback-free compilation. The two unchanged
mainline lit fixtures were copied to `lit/tests/` for subsequent verification;
they have not passed against a built mainline compiler. The original source
fixture retains its upstream `Ascend910_9589` alias; new driver fixtures use
`Ascend950PR_9589`. No simulator target substitution or runtime result is claimed.

## Compiler dependency blocker

A private standalone CMake wrapper reused existing LLVM/MLIR libraries and built
mainline NPU-IR under the evidence root, leaving shared build/install products
untouched. Initial wrapper configuration issues were resolved by matching the
Ascend feature defines and normal current-directory include behavior. The final
build then failed in `HIVMToLLVMIRTranslation.cpp` and
`HIVMRegbaseIntrinsToLLVMIRTranslation.cpp`: current NPU-IR references intrinsic
IDs absent from the pinned LLVM definitions, including `hivm_GET_COREID`,
`hivm_SET_COND`, `hivm_vdintlvv2` and `hivm_vcaddv2_v64f32_x`.

Freshly regenerating `IntrinsicsHIVM.h` from the pinned LLVM source with Ascend
defines succeeds and produces **byte-identical** output to the existing header
(SHA256 `645a6b8353a8d34e209c570772195b870f95bdf726472c22dac649c2598add17`).
Legacy intrinsics are present; the required IDs remain absent. This is not fixed
by merely regenerating a stale header. No fake intrinsic IDs, dropped translation
maps, compiler patches, pin changes or SET2D fallback were applied.

Evidence: `build.log`, `toolchain-wrapper/`, `intrinsics-regeneration.log`,
`pinned-IntrinsicsHIVM.h`, and `intrinsics-proof.json`. Invoking the existing
`b411e48a` compiler on the new mainline fixture fails with the older host-output
contract (`mainline-fixture.log`); it is not a valid mainline test substitute.

## Installation and next owner

**Not installed.** Matching-mainline checks are a required gate, so passing
CPU-only checks was not used to overwrite the working candidate. Its compiler
remains SHA256 `521d19e8e4c0d04c3c184cf8603b881c1a0f107f563beff37dc7238709172513`,
and the new helper is absent from site-packages. `candidate-before/` contains
rollback copies of compiler/utils/driver/init files; `task.patch` includes all
five frontend/test paths against the `461aa8f0` base. No rollback is needed now.

Next bounded action: the compiler/toolchain owner must provide or establish a
verified mainline build with compatible LLVM intrinsic definitions. Do not repair
that by modifying Triton, inventing IDs, deleting native lowering, changing pins,
or borrowing the SET2D branch. Then set `TRITON_TEST_MAINLINE_TOOL_DIR` and
`TRITON_TEST_HOST_TOOL_DIR` to the matched tools directory, and
`TRITON_TEST_HOST_CC` to `$CANN_ROOT/tools/bisheng_compiler/bin/bisheng`; rerun
the focused tests and the two mainline fixtures. Use a matching translator in
that prefix, not a differently versioned bundled tool.

After those checks pass, coordinate exclusive candidate ownership, recheck its
recorded hashes, apply only compiler.py plus the new helper with atomic writes,
and rerun the checks using normal installed imports. Preserve the old environment
and extension. Installation/reporting remains open; NPU runtime and numerical
validation belong to the testing agent after integration is verified. Unrelated
Planner edits and per-template adapter/uniform-ABI decisions remain untouched.
