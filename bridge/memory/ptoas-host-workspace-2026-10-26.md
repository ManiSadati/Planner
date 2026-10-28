# PTOAS host workspace callback export

Current installer direction, 2026-10-27: the user will use the
[shared-mainline host MLIR contract](ptoas-mainline-workspace-handoff-2026-10-27.md)
instead of this SET2D branch for now. The implementation and validation below
remain a historical record of the separately published branch.

Status: NPU-IR host extraction/export implemented and compiler-checked. Triton
integration and installation belong to the installer agent. BF16 SET2D remains
blocked by the independent intra-block PIPE_S restriction; no new NPU launches,
numerical matrix, or physical-A5 validation was performed.

This follows the [BF16 compiler investigation](set2d-bf16-compiler-followup-2026-10-26.md).
The user authorized the compiler half and requested an installer-agent prompt.
Copyable prompt: `$HOME/tmp/ptoas-host-workspace-20261026/installer-agent-prompt.txt`.

## Compiler contract

```sh
bishengir-compile input.mlir --emit-ptoas-vmi \
  --ptoas-host-output-file=kernel.host.ll [existing A5 options] \
  -o kernel.ptovmi.mlir
```

The equivalent standalone conversion option is:

```sh
bishengir-opt planned.mlir \
  '--convert-hivmave-to-ptoas-vmi=preserve-external-call-memrefs=true host-output-file=kernel.host.ll' \
  -o kernel.ptovmi.mlir
```

The converter exports marked HOST functions before converting device code. It
uses HACC's host/device separation utility and standard MLIR CPU lowering,
retaining native callback symbol names and a 64-bit index ABI. The host clone
does not inherit the NPU target/data layout. HOST definitions are removed from
device IR only after successful export. Kernel arguments, workspace addressing,
SET2D computation, native CCE, and synchronization mapping are unchanged.

The existing HACC host pipeline transforms host/device launch stubs; it is not
by itself a complete CPU callback compilation pipeline. This implementation
exports ordinary host LLVM IR and leaves host linking to the frontend. No
second device compilation or hardcoded workspace size is needed.

- HOST-bearing input without a host output fails explicitly.
- Requesting host output for callback-free input produces valid empty host LLVM
  IR, allowing the frontend to distinguish that case from a missing artifact.
- Host output cannot be stdout or alias the compiler's device output.
- HOST/device function dependencies, including device declarations, are rejected.
  Defined host helpers must carry the existing HACC HOST classification so the
  native separation utility retains their bodies. This metadata path does not
  implement CPU functions that launch device kernels.
- Unsupported CPU operations fail lowering/translation. File I/O errors fail
  compilation. No metadata is silently replaced with zero.
- A sidecar may exist when later device conversion fails. Consumers must require
  the compiler's successful exit before using either output.

## Owned source changes

All paths are within `$HOME/AscendNPU-IR`:

- `bishengir/lib/Conversion/HIVMAVEToPTOASVMI/PTOASHostCallbacks.{h,cpp}`:
  separation, dependency checks, CPU lowering and LLVM IR export.
- Existing converter `.cpp`, public conversion header, conversion `Passes.td`,
  and conversion CMake: pass option, factory, dependent dialects, and linkage.
- Compiler `Options.td`, regbase `PassPipeline.cpp` and
  `BiShengIRCompileMain.cpp`: public option, active integration, and output guards.
- New conversion fixtures `host-workspace-callback.mlir` and
  `host-callback-contracts.mlir`; extended `triton/emit-ptoas-vmi.mlir` compiler test.

Twelve source/test paths belong to this task. The separate SET2D and BF16
converter changes present before the task were preserved. A patch relative to
that starting state, rather than to clean HEAD, is recorded as `task.patch`.
No Triton source/installation, PTOAS source, submodule pin, template body,
selection pass, adapter, or resource was changed by this task.

## Provenance and private publication

Branch `mani/mmadl1-scratch-state`, NPU-IR
`b411e48a7537708178e1b1714dabc547e59da521` plus uncommitted changes.
Ascend LLVM `8ce9217f83f3376970d10428de4e2d1c15039e08`, PTOAS
`d36100d03255e5e8a858c497a4277708d8b889c9`, PTO LLVM
`fa2fd1f75d742fec91ecf40d98c9c7961647ec42`.

Evidence root: `$HOME/tmp/ptoas-host-workspace-20261026/`.
New CMake publication: `install/`. Bounded SET2D replay prefix:
`set2d-only-install/` (matching new binaries, four SET2D resources, native
dependencies). It isolates the previously documented unrelated MmadL1 adapter
restriction. Neither prefix replaces the shared install or older evidence tools.

Private `bishengir-compile` SHA256:
`3a6e87c50c9196a972b59f167fe592d3ae82d34cb88743fd7193b4315e132254`.
Private `bishengir-opt` SHA256:
`addb3baa5e5657250c57ae3b4f0d1dde9e3c783b5fc17642df2d747438372f6b`.
The installed compiler's version banner alone does not describe these dirty changes.

## Publication, 2026-10-27

At the user's request, the current SET2D instances, adapters, native/runtime and
compiler tests, BF16 caller-conversion fixes, and this workspace export were
copied to a separate worktree and pushed to GitCode `origin`
(`manisadati/AscendNPU-IR`) as `mani/set2d-ptodsl-workspace`:

- `51fc74fe2bea09f358d56281d32066159b210d0c`: SET2D development and coverage.
- `595ecac83324a96114787c3e120c573f105157fe`: HOST callback export (branch tip).

The branch starts from the same `b411e48a7` MmadL1 baseline as the validation
above. Worktree: `$HOME/worktrees/AscendNPU-IR-set2d-ptodsl-workspace`.
Its 36 source/test files match the original checkout's snapshot, with one
extra EOF blank line removed from a new compiler fixture. The new worktree is
clean; the original branch, dirty files and index were preserved. No additional
NPU runtime tests or installations were performed for publication. Evidence:
`$HOME/tmp/set2d-branch-publication-20261027/`.

The earlier "uncommitted" provenance above describes the tested development
snapshot. The private compiler's implementation is now represented by these
published commits. The installer prompt has been updated to identify the branch.

## Verification

- Rebuilt `bishengir-opt` and `bishengir-compile`; published to the new private
  prefix with CMake's `bishengir-publish-products` component. `git diff --check`
  passes. Shared installed compiler/backend hashes remain unchanged.
- **3/3 focused lit fixtures pass.** They cover mixed AIC/AIV separation,
  64-bit workspace return, helper arithmetic, another zero-valued callback,
  no-callback output, missing output, cross-domain definitions/declarations,
  stdout/I/O errors, CLI wiring, and colliding output paths.
- **70/70 relevant compiler tests pass**: the HIVMAVE-to-PTOAS-VMI directory,
  the VMI compiler fixture, and compiler command-line checks.
- Generated host LLVM modules linked using CANN's host `bisheng`:
  `bisheng -shared -fPIC -Wl,-z,defs host.ll -o callback.so`.
  CPU `ctypes` calls with `c_int64` return values **24, 14, 0, and 8**, validating
  actual exported symbols/ABI and helper arithmetic. These are compiler metadata
  checks, not NPU numerical tests.
- Replayed the unchanged cached frontend input/options for BF16 `small-odd-a`.
  The exported native-named callback links and returns **8 bytes per block**.
  The HOST metadata failure is gone; device compilation still fails on
  `pto.set_intra_block<PIPE_S>`. This is an expected remaining failure, not a
  successful BF16 kernel compilation or runtime result.
- Replayed `loaded-scalar-reuse`: exact BF16 SET2D PTO helper/body remains,
  no workspace callback enters device VMI, the host sidecar is empty, and pinned
  PTOAS emits a device object. Native ND2NZ, MmadL1 and output Fixpipe dependencies
  remain explicit fallback. No NPU execution was performed.

Logs: `focused-lit.log`, `broad-lit.log`, `artifact-checks.json`,
`artifact-checks.log`, `small-odd-a.log`, host `.ll`/`.so` artifacts,
`loaded-scalar-object.log`, and `summary.json`. `check-artifacts.sh` records the
bounded reproduction environment and invokes the preferred candidate Python.
`source-manifest.json`, `protected-manifest.json`, `revisions.json`, and
`task.patch` record source/tool evidence. Historical build/test failures are
retained; the final logs and manifests identify the checked revision.

## Installer integration still required

Have the shared A5 builder request `--ptoas-host-output-file` in its temporary
directory. After successful NPU-IR compilation, link the host LLVM IR to the
existing `libkernel.so`, consume native callback metadata before returning VMI,
and cache the resulting values. Preserve native binary versus VMI text returns,
option sharing, launcher workspace allocation and current resource restrictions.
Missing/failed artifact production must not silently select workspace_size=0.

Use the matching new compiler prefix, verify/update the preferred 3.6 candidate
against its source, retain rollback files, and leave the old environment intact.
The prompt details focused backend tests and installation reporting. Runtime
matrix ownership remains with the testing agent after PIPE_S is addressed.
