# Triton Ascend PTOAS End-to-End Integration Plan

Last updated: 2026-08-26

Status: proposed. No implementation has started.

## 1. Goal

Add a second Triton Ascend compilation flow that can be selected when running a
normal Triton Python program. The existing NPU-IR native flow must remain the
default and continue to use the existing host launcher.

The intended command-line experience is:

```bash
# Existing flow; remains the default.
python3 kernel.py

# New NPU-IR-to-PTOAS flow.
TRITON_ASCEND_COMPILE_FLOW=ptoas python3 kernel.py
```

An optional per-kernel launch option may also be supported later:

```python
kernel[grid](..., compile_flow="ptoas")
```

The environment variable is the primary interface because it selects the flow
without requiring edits to existing kernel programs. If both interfaces are
implemented, the per-kernel option should override the environment default.

The desired new flow is:

```text
Python Triton kernel
  -> TTIR
  -> TTAdapter MLIR
  -> NPU-IR pipeline through ConvertHIVMAVEToPTOASVMI
  -> PTOAS VMI MLIR
  -> PTOAS VPTO/device lowering
  -> raw AICore device object
  -> existing Triton CompiledKernel and NPULauncher
  -> rtDevBinaryRegister + rtFunctionRegister + rtKernelLaunch
```

## 2. Non-Goals

This work should not:

- add a second Triton backend or duplicate the Ascend driver;
- generate a standalone C++ host executable for every Python kernel;
- replace the existing NPU-IR native flow;
- recover production IR by parsing `--mlir-print-ir-after` diagnostic logs;
- pass a PTOAS host fat object directly to `rtDevBinaryRegister`;
- broaden the current AVE-to-VMI operation coverage as part of driver
  integration, except where a selected integration testcase exposes a required
  missing conversion;
- claim A2/A3 support before target mapping and PTOAS object compatibility are
  validated for those architectures.

## 3. Current Components and Ownership

The work spans three repositories. Each component should retain a clear
responsibility:

| Component | Responsibility |
| --- | --- |
| Triton Ascend | User-facing flow selection, stage orchestration, cache identity, metadata assembly, and invocation of the existing launcher |
| AscendNPU-IR | TTAdapter-to-PTOAS-VMI compilation through the selected bridge endpoint |
| PTOAS | PTOAS VMI/VPTO lowering and emission of a runtime-loadable device object |
| Existing Triton NPU driver | Device binary registration, Python argument extraction, workspace allocation, packed launch arguments, and kernel launch |

The Triton Ascend source checkout is not currently present as a standalone
repository under this workspace. The active installed files used to inspect
the current behavior are under:

```text
$HOME/.local/lib/python3.10/site-packages/triton/backends/ascend/
```

Implementation must be made in the equivalent files in a Triton Ascend source
checkout, not by treating the installed package as the source of record.

## 4. Confirmed Constraints

### 4.1 PTOAS output format

The current PTOAS default object path emits an x86-64 relocatable host fat
object. It contains an embedded `__aicore_rel_binary` section and host-side
registration code. The existing Triton Ascend loader instead passes the entire
`npubin` byte sequence directly to `rtDevBinaryRegister` with
`RT_DEV_BINARY_MAGIC_ELF_AIVEC` for vector kernels.

Therefore, the default PTOAS `.o` cannot simply be returned as Triton's
`npubin`. The clean integration requires PTOAS to expose the raw merged AICore
device ELF before host-stub compilation.

Relevant files:

```text
PTOAS/tools/ptoas/driver.cpp
PTOAS/tools/ptoas/ObjectEmission.cpp
PTOAS/tools/ptoas/ObjectEmission.h
PTOAS/tools/ptoas/ptoas.cpp
triton/backends/ascend/npu_utils.cpp
```

### 4.2 Host launch ABI

The existing generated launcher builds a packed argument structure containing
some or all of the following fields:

```text
FFTS address, when enabled
sync-block-lock pointer
workspace pointer
non-constexpr Triton kernel arguments
grid X, Y, and Z
device-debug pointer, when enabled
```

The PTOAS device entry function must consume the same logical fields in the
same order, with the same widths and alignment assumptions. Matching only the
kernel symbol is insufficient.

Relevant file:

```text
triton/backends/ascend/driver.py
```

### 4.3 Runtime metadata

The existing launcher depends on compilation metadata including:

- kernel name;
- mix mode;
- tensor kinds;
- workspace size;
- sync-block-lock count and initialization value;
- task type;
- final kernel argument layout.

The native NPU-IR path obtains some resource values from generated callback
functions in `libkernel.so`. The PTOAS path does not currently provide those
callbacks. The new flow therefore needs an explicit structured metadata
contract instead of silently defaulting missing values.

### 4.4 NPU-IR bridge endpoint

The current NPU-IR pipeline uses `BISHENGIR_ENABLE_PTOAS_BRIDGE` to add
`ConvertHIVMTemplatesToPTO` and `ConvertHIVMAVEToPTOASVMI`, but then continues
into later native-backend passes. Planner scripts recover the VMI module from a
pass dump even when a later pass fails.

The production Triton path needs a supported compiler output mode that writes
the verified PTOAS VMI module and exits successfully at that point.

## 5. Implementation Plan

### Phase 0: Freeze the object and launch contracts

Before changing the Python driver, use one known vector-add artifact to prove
the runtime contract.

Tasks:

1. Compare a native NPU-IR `npubin`, a PTOAS raw AICore object, and the PTOAS
   host fat object with `file`, `readelf`, and symbol inspection.
2. Verify the raw PTOAS object can be registered with the same
   `rtDevBinaryRegister` and `rtFunctionRegister` sequence used by Triton.
3. Record the exact exported entry symbol after PTOAS ABI name processing.
4. Compare the native and PTOAS entry argument lists, including hidden
   workspace, lock, and grid arguments.
5. Launch a small PTOAS object through a minimal copy of the Triton registration
   sequence before integrating it into Triton JIT compilation.

Files to inspect or use for the test:

```text
triton/backends/ascend/npu_utils.cpp
triton/backends/ascend/driver.py
PTOAS/tools/ptoas/ObjectEmission.cpp
Planner/bridge/triton-example/vector_add_large_ptoas_bridge_aicore.o
Planner/bridge/testcases/vadd_large/vector_add_large_kernel_ptoas_fatobj.o
```

Acceptance gate:

- a raw PTOAS AICore object registers successfully;
- the expected kernel symbol is found;
- the launch argument structure is documented field by field;
- no host fat object is passed directly to the Triton loader.

### Phase 1: Add a first-class PTOAS VMI output mode to NPU-IR

Add an explicit compiler mode such as:

```bash
bishengir-compile input.mlir \
  --emit-ptoas-vmi \
  --target=Ascend910_9589 \
  -o kernel.ptovmi.mlir
```

Behavior:

1. Run the normal Triton/NPU-IR preparation and optimization pipeline.
2. Run `ConvertHIVMTemplatesToPTO` at its current bridge position.
3. Run `ConvertHIVMAVEToPTOASVMI` at its current bridge position.
4. Verify the resulting module.
5. Write textual MLIR to `-o`.
6. Stop before `ConvertHIVMAVEToAVEIntrin`, LLVM lowering, and external
   `hivmc-a5` execution.
7. Return success when the VMI output was emitted successfully.

Files to change in AscendNPU-IR:

```text
bishengir/include/bishengir/Tools/bishengir-compile/Options.td
bishengir/lib/Tools/bishengir-compile/regbase/PassPipeline.cpp
bishengir/lib/Tools/bishengir-compile/regbase/BiShengIRCompileMain.cpp
bishengir/lib/Tools/bishengir-compile/BiShengIRCompileConfig.cpp
```

`BiShengIRCompileConfig.cpp` only needs a manual change if validation or CLI
handling is not fully generated from `Options.td`.

Compatibility:

- keep `BISHENGIR_ENABLE_PTOAS_BRIDGE` temporarily for existing Planner tools;
- make the Triton integration use the explicit output option;
- migrate Planner extraction scripts later, after the new endpoint is stable.

Tests to add:

```text
bishengir/test/bishengir-compile/
```

The tests should verify successful output, the final pass boundary, absence of
later AVE-intrinsic/LLVM lowering, and failure diagnostics for unsupported VMI
conversion.

### Phase 2: Make PTOAS target metadata explicit

The current conversion clears source module attributes and hardcodes
`pto.target_arch = "a5"` and vector kernel kind. This is acceptable only for a
strict A5-only prototype.

Tasks:

1. Derive the PTOAS architecture from the original HACC target before source
   attributes are removed, or pass it explicitly through conversion options.
2. Initially accept only the confirmed mapping
   `Ascend910_9589 -> pto.target_arch = "a5"`.
3. Emit a clear diagnostic for unsupported targets.
4. Preserve or derive kernel-kind information instead of assuming vector once
   cube or mixed kernels enter scope.

Files to change in AscendNPU-IR:

```text
bishengir/lib/Conversion/HIVMAVEToPTOASVMI/HIVMAVEToPTOASVMI.cpp
bishengir/include/bishengir/Conversion/Passes.td
bishengir/lib/Tools/bishengir-compile/regbase/PassPipeline.cpp
bishengir/test/Conversion/HIVMAVEToPTOASVMI/preflight-metadata-container.mlir
```

Acceptance gate:

- A5 metadata is derived or explicitly selected;
- unsupported architecture values fail before PTOAS is invoked;
- the emitted module contains the PTOAS attributes required by PTOAS.

### Phase 3: Add raw device-object emission to PTOAS

Add a PTOAS CLI output mode such as:

```bash
ptoas \
  --pto-arch=a5 \
  --pto-backend=vpto \
  --emit-device-object \
  kernel.ptovmi.mlir \
  -o kernel.npubin
```

The exact option name should follow PTOAS CLI conventions. Its semantic
contract is more important than the spelling: output must be the raw merged
device ELF, without compiling or linking a host registration stub.

Tasks:

1. Add a distinct output kind for a device object.
2. Reuse the existing VPTO-to-LLVM and device compilation path.
3. Reuse the existing cube/vector device-object merge logic when needed.
4. Write the merged device object directly to `-o`.
5. Skip VPTO host-stub generation and fat-object compilation in this mode.
6. Preserve existing default PTOAS output behavior.

Files to change in PTOAS:

```text
tools/ptoas/ptoas.cpp
tools/ptoas/driver.cpp
tools/ptoas/ObjectEmission.cpp
tools/ptoas/ObjectEmission.h
test/
```

Prototype fallback:

Extracting `__aicore_rel_binary` from the existing host fat object may be used
to validate assumptions, but it should not become the production API. Section
extraction ties Triton to the internal host-object layout and makes binary
validation harder.

Acceptance gate:

- `file`/`readelf` identify the output as an AICore device ELF rather than an
  x86 host object;
- the expected kernel symbol is exported;
- the object registers through the CANN runtime;
- existing PTOAS fat-object tests continue to pass.

### Phase 4: Define a structured runtime metadata contract

Define the metadata that NPU-IR/PTOAS must expose to Triton. Prefer a small JSON
sidecar or an equally structured compiler output over regex parsing of compiler
logs.

Minimum schema:

```json
{
  "kernel_name": "vector_add_kernel",
  "kernel_kind": "aiv",
  "task_type": 1,
  "workspace_size": 0,
  "lock_num": 0,
  "lock_init_value": 0,
  "arguments": [
    {"kind": "sync_block_lock", "type": "ptr"},
    {"kind": "workspace", "type": "ptr"},
    {"kind": "user", "type": "ptr"},
    {"kind": "grid_x", "type": "i32"},
    {"kind": "grid_y", "type": "i32"},
    {"kind": "grid_z", "type": "i32"}
  ]
}
```

The exact schema can change, but each field must have one clear owner. Values
that are only known after PTOAS lowering should be emitted by PTOAS. Values
already finalized by NPU-IR should be attached to the VMI module or its
sidecar before source attributes are removed.

Files likely to change:

```text
AscendNPU-IR/bishengir/lib/Conversion/HIVMAVEToPTOASVMI/HIVMAVEToPTOASVMI.cpp
AscendNPU-IR/bishengir/lib/Tools/bishengir-compile/regbase/BiShengIRCompileMain.cpp
PTOAS/tools/ptoas/driver.cpp
PTOAS/tools/ptoas/ObjectEmission.cpp
triton/backends/ascend/compiler.py
```

Metadata that Triton can continue deriving before NPU-IR conversion:

- source kernel name;
- `mix_mode`/`parallel_mode`, subject to final PTOAS validation;
- `tt.tensor_kind` values from TTAdapter;
- original Triton signature and constexpr removal.

Metadata that must not silently default without proof:

- workspace size;
- sync-block-lock allocation and initialization;
- final hidden-argument ordering;
- task type;
- any PTOAS ABI symbol suffix.

Acceptance gate:

- the launcher receives explicit resource values;
- the final device entry ABI can be mechanically compared with Triton's packed
  argument layout;
- a kernel requiring workspace executes correctly, not only a zero-workspace
  vector add.

### Phase 5: Add the Triton Ascend flow selector

Extend `NPUOptions` with a compilation-flow field. Suggested values:

```text
npuir   existing native flow and default
ptoas   new NPU-IR-to-PTOAS flow
```

Resolution order:

1. explicit `compile_flow=` kernel launch option, if present;
2. `TRITON_ASCEND_COMPILE_FLOW` environment variable;
3. default `npuir`.

The resolved flow must be included in `NPUOptions.hash()` so native and PTOAS
objects never share a cache entry.

Files to change in the Triton Ascend source checkout:

```text
triton/backends/ascend/compiler.py
triton/backends/ascend/utils.py
docs/ENVIRONMENT.md
```

`utils.py` should also resolve the PTOAS executable through a documented option
such as `TRITON_PTOAS_PATH`. Tool discovery must not hardcode this Planner
workspace.

Validation behavior:

- reject unknown flow names immediately;
- report the missing `bishengir-compile` or PTOAS path with the attempted path;
- reject unsupported target/flow combinations before launching subprocesses;
- include compiler stderr in raised errors without scraping it for artifacts.

### Phase 6: Add Triton compilation stages for the PTOAS branch

Keep one Ascend backend and one `npubin` runtime type. In
`AscendBackend.add_stages()`, select one of two stage graphs.

Existing flow:

```text
ttir -> ttadapter -> npubin
```

New flow:

```text
ttir -> ttadapter -> ptovmi -> npubin
```

The `ptovmi` stage should:

1. parse source metadata from TTAdapter before conversion removes TT/HACC
   attributes;
2. write TTAdapter MLIR into a temporary directory;
3. invoke `bishengir-compile --emit-ptoas-vmi` with the same relevant NPU-IR
   optimization options as the native flow;
4. read and return the verified PTOAS VMI text;
5. retain it in the Triton cache/dump as `<kernel>.ptovmi`.

The PTOAS `npubin` stage should:

1. write the PTOAS VMI text into a temporary file;
2. invoke PTOAS in raw-device-object mode;
3. load the structured runtime metadata;
4. validate the output object kind and expected kernel symbol;
5. return the raw object bytes as Triton's existing `npubin` stage result.

Primary file to change:

```text
triton/backends/ascend/compiler.py
```

Supporting file:

```text
triton/backends/ascend/utils.py
```

The backend must keep:

```python
self.binary_ext = "npubin"
```

Changing the binary extension or creating a separate backend is unnecessary as
long as the final bytes satisfy the existing runtime contract.

### Phase 7: Reuse the existing host launcher

No functional driver changes should be planned initially. If the raw object,
symbol, metadata, and argument ABI contracts are correct, the existing path
should continue to perform:

```text
CompiledKernel._init_handles
  -> NPULauncher generation
  -> NPUUtils.load_binary
  -> rtDevBinaryRegister
  -> rtFunctionRegister
  -> generated launch wrapper
  -> rtKernelLaunch
```

Files expected to remain functionally unchanged:

```text
triton/backends/ascend/driver.py
triton/backends/ascend/npu_utils.cpp
```

These files may receive assertions, diagnostics, or focused tests. A functional
ABI branch in the driver should be added only if PTOAS cannot produce the same
entry contract. That outcome should trigger a design review because it weakens
the goal of using the same generated host code.

### Phase 8: Testing and rollout

Tests should be added from the narrowest contract to the full Python flow.

#### NPU-IR tests

- `--emit-ptoas-vmi` writes a verified module and exits successfully;
- no later AVE-intrinsic or LLVM lowering runs;
- unsupported conversion produces a focused diagnostic;
- target metadata is correct and unsupported targets fail.

Locations:

```text
AscendNPU-IR/bishengir/test/bishengir-compile/
AscendNPU-IR/bishengir/test/Conversion/HIVMAVEToPTOASVMI/
```

#### PTOAS tests

- raw device-object mode emits the expected object kind;
- vector and, later, mixed/cube object merging works;
- the host fat-object path remains unchanged;
- symbol and ABI naming are stable;
- failure leaves no successful-looking output file.

Location:

```text
PTOAS/test/
```

#### Triton Ascend tests

- default flow remains `npuir`;
- environment and per-kernel options select `ptoas` correctly;
- invalid flow values fail before compilation;
- flow value changes the cache key;
- mocked subprocess tests verify NPU-IR and PTOAS command construction;
- `.ptovmi` and `.npubin` artifacts are stored and dumped correctly;
- missing metadata is rejected rather than silently replaced with unsafe
  defaults.

Location in the Triton Ascend source checkout:

```text
python/test/unit/backends/ascend/      # use the actual repository convention
```

#### End-to-end tests

Use the same Python source and Torch comparison for both flows:

```bash
python3 bridge/triton-example/vector_add.py

TRITON_ASCEND_COMPILE_FLOW=ptoas \
python3 bridge/triton-example/vector_add.py
```

Expand in this order:

1. small vector add;
2. large vector add with multiple physical blocks;
3. a kernel requiring nonzero workspace;
4. row softmax;
5. RMSNorm or another representative reduction kernel;
6. sanitizer/debug mode where supported;
7. cache hit on a second Python execution;
8. real A5 hardware validation after simulator validation.

Planner fixtures:

```text
bridge/triton-example/
bridge/testcases/vadd/
bridge/testcases/vadd_large/
bridge/testcases/row_softmax/
bridge/testcases/rmsnorm_256/
bridge/testcases/rmsnorm_1600/
```

## 6. Recommended Implementation Order

The cross-repository changes should land in this order:

1. Prove the raw-object registration and launch ABI contract.
2. Add and test NPU-IR `--emit-ptoas-vmi`.
3. Add and test PTOAS raw device-object emission.
4. Define and test the structured runtime metadata contract.
5. Add Triton flow selection and cache separation.
6. Add Triton `ptovmi` and PTOAS `npubin` stages.
7. Run the existing launcher unchanged with vector add.
8. Validate workspace and multi-block kernels.
9. Update Planner scripts to use the first-class compiler endpoints.
10. Run regression tests for the original flow and perform A5 validation.

This order keeps each boundary independently testable. It also prevents the
Triton integration from depending on diagnostic-log extraction or an
unverified object representation.

## 7. Main Risks

| Risk | Consequence | Mitigation |
| --- | --- | --- |
| PTOAS emits only a host fat object | `rtDevBinaryRegister` rejects the bytes | Add a raw device-object output mode before Triton integration |
| PTOAS changes the public kernel symbol | `rtFunctionRegister` cannot find the entry | Record and validate symbol naming in the metadata contract |
| Hidden argument order differs | Kernel reads incorrect pointers/scalars | Compare the final PTOAS entry ABI against the generated packed struct |
| Workspace metadata is missing | Null or undersized workspace causes runtime failure | Require explicit metadata and test a nonzero-workspace kernel |
| NPU-IR VMI output is obtained from logs | Fragile success/failure handling and malformed cache artifacts | Add a supported compiler output mode |
| Flow is omitted from the cache key | Native and PTOAS binaries collide | Store flow in `NPUOptions` and its hash |
| Target is hardcoded to A5 | Incorrect objects may be generated for other devices | Validate target mapping before invoking PTOAS |
| PTOAS and embedded PTO dialect revisions diverge | PTOAS rejects NPU-IR-produced MLIR | Pin/test the bridge dialect contract and add a version compatibility check |
| Existing driver is changed too early | Two launch ABIs develop and host-code reuse is lost | Treat driver reuse as an acceptance contract and review any exception |

## 8. Completion Criteria

The work is complete when all of the following are true:

- `python3 kernel.py` still uses the existing flow by default;
- `TRITON_ASCEND_COMPILE_FLOW=ptoas python3 kernel.py` compiles and launches the
  same kernel without a separate host executable;
- NPU-IR emits PTOAS VMI through a supported output mode;
- PTOAS emits a raw runtime-loadable AICore device object;
- the existing `NPULauncher` and `NPUUtils.load_binary` path is reused;
- kernel symbol, argument ABI, workspace, lock, and task metadata are explicit
  and validated;
- native and PTOAS flows use separate cache entries;
- vector add, a multi-block case, and a nonzero-workspace case pass numerical
  comparison;
- the original flow passes regression tests;
- simulator results and real A5 results are recorded separately.

## 9. Open Decisions

The following decisions should be resolved during the corresponding phase,
not guessed in the Triton driver:

1. What exact PTOAS CLI name should select raw device-object output?
2. Should runtime metadata be emitted by NPU-IR, PTOAS, or as a merged manifest
   assembled by the Triton stage?
3. Does PTOAS final ABI naming preserve the original Triton kernel symbol, or
   should the manifest carry a separate device-entry symbol?
4. Which component owns final workspace-size calculation for bridge-generated
   kernels?
5. Can the NPU-IR sync-block-lock contract be preserved directly, or should
   PTOAS own synchronization for the new path?
6. Should the first release expose only the environment selector, or both the
   environment and per-kernel launch option?
7. Which Triton Ascend source revision is compatible with the current
   AscendNPU-IR and PTOAS revisions?

