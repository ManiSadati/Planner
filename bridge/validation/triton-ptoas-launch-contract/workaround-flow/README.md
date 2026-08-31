# PTOAS Fat Object Extraction Workaround

Date: 2026-08-26

This directory records a no-PTOAS-source-change workaround for recovering a
runtime-loadable AICore object from a PTOAS host fat object.

## Inputs

Generated PTOAS VMI from the new NPU-IR endpoint:

```bash
AscendNPU-IR/build/bin/bishengir-compile \
  Planner/bridge/testcases/vadd/compile-input.mlir \
  --emit-ptoas-vmi \
  --target=Ascend910_9589 \
  --enable-auto-multi-buffer=true \
  --enable-auto-bind-sub-block=true \
  --disable-ffts \
  --limit-auto-multi-buffer-of-local-buffer=no-limit \
  --enable-auto-blockify-loop \
  --enable-hfusion-compile=true \
  --enable-hivm-compile=true \
  --enable-triton-kernel-compile=true \
  --mlir-disable-threading \
  --mlir-print-stacktrace-on-diagnostic \
  --enable-vf-merge-level=1 \
  -o Planner/bridge/validation/triton-ptoas-launch-contract/workaround-flow/vadd.ptoas-vmi.mlir
```

Result: success. The emitted MLIR contains PTOAS VMI and MTE operations.

## Complete Fresh Flow

### Generate the PTOAS host fat object

The CANN installation must contain the BiSheng frontend at
`tools/bisheng_compiler/bin/bisheng`. The installed PTOAS Python extension also
requires the LLVM shared-library directory in `LD_LIBRARY_PATH`.

```bash
export ASCEND_HOME_PATH=/home/a84369921/Ascend/cann-9.1.0-beta.3
export PATH="$ASCEND_HOME_PATH/x86_64-linux/bin:$ASCEND_HOME_PATH/tools/bisheng_compiler/bin:$PATH"
export LD_LIBRARY_PATH="/home/m00967009/Workspace/llvm-project/build-shared/lib:/home/m00967009/Workspace/PTOAS/install/lib:/home/m00967009/Workspace/AscendNPU-IR/build/lib:$ASCEND_HOME_PATH/x86_64-linux/lib64:${LD_LIBRARY_PATH:-}"

PTOAS/install/bin/ptoas \
  --pto-backend=vpto \
  --pto-arch=a5 \
  Planner/bridge/validation/triton-ptoas-launch-contract/workaround-flow/vadd.ptoas-vmi.mlir \
  -o Planner/bridge/validation/triton-ptoas-launch-contract/workaround-flow/vadd.ptoas-fatojb.o
```

Result: success. The generated artifact is an x86-64 relocatable host object
containing an `__aicore_rel_binary` section. The filename preserves the
`fatojb` spelling used for the validated artifact.

### Extract the AICore relocatable payload

```bash
WORKAROUND_DIR=/home/m00967009/Workspace/Planner/bridge/validation/triton-ptoas-launch-contract/workaround-flow

objcopy \
  --dump-section __aicore_rel_binary="$WORKAROUND_DIR/vadd.extracted-aicore-rel.o" \
  "$WORKAROUND_DIR/vadd.ptoas-fatojb.o" \
  "$WORKAROUND_DIR/vadd.ptoas-fat.copy.o"
```

`objcopy` writes a copy of the host object because its command form requires an
output object. Triton only consumes the extracted payload after the final link.

### Link the raw AICore executable

```bash
$ASCEND_HOME_PATH/tools/bisheng_compiler/bin/ld.lld \
  -m aicorelinux \
  -Ttext 0 \
  "$WORKAROUND_DIR/vadd.extracted-aicore-rel.o" \
  --allow-multiple-definition \
  -o "$WORKAROUND_DIR/vadd.npubin"
```

## Validation Commands

All commands run from `/home/m00967009/Workspace` after setting
`WORKAROUND_DIR` as above.

```bash
file \
  "$WORKAROUND_DIR/vadd.ptoas-fatojb.o" \
  "$WORKAROUND_DIR/vadd.extracted-aicore-rel.o" \
  "$WORKAROUND_DIR/vadd.npubin"

readelf -h -l "$WORKAROUND_DIR/vadd.npubin"
readelf -sW "$WORKAROUND_DIR/vadd.npubin"
readelf -SW "$WORKAROUND_DIR/vadd.npubin"
readelf -x __CCE_KernelArgSize "$WORKAROUND_DIR/vadd.npubin"
readelf -rW "$WORKAROUND_DIR/vadd.npubin"

sha256sum \
  "$WORKAROUND_DIR/vadd.ptoas-vmi.mlir" \
  "$WORKAROUND_DIR/vadd.ptoas-fatojb.o" \
  "$WORKAROUND_DIR/vadd.extracted-aicore-rel.o" \
  "$WORKAROUND_DIR/vadd.npubin"
```

## Fresh-Flow Results

Artifact identities:

```text
vadd.ptoas-fatojb.o:
  ELF64 relocatable, x86-64

vadd.extracted-aicore-rel.o:
  ELF64 relocatable, machine 0x1029

vadd.npubin:
  ELF64 executable, machine 0x1029, statically linked
```

The final object has the expected loader shape:

```text
Type:                  EXEC
Machine:               0x1029
Flags:                 0x990000
Entry point:           0x0
Program headers:       2
LOAD offset:           0x0000b0
LOAD file/memory size: 0x228
LOAD flags:            R E
LOAD alignment:        0x1000
Relocations:           none
```

The exported PTOAS kernel entry is:

```text
vector_add_kernel_mix_aiv
```

The object contains eight `.ParamInfo_*` records. Its packed kernel argument
size is `0x38` bytes:

```text
Hex dump of section '__CCE_KernelArgSize':
  0x00000000 38000000
```

## Runtime Metadata and Launch ABI Contract

This section records the metadata contract that the future Triton PTOAS stage
must satisfy before it can reuse the existing Ascend launcher unchanged.

### Runtime-facing metadata

The existing Triton Ascend loader receives `metadata.name` as two whitespace
separated fields:

```text
<logical_kernel_name> <mix_mode>
```

For this artifact the values are:

```text
logical_kernel_name: vector_add_kernel
mix_mode:            aiv
runtime symbol:      vector_add_kernel_mix_aiv
```

`triton/backends/ascend/driver.py` passes only the logical kernel name to
`load_kernel_binary`. `npu_utils.cpp` then calls:

```text
rtDevBinaryRegister(raw_npubin)
rtFunctionRegister(devbinHandle, func_stub_handle, stubName, logical_kernel_name, 0)
```

Therefore the PTOAS-produced `npubin` must keep CANN-visible metadata that maps
the logical kernel name to the ABI-renamed device entry symbol. The final linked
object satisfies that condition because it exports `vector_add_kernel_mix_aiv`
and contains `.ascend.meta.vector_add_kernel_mix_aiv`.

The Python metadata required by the launcher is:

```text
metadata.name          = "vector_add_kernel aiv"
metadata.kernel_name   = "vector_add_kernel"
metadata.mix_mode      = "aiv"
metadata.parallel_mode = "simd"
metadata.tensor_kinds  = [0, 0, 1]
metadata.shared        = 1
metadata.workspace_size = 0 or absent
metadata.lock_num       = 0 or absent
metadata.lock_init_value = 0
metadata.bs_task_type   = 0 or absent
metadata.required_ub_bits = 0 or parsed compiler value
```

For this vadd artifact, no runtime workspace or sync-block-lock storage is
required. The launcher still emits the two implicit pointer fields in normal
NPU mode, but fills them with `nullptr` when `workspace_size <= 0` and
`lock_num <= 0`.

One existing code mismatch should be handled during integration: the native
compiler path stores the sync lock initializer as `lock_init_val`, while
`driver.py` reads `lock_init_value`. The new PTOAS path should use
`lock_init_value` because that is the launcher-facing key.

### Packed argument layout

The PTOAS VMI entry function is:

```text
func.func @vector_add_kernel(
  %arg0: !pto.ptr<ui8, gm>,  // sync-block lock
  %arg1: !pto.ptr<ui8, gm>,  // workspace
  %arg2: !pto.ptr<f32, gm>,  // x
  %arg3: !pto.ptr<f32, gm>,  // y
  %arg4: !pto.ptr<f32, gm>,  // output
  %arg5: i32,                // grid X
  %arg6: i32,                // grid Y
  %arg7: i32                 // grid Z
)
```

The existing generated launcher builds this packed struct for non-SIMT NPU
kernels:

| Index | Field | Offset | Size | Source |
| --- | --- | ---: | ---: | --- |
| 0 | `syncBlockLock` | `0x00` | 8 | launcher implicit |
| 1 | `workspace_addr` | `0x08` | 8 | launcher implicit |
| 2 | user pointer arg0 | `0x10` | 8 | Triton signature |
| 3 | user pointer arg1 | `0x18` | 8 | Triton signature |
| 4 | user pointer arg2 | `0x20` | 8 | Triton signature |
| 5 | `gridX` | `0x28` | 4 | launcher grid |
| 6 | `gridY` | `0x2c` | 4 | launcher grid |
| 7 | `gridZ` | `0x30` | 4 | launcher grid |
| - | tail padding | `0x34` | 4 | struct alignment |

The final `npubin` agrees with this layout:

```text
__CCE_KernelArgSize = 0x38

.ParamSummary_vector_add_kernel_mix_aiv:
  record kind: 0x000c0010
  param count: 8
  last argument end offset: 0x34

.ParamInfo_vector_add_kernel_mix_aiv_0: offset 0x00, size 8
.ParamInfo_vector_add_kernel_mix_aiv_1: offset 0x08, size 8
.ParamInfo_vector_add_kernel_mix_aiv_2: offset 0x10, size 8
.ParamInfo_vector_add_kernel_mix_aiv_3: offset 0x18, size 8
.ParamInfo_vector_add_kernel_mix_aiv_4: offset 0x20, size 8
.ParamInfo_vector_add_kernel_mix_aiv_5: offset 0x28, size 4
.ParamInfo_vector_add_kernel_mix_aiv_6: offset 0x2c, size 4
.ParamInfo_vector_add_kernel_mix_aiv_7: offset 0x30, size 4
```

The relevant kernel-kind record is present at
`.AIV_Kernel_Type_vector_add_kernel_mix_aiv`, and the module-level VMI input
attributes are:

```text
pto.kernel_kind = #pto.kernel_kind<vector>
pto.target_arch = "a5"
```

### Integration gate

Before adding the Triton flow selector, the PTOAS compilation stage must:

1. Return the final raw AICore executable bytes, not the host fat object.
2. Set `metadata.name` to `<logical_kernel_name> <mix_mode>`.
3. Preserve the logical kernel name expected by `rtFunctionRegister`.
4. Set launcher metadata fields explicitly, especially `parallel_mode`,
   `workspace_size`, `lock_num`, `lock_init_value`, and `tensor_kinds`.
5. Verify `__CCE_KernelArgSize` and `.ParamInfo_*` offsets against the launcher
   struct for at least this vadd testcase.

Artifact hashes:

| Artifact | SHA256 |
| --- | --- |
| `vadd.ptoas-vmi.mlir` | `9c933aa0eeeba7bd52849a219b85d71888c88294a228e90533c99f7b55c32eff` |
| `vadd.ptoas-fatojb.o` | `ce0e50e07c512cd75994e590390d12333d227e4a492a2d029dc3952a632c9ae2` |
| `vadd.extracted-aicore-rel.o` | `3ec21e9a8aed9ccb89e55e89fc82f822db951751b012d1a983157ac060f4f6ec` |
| `vadd.npubin` | `bc3eb998ab00e1c3982201ec6264b6f2858ea9bba593c3578c9baad49676f06b` |

## Conclusion

The fresh chain from NPU-IR PTOAS VMI output through PTOAS fat-object emission,
section extraction, and final AICore linking succeeds. The resulting
`vadd.npubin` satisfies the statically inspectable object and symbol contract
needed by Triton's existing loader.

This remains a temporary integration mechanism because it depends on PTOAS's
internal `__aicore_rel_binary` section name and an external final link. Runtime
registration and launch are not proven on this server because it has no
attached NPU; the available VPTO simulator does not exercise CANN runtime
binary registration.

The earlier frozen `vadd_large` experiment established the same extraction and
link behavior before fresh PTOAS fat-object generation became available. Its
artifacts remain in the parent validation fixture for comparison.
