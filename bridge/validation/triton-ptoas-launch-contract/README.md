# Triton PTOAS Launch Contract Validation

Date: 2026-08-26

## Scope

This validation checks whether a PTOAS-emitted raw Ascend device object can be made compatible with the same Triton Ascend host-launch mechanism that currently launches Triton-native NPU-IR objects.

The fixture is `vector_add_large_kernel` from `/home/m00967009/Workspace/Planner/bridge/triton-example/vector_add_large.py`.

## Artifacts

See `artifact-manifest.md` for paths, hashes, source revisions, and observed environment values.

The three frozen comparison artifacts are:

| Role | Path |
| --- | --- |
| NPU-IR baseline raw object | `/home/m00967009/Workspace/Planner/bridge/triton-example/vector_add_large_npuir_baseline_aicore.o` |
| PTOAS bridge raw object | `/home/m00967009/Workspace/Planner/bridge/triton-example/vector_add_large_ptoas_bridge_aicore.o` |
| PTOAS host fat object negative control | `/home/m00967009/Workspace/Planner/bridge/testcases/vadd_large/vector_add_large_kernel_ptoas_fatobj.o` |

## Findings

1. The NPU-IR baseline and PTOAS bridge artifacts are both raw Ascend device executable ELFs:
   - ELF type: `EXEC`
   - machine: `0x1029`
   - flags: `0x990000`

2. The PTOAS fat object is not suitable for Triton raw-binary loading:
   - ELF type: `REL`
   - machine: x86-64
   - contains host wrapper/registration sections such as `__aicore_rel_binary`

3. The main raw-object contract difference is symbol naming:
   - native raw object exports `vector_add_large_kernel`
   - PTOAS raw object exports `vector_add_large_kernel_mix_aiv`
   - PTOAS host-stub generation strips `_mix_aiv` back to `vector_add_large_kernel` for host wrapper naming

4. The packed launch ABI for this fixture is 8 arguments and 56 bytes:
   - two hidden workspace-like pointers
   - three user tensor pointers
   - three `int32_t` logical grid values
   - total `__CCE_KernelArgSize = 0x38`

5. Runtime equivalence could not be completed in this session because `aclInit` fails with error `500000` for both:
   - the new direct raw-object probe
   - the existing `vadd_large/run_npu_from_fatobj.sh` fixture control

This means the current environment cannot yet prove `rtDevBinaryRegister`, `rtFunctionRegister`, or `rtKernelLaunch` behavior. The source/build side of the validation is complete; device/simulator execution remains pending.

## Probe

Probe source:

`/home/m00967009/Workspace/Planner/bridge/testcases/vadd_large/raw_object_probe.cpp`

The probe supports:

```sh
raw_object_probe --object <path> --symbol <device_entry_name> --register-only
raw_object_probe --object <path> --symbol <device_entry_name> --launch
```

Expected positive commands once ACL initialization works:

```sh
./raw_object_probe \
  --object /home/m00967009/Workspace/Planner/bridge/triton-example/vector_add_large_npuir_baseline_aicore.o \
  --symbol vector_add_large_kernel \
  --launch

./raw_object_probe \
  --object /home/m00967009/Workspace/Planner/bridge/triton-example/vector_add_large_ptoas_bridge_aicore.o \
  --symbol vector_add_large_kernel_mix_aiv \
  --launch
```

Expected negative commands once ACL initialization works:

```sh
./raw_object_probe \
  --object /home/m00967009/Workspace/Planner/bridge/triton-example/vector_add_large_ptoas_bridge_aicore.o \
  --symbol vector_add_large_kernel \
  --register-only

./raw_object_probe \
  --object /home/m00967009/Workspace/Planner/bridge/triton-example/vector_add_large_npuir_baseline_aicore.o \
  --symbol vector_add_large_kernel_mix_aiv \
  --register-only

./raw_object_probe \
  --object /home/m00967009/Workspace/Planner/bridge/testcases/vadd_large/vector_add_large_kernel_ptoas_fatobj.o \
  --symbol vector_add_large_kernel \
  --register-only
```

## Driver Contract

The future Triton driver integration should carry at least these separate fields in compiler metadata:

| Field | Native Flow Value | PTOAS Flow Value |
| --- | --- | --- |
| `logical_kernel_name` | `vector_add_large_kernel` | `vector_add_large_kernel` |
| `device_entry_name` | `vector_add_large_kernel` | `vector_add_large_kernel_mix_aiv` |
| `binary_format` | raw Ascend device ELF | raw Ascend device ELF |
| `binary_magic` | `RT_DEV_BINARY_MAGIC_ELF_AIVEC` | `RT_DEV_BINARY_MAGIC_ELF_AIVEC` |
| `arg_size` | `0x38` | `0x38` |
| `physical_block_count` for fixture | `8` | `8` |

The driver should use `device_entry_name` for `rtFunctionRegister` and `rtKernelLaunch`, while preserving `logical_kernel_name` for Python/Triton-facing identity.

## Files In This Directory

| File | Purpose |
| --- | --- |
| `artifact-manifest.md` | Frozen artifact paths, hashes, repository revisions, and observed environment |
| `static-elf-comparison.md` | ELF header/section/symbol comparison summary |
| `symbol-contract.md` | PTOAS symbol suffix source trace and driver naming implication |
| `launch-abi-contract.md` | Packed argument order, offsets, sizes, and launch dimensions |
| `probe-run-results.md` | Build/run attempts and current `aclInit` blocker |
| `negative-controls.md` | Negative-control matrix and current execution status |
| `*.readelf-*.txt` | Raw static `readelf` reports |
| `*.ascend-meta-hex.txt` | Raw Ascend metadata hex dumps |
| `*.arg-size-hex.txt` | Raw `__CCE_KernelArgSize` hex dumps |

## Completion Gate

Step 1 is fully complete only after a configured device or simulator environment can run the probe and show:

1. native raw object launches and compares successfully;
2. PTOAS raw object launches and compares successfully using `vector_add_large_kernel_mix_aiv`;
3. wrong-symbol registrations fail at `rtFunctionRegister`;
4. the PTOAS host fat object fails at `rtDevBinaryRegister`.

The current session completed all static, source, and build work, but runtime execution is blocked before registration by `aclInit failed, errorCode=500000`.
