# Negative Controls

Date: 2026-08-26

## Planned Runtime Controls

| Control | Command Intent | Expected Failure When ACL Initializes | Current Result |
| --- | --- | --- | --- |
| PTOAS raw object with logical symbol | Register `vector_add_large_ptoas_bridge_aicore.o` as `vector_add_large_kernel` | `rtFunctionRegister` should reject or fail to resolve the missing raw-object symbol | blocked at `aclInit failed, errorCode=500000` |
| Native raw object with PTOAS symbol | Register `vector_add_large_npuir_baseline_aicore.o` as `vector_add_large_kernel_mix_aiv` | `rtFunctionRegister` should reject or fail to resolve the missing raw-object symbol | blocked at `aclInit failed, errorCode=500000` |
| PTOAS host fat object as raw object | Register `vector_add_large_kernel_ptoas_fatobj.o` through `rtDevBinaryRegister` with `RT_DEV_BINARY_MAGIC_ELF_AIVEC` | `rtDevBinaryRegister` should reject the x86-64 relocatable host object | blocked at `aclInit failed, errorCode=500000` |

## Static Negative Evidence

The fat-object control is not a raw Ascend device executable:

| Artifact | ELF Type | Machine | Expected Raw Loader Outcome |
| --- | --- | --- | --- |
| `vector_add_large_kernel_ptoas_fatobj.o` | `REL` | x86-64 | reject before function registration |

By contrast, both raw-object candidates are `EXEC` ELFs for machine `0x1029` with flags `0x990000`.

## Probe Error Boundary

`raw_object_probe` reports failures at separate stages:

| Stage | Probe Message Prefix |
| --- | --- |
| Runtime initialization | `aclInit failed` / `aclrtSetDevice failed` |
| Raw binary registration | `rtDevBinaryRegister failed` |
| Function registration | `rtFunctionRegister failed` |
| Kernel launch | `rtKernelLaunch failed` |
| Stream/copy/compare | `aclrtSynchronizeStream failed`, `aclrtMemcpy(...) failed`, or `compare failed` |

The current session only validates the first boundary because ACL initialization fails for both the new raw-object probe and the existing fat-object fixture.
