# Kernel Symbol Contract

## Observed Symbols

| Artifact | Raw Device Entry Symbol | Logical Host Name |
| --- | --- | --- |
| NPU-IR baseline raw object | `vector_add_large_kernel` | `vector_add_large_kernel` |
| PTOAS bridge raw object | `vector_add_large_kernel_mix_aiv` | `vector_add_large_kernel` |
| PTOAS host fat object | device payload uses `vector_add_large_kernel_mix_aiv`; host wrapper exports `vector_add_large_kernel` | `vector_add_large_kernel` |

## PTOAS Naming Source

Relevant files:

| File | Lines | Meaning |
| --- | --- | --- |
| `/home/m00967009/Workspace/PTOAS/tools/ptoas/ObjectEmission.cpp` | 924-930 | Chooses VPTO vector public ABI suffix: `.vector` for newer CANN, `_mix_aiv` for older CANN. |
| `/home/m00967009/Workspace/PTOAS/tools/ptoas/ObjectEmission.cpp` | 1056-1069 | Renames external LLVM kernel functions by appending the selected ABI suffix. |
| `/home/m00967009/Workspace/PTOAS/tools/ptoas/VPTOHostStubEmission.cpp` | 34-41 | Strips `_mix_aiv` / `_mix_aic` to recover the logical host wrapper name. |

## Driver Implication

The new Triton driver path must distinguish two names:

| Metadata Field | Value for Current PTOAS Raw Object | Used For |
| --- | --- | --- |
| `logical_kernel_name` | `vector_add_large_kernel` | Python/Triton cache key, user-visible kernel name, wrapper naming |
| `device_entry_name` | `vector_add_large_kernel_mix_aiv` | `rtFunctionRegister` and launch registration |

For the current PTOAS bridge object, `rtFunctionRegister` should use `vector_add_large_kernel_mix_aiv`. Registering only `vector_add_large_kernel` would test the wrong symbol and is expected to fail for the raw PTOAS object.

Future driver integration should not infer this name from a hardcoded suffix. The compiler artifact metadata should carry the exact `device_entry_name`, because the suffix may be `_mix_aiv`, `.vector`, `_mix_aic`, or `.cube` depending on target and CANN ABI mode.
