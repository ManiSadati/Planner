# Triton PTOAS Launch Contract Artifact Manifest

Date: 2026-08-26

## Source Revisions

| Repository | Revision |
| --- | --- |
| AscendNPU-IR | `f95e4cef4d2cfa0520e894e96c96197e9207d13f` |
| PTOAS | `e4be6616eb4f19364e77ffe9641c10fa098b49c2` |

## Frozen Artifacts

| Role | Path | SHA256 | File Type |
| --- | --- | --- | --- |
| NPU-IR baseline raw device object | `/home/m00967009/Workspace/Planner/bridge/triton-example/vector_add_large_npuir_baseline_aicore.o` | `5fa18c83dfe3feb44b7dad61d23cc913b6bbe3140b1db803d2dd16d1c2f08ee2` | ELF64 executable, machine `0x1029`, statically linked |
| PTOAS bridge raw device object | `/home/m00967009/Workspace/Planner/bridge/triton-example/vector_add_large_ptoas_bridge_aicore.o` | `272c31dd444eb8c25f60633f587b73d54fff28f4d554dd00723e903682d8c990` | ELF64 executable, machine `0x1029`, statically linked |
| PTOAS host fat object control | `/home/m00967009/Workspace/Planner/bridge/testcases/vadd_large/vector_add_large_kernel_ptoas_fatobj.o` | `c3e04d2a27112917ff7501a9243d09392cfd9d099b726c5ff7fee68953e47fc9` | ELF64 relocatable, x86-64 |

## Observable Environment

| Item | Value |
| --- | --- |
| `ASCEND_HOME` | unset |
| `ASCEND_TOOLKIT_HOME` | unset |
| `LD_LIBRARY_PATH` | unset |
| `npu-smi` | not found on `PATH` |

The missing runtime environment data should be filled in before using this manifest as a formal regression baseline on a device or simulator machine.
