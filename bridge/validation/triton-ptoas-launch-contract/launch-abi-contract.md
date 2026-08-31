# Vector Add Large Launch ABI Contract

## Source Evidence

| Evidence | File |
| --- | --- |
| Triton launch grid `(rows,)`, with `rows = 1000` | `/home/m00967009/Workspace/Planner/bridge/triton-example/vector_add_large.py` |
| Manual host fixture argument order and physical block count | `/home/m00967009/Workspace/Planner/bridge/testcases/vadd_large/launch.cpp` |
| Packed argument size, NPU-IR baseline | `npuir-baseline.arg-size-hex.txt` |
| Packed argument size, PTOAS bridge | `ptoas-bridge.arg-size-hex.txt` |
| Raw metadata, NPU-IR baseline | `npuir-baseline.ascend-meta-hex.txt` |
| Raw metadata, PTOAS bridge | `ptoas-bridge.ascend-meta-hex.txt` |

Both raw objects report `__CCE_KernelArgSize = 0x38` bytes.

## Packed Arguments

| Index | Name in Host Fixture | Semantic Role | Offset | Size | Alignment Assumption | Runtime Value for Fixture |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | `workspace0` | sync-lock/workspace hidden pointer | `0x00` | 8 | 8 | device pointer, 1-byte allocation in current fixture |
| 1 | `workspace1` | workspace hidden pointer | `0x08` | 8 | 8 | device pointer, 1-byte allocation in current fixture |
| 2 | `input0` | user lhs tensor pointer | `0x10` | 8 | 8 | device pointer to `lhs` |
| 3 | `input1` | user rhs tensor pointer | `0x18` | 8 | 8 | device pointer to `rhs` |
| 4 | `output` | user output tensor pointer | `0x20` | 8 | 8 | device pointer to `out` |
| 5 | `tileCount` | logical grid X / logical tile count | `0x28` | 4 | 4 | `1000` |
| 6 | `outerCount` | logical grid Y | `0x2c` | 4 | 4 | `1` |
| 7 | `innerCount` | logical grid Z | `0x30` | 4 | 4 | `1` |

The total payload ends at byte `0x34`; the ABI section reports `0x38`, so the packed structure includes 4 bytes of tail padding/alignment.

## Launch Dimensions

| Dimension | Value |
| --- | --- |
| Logical Triton grid | `(1000,)` |
| Packed logical grid args | `tileCount = 1000`, `outerCount = 1`, `innerCount = 1` |
| Physical runtime `blockNum` in fixture | `8` |

The direct probe should preserve this split: pass `blockNum = 8` to `rtKernelLaunch`, while packing `1000, 1, 1` as the final three kernel arguments.

## Metadata Notes

The raw metadata sections contain one `.ParamSummary_*` entry and eight `.ParamInfo_*` entries for both the NPU-IR baseline and PTOAS bridge objects. The observed parameter offsets and sizes match the fixture C++ signature.

For this kernel, the two hidden workspace-like allocations appear unused by the algorithm, but the ABI still requires non-null device pointers at indexes 0 and 1. This does not generalize to kernels that require real workspace or lock initialization.
