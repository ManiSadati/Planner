# TODO: Simplify MmadL1 PTODSL template signatures

**Priority:** Low / deferred  
**Status:** Not started  
**Area:** AscendNPU-IR · PTODSL · MmadL1

## Problem

The typed MmadL1 PTODSL wrappers (for example, `mmadl1_f16_f32`) each declare **24 rank-4 metadata parameters** in addition to the 14 parameters used by `_mmadl1_body()`:

- `a_size_0`–`a_size_3`, `a_stride_0`–`a_stride_3`
- `b_size_0`–`b_size_3`, `b_stride_0`–`b_stride_3`
- `c_size_0`–`c_size_3`, `c_stride_0`–`c_stride_3`

These parameters describe tensor shapes and strides, but the current wrappers do not use or forward them. They make the function definitions long and repetitive across specializations.

## Proposed follow-up

Investigate a cleaner way to express the metadata arguments while **preserving the compiler-facing MLIR function signature**. Possibilities include `*metadata`, a reusable signature-generation mechanism, or another PTODSL-supported approach. Do not assume Python variadic parameters (`*args`/`**kwargs`) work under `@pto.func`: the compiler may require explicitly typed, fixed arguments.

## Acceptance criteria

- All MmadL1 specializations retain their existing externally visible symbols and compatible MLIR argument types/order.
- Metadata handling is less repetitive or a documented technical limitation explains why explicit parameters are required.
- Explicit-instantiation generation and existing PTODSL compilation tests still pass; compare generated signatures before and after.

## Relevant code

- `bishengir/lib/Template/lib/RegBase/Cube/mmadl1_64_ptodsl.py`: typed MmadL1 wrappers and `_mmadl1_body()`.
- `bishengir/lib/Template/lib/RegBase/Cube/cube_ptodsl_utils.py`: probe generation and metadata arguments (`*a_metadata`, `*b_metadata`, `*c_metadata`).

**Note:** This is a readability/refactoring task, not a known numerical correctness bug. Defer until higher-priority template work is complete.
