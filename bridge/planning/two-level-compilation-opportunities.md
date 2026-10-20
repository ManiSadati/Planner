# Key Technologies Enabled by NPU-IR and PTOAS

Date: 2026-10-19

Status: proposed research and development opportunities, not implemented
capabilities or an approved change to the current roadmap.

## Executive Summary

The strongest reason to combine NPU-IR and PTOAS is coordinated optimization
from tensor fusion and tiling down to register usage, data movement, and
hardware synchronization. Replacing CCE is an architectural foundation, not
by itself a performance improvement.

These technologies are not fundamentally impossible in either compiler alone.
The practical benefit is combining their strengths without rebuilding the
missing half: NPU-IR contributes tensor-level computation and scheduling
context; PTOAS exposes lower-level PTO/VMI/VPTO implementation decisions.

## 1. Register-Level Fusion Within Tensor-Fused Kernels

Example:

```text
Matmul -> scaling -> bias -> activation -> conversion -> store
```

NPU-IR identifies the fused computation and dependencies. PTOAS can optimize
the expanded vector implementation across template boundaries, when the
relevant bodies, alias information, and dependencies are visible.

The opportunity is to retain intermediate vector results in registers rather
than repeatedly storing and reloading them through UB. Necessary transfers
between Cube accumulator storage and the Vector side still remain.

Tensor-level fusion does not guarantee efficient register-level execution;
instruction optimization alone does not choose which tensor operations should
share a kernel. The combination connects those decisions.

Measure intermediate UB loads/stores, register pressure and spills, instruction
counts, and end-to-end execution time. Removing memory traffic is not a win if
it creates enough register pressure to make the kernel slower.

## 2. Joint Cube, Vector, and DMA Pipelining

Example: FlashAttention's `QK^T -> softmax -> PV` computation.

NPU-IR already supports Cube/Vector software pipelining, including a preload
mode intended for FlashAttention. The bridge must not claim to invent this
capability. See the [native pipeline documentation](https://ascendnpu-ir.gitcode.com/en/developer_guide/features/cv_pipelining.html).

The additional opportunity is coordinating the outer pipeline with the actual
expanded implementation: internal MmadL1 K slicing, L0 buffering, vector
instruction costs, DMA completion, Fixpipe, and unit-flag dependencies.

An outer schedule that looks balanced at tensor level can become unbalanced
after template expansion. Implementation-aware pipeline depth and buffering
could reduce idle periods and synchronization stalls. This needs explicit
dependency and resource ownership, not independently applied synchronization
passes from both compilers.

## 3. End-to-End Layout Optimization

The `qk_matmul` investigation provides a motivating example: a logical
transpose can introduce multiple movement and packing stages.

The proposed optimization scope is:

```text
GM input -> DMA/packing -> Cube layout -> Fixpipe -> vector-register layout
```

Choose layouts to minimize the whole conversion sequence rather than
optimizing each stage separately. NPU-IR contributes the logical access and
producer/consumer requirements; PTOAS contributes the instruction-level
consequences of alternative register layouts and transfers.

PTOAS developers report that the best vector layout depends on the balance
between computation and stores, not simply the number of layout conversions.
That supports investigating context-aware choices, but does not prove a gain
for our bridge. See [PTOAS issue #1391](https://github.com/hw-native-sys/PTOAS/issues/1391).

Measure packing/transposition instructions, bytes moved, temporary storage,
and execution time, including costs moved into another pipeline stage.

## 4. Hardware-Informed Tiling and Autotuning

NPU-IR chooses tensor tiles and core distribution. PTOAS can expose the
consequences of those choices: register pressure, instructions, scratch
requirements, and scheduling costs.

A proposed feedback loop would search `BLOCK_M/N/K`, pipeline depth,
buffering, and core distribution using final implementation costs. A tile
that reduces DMA traffic may increase spills or force inefficient internal
K partitioning; optimizing only one level can miss that tradeoff.

This requires a deliberate feedback interface and a validated cost model.
Connecting the compilers does not automatically provide either. Evaluate
search cost as well as kernel performance, and validate predictions on A5
hardware rather than treating simulator ticks as authoritative hardware time.

## 5. Fused Low-Precision Compute and Conversion

Example: quantized GEMM with scaling, activation, and output conversion.

NPU-IR retains mathematical semantics and precision requirements. PTOAS
exposes where hardware conversions and arithmetic actually occur. Together,
they could avoid unnecessary format round trips and place conversion/scaling
at suitable DMA, Fixpipe, or vector stages.

Preserve rounding, saturation, scale granularity, and accumulation precision.
This is a datatype-specific opportunity, not a claim that all low-precision
formats or conversion modes are already supported by the bridge.

## Architectural Conditions

- Template bodies must become visible PTO operations. Opaque native CCE calls
  remain useful compatibility fallbacks, but limit optimization at the PTO
  stage across those bodies. This is not a claim that CCE cannot optimize them.
- Preserve logical extents, physical layouts, offsets, dependency semantics,
  precision, and useful alias information at the handoff.
- The current late handoff inherits NPU-IR's memory and synchronization plan.
  Register-local optimizations may fit there; changing global layouts,
  buffering, or tiling can require an earlier handoff or an explicit feedback
  and replanning mechanism.
- Keep one owner for each memory and synchronization decision. Do not enable
  competing planners and assume that their results compose correctly.
- Validate semantic equivalence before making performance claims. Existing
  template test results do not establish these proposed optimization gains.

## Suggested First Demonstrator

Start with a fused GEMM epilogue, followed by FlashAttention if the smaller
experiment demonstrates value. This is a proposal, not authorization to begin
implementation or runtime sweeps.

1. Establish a native NPU-IR baseline and an equivalent PTODSL bridge result.
2. Inspect expanded IR for a specific redundant UB round trip or conversion.
3. Apply one bounded optimization, preserving numerical and sync contracts.
4. Compare traffic, instruction counts, spills, and A5 hardware runtime using
   identical inputs, shapes, launch configurations, and precision settings.

For current implementation and fallback policy, see
[PTODSL template development and fallback](../designs/ptodsl-template-development-and-fallback.md).
