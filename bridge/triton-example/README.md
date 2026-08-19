# Triton Bridge Examples

These Python files are source fixtures for the NPU-IR-to-PTOAS bridge. Real A5
validation and authoritative early IR generation still happen on an A5
machine. A CANN operator simulator is available on the Codex-accessible server
for limited functional checks; see
`NPUIR/coding-guide/simulator-workflow.md`.

The goal is to cover the operation families we need to map, starting from simple
cases and moving toward realistic fused kernels.

Some A5-generated MLIR dumps are currently checked in beside the source fixtures
as `*_kernel.mlir`. These are dumps right after
`AppendTargetDeviceSpec (hacc-append-device-spec)`, not original Python/Triton
source.

## Example Set

| File | Main purpose | Expected bridge signals |
|---|---|---|
| `cube_dotproduct.py` | 64x64 matmul seed already added by the human | `tl.dot`, cube/matmul path, L1/L0 staging, fixpipe/store |
| `dma_copy.py` | Pure copy with masked tail | GM->UB load, UB->GM store, simple DMA shape/stride |
| `vector_elementwise.py` | Pure vector arithmetic with masks | vector load, predicate/mask, add/mul/relu, masked store |
| `vector_dma_pipeline.py` | Multiple vector loads plus store | load/store DMA around vector arithmetic, scalar broadcast |
| `vector_add.py` | Minimal two-input vector addition | two GM loads, vector add, GM store, simulator smoke test |
| `vector_add_large.py` | 1000x2000 row-wise vector addition | multi-row launch, masked 2048-column vector tiles, large GM/UB movement |
| `row_softmax.py` | Row-wise softmax | vector reductions, exp, divide, masks, load/store |
| `rmsnorm.py` | Row-wise RMSNorm | reduction, sqrt/divide, weight load, vector scaling |
| `flash_attention_tiny.py` | Single-block attention row | dot-like score computation, softmax, value accumulation |

## A5 Dump Workflow

For each example that matters, generate at least:

- early IR before `convert-hivm-to-std`;
- IR after `convert-hivm-to-std`, when useful;
- command line / environment note used to generate the dump;
- compiler version or repo commit, if easy to capture.

Put generated dumps under:

```text
bridge/examples/npuir-early-ir/<example-name>/
```

Do not use synthetic IR as proof for bridge correctness when a real A5-generated
dump is needed.

For local replay from the current `*_kernel.mlir` files after NPU-IR is built,
see `NPUIR/coding-guide/device-spec-replay.md` and
`NPUIR/tools/replay_npuir_from_device_spec.sh`.

## Priority Order

1. `dma_copy.py`
2. `vector_elementwise.py`
3. `cube_dotproduct.py`
4. `row_softmax.py`
5. `rmsnorm.py`
6. `flash_attention_tiny.py`

`dma_copy.py` is first because it should expose the simplest GM->UB and UB->GM
rows for the DMA mapping plan.
