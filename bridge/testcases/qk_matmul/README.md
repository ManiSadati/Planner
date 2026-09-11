# QK Matmul

This testcase isolates the `Q @ K^T` Cube computation. It intentionally omits
scaling, masking, softmax, and the value tensor so the bridge can focus on the
matmul. The dimensions are
small enough for routine cycle-simulator smoke testing.

## Model Shape

The testcase uses batch size one and these dimensions:

```text
H_Q      = 8        query heads
SQ       = 64       query tokens
SK       = 64       cached key tokens
HEAD_DIM = 64

Q:      [8, 64, 64]
K:      [64, 64]
Scores: [8, 64, 64]
```

This is a reduced compiler/runtime fixture rather than a complete model-native
attention shape. It retains multiple query heads and one complete `64x64x64`
Cube tile per query head. All query heads share the same two-dimensional K
matrix.

## Reference Result

Each Q head is an identity matrix. K uses a deterministic, non-symmetric
pattern with exactly representable FP16 values. This makes the fixture
sensitive to whether the K transpose actually occurred. The reference is
computed with an actual FP32 batched matrix multiplication and converted to
FP16:

```text
Scores[head] = Q[head] @ transpose(K)
```

The identity Q makes each score matrix:

```text
Scores[head] = transpose(K)
```

The Python testcase prepares these tensors on CPU and copies them to the NPU
so simulator time is spent on the Triton kernel rather than input-generation
kernels.

## Triton Tiling

The kernel uses `64x64x64` blocks. One logical Triton program computes:

```text
[64, 64] @ [64, 64] -> [64, 64]
```

as one K-reduction iteration. The flattened launch contains:

```text
8 query heads * 1 query tile * 1 key tile = 8 programs
```

Flattening the head and matrix tile coordinates into `tl.program_id(0)` keeps
the testcase aligned with the current NPU-IR auto-blockify path. A complete
FlashAttention kernel would normally keep one query tile resident and loop
over key tiles while applying online softmax; this testcase materializes the
score tensor only to make Cube conversion and numerical comparison explicit.

## Bridge Flow

After changing the source dimensions, regenerate `input.mlir` before running
the bridge because it is compiler-generated and may still describe an older
shape:

```bash
cd "$HOME/Planner"
bridge/tools/run_comparison_flow.sh early-ir qk_matmul
```

Then run the default PTODSL bridge:

```bash
bridge/tools/run_comparison_flow.sh emit-vmi qk_matmul
bridge/tools/run_comparison_flow.sh emit-vpto qk_matmul
bridge/tools/run_comparison_flow.sh --clean-build bridge-sim qk_matmul
```

The current bridge simulation completes in 20,011 ticks. Its output columns
0-15 match the non-symmetric reference exactly, while columns 16-63 are zero.
This confirms the fixed implicit-transpose load specialization for the
currently produced slice and leaves a separate downstream Cube layout/output
coverage issue to resolve.

The score output now contains 32,768 FP16 values, or 64 KiB. Generated IR,
logs, build products, profiles, and fat objects are written under `out/` and
ignored by Git. The separate compiler regression test retains the `[1,256]`
implicit-transpose source-stride case; this reduced runtime fixture naturally
uses `[1,64]` because `HEAD_DIM=64`.
