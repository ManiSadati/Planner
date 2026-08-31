# QK Matmul

This testcase isolates the `Q @ K^T` Cube computation from a current
Qwen3.5-style full-attention layer. It intentionally omits scaling, masking,
softmax, and `P @ V` so the bridge can focus on the matmul and GQA addressing.

## Model Shape

The testcase uses batch size one and these dimensions:

```text
H_Q      = 32       query heads
H_KV     = 2        key/value heads
SQ       = 128      query tokens
SK       = 8192     cached key tokens
HEAD_DIM = 256

Q:      [32, 128, 256]
K:      [2, 8192, 256]
Scores: [32, 128, 8192]
```

Qwen3.5-397B-A17B uses 32 full-attention query heads, 2 KV heads, and a
256-element head dimension. Its native context can be much longer than 8192;
8192 is used here because it is large enough to exercise long tiled QK work
without making this already simulator-heavy fixture unnecessarily larger.

Official model references:

- <https://huggingface.co/Qwen/Qwen3.5-397B-A17B>
- <https://huggingface.co/Qwen/Qwen3.5-397B-A17B/blob/main/config.json>

## GQA Mapping

This is grouped-query attention, so 16 query heads share each KV head:

```text
kv_head = query_head / (H_Q / H_KV) = query_head / 16

query heads  0..15 -> KV head 0
query heads 16..31 -> KV head 1
```

Every Q element is one. KV head 0 contains ones and KV head 1 contains twos, so
the expected score for query head `h` is:

```text
(kv_head + 1) * HEAD_DIM
```

Query heads 0 through 15 therefore produce 256, while heads 16 through 31
produce 512. This checks the GQA mapping as well as the dot product. The Python
testcase prepares these tensors on CPU and copies them to the NPU so simulator
time is spent on the Triton kernel rather than input-generation kernels.

## Triton Tiling

The kernel uses `64x64x64` blocks. One logical Triton program computes:

```text
[64, 256] @ [256, 64] -> [64, 64]
```

as four K-reduction iterations. The flattened launch contains:

```text
32 query heads * 2 query tiles * 128 key tiles = 8192 programs
```

Flattening the head and matrix tile coordinates into `tl.program_id(0)` keeps
the testcase aligned with the current NPU-IR auto-blockify path. A complete
FlashAttention kernel would normally keep one query tile resident and loop
over key tiles while applying online softmax; this testcase materializes the
score tensor only to make Cube conversion and numerical comparison explicit.

## Bridge Flow

Current status:

- `early-ir` succeeds and the checked-in `input.mlir` contains the expected
  GQA offsets, four K-reduction iterations, and `64x64` `linalg.matmul` tiles;
- baseline `print-all` captures the normal NPU-IR pass trace, then reaches the
  environment's existing `hivmc-a5 --save-temps` rejection;
- the external-call `emit-vmi` experiment currently stops in
  `convert-hivmave-to-ptoas-vmi` because MIX-side
  `hivm.hir.sync_block_set` is not legalized yet;
- the host, build, comparison, simulator, and A5 fat-object runner files are
  prepared for use after that conversion gap is closed.

Generate the early IR and inspect the unchanged NPU-IR pipeline:

```bash
cd "$HOME/Planner"
bridge/tools/run_comparison_flow.sh early-ir qk_matmul
bridge/tools/run_comparison_flow.sh print-all qk_matmul
```

Capture every BiShengIR pass with the PTOAS bridge enabled, including the
conversion attempt:

```bash
bridge/tools/run_comparison_flow.sh \
  --bridge-mode external-calls bridge-print-all qk_matmul
```

The baseline and bridge-enabled traces are written to `out/after-all.log` and
`out/bridge-after-all.log`, respectively.

Exercise the CCE-template compatibility route:

```bash
bridge/tools/run_comparison_flow.sh \
  --bridge-mode external-calls emit-vmi qk_matmul
bridge/tools/run_comparison_flow.sh \
  --bridge-mode external-calls emit-vpto qk_matmul
bridge/tools/run_comparison_flow.sh \
  --clean-build --bridge-mode external-calls bridge-sim qk_matmul
```

The full `npu-sim` and `bridge-sim` runs can be slow because the output holds
33,554,432 FP16 scores, or 64 MiB. Generated IR, logs, build products, profiles,
and fat objects are written under `out/` and ignored by Git.
