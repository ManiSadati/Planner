# Flash Attention 64x64

This testcase is the smallest full causal Flash Attention composition used by
the NPU-IR-to-PTOAS bridge bring-up:

- one query head and one key/value head;
- `SQ = SK = HEAD_DIM = 64`;
- `BLOCK_M = BLOCK_N = 64`;
- one logical and physical launch block;
- one QK matmul, one softmax tile, one PV matmul, and final normalization.

Unlike `flash_atten_small`, this shape has no second key tile, no loop-carried
online-softmax update, no grouped-query head mapping, and no multi-block
interaction. The checked-in `input.mlir` was generated from `flash_atten.py`
with the comparison-flow `early-ir` command.

Run the bridge flow with:

```bash
bridge/tools/run_comparison_flow.sh emit-vmi flash_atten_64x64
bridge/tools/run_comparison_flow.sh emit-vpto flash_atten_64x64
bridge/tools/run_comparison_flow.sh bridge-sim flash_atten_64x64
```

At creation time, the generated early IR reaches the native NPU-IR lowering,
and the NumPy fixture produces finite `[1, 64, 64]` inputs and reference
output. The PTODSL bridge currently stops in
`convert-hivm-templates-to-pto` with a rewrite-driver convergence error. This
is therefore the first bridge bring-up boundary for the single-tile kernel,
before the divide-by-zero observed in the larger testcase.
