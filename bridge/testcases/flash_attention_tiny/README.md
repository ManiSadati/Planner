# Mask-free tiny flash attention

This testcase computes one complete non-causal 64-by-64 attention tile per
program. All dimensions exactly match the tile dimensions, so the Triton source
uses no boundary or causal masks and should require no predicate-memory
store/load operations.

`K` is transposed on the host and passed to the kernel as a contiguous
`[HEAD_DIM, SEQ_LEN]` tensor. The simulator fixture generates deterministic
FP16 inputs and compares the result with a NumPy FP32 softmax reference rounded
to FP16.

Run the complete bridge flow with:

```bash
Planner/bridge/tools/run_comparison_flow.sh early-ir flash_attention_tiny
Planner/bridge/tools/run_comparison_flow.sh emit-vmi flash_attention_tiny
Planner/bridge/tools/run_comparison_flow.sh emit-vpto flash_attention_tiny
Planner/bridge/tools/run_comparison_flow.sh bridge-sim flash_attention_tiny
```
