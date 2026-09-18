# Long-context single-head Flash Attention with D=64

This testcase uses:

- query length `SQ = 128`;
- key/value length `SK = 4096`;
- head dimension `D = 64`;
- one FP16 attention head.

It isolates the long-context loop from the split-head accumulation used by the
`D=128` testcase. Two Triton programs each own 64 query rows, and each program
iterates over 64 key/value tiles. Every QK and PV `tl.dot` is exactly
`[64,64] x [64,64] -> [64,64]`; there is no second partial QK dot.

The testcase is mask-free and has no boundary tiles because every dimension is
an exact multiple of 64. It is intended for real-hardware bridge validation;
the simulator is expected to be slow.

Compile it with:

```bash
Planner/bridge/tools/run_comparison_flow.sh early-ir flash_attention_128q_4096kv_d64
Planner/bridge/tools/run_comparison_flow.sh emit-vmi flash_attention_128q_4096kv_d64
Planner/bridge/tools/run_comparison_flow.sh emit-vpto flash_attention_128q_4096kv_d64
```

On the hardware system, build and run the generated VPTO fixture with the same
workflow used for the smaller Flash Attention testcases.
