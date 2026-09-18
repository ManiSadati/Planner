# Model-scale mask-free Flash Attention

This testcase uses a model-representative attention shape:

- query length `SQ = 128`;
- key/value length `SK = 4096`;
- head dimension `D = 128`;
- one FP16 attention head.

The 4096-token context and 128-element head match the Qwen3 Decode sample
configuration. The `S0=128, S1=4096` problem also appears in the A5 Flash
Attention performance configurations.

The kernel extends `flash_attention_128x128` without changing its microkernel
contracts:

- two Triton programs each own 64 query rows;
- each program iterates over 64 key/value tiles;
- QK accumulates two 64-element head chunks;
- PV computes two 64-column output chunks;
- every `tl.dot` is exactly `[64,64] x [64,64] -> [64,64]`;
- online softmax combines all 64 key/value tiles.

The testcase is mask-free and has no boundary tiles because every dimension is
an exact multiple of 64. It is intended for real-hardware bridge validation;
the simulator is expected to be slow.

Compile it with:

```bash
Planner/bridge/tools/run_comparison_flow.sh early-ir flash_attention_128q_4096kv_d128
Planner/bridge/tools/run_comparison_flow.sh emit-vmi flash_attention_128q_4096kv_d128
Planner/bridge/tools/run_comparison_flow.sh emit-vpto flash_attention_128q_4096kv_d128
```

On the hardware system, build and run the generated VPTO fixture with the same
workflow used for the smaller Flash Attention testcases.
