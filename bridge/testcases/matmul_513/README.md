# Matmul 513

This testcase computes a row-major `513x513` f16 matrix multiplication with
f32 accumulation and an f16 output. It uses `64x64x64` Triton blocks.

The non-multiple shape creates:

- 9 row tiles by 9 column tiles, or 81 logical output programs;
- 9 K iterations per output tile;
- masked M, N, and K tails containing one valid element in the final tile.

Compared with `matmul_64`, the generated IR should also expose views with
stride 513 and dynamic tile offsets. The host fixture launches 32 physical
Cube cores and passes 81 as the logical work count, matching the NPU-IR
auto-blockify ABI used by the other multi-tile fixtures.

Generate the early IR and inspect the normal NPU-IR pass trace:

```bash
cd "$HOME/Planner"
bridge/tools/run_comparison_flow.sh early-ir matmul_513
bridge/tools/run_comparison_flow.sh print-all matmul_513
```

Exercise the external CCE-call compatibility route:

```bash
bridge/tools/run_comparison_flow.sh \
  --bridge-mode external-calls emit-vmi matmul_513
bridge/tools/run_comparison_flow.sh \
  --bridge-mode external-calls emit-vpto matmul_513
bridge/tools/run_comparison_flow.sh \
  --clean-build --bridge-mode external-calls bridge-sim matmul_513
```

`early-ir` creates `input.mlir`. Generated VMI, VPTO, build products, logs,
profiles, and the fat object are written under `out/` and are ignored by Git.
