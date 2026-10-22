# Standalone triangular solve: predicate memory

This copies the entire `merge_16x16_to_64x64_inverse_kernel` from
`Q2TritonKernel/src/kernels/fla/ops/utils/solve_tril.py:135`, including its original
`do_not_specialize=['T']` setting. The original FLA host imports, wrapper and
`IS_VARLEN` heuristic are replaced with explicit arguments. No kernel operation
is changed. Original benchmark: `Q2TritonKernel/tests/fla_perf/test_solve_tril_perf.py`.

The relevant source predicate is `m_A = o_i[:, None] > o_i[None, :]`, used by
`tl.where` to retain the strict lower triangle. The existing
`Logs/q2_ave_requirements_oct19/captures/solve/before-bridge.mlir` contains
its PB32 predicate payload store at line 1515 and reload at line 1194.
The corresponding `solve_varlen` capture also contains predicate stores/reloads.
The diagonal predicate `m_I` is separate; a masked numeric store alone is not
predicate-memory evidence.

## Run and capture

From this directory, with the Triton-Ascend Python environment and compatible
CANN environment already configured:

```bash
python solve_tril_64.py
python solve_tril_64.py --length 64 --heads 2
python solve_tril_64.py --heads 16 --sequence-lengths 67 64
```

Defaults: FP32, B=1, T=1024, H=32, BT=64, USE_TMA=False,
IS_VARLEN=False, DOT_PRECISION=ieee, num_warps=2, num_stages=2,
sync_solver=True, multibuffer=True, unlimited local-buffer multibuffering.
These match the fixed-length capture's explicit configuration; two warps was
a harness choice, not a measured winning configuration. Debug diagnostics are
enabled. No additional sanitizer option is forced.

`--sequence-lengths` selects packed varlen mode, requires B=1, and sets T to
the sum of the supplied lengths. It builds cumulative lengths and chunk indices
on CPU. The example is a small tail check, not the original varlen workload.
The smaller fixed-length example also changes the captured specialization.
Reconfirm the predicate-memory pattern in fresh AVE IR for each configuration.

Native `compile_flow="npuir"` is the default and clears the bridge enable/mode
variables in the process. Use `--compile-flow ptoas` for a separate bridge run.

To retain compiler output, use fresh cache/dump directories:

```bash
run_dir=$(mktemp -d "$PWD/native-capture.XXXXXX")
export TRITON_KERNEL_DUMP=1
export TRITON_CACHE_DIR="$run_dir/cache"
export TRITON_DUMP_DIR="$run_dir/dump"
python solve_tril_64.py > "$run_dir/run.log" 2>&1
```

Without a physical NPU, run the same script through a configured CANN simulator,
for example on the previously used local Ascend950 setup:

```bash
msprof op simulator --soc-version=Ascend950PR_9589 --core-id=0 \
  --kernel-name=merge_16x16_to_64x64_inverse_kernel \
  --output="$run_dir/profile" python solve_tril_64.py --length 64 --heads 2
```

The simulator/runtime/compiler environment must already be configured; removing
the availability check does not enable execution. Require the Python `PASS`
comparison, not just a successful profiler exit. `debug=True` saves native
compiler diagnostics as `kernel.npuir.mlir` in the dump on the inspected backend.
For an all-pass dump, replay the printed native compiler invocation with its
supported IR-print flags. No `capture.py` harness is needed.

## Reference and checks

CPU inputs follow the benchmark's normalized-key Gram construction, with the
strict lower triangle kept. The CPU reference independently solves
`(I + A) X = I` with `torch.linalg.solve_triangular` in float64, then converts
to FP32. All output entries are compared with explicit atol=rtol=1e-4; this
is distinct from FLA's aggregate `assert_close` metric. Output zeros are created
on CPU and copied to the device, matching the original wrapper's treatment of
unwritten upper off-diagonal tiles.

Verified on creation: syntax; exact original kernel body; CPU reference inverse
residual; CPU Triton interpreter/reference agreement for B1/T64/H2,
B2/T128/H2, packed lengths [67,64]/H2 and [1,2,15,20]/H1.
Largest interpreter/reference absolute difference was below 1e-7.
These checks do not exercise NPU-IR or PTOAS. Native compilation, a fresh AVE
spill capture, and NPU/simulator numerical completion have not been run here.
