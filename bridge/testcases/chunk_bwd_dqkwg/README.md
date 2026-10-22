# Gated chunk backward: multiple continuous stores

This is the standalone original `chunk_bwd_kernel_dqkwg` from
`Q2TritonKernel/src/kernels/fla/ops/common/chunk_o.py:160`.
Its entire body, signature and `do_not_specialize=['T']` setting are preserved.
The normal `exp` and `exp2` helpers from `fla/ops/utils/op.py:29/31` are copied
unchanged, selecting the `FLA_USE_FAST_OPS=0` branch explicitly. FLA imports,
heuristics, autotuning, dispatch and the host wrapper are replaced by explicit
arguments. No compiler source or original kernel source is modified.

Original benchmark: `Q2TritonKernel/tests/fla_perf/test_chunk_bwd_dqkwg_perf.py`.
Original correctness test and reference functions:
`Q2TritonKernel/tests/fla/test_chunk_bwd_dqkwg.py`.

## Pattern to inspect

The original source at `chunk_o.py:308` computes two row reductions:

```python
b_dg = tl.sum(b_dq * b_q, axis=1) - tl.sum(b_dk * b_k, axis=1)
```

The retained capture
`Logs/q2_ave_requirements_oct19/captures/reused_dqkwg_H2_B4/before-bridge.mlir`
contains two intermediate UB result streams in the same row loop, at lines
1376 and 1383. Both use `ave.hir.masked_store <ONEPT_B32>` with
`hivm.is_continuous`. These are compiler-generated intermediate stores, not
two extra `tl.store` calls inserted into this testcase.

For this issue, first inspect fresh native AVE IR for both marked stores.
Then inspect native lowering for stream initialization, `vstus` updates,
loop-carried alignment/pointer state, and `vstas` completion after the loop.
Compare the same graph through VMI/PTOAS separately. The request concerns
continuous intent and eligible multiple-stream fusion; a measured performance
penalty has not been established. A smaller source specialization does not
prove the retained pattern reappears until its IR is inspected.

## Explicit specialization

| Setting | Value |
|---|---|
| Default dimensions | B=1, T=64, H=HV=2, K=V=128 |
| Tiles and grid | BT=64, BK=BV=128; grid=(1,T/64,B*H) |
| Input/output types | BF16 q/k/v/g/h/do/dh and dq/dk; FP32 dg |
| Flags | USE_G=True; USE_G_GAMMA/USE_EXP2/USE_DW/TRANSPOSE_STATE/IS_VARLEN=False |
| Launch | num_warps=2, num_stages=2, debug=True, sanitize_overflow=True |
| Backend options | sync_solver=True, multibuffer=True, unlimited local-buffer multibuffering, enable_auto_bind_sub_block=True, vf_merge_level=1 |
| Math/environment | normal helpers, TRITON_F32_DEFAULT=ieee, TRITON_ALL_BLOCKS_PARALLEL=1 |

The retained capture used B=4,T=131072,H=2 with the same tiles, enabled gate,
disabled optional branches and two warps/stages. The default reduces B and T
for simulation. The selected launch settings are not a measured autotuner
winner. The target comes from the active runtime/compiler; the retained capture
target was `Ascend950PR_9579`.

## Run and capture

From Workspace, after activating Triton-Ascend and configuring CANN:

```bash
source Planner/bridge/testcases/setup_triton_env.sh chunk_bwd_dqkwg
"$PYTHON_BIN" "$KERNEL_SCRIPT" > "$RUN_LOG" 2>&1
```

The environment helper automatically discovers this testcase. It puts cache,
dumps, profiles, logs and temporary files under this testcase's `out` directory.
It also prints the simulator command; using its exported variables:

```bash
msprof op simulator --soc-version="$SOC_VERSION" --core-id="$CORE_ID" \
  --kernel-name="$KERNEL_NAME" --output="$MSPROF_OUTPUT" \
  "$PYTHON_BIN" "$KERNEL_SCRIPT" > "$RUN_LOG" 2>&1
```

`--batch`, `--length` and `--heads` select dimensions; length must be a positive
multiple of 64. `--batch 4 --length 131072` restores the retained dimensions.
The default `compile_flow="npuir"` clears the bridge enable/mode variables in
the process. Use `--compile-flow ptoas` for a separate bridge check.

CPU inputs use the benchmark's random BF16 tensors, log-sigmoid gate and naive
h/dh construction. CPU-created NaNs initialize all outputs before transfer,
avoiding auxiliary NPU initialization kernels and exposing missing writes.
The kernel produces dq, dk and one dg partial per K tile; K=BK means NK=1.
All entries of dq/dk/dg are compared against the original copied reference.
The normal FLA criterion is relative RMS below 0.02, or maximum absolute error
at most 1e-6; warning/CI exceptions are omitted and non-finite values fail.
This is an aggregate error criterion, not a 2% bound on every element.

Require the Python `PASS: all dq/dk/dg entries compared` message before claiming
numerical success. Debug diagnostics are retained by the inspected backend as
`kernel.npuir.mlir`; an all-pass dump may require replaying the printed native
compiler command with the installation's supported IR-print flags. A dump with
multiple pass snapshots is not one parseable MLIR module.

## Validation at creation

- Syntax and original kernel signature/body/JIT setting checked; both normal
  math helpers and all three reference functions match the originals.
- Environment setup detects `chunk_bwd_kernel_dqkwg` and selects this directory's
  `out` paths in both Bash and Zsh.
- FP32-only CPU interpreter diagnostics passed for B1/T64/H2 and B2/T128/H1,
  checking every dq/dk/dg entry. Largest relative RMS was below 3e-7. The
  comparison also rejected injected NaN and zeroed dq outputs.
- The default BF16 interpreter attempt failed. An independent 16x16 BF16
  identity-dot probe returned 264257536 instead of 1 at [0,0], confirming a
  local interpreter limitation independent of this kernel. FP32 diagnostics
  do not validate the BF16 specialization.
- Native compilation, a fresh continuous-store capture, and BF16 NPU/simulator
  numerical execution have not been performed for this standalone case.
