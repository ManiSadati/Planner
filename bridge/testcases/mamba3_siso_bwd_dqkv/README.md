# Standalone Mamba-3 backward dQ/dK/dV: predicate memory

`mamba3_siso_bwd_dqkv.py` copies the entire kernel from
`Q2TritonKernel/tests/mamba/test_mamba3_siso_bwd_kernel_dqkv/mamba3_siso_fwd_kernel_npu.py:15`.
Only the single-config autotuner and host wrapper are replaced. No descriptor,
mask, volatile load, reduction or state update in the kernel is changed.
`reference.py` copies `compute_dqkv_pytorch` unchanged from that testcase's
`dqkv_torch_impl.py:5`; this runner uses only its fixed-length FP32 branch.
Original correctness driver: `test_mamba3_siso_bwd_kernel_dqkv.py` in the same
source directory. Its external `.pt` fixtures are not required here.

The source repeatedly applies a strict causal predicate,
`tl.arange(0, CHUNK_SIZE)[None, :] > tl.arange(0, CHUNK_SIZE)[:, None]`,
under `RECOMPUTE_MASK=True`, including the `dAinv = tl.where(...)` at original
lines 256–260. The retained
`Logs/q2_vmi_gap_reaudit_oct21/captures/mamba_volatile/before-bridge.mlir`
contains a PB32 predicate payload store at line 1556 and reloads at lines
970 and 1201. Reconfirm this in fresh AVE IR; the source mask alone does not
establish that the compiler stores a predicate in UB.

## Run and capture

From this directory, with the Triton-Ascend Python environment and compatible
CANN environment configured:

```bash
python mamba3_siso_bwd_dqkv.py
```

The explicit specialization follows the capture: B=1, T=128, H=Hqk=1,
Dqk=Dv=64, CHUNK_SIZE=32, FP32, RECOMPUTE_MASK=True,
HAS_D_OSSM_STATE=False, RETURN_D_ISSM_STATE=False, IS_VARLEN=False,
num_warps=4, num_stages=2. `debug=True` enables diagnostics, with no additional
sanitizer option forced. No host retiling is performed. The original direct
wrapper does not set the solve kernels' sync/multibuffer overrides, so this
runner leaves them at backend defaults.

Inputs are synthetic and created on CPU. Per-chunk negative cumulative decay
and forward state recurrence provide nonzero inter-chunk contributions.
`DA_CS_SUM` uses padded storage to preserve the capture's default batch/head
strides (128,128); the first four entries hold its four chunk sums. Active
input strides match the capture. Inactive optional-state strides are zero;
`D`/`Cu_Seqlens` remain None. All five enabled outputs start as CPU-created NaNs
and are copied to the device, so missing writes fail the comparison.

`--batch`, `--heads` and `--length` are available; length must be a positive
multiple of 32. Other settings alter the captured specialization and require
fresh pattern validation. Optional skip/output-state/input-state and varlen
branches are deliberately disabled for this captured case.

Native `compile_flow="npuir"` is default and clears bridge enable/mode variables
in the process. `--compile-flow ptoas` selects a separate bridge check.

```bash
run_dir=$(mktemp -d "$PWD/native-capture.XXXXXX")
export TRITON_KERNEL_DUMP=1
export TRITON_CACHE_DIR="$run_dir/cache"
export TRITON_DUMP_DIR="$run_dir/dump"
python mamba3_siso_bwd_dqkv.py > "$run_dir/run.log" 2>&1
```

Without a physical NPU, use a configured CANN simulator. The previously used
local Ascend950 command form is:

```bash
msprof op simulator --soc-version=Ascend950PR_9589 --core-id=0 \
  --kernel-name=mamba3_siso_bwd_kernel_dqkv \
  --output="$run_dir/profile" python mamba3_siso_bwd_dqkv.py
```

Require the Python comparisons to finish successfully; a profiler exit alone
is insufficient. `debug=True` saves native compiler diagnostics as
`kernel.npuir.mlir` in the dump on the inspected backend. For an all-pass dump,
replay the printed native compiler invocation with its supported IR-print flags.
Removing the NPU availability check does not enable hardware-free allocation.
No `capture.py` harness is used.

## Reference and checks

All enabled outputs—dQ, dK, dV, dADT and dQK_Dot—are compared to the copied
CPU reference with atol=rtol=1e-5, following the original direct driver's FP32
tolerances. dD and dInput_SSM_State are disabled and have no allocated outputs.
This checks the original kernel's numerical contract on synthetic inputs,
not the external-fixture accuracy test or an autograd oracle.

Verified on creation: syntax; exact original kernel and reference bodies;
default active strides; finite CPU reference outputs; CPU Triton interpreter
agreement for every enabled output at the default specialization. Largest
absolute difference was below 2e-7. These checks do not exercise NPU-IR or
PTOAS. Native compilation, a fresh AVE spill capture, and NPU/simulator numerical
completion have not been run here.
