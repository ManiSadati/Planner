# Q2/Q3 A5 Hardware Test Runner

`bridge/tools/run_q2_q3_hardware_tests.py` runs the original Q2 and Q3 pytest
benchmark cases on one A5 NPU. It does not modify the benchmark repositories,
compiler, or testcase sources.

The runner first uses pytest collection to obtain the exact parametrized node
IDs. It then feeds those node IDs to a rolling worker pool. At most `--jobs`
processes are active; as soon as one finishes, the next pending case starts.
Each case is a separate pytest process with its own raw log, JUnit report,
temporary directory, Triton cache, PyTorch extension cache, and working
directory. A per-case process-group watchdog terminates hung descendants.

## Prerequisites

Activate the same Python and CANN environment used by the successful manual
hardware commands. For the current A5 installation that was:

```bash
source "$HOME/.venv/python3.11_venv/bin/activate"
source /data/pri/Ascend/9.1.0.B087/cann-9.1.0/set_env.sh

python3 -c 'import pytest, torch, torch_npu, triton, einops; print(torch.npu.device_count())'
```

The final command should print `1` on the current machine. If a package import
fails, repair that environment before starting a batch.

Set the project and source locations. The source roots can instead be supplied
with `--q2-root` and `--q3-root`.

```bash
export PLANNER_ROOT="$HOME/Workspace/Planner"
export Q2_ROOT="$HOME/Workspace/Q2TritonKernel-main"
export Q3_ROOT="$HOME/Workspace/Q3TritonKernel-main"
cd "$PLANNER_ROOT"
```

`Q2_TRITON_ROOT`/`Q3_TRITON_ROOT` are also accepted and take precedence over
`Q2_ROOT`/`Q3_ROOT`. If none are set, the runner checks the common checkout
names under `$HOME/Workspace` and `$HOME`.

The runner supplies absolute import paths both through pytest's `pythonpath`
configuration and the child `PYTHONPATH`. Q2 receives `<Q2>/src/kernels` and
the Q2 root; Q3 receives the Q3 root. This preserves the import fix used by the
working manual commands, including Q2's local `fla` namespace package.

## Safe First Run

Collection-only preview (test functions are not executed):

```bash
python3 bridge/tools/run_q2_q3_hardware_tests.py \
  --repo all \
  --q2-root "$Q2_ROOT" \
  --q3-root "$Q3_ROOT" \
  --pattern '*chunk_bwd*' \
  --limit 2 \
  --dry-run
```

Run one selected case before a full campaign:

```bash
python3 bridge/tools/run_q2_q3_hardware_tests.py \
  --repo q2 \
  --q2-root "$Q2_ROOT" \
  --pattern '*chunk_bwd*' \
  --limit 1 \
  --jobs 1
```

`--dry-run` still asks pytest to collect the test modules, so module-level
Python code is imported, but no collected testcase is executed.

## Full and Filtered Runs

Run all supported Q2 and Q3 suites with the conservative default of two active
cases and a 900-second timeout per case:

```bash
python3 bridge/tools/run_q2_q3_hardware_tests.py \
  --repo all \
  --q2-root "$Q2_ROOT" \
  --q3-root "$Q3_ROOT"
```

Run only Q3 cases matching either of two shell-style patterns:

```bash
python3 bridge/tools/run_q2_q3_hardware_tests.py \
  --repo q3 \
  --q3-root "$Q3_ROOT" \
  --pattern '*fbgemm_perf*' \
  --pattern '*jagged*' \
  --jobs 2 \
  --timeout 1200
```

Patterns have OR semantics and match the full name, for example
`q3::tests/fbgemm_perf/test_jagged.py::test_sum[B4-D128]`. Quote patterns so
the shell does not expand them. `--limit N` is applied after filtering.

Concurrency and timeout can also be configured with environment variables:

```bash
export HARDWARE_TEST_JOBS=2
export HARDWARE_TEST_TIMEOUT=900
python3 bridge/tools/run_q2_q3_hardware_tests.py --repo q2
```

Do not add pytest `-n`/xdist around this runner. The runner already owns the
device-level concurrency. Increase `--jobs` only after confirming that two
simultaneous cases are stable on this single NPU.

### Device selection limitation

The test sources target the logical NPU named `npu`/device `0`. Consequently,
the runner accepts only `--device 0` and sets `NPU_DEVICE=0` for child tests.
On a multi-device host, select one physical device before launching the runner
and it will appear to the tests as logical device zero:

```bash
export ASCEND_RT_VISIBLE_DEVICES=3
python3 bridge/tools/run_q2_q3_hardware_tests.py --device 0 --repo q2
```

On the current one-NPU machine no mask is necessary. One invocation does not
schedule work across multiple NPUs.

## Included Suites

Q2 collection covers `tests/fla_perf` and retains the working-command exclusion
of `test_causal_conv1d_bwd_gpu_perf.py`. Q3 collections are performed
separately for:

- `tests/fbgemm_perf`
- `tests/fla_perf`
- `tests/minimax_m3_perf`
- `tests/recsys_example_perf`
- `tests/recsys_perf`
- `tests/vllm_ascend_perf`

Separate Q3 collection avoids pytest module-name collisions between suite-local
files such as `test_common.py`. Every parametrized pytest node ID is one queued
testcase. The final parameter ID becomes the report's `shape` value. If a test
has no parameter ID, the runner reports `unknown` rather than guessing.

Some Q3 cases depend on generated `.pt` inputs or other external data. Generate
those inputs with the testcase's adjacent generator before the run. A missing
input is retained and reported as a testcase failure; the runner does not alter
the source tree or manufacture test data.

## Results

Unless `--output-dir` is specified, output is written below:

```text
bridge/out/hardware-tests/run_<timestamp>/
  results.csv
  results.json
  failures.log
  collection/
  cases/<ordinal>_<case>_<hash>/
    raw.log
    junit.xml
    failure.log        # failed/timed-out cases only
    work/
    tmp/
    triton-cache/
```

An explicitly selected output directory must be new or empty; the runner will
not overwrite an earlier campaign.

The terminal prints a compact table after the queue drains. `results.csv` has
exactly these columns:

```text
testcase_name,shape,status,reason
```

`results.json` contains the same fields plus the command, duration, exit code,
raw-log path, and reduced-log path. Status is one of `passed`, `failed`,
`timed out`, or `skipped`. The complete `raw.log` is retained for every case.

Each failure has a compact `failure.log`, and all such excerpts are combined in
the top-level `failures.log`. The original `Planner/failure.log` example was
empty, so the runner uses a conservative labeled format containing testcase,
shape, status, exact command, exit code, raw-log path, and selected traceback,
compiler, Triton, ACL/NPU, and runtime diagnostics with nearby context. If it
cannot isolate a trustworthy error, the reason explicitly points to that
reduced log instead of inventing a diagnosis.

Example terminal report:

```text
testcase_name                                      | shape       | status    | reason
---------------------------------------------------+-------------+-----------+-----------------------------
q2::tests/fla_perf/test_chunk.py::test_fwd[B1-T64] | B1-T64      | passed    |
q3::tests/fbgemm_perf/test_jagged.py::test_sum[B4] | B4          | failed    | RuntimeError: ACL_ERROR_...
q3::tests/fla_perf/test_kda.py::test_kda[B2-H8]    | B2-H8       | timed out | Exceeded timeout of 900 seconds
```

The process exits `0` when every executed case passed or skipped, `1` when a
case or collection failed/timed out, and `2` for invalid arguments or an empty
selection. A collection failure for one suite is recorded as a synthetic
`<repo>::collection::<suite>` result and does not prevent other suites from
being collected or run.
