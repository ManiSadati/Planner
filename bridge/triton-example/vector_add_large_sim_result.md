# vector_add_large simulator result

Date: 2026-08-18

Kernel: `vector_add_large_kernel`

Shape: `1000 x 2000` `float32`

Launch shape: one Triton program per row, with `BLOCK_COLS = 2048` and a mask for the valid `2000` columns.

Command shape:

```bash
cd "$HOME/Planner"
source bridge/tools/source_vector_add_large_simulator_env.sh

msprof op simulator \
  --kernel-name=vector_add_large_kernel \
  --soc-version=Ascend950PR_9589 \
  --core-id=0 \
  --output="$OUT/profile" \
  python3 "$HOME/Planner/bridge/triton-example/vector_add_large.py"
```

Equivalent wrapper:

```bash
cd "$HOME/Planner"
bridge/tools/run_vector_add_large_simulator.sh
```

Functional result:

```text
shape: (1000, 2000)
numel: 2000000
block_cols: 2048
max error: 0.0
allclose: True
```

Simulator timing summary:

```text
core0.veccore0: duration_time 27.64 us, running_time 23.35 us
core0.veccore1: duration_time 28.33 us, running_time 23.98 us
```

Generated artifacts:

- Initial TTIR: `$HOME/tmp/npuir-simulator/vector-add-large-20260818T185550Z/dump/I5q3AMeYluis82BbXstG193DDLbRsXo1cMdGsprtlco/kernel.ttir.mlir`
- Initial TTAdapter MLIR: `$HOME/tmp/npuir-simulator/vector-add-large-20260818T185550Z/dump/I5q3AMeYluis82BbXstG193DDLbRsXo1cMdGsprtlco/kernel.ttadapter.mlir`
- `AppendTargetDeviceSpec` fixture copied to: `bridge/triton-example/vector_add_large.mlir`
- Full simulator profile: `$HOME/tmp/npuir-simulator/vector-add-large-20260818T185550Z/profile/OPPROF_20260818185550_XDHMMTUZMWZTLSDO`

Note: replaying the dumped TTAdapter through `bishengir-compile` printed the
requested `AppendTargetDeviceSpec` IR and then crashed later in the pipeline.
The checked-in `vector_add_large.mlir` is only claiming the printed
`AppendTargetDeviceSpec` level.
