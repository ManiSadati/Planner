# NPU-IR Tools

These scripts are NPU-IR-only helpers. They live here instead of `bridge/tools`
because they do not invoke PTOAS.

| Script | Purpose |
|---|---|
| `setup_npuir_simulator_env.sh` | Create/update the Python simulator venv and apply the local Triton Ascend compatibility patch. |
| `source_npuir_simulator_env.sh` | Sourceable base environment for NPU-IR simulator runs. |
| `source_vector_add_large_simulator_env.sh` | Sourceable complete environment/output setup for the large vector-add simulator run. |
| `run_vector_add_simulator.sh` | One-shot baseline NPU-IR simulator run for `vector_add.py`. |
| `run_vector_add_large_simulator.sh` | One-shot baseline NPU-IR simulator run for `vector_add_large.py`. |
| `replay_npuir_from_device_spec.sh` | Replay checked-in `*_kernel.mlir` dumps from `AppendTargetDeviceSpec` through local `bishengir-compile` pass dumps. |

Cross-repo comparison stays in:

```text
bridge/tools/run_npuir_ptoas_bridge_tests.sh
```
