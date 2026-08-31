# Raw Object Probe Run Results

Date: 2026-08-26

## Build

The probe source builds successfully with CANN 9.1 beta headers and libraries:

```sh
export ASCEND_HOME_PATH=/home/a84369921/Ascend/cann-9.1.0-beta.3

g++ -std=c++17 -Wall -Wextra \
  -I $ASCEND_HOME_PATH/x86_64-linux/include \
  -I $ASCEND_HOME_PATH/x86_64-linux/pkg_inc \
  /home/m00967009/Workspace/Planner/bridge/testcases/vadd_large/raw_object_probe.cpp \
  -L $ASCEND_HOME_PATH/x86_64-linux/lib64 \
  -Wl,--allow-shlib-undefined -lascendcl -lruntime \
  -o /home/m00967009/Workspace/Planner/bridge/testcases/vadd_large/raw_object_probe
```

## Launch Attempts

The fixture inputs were generated with:

```sh
python3 /home/m00967009/Workspace/Planner/bridge/testcases/vadd_large/gen_data.py
```

| Attempt | Object | Symbol | Mode | Result |
| --- | --- | --- | --- | --- |
| Native raw object | `vector_add_large_npuir_baseline_aicore.o` | `vector_add_large_kernel` | `--launch` | blocked at `aclInit failed, errorCode=500000` |
| PTOAS raw object | `vector_add_large_ptoas_bridge_aicore.o` | `vector_add_large_kernel_mix_aiv` | `--launch` | blocked at `aclInit failed, errorCode=500000` |
| Existing fat-object fixture control | `vector_add_large_kernel_ptoas_fatobj.o` | host wrapper | existing `run_npu_from_fatobj.sh` | blocked at `[ERROR] aclInit failed: 500000` |

Because the existing fat-object flow fails at the same initialization boundary, the current machine/session cannot validate registration or launch behavior. The source and build checks are complete, but numeric runtime equivalence remains pending until ACL initialization succeeds on a configured device or simulator environment.
