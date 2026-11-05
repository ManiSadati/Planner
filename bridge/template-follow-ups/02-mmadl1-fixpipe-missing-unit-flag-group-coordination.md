# MmadL1 and Fixpipe: missing PTODSL unit-flag group coordination

## Issue

**Status: documented for later implementation. The PTODSL group-state behavior has not been implemented or validated.**

Native MmadL1 and Fixpipe now share a small piece of state through `unit_flag_group_id`. MmadL1 can decide that a particular operation needs explicit synchronization, and Fixpipe can read that decision before copying its result.

Our PTODSL replacements still receive only the unit-flag mode. They do not receive the group ID or reproduce this shared-state behavior. Adding an argument alone will not close the gap: the implementations must also read and update the state and emit the corresponding synchronization.

This note compares the older `celine/new-templates` branch (`471d351c5`) with the native code and PTODSL replacements on `celine/merge-new-templates` (HEAD `26b97628e` when inspected, 2026-11-05). Source links point to the local checkout; old snippets were read with `git show celine/new-templates:<path>`.

## What the two arguments mean

| Argument | Simple meaning |
|---|---|
| `unit_flag_mode` | How this operation should use the hardware unit flag. |
| `unit_flag_group_id` | Which software state entry connects the related MmadL1 and Fixpipe operations. |

Mode `0` is disabled, `2` is enabled without updating the flag, and `3` is enabled with an update. The instruction actions differ between MmadL1 and Fixpipe: for example, mode `3` sets the flag on the matrix instruction and checks/clears it on the corresponding Fixpipe transfer.

The group ID is not another hardware unit-flag mode. For example, `group_id=7` means “use shared state entry 7.” Related operations must receive the same assigned ID; unrelated groups must not accidentally share their state. The native implementation currently has room for 100 group IDs.

## What changed in the native arguments

In [LocalMmad.cpp](/home/c50066216/AscendNPU-IR/bishengir/lib/Template/lib/RegBase/Cube/LocalMmad.cpp), the end of `mma_tile_core`'s parameter list used to be:

```cpp
              int64_t back_pipe_m_pipe_mte1_db_event0,
              int64_t back_pipe_m_pipe_mte1_db_event1,
              uint8_t unit_flag) {
```

It is now:

```cpp
              int64_t kloop_db_cond, int64_t back_pipe_m_pipe_mte1_db_event0,
              int64_t back_pipe_m_pipe_mte1_db_event1, UNIT_FLAG unit_flag_mode,
              int64_t unit_flag_group_id) {
```

The older implementation passed the supplied flag down to `L1Mmad`. The newer implementation first checks shared state and may change the effective mode.

Fixpipe received the extra argument too. In [compat/Fixpipe/Fixpipe.cpp](/home/c50066216/AscendNPU-IR/bishengir/lib/Template/lib/RegBase/Cube/compat/Fixpipe/Fixpipe.cpp), the end of `copy_matrix_cc_to_gm_normal_2d_to_2d_core` used to be:

```cpp
                                          int64_t pre_relu, bool channel_split,
                                          uint8_t unit_flag) {
```

It is now:

```cpp
    int64_t pre_quant, float32_t quant_scale, int64_t pre_relu,
    bool channel_split, UNIT_FLAG unit_flag_mode, int64_t unit_flag_group_id) {
```

This GM function is one concrete signature example. The current native UB Fixpipe paths also call the shared resolver below.

## What the new native implementation does

The state lives in [SyncUtils.cpp](/home/c50066216/AscendNPU-IR/bishengir/lib/Template/lib/Synchronization/SyncUtils/SyncUtils.cpp):

```cpp
__aicore__ uint8_t &getUnitFlagDisableStatus(int64_t unit_flag_group_id) {
  static uint8_t unit_flag_disable_status[MAX_BLOCK_NUM]
                                         [MAX_UNIT_FLAG_GROUP_ID] = {0};
  return unit_flag_disable_status[get_block_idx() % MAX_BLOCK_NUM]
                                 [unit_flag_group_id];
}
```

This returns a shared byte for the current block and group. A true value means this group needs the explicit synchronization path. It does not store a replacement mode number.

In `mma_tile_core`, the native code handles a requested mode `3` as follows:

```cpp
    } else if (unit_flag_mode == UNIT_FLAG::ENABLED_WITH_UPDATE) {
      unit_flag_was_disable = false;
      if ((actualN != l0c_n_size) || (actualM != l0c_m_size)) {
        unit_flag_was_disable = true;
        unit_flag_mode = UNIT_FLAG::ENABLED_WITHOUT_UPDATE;
      }
    }
```

Here `actualM` and `actualN` describe the matrix result being computed. `l0c_m_size` and `l0c_n_size` describe the allocated accumulator tile. If those extents differ, MmadL1 changes its effective mode from `3` to `2` and marks the group as disabled for the coordination path.

Fixpipe then calls this exact resolver:

```cpp
__aicore__ __attribute__((always_inline)) UNIT_FLAG
resolveUnitFlagMode(UNIT_FLAG unit_flag_mode, int64_t unit_flag_group_id) {
  if (unit_flag_mode != UNIT_FLAG::DISABLED) {
    if (getUnitFlagDisableStatus(unit_flag_group_id)) {
      INTRINSIC(set_flag, PIPE_M, PIPE_FIX, LIB_EVENT_ID0);
      INTRINSIC(wait_flag, PIPE_M, PIPE_FIX, LIB_EVENT_ID0);
      return UNIT_FLAG::DISABLED;
    }
  }
  return unit_flag_mode;
}
```

Fixpipe does **not** simply copy MmadL1's mode `2`. It emits an explicit M→FIX synchronization pair and uses mode `0` for its transfer when the shared marker is set.

The native MmadL1 implementation also checks state left by a previous operation. When the marker is set, it emits a FIX→M synchronization pair before continuing. Its mode-`2` path can clear the marker and switch to mode `0`. A future implementation must preserve these transitions across successive calls, not just handle one producer/consumer pair.

### Simple example

The following is an explanatory example, not a captured testcase:

```text
Allocated accumulator: 64 × 64
Actual result:         48 × 64
Requested mode:        3
Shared group ID:       7

MmadL1:
  Actual result does not cover the accumulator tile.
  Set group[7].disabled = true.
  Execute with effective mode 2.

Fixpipe with the same group ID:
  Read group[7].disabled == true.
  Emit explicit M→FIX set_flag/wait_flag.
  Execute the transfer with effective mode 0.
```

For a matching full tile, with a clear marker, this particular partial-tile fallback is not triggered. The native batched path has an additional fallback condition for a batch larger than one; bias and batched variants must be reviewed when extending coverage.

## What our PTODSL path currently has

In [mmadl1_ptodsl.py](/home/c50066216/AscendNPU-IR/bishengir/lib/Template/lib/RegBase/Cube/mmadl1_ptodsl.py), the MmadL1 helper's parameter list still ends with:

```python
    c_stride_3: pto.i64,
    unit_flag: pto.i8,
):
```

The matching bridge in [MmadL1PTODSLAdapter.cpp](/home/c50066216/AscendNPU-IR/bishengir/lib/Conversion/HIVMTemplatesToPTO/Cube/MmadL1PTODSLAdapter.cpp) appends only the mode:

```cpp
  if (supportsUnitFlag)
    arguments.push_back(op.getUnitFlagModeLibValue(rewriter));
```

The PTODSL body treats nonzero incoming modes as enabled and chooses an updating matrix instruction for the final internal K slice. It does not reproduce the native group-state resolver or the full native distinction between incoming modes `2` and `3`. That behavior must be reviewed as part of the implementation.

In [fixpipe_ptodsl.py](/home/c50066216/AscendNPU-IR/bishengir/lib/Template/lib/RegBase/Cube/fixpipe_ptodsl.py), the single-UB helper currently receives:

```python
    sub_block_id: pto.i64,
    unit_flag: pto.i8,
    pre_quant: pto.i64,
    quant_scale: pto.f32,
    pre_relu: pto.i64,
```

Its transfer selection checks `unit_flag` directly:

```python
        with pto.if_(unit_flag == pto.const(3, dtype=pto.i8)) as update:
            with update.then_:
                transfer(pto.AccStoreUnitFlagCtrl.CHECK_AND_CLEAR, relu)
            with update.else_:
                with pto.if_(unit_flag == pto.const(2, dtype=pto.i8)) as check:
                    with check.then_:
                        transfer(pto.AccStoreUnitFlagCtrl.CHECK_ONLY, relu)
                    with check.else_:
                        transfer(None, relu)
```

There is no group ID or shared-marker lookup in this helper. [MatmulRegionToPTO.cpp](/home/c50066216/AscendNPU-IR/bishengir/lib/Conversion/HIVMTemplatesToPTO/Cube/MatmulRegionToPTO.cpp) likewise forwards the Fixpipe mode but not the group ID.

## Work to implement later

1. Forward the compiler-assigned group ID through both bridge adapters and add it to the corresponding PTODSL helper signatures.
2. Implement the native state transitions: partial-tile fallback, mode handling, explicit M→FIX and FIX→M synchronization, and marker clearing.
3. Ensure related PTODSL helpers share the same state. If one operation uses PTODSL and its partner uses native fallback, they must coordinate through compatible shared state too; separate private tables would not communicate.
4. Regenerate the affected MLIR resources and update argument-count checks and tests. The current MmadL1 generator self-test expects 39 arguments, ending in the mode.
5. Verify full tiles, partial tiles, repeated calls that enter and leave fallback, separate group IDs, and supported mixtures of native and PTODSL calls. Check generated synchronization as well as numerical results. Review batched and bias variants explicitly before claiming coverage.

## What has and has not been demonstrated

The missing group-state implementation is visible in the code comparison. We have **not demonstrated a runtime failure caused specifically by this gap**.

The enabled-sequence cases that emitted mode `0` involve the compiler's unit-flag eligibility decisions. The FP8 NN stride bug and INT8 Nd2Nz argument-adapter failures are separate issues. Their failures should not be presented as evidence that missing group-ID handling caused an error.
