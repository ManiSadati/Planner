# Matmul64 Template Selection IR

These snapshots record the native NPU-IR and PTODSL bridge dispatch boundaries
for the f16/f32 NN `matmul_64` testcase. They were captured from AscendNPU-IR
commit `821f863432bf5747d1d4f58f6ebebf9a8358a69d` on branch
`mani/selection-pass`.

The source testcase lives at:

```text
$ASCEND_NPU_IR_ROOT/bishengir/test/Integration/PTOASBridge/testcases/matmul_64/
```

[`matmul_64.py`](matmul_64.py) is a checked-in copy of the Triton kernel used
to produce these snapshots. The NPU-IR testcase directory remains the source
of truth for future regeneration.

## Snapshots

1. [`01-native-after-mark-disable-load.mlir`](01-native-after-mark-disable-load.mlir)
   is the last shared structured boundary. It contains
   `hivm.hir.mmadL1` with f16 L1 inputs, an f32 L0C output, and runtime
   `M=K=N=64`.
2. [`02-native-after-convert-hivm-to-std.mlir`](02-native-after-convert-hivm-to-std.mlir)
   shows the unchanged native path selecting and calling
   `@mma_tile_half_to_float`.
3. [`03-bridge-after-template-selection.mlir`](03-bridge-after-template-selection.mlir)
   shows `select-hivm-ptodsl-templates` retaining the structured operation and
   attaching the native identity `mma_tile_half_to_float` and PTODSL identity
   `__pto_mma_tile_half_to_float`.
4. [`04-bridge-after-template-conversion.mlir`](04-bridge-after-template-conversion.mlir)
   shows `convert-hivm-templates-to-pto` replacing the structured operation
   with a call to `@__pto_mma_tile_half_to_float` and importing the helper's PTO
   implementation into the module.

Together, snapshots 2 and 3 demonstrate that native NPU-IR and the bridge begin
from the same native template identity. Snapshot 4 demonstrates that the
bridge resolves that identity to the checked-in PTODSL-generated helper rather
than the native CCE bitcode implementation.

The files are review artifacts. Regenerate them from the source testcase after
changes to template selection, argument adaptation, or PTODSL materialization;
do not treat them as compiler inputs.
