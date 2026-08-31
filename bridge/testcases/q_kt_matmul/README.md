# Q x Contiguous KT Matmul

This testcase is the physical-layout control for `qk_matmul`. It computes the
same grouped-query-attention score matrix, but receives K-transpose as a
physically contiguous input instead of constructing a transposed view of
row-major K.

## Shapes

Batch size one is implicit:

```text
Q:      [32, 128, 256]   f16
KT:     [2, 256, 8192]   f16, physically contiguous
Scores: [32, 128, 8192]  f16
```

Query heads 0 through 15 use KT head 0. Query heads 16 through 31 use KT head
1. KT head 0 contains ones and KT head 1 contains twos, so the expected scores
are 256 and 512, respectively.

Each Triton program computes one `64x64` score tile. The K reduction has four
`64`-element steps:

```text
[64, 256] @ [256, 64] -> [64, 64]
```

## Why This Case Exists

`qk_matmul` stores K as `[2,8192,256]` and reads a logical transpose with tile
strides `[1,256]`. NPU-IR therefore creates an AIV path that loads K through
UB, performs a vector transpose/repack, and copies the result to L1 before
Cube MMA.

This testcase stores KT directly as `[2,256,8192]`. Its tiles have strides
`[8192,1]`, matching a normal matmul B operand. The expected NPU-IR lowering is
Cube-only: one direct ND2NZ load for Q, one for KT, MMA in L0, and fixpipe output
to GM. The pass trace should contain no generated AIV function and no
`hivm.hir.sync_block_set` or `hivm.hir.sync_block_wait` operation.

## Commands

```bash
cd "$HOME/Planner"
bridge/tools/run_comparison_flow.sh early-ir q_kt_matmul
bridge/tools/run_comparison_flow.sh \
  --bridge-mode external-calls bridge-print-all q_kt_matmul
bridge/tools/run_comparison_flow.sh \
  --bridge-mode external-calls emit-vmi q_kt_matmul
bridge/tools/run_comparison_flow.sh \
  --bridge-mode external-calls emit-vpto q_kt_matmul
bridge/tools/run_comparison_flow.sh \
  --clean-build --bridge-mode external-calls bridge-sim q_kt_matmul
```

Generated IR, logs, build products, profiles, and fat objects are written under
`out/` and ignored by Git.

## Current Result

The generated early IR uses `[256,1]` Q tiles and `[8192,1]` KT tiles. The
external-call bridge produces one Cube kernel with two `nd2nz_half` calls, no
AIV function, and no `sync_block_set` or `sync_block_wait` operation.

`emit-vmi` and `emit-vpto` both succeed. The bridge simulator also compiles the
VPTO, creates the A5 fat object, links the fixture, and launches all 32 physical
AIC cores. The full 8192-logical-tile cycle simulation did not produce a block
completion or output tensor within a 20-minute local test window, so numerical
comparison at this full shape remains unverified. This is a simulator-duration
limit, not a conversion, build, link, or launch error.

## A5 Hardware Run

The checked-in `q_kt_matmul_kernel_ptoas_fatobj.o` is the generated A5 device
object. On the A5 server, source its CANN environment and run:

```bash
cd "$HOME/Planner/bridge/testcases/q_kt_matmul"
./run_npu_from_fatobj.sh
```

The script compiles `launch.cpp`, links the fat object into the kernel shared
library, builds the ACL host executable, generates Q/KT input files, runs the
kernel, and compares `output.bin` against the expected 256/512 scores. Set
`ACL_DEVICE_ID` when the target device is not device zero.
