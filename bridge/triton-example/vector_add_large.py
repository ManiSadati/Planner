import os
import sys

import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def vector_add_large_kernel(
    lhs_ptr,
    rhs_ptr,
    out_ptr,
    NUM_COLS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    row = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_COLS)
    offsets = row * NUM_COLS + col_offsets
    mask = col_offsets < NUM_COLS

    lhs = tl.load(lhs_ptr + offsets, mask=mask)
    rhs = tl.load(rhs_ptr + offsets, mask=mask)
    result = lhs + rhs
    tl.store(out_ptr + offsets, result, mask=mask)


def main():
    rows = 1000
    cols = 2000
    block_cols = triton.next_power_of_2(cols)
    numel = rows * cols

    lhs_cpu = torch.arange(numel, dtype=torch.float32).reshape(rows, cols) * 0.001
    rhs_cpu = torch.arange(numel, dtype=torch.float32).reshape(rows, cols) * -0.002
    lhs = lhs_cpu.npu()
    rhs = rhs_cpu.npu()
    out = torch.empty_like(lhs)

    vector_add_large_kernel[(rows,)](
        lhs,
        rhs,
        out,
        NUM_COLS=cols,
        BLOCK_COLS=block_cols,
    )

    out_cpu = out.cpu()
    reference = lhs_cpu + rhs_cpu
    print("shape:", tuple(out_cpu.shape))
    print("numel:", numel)
    print("block_cols:", block_cols)
    print("max error:", (out_cpu - reference).abs().max().item())
    print("allclose:", torch.allclose(out_cpu, reference))

    # CANN 9.1 beta can fault during TorchNPU teardown after simulator success.
    if os.getenv("TRITON_SIMULATOR_CLEAN_EXIT") == "1":
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
