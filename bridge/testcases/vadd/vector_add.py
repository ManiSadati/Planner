import os
import sys

import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def vector_add_kernel(
    lhs_ptr,
    rhs_ptr,
    out_ptr,
    TILE_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, TILE_SIZE)

    lhs = tl.load(lhs_ptr + offsets)
    rhs = tl.load(rhs_ptr + offsets)
    result = lhs + rhs
    tl.store(out_ptr + offsets, result)


def main():
    tile_size = 256
    lhs_cpu = torch.linspace(-1.0, 1.0, tile_size, dtype=torch.float32)
    rhs_cpu = torch.linspace(2.0, -2.0, tile_size, dtype=torch.float32)
    lhs = lhs_cpu.npu()
    rhs = rhs_cpu.npu()
    out = torch.empty_like(lhs)

    vector_add_kernel[(1,)](lhs, rhs, out, TILE_SIZE=tile_size)

    out_cpu = out.cpu()
    reference = lhs_cpu + rhs_cpu
    print("max error:", (out_cpu - reference).abs().max().item())
    print("allclose:", torch.allclose(out_cpu, reference))

    # CANN 9.1 beta can fault during TorchNPU teardown after simulator success.
    if os.getenv("TRITON_SIMULATOR_CLEAN_EXIT") == "1":
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
