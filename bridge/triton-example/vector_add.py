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
    torch.manual_seed(0)

    tile_size = 256
    lhs = torch.randn((tile_size,), device="npu", dtype=torch.float32)
    rhs = torch.randn((tile_size,), device="npu", dtype=torch.float32)
    out = torch.empty_like(lhs)

    vector_add_kernel[(1,)](lhs, rhs, out, TILE_SIZE=tile_size)

    reference = lhs + rhs
    print("max error:", (out - reference).abs().max().item())
    print("allclose:", torch.allclose(out, reference))


if __name__ == "__main__":
    main()
