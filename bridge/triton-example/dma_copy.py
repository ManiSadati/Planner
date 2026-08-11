import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def dma_copy_kernel(src_ptr, dst_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    values = tl.load(src_ptr + offsets, mask=mask, other=0.0)
    tl.store(dst_ptr + offsets, values, mask=mask)


def main():
    torch.manual_seed(0)

    n_elements = 4096 + 17
    src = torch.randn((n_elements,), device="npu", dtype=torch.float32)
    dst = torch.empty_like(src)

    block = 256
    grid = (triton.cdiv(n_elements, block),)
    dma_copy_kernel[grid](src, dst, n_elements, BLOCK=block)

    print("max error:", (dst - src).abs().max().item())
    print("allclose:", torch.allclose(dst, src))


if __name__ == "__main__":
    main()
