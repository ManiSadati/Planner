import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def vector_elementwise_kernel(a_ptr, b_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    mixed = (a + b) * 0.5
    relu = tl.where(mixed > 0.0, mixed, 0.0)

    tl.store(out_ptr + offsets, relu, mask=mask)


def main():
    torch.manual_seed(0)

    n_elements = 2048 + 31
    a = torch.randn((n_elements,), device="npu", dtype=torch.float32)
    b = torch.randn((n_elements,), device="npu", dtype=torch.float32)
    out = torch.empty_like(a)

    block = 256
    grid = (triton.cdiv(n_elements, block),)
    vector_elementwise_kernel[grid](a, b, out, n_elements, BLOCK=block)

    ref = torch.relu((a + b) * 0.5)
    print("max error:", (out - ref).abs().max().item())
    print("allclose:", torch.allclose(out, ref))


if __name__ == "__main__":
    main()
