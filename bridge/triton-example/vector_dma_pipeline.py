import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def vector_dma_pipeline_kernel(
    x_ptr,
    residual_ptr,
    bias_ptr,
    out_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)

    y = (x * 1.25) + residual + bias
    y = tl.where(y > 6.0, 6.0, y)
    y = tl.where(y < -6.0, -6.0, y)

    tl.store(out_ptr + offsets, y, mask=mask)


def main():
    torch.manual_seed(0)

    n_elements = 4096 + 13
    x = torch.randn((n_elements,), device="npu", dtype=torch.float32)
    residual = torch.randn((n_elements,), device="npu", dtype=torch.float32)
    bias = torch.randn((n_elements,), device="npu", dtype=torch.float32)
    out = torch.empty_like(x)

    block = 256
    grid = (triton.cdiv(n_elements, block),)
    vector_dma_pipeline_kernel[grid](x, residual, bias, out, n_elements, BLOCK=block)

    ref = torch.clamp((x * 1.25) + residual + bias, min=-6.0, max=6.0)
    print("max error:", (out - ref).abs().max().item())
    print("allclose:", torch.allclose(out, ref))


if __name__ == "__main__":
    main()
