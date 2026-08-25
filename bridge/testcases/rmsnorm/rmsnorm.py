import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def rmsnorm_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n_cols
    base = row * n_cols

    x = tl.load(x_ptr + base + offsets, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    square_sum = tl.sum(tl.where(mask, x * x, 0.0), axis=0)
    inv_rms = 1.0 / tl.sqrt(square_sum / n_cols + eps)
    y = x * inv_rms * weight

    tl.store(out_ptr + base + offsets, y, mask=mask)


def main():
    torch.manual_seed(0)

    n_rows = 1600
    n_cols = 1600
    eps = 1.0e-5
    x = torch.randn((n_rows, n_cols), device="npu", dtype=torch.float32)
    weight = torch.randn((n_cols,), device="npu", dtype=torch.float32)
    out = torch.empty_like(x)

    block = triton.next_power_of_2(n_cols)
    rmsnorm_kernel[(n_rows,)](x, weight, out, n_cols, eps, BLOCK=block)

    ref = x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps) * weight
    print("max error:", (out - ref).abs().max().item())
    print("allclose:", torch.allclose(out, ref, atol=1e-5, rtol=1e-5))


if __name__ == "__main__":
    main()
