import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def row_softmax_kernel(x_ptr, out_ptr, n_cols: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n_cols
    base = row * n_cols

    x = tl.load(x_ptr + base + offsets, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)
    numerator = tl.exp(x)
    denominator = tl.sum(numerator, axis=0)
    y = numerator / denominator

    tl.store(out_ptr + base + offsets, y, mask=mask)


def main():
    torch.manual_seed(0)

    n_rows = 16
    n_cols = 256
    x = torch.randn((n_rows, n_cols), device="npu", dtype=torch.float32)
    out = torch.empty_like(x)

    block = triton.next_power_of_2(n_cols)
    row_softmax_kernel[(n_rows,)](x, out, n_cols, BLOCK=block)

    ref = torch.softmax(x, dim=-1)
    print("max error:", (out - ref).abs().max().item())
    print("allclose:", torch.allclose(out, ref, atol=1e-5, rtol=1e-5))


if __name__ == "__main__":
    main()
