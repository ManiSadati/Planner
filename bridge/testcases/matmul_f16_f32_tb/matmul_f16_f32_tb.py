import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def matmul_f16_f32_tb_kernel(a_ptr, b_ptr, c_ptr):
    offs_m = tl.arange(0, 64)
    offs_n = tl.arange(0, 64)
    offs_k = tl.arange(0, 64)

    a = tl.load(a_ptr + offs_m[:, None] * 64 + offs_k[None, :])
    # Load B in its physical [N, K] order, then make the source transpose
    # explicit so NPU-IR can represent the distinct b_transpose MMAD contract.
    b_physical = tl.load(b_ptr + offs_n[:, None] * 64 + offs_k[None, :])
    b = tl.trans(b_physical)
    c = tl.dot(a, b)
    tl.store(c_ptr + offs_m[:, None] * 64 + offs_n[None, :], c)


def main():
    a = torch.eye(64, device="npu", dtype=torch.float16)
    b = torch.arange(1, 65, device="npu", dtype=torch.float16)[:, None]
    b = b.repeat(1, 64)
    c = torch.empty((64, 64), device="npu", dtype=torch.float16)

    matmul_f16_f32_tb_kernel[(1,)](a, b, c)

    reference = torch.matmul(a, b.transpose(0, 1))
    print("max error:", (c - reference).abs().max().item())
    print("allclose:", torch.allclose(c, reference, atol=1e-2, rtol=1e-2))


if __name__ == "__main__":
    main()
