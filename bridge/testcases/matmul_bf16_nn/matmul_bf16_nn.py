import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def matmul_bf16_nn_kernel(a_ptr, b_ptr, c_ptr):
    offs_m = tl.arange(0, 64)
    offs_n = tl.arange(0, 64)
    offs_k = tl.arange(0, 64)

    a = tl.load(a_ptr + offs_m[:, None] * 64 + offs_k[None, :])
    b = tl.load(b_ptr + offs_k[:, None] * 64 + offs_n[None, :])
    c = tl.dot(a, b)
    tl.store(c_ptr + offs_m[:, None] * 64 + offs_n[None, :], c)


def main():
    a = torch.ones((64, 64), device="npu", dtype=torch.bfloat16)
    b = torch.ones((64, 64), device="npu", dtype=torch.bfloat16)
    c = torch.empty((64, 64), device="npu", dtype=torch.bfloat16)

    matmul_bf16_nn_kernel[(1,)](a, b, c)

    reference = torch.full((64, 64), 64.0, device="npu", dtype=torch.bfloat16)
    print("max error:", (c - reference).abs().max().item())
    print("allclose:", torch.allclose(c, reference, atol=0.0, rtol=0.0))


if __name__ == "__main__":
    main()
