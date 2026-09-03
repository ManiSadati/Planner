import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def matmul_i8_i32_nn_kernel(a_ptr, b_ptr, c_ptr):
    offs_m = tl.arange(0, 64)
    offs_n = tl.arange(0, 64)
    offs_k = tl.arange(0, 64)

    a = tl.load(a_ptr + offs_m[:, None] * 64 + offs_k[None, :])
    b = tl.load(b_ptr + offs_k[:, None] * 64 + offs_n[None, :])
    c = tl.dot(a, b, out_dtype=tl.int32)
    tl.store(c_ptr + offs_m[:, None] * 64 + offs_n[None, :], c)


def main():
    a = torch.ones((64, 64), device="npu", dtype=torch.int8)
    b = torch.ones((64, 64), device="npu", dtype=torch.int8)
    c = torch.empty((64, 64), device="npu", dtype=torch.int32)

    matmul_i8_i32_nn_kernel[(1,)](a, b, c)

    reference = torch.full((64, 64), 64, device="npu", dtype=torch.int32)
    print("exact:", torch.equal(c, reference))


if __name__ == "__main__":
    main()
