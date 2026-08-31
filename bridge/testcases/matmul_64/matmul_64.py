import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def matmul64_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
):
    # One program computes the entire output matrix.
    pid = tl.program_id(0)

    # Only launch one program.
    if pid != 0:
        return

    BLOCK = 64

    # Row and column indices.
    offs_m = tl.arange(0, 64)
    offs_n = tl.arange(0, 64)
    offs_k = tl.arange(0, 64)

    # A is row-major 64x64.
    a_ptrs = A_ptr + offs_m[:, None] * BLOCK + offs_k[None, :]

    # B is row-major 64x64.
    b_ptrs = B_ptr + offs_k[:, None] * BLOCK + offs_n[None, :]

    a = tl.load(a_ptrs)
    b = tl.load(b_ptrs)

    c = tl.dot(a, b)

    # C is row-major 64x64.
    c_ptrs = C_ptr + offs_m[:, None] * BLOCK + offs_n[None, :]
    tl.store(c_ptrs, c)


def main():
    torch.manual_seed(0)

    A = torch.randn((64, 64), device="npu", dtype=torch.float16)
    B = torch.randn((64, 64), device="npu", dtype=torch.float16)
    C = torch.empty((64, 64), device="npu", dtype=torch.float16)

    matmul64_kernel[(1,)](A, B, C)

    ref = torch.matmul(A, B)

    print("max error:", (C - ref).abs().max().item())
    print("allclose:", torch.allclose(C, ref, atol=1e-2, rtol=1e-2))


if __name__ == "__main__":
    main()
