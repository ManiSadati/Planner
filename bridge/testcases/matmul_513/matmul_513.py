import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def matmul513_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_block in range(0, tl.cdiv(K, BLOCK_K)):
        k = k_block * BLOCK_K + offs_k
        a_ptrs = A_ptr + offs_m[:, None] * K + k[None, :]
        b_ptrs = B_ptr + k[:, None] * N + offs_n[None, :]

        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & (k[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        accumulator += tl.dot(a, b)

    c_ptrs = C_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(
        c_ptrs,
        accumulator,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def main():
    m = 513
    n = 513
    k = 513
    block_m = 64
    block_n = 64
    block_k = 64

    # Keep setup cheap in the operator simulator. The non-multiple shape and
    # masks, rather than random data, are the focus of this testcase.
    a = torch.ones((m, k), device="npu", dtype=torch.float16)
    b = torch.ones((k, n), device="npu", dtype=torch.float16)
    c = torch.empty((m, n), device="npu", dtype=torch.float16)

    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
    matmul513_kernel[grid](
        a,
        b,
        c,
        M=m,
        N=n,
        K=k,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )

    reference = torch.full((m, n), float(k), device="npu", dtype=torch.float16)
    difference = (c - reference).abs()
    print("max error:", difference.max().item())
    print("allclose:", torch.allclose(c, reference, atol=0.5, rtol=0.0))


if __name__ == "__main__":
    main()
