import os
import sys
from pathlib import Path

import torch
import torch_npu
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from compile_timing import enable_compile_timing


@triton.jit
def matmul1600_kernel(
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
    enable_compile_timing()

    torch.manual_seed(0)

    m = 1600
    n = 1600
    k = 1600
    block_m = 64
    block_n = 64
    block_k = 64

    A = torch.randn((m, k), device="npu", dtype=torch.float16)
    B = torch.randn((k, n), device="npu", dtype=torch.float16)
    C = torch.empty((m, n), device="npu", dtype=torch.float16)

    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
    matmul1600_kernel[grid](
        A,
        B,
        C,
        M=m,
        N=n,
        K=k,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )

    ref = torch.matmul(A, B)

    print("max error:", (C - ref).abs().max().item())
    print("allclose:", torch.allclose(C, ref, atol=1e-2, rtol=1e-2))

    # CANN 9.1 beta can fault during TorchNPU teardown after simulator success.
    if os.getenv("TRITON_SIMULATOR_CLEAN_EXIT") == "1":
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
