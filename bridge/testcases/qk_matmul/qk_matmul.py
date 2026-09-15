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
def qk_matmul_kernel(
    q_ptr,
    k_ptr,
    scores_ptr,
    SQ: tl.constexpr,
    SK: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    H_Q: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(SQ, BLOCK_M)
    num_pid_n = tl.cdiv(SK, BLOCK_N)
    tiles_per_head = num_pid_m * num_pid_n

    q_head = pid // tiles_per_head
    tile_id = pid % tiles_per_head
    pid_m = tile_id // num_pid_n
    pid_n = tile_id % num_pid_n
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_block in range(0, tl.cdiv(HEAD_DIM, BLOCK_K)):
        k_dim = k_block * BLOCK_K + offs_k
        q_ptrs = (
            q_ptr
            + q_head * SQ * HEAD_DIM
            + offs_m[:, None] * HEAD_DIM
            + k_dim[None, :]
        )
        # K is shared by all query heads and stored as [SK, HEAD_DIM].
        # Swapping the pointer dimensions loads a K-transpose tile.
        kt_ptrs = (
            k_ptr
            + offs_n[None, :] * HEAD_DIM
            + k_dim[:, None]
        )

        q = tl.load(
            q_ptrs,
            mask=(offs_m[:, None] < SQ) & (k_dim[None, :] < HEAD_DIM),
            other=0.0,
        )
        kt = tl.load(
            kt_ptrs,
            mask=(offs_n[None, :] < SK) & (k_dim[:, None] < HEAD_DIM),
            other=0.0,
        )
        accumulator += tl.dot(q, kt)

    score_ptrs = (
        scores_ptr
        + q_head * SQ * SK
        + offs_m[:, None] * SK
        + offs_n[None, :]
    )
    tl.store(
        score_ptrs,
        accumulator,
        mask=(offs_m[:, None] < SQ) & (offs_n[None, :] < SK),
    )


def main():
    enable_compile_timing()

    h_q = 8
    sq = 64
    sk = 64
    head_dim = 64
    block_m = 64
    block_n = 64
    block_k = 64

    q_host = torch.ones((h_q, sq, head_dim), dtype=torch.float16)
    k_host = torch.ones((sk, head_dim), dtype=torch.float16)
    q = q_host.to("npu")
    k = k_host.to("npu")
    scores = torch.empty((h_q, sq, sk), device="npu", dtype=torch.float16)

    num_programs = (
        h_q * triton.cdiv(sq, block_m) * triton.cdiv(sk, block_n)
    )
    qk_matmul_kernel[(num_programs,)](
        q,
        k,
        scores,
        SQ=sq,
        SK=sk,
        HEAD_DIM=head_dim,
        H_Q=h_q,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )

    scores_host = scores.cpu()
    reference = torch.matmul(
        q_host.float(), k_host.float().transpose(0, 1)
    ).to(torch.float16)
    difference = (scores_host - reference).abs()
    print("max error:", difference.max().item())
    print("allclose:", torch.allclose(scores_host, reference, atol=0.5, rtol=0.0))

    # CANN 9.1 beta can fault during TorchNPU teardown after simulator success.
    if os.getenv("TRITON_SIMULATOR_CLEAN_EXIT") == "1":
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
