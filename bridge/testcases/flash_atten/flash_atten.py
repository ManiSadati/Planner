import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def flash_atten_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    SQ: tl.constexpr,
    SK: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    H_Q: tl.constexpr,
    H_KV: tl.constexpr,
    SM_SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # A program owns one BLOCK_M-sized query tile for one query head.
    pid = tl.program_id(0)
    num_query_tiles = tl.cdiv(SQ, BLOCK_M)
    q_head = pid // num_query_tiles
    query_tile = pid % num_query_tiles

    # Grouped-query attention maps several Q heads to the same K/V head.
    kv_head = q_head // (H_Q // H_KV)

    offs_m = query_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    q_mask = offs_m[:, None] < SQ
    q_ptrs = (
        q_ptr
        + q_head * SQ * HEAD_DIM
        + offs_m[:, None] * HEAD_DIM
        + offs_d[None, :]
    )
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # Per-query-row online-softmax state. The output accumulator stays in
    # float32 while the kernel streams over the K/V sequence in BLOCK_N tiles.
    row_max = tl.full((BLOCK_M,), -1.0e9, tl.float32)
    row_sum = tl.zeros((BLOCK_M,), tl.float32)
    out_acc = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)

    for start_n in range(0, SK, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        # K is row-major [H_KV, SK, HEAD_DIM]. Reversing the two tile axes
        # loads a logical [HEAD_DIM, BLOCK_N] K-transpose tile for Q @ K^T.
        k_ptrs = (
            k_ptr
            + kv_head * SK * HEAD_DIM
            + offs_n[None, :] * HEAD_DIM
            + offs_d[:, None]
        )
        k = tl.load(
            k_ptrs,
            mask=offs_n[None, :] < SK,
            other=0.0,
        )
        scores = tl.dot(q, k).to(tl.float32) * SM_SCALE

        score_mask = (offs_m[:, None] < SQ) & (offs_n[None, :] < SK)
        if CAUSAL:
            score_mask = score_mask & (offs_n[None, :] <= offs_m[:, None])
        scores = tl.where(score_mask, scores, -1.0e9)

        # Merge this tile into the running softmax. When a new maximum is
        # found, alpha rescales all contributions accumulated so far.
        tile_max = tl.max(scores, axis=1)
        new_max = tl.maximum(row_max, tile_max)
        alpha = tl.exp(row_max - new_max)
        probabilities = tl.exp(scores - new_max[:, None])
        tile_sum = tl.sum(probabilities, axis=1)

        v_ptrs = (
            v_ptr
            + kv_head * SK * HEAD_DIM
            + offs_n[:, None] * HEAD_DIM
            + offs_d[None, :]
        )
        v = tl.load(
            v_ptrs,
            mask=offs_n[:, None] < SK,
            other=0.0,
        )
        out_acc = (
            out_acc * alpha[:, None]
            + tl.dot(probabilities.to(tl.float16), v)
        )
        row_sum = row_sum * alpha + tile_sum
        row_max = new_max

    output = out_acc / row_sum[:, None]
    out_ptrs = (
        out_ptr
        + q_head * SQ * HEAD_DIM
        + offs_m[:, None] * HEAD_DIM
        + offs_d[None, :]
    )
    tl.store(out_ptrs, output, mask=q_mask)


def main():
    torch.manual_seed(0)

    h_q = 4
    h_kv = 2
    seq_len = 256
    head_dim = 64
    block_m = 64
    block_n = 64
    sm_scale = head_dim**-0.5

    # Generate inputs on the host so an NPU simulator only executes the
    # kernel under test, not three random-number generation kernels.
    q_host = torch.randn((h_q, seq_len, head_dim), dtype=torch.float16)
    k_host = torch.randn((h_kv, seq_len, head_dim), dtype=torch.float16)
    v_host = torch.randn((h_kv, seq_len, head_dim), dtype=torch.float16)
    q = q_host.to("npu")
    k = k_host.to("npu")
    v = v_host.to("npu")
    out = torch.empty_like(q)

    num_programs = h_q * triton.cdiv(seq_len, block_m)
    flash_atten_kernel[(num_programs,)](
        q,
        k,
        v,
        out,
        SQ=seq_len,
        SK=seq_len,
        HEAD_DIM=head_dim,
        H_Q=h_q,
        H_KV=h_kv,
        SM_SCALE=sm_scale,
        CAUSAL=True,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
    )

    # PyTorch reference for the same GQA mapping and causal attention.
    kv_heads = torch.arange(h_q) // (h_q // h_kv)
    scores = (
        torch.matmul(q_host, k_host[kv_heads].transpose(-1, -2)) * sm_scale
    )
    causal_mask = torch.triu(
        torch.ones((seq_len, seq_len), dtype=torch.bool), diagonal=1
    )
    scores = scores.masked_fill(causal_mask, float("-inf"))
    reference = torch.matmul(torch.softmax(scores, dim=-1), v_host[kv_heads])

    out_host = out.cpu()
    difference = (out_host - reference).abs()
    print("max error:", difference.max().item())
    print(
        "allclose:",
        torch.allclose(out_host, reference, atol=2.0e-2, rtol=2.0e-2),
    )


if __name__ == "__main__":
    main()
