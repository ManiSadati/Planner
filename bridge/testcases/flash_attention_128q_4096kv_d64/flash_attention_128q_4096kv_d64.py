import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def flash_attention_128q_4096kv_d64_kernel(
    q_ptr,
    k_t_ptr,
    v_ptr,
    out_ptr,
    scale: tl.constexpr,
    SQ: tl.constexpr,
    SK: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compute single-head, mask-free attention using 64-by-64 matmuls."""
    query_tile = tl.program_id(0)

    offs_m = query_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    q = tl.load(q_ptr + offs_m[:, None] * HEAD_DIM + offs_d[None, :])

    row_max = tl.full((BLOCK_M,), -1.0e9, tl.float32)
    row_sum = tl.zeros((BLOCK_M,), tl.float32)
    output = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    # SK=4096 gives 64 iterations. Each QK and PV operation is one
    # 64-by-64 matmul because the complete head dimension is 64.
    for start_n in range(0, SK, BLOCK_N):
        key_offsets = start_n + offs_n

        k_t = tl.load(
            k_t_ptr + offs_d[:, None] * SK + key_offsets[None, :]
        )
        scores = tl.dot(q, k_t).to(tl.float32)
        scores *= scale

        tile_max = tl.max(scores, axis=1)
        new_max = tl.maximum(row_max, tile_max)
        alpha = tl.exp(row_max - new_max)
        probabilities = tl.exp(scores - new_max[:, None])
        tile_sum = tl.sum(probabilities, axis=1)

        v = tl.load(
            v_ptr + key_offsets[:, None] * HEAD_DIM + offs_d[None, :]
        )
        output = (
            output * alpha[:, None]
            + tl.dot(probabilities.to(tl.float16), v)
        )
        row_sum = row_sum * alpha + tile_sum
        row_max = new_max

    output /= row_sum[:, None]
    tl.store(
        out_ptr + offs_m[:, None] * HEAD_DIM + offs_d[None, :],
        output,
    )


def main():
    torch.manual_seed(0)

    query_len = 128
    kv_len = 4096
    head_dim = 64
    block_size = 64
    scale = head_dim**-0.5

    q_host = torch.randn((query_len, head_dim), dtype=torch.float16)
    k_host = torch.randn((kv_len, head_dim), dtype=torch.float16)
    v_host = torch.randn((kv_len, head_dim), dtype=torch.float16)

    q = q_host.to("npu")
    k_t = k_host.t().contiguous().to("npu")
    v = v_host.to("npu")
    out = torch.empty_like(q)

    flash_attention_128q_4096kv_d64_kernel[(query_len // block_size,)](
        q,
        k_t,
        v,
        out,
        scale,
        SQ=query_len,
        SK=kv_len,
        HEAD_DIM=head_dim,
        BLOCK_M=block_size,
        BLOCK_N=block_size,
        BLOCK_D=block_size,
    )

    out_host = out.cpu()
    reference = torch.softmax(
        (q_host @ k_host.t()) * scale, dim=-1
    ) @ v_host
    difference = (out_host - reference).abs()
    print("max error:", difference.max().item())
    print(
        "allclose:",
        torch.allclose(out_host, reference, atol=5e-2, rtol=5e-2),
    )


if __name__ == "__main__":
    main()
