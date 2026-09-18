import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def flash_attention_tiny_kernel(
    q_ptr,
    k_t_ptr,
    v_ptr,
    out_ptr,
    scale: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compute one exact-tile, non-causal 64-by-64 attention problem."""
    offs_m = tl.arange(0, BLOCK_N)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    # BLOCK_N and BLOCK_D exactly match the input dimensions, so none of the
    # loads or stores requires a boundary predicate.
    q = tl.load(q_ptr + offs_m[:, None] * BLOCK_D + offs_d[None, :])
    k_t = tl.load(k_t_ptr + offs_d[:, None] * BLOCK_N + offs_n[None, :])
    scores = tl.dot(q, k_t).to(tl.float32) * scale

    scores = scores - tl.max(scores, axis=1)[:, None]
    probabilities = tl.exp(scores)
    probabilities = probabilities / tl.sum(probabilities, axis=1)[:, None]

    v = tl.load(v_ptr + offs_n[:, None] * BLOCK_D + offs_d[None, :])
    output = tl.dot(probabilities.to(tl.float16), v)

    tl.store(out_ptr + offs_m[:, None] * BLOCK_D + offs_d[None, :], output)


def main():
    torch.manual_seed(0)

    seq_len = 64
    head_dim = 64
    scale = head_dim**-0.5

    q = torch.randn((seq_len, head_dim), device="npu", dtype=torch.float16)
    k = torch.randn((seq_len, head_dim), device="npu", dtype=torch.float16)
    v = torch.randn((seq_len, head_dim), device="npu", dtype=torch.float16)
    k_t = k.t().contiguous()
    out = torch.empty_like(q)

    flash_attention_tiny_kernel[(1,)](
        q,
        k_t,
        v,
        out,
        scale,
        BLOCK_N=seq_len,
        BLOCK_D=head_dim,
    )

    reference = torch.softmax((q @ k.t()) * scale, dim=-1) @ v
    difference = (out - reference).abs()
    print("max error:", difference.max().item())
    print("allclose:", torch.allclose(out, reference, atol=5e-2, rtol=5e-2))


if __name__ == "__main__":
    main()
