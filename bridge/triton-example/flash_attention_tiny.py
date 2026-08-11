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
    row = tl.program_id(0)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    q = tl.load(q_ptr + row * BLOCK_D + offs_d)[None, :]
    k_t = tl.load(k_t_ptr + offs_d[:, None] * BLOCK_N + offs_n[None, :])
    scores = tl.dot(q, k_t).to(tl.float32) * scale

    scores = scores - tl.max(scores, axis=1)[:, None]
    probs = tl.exp(scores)
    probs = probs / tl.sum(probs, axis=1)[:, None]

    v = tl.load(v_ptr + offs_n[:, None] * BLOCK_D + offs_d[None, :])
    out = tl.dot(probs.to(tl.float16), v)

    tl.store(out_ptr + row * BLOCK_D + offs_d[None, :], out)


def main():
    torch.manual_seed(0)

    seq_len = 64
    head_dim = 64
    scale = 1.0 / (head_dim ** 0.5)

    q = torch.randn((seq_len, head_dim), device="npu", dtype=torch.float16)
    k = torch.randn((seq_len, head_dim), device="npu", dtype=torch.float16)
    v = torch.randn((seq_len, head_dim), device="npu", dtype=torch.float16)
    k_t = k.t().contiguous()
    out = torch.empty_like(q)

    flash_attention_tiny_kernel[(seq_len,)](
        q,
        k_t,
        v,
        out,
        scale,
        BLOCK_N=seq_len,
        BLOCK_D=head_dim,
    )

    ref = torch.softmax((q @ k.t()) * scale, dim=-1) @ v
    print("max error:", (out - ref).abs().max().item())
    print("allclose:", torch.allclose(out, ref, atol=1e-2, rtol=1e-2))


if __name__ == "__main__":
    main()
