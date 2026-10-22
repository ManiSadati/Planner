# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Original CPU state builders and numerical oracle for chunk_bwd_dqkwg.

Copied from Q2TritonKernel/tests/fla/test_chunk_bwd_dqkwg.py.
Only imports/test harness are omitted; functions are unchanged except blank-line
whitespace. This standalone runner uses H=HV, USE_G=True and fixed full chunks.
"""

import torch

CHUNK_SIZE = 64


def naive_chunk_fwd_h(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    chunk_size: int = CHUNK_SIZE,
):
    """Naive forward for chunk-level hidden state h."""
    B, T, H, K, V = *k.shape, v.shape[-1]
    NT = (T + chunk_size - 1) // chunk_size
    orig_dtype = k.dtype

    k_f = k.to(torch.float32)
    v_f = v.to(torch.float32)
    g_f = g.to(torch.float32) if g is not None else None

    h = torch.zeros((B, NT, H, K, V), dtype=torch.float32, device=k.device)

    for b in range(B):
        for i_h in range(H):
            h_state = torch.zeros((K, V), dtype=torch.float32, device=k.device)

            for i_t in range(NT):
                s = i_t * chunk_size
                e = min(s + chunk_size, T)

                k_chunk = k_f[b, s:e, i_h]
                v_chunk = v_f[b, s:e, i_h]

                if g_f is not None:
                    g_last = g_f[b, e - 1, i_h]
                    h_state = h_state * torch.exp(g_last)
                    g_chunk = g_f[b, s:e, i_h]
                    v_chunk = v_chunk * torch.exp(g_last - g_chunk).unsqueeze(-1)

                h_state = h_state + torch.matmul(k_chunk.transpose(-1, -2), v_chunk)
                h[b, i_t, i_h] = h_state

    return h.to(orig_dtype)


def naive_chunk_bwd_dh(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    g: torch.Tensor | None = None,
    scale: float = None,
    chunk_size: int = CHUNK_SIZE,
):
    """Naive backward for chunk-level gradient dh."""
    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = do.shape[2]
    NT = (T + chunk_size - 1) // chunk_size
    orig_dtype = q.dtype

    if scale is None:
        scale = K ** -0.5

    q_f = q.to(torch.float32)
    do_f = do.to(torch.float32)
    g_f = g.to(torch.float32) if g is not None else None

    dh = torch.zeros((B, NT, HV, K, V), dtype=torch.float32, device=q.device)

    for b in range(B):
        for i_h in range(HV):
            dh_state = torch.zeros((K, V), dtype=torch.float32, device=q.device)

            for i_t in range(NT - 1, -1, -1):
                s = i_t * chunk_size
                e = min(s + chunk_size, T)

                q_chunk = q_f[b, s:e, i_h // (HV // H)]
                do_chunk = do_f[b, s:e, i_h]

                if g_f is not None:
                    g_last = g_f[b, e - 1, i_h]
                    dh_state = dh_state * torch.exp(g_last)
                    g_chunk = g_f[b, s:e, i_h]
                    do_chunk = do_chunk * torch.exp(g_chunk - g_last).unsqueeze(-1)

                dh_state = dh_state + torch.matmul(q_chunk.transpose(-1, -2), do_chunk)
                dh[b, i_t, i_h // (HV // H)] = dh_state

    return dh.to(orig_dtype)


def naive_chunk_bwd_dqkwg(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    dh: torch.Tensor,
    do: torch.Tensor,
    g: torch.Tensor | None = None,
    scale: float | None = None,
    chunk_size: int = CHUNK_SIZE,
    cast_dqdk_to_orig_dtype: bool = True,
    cast_dg_to_orig_dtype: bool = False,
):
    """Naive backward for dq, dk, dg at chunk level."""
    B, T, H, K = q.shape
    V = v.shape[-1]
    HV = do.shape[2]
    NT = (T + chunk_size - 1) // chunk_size
    orig_dtype = q.dtype

    if scale is None:
        scale = K ** -0.5

    q_f = q.to(torch.float32)
    k_f = k.to(torch.float32)
    v_f = v.to(torch.float32)
    h_f = h.to(torch.float32)
    dh_f = dh.to(torch.float32)
    do_f = do.to(torch.float32)
    g_f = g.to(torch.float32) if g is not None else None

    dq = torch.zeros((B, T, HV, K), dtype=torch.float32, device=q.device)
    dk = torch.zeros((B, T, HV, K), dtype=torch.float32, device=q.device)
    dg = torch.zeros((B, T, HV), dtype=torch.float32, device=q.device) if g is not None else None

    for b in range(B):
        for i_t in range(NT):
            s = i_t * chunk_size
            e = min(s + chunk_size, T)
            bt = e - s

            for i_h in range(HV):
                q_chunk = q_f[b, s:e, i_h // (HV // H)]
                k_chunk = k_f[b, s:e, i_h // (HV // H)]
                v_chunk = v_f[b, s:e, i_h]
                do_chunk = do_f[b, s:e, i_h]

                h_state = h_f[b, i_t, i_h // (HV // H)]
                dh_state = dh_f[b, i_t, i_h // (HV // H)]

                # Inter-chunk gradients
                dq_inter = torch.matmul(do_chunk, h_state.transpose(-1, -2))
                dk_inter = torch.matmul(v_chunk, dh_state.transpose(-1, -2))

                # Intra-chunk: ds = do @ v^T
                ds = torch.matmul(do_chunk, v_chunk.transpose(-1, -2))

                # Causal mask
                causal = torch.tril(torch.ones(bt, bt, device=q.device, dtype=torch.float32))

                if g_f is not None:
                    g_chunk = g_f[b, s:e, i_h]
                    g_last = g_f[b, e - 1, i_h]

                    # dg_last from inter-chunk
                    dg_last = torch.sum(h_state * dh_state) * torch.exp(g_last)

                    # Apply gating
                    dq_inter = dq_inter * torch.exp(g_chunk).unsqueeze(-1)
                    dk_inter = dk_inter * torch.exp(g_last - g_chunk).unsqueeze(-1)

                    # dg_last contribution from dk_inter
                    dg_last = dg_last + torch.sum(dk_inter * k_chunk)

                    # Intra-chunk with gating
                    g_diff = g_chunk.unsqueeze(-1) - g_chunk.unsqueeze(-2)
                    ds = ds * causal * torch.exp(g_diff) * scale

                    # Accumulate
                    dq_chunk = dq_inter * scale + torch.matmul(ds, k_chunk)
                    dk_chunk = dk_inter + torch.matmul(ds.transpose(-1, -2), q_chunk)

                    # dg
                    dg_chunk = torch.sum(dq_chunk * q_chunk, dim=-1) - torch.sum(dk_chunk * k_chunk, dim=-1)
                    dg_chunk[-1] += dg_last

                    dg[b, s:e, i_h] = dg_chunk
                else:
                    ds = ds * causal * scale

                    dq_chunk = dq_inter * scale + torch.matmul(ds, k_chunk)
                    dk_chunk = dk_inter + torch.matmul(ds.transpose(-1, -2), q_chunk) * scale

                dq[b, s:e, i_h] = dq_chunk
                dk[b, s:e, i_h] = dk_chunk

    # Aggregate for H != HV
    if H != HV:
        dq = dq.view(B, T, H, HV // H, K).sum(dim=3)
        dk = dk.view(B, T, H, HV // H, K).sum(dim=3)

    if cast_dqdk_to_orig_dtype:
        dq_out = dq.to(orig_dtype)
        dk_out = dk.to(orig_dtype)
    else:
        dq_out = dq
        dk_out = dk

    if dg is None:
        dg_out = None
    elif cast_dg_to_orig_dtype:
        dg_out = dg.to(orig_dtype)
    else:
        dg_out = dg

    return dq_out, dk_out, dg_out
