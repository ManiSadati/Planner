"""Original CPU reference, used here only for fixed-length FP32 inputs.

Copied unchanged from Q2TritonKernel/tests/mamba/
test_mamba3_siso_bwd_kernel_dqkv/dqkv_torch_impl.py:5.
"""

import torch
from typing import Optional, Tuple


def compute_dqkv_pytorch(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        da_cs: torch.Tensor,
        da_cs_sum: torch.Tensor,
        qk_dot: torch.Tensor,
        SSM_States: torch.Tensor,
        do: torch.Tensor,
        d_ossm_state: Optional[torch.Tensor] = None,
        d_ov_state: Optional[torch.Tensor] = None,
        D: Optional[torch.Tensor] = None,
        chunk_size: int = 64,
        has_input_state: bool = False,
        Cu_Seqlens: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[
    torch.Tensor]]:
    """
    PyTorch reference implementation for Mamba-3 backward pass.

    Args:
        q: Query tensor (batch, seqlen, nheads_qk, headdim_qk)
        k: Key tensor (batch, seqlen, nheads_qk, headdim_qk)
        v: Value tensor (batch, seqlen, nheads, headdim_v)
        da_cs: Cumulative decay per chunk (batch, nheads, seqlen)
        da_cs_sum: Sum of decay per chunk (batch, nheads, nchunks)
        qk_dot: QK dot products from forward (batch, nheads, seqlen)
        SSM_States: SSM states from forward pass (batch, nheads, headdim_v, nchunks * headdim_qk)
        do: Output gradient (batch, seqlen, nheads, headdim_v)
        d_ossm_state: Gradient of output SSM states (num_sequences, nheads, headdim_v, headdim_qk)
        d_ov_state: Gradient of output V state (num_sequences, nheads, headdim_v)
        D: Optional skip connection weight (nheads,)
        chunk_size: Chunk size (default: 64)
        has_input_state: Whether to compute gradient for input states
        Cu_Seqlens: Cumulative sequence lengths for varlen mode

    Returns:
        Tuple of (dQ_mid, dK_mid, dV, dADT, dQK_dot, dD, d_issm_state)
    """
    batch, seqlen, nheads_qk, headdim_qk = q.shape
    _, _, nheads, headdim_v = v.shape
    is_varlen = Cu_Seqlens is not None

    if is_varlen:
        num_sequences = Cu_Seqlens.shape[0] - 1
        assert batch == 1
        nchunks = num_sequences + seqlen // chunk_size
    else:
        num_sequences = batch
        nchunks = (seqlen + chunk_size - 1) // chunk_size

    assert nheads % nheads_qk == 0, "nheads must be divisible by nheads_qk"

    device = q.device
    dtype = q.dtype

    dq = torch.zeros((batch, seqlen, nheads, headdim_qk), dtype=dtype, device=device)
    dk = torch.zeros((batch, seqlen, nheads, headdim_qk), dtype=dtype, device=device)
    dv = torch.zeros((batch, seqlen, nheads, headdim_v), dtype=dtype, device=device)
    dAdt = torch.zeros((batch, nheads, seqlen), dtype=dtype, device=device)
    dQK = torch.zeros((batch, nheads, seqlen), dtype=dtype, device=device)
    dD = torch.zeros((num_sequences, nheads), dtype=torch.float32, device=device) if D is not None else None
    d_issm_state = torch.zeros((num_sequences, nheads, headdim_v, headdim_qk), dtype=torch.float32,
                               device=device) if has_input_state else None

    for b in range(batch):
        for h in range(nheads):
            head_idx_qk = h // (nheads // nheads_qk)

            q_h = q[b, :, head_idx_qk, :].float()
            k_h = k[b, :, head_idx_qk, :].float()
            v_h = v[b, :, h, :].float()
            da_cs_h = da_cs[b, h, :].float()
            da_cs_sum_h = da_cs_sum[b, h, :].float()
            qk_dot_h = qk_dot[b, h, :].float()
            do_h = do[b, :, h, :].float()
            ssm_states_h = SSM_States[b, h, :, :].float()

            if d_ossm_state is not None:
                d_ssm_states_acc = d_ossm_state[b, h, :, :].float()
            else:
                d_ssm_states_acc = torch.zeros(headdim_v, headdim_qk, dtype=torch.float32, device=device)

            if D is not None:
                D_val = D[h].float()
                dD_acc = torch.zeros(1, dtype=torch.float32, device=device)

            dM_rev_vector_full = torch.zeros(seqlen, dtype=torch.float32, device=device)

            for chunk_idx_loop in range(nchunks):
                chunk_idx = nchunks - 1 - chunk_idx_loop
                chunk_start = chunk_idx * chunk_size
                chunk_end = min(chunk_start + chunk_size, seqlen)
                actual_chunk_size = chunk_end - chunk_start

                offs_cs = torch.arange(chunk_start, chunk_end, device=device)

                da_cs_chunk = da_cs_h[chunk_start:chunk_end]
                da_cs_chunk_sum = da_cs_sum_h[chunk_idx]

                do_block = do_h[chunk_start:chunk_end, :]
                v_block = v_h[chunk_start:chunk_end, :]
                q_block = q_h[chunk_start:chunk_end, :]
                k_block = k_h[chunk_start:chunk_end, :]
                ssm_states_block = ssm_states_h[:, chunk_idx * headdim_qk:(chunk_idx + 1) * headdim_qk]

                da_cs_rev = da_cs_chunk_sum - da_cs_chunk
                exp_da_cs_rev = torch.exp2(da_cs_rev)
                exp_da_cs = torch.exp2(da_cs_chunk)

                row_idx = torch.arange(actual_chunk_size, device=device)
                col_idx = torch.arange(actual_chunk_size, device=device)
                causal_mask = col_idx[None, :] > row_idx[:, None]
                decay_diff = da_cs_chunk[None, :] - da_cs_chunk[:, None]
                causal_decay_mask = torch.where(causal_mask,
                                                torch.exp2(torch.minimum(decay_diff, torch.zeros_like(decay_diff))),
                                                torch.zeros(actual_chunk_size, actual_chunk_size, device=device))

                dAinv = torch.matmul(v_block, do_block.t())
                dAinv = dAinv * causal_decay_mask
                dAinv = dAinv * torch.matmul(k_block, q_block.t())
                dM_rev_vector = dAinv.sum(dim=0) - dAinv.sum(dim=1)

                dp_t_block = torch.matmul(v_block, do_block.t())
                dp_t_block = dp_t_block * causal_decay_mask
                acc_dk = torch.matmul(dp_t_block, q_block)
                acc_dk = acc_dk + torch.matmul(v_block, d_ssm_states_acc) * exp_da_cs_rev[:, None]
                dk[b, chunk_start:chunk_end, h, :] = acc_dk.to(dtype)

                s_block = torch.matmul(v_block, do_block.t())
                s_block = s_block * causal_decay_mask
                acc_dq = torch.matmul(s_block.t(), k_block)
                acc_dq = acc_dq + torch.matmul(do_block, ssm_states_block) * exp_da_cs[:, None]
                dq[b, chunk_start:chunk_end, h, :] = acc_dq.to(dtype)

                p_t_block = torch.matmul(k_block, q_block.t())
                p_t_block = p_t_block * causal_decay_mask
                acc_dv = torch.matmul(p_t_block, do_block)
                acc_dv = acc_dv + torch.matmul(k_block, d_ssm_states_acc.t()) * exp_da_cs_rev[:, None]

                if D is not None:
                    acc_dv = acc_dv + do_block * (D_val + qk_dot_h[chunk_start:chunk_end, None])
                else:
                    acc_dv = acc_dv + do_block * qk_dot_h[chunk_start:chunk_end, None]

                dv[b, chunk_start:chunk_end, h, :] = acc_dv.to(dtype)

                dQK_dot_block = torch.matmul(do_block * v_block, torch.ones(headdim_v, 1, device=device))
                dQK[b, h, chunk_start:chunk_end] = dQK_dot_block.squeeze(-1).to(dtype)

                if D is not None:
                    dD_acc = dD_acc + dQK_dot_block.sum()

                QS = torch.matmul(q_block, ssm_states_block.t())
                dM_rev_vector = dM_rev_vector + (QS * do_block).sum(dim=1) * exp_da_cs

                SSM_States_reloaded = ssm_states_h[:, chunk_idx * headdim_qk:(chunk_idx + 1) * headdim_qk]
                dM_scalar = (SSM_States_reloaded * d_ssm_states_acc).sum() * torch.exp2(da_cs_chunk_sum)

                dSK = torch.matmul(k_block, d_ssm_states_acc.t())
                dM_vector = (dSK * v_block).sum(dim=1) * exp_da_cs_rev

                dM_rev_vector_full[chunk_start:chunk_end] = dM_rev_vector + (
                            dM_rev_vector.sum() + dM_scalar) + torch.cumsum(dM_vector - dM_rev_vector,
                                                                            dim=0) - dM_vector

                do_scaled = do_block * exp_da_cs[:, None]
                d_ssm_states_acc = torch.exp2(da_cs_chunk_sum) * d_ssm_states_acc + torch.matmul(do_scaled.t(), q_block)

            dAdt[b, h, :] = dM_rev_vector_full.to(dtype)

            if D is not None:
                dD[b, h] = dD_acc

            if has_input_state:
                d_issm_state[b, h, :, :] = d_ssm_states_acc

    if d_ov_state is not None:
        if is_varlen:
            last_token_idx = Cu_Seqlens[1:] - 1
            dv[0, last_token_idx] += d_ov_state
        else:
            dv[:, -1, :, :] += d_ov_state

    dD = dD.sum(dim=0) if dD is not None else None
    return dq, dk, dv, dAdt, dQK, dD, d_issm_state
