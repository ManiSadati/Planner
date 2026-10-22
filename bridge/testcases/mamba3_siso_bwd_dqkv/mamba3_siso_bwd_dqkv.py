"""Standalone original Mamba-3 backward dQ/dK/dV kernel.

Source: Q2TritonKernel/tests/mamba/test_mamba3_siso_bwd_kernel_dqkv/
mamba3_siso_fwd_kernel_npu.py:15. The entire kernel body is unchanged.
Only the single-config autotune wrapper and host wrapper are replaced.
Default B=1,T=128,H=Hqk=1,Dqk=Dv=64,CHUNK_SIZE=32,FP32,
RECOMPUTE_MASK=True,HAS_D_OSSM_STATE=False,RETURN_D_ISSM_STATE=False,
IS_VARLEN=False,num_warps=4,num_stages=2 follows the retained capture.
No host retiling or optional skip/final-state branch is enabled.
Inputs/reference/output initialization run on CPU; native NPU-IR is default.
This is a synthetic input set, not the external .pt accuracy-test fixture.
A fresh AVE dump is required to reconfirm predicate UB stores/reloads.
"""

import argparse
import os

import torch
import triton
import triton.language as tl

from reference import compute_dqkv_pytorch


@triton.jit
def mamba3_siso_bwd_kernel_dqkv(
        # Input tensors
        Q, K, V, DA_CS, DA_CS_SUM, QK_Dot, D, SSM_States, dO, d_OSSM_State, Cu_Seqlens,  # dO is scaled with Z
        # Output tensors
        dQ, dK, dV, dADT, dQK_Dot, dD, d_ISSM_State,  # dQK_Dot is scaled with scale
        # Strides for Inputs
        # Strides for Q: (batch, seqlen, nheads_qk, HEADDIM_QK)
        stride_q_batch, stride_q_seqlen, stride_q_head, stride_q_qkdim,
        # Strides for K: (batch, seqlen, nheads_qk, HEADDIM_QK)
        stride_k_batch, stride_k_seqlen, stride_k_head, stride_k_qkdim,
        # Strides for V: (batch, seqlen, nheads, HEADDIM_V)
        stride_v_batch, stride_v_seqlen, stride_v_head, stride_v_vdim,
        # Strides for DA_CS: (batch, nheads, seqlen)
        stride_da_cs_batch, stride_da_cs_head, stride_da_cs_seqlen,
        # Strides for DA_CS_SUM: (batch, nheads, nchunks)
        stride_da_cs_sum_batch, stride_da_cs_sum_head, stride_da_cs_sum_seqlen,
        # Strides for QK (QK dot products): (batch, nheads, nchunks*CHUNK_SIZE)
        stride_qk_dot_batch, stride_qk_dot_head, stride_qk_dot_seqlen,
        # Strides for D: (nheads,)
        stride_d_head,
        # Strides for SSM_States: (batch, nheads, HEADDIM_V, nchunks*HEADDIM_QK)
        stride_ssm_states_batch, stride_ssm_states_head, stride_ssm_states_vdim, stride_ssm_states_qkdim,
        # Strides for dO: (batch, seqlen, nheads, HEADDIM_V)
        stride_do_batch, stride_do_seqlen, stride_do_head, stride_do_vdim,
        # Strides for d_OSSM_State: (num_sequences, nheads, HEADDIM_V, HEADDIM_QK)
        stride_d_ossm_state_batch, stride_d_ossm_state_head, stride_d_ossm_state_vdim, stride_d_ossm_state_qkdim,
        # Strides for Cu_Seqlens: (num_sequences + 1,)
        stride_cu_seqlen,
        # Strides for Outputs
        # Strides for dQ: (batch, seqlen, nheads, HEADDIM_QK)
        stride_dq_batch, stride_dq_seqlen, stride_dq_head, stride_dq_qkdim,
        # Strides for dK: (batch, seqlen, nheads, HEADDIM_QK)
        stride_dk_batch, stride_dk_seqlen, stride_dk_head, stride_dk_qkdim,
        # Strides for dV: (batch, seqlen, nheads, HEADDIM_V)
        stride_dv_batch, stride_dv_seqlen, stride_dv_head, stride_dv_vdim,
        # Strides for dAdt: (batch, nheads, seqlen)
        stride_dadt_batch, stride_dadt_head, stride_dadt_seqlen,
        # Strides for dQK_dot: (batch, nheads, seqlen)
        stride_dQK_dot_batch, stride_dQK_dot_head, stride_dQK_dot_seqlen,
        # Strides for dD: (nheads,)
        stride_dd_batch, stride_dd_head,
        # Strides for d_ISSM_State: (num_sequences, nheads, HEADDIM_V, HEADDIM_QK)
        stride_d_issm_state_batch, stride_d_issm_state_head, stride_d_issm_state_vdim, stride_d_issm_state_qkdim,
        # Dimensions
        seqlen, nheads_qk, headdim_qk, headdim_v,
        CHUNK_SIZE: tl.constexpr,
        HEADDIM_QK: tl.constexpr,
        HEADDIM_V: tl.constexpr,
        RECOMPUTE_MASK: tl.constexpr,
        HAS_D_OSSM_STATE: tl.constexpr,
        RETURN_D_ISSM_STATE: tl.constexpr,
        IS_VARLEN: tl.constexpr,
):
    """
    Backward kernel for Mamba-3 attention mechanism.

    Each program instance handles one (head, batch/seq) pair and iterates through
    all chunks in reverse order. This reverse iteration is necessary because
    state gradients flow backward through the sequence.

    The kernel computes:
        - dQ, dK: Gradients for query/key from both intra-chunk attention and inter-chunk states
        - dV: Gradient for values
        - dADT: Gradient for the decay parameter (A * dt)
        - dQK_Dot: Gradient for the QK dot product term
        - dD: Gradient for the skip connection (if present)
        - dISSM_State: Gradient for the input SSM state (if present)

    Grid:
        - Normal mode: (nheads, batch)
        - Varlen mode: (nheads, num_sequences)
    """
    # ==================== Program Indexing ====================
    pid_head = tl.program_id(0)
    pid_batch = tl.program_id(1)

    if IS_VARLEN:
        pid_seq = pid_batch
        pid_batch = 0
        cu_seqlen = tl.load(Cu_Seqlens + pid_seq * stride_cu_seqlen).to(tl.int32)
        cu_seqlen_next = tl.load(Cu_Seqlens + (pid_seq + 1) * stride_cu_seqlen).to(tl.int32)
        seqlen = cu_seqlen_next - cu_seqlen
        cu_chunks = pid_seq + cu_seqlen // CHUNK_SIZE
    else:
        cu_seqlen = 0
        cu_chunks = 0
        pid_seq = 0

    # Compute Q/K head index for GQA (grouped query attention)
    # Multiple output heads may share the same Q/K head
    nheads = tl.num_programs(0)
    head_idx_qk = pid_head // (nheads // nheads_qk)

    # Input Pointer Offsets
    q_offset = pid_batch * stride_q_batch + head_idx_qk * stride_q_head + IS_VARLEN * cu_seqlen * stride_q_seqlen
    k_offset = pid_batch * stride_k_batch + head_idx_qk * stride_k_head + IS_VARLEN * cu_seqlen * stride_k_seqlen
    v_offset = pid_batch * stride_v_batch + pid_head * stride_v_head + IS_VARLEN * cu_seqlen * stride_v_seqlen
    da_cs_offset = pid_batch * stride_da_cs_batch + pid_head * stride_da_cs_head + IS_VARLEN * cu_seqlen * stride_da_cs_seqlen
    da_cs_sum_offset = pid_batch * stride_da_cs_sum_batch + pid_head * stride_da_cs_sum_head + IS_VARLEN * cu_chunks * stride_da_cs_sum_seqlen
    qk_dot_offset = pid_batch * stride_qk_dot_batch + pid_head * stride_qk_dot_head + IS_VARLEN * cu_seqlen * stride_qk_dot_seqlen
    ssm_states_offset = pid_batch * stride_ssm_states_batch + pid_head * stride_ssm_states_head + IS_VARLEN * cu_chunks * HEADDIM_QK * stride_ssm_states_qkdim
    do_offset = pid_batch * stride_do_batch + pid_head * stride_do_head + IS_VARLEN * cu_seqlen * stride_do_seqlen
    if HAS_D_OSSM_STATE:
        d_ossm_state_offset = (
                                          pid_batch + IS_VARLEN * pid_seq) * stride_d_ossm_state_batch + pid_head * stride_d_ossm_state_head

    # Load skip connection value D if present
    if D is not None:
        D_offset = pid_head * stride_d_head
        D_val = tl.load(D + D_offset)

    # Output Pointer Offsets
    dq_offset = pid_batch * stride_dq_batch + pid_head * stride_dq_head + IS_VARLEN * cu_seqlen * stride_dq_seqlen
    dk_offset = pid_batch * stride_dk_batch + pid_head * stride_dk_head + IS_VARLEN * cu_seqlen * stride_dk_seqlen
    dv_offset = pid_batch * stride_dv_batch + pid_head * stride_dv_head + IS_VARLEN * cu_seqlen * stride_dv_seqlen
    dadt_offset = pid_batch * stride_dadt_batch + pid_head * stride_dadt_head + IS_VARLEN * cu_seqlen * stride_dadt_seqlen
    dQK_dot_offset = pid_batch * stride_dQK_dot_batch + pid_head * stride_dQK_dot_head + IS_VARLEN * cu_seqlen * stride_dQK_dot_seqlen

    if D is not None:
        dD_offset = pid_head * stride_dd_head + pid_batch * stride_dd_batch + IS_VARLEN * pid_seq * stride_dd_batch
        dD_acc = tl.zeros([1], dtype=tl.float32)

    if RETURN_D_ISSM_STATE:
        d_issm_state_offset = (
                                          pid_batch + IS_VARLEN * pid_seq) * stride_d_issm_state_batch + pid_head * stride_d_issm_state_head

    # Accumulates gradients flowing backward through states across chunks
    if HAS_D_OSSM_STATE:
        d_ssm_ptrs = d_OSSM_State + d_ossm_state_offset + tl.arange(0, HEADDIM_V)[:,
                                                          None] * stride_d_ossm_state_vdim + tl.arange(0, HEADDIM_QK)[
                                                                                             None,
                                                                                             :] * stride_d_ossm_state_qkdim
        d_ssm_states_mask = (tl.arange(0, HEADDIM_V)[:, None] < headdim_v) & (
                    tl.arange(0, HEADDIM_QK)[None, :] < headdim_qk)
        d_ssm_states_acc = tl.load(d_ssm_ptrs, mask=d_ssm_states_mask, other=0.0).to(tl.float32)
    else:
        d_ssm_states_acc = tl.zeros([HEADDIM_V, HEADDIM_QK], dtype=tl.float32)

    num_chunks = tl.cdiv(seqlen, CHUNK_SIZE)

    #  TMA Descriptors for Efficient Memory Access
    q_desc = tl.make_tensor_descriptor(
        Q + q_offset,
        shape=[seqlen, headdim_qk],
        strides=[stride_q_seqlen, stride_q_qkdim],
        block_shape=[CHUNK_SIZE, HEADDIM_QK],
    )
    k_desc = tl.make_tensor_descriptor(
        K + k_offset,
        shape=[seqlen, headdim_qk],
        strides=[stride_k_seqlen, stride_k_qkdim],
        block_shape=[CHUNK_SIZE, HEADDIM_QK],
    )
    v_desc = tl.make_tensor_descriptor(
        V + v_offset,
        shape=[seqlen, headdim_v],
        strides=[stride_v_seqlen, stride_v_vdim],
        block_shape=[CHUNK_SIZE, HEADDIM_V],
    )
    ssm_states_desc = tl.make_tensor_descriptor(
        SSM_States + ssm_states_offset,
        shape=[headdim_v, num_chunks * headdim_qk],
        strides=[stride_ssm_states_vdim, stride_ssm_states_qkdim],
        block_shape=[HEADDIM_V, HEADDIM_QK],
    )
    do_desc = tl.make_tensor_descriptor(
        dO + do_offset,
        shape=[seqlen, headdim_v],
        strides=[stride_do_seqlen, stride_do_vdim],
        block_shape=[CHUNK_SIZE, HEADDIM_V],
    )
    dq_desc = tl.make_tensor_descriptor(
        dQ + dq_offset,
        shape=[seqlen, headdim_qk],
        strides=[stride_dq_seqlen, stride_dq_qkdim],
        block_shape=[CHUNK_SIZE, HEADDIM_QK],
    )
    dk_desc = tl.make_tensor_descriptor(
        dK + dk_offset,
        shape=[seqlen, headdim_qk],
        strides=[stride_dk_seqlen, stride_dk_qkdim],
        block_shape=[CHUNK_SIZE, HEADDIM_QK],
    )
    dv_desc = tl.make_tensor_descriptor(
        dV + dv_offset,
        shape=[seqlen, headdim_v],
        strides=[stride_dv_seqlen, stride_dv_vdim],
        block_shape=[CHUNK_SIZE, HEADDIM_V],
    )

    for chunk_idx_loop in range(num_chunks):
        chunk_idx = num_chunks - 1 - chunk_idx_loop  # Reverse order for backward pass
        chunk_start = chunk_idx * CHUNK_SIZE

        # Sequence-length mask for non-TMA loads/stores
        offs_cs = chunk_start + tl.arange(0, CHUNK_SIZE)
        seq_mask = offs_cs < seqlen

        # ============================================================
        # Load Decay Values
        # We load these first to overlap computation with TMA loads
        # ============================================================
        da_cs_ptrs = DA_CS + da_cs_offset + offs_cs * stride_da_cs_seqlen
        da_cs = tl.load(da_cs_ptrs, mask=seq_mask, other=0.0)  # Cumulative decay within chunk: (CHUNK_SIZE,)

        da_cs_sum_ptrs = DA_CS_SUM + da_cs_sum_offset + chunk_idx * stride_da_cs_sum_seqlen
        da_cs_chunk_sum = tl.load(da_cs_sum_ptrs)  # Total decay for this chunk: scalar

        # ============================================================
        # Load Q, K, V, dO, SSM_States via TMA
        # ============================================================
        do_block = do_desc.load([chunk_start, 0])  # (CHUNK_SIZE, HEADDIM_V)
        v_block = v_desc.load([chunk_start, 0])  # (CHUNK_SIZE, HEADDIM_V)
        q_block = q_desc.load([chunk_start, 0])  # (CHUNK_SIZE, HEADDIM_QK)
        k_block = k_desc.load([chunk_start, 0])  # (CHUNK_SIZE, HEADDIM_QK)
        ssm_states_block = ssm_states_desc.load([0, chunk_idx * headdim_qk])  # (HEADDIM_V, HEADDIM_QK)

        # ============================================================
        # Compute Decay Scaling Factors
        # ============================================================
        # Reverse cumsum: how much decay from position i to end of chunk
        da_cs_rev = da_cs_chunk_sum - da_cs
        exp_da_cs_rev = tl.math.exp2(da_cs_rev)  # For scaling inter-chunk contributions
        exp_da_cs = tl.math.exp2(da_cs)  # For scaling intra-chunk contributions

        # Compute strictly causal mask with exponential decay (this is L^T)
        if not RECOMPUTE_MASK:
            causal_decay_mask = tl.where(
                tl.arange(0, CHUNK_SIZE)[None, :] > tl.arange(0, CHUNK_SIZE)[:, None],
                tl.math.exp2(tl.minimum(da_cs[None, :] - da_cs[:, None], 0.0)),
                0.0
            )

        # ============================================================
        # Compute dADT Gradient (Part 1): From Intra-chunk Attention
        # This is register-heavy so we compute it early before spilling
        # ============================================================
        # Gradient contribution from (QK^T ⊙ L) V term
        dAinv = tl.dot(v_block, tl.trans(do_block))  # V @ dO^T
        if RECOMPUTE_MASK:
            dAinv *= tl.math.exp2(tl.minimum(da_cs[None, :] - da_cs[:, None], 0.0))
            dAinv = tl.where(
                tl.arange(0, CHUNK_SIZE)[None, :] > tl.arange(0, CHUNK_SIZE)[:, None],
                dAinv,
                0.0
            )
        else:
            dAinv *= causal_decay_mask
        dAinv *= tl.dot(k_block, tl.trans(q_block))  # Element-wise with K @ Q^T
        dM_rev_vector = tl.sum(dAinv, axis=0) - tl.sum(dAinv, axis=1)  # (CHUNK_SIZE,)

        # ============================================================
        # Compute dK: Key Gradient
        # dK = (V @ dO^T ⊙ mask)^T @ Q + V @ dStates * scale
        # ============================================================
        # Intra-chunk: dP^T @ Q where dP = dO @ V^T ⊙ mask
        dp_t_block = tl.dot(v_block, tl.trans(do_block))  # V @ dO^T: (CHUNK_SIZE, CHUNK_SIZE)
        if RECOMPUTE_MASK:
            dp_t_block *= tl.math.exp2(tl.minimum(da_cs[None, :] - da_cs[:, None], 0.0))
            dp_t_block = tl.where(
                tl.arange(0, CHUNK_SIZE)[None, :] > tl.arange(0, CHUNK_SIZE)[:, None],
                dp_t_block,
                0.0
            )
        else:
            dp_t_block *= causal_decay_mask

        acc_dk = tl.dot(dp_t_block.to(q_block.dtype), q_block)  # (CHUNK_SIZE, HEADDIM_QK)

        # Inter-chunk: gradient flowing through accumulated states
        acc_dk += tl.dot(v_block, d_ssm_states_acc.to(v_block.dtype)) * exp_da_cs_rev[:, None]

        dk_desc.store([chunk_start, 0], acc_dk.to(q_block.dtype))

        # ============================================================
        # Compute dQ: Query Gradient
        # dQ = (V @ dO^T ⊙ mask) @ K + dO @ States * scale
        # ============================================================
        # Intra-chunk: S^T @ K where S = V @ dO^T ⊙ mask
        s_block = tl.dot(v_block, tl.trans(do_block))  # (CHUNK_SIZE, CHUNK_SIZE)
        if RECOMPUTE_MASK:
            s_block *= tl.math.exp2(tl.minimum(da_cs[None, :] - da_cs[:, None], 0.0))
            s_block = tl.where(
                tl.arange(0, CHUNK_SIZE)[None, :] > tl.arange(0, CHUNK_SIZE)[:, None],
                s_block,
                0.0
            )
        else:
            s_block *= causal_decay_mask

        acc_dq = tl.dot(tl.trans(s_block).to(k_block.dtype), k_block)  # (CHUNK_SIZE, HEADDIM_QK)

        # Inter-chunk: gradient through states from previous chunks
        acc_dq += tl.dot(do_block, ssm_states_block) * exp_da_cs[:, None]

        dq_desc.store([chunk_start, 0], acc_dq.to(q_block.dtype))

        # ============================================================
        # Compute dV: Value Gradient
        # dV = (K @ Q^T ⊙ mask) @ dO + K @ dStates^T * scale + dO * (D + qk_dot)
        # ============================================================
        # Intra-chunk: P^T @ dO where P = Q @ K^T ⊙ mask
        p_t_block = tl.dot(k_block, tl.trans(q_block))  # K @ Q^T: (CHUNK_SIZE, CHUNK_SIZE)
        if RECOMPUTE_MASK:
            p_t_block *= tl.math.exp2(tl.minimum(da_cs[None, :] - da_cs[:, None], 0.0))
            p_t_block = tl.where(
                tl.arange(0, CHUNK_SIZE)[None, :] > tl.arange(0, CHUNK_SIZE)[:, None],
                p_t_block,
                0.0
            )
        else:
            p_t_block *= causal_decay_mask

        acc_dv = tl.dot(p_t_block.to(do_block.dtype), do_block)  # (CHUNK_SIZE, HEADDIM_V)

        # Inter-chunk: gradient through states
        acc_dv += tl.dot(k_block, tl.trans(d_ssm_states_acc).to(k_block.dtype)) * exp_da_cs_rev[:, None]

        # Skip connection gradient contribution
        # Load dO again with volatile to avoid cache conflicts
        dO_reloaded = tl.load(
            dO + do_offset + offs_cs[:, None] * stride_do_seqlen +
            tl.arange(0, HEADDIM_V)[None, :] * stride_do_vdim,
            mask=seq_mask[:, None] & (tl.arange(0, HEADDIM_V)[None, :] < headdim_v),
            other=0.0,
            volatile=True
        )

        qk_dot = tl.load(QK_Dot + qk_dot_offset + offs_cs * stride_qk_dot_seqlen, mask=seq_mask, other=0.0)
        if D is not None:
            acc_dv += dO_reloaded * (D_val + qk_dot[:, None])
        else:
            acc_dv += dO_reloaded * qk_dot[:, None]

        dv_desc.store([chunk_start, 0], acc_dv.to(do_block.dtype))

        # ============================================================
        # Compute dQK_Dot and dD: Skip Connection Gradients
        # ============================================================
        v_block_reloaded = tl.load(
            V + v_offset + offs_cs[:, None] * stride_v_seqlen +
            tl.arange(0, HEADDIM_V)[None, :] * stride_v_vdim,
            mask=seq_mask[:, None] & (tl.arange(0, HEADDIM_V)[None, :] < headdim_v),
            other=0.0,
            volatile=True
        )

        # dQK_dot = sum_v(dO * V) for each position
        dQK_dot_block = tl.sum(dO_reloaded * v_block_reloaded, axis=1)

        tl.store(
            dQK_Dot + dQK_dot_offset + offs_cs * stride_dQK_dot_seqlen,
            dQK_dot_block,
            mask=seq_mask
        )

        # Accumulate dD gradient
        if D is not None:
            dD_acc += tl.sum(dQK_dot_block)

        # ============================================================
        # Compute dADT Gradient (Part 2): From Inter-chunk States
        # ============================================================
        # Gradient from Q @ States^T term
        QS = tl.dot(q_block, tl.trans(ssm_states_block))  # (CHUNK_SIZE, HEADDIM_V)
        dM_rev_vector += tl.sum(QS * dO_reloaded, axis=1) * exp_da_cs  # (CHUNK_SIZE,)

        # ============================================================
        # Compute dADT Gradient (Part 3): From State Accumulation
        # ============================================================
        # Gradient flowing through d_ssm_states_acc @ SSM_States
        SSM_States_ptrs = (SSM_States + ssm_states_offset +
                           tl.arange(0, HEADDIM_V)[:, None] * stride_ssm_states_vdim +
                           (chunk_idx * headdim_qk + tl.arange(0, HEADDIM_QK)[None, :]) * stride_ssm_states_qkdim)
        SSM_States_mask = (tl.arange(0, HEADDIM_V)[:, None] < headdim_v) & (
                    (chunk_idx * headdim_qk + tl.arange(0, HEADDIM_QK)[None, :]) < num_chunks * headdim_qk)

        SSM_States_reloaded = tl.load(SSM_States_ptrs, volatile=True, mask=SSM_States_mask)  # (HEADDIM_V, HEADDIM_QK)
        dM_scalar = tl.sum(SSM_States_reloaded * d_ssm_states_acc) * tl.math.exp2(da_cs_chunk_sum)

        # ============================================================
        # Compute dADT Gradient (Part 4): From K @ dStates
        # ============================================================
        dSK = tl.dot(k_block, tl.trans(d_ssm_states_acc).to(k_block.dtype))  # (CHUNK_SIZE, HEADDIM_V)
        dM_vector = tl.sum(dSK * v_block_reloaded, axis=1) * exp_da_cs_rev  # (CHUNK_SIZE,)

        # ============================================================
        # Combine dADT Gradient Components via Reverse Cumsum
        # ============================================================
        dM_rev_vector += (tl.sum(dM_rev_vector) + dM_scalar) + tl.cumsum(dM_vector - dM_rev_vector) - dM_vector

        # Store dADT
        dadt_ptrs = dADT + dadt_offset + offs_cs * stride_dadt_seqlen
        tl.store(dadt_ptrs, dM_rev_vector, mask=seq_mask)

        # ============================================================
        # Accumulate State Gradients for Previous Chunks
        # ============================================================
        dO_reloaded *= exp_da_cs[:, None]
        d_ssm_states_acc = (tl.math.exp2(da_cs_chunk_sum) * d_ssm_states_acc +
                            tl.dot(tl.trans(dO_reloaded).to(q_block.dtype), q_block))

    # Store Final dD Gradient
    if D is not None:
        tl.store(dD + dD_offset + tl.arange(0, 1), dD_acc)

    # Store d_ISSM_State
    if RETURN_D_ISSM_STATE:
        d_ISSM_State_ptrs = d_ISSM_State + d_issm_state_offset + tl.arange(0, HEADDIM_V)[:,
                                                                 None] * stride_d_issm_state_vdim + tl.arange(0,
                                                                                                              HEADDIM_QK)[
                                                                                                    None,
                                                                                                    :] * stride_d_issm_state_qkdim
        d_ISSM_State_mask = (tl.arange(0, HEADDIM_V)[:, None] < headdim_v) & (
                    tl.arange(0, HEADDIM_QK)[None, :] < headdim_qk)
        tl.store(d_ISSM_State_ptrs, d_ssm_states_acc, mask=d_ISSM_State_mask)



def make_inputs(batch=1, length=128, heads=1):
    """CPU-only FP32 inputs with per-chunk cumulative decay and forward states."""
    torch.manual_seed(42)
    shape = (batch, length, heads, 64)
    q, k, v, do = [torch.randn(shape) * 0.1 for _ in range(4)]
    chunks = length // 32
    decay = -torch.rand(batch, heads, chunks, 32) * 0.02
    da_cs = decay.cumsum(-1).reshape(batch, heads, length)
    # Padding preserves the capture's DA_CS_SUM batch/head strides (128,128).
    da_cs_sum = torch.zeros(batch, heads, length)
    da_cs_sum[..., :chunks] = da_cs.reshape(batch, heads, chunks, 32)[..., -1]
    state = torch.zeros(batch, heads, 64, 64)
    states = []
    for chunk in range(chunks):
        states.append(state)
        start = chunk * 32
        cs = da_cs[..., start:start + 32]
        total = da_cs_sum[..., chunk]
        weights = torch.exp2(total[..., None] - cs)
        kc = k[:, start:start + 32].permute(0, 2, 1, 3)
        vc = v[:, start:start + 32].permute(0, 2, 1, 3)
        state = torch.exp2(total)[..., None, None] * state + (
            vc * weights[..., None]).transpose(-1, -2) @ kc
    ssm_states = torch.cat(states, dim=-1)
    qk_dot = torch.empty(batch, heads, length)
    qk_dot.copy_((q * k).sum(-1).permute(0, 2, 1))
    return dict(q=q, k=k, v=v, da_cs=da_cs, da_cs_sum=da_cs_sum,
                qk_dot=qk_dot, SSM_States=ssm_states, do=do)


def launch(inputs, outputs, batch, length, heads, compile_flow):
    q, k, v, cs, sums, qk, states, do = (
        inputs[name] for name in ('q', 'k', 'v', 'da_cs', 'da_cs_sum', 'qk_dot', 'SSM_States', 'do'))
    dq, dk, dv, dadt, dqk = outputs
    # Inactive pointer/stride arguments are explicit; active strides come from tensors.
    mamba3_siso_bwd_kernel_dqkv[(heads, batch)](
        q, k, v, cs, sums, qk, None, states, do, None, None,
        dq, dk, dv, dadt, dqk, None, None,
        *q.stride(), *k.stride(), *v.stride(),
        *cs.stride(), *sums.stride(), *qk.stride(),
        64, *states.stride(), *do.stride(),
        0, 0, 0, 0, 64,
        *dq.stride(), *dk.stride(), *dv.stride(),
        *dadt.stride(), *dqk.stride(),
        0, 0, 0, 0, 0, 0,
        length, heads, 64, 64,
        CHUNK_SIZE=32, HEADDIM_QK=64, HEADDIM_V=64,
        RECOMPUTE_MASK=True, HAS_D_OSSM_STATE=False,
        RETURN_D_ISSM_STATE=False, IS_VARLEN=False,
        num_warps=4, num_stages=2, debug=True, compile_flow=compile_flow,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--length', type=int, default=128)
    parser.add_argument('--heads', type=int, default=1)
    parser.add_argument('--compile-flow', choices=('npuir', 'ptoas'), default='npuir')
    args = parser.parse_args()
    if min(args.batch, args.length, args.heads) < 1 or args.length % 32:
        parser.error('batch/heads must be positive and length a positive multiple of 32')
    os.environ['TRITON_ASCEND_COMPILE_FLOW'] = args.compile_flow
    if args.compile_flow == 'npuir':
        os.environ.pop('BISHENGIR_ENABLE_PTOAS_BRIDGE', None)
        os.environ.pop('BISHENGIR_PTOAS_BRIDGE_MODE', None)
    else:
        os.environ['BISHENGIR_ENABLE_PTOAS_BRIDGE'] = '1'
        os.environ['BISHENGIR_PTOAS_BRIDGE_MODE'] = 'ptodsl'
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'

    import torch_npu  # Registers the real or simulated NPU runtime.
    if not torch.npu.is_available():
        raise RuntimeError('Run on an NPU or under the configured CANN msprof op simulator')

    cpu_inputs = make_inputs(args.batch, args.length, args.heads)
    # This branch produces five tensors; dD/dInput_SSM_State are disabled.
    expected = compute_dqkv_pytorch(**cpu_inputs, chunk_size=32)
    inputs = {name: value.npu() for name, value in cpu_inputs.items()}
    outputs = [torch.full_like(value, float('nan')).npu() for value in expected[:5]]
    print(f'flow={args.compile_flow}, B={args.batch}, T={args.length}, '
          f'H=Hqk={args.heads}, Dqk=Dv=64, CHUNK_SIZE=32, '
          'RECOMPUTE_MASK=True, num_warps=4, num_stages=2', flush=True)
    launch(inputs, outputs, args.batch, args.length, args.heads, args.compile_flow)
    torch.npu.synchronize()
    # FP32 1e-5 elementwise tolerances follow the original direct test helper.
    for name, actual, wanted in zip(('dQ', 'dK', 'dV', 'dADT', 'dQK_Dot'), outputs, expected):
        actual = actual.cpu()
        torch.testing.assert_close(actual, wanted, atol=1e-5, rtol=1e-5)
        print(f'PASS {name}: max absolute error={(actual - wanted).abs().max().item():.8g}', flush=True)
    print('PASS: all five enabled outputs compared', flush=True)


if __name__ == '__main__':
    main()
