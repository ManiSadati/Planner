# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Standalone original gated chunk backward kernel: multiple continuous stores.

Source: Q2TritonKernel/src/kernels/fla/ops/common/chunk_o.py:160.
Benchmark: Q2TritonKernel/tests/fla_perf/test_chunk_bwd_dqkwg_perf.py:129.
Normal exp/exp2 helpers: Q2TritonKernel/src/kernels/fla/ops/utils/op.py:29/31.
The full kernel, helper bodies and do_not_specialize=['T'] are preserved.
FLA host imports, heuristic flags, autotuning and wrapper dispatch are replaced
with explicit launch parameters. FLA_USE_FAST_OPS=0 is selected explicitly.

Default small case: B=1,T=64,H=HV=2,K=V=128,BT=64,BK=BV=128, BF16 inputs,
BF16 dq/dk, FP32 dg, USE_G=True, USE_EXP2=False, USE_G_GAMMA=False,
USE_DW=False, TRANSPOSE_STATE=False, IS_VARLEN=False; two warps/stages.
The retained evidence used B=4,T=131072 with the same H/tiles/flags. Use
--batch 4 --length 131072 to restore those dimensions. Two warps/stages are
explicit capture settings, not a measured autotuner winner.

CPU inputs/state builders/reference avoid auxiliary NPU initialization kernels.
Native compile_flow='npuir' is default; --compile-flow ptoas selects the bridge.
Run on an NPU or under configured CANN simulation. Fresh native AVE IR must
confirm that both row-reduction stores are ONEPT_B32 with hivm.is_continuous;
no store or loop is manually added to the original Triton code.
"""

import argparse
import os

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from reference import naive_chunk_fwd_h, naive_chunk_bwd_dh, naive_chunk_bwd_dqkwg


@triton.jit
def exp(x): return tl.exp(x.to(tl.float32))


@triton.jit
def exp2(x): return tl.math.exp2(x.to(tl.float32))


@triton.jit(do_not_specialize=['T'])
def chunk_bwd_kernel_dqkwg(
    q,
    k,
    v,
    g,
    g_gamma,
    h,
    do,
    dh,
    dq,
    dk,
    dw,
    dv,
    dg,
    cu_seqlens,
    chunk_indices,
    scale,
    B: tl.constexpr,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_G_GAMMA: tl.constexpr,
    USE_EXP2: tl.constexpr,
    USE_DW: tl.constexpr,
    TRANSPOSE_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // HV, i_bh % HV

    all = B * T
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    # offset calculation
    v += (bos * HV + i_h) * V
    do += (bos * HV + i_h) * V
    h += (i_tg * HV + i_h).to(tl.int64) * K*V
    dh += (i_tg * HV + i_h).to(tl.int64) * K*V
    q += (bos * H + i_h // (HV // H)) * K
    k += (bos * H + i_h // (HV // H)) * K
    dq += (bos * HV + i_h) * K
    dk += (bos * HV + i_h) * K

    # for delta rule only
    if USE_DW:
        dw += (bos * HV + i_h) * K
        dv += (bos * HV + i_h) * V

    if USE_G:
        dg += i_k * all * HV
        b_dg_last = tl.zeros([1], dtype=tl.float32) if USE_G else None
    if USE_G_GAMMA:
        b_gamma = tl.load(g_gamma + i_h)
        b_g = b_gamma * (tl.arange(0, BT) + 1)
        b_g_last = b_gamma * min(BT, T - i_t * BT)
    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_ds = tl.zeros([BT, BT], dtype=tl.float32)
    b_dw = tl.zeros([BT, BK], dtype=tl.float32) if USE_DW else None

    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v, (T, V), (HV*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_do = tl.make_block_ptr(do, (T, V), (HV*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        if TRANSPOSE_STATE:
            p_h = tl.make_block_ptr(h, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0))
            p_dh = tl.make_block_ptr(dh, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0))
        else:
            p_h = tl.make_block_ptr(h, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
            p_dh = tl.make_block_ptr(dh, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        # [BT, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        # [BV, BK]
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_dh = tl.load(p_dh, boundary_check=(0, 1))
        if USE_G:
            b_dg_last += (tl.sum(b_h * b_dh))
        # [BT, BV] @ [BV, BT] -> [BT, BT]
        b_ds += tl.dot(b_do, tl.trans(b_v))
        # [BT, BV] @ [BV, BK] -> [BT, BK]
        b_dq += tl.dot(b_do, b_h.to(b_do.dtype))
        # [BT, BV] @ [BV, BK] -> [BT, BK]
        b_dk += tl.dot(b_v, b_dh.to(b_v.dtype))
        if USE_DW:
            p_dv = tl.make_block_ptr(dv, (T, V), (HV*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            b_dv = tl.load(p_dv, boundary_check=(0, 1))
            b_dw += tl.dot(b_dv.to(b_v.dtype), b_h.to(b_v.dtype))

    if USE_DW:
        p_dw = tl.make_block_ptr(dw, (T, K), (HV*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        tl.store(p_dw, -b_dw.to(p_dw.dtype.element_ty), boundary_check=(0, 1))

    tl.debug_barrier()
    p_q = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))

    p_dq = tl.make_block_ptr(dq, (T, K), (HV*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dk = tl.make_block_ptr(dk, (T, K), (HV*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
    if USE_G:
        g += bos * HV + i_h
        dg += bos * HV + i_h
        p_g = tl.make_block_ptr(g, (T,), (HV,), (i_t * BT,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,))
        b_g_last = tl.load(g + (min(i_t * BT + BT, T) - 1) * HV)
        if USE_EXP2:
            b_dg_last *= exp2(b_g_last)
            b_dq = b_dq * exp2(b_g)[:, None] * scale
        else:
            b_dg_last *= exp(b_g_last)
            b_dq = b_dq * exp(b_g)[:, None] * scale

        if USE_EXP2:
            b_dk = b_dk * tl.where(m_t, exp2(-b_g + b_g_last), 0)[:, None]
        else:
            b_dk = b_dk * tl.where(m_t, exp(-b_g + b_g_last), 0)[:, None]
        b_dg_last += tl.sum(b_dk * b_k)

        if USE_EXP2:
            b_ds = tl.where(m_A, b_ds * exp2(b_g[:, None] - b_g[None, :]), 0) * scale
        else:
            b_ds = tl.where(m_A, b_ds * exp(b_g[:, None] - b_g[None, :]), 0) * scale

        b_ds = b_ds.to(b_k.dtype)
        # [BT, BK]
        b_dq += tl.dot(b_ds, b_k)
        b_dk += tl.dot(tl.trans(b_ds), b_q)

        b_dg = tl.sum(b_dq * b_q, axis=1) - tl.sum(b_dk * b_k, axis=1)

        p_dg = tl.make_block_ptr(dg, (T,), (HV,), (i_t * BT,), (BT,), (0,))
        # (SY 09/21) revcumsum in a separate kernel due to strange triton compiler issue
        # b_dg = tl.dot(tl.where(o_t[:, None] <= o_t[None, :], 1., 0.), b_dg, allow_tf32=False) + b_dg_last)
        b_dg = tl.where(o_t < min(i_t * BT + BT, T) - 1, b_dg, b_dg + b_dg_last)
        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0,))

    elif USE_G_GAMMA:
        if USE_EXP2:
            b_dq = b_dq * exp2(b_g)[:, None] * scale
            b_dk = b_dk * tl.where(m_t, exp2(-b_g + b_g_last), 0)[:, None]
            b_ds = tl.where(m_A, b_ds * exp2(b_g[:, None] - b_g[None, :]), 0) * scale
        else:
            b_dq = b_dq * exp(b_g)[:, None] * scale
            b_dk = b_dk * tl.where(m_t, exp(-b_g + b_g_last), 0)[:, None]
            b_ds = tl.where(m_A, b_ds * exp(b_g[:, None] - b_g[None, :]), 0) * scale
        b_ds = b_ds.to(b_k.dtype)
        # [BT, BK]
        b_dq += tl.dot(b_ds, b_k)
        b_dk += tl.dot(tl.trans(b_ds), b_q)
        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))

    else:
        b_ds = tl.where(m_A, b_ds, 0)
        b_ds = b_ds.to(b_k.dtype)
        b_dq += tl.dot(b_ds, b_k)
        b_dk += tl.dot(tl.trans(b_ds), b_q) * scale
        b_dq *= scale
        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))



def make_inputs(batch=1, length=64, heads=2):
    """CPU versions of the benchmark's inputs and naive h/dh construction."""
    torch.manual_seed(42)
    shape = (batch, length, heads, 128)
    q = torch.randn(shape, dtype=torch.bfloat16)
    k = torch.randn(shape, dtype=torch.bfloat16)
    v = torch.randn(shape, dtype=torch.bfloat16)
    g = F.logsigmoid(torch.randn((batch, length, heads), dtype=torch.bfloat16))
    do = torch.randn(shape, dtype=torch.bfloat16)
    h = naive_chunk_fwd_h(k=k, v=v, g=g, chunk_size=64)
    dh = naive_chunk_bwd_dh(q=q, k=k, v=v, do=do, g=g,
                           scale=128 ** -0.5, chunk_size=64)
    return dict(q=q, k=k, v=v, g=g, h=h, do=do, dh=dh)


def reference(inputs):
    """Original chunk-level correctness oracle; dg remains FP32."""
    return naive_chunk_bwd_dqkwg(**inputs, scale=128 ** -0.5, chunk_size=64)


def launch(inputs, outputs, compile_flow='npuir'):
    batch, length, heads, _ = inputs['q'].shape
    dq, dk, dg = outputs
    chunk_bwd_kernel_dqkwg[(1, length // 64, batch * heads)](
        **inputs, g_gamma=None, dq=dq, dk=dk, dw=None, dv=None, dg=dg,
        cu_seqlens=None, chunk_indices=None, scale=128 ** -0.5,
        B=batch, T=length, H=heads, HV=heads, K=128, V=128,
        BT=64, BK=128, BV=128, USE_G=True, USE_G_GAMMA=False,
        USE_EXP2=False, USE_DW=False, TRANSPOSE_STATE=False, IS_VARLEN=False,
        num_warps=2, num_stages=2, debug=True, sanitize_overflow=True,
        compile_flow=compile_flow, sync_solver=True, multibuffer=True,
        limit_auto_multi_buffer_of_local_buffer='no-limit',
        enable_auto_bind_sub_block=True, vf_merge_level=1,
    )


def compare_outputs(actual, expected):
    """Strict normal FLA RMS criterion, without its warning/CI exceptions."""
    for name, got, wanted in zip(('dq', 'dk', 'dg'), actual, expected):
        got, wanted = got.detach().cpu().float(), wanted.detach().cpu().float()
        if got.shape != wanted.shape:
            raise AssertionError(f'{name}: shape {got.shape} != {wanted.shape}')
        if not torch.isfinite(got).all() or not torch.isfinite(wanted).all():
            raise AssertionError(f'{name}: non-finite output/reference')
        diff = got - wanted
        max_error = diff.abs().max().item()
        ratio = diff.square().mean().sqrt().item() / (
            wanted.square().mean().sqrt().item() + 1e-8)
        if max_error > 1e-6 and ratio >= 0.02:
            raise AssertionError(f'{name}: relative RMS {ratio:.8g} >= 0.02; '
                                 f'max absolute error={max_error:.8g}')
        print(f'PASS {name}: relative RMS={ratio:.8g}, '
              f'max absolute error={max_error:.8g}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--length', type=int, default=64)
    parser.add_argument('--heads', type=int, default=2)
    parser.add_argument('--compile-flow', choices=('npuir', 'ptoas'), default='npuir')
    args = parser.parse_args()
    if min(args.batch, args.length, args.heads) < 1 or args.length % 64:
        parser.error('batch/heads must be positive and length a positive multiple of 64')

    os.environ['TRITON_ASCEND_COMPILE_FLOW'] = args.compile_flow
    if args.compile_flow == 'npuir':
        os.environ.pop('BISHENGIR_ENABLE_PTOAS_BRIDGE', None)
        os.environ.pop('BISHENGIR_PTOAS_BRIDGE_MODE', None)
    else:
        os.environ['BISHENGIR_ENABLE_PTOAS_BRIDGE'] = '1'
        os.environ['BISHENGIR_PTOAS_BRIDGE_MODE'] = 'ptodsl'
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'
    os.environ['TRITON_ALL_BLOCKS_PARALLEL'] = '1'

    import torch_npu
    if not torch.npu.is_available():
        raise RuntimeError('Run on an NPU or under the configured CANN msprof op simulator')

    cpu_inputs = make_inputs(args.batch, args.length, args.heads)
    expected = reference(cpu_inputs)
    inputs = {name: value.npu() for name, value in cpu_inputs.items()}
    dq = torch.full_like(expected[0], float('nan')).npu()
    dk = torch.full_like(expected[1], float('nan')).npu()
    # The raw kernel writes one dg partial per K tile; NK=1 for K=BK=128.
    dg = torch.full((1, args.batch, args.length, args.heads), float('nan'),
                    dtype=torch.float32).npu()
    print(f'flow={args.compile_flow}, B={args.batch}, T={args.length}, '
          f'H=HV={args.heads}, K=V=BK=BV=128, BT=64, USE_G=True, '
          'USE_EXP2=False, num_warps=2, num_stages=2', flush=True)
    launch(inputs, (dq, dk, dg), args.compile_flow)
    torch.npu.synchronize()
    compare_outputs((dq, dk, dg[0]), expected)
    print('PASS: all dq/dk/dg entries compared', flush=True)


if __name__ == '__main__':
    main()
