# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Standalone original solve_tril_mxr.py 64x64 merge kernel.

Source: Q2TritonKernel/src/kernels/fla/ops/utils/solve_tril_mxr.py:135.
Benchmark: Q2TritonKernel/tests/fla_perf/test_solve_tril_mxr_perf.py.
The complete kernel body and do_not_specialize=['T'] are preserved.
Host FLA imports/input_guard/IS_VARLEN heuristic are replaced by explicit args.
Default B=1,T=1024,H=32,BT=64,FP32 matches the fixed-length capture recipe;
num_warps=2 is the capture harness choice, not a measured autotuner winner.
Use --sequence-lengths for a packed varlen specialization, or --length 64
--heads 2 for a smaller simulator run. Native NPU-IR is the default.
Inputs/reference/zero output initialization run on CPU before transfer.
A fresh AVE dump is required to reconfirm predicate UB stores/reloads.
"""

import argparse
import os

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit(do_not_specialize=['T'])
def merge_16x16_to_64x64_inverse_kernel(
    A,
    Ai,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    USE_TMA: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2), tl.load(chunk_indices + i_t * 2 + 1)
        bos, eos = tl.load(cu_seqlens + i_n), tl.load(cu_seqlens + i_n + 1)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    o_i = tl.arange(0, 16)
    o_j = tl.arange(0, 16)
    m_A = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]
    A += (bos * H + i_h) * BT
    Ai += (bos * H + i_h) * BT

    r_off_11 = i_t * BT
    r_off_22 = i_t * BT + 16
    r_off_33 = i_t * BT + 32
    r_off_44 = i_t * BT + 48
    m_diag = (r_off_11 + o_i[:, None] < T) & (o_j[None, :] < BT)
    m_diag_22 = (r_off_22 + o_i[:, None] < T) & (16 + o_j[None, :] < BT)
    m_diag_33 = (r_off_33 + o_i[:, None] < T) & (32 + o_j[None, :] < BT)
    m_diag_44 = (r_off_44 + o_i[:, None] < T) & (48 + o_j[None, :] < BT)
    b_Ai_11 = tl.load(A + (r_off_11 + o_i[:, None]) * H * BT + o_j[None, :], mask=m_diag, other=0.0).to(tl.float32)
    b_Ai_22 = tl.load(A + (r_off_22 + o_i[:, None]) * H * BT + 16 + o_j[None, :], mask=m_diag_22, other=0.0).to(tl.float32)
    b_Ai_33 = tl.load(A + (r_off_33 + o_i[:, None]) * H * BT + 32 + o_j[None, :], mask=m_diag_33, other=0.0).to(tl.float32)
    b_Ai_44 = tl.load(A + (r_off_44 + o_i[:, None]) * H * BT + 48 + o_j[None, :], mask=m_diag_44, other=0.0).to(tl.float32)

    # [16, 16]
    b_Ai_11 = -tl.where(m_A, b_Ai_11, 0)
    b_Ai_22 = -tl.where(m_A, b_Ai_22, 0)
    b_Ai_33 = -tl.where(m_A, b_Ai_33, 0)
    b_Ai_44 = -tl.where(m_A, b_Ai_44, 0)

    # tri_inv_mch for diagonal block 11
    # X = I - A = I + b_Ai_11,  Y = A@A = b_Ai_11 @ b_Ai_11
    # 3 iterations: X += X@Y; Y = Y@Y

    remaining_rows = T - i_t * BT
    f_I = m_I.to(tl.float32)
    if (not IS_VARLEN) or remaining_rows >= 16:
        X_11 = f_I + b_Ai_11
        Y_11 = tl.dot(b_Ai_11, b_Ai_11, input_precision=DOT_PRECISION)
        X_11 = X_11 + tl.dot(X_11, Y_11, input_precision=DOT_PRECISION)
        Y_11 = tl.dot(Y_11, Y_11, input_precision=DOT_PRECISION)
        X_11 = X_11 + tl.dot(X_11, Y_11, input_precision=DOT_PRECISION)
        Y_11 = tl.dot(Y_11, Y_11, input_precision=DOT_PRECISION)
        X_11 = X_11 + tl.dot(X_11, Y_11, input_precision=DOT_PRECISION)
        b_Ai_11 = X_11
    else:
        for i in range(2, min(16, T - i_t * BT)):
            b_a_11 = -tl.load(A + (i_t * BT + i) * H*BT + o_i)
            b_a_11 = tl.where(o_i < i, b_a_11, 0.)
            b_a_11 += tl.sum(b_a_11[:, None] * b_Ai_11, 0)
            b_Ai_11 = tl.where((o_i == i)[:, None], b_a_11, b_Ai_11)
        b_Ai_11 += m_I

    if (not IS_VARLEN) or remaining_rows >= 32:
        # tri_inv_mch for diagonal block 22
        X_22 = f_I + b_Ai_22
        Y_22 = tl.dot(b_Ai_22, b_Ai_22, input_precision=DOT_PRECISION)
        X_22 = X_22 + tl.dot(X_22, Y_22, input_precision=DOT_PRECISION)
        Y_22 = tl.dot(Y_22, Y_22, input_precision=DOT_PRECISION)
        X_22 = X_22 + tl.dot(X_22, Y_22, input_precision=DOT_PRECISION)
        Y_22 = tl.dot(Y_22, Y_22, input_precision=DOT_PRECISION)
        X_22 = X_22 + tl.dot(X_22, Y_22, input_precision=DOT_PRECISION)
        b_Ai_22 = X_22
    else:
        for i in range(16 + 2, min(32, T - i_t * BT)):
            b_a_22 = -tl.load(A + (i_t * BT + i) * H*BT + o_i + 16)
            b_a_22 = tl.where(o_i < i - 16, b_a_22, 0.)
            b_a_22 += tl.sum(b_a_22[:, None] * b_Ai_22, 0)
            b_Ai_22 = tl.where((o_i == i - 16)[:, None], b_a_22, b_Ai_22)
        b_Ai_22 += m_I

    if (not IS_VARLEN) or remaining_rows >= 48:
        # tri_inv_mch for diagonal block 33
        X_33 = f_I + b_Ai_33
        Y_33 = tl.dot(b_Ai_33, b_Ai_33, input_precision=DOT_PRECISION)
        X_33 = X_33 + tl.dot(X_33, Y_33, input_precision=DOT_PRECISION)
        Y_33 = tl.dot(Y_33, Y_33, input_precision=DOT_PRECISION)
        X_33 = X_33 + tl.dot(X_33, Y_33, input_precision=DOT_PRECISION)
        Y_33 = tl.dot(Y_33, Y_33, input_precision=DOT_PRECISION)
        X_33 = X_33 + tl.dot(X_33, Y_33, input_precision=DOT_PRECISION)
        b_Ai_33 = X_33
    else:
        for i in range(32 + 2, min(48, T - i_t * BT)):
            b_a_33 = -tl.load(A + (i_t * BT + i) * H*BT + o_i + 32)
            b_a_33 = tl.where(o_i < i - 32, b_a_33, 0.)
            b_a_33 += tl.sum(b_a_33[:, None] * b_Ai_33, 0)
            b_Ai_33 = tl.where((o_i == i - 32)[:, None], b_a_33, b_Ai_33)
        b_Ai_33 += m_I

    if (not IS_VARLEN) or remaining_rows >= 64:
        # tri_inv_mch for diagonal block 44
        X_44 = f_I + b_Ai_44
        Y_44 = tl.dot(b_Ai_44, b_Ai_44, input_precision=DOT_PRECISION)
        X_44 = X_44 + tl.dot(X_44, Y_44, input_precision=DOT_PRECISION)
        Y_44 = tl.dot(Y_44, Y_44, input_precision=DOT_PRECISION)
        X_44 = X_44 + tl.dot(X_44, Y_44, input_precision=DOT_PRECISION)
        Y_44 = tl.dot(Y_44, Y_44, input_precision=DOT_PRECISION)
        X_44 = X_44 + tl.dot(X_44, Y_44, input_precision=DOT_PRECISION)
        b_Ai_44 = X_44
    else:
        for i in range(48 + 2, min(64, T - i_t * BT)):
            b_a_44 = -tl.load(A + (i_t * BT + i) * H*BT + o_i + 48)
            b_a_44 = tl.where(o_i < i - 48, b_a_44, 0.)
            b_a_44 += tl.sum(b_a_44[:, None] * b_Ai_44, 0)
            b_Ai_44 = tl.where((o_i == i - 48)[:, None], b_a_44, b_Ai_44)
        b_Ai_44 += m_I


    m_off_21 = (i_t * BT + 16 + o_i[:, None] < T) & (o_j[None, :] < BT)
    m_off_31 = (i_t * BT + 32 + o_i[:, None] < T) & (o_j[None, :] < BT)
    m_off_32 = (i_t * BT + 32 + o_i[:, None] < T) & (16 + o_j[None, :] < BT)
    m_off_41 = (i_t * BT + 48 + o_i[:, None] < T) & (o_j[None, :] < BT)
    m_off_42 = (i_t * BT + 48 + o_i[:, None] < T) & (16 + o_j[None, :] < BT)
    m_off_43 = (i_t * BT + 48 + o_i[:, None] < T) & (32 + o_j[None, :] < BT)
    b_A_21 = tl.load(A + (i_t * BT + 16 + o_i[:, None]) * H * BT + o_j[None, :], mask=m_off_21, other=0.0).to(tl.float32)
    b_A_31 = tl.load(A + (i_t * BT + 32 + o_i[:, None]) * H * BT + o_j[None, :], mask=m_off_31, other=0.0).to(tl.float32)
    b_A_32 = tl.load(A + (i_t * BT + 32 + o_i[:, None]) * H * BT + 16 + o_j[None, :], mask=m_off_32, other=0.0).to(tl.float32)
    b_A_41 = tl.load(A + (i_t * BT + 48 + o_i[:, None]) * H * BT + o_j[None, :], mask=m_off_41, other=0.0).to(tl.float32)
    b_A_42 = tl.load(A + (i_t * BT + 48 + o_i[:, None]) * H * BT + 16 + o_j[None, :], mask=m_off_42, other=0.0).to(tl.float32)
    b_A_43 = tl.load(A + (i_t * BT + 48 + o_i[:, None]) * H * BT + 32 + o_j[None, :], mask=m_off_43, other=0.0).to(tl.float32)

    # MBH off-diagonal: recursive 2x2 block inversion
    # Level 2: 16x16 -> 32x32 pairs — X21 = -X22@A21@X11, X43 = -X44@A43@X33
    b_Ai_21 = -tl.dot(tl.dot(b_Ai_22, b_A_21, input_precision=DOT_PRECISION), b_Ai_11, input_precision=DOT_PRECISION)
    b_Ai_43 = -tl.dot(tl.dot(b_Ai_44, b_A_43, input_precision=DOT_PRECISION), b_Ai_33, input_precision=DOT_PRECISION)

    # Level 1: 32x32 -> 64x64 cross terms — intermediates B21 @ X_UL
    P00 = tl.dot(b_A_31, b_Ai_11, input_precision=DOT_PRECISION) + tl.dot(b_A_32, b_Ai_21, input_precision=DOT_PRECISION)
    P01 = tl.dot(b_A_32, b_Ai_22, input_precision=DOT_PRECISION)
    P10 = tl.dot(b_A_41, b_Ai_11, input_precision=DOT_PRECISION) + tl.dot(b_A_42, b_Ai_21, input_precision=DOT_PRECISION)
    P11 = tl.dot(b_A_42, b_Ai_22, input_precision=DOT_PRECISION)

    b_Ai_31 = -tl.dot(b_Ai_33, P00, input_precision=DOT_PRECISION)
    b_Ai_32 = -tl.dot(b_Ai_33, P01, input_precision=DOT_PRECISION)
    b_Ai_41 = -tl.dot(b_Ai_43, P00, input_precision=DOT_PRECISION) - tl.dot(b_Ai_44, P10, input_precision=DOT_PRECISION)
    b_Ai_42 = -tl.dot(b_Ai_43, P01, input_precision=DOT_PRECISION) - tl.dot(b_Ai_44, P11, input_precision=DOT_PRECISION)

    Ai_dt = Ai.dtype.element_ty
    m_s11 = (i_t * BT + o_i[:, None] < T) & (o_j[None, :] < BT)
    m_s22 = (i_t * BT + 16 + o_i[:, None] < T) & (16 + o_j[None, :] < BT)
    m_s33 = (i_t * BT + 32 + o_i[:, None] < T) & (32 + o_j[None, :] < BT)
    m_s44 = (i_t * BT + 48 + o_i[:, None] < T) & (48 + o_j[None, :] < BT)
    m_s21 = (i_t * BT + 16 + o_i[:, None] < T) & (o_j[None, :] < BT)
    m_s31 = (i_t * BT + 32 + o_i[:, None] < T) & (o_j[None, :] < BT)
    m_s32 = (i_t * BT + 32 + o_i[:, None] < T) & (16 + o_j[None, :] < BT)
    m_s41 = (i_t * BT + 48 + o_i[:, None] < T) & (o_j[None, :] < BT)
    m_s42 = (i_t * BT + 48 + o_i[:, None] < T) & (16 + o_j[None, :] < BT)
    m_s43 = (i_t * BT + 48 + o_i[:, None] < T) & (32 + o_j[None, :] < BT)
    tl.store(Ai + (i_t * BT + o_i[:, None]) * H * BT + o_j[None, :], b_Ai_11.to(Ai_dt, fp_downcast_rounding="rtne"), mask=m_s11)
    tl.store(Ai + (i_t * BT + 16 + o_i[:, None]) * H * BT + 16 + o_j[None, :], b_Ai_22.to(Ai_dt, fp_downcast_rounding="rtne"), mask=m_s22)
    tl.store(Ai + (i_t * BT + 32 + o_i[:, None]) * H * BT + 32 + o_j[None, :], b_Ai_33.to(Ai_dt, fp_downcast_rounding="rtne"), mask=m_s33)
    tl.store(Ai + (i_t * BT + 48 + o_i[:, None]) * H * BT + 48 + o_j[None, :], b_Ai_44.to(Ai_dt, fp_downcast_rounding="rtne"), mask=m_s44)
    tl.store(Ai + (i_t * BT + 16 + o_i[:, None]) * H * BT + o_j[None, :], b_Ai_21.to(Ai_dt, fp_downcast_rounding="rtne"), mask=m_s21)
    tl.store(Ai + (i_t * BT + 32 + o_i[:, None]) * H * BT + o_j[None, :], b_Ai_31.to(Ai_dt, fp_downcast_rounding="rtne"), mask=m_s31)
    tl.store(Ai + (i_t * BT + 32 + o_i[:, None]) * H * BT + 16 + o_j[None, :], b_Ai_32.to(Ai_dt, fp_downcast_rounding="rtne"), mask=m_s32)
    tl.store(Ai + (i_t * BT + 48 + o_i[:, None]) * H * BT + o_j[None, :], b_Ai_41.to(Ai_dt, fp_downcast_rounding="rtne"), mask=m_s41)
    tl.store(Ai + (i_t * BT + 48 + o_i[:, None]) * H * BT + 16 + o_j[None, :], b_Ai_42.to(Ai_dt, fp_downcast_rounding="rtne"), mask=m_s42)
    tl.store(Ai + (i_t * BT + 48 + o_i[:, None]) * H * BT + 32 + o_j[None, :], b_Ai_43.to(Ai_dt, fp_downcast_rounding="rtne"), mask=m_s43)


def make_inputs(batch, length, heads, sequence_lengths=None):
    """Normalized-key Gram matrices, as in the original solve benchmark."""
    torch.manual_seed(42)
    lengths = sequence_lengths if sequence_lengths else [length] * batch
    A = torch.zeros((batch, length, heads, 64), dtype=torch.float32)
    chunks = []
    bos = 0
    for seq, seq_length in enumerate(lengths):
        b = 0 if sequence_lengths else seq
        begin = bos if sequence_lengths else 0
        for chunk in range(triton.cdiv(seq_length, 64)):
            start = begin + chunk * 64
            count = min(64, seq_length - chunk * 64)
            keys = F.normalize(torch.randn(heads, count, 64), dim=-1)
            lower = (keys @ keys.transpose(-1, -2)).tril(-1)
            A[b, start:start + count, :, :count] = lower.permute(1, 0, 2)
            chunks.append((seq, chunk, b, start, count))
        bos += seq_length
    return A, chunks


def reference(A, chunks):
    """Independently solve (I + A) X = I in CPU float64 per chunk/head."""
    result = torch.zeros_like(A)
    for _, _, b, start, count in chunks:
        lower = A[b, start:start + count, :, :count].permute(1, 0, 2).double()
        eye = torch.eye(count, dtype=torch.float64).expand(A.shape[2], -1, -1)
        inverse = torch.linalg.solve_triangular(eye + lower, eye, upper=False)
        result[b, start:start + count, :, :count] = inverse.permute(1, 0, 2).float()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--length', type=int, default=1024)
    parser.add_argument('--heads', type=int, default=32)
    parser.add_argument('--sequence-lengths', type=int, nargs='+',
                        help='Packed variable-length sequences; requires batch=1; overrides length')
    parser.add_argument('--compile-flow', choices=('npuir', 'ptoas'), default='npuir')
    args = parser.parse_args()
    if min(args.batch, args.length, args.heads) < 1:
        parser.error('batch, length and heads must be positive')
    if args.sequence_lengths:
        if args.batch != 1 or min(args.sequence_lengths) < 1:
            parser.error('packed sequences require batch=1 and positive sequence lengths')
        args.length = sum(args.sequence_lengths)
    elif args.length % 64:
        parser.error('fixed length must be a multiple of 64; use --sequence-lengths for tails')

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

    A_cpu, chunks = make_inputs(args.batch, args.length, args.heads, args.sequence_lengths)
    expected = reference(A_cpu, chunks)
    # The original wrapper zeros Ai: upper off-diagonal tiles are never written.
    A = A_cpu.npu()
    output = torch.zeros_like(A_cpu).npu()
    cu_seqlens = chunk_indices = None
    if args.sequence_lengths:
        cu_cpu = torch.tensor([0] + args.sequence_lengths, dtype=torch.int32).cumsum(0).to(torch.int32)
        indices_cpu = torch.tensor([(seq, chunk) for seq, chunk, _, _, _ in chunks], dtype=torch.int32)
        cu_seqlens, chunk_indices = cu_cpu.npu(), indices_cpu.npu()
    grid = (len(chunks), args.heads) if args.sequence_lengths else (
        triton.cdiv(args.length, 64), args.batch * args.heads)
    print(f'flow={args.compile_flow}, B={args.batch}, T={args.length}, H={args.heads}, '
          f'BT=64, IS_VARLEN={bool(args.sequence_lengths)}, grid={grid}, '
          'num_warps=2, num_stages=2, DOT_PRECISION=ieee', flush=True)
    merge_16x16_to_64x64_inverse_kernel[grid](
        A=A, Ai=output, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
        T=args.length, H=args.heads, BT=64, USE_TMA=False,
        IS_VARLEN=bool(args.sequence_lengths), DOT_PRECISION='ieee',
        num_warps=2, num_stages=2, debug=True, compile_flow=args.compile_flow,
        sync_solver=True, multibuffer=True,
        limit_auto_multi_buffer_of_local_buffer='no-limit',
    )
    torch.npu.synchronize()
    actual = output.cpu()
    # Explicit elementwise tolerances, distinct from FLA's assert_close metric.
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
    print(f'PASS: all output entries compared; max absolute error='
          f'{(actual - expected).abs().max().item():.8g}', flush=True)


if __name__ == '__main__':
    main()
