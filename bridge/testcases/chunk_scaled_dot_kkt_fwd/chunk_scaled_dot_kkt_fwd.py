# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors


"""Standalone KKT testcase for the original chunk-scaled-dot Triton kernel.

Original kernel: Q2TritonKernel/src/kernels/fla/ops/common/chunk_scaled_dot_kkt.py:32.
Original normal exp helper: Q2TritonKernel/src/kernels/fla/ops/utils/op.py:29.
Benchmark: Q2TritonKernel/tests/fla_perf/test_chunk_scaled_dot_kkt_fwd_perf.py.

The kernel and normal exp helper bodies are unchanged. FLA imports, wrapper,
heuristics and autotuning are replaced with explicit launch settings:
B=1, T=64, H=HV=2, K=128, BT=64, BK=32, two warps/stages, USE_G=True,
IS_VARLEN=False, BF16 inputs and output. The small T=64 simulator case differs
from the retained T=131072 capture; use --length 131072 to restore that length.
The normal FLA_USE_FAST_OPS=0 math branch is fixed explicitly by the copied
helper. This is a selected specialization, not a measured autotuner winner.

Run on an NPU or under the configured CANN simulator:
    python chunk_scaled_dot_kkt_fwd.py
    python chunk_scaled_dot_kkt_fwd.py --length 131072

Native NPU-IR is the default; --compile-flow ptoas selects the bridge.
Use TRITON_KERNEL_DUMP=1 with fresh TRITON_CACHE_DIR/TRITON_DUMP_DIR directories
to retain IR. debug=True enables backend diagnostics; no additional sanitizer
setting is forced. Check the resulting IR before claiming the predicate-memory
pattern from the older capture has been reproduced. No capture.py is used.
"""

import argparse
import os

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def exp(x): return tl.exp(x.to(tl.float32))


@triton.jit(do_not_specialize=['T'])
def chunk_scaled_dot_kkt_fwd_kernel(
    k,
    g,
    beta,
    A,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_G: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // HV, i_bh % HV
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T

    p_b = tl.make_block_ptr(beta + bos*HV + i_h, (T,), (HV,), (i_t * BT,), (BT,), (0,))
    b_b = tl.load(p_b, boundary_check=(0,))

    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(k + (bos*H + i_h // (HV // H)) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_A += tl.dot(b_k, tl.trans(b_k))

    if USE_G:
        p_g = tl.make_block_ptr(g + bos*HV + i_h, (T,), (HV,), (i_t * BT,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,))
        b_g_diff = b_g[:, None] - b_g[None, :]
        b_A *= exp(b_g_diff)
    b_A *= b_b[:, None]

    m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0)
    p_A = tl.make_block_ptr(A + (bos*HV + i_h) * BT, (T, BT), (BT*HV, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    tl.store(p_A, b_A.to(p_A.dtype.element_ty), boundary_check=(0, 1))


def make_inputs(batch, length, heads):
    """Build the benchmark-style inputs on the CPU for simulator use."""
    torch.manual_seed(42)
    k = F.normalize(torch.randn(batch, length, heads, 128), dim=-1).to(torch.bfloat16)
    g = F.logsigmoid(torch.randn(batch, length, heads, dtype=torch.bfloat16) * 2)
    g = torch.clamp(g, min=-4.0)
    beta = torch.rand(batch, length, heads, dtype=torch.bfloat16).sigmoid()
    return k, g, beta


def reference(k, g, beta):
    """CPU reference with FP32 dot accumulation and BF16 output.

    For each 64-row chunk and head:
        A[i,j] = beta[i] * dot(k[i], k[j]) * exp(g[i] - g[j]) if i > j,
                 0 otherwise.
    The BF16 gate subtraction precedes the helper's FP32 exp conversion.
    """
    batch, length, heads, _ = k.shape
    output = torch.empty((batch, length, heads, 64), dtype=torch.bfloat16)
    causal = torch.tril(torch.ones((64, 64), dtype=torch.bool), diagonal=-1)
    for start in range(0, length, 64):
        for head in range(heads):
            keys = k[:, start:start + 64, head, :].float()
            scores = keys @ keys.transpose(-1, -2)
            gates = g[:, start:start + 64, head]
            gate_difference = (gates.unsqueeze(-1) - gates.unsqueeze(-2)).float()
            scores *= torch.exp(gate_difference)
            scores *= beta[:, start:start + 64, head].float().unsqueeze(-1)
            output[:, start:start + 64, head, :] = scores.masked_fill(~causal, 0).to(torch.bfloat16)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--length', type=int, default=64)
    parser.add_argument('--heads', type=int, default=2)
    parser.add_argument('--compile-flow', choices=('npuir', 'ptoas'), default='npuir')
    args = parser.parse_args()
    if min(args.batch, args.length, args.heads) < 1:
        parser.error('batch, length and heads must be positive')
    if args.length % 64:
        parser.error('length must be a multiple of 64 for this testcase')

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

    k_cpu, g_cpu, beta_cpu = make_inputs(args.batch, args.length, args.heads)
    k, g, beta = k_cpu.npu(), g_cpu.npu(), beta_cpu.npu()
    output = torch.empty((args.batch, args.length, args.heads, 64),
                         dtype=torch.bfloat16, device='npu')
    print(f'flow={args.compile_flow}, B={args.batch}, T={args.length}, '
          f'H=HV={args.heads}, K=128, BT=64, BK=32, '
          'num_warps=2, num_stages=2, USE_G=True, IS_VARLEN=False', flush=True)

    chunk_scaled_dot_kkt_fwd_kernel[(triton.cdiv(args.length, 64), args.batch * args.heads)](
        k=k, g=g, beta=beta, A=output, cu_seqlens=None, chunk_indices=None,
        T=args.length, H=args.heads, HV=args.heads, K=128, BT=64, BK=32,
        IS_VARLEN=False, USE_G=True,
        num_warps=2, num_stages=2, debug=True, compile_flow=args.compile_flow,
        sync_solver=True, multibuffer=True,
        limit_auto_multi_buffer_of_local_buffer='no-limit',
        enable_auto_bind_sub_block=True,
    )
    torch.npu.synchronize()
    actual = output.cpu()
    expected = reference(k_cpu, g_cpu, beta_cpu)
    # Explicit elementwise tolerances for BF16 output; this is not the FLA
    # assert_close error metric used by the repository's accuracy test.
    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
    mask = torch.triu(torch.ones(64, 64, dtype=torch.bool)).repeat(args.length // 64, 1)
    upper_values = actual.permute(0, 2, 1, 3).masked_select(mask)
    if torch.count_nonzero(upper_values).item():
        raise AssertionError('Causal diagonal/upper triangle must be zero')
    max_error = (actual.float() - expected.float()).abs().max().item()
    print(f'PASS: all outputs compared; max absolute error={max_error:.8g}', flush=True)


if __name__ == '__main__':
    main()
