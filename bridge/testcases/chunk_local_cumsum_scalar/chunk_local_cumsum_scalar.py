# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors


"""Standalone copy of the original local-cumsum Triton kernel.

Source: Q2TritonKernel/src/kernels/fla/ops/utils/cumsum.py:31.
The function body and triton.jit decorator are unchanged. The FLA wrapper,
heuristics and autotuner are replaced with explicit launch settings.
Default case: B=1, T=1024, H=32, BT=64, FP16 input, FP32 output.

Run on an NPU with the configured Triton-Ascend/CANN environment:
    python chunk_local_cumsum_scalar.py
    python chunk_local_cumsum_scalar.py --reverse

Save compiler IR using fresh TRITON_CACHE_DIR / TRITON_DUMP_DIR directories
and TRITON_KERNEL_DUMP=1. debug=True also retains native compiler diagnostics.
Native NPU-IR is the default; --compile-flow ptoas selects the bridge instead.
No capture.py or imports from the FLA/Logs repositories are used.
"""

import argparse
import os

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=['T'])
def chunk_local_cumsum_scalar_kernel(
    s,
    o,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    BT: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if HEAD_FIRST:
        p_s = tl.make_block_ptr(s + bos*H + i_h*T, (T,), (1,), (i_t * BT,), (BT,), (0,))
        p_o = tl.make_block_ptr(o + bos*H + i_h*T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    else:
        p_s = tl.make_block_ptr(s + bos*H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
        p_o = tl.make_block_ptr(o + bos*H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    # [BT]
    b_s = tl.load(p_s, boundary_check=(0,)).to(tl.float32)
    b_o = tl.cumsum(b_s, axis=0)
    if REVERSE:
        b_z = tl.sum(b_s, axis=0)
        b_o = -b_o + b_z[None] + b_s
    if HAS_SCALE:
        b_o *= scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0,))


def reference(x, reverse=False):
    """CPU FP32 reference: each 64-element sequence chunk starts a new sum."""
    chunks = x.float().split(64, dim=1)
    if reverse:
        return torch.cat([chunk.flip(1).cumsum(1).flip(1) for chunk in chunks], dim=1)
    return torch.cat([chunk.cumsum(1) for chunk in chunks], dim=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--length', type=int, default=1024)
    parser.add_argument('--heads', type=int, default=32)
    parser.add_argument('--reverse', action='store_true')
    parser.add_argument('--num-warps', type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument('--compile-flow', choices=('npuir', 'ptoas'), default='npuir')
    args = parser.parse_args()
    if min(args.batch, args.length, args.heads) < 1:
        parser.error('batch, length and heads must be positive')
    if args.length % 64:
        parser.error('length must be a multiple of 64 for this benchmark reproducer')

    # Native compilation must not inherit an opt-in AVE-to-VMI bridge setting.
    os.environ['TRITON_ASCEND_COMPILE_FLOW'] = args.compile_flow
    if args.compile_flow == 'npuir':
        os.environ.pop('BISHENGIR_ENABLE_PTOAS_BRIDGE', None)
        os.environ.pop('BISHENGIR_PTOAS_BRIDGE_MODE', None)
    else:
        os.environ['BISHENGIR_ENABLE_PTOAS_BRIDGE'] = '1'
        os.environ['BISHENGIR_PTOAS_BRIDGE_MODE'] = 'ptodsl'
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'
    os.environ['TRITON_ALL_BLOCKS_PARALLEL'] = '1'

    import torch_npu  # Registers the NPU device with PyTorch.

    if not torch.npu.is_available():
        raise RuntimeError('An available NPU is required for this execution check')
    torch.manual_seed(42)
    # The simulator may execute only the selected Triton kernel, so initialize
    # input on the CPU rather than relying on an NPU random-number kernel.
    input_cpu = torch.randn((args.batch, args.length, args.heads), dtype=torch.float16)
    x = input_cpu.npu()
    result = torch.empty_like(x, dtype=torch.float32)
    print(f'flow={args.compile_flow}, shape={tuple(x.shape)}, BT=64, '
          f'reverse={args.reverse}, num_warps={args.num_warps}', flush=True)

    chunk_local_cumsum_scalar_kernel[(triton.cdiv(args.length, 64), args.batch * args.heads)](
        s=x, o=result, scale=None, cu_seqlens=None, chunk_indices=None,
        T=args.length, B=args.batch, H=args.heads, BT=64,
        REVERSE=args.reverse, HAS_SCALE=False, IS_VARLEN=False, HEAD_FIRST=False,
        num_warps=args.num_warps, num_stages=2, debug=True,
        compile_flow=args.compile_flow,
    )
    torch.npu.synchronize()
    actual = result.cpu()
    expected = reference(input_cpu, reverse=args.reverse)
    # Match the tolerance in tests/fla/test_chunk_local_cumsum.py.
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)
    print(f'PASS: max absolute error={(actual - expected).abs().max().item():.8g}')


if __name__ == '__main__':
    main()
