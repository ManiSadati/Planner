import os
import sys
from pathlib import Path

import torch
import torch_npu
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from compile_timing import enable_compile_timing


@triton.jit
def mixed_matmul_bias_kernel(a_ptr, b_ptr, bias_ptr, output_ptr):
    offs_m = tl.arange(0, 64)
    offs_n = tl.arange(0, 64)
    offs_k = tl.arange(0, 64)

    a = tl.load(a_ptr + offs_m[:, None] * 64 + offs_k[None, :])
    b = tl.load(b_ptr + offs_k[:, None] * 64 + offs_n[None, :])
    bias = tl.load(bias_ptr + offs_m[:, None] * 64 + offs_n[None, :])

    matmul = tl.dot(a, b).to(tl.float16)
    result = matmul * bias + bias
    tl.store(output_ptr + offs_m[:, None] * 64 + offs_n[None, :], result)


def main():
    enable_compile_timing()

    a_host = torch.ones((64, 64), dtype=torch.float16)
    b_host = torch.ones((64, 64), dtype=torch.float16)
    bias_host = (
        (torch.arange(64 * 64, dtype=torch.float16).reshape(64, 64) % 16)
        * 0.25
    )
    a = a_host.to("npu")
    b = b_host.to("npu")
    bias = bias_host.to("npu")
    output = torch.empty((64, 64), device="npu", dtype=torch.float16)

    mixed_matmul_bias_kernel[(1,)](a, b, bias, output)

    matmul_reference = torch.matmul(a_host, b_host)
    reference = matmul_reference * bias_host + bias_host
    output_host = output.cpu()
    print("max error:", (output_host - reference).abs().max().item())
    print(
        "allclose:",
        torch.allclose(output_host, reference, atol=1e-2, rtol=1e-2),
    )

    # CANN 9.1 beta can fault during TorchNPU teardown after simulator success.
    if os.getenv("TRITON_SIMULATOR_CLEAN_EXIT") == "1":
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
