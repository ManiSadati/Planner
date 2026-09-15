#!/usr/bin/env python3

import numpy as np


M = 64
K = 64
N = 64

input_a = np.ones((M, K), dtype=np.float16)
input_b = np.ones((K, N), dtype=np.float16)
bias = ((np.arange(M * N).reshape(M, N) % 16) * 0.25).astype(np.float16)
matmul = (input_a.astype(np.float32) @ input_b.astype(np.float32)).astype(np.float16)
golden = (matmul * bias + bias).astype(np.float16)

input_a.tofile("input_a.bin")
input_b.tofile("input_b.bin")
bias.tofile("bias.bin")
golden.tofile("golden.bin")

print("[INFO] generated 64x64 FP16 A, B, bias, and golden output")
