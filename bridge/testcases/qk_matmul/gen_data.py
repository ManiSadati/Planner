#!/usr/bin/env python3
import numpy as np

H_Q = 8
SQ = 64
SK = 64
HEAD_DIM = 64

q = np.broadcast_to(
    np.eye(SQ, HEAD_DIM, dtype=np.float16),
    (H_Q, SQ, HEAD_DIM),
).copy()
k_row = np.arange(SK, dtype=np.int32)[:, None]
k_col = np.arange(HEAD_DIM, dtype=np.int32)[None, :]
k = (((3 * k_row + k_col) % 17) - 8).astype(np.float16) / np.float16(8)

q_f32 = q.astype(np.float32)
k_f32 = k.astype(np.float32)
golden = np.matmul(q_f32, k_f32.transpose(1, 0)).astype(np.float16)

q.tofile("input_q.bin")
k.tofile("input_k.bin")
golden.tofile("golden.bin")

print(
    "[INFO] generated identity Q [8,64,64], non-symmetric K [64,64], "
    "and scores [8,64,64]"
)
