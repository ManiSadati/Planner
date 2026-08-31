#!/usr/bin/env python3
import numpy as np

H_Q = 32
H_KV = 2
SQ = 128
SK = 8192
HEAD_DIM = 256

q = np.ones((H_Q, SQ, HEAD_DIM), dtype=np.float16)
k = np.ones((H_KV, SK, HEAD_DIM), dtype=np.float16)
k[1].fill(2.0)

kv_heads = np.arange(H_Q) // (H_Q // H_KV)
expected_per_head = (kv_heads + 1) * HEAD_DIM
golden = np.broadcast_to(
    expected_per_head[:, None, None], (H_Q, SQ, SK)
).astype(np.float16)

q.tofile("input_q.bin")
k.tofile("input_k.bin")
golden.tofile("golden.bin")

print(
    "[INFO] generated Q [32,128,256], K [2,8192,256], "
    "and scores [32,128,8192]"
)
