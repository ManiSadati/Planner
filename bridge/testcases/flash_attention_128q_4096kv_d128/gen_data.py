#!/usr/bin/env python3

import numpy as np


QUERY_LEN = 128
KV_LEN = 4096
HEAD_DIM = 128
SM_SCALE = HEAD_DIM**-0.5

rng = np.random.default_rng(20260918)
q = rng.normal(0.0, 0.5, size=(QUERY_LEN, HEAD_DIM)).astype(np.float16)
k = rng.normal(0.0, 0.5, size=(KV_LEN, HEAD_DIM)).astype(np.float16)
v = rng.normal(0.0, 0.5, size=(KV_LEN, HEAD_DIM)).astype(np.float16)
k_t = np.ascontiguousarray(k.T)

scores = np.matmul(q.astype(np.float32), k_t.astype(np.float32))
scores *= np.float32(SM_SCALE)
scores -= np.max(scores, axis=1, keepdims=True)
probabilities = np.exp(scores)
probabilities /= np.sum(probabilities, axis=1, keepdims=True)
golden = np.matmul(probabilities, v.astype(np.float32)).astype(np.float16)

q.tofile("input_q.bin")
k_t.tofile("input_k_t.bin")
v.tofile("input_v.bin")
golden.tofile("golden.bin")

print(
    "[INFO] generated FP16 Q[128,128], KT[128,4096], "
    "V[4096,128], and output[128,128]"
)
