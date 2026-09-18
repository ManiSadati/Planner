#!/usr/bin/env python3

import numpy as np


SEQ_LEN = 64
HEAD_DIM = 64
SM_SCALE = HEAD_DIM**-0.5

rng = np.random.default_rng(20260917)
q = rng.normal(0.0, 0.5, size=(SEQ_LEN, HEAD_DIM)).astype(np.float16)
k = rng.normal(0.0, 0.5, size=(SEQ_LEN, HEAD_DIM)).astype(np.float16)
v = rng.normal(0.0, 0.5, size=(SEQ_LEN, HEAD_DIM)).astype(np.float16)
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

print("[INFO] generated mask-free Q/K-transpose/V/output [64,64] FP16 tensors")
