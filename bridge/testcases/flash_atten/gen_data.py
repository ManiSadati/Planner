#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import numpy as np

H_Q = 4
H_KV = 2
SQ = 256
SK = 256
HEAD_DIM = 64
SM_SCALE = HEAD_DIM ** -0.5

np.random.seed(20260903)
q = np.random.normal(0.0, 0.5, size=(H_Q, SQ, HEAD_DIM)).astype(np.float16)
k = np.random.normal(0.0, 0.5, size=(H_KV, SK, HEAD_DIM)).astype(np.float16)
v = np.random.normal(0.0, 0.5, size=(H_KV, SK, HEAD_DIM)).astype(np.float16)

kv_heads = np.arange(H_Q) // (H_Q // H_KV)
scores = np.matmul(q.astype(np.float32), np.swapaxes(k[kv_heads].astype(np.float32), -1, -2))
scores *= np.float32(SM_SCALE)
causal_mask = np.triu(np.ones((SQ, SK), dtype=bool), k=1)
scores = np.where(causal_mask[None, :, :], -np.inf, scores)
scores -= np.max(scores, axis=-1, keepdims=True)
probs = np.exp(scores)
probs /= np.sum(probs, axis=-1, keepdims=True)
golden = np.matmul(probs.astype(np.float32), v[kv_heads].astype(np.float32)).astype(np.float16)

q.tofile("input_q.bin")
k.tofile("input_k.bin")
v.tofile("input_v.bin")
golden.tofile("golden.bin")

print(
    "[INFO] generated Q [4,256,64], K/V [2,256,64], "
    "and output [4,256,64] FP16 tensors"
)
