#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import numpy as np

M = 64
K = 64
N = 64

rng = np.random.default_rng(0)
input_a = rng.standard_normal((M, K)).astype(np.float16)
input_b = rng.standard_normal((K, N)).astype(np.float16)

# tl.dot on f16 inputs accumulates in f32; the Triton kernel stores to an f16 C.
golden = (input_a.astype(np.float32) @ input_b.astype(np.float32)).astype(np.float16)

input_a.tofile("input_a.bin")
input_b.tofile("input_b.bin")
golden.tofile("golden.bin")

print(f"[INFO] generated 64x64 FP16 A, B, and matmul golden ({M * N} outputs)")
