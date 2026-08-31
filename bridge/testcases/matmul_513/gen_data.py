#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import numpy as np

M = 513
K = 513
N = 513

input_a = np.ones((M, K), dtype=np.float16)
input_b = np.ones((K, N), dtype=np.float16)
golden = np.full((M, N), K, dtype=np.float16)

input_a.tofile("input_a.bin")
input_b.tofile("input_b.bin")
golden.tofile("golden.bin")

print(f"[INFO] generated 513x513 FP16 A, B, and matmul golden ({M * N} outputs)")
