#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import sys

import numpy as np

M = 513
N = 513
ELEM_COUNT = M * N
ATOL = 0.5
RTOL = 0.0

golden = np.fromfile("golden.bin", dtype=np.float16)
output = np.fromfile("output.bin", dtype=np.float16)

if golden.size != ELEM_COUNT or output.size != ELEM_COUNT:
    print(f"[ERROR] shape mismatch: golden={golden.size}, output={output.size}")
    sys.exit(2)

if np.allclose(golden, output, atol=ATOL, rtol=RTOL, equal_nan=True):
    max_diff = float(
        np.max(np.abs(golden.astype(np.float32) - output.astype(np.float32)))
    )
    print(f"[INFO] compare passed: max abs diff={max_diff}")
    sys.exit(0)

abs_diff = np.abs(golden.astype(np.float32) - output.astype(np.float32))
idx = int(np.argmax(abs_diff))
row = idx // N
col = idx % N
print(
    f"[ERROR] compare failed: max diff={float(abs_diff[idx])} at "
    f"row={row} col={col} golden={float(golden[idx])} "
    f"output={float(output[idx])}"
)
sys.exit(2)
