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

H_Q = 4
SQ = 256
HEAD_DIM = 64
ELEM_COUNT = H_Q * SQ * HEAD_DIM
ATOL = 5e-2
RTOL = 5e-2

golden = np.fromfile("golden.bin", dtype=np.float16)
output = np.fromfile("output.bin", dtype=np.float16)

if golden.size != ELEM_COUNT or output.size != ELEM_COUNT:
    print(f"[ERROR] shape mismatch: golden={golden.size}, output={output.size}")
    sys.exit(2)

golden_f32 = golden.astype(np.float32)
output_f32 = output.astype(np.float32)
abs_diff = np.abs(golden_f32 - output_f32)

if np.allclose(golden_f32, output_f32, atol=ATOL, rtol=RTOL, equal_nan=False):
    print(f"[INFO] compare passed: max abs diff={float(np.max(abs_diff))}")
    sys.exit(0)

idx = int(np.argmax(abs_diff))
head = idx // (SQ * HEAD_DIM)
head_offset = idx % (SQ * HEAD_DIM)
row = head_offset // HEAD_DIM
col = head_offset % HEAD_DIM
print(
    f"[ERROR] compare failed: max diff={float(abs_diff[idx])} at "
    f"head={head} row={row} col={col} golden={float(golden_f32[idx])} "
    f"output={float(output_f32[idx])}"
)
sys.exit(2)
