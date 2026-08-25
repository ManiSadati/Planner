#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

import numpy as np

ROWS = 1600
COLS = 1600
EPS = np.float32(1.0e-5)
ELEM_COUNT = ROWS * COLS

np.random.seed(20260825)
input_data = np.random.normal(0.0, 1.0, size=(ROWS, COLS)).astype(np.float32)
weight = np.random.normal(0.0, 1.0, size=(COLS,)).astype(np.float32)

mean_square = np.mean(input_data * input_data, axis=1, keepdims=True).astype(np.float32)
inv_rms = (np.float32(1.0) / np.sqrt(mean_square + EPS)).astype(np.float32)
golden = (input_data * inv_rms * weight.reshape(1, COLS)).astype(np.float32)

input_data.tofile("input.bin")
weight.tofile("weight.bin")
golden.tofile("golden.bin")

print(
    f"[INFO] generated input.bin weight.bin golden.bin "
    f"({ELEM_COUNT} input/output float32 elements, {COLS} weight elements)"
)
