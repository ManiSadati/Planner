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

ROWS = 64
COLS = 256
ELEM_COUNT = ROWS * COLS

np.random.seed(20260820)
input_data = np.random.uniform(-8.0, 8.0, size=(ROWS, COLS)).astype(np.float32)
shifted = input_data - np.max(input_data, axis=1, keepdims=True)
exp_data = np.exp(shifted).astype(np.float32)
golden = (exp_data / np.sum(exp_data, axis=1, keepdims=True)).astype(np.float32)

input_data.tofile("input.bin")
golden.tofile("golden.bin")

print(f"[INFO] generated input.bin golden.bin ({ELEM_COUNT} float32 elements)")
