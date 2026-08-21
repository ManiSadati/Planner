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

TILE_COUNT = 1000
TILE_ELEMS = 2000
ELEM_COUNT = TILE_COUNT * TILE_ELEMS

values = np.arange(ELEM_COUNT, dtype=np.float32).reshape(TILE_COUNT, TILE_ELEMS)
input0 = (values * np.float32(0.001)).astype(np.float32).reshape(-1)
input1 = (values * np.float32(-0.002)).astype(np.float32).reshape(-1)
golden = (input0 + input1).astype(np.float32)

input0.tofile("input0.bin")
input1.tofile("input1.bin")
golden.tofile("golden.bin")

print(f"[INFO] generated input0.bin input1.bin golden.bin ({ELEM_COUNT} float32 elements)")
