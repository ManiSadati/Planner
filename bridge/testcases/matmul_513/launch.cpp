// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include <cstdint>

#ifndef AICORE
#define AICORE [aicore]
#endif

extern "C" __global__ AICORE void matmul513_kernel(
    __gm__ uint8_t *workspace0, __gm__ uint8_t *workspace1,
    __gm__ __fp16 *inputA, __gm__ __fp16 *inputB, __gm__ __fp16 *output,
    int32_t logicalBlockCount, int32_t outerCount, int32_t innerCount);

void LaunchMatmul513Kernel(void *workspace0, void *workspace1, void *inputA,
                           void *inputB, void *output, void *stream) {
    constexpr int32_t kLogicalBlockCount = 9 * 9;
    constexpr int32_t kPhysicalBlockCount = 32;
    matmul513_kernel<<<kPhysicalBlockCount, nullptr, stream>>>(
        (__gm__ uint8_t *)workspace0,
        (__gm__ uint8_t *)workspace1,
        (__gm__ __fp16 *)inputA,
        (__gm__ __fp16 *)inputB,
        (__gm__ __fp16 *)output,
        kLogicalBlockCount,
        1,
        1);
}
