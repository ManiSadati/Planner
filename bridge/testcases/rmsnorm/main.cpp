// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "acl/acl.h"
#include "test_common.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>

using namespace PtoTestCommon;

void LaunchRmsNormKernel(uint8_t *workspace0, uint8_t *workspace1, float *input,
                         float *weight, float *output, void *stream);

namespace {

constexpr size_t kRows = 16;
constexpr size_t kCols = 256;
constexpr size_t kInputElems = kRows * kCols;
constexpr size_t kWeightElems = kCols;
constexpr size_t kInputBytes = kInputElems * sizeof(float);
constexpr size_t kWeightBytes = kWeightElems * sizeof(float);
constexpr size_t kWorkspaceBytes = 1;

bool CheckAcl(aclError err, const char *what) {
    if (err == ACL_SUCCESS) {
        return true;
    }
    std::fprintf(stderr, "[ERROR] %s failed: %d\n", what, static_cast<int>(err));
    return false;
}

} // namespace

int main() {
    int rc = 0;
    int deviceId = 0;
    if (const char *envDevice = std::getenv("ACL_DEVICE_ID")) {
        deviceId = std::atoi(envDevice);
    }

    float *inputHost = nullptr;
    float *weightHost = nullptr;
    float *outputHost = nullptr;
    float *inputDevice = nullptr;
    float *weightDevice = nullptr;
    float *outputDevice = nullptr;
    uint8_t *workspace0Device = nullptr;
    uint8_t *workspace1Device = nullptr;
    aclrtStream stream = nullptr;

    if (!CheckAcl(aclInit(nullptr), "aclInit") ||
        !CheckAcl(aclrtSetDevice(deviceId), "aclrtSetDevice") ||
        !CheckAcl(aclrtCreateStream(&stream), "aclrtCreateStream") ||
        !CheckAcl(aclrtMallocHost(reinterpret_cast<void **>(&inputHost), kInputBytes),
                  "aclrtMallocHost(input)") ||
        !CheckAcl(aclrtMallocHost(reinterpret_cast<void **>(&weightHost), kWeightBytes),
                  "aclrtMallocHost(weight)") ||
        !CheckAcl(aclrtMallocHost(reinterpret_cast<void **>(&outputHost), kInputBytes),
                  "aclrtMallocHost(output)") ||
        !CheckAcl(aclrtMalloc(reinterpret_cast<void **>(&inputDevice), kInputBytes,
                              ACL_MEM_MALLOC_HUGE_FIRST),
                  "aclrtMalloc(input)") ||
        !CheckAcl(aclrtMalloc(reinterpret_cast<void **>(&weightDevice), kWeightBytes,
                              ACL_MEM_MALLOC_HUGE_FIRST),
                  "aclrtMalloc(weight)") ||
        !CheckAcl(aclrtMalloc(reinterpret_cast<void **>(&outputDevice), kInputBytes,
                              ACL_MEM_MALLOC_HUGE_FIRST),
                  "aclrtMalloc(output)") ||
        !CheckAcl(aclrtMalloc(reinterpret_cast<void **>(&workspace0Device), kWorkspaceBytes,
                              ACL_MEM_MALLOC_HUGE_FIRST),
                  "aclrtMalloc(workspace0)") ||
        !CheckAcl(aclrtMalloc(reinterpret_cast<void **>(&workspace1Device), kWorkspaceBytes,
                              ACL_MEM_MALLOC_HUGE_FIRST),
                  "aclrtMalloc(workspace1)")) {
        rc = 1;
        goto cleanup;
    }

    {
        size_t inputSize = kInputBytes;
        size_t weightSize = kWeightBytes;
        if (!ReadFile("input.bin", inputSize, inputHost, kInputBytes) ||
            !ReadFile("weight.bin", weightSize, weightHost, kWeightBytes)) {
            std::fprintf(stderr, "[ERROR] failed to read input.bin or weight.bin\n");
            rc = 1;
            goto cleanup;
        }
    }

    if (!CheckAcl(aclrtMemcpy(inputDevice, kInputBytes, inputHost, kInputBytes,
                              ACL_MEMCPY_HOST_TO_DEVICE),
                  "aclrtMemcpy(input H2D)") ||
        !CheckAcl(aclrtMemcpy(weightDevice, kWeightBytes, weightHost, kWeightBytes,
                              ACL_MEMCPY_HOST_TO_DEVICE),
                  "aclrtMemcpy(weight H2D)")) {
        rc = 1;
        goto cleanup;
    }

    LaunchRmsNormKernel(workspace0Device, workspace1Device, inputDevice, weightDevice,
                        outputDevice, stream);

    if (!CheckAcl(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream") ||
        !CheckAcl(aclrtMemcpy(outputHost, kInputBytes, outputDevice, kInputBytes,
                              ACL_MEMCPY_DEVICE_TO_HOST),
                  "aclrtMemcpy(output D2H)")) {
        rc = 1;
        goto cleanup;
    }

    if (!WriteFile("output.bin", outputHost, kInputBytes)) {
        std::fprintf(stderr, "[ERROR] failed to write output.bin\n");
        rc = 1;
        goto cleanup;
    }

    std::printf("[INFO] wrote output.bin\n");

cleanup:
    if (workspace0Device != nullptr) aclrtFree(workspace0Device);
    if (workspace1Device != nullptr) aclrtFree(workspace1Device);
    if (inputDevice != nullptr) aclrtFree(inputDevice);
    if (weightDevice != nullptr) aclrtFree(weightDevice);
    if (outputDevice != nullptr) aclrtFree(outputDevice);
    if (inputHost != nullptr) aclrtFreeHost(inputHost);
    if (weightHost != nullptr) aclrtFreeHost(weightHost);
    if (outputHost != nullptr) aclrtFreeHost(outputHost);
    if (stream != nullptr) aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return rc;
}
