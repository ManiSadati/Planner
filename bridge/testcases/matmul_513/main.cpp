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

void LaunchMatmul513Kernel(void *workspace0, void *workspace1, void *inputA,
                           void *inputB, void *output, void *stream);

namespace {

constexpr size_t kRows = 513;
constexpr size_t kCols = 513;
constexpr size_t kElemCount = kRows * kCols;
constexpr size_t kDataBytes = kElemCount * sizeof(uint16_t);
constexpr size_t kWorkspaceBytes = 1;

bool CheckAcl(aclError err, const char *what) {
    if (err == ACL_SUCCESS) {
        return true;
    }
    std::fprintf(stderr, "[ERROR] %s failed: %d\n", what,
                 static_cast<int>(err));
    return false;
}

} // namespace

int main() {
    int rc = 0;
    int deviceId = 0;
    if (const char *envDevice = std::getenv("ACL_DEVICE_ID")) {
        deviceId = std::atoi(envDevice);
    }

    uint16_t *inputAHost = nullptr;
    uint16_t *inputBHost = nullptr;
    uint16_t *outputHost = nullptr;
    uint16_t *inputADevice = nullptr;
    uint16_t *inputBDevice = nullptr;
    uint16_t *outputDevice = nullptr;
    uint8_t *workspace0Device = nullptr;
    uint8_t *workspace1Device = nullptr;
    aclrtStream stream = nullptr;

    if (!CheckAcl(aclInit(nullptr), "aclInit") ||
        !CheckAcl(aclrtSetDevice(deviceId), "aclrtSetDevice") ||
        !CheckAcl(aclrtCreateStream(&stream), "aclrtCreateStream") ||
        !CheckAcl(aclrtMallocHost(reinterpret_cast<void **>(&inputAHost),
                                  kDataBytes),
                  "aclrtMallocHost(inputA)") ||
        !CheckAcl(aclrtMallocHost(reinterpret_cast<void **>(&inputBHost),
                                  kDataBytes),
                  "aclrtMallocHost(inputB)") ||
        !CheckAcl(aclrtMallocHost(reinterpret_cast<void **>(&outputHost),
                                  kDataBytes),
                  "aclrtMallocHost(output)") ||
        !CheckAcl(aclrtMalloc(reinterpret_cast<void **>(&inputADevice),
                              kDataBytes, ACL_MEM_MALLOC_HUGE_FIRST),
                  "aclrtMalloc(inputA)") ||
        !CheckAcl(aclrtMalloc(reinterpret_cast<void **>(&inputBDevice),
                              kDataBytes, ACL_MEM_MALLOC_HUGE_FIRST),
                  "aclrtMalloc(inputB)") ||
        !CheckAcl(aclrtMalloc(reinterpret_cast<void **>(&outputDevice),
                              kDataBytes, ACL_MEM_MALLOC_HUGE_FIRST),
                  "aclrtMalloc(output)") ||
        !CheckAcl(aclrtMalloc(reinterpret_cast<void **>(&workspace0Device),
                              kWorkspaceBytes, ACL_MEM_MALLOC_HUGE_FIRST),
                  "aclrtMalloc(workspace0)") ||
        !CheckAcl(aclrtMalloc(reinterpret_cast<void **>(&workspace1Device),
                              kWorkspaceBytes, ACL_MEM_MALLOC_HUGE_FIRST),
                  "aclrtMalloc(workspace1)")) {
        rc = 1;
        goto cleanup;
    }

    {
        size_t inputASize = kDataBytes;
        size_t inputBSize = kDataBytes;
        if (!ReadFile("input_a.bin", inputASize, inputAHost, kDataBytes) ||
            inputASize != kDataBytes ||
            !ReadFile("input_b.bin", inputBSize, inputBHost, kDataBytes) ||
            inputBSize != kDataBytes) {
            std::fprintf(stderr, "[ERROR] failed to read FP16 matrix inputs\n");
            rc = 1;
            goto cleanup;
        }
    }

    if (!CheckAcl(aclrtMemcpy(inputADevice, kDataBytes, inputAHost, kDataBytes,
                              ACL_MEMCPY_HOST_TO_DEVICE),
                  "aclrtMemcpy(inputA H2D)") ||
        !CheckAcl(aclrtMemcpy(inputBDevice, kDataBytes, inputBHost, kDataBytes,
                              ACL_MEMCPY_HOST_TO_DEVICE),
                  "aclrtMemcpy(inputB H2D)")) {
        rc = 1;
        goto cleanup;
    }

    LaunchMatmul513Kernel(workspace0Device, workspace1Device, inputADevice,
                          inputBDevice, outputDevice, stream);

    if (!CheckAcl(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream") ||
        !CheckAcl(aclrtMemcpy(outputHost, kDataBytes, outputDevice, kDataBytes,
                              ACL_MEMCPY_DEVICE_TO_HOST),
                  "aclrtMemcpy(output D2H)")) {
        rc = 1;
        goto cleanup;
    }

    if (!WriteFile("output.bin", outputHost, kDataBytes)) {
        std::fprintf(stderr, "[ERROR] failed to write output.bin\n");
        rc = 1;
        goto cleanup;
    }

    std::printf("[INFO] wrote output.bin (%zu FP16 elements)\n", kElemCount);

cleanup:
    if (workspace0Device != nullptr) aclrtFree(workspace0Device);
    if (workspace1Device != nullptr) aclrtFree(workspace1Device);
    if (inputADevice != nullptr) aclrtFree(inputADevice);
    if (inputBDevice != nullptr) aclrtFree(inputBDevice);
    if (outputDevice != nullptr) aclrtFree(outputDevice);
    if (inputAHost != nullptr) aclrtFreeHost(inputAHost);
    if (inputBHost != nullptr) aclrtFreeHost(inputBHost);
    if (outputHost != nullptr) aclrtFreeHost(outputHost);
    if (stream != nullptr) aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return rc;
}
