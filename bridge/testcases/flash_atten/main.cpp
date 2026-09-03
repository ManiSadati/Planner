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

void LaunchFlashAttenKernel(void *workspace0, void *workspace1, void *q,
                            void *k, void *v, void *out, void *stream);

namespace {

constexpr size_t kHq = 4;
constexpr size_t kHkv = 2;
constexpr size_t kSeqLen = 256;
constexpr size_t kHeadDim = 64;
constexpr size_t kQElems = kHq * kSeqLen * kHeadDim;
constexpr size_t kKVElems = kHkv * kSeqLen * kHeadDim;
constexpr size_t kQBytes = kQElems * sizeof(uint16_t);
constexpr size_t kKVBytes = kKVElems * sizeof(uint16_t);
constexpr size_t kOutputBytes = kQBytes;
constexpr size_t kSyncBlockBytes = 4096;
constexpr size_t kWorkspaceBytes = 4096;

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

    uint16_t *qHost = nullptr;
    uint16_t *kHost = nullptr;
    uint16_t *vHost = nullptr;
    uint16_t *outputHost = nullptr;
    uint16_t *qDevice = nullptr;
    uint16_t *kDevice = nullptr;
    uint16_t *vDevice = nullptr;
    uint16_t *outputDevice = nullptr;
    uint8_t *workspace0Device = nullptr;
    uint8_t *workspace1Device = nullptr;
    aclrtStream stream = nullptr;

    if (!CheckAcl(aclInit(nullptr), "aclInit") ||
        !CheckAcl(aclrtSetDevice(deviceId), "aclrtSetDevice") ||
        !CheckAcl(aclrtCreateStream(&stream), "aclrtCreateStream") ||
        !CheckAcl(aclrtMallocHost(reinterpret_cast<void **>(&qHost), kQBytes),
                  "aclrtMallocHost(Q)") ||
        !CheckAcl(aclrtMallocHost(reinterpret_cast<void **>(&kHost), kKVBytes),
                  "aclrtMallocHost(K)") ||
        !CheckAcl(aclrtMallocHost(reinterpret_cast<void **>(&vHost), kKVBytes),
                  "aclrtMallocHost(V)") ||
        !CheckAcl(aclrtMallocHost(reinterpret_cast<void **>(&outputHost),
                                  kOutputBytes),
                  "aclrtMallocHost(output)") ||
        !CheckAcl(aclrtMalloc(reinterpret_cast<void **>(&qDevice), kQBytes,
                              ACL_MEM_MALLOC_HUGE_FIRST),
                  "aclrtMalloc(Q)") ||
        !CheckAcl(aclrtMalloc(reinterpret_cast<void **>(&kDevice), kKVBytes,
                              ACL_MEM_MALLOC_HUGE_FIRST),
                  "aclrtMalloc(K)") ||
        !CheckAcl(aclrtMalloc(reinterpret_cast<void **>(&vDevice), kKVBytes,
                              ACL_MEM_MALLOC_HUGE_FIRST),
                  "aclrtMalloc(V)") ||
        !CheckAcl(aclrtMalloc(reinterpret_cast<void **>(&outputDevice),
                              kOutputBytes, ACL_MEM_MALLOC_HUGE_FIRST),
                  "aclrtMalloc(output)") ||
        !CheckAcl(aclrtMalloc(reinterpret_cast<void **>(&workspace0Device),
                              kSyncBlockBytes, ACL_MEM_MALLOC_HUGE_FIRST),
                  "aclrtMalloc(workspace0)") ||
        !CheckAcl(aclrtMalloc(reinterpret_cast<void **>(&workspace1Device),
                              kWorkspaceBytes, ACL_MEM_MALLOC_HUGE_FIRST),
                  "aclrtMalloc(workspace1)")) {
        rc = 1;
        goto cleanup;
    }

    {
        size_t qSize = kQBytes;
        size_t kSize = kKVBytes;
        size_t vSize = kKVBytes;
        if (!ReadFile("input_q.bin", qSize, qHost, kQBytes) ||
            qSize != kQBytes ||
            !ReadFile("input_k.bin", kSize, kHost, kKVBytes) ||
            kSize != kKVBytes ||
            !ReadFile("input_v.bin", vSize, vHost, kKVBytes) ||
            vSize != kKVBytes) {
            std::fprintf(stderr, "[ERROR] failed to read Q/K/V inputs\n");
            rc = 1;
            goto cleanup;
        }
    }

    if (!CheckAcl(aclrtMemcpy(qDevice, kQBytes, qHost, kQBytes,
                              ACL_MEMCPY_HOST_TO_DEVICE),
                  "aclrtMemcpy(Q H2D)") ||
        !CheckAcl(aclrtMemcpy(kDevice, kKVBytes, kHost, kKVBytes,
                              ACL_MEMCPY_HOST_TO_DEVICE),
                  "aclrtMemcpy(K H2D)") ||
        !CheckAcl(aclrtMemcpy(vDevice, kKVBytes, vHost, kKVBytes,
                              ACL_MEMCPY_HOST_TO_DEVICE),
                  "aclrtMemcpy(V H2D)")) {
        rc = 1;
        goto cleanup;
    }

    LaunchFlashAttenKernel(workspace0Device, workspace1Device, qDevice, kDevice,
                           vDevice, outputDevice, stream);

    if (!CheckAcl(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream") ||
        !CheckAcl(aclrtMemcpy(outputHost, kOutputBytes, outputDevice,
                              kOutputBytes, ACL_MEMCPY_DEVICE_TO_HOST),
                  "aclrtMemcpy(output D2H)")) {
        rc = 1;
        goto cleanup;
    }

    if (!WriteFile("output.bin", outputHost, kOutputBytes)) {
        std::fprintf(stderr, "[ERROR] failed to write output.bin\n");
        rc = 1;
        goto cleanup;
    }

    std::printf("[INFO] wrote output.bin (%zu FP16 elements)\n", kQElems);

cleanup:
    if (workspace0Device != nullptr) aclrtFree(workspace0Device);
    if (workspace1Device != nullptr) aclrtFree(workspace1Device);
    if (qDevice != nullptr) aclrtFree(qDevice);
    if (kDevice != nullptr) aclrtFree(kDevice);
    if (vDevice != nullptr) aclrtFree(vDevice);
    if (outputDevice != nullptr) aclrtFree(outputDevice);
    if (qHost != nullptr) aclrtFreeHost(qHost);
    if (kHost != nullptr) aclrtFreeHost(kHost);
    if (vHost != nullptr) aclrtFreeHost(vHost);
    if (outputHost != nullptr) aclrtFreeHost(outputHost);
    if (stream != nullptr) aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return rc;
}
