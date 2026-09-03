// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.

#pragma once

#include "acl/acl.h"
#include "test_common.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#ifndef MATMUL_LAUNCH_FUNCTION
#error "MATMUL_LAUNCH_FUNCTION must name the testcase launch wrapper"
#endif
#ifndef MATMUL_INPUT_A_BYTES
#error "MATMUL_INPUT_A_BYTES must be defined"
#endif
#ifndef MATMUL_INPUT_B_BYTES
#error "MATMUL_INPUT_B_BYTES must be defined"
#endif
#ifndef MATMUL_OUTPUT_BYTES
#error "MATMUL_OUTPUT_BYTES must be defined"
#endif

using namespace PtoTestCommon;

void MATMUL_LAUNCH_FUNCTION(void *workspace0, void *workspace1, void *inputA,
                            void *inputB, void *output, void *stream);

namespace {

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

    uint8_t *inputAHost = nullptr;
    uint8_t *inputBHost = nullptr;
    uint8_t *outputHost = nullptr;
    uint8_t *inputADevice = nullptr;
    uint8_t *inputBDevice = nullptr;
    uint8_t *outputDevice = nullptr;
    uint8_t *workspace0Device = nullptr;
    uint8_t *workspace1Device = nullptr;
    aclrtStream stream = nullptr;

    if (!CheckAcl(aclInit(nullptr), "aclInit") ||
        !CheckAcl(aclrtSetDevice(deviceId), "aclrtSetDevice") ||
        !CheckAcl(aclrtCreateStream(&stream), "aclrtCreateStream") ||
        !CheckAcl(aclrtMallocHost(reinterpret_cast<void **>(&inputAHost),
                                  MATMUL_INPUT_A_BYTES),
                  "aclrtMallocHost(inputA)") ||
        !CheckAcl(aclrtMallocHost(reinterpret_cast<void **>(&inputBHost),
                                  MATMUL_INPUT_B_BYTES),
                  "aclrtMallocHost(inputB)") ||
        !CheckAcl(aclrtMallocHost(reinterpret_cast<void **>(&outputHost),
                                  MATMUL_OUTPUT_BYTES),
                  "aclrtMallocHost(output)") ||
        !CheckAcl(aclrtMalloc(reinterpret_cast<void **>(&inputADevice),
                              MATMUL_INPUT_A_BYTES,
                              ACL_MEM_MALLOC_HUGE_FIRST),
                  "aclrtMalloc(inputA)") ||
        !CheckAcl(aclrtMalloc(reinterpret_cast<void **>(&inputBDevice),
                              MATMUL_INPUT_B_BYTES,
                              ACL_MEM_MALLOC_HUGE_FIRST),
                  "aclrtMalloc(inputB)") ||
        !CheckAcl(aclrtMalloc(reinterpret_cast<void **>(&outputDevice),
                              MATMUL_OUTPUT_BYTES,
                              ACL_MEM_MALLOC_HUGE_FIRST),
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
        size_t inputASize = MATMUL_INPUT_A_BYTES;
        size_t inputBSize = MATMUL_INPUT_B_BYTES;
        if (!ReadFile("input_a.bin", inputASize, inputAHost,
                      MATMUL_INPUT_A_BYTES) ||
            inputASize != MATMUL_INPUT_A_BYTES ||
            !ReadFile("input_b.bin", inputBSize, inputBHost,
                      MATMUL_INPUT_B_BYTES) ||
            inputBSize != MATMUL_INPUT_B_BYTES) {
            std::fprintf(stderr, "[ERROR] failed to read matrix inputs\n");
            rc = 1;
            goto cleanup;
        }
    }

    if (!CheckAcl(aclrtMemcpy(inputADevice, MATMUL_INPUT_A_BYTES, inputAHost,
                              MATMUL_INPUT_A_BYTES,
                              ACL_MEMCPY_HOST_TO_DEVICE),
                  "aclrtMemcpy(inputA H2D)") ||
        !CheckAcl(aclrtMemcpy(inputBDevice, MATMUL_INPUT_B_BYTES, inputBHost,
                              MATMUL_INPUT_B_BYTES,
                              ACL_MEMCPY_HOST_TO_DEVICE),
                  "aclrtMemcpy(inputB H2D)")) {
        rc = 1;
        goto cleanup;
    }

    MATMUL_LAUNCH_FUNCTION(workspace0Device, workspace1Device, inputADevice,
                           inputBDevice, outputDevice, stream);

    if (!CheckAcl(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream") ||
        !CheckAcl(aclrtMemcpy(outputHost, MATMUL_OUTPUT_BYTES, outputDevice,
                              MATMUL_OUTPUT_BYTES,
                              ACL_MEMCPY_DEVICE_TO_HOST),
                  "aclrtMemcpy(output D2H)")) {
        rc = 1;
        goto cleanup;
    }

    if (!WriteFile("output.bin", outputHost, MATMUL_OUTPUT_BYTES)) {
        std::fprintf(stderr, "[ERROR] failed to write output.bin\n");
        rc = 1;
        goto cleanup;
    }

    std::printf("[INFO] wrote output.bin (%zu bytes)\n",
                static_cast<size_t>(MATMUL_OUTPUT_BYTES));

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
