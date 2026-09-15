#include "acl/acl.h"
#include "test_common.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

using namespace PtoTestCommon;

void LaunchMixedMatmulBiasKernel(void *syncWorkspace, void *workspace,
                                 void *inputA, void *inputB, void *bias,
                                 void *output, void *stream);

namespace {

constexpr size_t kElementCount = 64 * 64;
constexpr size_t kDataBytes = kElementCount * sizeof(uint16_t);
constexpr size_t kWorkspaceBytes = 4096;

bool CheckAcl(aclError error, const char *operation) {
    if (error == ACL_SUCCESS)
        return true;
    std::fprintf(stderr, "[ERROR] %s failed: %d\n", operation,
                 static_cast<int>(error));
    return false;
}

} // namespace

int main() {
    int result = 0;
    int deviceId = 0;
    if (const char *value = std::getenv("ACL_DEVICE_ID"))
        deviceId = std::atoi(value);

    uint16_t *inputAHost = nullptr;
    uint16_t *inputBHost = nullptr;
    uint16_t *biasHost = nullptr;
    uint16_t *outputHost = nullptr;
    uint16_t *inputADevice = nullptr;
    uint16_t *inputBDevice = nullptr;
    uint16_t *biasDevice = nullptr;
    uint16_t *outputDevice = nullptr;
    uint8_t *syncWorkspaceDevice = nullptr;
    uint8_t *workspaceDevice = nullptr;
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
        !CheckAcl(aclrtMallocHost(reinterpret_cast<void **>(&biasHost),
                                  kDataBytes),
                  "aclrtMallocHost(bias)") ||
        !CheckAcl(aclrtMallocHost(reinterpret_cast<void **>(&outputHost),
                                  kDataBytes),
                  "aclrtMallocHost(output)") ||
        !CheckAcl(aclrtMalloc(reinterpret_cast<void **>(&inputADevice),
                              kDataBytes, ACL_MEM_MALLOC_HUGE_FIRST),
                  "aclrtMalloc(inputA)") ||
        !CheckAcl(aclrtMalloc(reinterpret_cast<void **>(&inputBDevice),
                              kDataBytes, ACL_MEM_MALLOC_HUGE_FIRST),
                  "aclrtMalloc(inputB)") ||
        !CheckAcl(aclrtMalloc(reinterpret_cast<void **>(&biasDevice), kDataBytes,
                              ACL_MEM_MALLOC_HUGE_FIRST),
                  "aclrtMalloc(bias)") ||
        !CheckAcl(aclrtMalloc(reinterpret_cast<void **>(&outputDevice),
                              kDataBytes, ACL_MEM_MALLOC_HUGE_FIRST),
                  "aclrtMalloc(output)") ||
        !CheckAcl(aclrtMalloc(reinterpret_cast<void **>(&syncWorkspaceDevice),
                              kWorkspaceBytes, ACL_MEM_MALLOC_HUGE_FIRST),
                  "aclrtMalloc(syncWorkspace)") ||
        !CheckAcl(aclrtMalloc(reinterpret_cast<void **>(&workspaceDevice),
                              kWorkspaceBytes, ACL_MEM_MALLOC_HUGE_FIRST),
                  "aclrtMalloc(workspace)")) {
        result = 1;
        goto cleanup;
    }

    {
        size_t inputASize = kDataBytes;
        size_t inputBSize = kDataBytes;
        size_t biasSize = kDataBytes;
        if (!ReadFile("input_a.bin", inputASize, inputAHost, kDataBytes) ||
            inputASize != kDataBytes ||
            !ReadFile("input_b.bin", inputBSize, inputBHost, kDataBytes) ||
            inputBSize != kDataBytes ||
            !ReadFile("bias.bin", biasSize, biasHost, kDataBytes) ||
            biasSize != kDataBytes) {
            std::fprintf(stderr, "[ERROR] failed to read input files\n");
            result = 1;
            goto cleanup;
        }
    }

    if (!CheckAcl(aclrtMemcpy(inputADevice, kDataBytes, inputAHost, kDataBytes,
                              ACL_MEMCPY_HOST_TO_DEVICE),
                  "aclrtMemcpy(inputA H2D)") ||
        !CheckAcl(aclrtMemcpy(inputBDevice, kDataBytes, inputBHost, kDataBytes,
                              ACL_MEMCPY_HOST_TO_DEVICE),
                  "aclrtMemcpy(inputB H2D)") ||
        !CheckAcl(aclrtMemcpy(biasDevice, kDataBytes, biasHost, kDataBytes,
                              ACL_MEMCPY_HOST_TO_DEVICE),
                  "aclrtMemcpy(bias H2D)")) {
        result = 1;
        goto cleanup;
    }

    LaunchMixedMatmulBiasKernel(syncWorkspaceDevice, workspaceDevice,
                                inputADevice, inputBDevice, biasDevice,
                                outputDevice, stream);

    if (!CheckAcl(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream") ||
        !CheckAcl(aclrtMemcpy(outputHost, kDataBytes, outputDevice, kDataBytes,
                              ACL_MEMCPY_DEVICE_TO_HOST),
                  "aclrtMemcpy(output D2H)")) {
        result = 1;
        goto cleanup;
    }

    if (!WriteFile("output.bin", outputHost, kDataBytes)) {
        std::fprintf(stderr, "[ERROR] failed to write output.bin\n");
        result = 1;
        goto cleanup;
    }
    std::printf("[INFO] wrote output.bin (%zu FP16 elements)\n", kElementCount);

cleanup:
    if (syncWorkspaceDevice != nullptr) aclrtFree(syncWorkspaceDevice);
    if (workspaceDevice != nullptr) aclrtFree(workspaceDevice);
    if (inputADevice != nullptr) aclrtFree(inputADevice);
    if (inputBDevice != nullptr) aclrtFree(inputBDevice);
    if (biasDevice != nullptr) aclrtFree(biasDevice);
    if (outputDevice != nullptr) aclrtFree(outputDevice);
    if (inputAHost != nullptr) aclrtFreeHost(inputAHost);
    if (inputBHost != nullptr) aclrtFreeHost(inputBHost);
    if (biasHost != nullptr) aclrtFreeHost(biasHost);
    if (outputHost != nullptr) aclrtFreeHost(outputHost);
    if (stream != nullptr) aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return result;
}
