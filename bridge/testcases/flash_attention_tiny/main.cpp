#include "acl/acl.h"
#include "test_common.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

using namespace PtoTestCommon;

void LaunchFlashAttentionTinyKernel(void *workspace0, void *workspace1,
                                    void *q, void *kT, void *v, void *out,
                                    void *stream);

namespace {

constexpr size_t kSeqLen = 64;
constexpr size_t kHeadDim = 64;
constexpr size_t kElems = kSeqLen * kHeadDim;
constexpr size_t kTensorBytes = kElems * sizeof(uint16_t);
constexpr size_t kWorkspaceBytes = 4096;

bool CheckAcl(aclError error, const char *what) {
    if (error == ACL_SUCCESS) {
        return true;
    }
    std::fprintf(stderr, "[ERROR] %s failed: %d\n", what,
                 static_cast<int>(error));
    return false;
}

} // namespace

int main() {
    int result = 0;
    int deviceId = 0;
    if (const char *value = std::getenv("ACL_DEVICE_ID")) {
        deviceId = std::atoi(value);
    }

    uint16_t *qHost = nullptr;
    uint16_t *kTHost = nullptr;
    uint16_t *vHost = nullptr;
    uint16_t *outputHost = nullptr;
    uint16_t *qDevice = nullptr;
    uint16_t *kTDevice = nullptr;
    uint16_t *vDevice = nullptr;
    uint16_t *outputDevice = nullptr;
    uint8_t *workspace0Device = nullptr;
    uint8_t *workspace1Device = nullptr;
    aclrtStream stream = nullptr;

    if (!CheckAcl(aclInit(nullptr), "aclInit") ||
        !CheckAcl(aclrtSetDevice(deviceId), "aclrtSetDevice") ||
        !CheckAcl(aclrtCreateStream(&stream), "aclrtCreateStream") ||
        !CheckAcl(aclrtMallocHost(reinterpret_cast<void **>(&qHost),
                                  kTensorBytes), "aclrtMallocHost(Q)") ||
        !CheckAcl(aclrtMallocHost(reinterpret_cast<void **>(&kTHost),
                                  kTensorBytes), "aclrtMallocHost(KT)") ||
        !CheckAcl(aclrtMallocHost(reinterpret_cast<void **>(&vHost),
                                  kTensorBytes), "aclrtMallocHost(V)") ||
        !CheckAcl(aclrtMallocHost(reinterpret_cast<void **>(&outputHost),
                                  kTensorBytes), "aclrtMallocHost(output)") ||
        !CheckAcl(aclrtMalloc(reinterpret_cast<void **>(&qDevice), kTensorBytes,
                              ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(Q)") ||
        !CheckAcl(aclrtMalloc(reinterpret_cast<void **>(&kTDevice), kTensorBytes,
                              ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(KT)") ||
        !CheckAcl(aclrtMalloc(reinterpret_cast<void **>(&vDevice), kTensorBytes,
                              ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(V)") ||
        !CheckAcl(aclrtMalloc(reinterpret_cast<void **>(&outputDevice),
                              kTensorBytes, ACL_MEM_MALLOC_HUGE_FIRST),
                  "aclrtMalloc(output)") ||
        !CheckAcl(aclrtMalloc(reinterpret_cast<void **>(&workspace0Device),
                              kWorkspaceBytes, ACL_MEM_MALLOC_HUGE_FIRST),
                  "aclrtMalloc(workspace0)") ||
        !CheckAcl(aclrtMalloc(reinterpret_cast<void **>(&workspace1Device),
                              kWorkspaceBytes, ACL_MEM_MALLOC_HUGE_FIRST),
                  "aclrtMalloc(workspace1)")) {
        result = 1;
        goto cleanup;
    }

    {
        size_t qSize = kTensorBytes;
        size_t kTSize = kTensorBytes;
        size_t vSize = kTensorBytes;
        if (!ReadFile("input_q.bin", qSize, qHost, kTensorBytes) ||
            qSize != kTensorBytes ||
            !ReadFile("input_k_t.bin", kTSize, kTHost, kTensorBytes) ||
            kTSize != kTensorBytes ||
            !ReadFile("input_v.bin", vSize, vHost, kTensorBytes) ||
            vSize != kTensorBytes) {
            std::fprintf(stderr, "[ERROR] failed to read Q/K-transpose/V inputs\n");
            result = 1;
            goto cleanup;
        }
    }

    if (!CheckAcl(aclrtMemcpy(qDevice, kTensorBytes, qHost, kTensorBytes,
                              ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy(Q H2D)") ||
        !CheckAcl(aclrtMemcpy(kTDevice, kTensorBytes, kTHost, kTensorBytes,
                              ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy(KT H2D)") ||
        !CheckAcl(aclrtMemcpy(vDevice, kTensorBytes, vHost, kTensorBytes,
                              ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy(V H2D)")) {
        result = 1;
        goto cleanup;
    }

    LaunchFlashAttentionTinyKernel(workspace0Device, workspace1Device, qDevice,
                                   kTDevice, vDevice, outputDevice, stream);

    if (!CheckAcl(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream") ||
        !CheckAcl(aclrtMemcpy(outputHost, kTensorBytes, outputDevice,
                              kTensorBytes, ACL_MEMCPY_DEVICE_TO_HOST),
                  "aclrtMemcpy(output D2H)")) {
        result = 1;
        goto cleanup;
    }
    if (!WriteFile("output.bin", outputHost, kTensorBytes)) {
        std::fprintf(stderr, "[ERROR] failed to write output.bin\n");
        result = 1;
    } else {
        std::printf("[INFO] wrote output.bin (%zu FP16 elements)\n", kElems);
    }

cleanup:
    if (workspace0Device != nullptr) aclrtFree(workspace0Device);
    if (workspace1Device != nullptr) aclrtFree(workspace1Device);
    if (qDevice != nullptr) aclrtFree(qDevice);
    if (kTDevice != nullptr) aclrtFree(kTDevice);
    if (vDevice != nullptr) aclrtFree(vDevice);
    if (outputDevice != nullptr) aclrtFree(outputDevice);
    if (qHost != nullptr) aclrtFreeHost(qHost);
    if (kTHost != nullptr) aclrtFreeHost(kTHost);
    if (vHost != nullptr) aclrtFreeHost(vHost);
    if (outputHost != nullptr) aclrtFreeHost(outputHost);
    if (stream != nullptr) aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return result;
}
