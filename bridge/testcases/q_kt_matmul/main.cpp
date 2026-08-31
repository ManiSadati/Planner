#include "acl/acl.h"
#include "test_common.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>

using namespace PtoTestCommon;

void LaunchQKtMatmulKernel(void *workspace0, void *workspace1, void *inputQ,
                           void *inputKt, void *scores, void *stream);

namespace {

constexpr size_t kHq = 32;
constexpr size_t kHkv = 2;
constexpr size_t kSq = 128;
constexpr size_t kSk = 8192;
constexpr size_t kHeadDim = 256;
constexpr size_t kQBytes = kHq * kSq * kHeadDim * sizeof(uint16_t);
constexpr size_t kKtBytes = kHkv * kHeadDim * kSk * sizeof(uint16_t);
constexpr size_t kScoreElems = kHq * kSq * kSk;
constexpr size_t kScoreBytes = kScoreElems * sizeof(uint16_t);
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

    uint16_t *qHost = nullptr;
    uint16_t *ktHost = nullptr;
    uint16_t *scoresHost = nullptr;
    uint16_t *qDevice = nullptr;
    uint16_t *ktDevice = nullptr;
    uint16_t *scoresDevice = nullptr;
    uint8_t *workspace0Device = nullptr;
    uint8_t *workspace1Device = nullptr;
    aclrtStream stream = nullptr;

    if (!CheckAcl(aclInit(nullptr), "aclInit") ||
        !CheckAcl(aclrtSetDevice(deviceId), "aclrtSetDevice") ||
        !CheckAcl(aclrtCreateStream(&stream), "aclrtCreateStream") ||
        !CheckAcl(aclrtMallocHost(reinterpret_cast<void **>(&qHost), kQBytes),
                  "aclrtMallocHost(Q)") ||
        !CheckAcl(aclrtMallocHost(reinterpret_cast<void **>(&ktHost), kKtBytes),
                  "aclrtMallocHost(KT)") ||
        !CheckAcl(aclrtMallocHost(reinterpret_cast<void **>(&scoresHost),
                                  kScoreBytes),
                  "aclrtMallocHost(scores)") ||
        !CheckAcl(aclrtMalloc(reinterpret_cast<void **>(&qDevice), kQBytes,
                              ACL_MEM_MALLOC_HUGE_FIRST),
                  "aclrtMalloc(Q)") ||
        !CheckAcl(aclrtMalloc(reinterpret_cast<void **>(&ktDevice), kKtBytes,
                              ACL_MEM_MALLOC_HUGE_FIRST),
                  "aclrtMalloc(KT)") ||
        !CheckAcl(aclrtMalloc(reinterpret_cast<void **>(&scoresDevice),
                              kScoreBytes, ACL_MEM_MALLOC_HUGE_FIRST),
                  "aclrtMalloc(scores)") ||
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
        size_t qSize = kQBytes;
        size_t ktSize = kKtBytes;
        if (!ReadFile("input_q.bin", qSize, qHost, kQBytes) ||
            qSize != kQBytes ||
            !ReadFile("input_kt.bin", ktSize, ktHost, kKtBytes) ||
            ktSize != kKtBytes) {
            std::fprintf(stderr, "[ERROR] failed to read Q/KT inputs\n");
            rc = 1;
            goto cleanup;
        }
    }

    if (!CheckAcl(aclrtMemcpy(qDevice, kQBytes, qHost, kQBytes,
                              ACL_MEMCPY_HOST_TO_DEVICE),
                  "aclrtMemcpy(Q H2D)") ||
        !CheckAcl(aclrtMemcpy(ktDevice, kKtBytes, ktHost, kKtBytes,
                              ACL_MEMCPY_HOST_TO_DEVICE),
                  "aclrtMemcpy(KT H2D)")) {
        rc = 1;
        goto cleanup;
    }

    LaunchQKtMatmulKernel(workspace0Device, workspace1Device, qDevice, ktDevice,
                          scoresDevice, stream);

    if (!CheckAcl(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream") ||
        !CheckAcl(aclrtMemcpy(scoresHost, kScoreBytes, scoresDevice,
                              kScoreBytes, ACL_MEMCPY_DEVICE_TO_HOST),
                  "aclrtMemcpy(scores D2H)")) {
        rc = 1;
        goto cleanup;
    }

    if (!WriteFile("output.bin", scoresHost, kScoreBytes)) {
        std::fprintf(stderr, "[ERROR] failed to write output.bin\n");
        rc = 1;
        goto cleanup;
    }

    std::printf("[INFO] wrote output.bin (%zu FP16 scores)\n", kScoreElems);

cleanup:
    if (workspace0Device != nullptr) aclrtFree(workspace0Device);
    if (workspace1Device != nullptr) aclrtFree(workspace1Device);
    if (qDevice != nullptr) aclrtFree(qDevice);
    if (ktDevice != nullptr) aclrtFree(ktDevice);
    if (scoresDevice != nullptr) aclrtFree(scoresDevice);
    if (qHost != nullptr) aclrtFreeHost(qHost);
    if (ktHost != nullptr) aclrtFreeHost(ktHost);
    if (scoresHost != nullptr) aclrtFreeHost(scoresHost);
    if (stream != nullptr) aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return rc;
}
