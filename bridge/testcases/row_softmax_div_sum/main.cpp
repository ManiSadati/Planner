#include "acl/acl.h"
#include "test_common.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
using namespace PtoTestCommon;
void LaunchRowSoftmaxDivSumKernel(uint8_t *workspace0, uint8_t *workspace1, float *input, float *output, void *stream);
namespace {
constexpr size_t kRows = 64, kCols = 256, kDataBytes = kRows * kCols * sizeof(float), kWorkspaceBytes = 1;
bool CheckAcl(aclError err, const char *what) {
    if (err == ACL_SUCCESS) return true;
    std::fprintf(stderr, "[ERROR] %s failed: %d\n", what, static_cast<int>(err));
    return false;
}
}
int main() {
    int rc = 0;
    int deviceId = std::getenv("ACL_DEVICE_ID") ? std::atoi(std::getenv("ACL_DEVICE_ID")) : 0;
    float *inputHost = nullptr, *outputHost = nullptr, *inputDevice = nullptr, *outputDevice = nullptr;
    uint8_t *workspace0Device = nullptr, *workspace1Device = nullptr;
    aclrtStream stream = nullptr;
    if (!CheckAcl(aclInit(nullptr), "aclInit") || !CheckAcl(aclrtSetDevice(deviceId), "aclrtSetDevice") || !CheckAcl(aclrtCreateStream(&stream), "aclrtCreateStream") || !CheckAcl(aclrtMallocHost(reinterpret_cast<void **>(&inputHost), kDataBytes), "aclrtMallocHost(input)") || !CheckAcl(aclrtMallocHost(reinterpret_cast<void **>(&outputHost), kDataBytes), "aclrtMallocHost(output)") || !CheckAcl(aclrtMalloc(reinterpret_cast<void **>(&inputDevice), kDataBytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(input)") || !CheckAcl(aclrtMalloc(reinterpret_cast<void **>(&outputDevice), kDataBytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(output)") || !CheckAcl(aclrtMalloc(reinterpret_cast<void **>(&workspace0Device), kWorkspaceBytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(workspace0)") || !CheckAcl(aclrtMalloc(reinterpret_cast<void **>(&workspace1Device), kWorkspaceBytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(workspace1)")) { rc = 1; goto cleanup; }
    { size_t inputSize = kDataBytes; if (!ReadFile("input.bin", inputSize, inputHost, kDataBytes)) { std::fprintf(stderr, "[ERROR] failed to read input.bin\n"); rc = 1; goto cleanup; } }
    if (!CheckAcl(aclrtMemcpy(inputDevice, kDataBytes, inputHost, kDataBytes, ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy(input H2D)")) { rc = 1; goto cleanup; }
    LaunchRowSoftmaxDivSumKernel(workspace0Device, workspace1Device, inputDevice, outputDevice, stream);
    if (!CheckAcl(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream") || !CheckAcl(aclrtMemcpy(outputHost, kDataBytes, outputDevice, kDataBytes, ACL_MEMCPY_DEVICE_TO_HOST), "aclrtMemcpy(output D2H)")) { rc = 1; goto cleanup; }
    if (!WriteFile("output.bin", outputHost, kDataBytes)) { std::fprintf(stderr, "[ERROR] failed to write output.bin\n"); rc = 1; }
cleanup:
    if (workspace0Device) aclrtFree(workspace0Device); if (workspace1Device) aclrtFree(workspace1Device); if (inputDevice) aclrtFree(inputDevice); if (outputDevice) aclrtFree(outputDevice); if (inputHost) aclrtFreeHost(inputHost); if (outputHost) aclrtFreeHost(outputHost); if (stream) aclrtDestroyStream(stream); aclrtResetDevice(deviceId); aclFinalize(); return rc;
}
