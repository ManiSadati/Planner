#include <cstdint>

#ifndef AICORE
#define AICORE [aicore]
#endif

extern "C" __global__ AICORE void q_kt_matmul_kernel(
    __gm__ uint8_t *workspace0, __gm__ uint8_t *workspace1,
    __gm__ __fp16 *inputQ, __gm__ __fp16 *inputKt, __gm__ __fp16 *scores,
    int32_t logicalBlockCount, int32_t outerCount, int32_t innerCount);

void LaunchQKtMatmulKernel(void *workspace0, void *workspace1, void *inputQ,
                           void *inputKt, void *scores, void *stream) {
    constexpr int32_t kLogicalBlockCount = 32 * 2 * 128;
    constexpr int32_t kPhysicalBlockCount = 32;
    q_kt_matmul_kernel<<<kPhysicalBlockCount, nullptr, stream>>>(
        (__gm__ uint8_t *)workspace0,
        (__gm__ uint8_t *)workspace1,
        (__gm__ __fp16 *)inputQ,
        (__gm__ __fp16 *)inputKt,
        (__gm__ __fp16 *)scores,
        kLogicalBlockCount,
        1,
        1);
}
