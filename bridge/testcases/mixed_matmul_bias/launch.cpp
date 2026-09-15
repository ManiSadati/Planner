#include <cstdint>

#ifndef AICORE
#define AICORE [aicore]
#endif

extern "C" __global__ AICORE void mixed_matmul_bias_kernel(
    __gm__ uint8_t *syncWorkspace, __gm__ uint8_t *workspace,
    __gm__ __fp16 *inputA, __gm__ __fp16 *inputB, __gm__ __fp16 *bias,
    __gm__ __fp16 *output, int32_t logicalBlockCount, int32_t outerCount,
    int32_t innerCount);

void LaunchMixedMatmulBiasKernel(void *syncWorkspace, void *workspace,
                                 void *inputA, void *inputB, void *bias,
                                 void *output, void *stream) {
    constexpr int32_t kLogicalBlockCount = 1;
    constexpr int32_t kPhysicalBlockCount = 32;
    mixed_matmul_bias_kernel<<<kPhysicalBlockCount, nullptr, stream>>>(
        (__gm__ uint8_t *)syncWorkspace,
        (__gm__ uint8_t *)workspace,
        (__gm__ __fp16 *)inputA,
        (__gm__ __fp16 *)inputB,
        (__gm__ __fp16 *)bias,
        (__gm__ __fp16 *)output,
        kLogicalBlockCount,
        1,
        1);
}
