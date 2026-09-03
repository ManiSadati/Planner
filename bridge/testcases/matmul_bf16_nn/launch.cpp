#include <cstdint>

#ifndef AICORE
#define AICORE [aicore]
#endif

extern "C" __global__ AICORE void matmul_bf16_nn_kernel(
    __gm__ uint8_t *workspace0, __gm__ uint8_t *workspace1,
    __gm__ uint16_t *inputA, __gm__ uint16_t *inputB,
    __gm__ uint16_t *output, int32_t scalar0, int32_t scalar1,
    int32_t scalar2);

void LaunchMatmulBf16NNKernel(void *workspace0, void *workspace1, void *inputA,
                              void *inputB, void *output, void *stream) {
    matmul_bf16_nn_kernel<<<1, nullptr, stream>>>(
        (__gm__ uint8_t *)workspace0, (__gm__ uint8_t *)workspace1,
        (__gm__ uint16_t *)inputA, (__gm__ uint16_t *)inputB,
        (__gm__ uint16_t *)output, 1, 1, 1);
}
