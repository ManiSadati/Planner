#include <cstdint>

#ifndef AICORE
#define AICORE [aicore]
#endif

extern "C" __global__ AICORE void matmul_i8_i32_nn_kernel(
    __gm__ uint8_t *workspace0, __gm__ uint8_t *workspace1,
    __gm__ int8_t *inputA, __gm__ int8_t *inputB, __gm__ int32_t *output,
    int32_t scalar0, int32_t scalar1, int32_t scalar2);

void LaunchMatmulI8I32NNKernel(void *workspace0, void *workspace1, void *inputA,
                               void *inputB, void *output, void *stream) {
    matmul_i8_i32_nn_kernel<<<1, nullptr, stream>>>(
        (__gm__ uint8_t *)workspace0, (__gm__ uint8_t *)workspace1,
        (__gm__ int8_t *)inputA, (__gm__ int8_t *)inputB,
        (__gm__ int32_t *)output, 1, 1, 1);
}
