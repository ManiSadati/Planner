#include <cstdint>

#ifndef AICORE
#define AICORE [aicore]
#endif

extern "C" __global__ AICORE void matmul_f32_hf32_nn_kernel(
    __gm__ uint8_t *workspace0, __gm__ uint8_t *workspace1,
    __gm__ float *inputA, __gm__ float *inputB, __gm__ float *output,
    int32_t scalar0, int32_t scalar1, int32_t scalar2);

void LaunchMatmulF32HF32NNKernel(void *workspace0, void *workspace1,
                                 void *inputA, void *inputB, void *output,
                                 void *stream) {
    matmul_f32_hf32_nn_kernel<<<1, nullptr, stream>>>(
        (__gm__ uint8_t *)workspace0, (__gm__ uint8_t *)workspace1,
        (__gm__ float *)inputA, (__gm__ float *)inputB, (__gm__ float *)output,
        1, 1, 1);
}
