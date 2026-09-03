#include <cstdint>

#ifndef AICORE
#define AICORE [aicore]
#endif

extern "C" __global__ AICORE void matmul_f16_f32_tb_kernel(
    __gm__ uint8_t *workspace0, __gm__ uint8_t *workspace1,
    __gm__ __fp16 *inputA, __gm__ __fp16 *inputB, __gm__ __fp16 *output,
    int32_t scalar0, int32_t scalar1, int32_t scalar2);

void LaunchMatmulF16F32TBKernel(void *workspace0, void *workspace1,
                                void *inputA, void *inputB, void *output,
                                void *stream) {
    matmul_f16_f32_tb_kernel<<<1, nullptr, stream>>>(
        (__gm__ uint8_t *)workspace0, (__gm__ uint8_t *)workspace1,
        (__gm__ __fp16 *)inputA, (__gm__ __fp16 *)inputB,
        (__gm__ __fp16 *)output, 1, 1, 1);
}
