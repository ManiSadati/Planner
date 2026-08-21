#include <stdint.h>
#ifndef AICORE
#define AICORE [aicore]
#endif
extern "C" __global__ AICORE void row_softmax_exp_kernel(__gm__ uint8_t *workspace0, __gm__ uint8_t *workspace1, __gm__ float *input, __gm__ float *output, int32_t dim0, int32_t dim1, int32_t dim2);
void LaunchRowSoftmaxExpKernel(uint8_t *workspace0, uint8_t *workspace1, float *input, float *output, void *stream) {
    constexpr int32_t kRows = 64;
    row_softmax_exp_kernel<<<kRows, nullptr, stream>>>((__gm__ uint8_t *)workspace0, (__gm__ uint8_t *)workspace1, (__gm__ float *)input, (__gm__ float *)output, kRows, 1, 1);
}
