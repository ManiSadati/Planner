#include <cstdint>

#ifndef AICORE
#define AICORE [aicore]
#endif

extern "C" __global__ AICORE void flash_attention_tiny_kernel(
    __gm__ uint8_t *workspace0, __gm__ uint8_t *workspace1,
    __gm__ __fp16 *q, __gm__ __fp16 *kT, __gm__ __fp16 *v,
    __gm__ __fp16 *out, int32_t logicalBlockCount, int32_t outerCount,
    int32_t innerCount);

void LaunchFlashAttentionTinyKernel(void *workspace0, void *workspace1,
                                    void *q, void *kT, void *v, void *out,
                                    void *stream) {
    constexpr int32_t kLogicalBlockCount = 1;
    constexpr int32_t kPhysicalBlockCount = 1;
    flash_attention_tiny_kernel<<<kPhysicalBlockCount, nullptr, stream>>>(
        (__gm__ uint8_t *)workspace0,
        (__gm__ uint8_t *)workspace1,
        (__gm__ __fp16 *)q,
        (__gm__ __fp16 *)kT,
        (__gm__ __fp16 *)v,
        (__gm__ __fp16 *)out,
        kLogicalBlockCount,
        1,
        1);
}
