#define MATMUL_LAUNCH_FUNCTION LaunchMatmulBf16NNKernel
#define MATMUL_INPUT_A_BYTES (64 * 64 * 2)
#define MATMUL_INPUT_B_BYTES (64 * 64 * 2)
#define MATMUL_OUTPUT_BYTES (64 * 64 * 2)
#include "../common/matmul_main.h"
