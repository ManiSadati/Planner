#define MATMUL_LAUNCH_FUNCTION LaunchMatmulF16F32TBKernel
#define MATMUL_INPUT_A_BYTES (64 * 64 * 2)
#define MATMUL_INPUT_B_BYTES (64 * 64 * 2)
#define MATMUL_OUTPUT_BYTES (64 * 64 * 2)
#include "../common/matmul_main.h"
