#define MATMUL_LAUNCH_FUNCTION LaunchMatmulF32HF32NNKernel
#define MATMUL_INPUT_A_BYTES (64 * 64 * 4)
#define MATMUL_INPUT_B_BYTES (64 * 64 * 4)
#define MATMUL_OUTPUT_BYTES (64 * 64 * 4)
#include "../common/matmul_main.h"
