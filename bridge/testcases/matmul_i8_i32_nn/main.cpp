#define MATMUL_LAUNCH_FUNCTION LaunchMatmulI8I32NNKernel
#define MATMUL_INPUT_A_BYTES (64 * 64)
#define MATMUL_INPUT_B_BYTES (64 * 64)
#define MATMUL_OUTPUT_BYTES (64 * 64 * 4)
#include "../common/matmul_main.h"
