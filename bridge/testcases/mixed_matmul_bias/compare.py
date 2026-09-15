#!/usr/bin/env python3

import sys

import numpy as np


ELEMENT_COUNT = 64 * 64
golden = np.fromfile("golden.bin", dtype=np.float16)
output = np.fromfile("output.bin", dtype=np.float16)

if golden.size != ELEMENT_COUNT or output.size != ELEMENT_COUNT:
    print(f"[ERROR] shape mismatch: golden={golden.size}, output={output.size}")
    sys.exit(2)

golden_f32 = golden.astype(np.float32)
output_f32 = output.astype(np.float32)
absolute_difference = np.abs(golden_f32 - output_f32)

if np.allclose(golden_f32, output_f32, atol=1e-2, rtol=1e-2, equal_nan=True):
    print(f"[INFO] compare passed: max abs diff={float(absolute_difference.max())}")
    sys.exit(0)

index = int(np.argmax(absolute_difference))
row, column = divmod(index, 64)
print(
    f"[ERROR] compare failed: max diff={float(absolute_difference[index])} at "
    f"row={row} col={column} golden={float(golden_f32[index])} "
    f"output={float(output_f32[index])}"
)
sys.exit(2)
