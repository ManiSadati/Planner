#!/usr/bin/env python3

import sys

import numpy as np


QUERY_LEN = 128
HEAD_DIM = 64
ELEM_COUNT = QUERY_LEN * HEAD_DIM
ATOL = 5e-2
RTOL = 5e-2

golden = np.fromfile("golden.bin", dtype=np.float16)
output = np.fromfile("output.bin", dtype=np.float16)

if golden.size != ELEM_COUNT or output.size != ELEM_COUNT:
    print(f"[ERROR] shape mismatch: golden={golden.size}, output={output.size}")
    sys.exit(2)

golden_f32 = golden.astype(np.float32)
output_f32 = output.astype(np.float32)
abs_diff = np.abs(golden_f32 - output_f32)

if np.allclose(golden_f32, output_f32, atol=ATOL, rtol=RTOL, equal_nan=False):
    print(f"[INFO] compare passed: max abs diff={float(np.max(abs_diff))}")
    sys.exit(0)

idx = int(np.argmax(abs_diff))
row = idx // HEAD_DIM
col = idx % HEAD_DIM
print(
    f"[ERROR] compare failed: max diff={float(abs_diff[idx])} at "
    f"row={row} col={col} golden={float(golden_f32[idx])} "
    f"output={float(output_f32[idx])}"
)
sys.exit(2)
