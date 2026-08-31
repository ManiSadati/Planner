#!/usr/bin/env python3
import sys

import numpy as np

H_Q = 32
SQ = 128
SK = 8192
ELEM_COUNT = H_Q * SQ * SK
ATOL = 0.5
RTOL = 0.0

golden = np.fromfile("golden.bin", dtype=np.float16)
output = np.fromfile("output.bin", dtype=np.float16)

if golden.size != ELEM_COUNT or output.size != ELEM_COUNT:
    print(f"[ERROR] shape mismatch: golden={golden.size}, output={output.size}")
    sys.exit(2)

golden_f32 = golden.astype(np.float32)
output_f32 = output.astype(np.float32)
abs_diff = np.abs(golden_f32 - output_f32)

if np.allclose(golden_f32, output_f32, atol=ATOL, rtol=RTOL, equal_nan=True):
    print(f"[INFO] compare passed: max abs diff={float(np.max(abs_diff))}")
    sys.exit(0)

idx = int(np.argmax(abs_diff))
head = idx // (SQ * SK)
head_offset = idx % (SQ * SK)
row = head_offset // SK
col = head_offset % SK
print(
    f"[ERROR] compare failed: max diff={float(abs_diff[idx])} at "
    f"head={head} row={row} col={col} golden={float(golden_f32[idx])} "
    f"output={float(output_f32[idx])}"
)
sys.exit(2)
