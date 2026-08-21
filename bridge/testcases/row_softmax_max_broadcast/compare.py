#!/usr/bin/env python3
import sys
import numpy as np

ROWS = 64
COLS = 256
ATOL = 2e-4
RTOL = 2e-4

golden = np.fromfile("golden.bin", dtype=np.float32)
output = np.fromfile("output.bin", dtype=np.float32)
if golden.size != ROWS * COLS or output.size != ROWS * COLS:
    print(f"[ERROR] shape mismatch: golden={golden.size}, output={output.size}")
    sys.exit(2)
if np.allclose(golden, output, atol=ATOL, rtol=RTOL, equal_nan=True):
    print("[INFO] compare passed")
    sys.exit(0)
diff = np.abs(golden.astype(np.float64) - output.astype(np.float64))
idx = int(np.nanargmax(diff))
print(f"[ERROR] compare failed: max diff={float(diff[idx])} at idx={idx} row={idx // COLS} col={idx % COLS} golden={float(golden[idx])} output={float(output[idx])}")
sys.exit(2)
