#!/usr/bin/env python3
import sys
import numpy as np


golden = np.fromfile("golden.bin", dtype=np.int32)
output = np.fromfile("output.bin", dtype=np.int32)
if golden.size != 4096 or output.size != 4096:
    print(f"[ERROR] shape mismatch: golden={golden.size}, output={output.size}")
    sys.exit(2)
if np.array_equal(golden, output):
    print("[INFO] compare passed: exact INT32 match")
    sys.exit(0)
idx = int(np.argmax(np.abs(golden.astype(np.int64) - output.astype(np.int64))))
print(f"[ERROR] compare failed at index={idx}: golden={golden[idx]}, output={output[idx]}")
sys.exit(2)
