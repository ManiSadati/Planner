#!/usr/bin/env python3
import sys
import numpy as np


def from_bf16(values):
    return (values.astype(np.uint32) << 16).view(np.float32)


golden = from_bf16(np.fromfile("golden.bin", dtype=np.uint16))
output = from_bf16(np.fromfile("output.bin", dtype=np.uint16))
if golden.size != 4096 or output.size != 4096:
    print(f"[ERROR] shape mismatch: golden={golden.size}, output={output.size}")
    sys.exit(2)
if np.array_equal(golden, output):
    print("[INFO] compare passed: exact BF16 match")
    sys.exit(0)
diff = np.abs(golden - output)
idx = int(np.argmax(diff))
print(f"[ERROR] compare failed: max diff={float(diff[idx])} at index={idx}")
sys.exit(2)
