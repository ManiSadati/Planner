#!/usr/bin/env python3
import sys
import numpy as np


golden = np.fromfile("golden.bin", dtype=np.float16)
output = np.fromfile("output.bin", dtype=np.float16)
if golden.size != 4096 or output.size != 4096:
    print(f"[ERROR] shape mismatch: golden={golden.size}, output={output.size}")
    sys.exit(2)
if np.allclose(golden, output, atol=1e-2, rtol=1e-2):
    print(f"[INFO] compare passed: max abs diff={float(np.max(np.abs(golden - output)))}")
    sys.exit(0)
idx = int(np.argmax(np.abs(golden.astype(np.float32) - output.astype(np.float32))))
print(f"[ERROR] compare failed at index={idx}: golden={golden[idx]}, output={output[idx]}")
sys.exit(2)
