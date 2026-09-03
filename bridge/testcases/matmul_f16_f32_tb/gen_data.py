#!/usr/bin/env python3
import numpy as np


input_a = np.eye(64, dtype=np.float16)
input_b = np.repeat(np.arange(1, 65, dtype=np.float32)[:, None], 64, axis=1).astype(np.float16)
golden = (input_a.astype(np.float32) @ input_b.astype(np.float32).T).astype(np.float16)
input_a.tofile("input_a.bin")
input_b.tofile("input_b.bin")
golden.tofile("golden.bin")
print("[INFO] generated asymmetric 64x64 F16 A, physical [N,K] B, and A@B.T golden")
