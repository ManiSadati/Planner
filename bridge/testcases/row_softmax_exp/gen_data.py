#!/usr/bin/env python3
import numpy as np
ROWS = 64
COLS = 256
np.random.seed(20260822)
input_data = np.random.uniform(-10.0, 0.0, size=(ROWS, COLS)).astype(np.float32)
golden = np.exp(input_data).astype(np.float32)
input_data.tofile("input.bin")
golden.tofile("golden.bin")
print(f"[INFO] generated input.bin golden.bin ({ROWS * COLS} float32 elements)")
