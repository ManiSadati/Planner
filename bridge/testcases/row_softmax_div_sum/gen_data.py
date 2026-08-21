#!/usr/bin/env python3
import numpy as np
ROWS = 64
COLS = 256
np.random.seed(20260824)
input_data = np.random.uniform(0.001, 1.0, size=(ROWS, COLS)).astype(np.float32)
golden = (input_data / np.sum(input_data, axis=1, keepdims=True)).astype(np.float32)
input_data.tofile("input.bin")
golden.tofile("golden.bin")
print(f"[INFO] generated input.bin golden.bin ({ROWS * COLS} float32 elements)")
