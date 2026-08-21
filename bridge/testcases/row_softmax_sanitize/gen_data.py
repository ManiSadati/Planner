#!/usr/bin/env python3
import numpy as np
ROWS = 64
COLS = 256
np.random.seed(20260825)
input_data = np.random.uniform(-8.0, 8.0, size=(ROWS, COLS)).astype(np.float32)
input_data[0, 0] = np.nan
input_data[1, 7] = np.inf
input_data[2, 11] = -np.inf
golden = np.fmax(input_data, np.float32(-np.inf)).astype(np.float32)
input_data.tofile("input.bin")
golden.tofile("golden.bin")
print(f"[INFO] generated input.bin golden.bin ({ROWS * COLS} float32 elements)")
