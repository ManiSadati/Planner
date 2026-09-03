#!/usr/bin/env python3
import numpy as np


np.ones((64, 64), dtype=np.int8).tofile("input_a.bin")
np.ones((64, 64), dtype=np.int8).tofile("input_b.bin")
np.full((64, 64), 64, dtype=np.int32).tofile("golden.bin")
print("[INFO] generated 64x64 INT8 inputs and INT32 output golden")
