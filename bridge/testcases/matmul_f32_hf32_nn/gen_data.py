#!/usr/bin/env python3
import numpy as np


np.ones((64, 64), dtype=np.float32).tofile("input_a.bin")
np.ones((64, 64), dtype=np.float32).tofile("input_b.bin")
np.full((64, 64), 64.0, dtype=np.float32).tofile("golden.bin")
print("[INFO] generated 64x64 F32 inputs and HF32-path output golden")
