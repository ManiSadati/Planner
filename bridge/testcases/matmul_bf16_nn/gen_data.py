#!/usr/bin/env python3
import numpy as np


def to_bf16(values):
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)
    return (rounded >> 16).astype(np.uint16)


to_bf16(np.ones((64, 64), dtype=np.float32)).tofile("input_a.bin")
to_bf16(np.ones((64, 64), dtype=np.float32)).tofile("input_b.bin")
to_bf16(np.full((64, 64), 64.0, dtype=np.float32)).tofile("golden.bin")
print("[INFO] generated 64x64 BF16 inputs and output golden")
