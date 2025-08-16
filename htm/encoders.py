"""Scalar encoding utilities."""

import numpy as np


class ScalarEncoder:
    """Encode scalar values as binary arrays."""

    def __init__(self, min_val=0, max_val=100, n_bits=100, w=11):
        self.min_val = min_val
        self.max_val = max_val
        self.n_bits = n_bits
        self.w = w
        self.resolution = (max_val - min_val) / (n_bits - w)

    def encode(self, value):
        """Create SDR for scalar value."""
        value = float(np.clip(value, self.min_val, self.max_val))
        center = int((value - self.min_val) / self.resolution) if self.resolution > 0 else 0
        sdr = np.zeros(self.n_bits, dtype=np.int32)
        half_w = self.w // 2
        for i in range(center - half_w, center + half_w + 1):
            if 0 <= i < self.n_bits:
                sdr[i] = 1
        return sdr
