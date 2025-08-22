"""Lightweight torch helpers for the BIO variant."""

import warnings
from typing import Set

import torch


def get_device(device_str: str) -> torch.device:
    """Return a torch.device, falling back to CPU if CUDA unavailable."""
    dev = torch.device(device_str)
    if dev.type == "cuda" and not torch.cuda.is_available():
        warnings.warn("CUDA not available, falling back to CPU")
        dev = torch.device("cpu")
    return dev


def set_to_bool_vec(active: Set[int], size: int, device: torch.device) -> torch.Tensor:
    """Convert a set of active indices to a boolean vector on ``device``."""
    vec = torch.zeros(size, dtype=torch.bool, device=device)
    if active:
        idx = torch.tensor(list(active), dtype=torch.int64, device=device)
        vec[idx] = True
    return vec
