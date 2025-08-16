import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class MetaParams:
    enabled: bool = False
    rungs: List[float] = field(default_factory=lambda: [0.30, 0.40, 0.50])
    rung_min_margin: Dict[float, int] = field(default_factory=lambda: {0.30: 0, 0.40: 2, 0.50: 4})
    rung_max_entropy: Dict[float, int] = field(default_factory=lambda: {0.30: 999, 0.40: 2, 0.50: 1})
    decay_beta: float = 2.0
    decay_floor: float = 0.10
    eps: float = 1e-6

def effective_dec(perms: np.ndarray, base_dec: float, params: MetaParams) -> np.ndarray:
    g = np.exp(-params.decay_beta * perms)
    g = np.maximum(g, params.decay_floor)
    return base_dec * g

def next_rung(p: float, rungs: List[float]) -> Optional[float]:
    for r in rungs:
        if p < r:
            return r
    return None

def apply_gates(perms: np.ndarray, inc_try: np.ndarray, margin: float, entropy: int, params: MetaParams) -> np.ndarray:
    out = inc_try.copy()
    for i, p in enumerate(perms):
        r = next_rung(p, params.rungs)
        if r is None:
            continue
        min_margin = params.rung_min_margin.get(r, 0)
        max_ent = params.rung_max_entropy.get(r, 999)
        if margin < min_margin or entropy > max_ent:
            target = r - params.eps
            if p + out[i] >= target:
                out[i] = max(0.0, target - p)
    return out
