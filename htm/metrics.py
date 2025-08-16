"""Common metric utilities for HTM experiments."""

from typing import List, Dict, Set
import numpy as np


def accuracy_from_anomaly(results: List[Dict[str, float]]) -> float:
    """Return mean accuracy given a list of result dicts with 'anomaly_score'."""
    if not results:
        return 0.0
    return float(np.mean([1.0 - r.get('anomaly_score', 1.0) for r in results]))


def capture_transition_reprs(network, seq: List[int], encoder) -> Dict[str, Set[int]]:
    """Capture predicted-driven activations for each transition in a sequence."""
    reps: Dict[str, Set[int]] = {}
    if not seq:
        return reps
    network.reset_sequence()
    network.compute(encoder.encode(seq[0]), learn=False)
    for i in range(1, len(seq)):
        res = network.compute(encoder.encode(seq[i]), learn=False)
        preds = set(res['predictive_cells'])
        act = set(res['active_cells'])
        reps[f"{seq[i-1]}->{seq[i]}"] = preds & act if preds else set()
    return reps


def stability_overlap(before: Dict[str, Set[int]], after: Dict[str, Set[int]]) -> float:
    """Mean overlap of transition representations before and after learning."""
    overlaps = []
    for k in before.keys():
        b0 = before[k]
        b1 = after.get(k, set())
        overlaps.append(len(b0 & b1) / (len(b0) or 1))
    return float(np.mean(overlaps)) if overlaps else 0.0


def jaccard_stability(preds_before: Dict[str, Set[int]], preds_after: Dict[str, Set[int]]) -> float:
    """Mean Jaccard index between pre/post prediction sets for transitions."""
    scores = []
    for k in preds_before.keys():
        a = preds_before[k]
        b = preds_after.get(k, set())
        union = a | b
        scores.append(len(a & b) / (len(union) or 1))
    return float(np.mean(scores)) if scores else 0.0
