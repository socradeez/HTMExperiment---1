"""Sequence learning experiment utilities."""

from typing import List, Callable, Tuple, Dict
import numpy as np


def train_curve(builder: Callable[[int], Tuple[object, object]],
                sequence: List[int], encoder,
                seeds: List[int] = None, epochs: int = 30) -> Dict[str, object]:
    """Return accuracy curves for baseline and confidence networks."""
    seeds = seeds or [0]
    baseline_acc_all = []
    confidence_acc_all = []
    for s in seeds:
        baseline, confidence = builder(s)
        baseline_accuracy: List[float] = []
        confidence_accuracy: List[float] = []
        for _ in range(epochs):
            epoch_baseline_acc = []
            epoch_confidence_acc = []
            baseline.reset_sequence()
            confidence.reset_sequence()
            for v in sequence:
                inp = encoder.encode(v)
                res_b = baseline.compute(inp)
                res_c = confidence.compute(inp)
                epoch_baseline_acc.append(1.0 - res_b['anomaly_score'])
                epoch_confidence_acc.append(1.0 - res_c['anomaly_score'])
            baseline_accuracy.append(float(np.mean(epoch_baseline_acc)))
            confidence_accuracy.append(float(np.mean(epoch_confidence_acc)))
        baseline_acc_all.append(baseline_accuracy)
        confidence_acc_all.append(confidence_accuracy)
    baseline_mean = np.mean(baseline_acc_all, axis=0).tolist()
    confidence_mean = np.mean(confidence_acc_all, axis=0).tolist()
    baseline_final = [acc[-1] for acc in baseline_acc_all]
    confidence_final = [acc[-1] for acc in confidence_acc_all]
    return {
        'baseline_accuracy': baseline_mean,
        'confidence_accuracy': confidence_mean,
        'baseline_final': baseline_final,
        'confidence_final': confidence_final,
    }
