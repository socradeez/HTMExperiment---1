"""Sequence length scaling experiment."""

from typing import List, Callable, Tuple, Dict
import numpy as np


def run_experiment(builder: Callable[[int], Tuple[object, object]],
                   lengths: List[int], seeds: List[int], encoder,
                   epochs_fn=lambda L: max(10, L // 5)) -> Dict[str, object]:
    """Return accuracy and retention as sequence length varies."""
    results = {
        "lengths": lengths,
        "baseline_acc_mean": [],
        "confidence_acc_mean": [],
        "baseline_ret_mean": [],
        "confidence_ret_mean": [],
    }
    for L in lengths:
        b_acc, c_acc, b_ret, c_ret = [], [], [], []
        for s in seeds:
            base, conf = builder(s)
            seqA = list(range(10, 10 + L))
            seqB = list(range(40, 40 + L))
            def train_on(net, seq, epochs):
                for _ in range(epochs):
                    net.reset_sequence()
                    for v in seq:
                        net.compute(encoder.encode(v))
            def eval_on(net, seq):
                net.reset_sequence()
                accs = []
                for v in seq:
                    r = net.compute(encoder.encode(v), learn=False)
                    accs.append(1.0 - r['anomaly_score'])
                return float(np.mean(accs))
            epochs = epochs_fn(L)
            train_on(base, seqA, epochs)
            train_on(conf, seqA, epochs)
            b_acc.append(eval_on(base, seqA))
            c_acc.append(eval_on(conf, seqA))
            train_on(base, seqB, epochs)
            train_on(conf, seqB, epochs)
            b_ret.append(eval_on(base, seqA))
            c_ret.append(eval_on(conf, seqA))
        results["baseline_acc_mean"].append(float(np.mean(b_acc)))
        results["confidence_acc_mean"].append(float(np.mean(c_acc)))
        results["baseline_ret_mean"].append(float(np.mean(b_ret)))
        results["confidence_ret_mean"].append(float(np.mean(c_ret)))
    return results
