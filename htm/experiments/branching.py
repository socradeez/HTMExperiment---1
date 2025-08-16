"""Branching context disambiguation experiment."""

from typing import List, Callable, Tuple, Dict
import numpy as np


def run_experiment(builder: Callable[[int], Tuple[object, object]],
                   encoder,
                   prefix: int = 3, length: int = 15,
                   seeds: List[int] = None, train_epochs: int = 30) -> Dict[str, object]:
    """Return branching and post-branch accuracy metrics."""
    seeds = seeds or [0]

    def build_AB(prefix: int, length: int):
        P = list(range(10, 10 + prefix))
        A = P + list(range(20, 20 + (length - prefix)))
        B = P + list(range(40, 40 + (length - prefix)))
        return A, B

    A, B = build_AB(prefix, length)
    results = {"prefix": prefix, "length": length, "baseline": {}, "confidence": {}}

    for model_name in ["baseline", "confidence"]:
        branch_acc = []
        post_acc = []
        for s in seeds:
            base, conf = builder(s)
            net = base if model_name == "baseline" else conf
            for _ in range(train_epochs):
                net.reset_sequence()
                for v in A:
                    net.compute(encoder.encode(v))
                net.reset_sequence()
                for v in B:
                    net.compute(encoder.encode(v))
            def eval_branch(net, seq_prev, seq_next):
                net.reset_sequence()
                for i, v in enumerate(seq_prev):
                    if i == prefix:
                        break
                    net.compute(encoder.encode(v), learn=False)
                r0 = net.compute(encoder.encode(seq_next[prefix]), learn=False)
                a0 = float(1.0 - r0['anomaly_score'])
                post = []
                for j in range(prefix+1, min(prefix+4, len(seq_next))):
                    rj = net.compute(encoder.encode(seq_next[j]), learn=False)
                    post.append(float(1.0 - rj['anomaly_score']))
                return a0, float(np.mean(post)) if post else a0
            a0, a_post = eval_branch(net, A, A)
            b0, b_post = eval_branch(net, B, B)
            branch_acc.append((a0 + b0) / 2.0)
            post_acc.append((a_post + b_post) / 2.0)
        results[model_name]["branch_acc_mean"] = float(np.mean(branch_acc))
        results[model_name]["branch_acc_std"] = float(np.std(branch_acc))
        results[model_name]["post_acc_mean"] = float(np.mean(post_acc))
        results[model_name]["post_acc_std"] = float(np.std(post_acc))
    return results
