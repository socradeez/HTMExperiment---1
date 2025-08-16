"""Continual learning experiment utilities."""

from typing import List, Callable, Tuple, Dict
import numpy as np

from ..metrics import capture_transition_reprs, stability_overlap


def run_experiment(builder: Callable[[int], Tuple[object, object]],
                   seq_a: List[int], seq_b: List[int], encoder,
                   seeds: List[int] = None, epochs: int = 25) -> Dict[str, object]:
    """Return continual learning metrics for baseline and confidence networks."""
    seeds = seeds or [0]
    baseline_seq_a = []
    baseline_seq_b = []
    baseline_seq_a_after = []
    confidence_seq_a = []
    confidence_seq_b = []
    confidence_seq_a_after = []
    baseline_stabilities = []
    confidence_stabilities = []

    for s in seeds:
        baseline, confidence = builder(s)
        for net, acc_list in [(baseline, baseline_seq_a), (confidence, confidence_seq_a)]:
            for _ in range(epochs):
                net.reset_sequence()
                epoch_acc = []
                for v in seq_a:
                    res = net.compute(encoder.encode(v))
                    epoch_acc.append(1.0 - res['anomaly_score'])
                acc_list.append(float(np.mean(epoch_acc)))
        reps_base_A = capture_transition_reprs(baseline, seq_a, encoder)
        reps_conf_A = capture_transition_reprs(confidence, seq_a, encoder)
        for net, acc_list in [(baseline, baseline_seq_b), (confidence, confidence_seq_b)]:
            for _ in range(epochs):
                net.reset_sequence()
                epoch_acc = []
                for v in seq_b:
                    res = net.compute(encoder.encode(v))
                    epoch_acc.append(1.0 - res['anomaly_score'])
                acc_list.append(float(np.mean(epoch_acc)))
        reps_base_after = capture_transition_reprs(baseline, seq_a, encoder)
        reps_conf_after = capture_transition_reprs(confidence, seq_a, encoder)
        for net, acc_list in [(baseline, baseline_seq_a_after), (confidence, confidence_seq_a_after)]:
            net.reset_sequence()
            test_acc = []
            for v in seq_a:
                res = net.compute(encoder.encode(v), learn=False)
                test_acc.append(1.0 - res['anomaly_score'])
            acc_list.append(float(np.mean(test_acc)))
        baseline_stabilities.append(stability_overlap(reps_base_A, reps_base_after))
        confidence_stabilities.append(stability_overlap(reps_conf_A, reps_conf_after))

    avg_baseline_stability = float(np.mean(baseline_stabilities)) if baseline_stabilities else 0.0
    avg_confidence_stability = float(np.mean(confidence_stabilities)) if confidence_stabilities else 0.0

    baseline_results = {
        'seq_a': [float(np.mean(baseline_seq_a))],
        'seq_b_during': [float(np.mean(baseline_seq_b))],
        'seq_a_after': [float(np.mean(baseline_seq_a_after))],
    }
    confidence_results = {
        'seq_a': [float(np.mean(confidence_seq_a))],
        'seq_b_during': [float(np.mean(confidence_seq_b))],
        'seq_a_after': [float(np.mean(confidence_seq_a_after))],
    }

    def get_permanence_stats(network):
        all_perms = []
        for cell_segments in network.tm.segments.values():
            for segment in cell_segments:
                for _, perm in segment['synapses']:
                    all_perms.append(perm)
        if all_perms:
            return {
                'mean': float(np.mean(all_perms)),
                'high_perm_ratio': float(np.mean([p > 0.7 for p in all_perms])),
                'very_high_perm_ratio': float(np.mean([p > 0.85 for p in all_perms])),
                'max': float(np.max(all_perms)),
                'total_synapses': int(len(all_perms)),
            }
        return {'mean': 0.0, 'high_perm_ratio': 0.0, 'very_high_perm_ratio': 0.0, 'max': 0.0, 'total_synapses': 0}

    baseline_perm_stats = get_permanence_stats(baseline)
    confidence_perm_stats = get_permanence_stats(confidence)

    return {
        'baseline': baseline_results,
        'confidence': confidence_results,
        'baseline_stability': avg_baseline_stability,
        'confidence_stability': avg_confidence_stability,
        'baseline_perm_stats': baseline_perm_stats,
        'confidence_perm_stats': confidence_perm_stats,
    }
