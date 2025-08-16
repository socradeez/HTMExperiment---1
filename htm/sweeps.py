import os
import csv
import json
from typing import Sequence, Dict, Tuple

import numpy as np

from .encoders import ScalarEncoder
from .network import ConfidenceHTMNetwork
from .metrics import capture_transition_reprs, stability_overlap
from .plotting import set_matplotlib_headless, plot_hardening_heatmaps


def run_hardening_sweep(
    *,
    rates: Sequence[float] = (0.0, 0.02, 0.05, 0.1, 0.2),
    thresholds: Sequence[float] = (0.55, 0.6, 0.65, 0.7),
    seeds: Sequence[int] = (0, 1, 2),
    epochs_per_phase: int = 25,
    outdir: str = "sweep_results",
) -> Dict[str, str]:
    """Run a hardening parameter sweep and save CSV/JSON/PNGs.

    Returns paths to the generated artifacts.
    """
    set_matplotlib_headless()
    os.makedirs(outdir, exist_ok=True)

    encoder = ScalarEncoder(min_val=0, max_val=10, n_bits=100)
    seq_a = [1, 2, 3, 4, 5]
    seq_b = [1, 2, 6, 7, 5]

    csv_rows = []
    agg: Dict[Tuple[float, float], Dict[str, list]] = {}
    for r in rates:
        for t in thresholds:
            agg[(r, t)] = {"init": [], "ret": [], "stab": []}

    for rate in rates:
        for thr in thresholds:
            for seed in seeds:
                tm_params = {
                    "cells_per_column": 8,
                    "activation_threshold": 10,
                    "learning_threshold": 8,
                    "initial_permanence": 0.5,
                    "permanence_increment": 0.02,
                    "permanence_decrement": 0.005,
                    "max_synapses_per_segment": 16,
                    "seed": seed,
                    "hardening_rate": rate,
                    "hardening_threshold": thr,
                }
                sp_params = {
                    "seed": seed,
                    "column_count": 100,
                    "sparsity": 0.1,
                    "boost_strength": 0.0,
                }
                net = ConfidenceHTMNetwork(
                    input_size=100, tm_params=tm_params, sp_params=sp_params
                )

                # train on sequence A
                for _ in range(epochs_per_phase):
                    net.reset_sequence()
                    for v in seq_a:
                        net.compute(encoder.encode(v))

                # evaluate initial accuracy on A
                net.reset_sequence()
                accs = []
                for v in seq_a:
                    res = net.compute(encoder.encode(v), learn=False)
                    accs.append(1.0 - res["anomaly_score"])
                initial_acc = float(accs[-1]) if accs else 0.0
                reps_before = capture_transition_reprs(net, seq_a, encoder)

                # train on sequence B
                for _ in range(epochs_per_phase):
                    net.reset_sequence()
                    for v in seq_b:
                        net.compute(encoder.encode(v))

                # evaluate retention on A
                net.reset_sequence()
                accs = []
                for v in seq_a:
                    res = net.compute(encoder.encode(v), learn=False)
                    accs.append(1.0 - res["anomaly_score"])
                retention_acc = float(accs[-1]) if accs else 0.0
                reps_after = capture_transition_reprs(net, seq_a, encoder)

                stab = stability_overlap(reps_before, reps_after)
                mean_conf = (
                    float(np.mean(net.tm.system_confidence))
                    if net.tm.system_confidence
                    else 0.0
                )
                frac_conf = net.tm._conf_over_thr_steps / max(1, net.tm._total_steps)
                mean_hard = net.tm._hardness_sum / max(1, net.tm._hardness_count)
                updates = net.tm._hardening_updates
                decays = net.tm._hardness_decays

                csv_rows.append(
                    {
                        "hardening_rate": rate,
                        "hardening_threshold": thr,
                        "seed": seed,
                        "initial_accuracy": initial_acc,
                        "retention_accuracy": retention_acc,
                        "representation_stability": stab,
                        "mean_conf": mean_conf,
                        "frac_conf_ge_thr": frac_conf,
                        "mean_hardness": mean_hard,
                        "hardening_updates": updates,
                        "hardness_decays": decays,
                    }
                )
                agg[(rate, thr)]["init"].append(initial_acc)
                agg[(rate, thr)]["ret"].append(retention_acc)
                agg[(rate, thr)]["stab"].append(stab)

    # write CSV
    csv_path = os.path.join(outdir, "hardening_sweep.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "hardening_rate",
                "hardening_threshold",
                "seed",
                "initial_accuracy",
                "retention_accuracy",
                "representation_stability",
                "mean_conf",
                "frac_conf_ge_thr",
                "mean_hardness",
                "hardening_updates",
                "hardness_decays",
            ],
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    # aggregate to JSON and matrices
    summary = {}
    ret_mat = np.zeros((len(thresholds), len(rates)))
    stab_mat = np.zeros((len(thresholds), len(rates)))
    for i, thr in enumerate(thresholds):
        for j, rate in enumerate(rates):
            vals = agg[(rate, thr)]
            key = f"rate_{rate}_thr_{thr}"
            init_vals = np.array(vals["init"])
            ret_vals = np.array(vals["ret"])
            stab_vals = np.array(vals["stab"])
            summary[key] = {
                "initial_accuracy_mean": float(np.mean(init_vals)),
                "initial_accuracy_std": float(np.std(init_vals)),
                "retention_accuracy_mean": float(np.mean(ret_vals)),
                "retention_accuracy_std": float(np.std(ret_vals)),
                "representation_stability_mean": float(np.mean(stab_vals)),
                "representation_stability_std": float(np.std(stab_vals)),
            }
            ret_mat[i, j] = np.mean(ret_vals)
            stab_mat[i, j] = np.mean(stab_vals)

    json_path = os.path.join(outdir, "hardening_sweep_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    # plots
    heatmap_path = os.path.join(outdir, "hardening_heatmaps.png")
    plot_hardening_heatmaps(ret_mat, stab_mat, rates, thresholds, heatmap_path)

    scatter_path = os.path.join(outdir, "hardening_pareto.png")
    import matplotlib.pyplot as plt

    plt.figure()
    xs = []
    ys = []
    labels = []
    for rate in rates:
        for thr in thresholds:
            vals = agg[(rate, thr)]
            xs.append(float(np.mean(vals["stab"])))
            ys.append(float(np.mean(vals["ret"])))
            labels.append(f"r{rate}-t{thr}")
    plt.scatter(xs, ys)
    for x, y, lbl in zip(xs, ys, labels):
        plt.text(x, y, lbl, fontsize=8)
    plt.xlabel("Representation Stability")
    plt.ylabel("Retention Accuracy")
    plt.title("Retention vs Stability")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(scatter_path, dpi=150, bbox_inches="tight")
    plt.close()

    return {
        "csv": csv_path,
        "json": json_path,
        "heatmaps": heatmap_path,
        "pareto": scatter_path,
    }
