import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from htm import ScalarEncoder, HTMNetwork, ConfidenceHTMNetwork
from htm.experiments import (
    sequence_learning,
    continual_learning,
    scaling,
    branching,
)
from htm.plotting import set_matplotlib_headless, plot_main_dashboard, plot_scaling


def _parse_list(arg: str):
    return [int(x) for x in arg.split(',') if x]


def parse_args():
    parser = argparse.ArgumentParser(description="Run HTM confidence experiments")
    parser.add_argument("--seeds", default="0,1,2", help="Comma-separated random seeds")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs for sequence learning")
    parser.add_argument("--epochs-per-phase", type=int, default=25, dest="epochs_per_phase",
                        help="Epochs per phase for continual learning")
    parser.add_argument("--lengths", default="5,10,20,40,60,80",
                        help="Comma-separated sequence lengths for scaling study")
    parser.add_argument("--outdir", default="results", help="Directory for outputs")
    return parser.parse_args()


def build_networks(seed: int):
    sp_params = {
        "column_count": 100,
        "sparsity": 0.1,
        "boost_strength": 0.0,
        "seed": seed,
    }
    tm_params = {
        "cells_per_column": 8,
        "activation_threshold": 10,
        "learning_threshold": 8,
        "max_synapses_per_segment": 16,
        "initial_permanence": 0.5,
        "permanence_increment": 0.02,
        "permanence_decrement": 0.005,
        "seed": seed,
    }
    baseline = HTMNetwork(input_size=100, sp_params=sp_params, tm_params=tm_params)
    confidence = ConfidenceHTMNetwork(input_size=100, sp_params=sp_params, tm_params=tm_params)
    return baseline, confidence


def main():
    args = parse_args()
    seeds = _parse_list(args.seeds)
    lengths = _parse_list(args.lengths)
    os.makedirs(args.outdir, exist_ok=True)

    encoder = ScalarEncoder(min_val=0, max_val=100, n_bits=100)

    seq_results = sequence_learning.train_curve(
        build_networks,
        sequence=[1, 2, 3, 4, 5],
        encoder=encoder,
        seeds=seeds,
        epochs=args.epochs,
    )

    cl_results = continual_learning.run_experiment(
        build_networks,
        seq_a=[1, 2, 3, 4, 5],
        seq_b=[1, 2, 6, 7, 5],
        encoder=encoder,
        seeds=seeds,
        epochs=args.epochs_per_phase,
    )

    sc_results = scaling.run_experiment(
        build_networks,
        lengths=lengths,
        seeds=seeds,
        encoder=encoder,
    )

    br_results = branching.run_experiment(
        build_networks,
        encoder=encoder,
        prefix=3,
        length=15,
        seeds=seeds,
    )

    results = {
        "sequence_learning": seq_results,
        "continual_learning": cl_results,
        "scaling": sc_results,
        "branching": br_results,
    }

    out_json = os.path.join(args.outdir, "htm_test_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    set_matplotlib_headless()
    dash_dict = {
        "sequence_comparison": seq_results,
        "continual_learning": cl_results,
        "branching_context": br_results,
    }
    plot_main_dashboard(dash_dict, save_path=os.path.join(args.outdir, "htm_confidence_results.png"))
    plot_scaling(sc_results, save_path=os.path.join(args.outdir, "htm_scaling.png"))

    print("Sequence final accuracy (baseline/confidence): {:.3f} / {:.3f}".format(
        seq_results['baseline_accuracy'][-1], seq_results['confidence_accuracy'][-1]))
    print("Continual learning retention (baseline/confidence): {:.3f} / {:.3f}".format(
        cl_results['baseline']['seq_a_after'][0], cl_results['confidence']['seq_a_after'][0]))
    print("Branching accuracy (branch/post, baseline): {:.3f}/{:.3f}".format(
        br_results['baseline']['branch_acc_mean'], br_results['baseline']['post_acc_mean']))
    print("Branching accuracy (branch/post, confidence): {:.3f}/{:.3f}".format(
        br_results['confidence']['branch_acc_mean'], br_results['confidence']['post_acc_mean']))
    print("Scaling lengths:", sc_results['lengths'])
    print("Baseline accuracy vs length:", sc_results['baseline_acc_mean'])
    print("Confidence accuracy vs length:", sc_results['confidence_acc_mean'])


if __name__ == "__main__":
    main()
