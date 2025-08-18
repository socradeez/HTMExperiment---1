"""Quick BIO sweep producing baseline-compatible figures."""

import argparse
import os
from itertools import product

import numpy as np
import pandas as pd

from .config import BioModelConfig, BioRunConfig
from . import runner
from . import plots_bio


def build_interleave_schedule(L: int, S: int, occurrences: int):
    tokens = []
    for _ in range(occurrences):
        for t in range(L):
            for s in range(S):
                tokens.append(f"S{s}_T{t}")
    token_pos_map = {f"S{s}_T{t}": t for s in range(S) for t in range(L)}
    return tokens, token_pos_map


def run_once(L: int, S: int, O: int, seed: int, args) -> dict:
    tokens, token_pos_map = build_interleave_schedule(L, S, args.occurrences)
    model_cfg = BioModelConfig(device=args.device)
    run_cfg = BioRunConfig(
        explicit_step_tokens=tokens,
        token_pos_map=token_pos_map,
        steps=len(tokens),
        outdir=args.out,
        seed=seed,
        schedule_name=f"L{L}_S{S}_O{O}_seed{seed}",
        dry_run=False,
        overlap_pct=O,
    )
    run_dir = runner.main(model_cfg, run_cfg)
    if args.plots:
        plots_bio.make_all(run_dir)
    metrics_path = os.path.join(run_dir, "metrics.csv")
    bio_path = os.path.join(run_dir, "metrics_bio.csv")
    mdf = pd.read_csv(metrics_path) if os.path.exists(metrics_path) else pd.DataFrame()
    bdf = pd.read_csv(bio_path) if os.path.exists(bio_path) else pd.DataFrame()
    return {
        "L": L,
        "S": S,
        "O": O,
        "seed": seed,
        "precision_mean": float(mdf["precision"].mean()) if "precision" in mdf else 0.0,
        "recall_mean": float(mdf["recall"].mean()) if "recall" in mdf else 0.0,
        "winners_per_column_mean": float(bdf["winners_per_column_mean"].mean())
        if "winners_per_column_mean" in bdf
        else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Quick BIO sweep")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="runs/bio_quick")
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--occurrences", type=int, default=40)
    parser.add_argument("--lengths", default="4,16")
    parser.add_argument("--seq-counts", default="1,4")
    parser.add_argument("--overlaps", default="0,50")
    parser.add_argument("--plots", dest="plots", action="store_true")
    parser.add_argument("--no-plots", dest="plots", action="store_false")
    parser.set_defaults(plots=True)
    args = parser.parse_args()

    lengths = [int(x) for x in args.lengths.split(",") if x]
    seq_counts = [int(x) for x in args.seq_counts.split(",") if x]
    overlaps = [int(x) for x in args.overlaps.split(",") if x]

    results = []
    for L, S, O in product(lengths, seq_counts, overlaps):
        for seed in range(args.seeds):
            res = run_once(L, S, O, seed, args)
            results.append(res)

    df = pd.DataFrame(results)
    rows = []
    for (L, S, O), grp in df.groupby(["L", "S", "O"]):
        rows.append(
            {
                "L": L,
                "S": S,
                "O": O,
                "precision_mean": grp["precision_mean"].mean(),
                "recall_mean": grp["recall_mean"].mean(),
                "winners_per_column_mean": grp["winners_per_column_mean"].mean(),
            }
        )
        print(
            f"(L{L}, S{S}, O{O}) -> precision_mean {rows[-1]['precision_mean']:.3f}, "
            f"recall_mean {rows[-1]['recall_mean']:.3f}, "
            f"winners_per_column_mean {rows[-1]['winners_per_column_mean']:.3f}"
        )
    os.makedirs(args.out, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(args.out, "bio_quick_summary.csv"), index=False)


if __name__ == "__main__":
    main()
