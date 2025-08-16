import argparse
import os
from typing import List, Dict

import numpy as np
import pandas as pd

from config import ModelConfig, RunConfig
import run as run_mod
from input_gen import make_sequence_tokens, build_token_sdrs


def build_schedule(S: int, L: int, occurrences: int, schedule: str) -> (List[str], Dict[str, int]):
    tokens = [[f"S{s}_T{t}" for t in range(L)] for s in range(S)]
    token_pos_map = {f"S{s}_T{t}": t for s in range(S) for t in range(L)}
    explicit: List[str] = []
    if schedule == "blocked":
        for s in range(S):
            seq = tokens[s]
            for _ in range(occurrences):
                explicit.extend(seq)
    elif schedule == "interleave":
        for _ in range(occurrences):
            for t in range(L):
                for s in range(S):
                    explicit.append(tokens[s][t])
    elif schedule == "blocked_then_interleave":
        half = occurrences // 2
        b_tokens, _ = build_schedule(S, L, half, "blocked")
        i_tokens, _ = build_schedule(S, L, occurrences - half, "interleave")
        explicit = b_tokens + i_tokens
    else:
        raise ValueError(f"Unknown schedule {schedule}")
    return explicit, token_pos_map


def summarize_run(outdir: str, run_cfg: RunConfig, model_cfg: ModelConfig) -> Dict[str, float]:
    df = pd.read_csv(os.path.join(outdir, "metrics.csv"))
    tau = run_cfg.convergence_tau
    M = run_cfg.convergence_M
    times = []
    for _, grp in df.groupby("input_id"):
        vals = grp["stability_jaccard_ema"].to_list()
        reached = np.nan
        for i in range(len(vals) - M + 1):
            if all(v >= tau for v in vals[i:i + M]):
                reached = i + 1
                break
        times.append(reached)
    precision_mean = float(df["precision"].mean())
    recall_mean = float(df["recall"].mean())
    f1_mean = float(df["f1"].mean())
    bursting_rate_mean = float(df["bursting_columns"].mean() / model_cfg.k_active_columns)
    predicted_cells_mean = float(df["predicted_cells"].mean())
    active_cells_mean = float(df["active_cells"].mean())
    tail_vals = [grp["stability_jaccard_ema"].tail(5).mean() for _, grp in df.groupby("input_id")]
    stability_floor = float(min(tail_vals)) if tail_vals else float("nan")
    return {
        "time_to_stability_mean": float(np.nanmean(times)),
        "time_to_stability_median": float(np.nanmedian(times)),
        "precision_mean": precision_mean,
        "recall_mean": recall_mean,
        "f1_mean": f1_mean,
        "bursting_rate_mean": bursting_rate_mean,
        "predicted_cells_mean": predicted_cells_mean,
        "active_cells_mean": active_cells_mean,
        "stability_floor": stability_floor,
    }


def run_condition(L: int, S: int, O: int, schedule: str, occ: int, seed: int, out_root: str,
                  model_cfg: ModelConfig, run_cfg_template: RunConfig) -> Dict[str, float]:
    tokens = make_sequence_tokens(S, L)
    rng = np.random.default_rng(seed)
    token_sdrs = build_token_sdrs(
        tokens,
        input_size=model_cfg.input_size,
        on_bits=run_cfg_template.sdr_on_bits,
        overlap_pct=O,
        rng=rng,
    )
    explicit, token_pos_map = build_schedule(S, L, occ, schedule)
    run_name = f"L{L}_S{S}_O{O}_{schedule}_seed{seed}"
    run_cfg = RunConfig(
        seed=seed,
        steps=0,
        learn=True,
        figure_mode="single",
        annotate_formulas=False,
        per_input_plots_cells=False,
        per_input_plots_columns=False,
        diagnostics_print=False,
        output_dir=out_root,
        sdr_on_bits=run_cfg_template.sdr_on_bits,
        sequence="unused",
        stability_window=run_cfg_template.stability_window,
        ema_threshold=run_cfg_template.ema_threshold,
        convergence_tau=run_cfg_template.convergence_tau,
        convergence_M=run_cfg_template.convergence_M,
        input_flip_bits=run_cfg_template.input_flip_bits,
        explicit_step_tokens=explicit,
        token_pos_map=token_pos_map,
        schedule_name=schedule,
        run_name=run_name,
    )

    original_builder = run_mod.build_inputs

    def patched_builder(rng_, cfg, model_cfg_, tokens_unique=None):
        return token_sdrs

    run_mod.build_inputs = patched_builder
    try:
        outdir = run_mod.main(model_cfg, run_cfg)
    finally:
        run_mod.build_inputs = original_builder

    summary = summarize_run(outdir, run_cfg, model_cfg)
    summary.update({
        "length": L,
        "seq_count": S,
        "overlap": O,
        "schedule": schedule,
        "seed": seed,
    })
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", default="4,8,16,32")
    parser.add_argument("--seq-counts", default="1,2,4,8")
    parser.add_argument("--overlaps", default="0,25,50")
    parser.add_argument(
        "--schedule",
        default="interleave",
        choices=["blocked", "interleave", "blocked_then_interleave"],
    )
    parser.add_argument("--occurrences", type=int, default=60)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--out", default="runs/capacity")
    args = parser.parse_args()

    lengths = [int(x) for x in args.lengths.split(",") if x]
    seq_counts = [int(x) for x in args.seq_counts.split(",") if x]
    overlaps = [int(x) for x in args.overlaps.split(",") if x]

    os.makedirs(args.out, exist_ok=True)

    model_cfg = ModelConfig(synapses_per_column=64)
    run_cfg_template = RunConfig()

    rows = []
    for L in lengths:
        for S in seq_counts:
            for O in overlaps:
                for seed in range(args.seeds):
                    rows.append(
                        run_condition(
                            L,
                            S,
                            O,
                            args.schedule,
                            args.occurrences,
                            seed,
                            args.out,
                            model_cfg,
                            run_cfg_template,
                        )
                    )

    summary_df = pd.DataFrame(rows)
    summary_path = os.path.join(args.out, "capacity_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    group_cols = ["length", "seq_count", "overlap", "schedule"]
    grouped = summary_df.groupby(group_cols)
    for key, grp in grouped:
        t_mu = grp["time_to_stability_mean"].mean()
        t_sd = grp["time_to_stability_mean"].std()
        r_mu = grp["recall_mean"].mean()
        r_sd = grp["recall_mean"].std()
        sf_mu = grp["stability_floor"].mean()
        sf_sd = grp["stability_floor"].std()
        print(
            f"L{key[0]} S{key[1]} O{key[2]} {key[3]}: "
            f"t_stab={t_mu:.2f}±{t_sd:.2f}, "
            f"recall={r_mu:.2f}±{r_sd:.2f}, "
            f"stability_floor={sf_mu:.2f}±{sf_sd:.2f}"
        )

