import argparse
import os
import csv
from typing import List, Dict

import numpy as np

from config import ModelConfig, RunConfig
import run as run_mod
from input_gen import make_sequence_tokens, build_token_sdrs_between_sequences
from plotting import plot_baseline_meta_sweep


def _parse_csv_ints(values: List[str]) -> List[int]:
    out: List[int] = []
    for v in values:
        for part in v.split(','):
            part = part.strip()
            if part:
                out.append(int(part))
    return out


def _build_schedule(num_sequences: int, seq_length: int, repetitions: int) -> (List[str], Dict[str, int]):
    sequences = [[f"S{s}_T{t}" for t in range(seq_length)] for s in range(num_sequences)]
    explicit: List[str] = []
    for _ in range(repetitions):
        for seq in sequences:
            explicit.extend(seq)
    token_pos_map = {f"S{s}_T{t}": t for s in range(num_sequences) for t in range(seq_length)}
    return explicit, token_pos_map


def _summarize_metrics(outdir: str) -> Dict[str, float]:
    path = os.path.join(outdir, "metrics.csv")
    if not os.path.exists(path):
        return {}
    sums: Dict[str, float] = {}
    count = 0
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            count += 1
            for k, v in row.items():
                try:
                    val = float(v)
                except (TypeError, ValueError):
                    continue
                sums[k] = sums.get(k, 0.0) + val
    return {k: v / count for k, v in sums.items()} if count else {}


def run_condition(num_sequences: int, seq_length: int, overlap: int, repetitions: int,
                  seed: int, plots: List[str], out_root: str, model_cfg: ModelConfig,
                  run_cfg_template: RunConfig, backend: str) -> Dict[str, float]:
    tokens = make_sequence_tokens(num_sequences, seq_length)
    rng = np.random.default_rng(seed)
    token_sdrs = build_token_sdrs_between_sequences(
        tokens,
        input_size=model_cfg.input_size,
        on_bits=run_cfg_template.sdr_on_bits,
        overlap_pct=overlap,
        rng=rng,
    )
    explicit, token_pos_map = _build_schedule(num_sequences, seq_length, repetitions)
    run_name = f"S{num_sequences}_O{overlap}_seed{seed}"
    run_cfg = RunConfig(
        seed=seed,
        steps=0,
        learn=True,
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
        run_name=run_name,
        backend=backend,
        plots=plots,
    )
    original_builder = run_mod.build_inputs
    def patched_builder(rng_, cfg, model_cfg_, tokens_unique=None):
        return token_sdrs
    run_mod.build_inputs = patched_builder
    try:
        outdir = run_mod.main(model_cfg, run_cfg)
    finally:
        run_mod.build_inputs = original_builder
    summary = _summarize_metrics(outdir)
    summary.update({
        "num_sequences": num_sequences,
        "seq_length": seq_length,
        "overlap": overlap,
        "seed": seed,
        "run_dir": outdir,
    })
    return summary


def main(args):
    num_sequences_list = _parse_csv_ints(args.num_sequences)
    seq_length_list = _parse_csv_ints(args.seq_length)
    overlap_list = _parse_csv_ints(args.overlap)
    model_cfg = ModelConfig()
    run_cfg_template = RunConfig()
    os.makedirs(args.out, exist_ok=True)
    rows = []
    for L in seq_length_list:
        for S in num_sequences_list:
            for O in overlap_list:
                for seed in range(args.seeds):
                    rows.append(run_condition(
                        S,
                        L,
                        O,
                        args.repetitions,
                        seed,
                        args.plots,
                        args.out,
                        model_cfg,
                        run_cfg_template,
                        args.backend,
                    ))
    if rows:
        summary_path = os.path.join(args.out, "baseline_sweep_summary.csv")
        with open(summary_path, "w", newline="") as f:
            fieldnames = sorted({k for row in rows for k in row.keys()})
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print("Sweep complete. Summary saved to", summary_path)
        csv_paths = [os.path.join(r["run_dir"], "metrics.csv") for r in rows]
        labels = [os.path.basename(r["run_dir"]) for r in rows]
        plot_dir = os.path.join(args.out, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        plot_baseline_meta_sweep(csv_paths, labels, plot_dir)
    else:
        print("No runs executed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--num_sequences", nargs="+", default=["1"])
    parser.add_argument("--seq_length", nargs="+", default=["4"])
    parser.add_argument("--overlap", nargs="+", default=["0"])
    parser.add_argument("--plots", nargs="*", default=None)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--out", default="runs/sweep")
    parser.add_argument("--backend", default="torch", choices=["numpy", "torch"])
    args = parser.parse_args()
    main(args)
