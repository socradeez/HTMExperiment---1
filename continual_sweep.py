import argparse
import os
import csv

from config import ModelConfig, RunConfig
import run as run_mod
from input_gen import generate_noisy_stream


def summarize_run(outdir: str) -> dict:
    path = os.path.join(outdir, "metrics.csv")
    if not os.path.exists(path):
        return {}
    seq_vals = []
    noise_vals = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                val = float(row.get("stability_jaccard_last", "nan"))
            except ValueError:
                continue
            if row.get("is_noise_step") == "1":
                noise_vals.append(val)
            else:
                seq_vals.append(val)
    mean_seq = float(sum(seq_vals) / len(seq_vals)) if seq_vals else float("nan")
    mean_noise = float(sum(noise_vals) / len(noise_vals)) if noise_vals else float("nan")
    return {"stability_seq": mean_seq, "stability_noise": mean_noise}


def run_condition(V: int, N_total: int, gap_mean: int, K: int, L: int, noise_vocab: str,
                  seed: int, out_root: str, model_cfg: ModelConfig, run_cfg_template: RunConfig):
    sequences = [list(range(s * L, s * L + L)) for s in range(K)]
    stream = generate_noisy_stream(
        V=V,
        N_total=N_total,
        sequences=sequences,
        gap_dist=("poisson", gap_mean),
        noise_vocab=noise_vocab,
        seed=seed,
    )
    run_name = f"gap{gap_mean}_K{K}_L{L}_{noise_vocab}_seed{seed}"
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
        explicit_step_tokens=stream["tokens"],
        step_is_noise=stream["is_noise"],
        step_seq_id=stream["seq_id"],
        step_seq_pos=stream["seq_pos"],
        step_occurrence_id=stream["occurrence_id"],
        step_phase_in_sequence=stream["phase"],
        run_name=run_name,
        backend="torch",
        device=run_cfg_template.device,
    )
    outdir = run_mod.main(model_cfg, run_cfg)
    summary = summarize_run(outdir)
    summary.update({
        "gap_mean": gap_mean,
        "K": K,
        "L": L,
        "noise_vocab": noise_vocab,
        "seed": seed,
        "run_dir": outdir,
    })
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab", type=int, default=100)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--gap_means", default="2,6,12,24")
    parser.add_argument("--seq_counts", default="1,2,4")
    parser.add_argument("--seq_lengths", default="4,8,16")
    parser.add_argument("--noise_types", default="in_dist")
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--out", default="runs/continual")
    args = parser.parse_args()

    gap_means = [int(x) for x in args.gap_means.split(",") if x]
    seq_counts = [int(x) for x in args.seq_counts.split(",") if x]
    seq_lengths = [int(x) for x in args.seq_lengths.split(",") if x]
    noise_types = [x.strip() for x in args.noise_types.split(",") if x]

    os.makedirs(args.out, exist_ok=True)
    model_cfg = ModelConfig()
    run_cfg_template = RunConfig(device="cpu")

    rows = []
    for g in gap_means:
        for k in seq_counts:
            for l in seq_lengths:
                for nv in noise_types:
                    for seed in range(args.seeds):
                        rows.append(run_condition(args.vocab, args.steps, g, k, l, nv,
                                                  seed, args.out, model_cfg, run_cfg_template))
    if rows:
        summary_path = os.path.join(args.out, "continual_sweep_summary.csv")
        fieldnames = sorted({k for row in rows for k in row.keys()})
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print("Sweep complete. Summary saved to", summary_path)
    else:
        print("No runs executed")
