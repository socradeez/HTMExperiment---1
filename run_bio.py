"""Entry point for the BIO variant."""

import argparse

from htm_bio.config import BioModelConfig, BioRunConfig
from htm_bio import runner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BIO variant")
    parser.add_argument("--dry-run", action="store_true", help="Set up run without executing")
    parser.add_argument("--out", type=str, default="runs", help="Output directory base")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--steps", type=int, default=10, help="Number of steps to run")
    parser.add_argument("--winners-per-column", type=int, default=1)
    parser.add_argument("--bias-gain", type=float, default=1.0)
    parser.add_argument("--bias-cap", type=float, default=1.0)
    parser.add_argument("--ff-threshold", type=float, default=1.0)
    parser.add_argument("--meta", action="store_true", help="Enable metaplastic gating")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_cfg = BioModelConfig(
        device=args.device,
        winners_per_column=args.winners_per_column,
        bias_gain=args.bias_gain,
        bias_cap=args.bias_cap,
        ff_threshold=args.ff_threshold,
    )
    model_cfg.meta.enabled = args.meta
    run_cfg = BioRunConfig(outdir=args.out, seed=args.seed, steps=args.steps, dry_run=args.dry_run)
    run_dir = runner.main(model_cfg, run_cfg)
    print("Run directory:", run_dir)


if __name__ == "__main__":
    main()
