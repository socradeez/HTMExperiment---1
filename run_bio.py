"""Entry point for the BIO scaffold."""

import argparse

from htm_bio.config import BioModelConfig, BioRunConfig
from htm_bio import runner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BIO variant scaffold")
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="Run without activation/learning (default)")
    parser.add_argument("--out", type=str, default="runs", help="Output directory base")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_cfg = BioModelConfig(device=args.device)
    run_cfg = BioRunConfig(outdir=args.out, seed=args.seed, dry_run=args.dry_run)
    run_dir = runner.main(model_cfg, run_cfg)
    print("Run directory:", run_dir)


if __name__ == "__main__":
    main()
