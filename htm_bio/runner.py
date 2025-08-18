"""Minimal runner for the BIO scaffold."""

import json
import os
import time
from dataclasses import asdict

from .config import BioModelConfig, BioRunConfig
from . import metrics_bio


def main(model_cfg: BioModelConfig, run_cfg: BioRunConfig) -> str:
    """Set up run artifacts and optionally execute a dry run.

    Returns the path to the run directory.
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    sched = run_cfg.schedule_name or "bio"
    run_dir = os.path.join(run_cfg.outdir, "bio", f"{timestamp}_{sched}")
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "config_model_bio.json"), "w") as f:
        json.dump(asdict(model_cfg), f, indent=2, sort_keys=True)

    run_dict = asdict(run_cfg)
    tokens = run_dict.pop("explicit_step_tokens")
    if tokens is not None:
        run_dict["explicit_step_tokens_len"] = len(tokens)
    with open(os.path.join(run_dir, "config_run_bio.json"), "w") as f:
        json.dump(run_dict, f, indent=2, sort_keys=True)

    metrics_path = metrics_bio.init_metrics(run_dir)
    if run_cfg.dry_run:
        metrics_bio.append_row(metrics_path, {"notes": "dry-run"})
        print("[BIO] scaffold ready (dry-run). No activation/learning performed.")
        return run_dir

    print("[BIO] TODO: implement activation/learning")
    return run_dir
