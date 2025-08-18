"""Runner for the BIO variant."""

import json
import os
import time
from dataclasses import asdict
from typing import Set

import numpy as np
import torch

from config import ModelConfig as BaseModelConfig
from torch_backend import TorchSP
from input_gen import build_token_sdrs

from .config import BioModelConfig, BioRunConfig
from .predictive_bias import SubthresholdPredictor
from .inhibition import ColumnInhibition
from .tm import BioTM
from . import metrics_bio


def _write_configs(model_cfg: BioModelConfig, run_cfg: BioRunConfig, run_dir: str) -> None:
    with open(os.path.join(run_dir, "config_model_bio.json"), "w") as f:
        json.dump(asdict(model_cfg), f, indent=2, sort_keys=True)
    run_dict = asdict(run_cfg)
    tokens = run_dict.pop("explicit_step_tokens")
    if tokens is not None:
        run_dict["explicit_step_tokens_len"] = len(tokens)
    with open(os.path.join(run_dir, "config_run_bio.json"), "w") as f:
        json.dump(run_dict, f, indent=2, sort_keys=True)


def main(model_cfg: BioModelConfig, run_cfg: BioRunConfig) -> str:
    """Execute a BIO run and return the output directory."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    sched = run_cfg.schedule_name or "bio"
    run_dir = os.path.join(run_cfg.outdir, "bio", f"{timestamp}_{sched}")
    os.makedirs(run_dir, exist_ok=True)
    _write_configs(model_cfg, run_cfg, run_dir)

    metrics_path = metrics_bio.init_metrics(run_dir)
    if run_cfg.dry_run:
        metrics_bio.append_row(metrics_path, {"notes": "dry-run"})
        print("[BIO] scaffold ready (dry-run). No activation/learning performed.")
        return run_dir

    rng = np.random.default_rng(run_cfg.seed)
    base_cfg = BaseModelConfig(
        input_size=model_cfg.input_size,
        num_columns=model_cfg.num_columns,
        cells_per_column=model_cfg.cells_per_column,
        k_active_columns=model_cfg.k_active_columns,
        segment_activation_threshold=model_cfg.segment_activation_threshold,
    )
    sp = TorchSP(base_cfg, rng, model_cfg.device)
    tm = BioTM(base_cfg, rng, model_cfg.device)
    predictor = SubthresholdPredictor(
        model_cfg.bias_gain,
        model_cfg.bias_cap,
        model_cfg.segment_activation_threshold,
    )
    inhibition = ColumnInhibition(model_cfg.inhibition_strength, model_cfg.winners_per_column)

    tokens = run_cfg.explicit_step_tokens or run_cfg.sequence.split(">")
    steps = run_cfg.steps or len(tokens)
    token_map = build_token_sdrs(tokens, model_cfg.input_size, on_bits=20, overlap_pct=0, rng=rng)

    active_cells_prev: Set[int] = set()
    predicted_prev = torch.empty(0, dtype=torch.int64, device=sp.device)

    for step in range(steps):
        tok = tokens[step % len(tokens)]
        bits = token_map[tok]
        x_bool = torch.zeros(model_cfg.input_size, dtype=torch.bool, device=sp.device)
        x_bool[bits] = True
        overlaps = sp.compute_overlap(x_bool)
        active_cols = sp.k_wta(overlaps, model_cfg.k_active_columns)

        evidence, predicted_cells, active_segments = tm.compute_distal_evidence(active_cells_prev)
        bias = predictor.compute_bias(evidence)
        active_cells = inhibition.select_winners(
            active_cols, bias, model_cfg.cells_per_column, model_cfg.winners_per_column
        )

        sp.learn(x_bool, active_cols)
        tm.learn(active_cells_prev, active_cols, active_cells, active_segments, predicted_prev)

        winners_per_col = (
            float(active_cells.numel()) / float(active_cols.numel())
            if active_cols.numel() > 0
            else 0.0
        )
        winner_mean = (
            float(bias[active_cells].mean().item()) if active_cells.numel() > 0 else 0.0
        )
        nonwinner_biases = []
        for col in active_cols.tolist():
            start = col * model_cfg.cells_per_column
            end = start + model_cfg.cells_per_column
            for c in range(start, end):
                if c not in active_cells.tolist():
                    nonwinner_biases.append(bias[c].item())
        nonwinner_mean = float(np.mean(nonwinner_biases)) if nonwinner_biases else 0.0
        pred_cols_prev = set((predicted_prev // model_cfg.cells_per_column).cpu().tolist())
        actual_cols = set(active_cols.cpu().tolist())
        precision = (
            len(pred_cols_prev & actual_cols) / len(pred_cols_prev) if pred_cols_prev else 0.0
        )
        recall = (
            len(pred_cols_prev & actual_cols) / len(actual_cols) if actual_cols else 0.0
        )
        metrics_bio.append_row(
            metrics_path,
            {
                "within_column_winner_count": winners_per_col,
                "distal_bias_winner_mean": winner_mean,
                "distal_bias_nonwinner_mean": nonwinner_mean,
                "column_precision": precision,
                "column_recall": recall,
                "predicted_columns": len(pred_cols_prev),
                "notes": tok,
            },
        )
        predicted_prev = predicted_cells
        active_cells_prev = set(active_cells.cpu().tolist())

    return run_dir
