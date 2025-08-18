"""Runner for the BIO variant."""

import json
import os
import time
from dataclasses import asdict
from typing import Set

import numpy as np
import torch

from config import ModelConfig as BaseModelConfig
from input_gen import build_token_sdrs
from torch_backend import TorchSP

from .config import BioModelConfig, BioRunConfig
from .predictive_bias import SubthresholdPredictor
from .inhibition import ColumnInhibition
from .tm import BioTM
from . import metrics_bio
from . import compat_metrics


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
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    sched = run_cfg.schedule_name or "bio"
    run_dir = os.path.join(run_cfg.outdir, "bio", f"{timestamp}_{sched}")
    os.makedirs(run_dir, exist_ok=True)
    _write_configs(model_cfg, run_cfg, run_dir)
    metrics_path = metrics_bio.init_metrics(run_dir)
    compat_writer = None
    if run_cfg.dry_run:
        metrics_bio.append_row(metrics_path, {"notes": "dry-run"})
        print("[BIO] scaffold ready (dry-run). No activation/learning performed.")
        return run_dir
    if run_cfg.compat_metrics:
        num_cells = model_cfg.num_columns * model_cfg.cells_per_column
        compat_writer = compat_metrics.open_writer(run_dir, num_cells, model_cfg.num_columns)

    rng = np.random.default_rng(run_cfg.seed)
    base_cfg = BaseModelConfig(
        input_size=model_cfg.input_size,
        num_columns=model_cfg.num_columns,
        cells_per_column=model_cfg.cells_per_column,
        k_active_columns=model_cfg.k_active_columns,
        segment_activation_threshold=model_cfg.segment_activation_threshold,
    )
    base_cfg.meta = model_cfg.meta
    sp = TorchSP(base_cfg, rng, model_cfg.device)
    tm = BioTM(model_cfg, rng, model_cfg.device)
    predictor = SubthresholdPredictor()
    inhibition = ColumnInhibition(model_cfg.inhibition_strength, model_cfg.winners_per_column)

    tokens = run_cfg.explicit_step_tokens or run_cfg.sequence.split(">")
    steps = run_cfg.steps or len(tokens)
    token_map = build_token_sdrs(
        tokens,
        model_cfg.input_size,
        on_bits=20,
        overlap_pct=run_cfg.overlap_pct,
        rng=rng,
        cross_sequence_reuse=True,
    )

    active_prev: Set[int] = set()
    predicted_cols_prev: Set[int] = set()
    predicted_cells_prev: Set[int] = set()

    for t in range(steps):
        tok = tokens[t % len(tokens)]
        bits = token_map[tok]
        x_bool = torch.zeros(model_cfg.input_size, dtype=torch.bool, device=sp.device)
        x_bool[bits] = True
        overlaps = sp.compute_overlap(x_bool)
        active_cols = sp.k_wta(overlaps, model_cfg.k_active_columns)

        bias, seg_counts = predictor.compute_bias(tm, active_prev)
        active_cells, bursting = tm.activate_cells(active_cols, bias, inhibition)

        sp.learn(x_bool, active_cols)
        tm.learn(active_prev, active_cells, bursting, seg_counts, model_cfg.meta)

        winner_counts = active_cells.view(model_cfg.num_columns, model_cfg.cells_per_column).sum(dim=1)
        winners_per_col_mean = (
            winner_counts[active_cols].float().mean().item() if active_cols.numel() > 0 else 0.0
        )
        winner_cells = int(active_cells.sum().item())
        winner_bias_mean = bias[active_cells].mean().item() if winner_cells > 0 else 0.0
        nonwin_mask = torch.zeros_like(active_cells)
        for col in active_cols.tolist():
            start = col * model_cfg.cells_per_column
            end = start + model_cfg.cells_per_column
            nonwin_mask[start:end] = True
        nonwin_mask &= ~active_cells
        nonwin_mean = bias[nonwin_mask].mean().item() if nonwin_mask.any() else 0.0

        margins = seg_counts - model_cfg.segment_activation_threshold
        predicted_cells = tm.seg_owner_cell[torch.nonzero(margins > 0, as_tuple=False).squeeze(1)]
        predicted_cols = set((predicted_cells // model_cfg.cells_per_column).tolist())
        col_precision = (
            len(predicted_cols_prev & set(active_cols.tolist())) / len(predicted_cols_prev)
            if predicted_cols_prev
            else 0.0
        )
        col_recall = (
            len(predicted_cols_prev & set(active_cols.tolist())) / active_cols.numel()
            if active_cols.numel() > 0
            else 0.0
        )
        cell_precision = (
            len(predicted_cells_prev & set(torch.nonzero(active_cells, as_tuple=False).squeeze(1).tolist()))
            / len(predicted_cells_prev)
            if predicted_cells_prev
            else 0.0
        )
        cell_recall = (
            len(predicted_cells_prev & set(torch.nonzero(active_cells, as_tuple=False).squeeze(1).tolist()))
            / winner_cells
            if winner_cells > 0
            else 0.0
        )

        metrics_bio.append_row(
            metrics_path,
            {
                "t": t,
                "input_id": tok,
                "pos_in_seq": t,
                "active_columns": int(active_cols.numel()),
                "bursting_columns": int(bursting.numel()),
                "winner_cells": winner_cells,
                "winners_per_column_mean": winners_per_col_mean,
                "distal_bias_winner_mean": winner_bias_mean,
                "distal_bias_nonwinner_mean": nonwin_mean,
                "predicted_columns": len(predicted_cols),
                "column_precision": col_precision,
                "column_recall": col_recall,
                "cell_precision": cell_precision,
                "cell_recall": cell_recall,
                "notes": "bio_v1",
            },
        )

        if compat_writer is not None:
            active_cell_ids = torch.nonzero(active_cells, as_tuple=False).squeeze(1).tolist()
            compat_metrics.write_compat_row(
                compat_writer,
                t,
                tok,
                t,
                active_cols.tolist(),
                bursting.tolist(),
                active_cell_ids,
                predicted_cells.tolist(),
                model_cfg.k_active_columns,
            )

        predicted_cols_prev = predicted_cols
        predicted_cells_prev = set(predicted_cells.tolist())
        active_prev = set(torch.nonzero(active_cells, as_tuple=False).squeeze(1).tolist())

    if compat_writer is not None:
        compat_writer.close()
    return run_dir
