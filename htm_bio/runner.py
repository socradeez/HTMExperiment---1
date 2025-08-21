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
    unique_tokens = list(dict.fromkeys(tokens))
    token_map = build_token_sdrs(
        unique_tokens,
        model_cfg.input_size,
        on_bits=20,
        overlap_pct=run_cfg.overlap_pct,
        rng=rng,
        cross_sequence_reuse=True,
    )

    active_prev: Set[int] = set()
    winner_prev: Set[int] = set()
    predicted_cols_prev: Set[int] = set()
    predicted_cells_prev: Set[int] = set()

    for t in range(steps):
        tok = tokens[t % len(tokens)]
        bits = token_map[tok]
        x_bool = torch.zeros(model_cfg.input_size, dtype=torch.bool, device=sp.device)
        x_bool[bits] = True
        overlaps = sp.compute_overlap(x_bool)
        active_cols_sp = sp.k_wta(overlaps, model_cfg.k_active_columns)

        bias, seg_counts = predictor.compute_bias(tm, active_prev)
        margins = seg_counts - model_cfg.segment_activation_threshold

        mask = margins > 0
        if mask.any():
            owner_ids = tm.seg_owner_cell[mask]
            seg_margins = margins[mask]
            pred_dict: dict[int, float] = {}
            for i in range(owner_ids.numel()):
                owner = int(owner_ids[i])
                m = float(seg_margins[i])
                if owner not in pred_dict or m > pred_dict[owner]:
                    pred_dict[owner] = m
            predicted_cells = torch.tensor(list(pred_dict.keys()), dtype=torch.int64, device=tm.device)
            pred_margins = torch.tensor(list(pred_dict.values()), device=tm.device)
        else:
            predicted_cells = torch.empty(0, dtype=torch.int64, device=tm.device)
            pred_margins = torch.empty(0, dtype=torch.float32, device=tm.device)
        pred_cols_per_cell = predicted_cells // model_cfg.cells_per_column
        predicted_cols_tensor = pred_cols_per_cell.unique()

        sp_set = set(active_cols_sp.tolist())
        pred_set = set(predicted_cols_tensor.tolist())
        union_set = sp_set | pred_set
        both_set = sp_set & pred_set
        pred_only_set = pred_set - sp_set
        sp_only_set = sp_set - pred_set

        active_cols_union = torch.tensor(sorted(union_set), dtype=torch.int64, device=tm.device)

        winners_list = []
        winners_from_pred = 0

        for col in pred_set:
            mask_col = pred_cols_per_cell == col
            cells_in_col = predicted_cells[mask_col]
            margins_in_col = pred_margins[mask_col]
            if cells_in_col.numel() > 0:
                k = min(model_cfg.winners_per_column, cells_in_col.numel())
                if cells_in_col.numel() > k:
                    topk = torch.topk(margins_in_col, k).indices
                    cells_in_col = cells_in_col[topk]
                winners_list.extend(cells_in_col.tolist())
                winners_from_pred += cells_in_col.numel()

        active_cols_sp_only = torch.tensor(sorted(sp_only_set), dtype=torch.int64, device=tm.device)
        winners_sp = torch.empty(0, dtype=torch.int64, device=tm.device)
        bursting = torch.empty(0, dtype=torch.int64, device=tm.device)
        if active_cols_sp_only.numel() > 0:
            winners_sp, bursting = inhibition.select_winners(
                active_cols_sp_only,
                bias,
                model_cfg.cells_per_column,
                model_cfg.winners_per_column,
            )
            winners_list.extend(winners_sp.tolist())

        winners = torch.tensor(winners_list, dtype=torch.int64, device=tm.device)
        active_cells = torch.zeros(tm.num_cells, dtype=torch.bool, device=tm.device)
        if winners.numel() > 0:
            active_cells[winners] = True
        for col in bursting.tolist():
            start = col * model_cfg.cells_per_column
            end = start + model_cfg.cells_per_column
            active_cells[start:end] = True

        sp.learn(x_bool, active_cols_sp)
        pre_ratio, cur_ratio = tm.learn(winner_prev, active_cells, bursting, seg_counts, model_cfg.meta)

        active_cols = active_cols_union
        winner_counts = active_cells.view(model_cfg.num_columns, model_cfg.cells_per_column).sum(dim=1)
        burst_set = set(bursting.tolist())
        nonburst_cols = [c for c in union_set if c not in burst_set]
        nonburst_tensor = torch.tensor(nonburst_cols, dtype=torch.int64, device=tm.device)
        winners_per_col_mean = (
            winner_counts[nonburst_tensor].float().mean().item() if nonburst_tensor.numel() > 0 else 0.0
        )
        bias_eps = 1e-9
        winner_cells = int(active_cells.sum().item())
        winner_bias_mean = bias[active_cells].mean().item() if winner_cells > 0 else 0.0
        nonwin_mask = torch.zeros_like(active_cells)
        for col in active_cols.tolist():
            start = col * model_cfg.cells_per_column
            end = start + model_cfg.cells_per_column
            nonwin_mask[start:end] = True
        nonwin_mask &= ~active_cells
        nonwin_mean = bias[nonwin_mask].mean().item() if nonwin_mask.any() else 0.0
        bias_matrix = bias.view(model_cfg.num_columns, model_cfg.cells_per_column)
        columns_with_bias = int((bias_matrix[active_cols] > bias_eps).any(dim=1).sum().item())
        nonzero_bias_cells = int((bias > bias_eps).sum().item())

        predicted_not_winner = (
            int((~active_cells[predicted_cells]).sum().item())
            if predicted_cells.numel() > 0
            else 0
        )
        col_precision = (
            len(predicted_cols_prev & union_set) / len(predicted_cols_prev)
            if predicted_cols_prev
            else 0.0
        )
        col_recall = (
            len(predicted_cols_prev & union_set) / len(active_cols)
            if active_cols.numel() > 0
            else 0.0
        )
        active_cell_ids_list = torch.nonzero(active_cells, as_tuple=False).squeeze(1).tolist()
        cell_precision = (
            len(predicted_cells_prev & set(active_cell_ids_list)) / len(predicted_cells_prev)
            if predicted_cells_prev
            else 0.0
        )
        cell_recall = (
            len(predicted_cells_prev & set(active_cell_ids_list)) / winner_cells
            if winner_cells > 0
            else 0.0
        )

        active_mask = torch.zeros(tm.num_segments, dtype=torch.bool, device=tm.device)
        active_mask[: seg_counts.shape[0]] = seg_counts >= model_cfg.segment_activation_threshold
        if active_mask.any():
            owner_ids = tm.seg_owner_cell[active_mask]
            owner_active = active_cells[owner_ids]
            tp_segments = int(owner_active.sum().item())
            fp_segments = int((~owner_active).sum().item())
        else:
            tp_segments = fp_segments = 0

        cols_sp = len(sp_set)
        cols_pred = len(pred_set)
        cols_union = len(union_set)
        cols_both = len(both_set)
        cols_pred_only = len(pred_only_set)
        cols_sp_only = len(sp_only_set)

        metrics_bio.append_row(
            metrics_path,
            {
                "t": t,
                "input_id": tok,
                "pos_in_seq": t,
                "active_columns": cols_union,
                "bursting_columns": int(bursting.numel()),
                "cols_sp": cols_sp,
                "cols_pred": cols_pred,
                "cols_union": cols_union,
                "cols_both": cols_both,
                "cols_pred_only": cols_pred_only,
                "cols_sp_only": cols_sp_only,
                "winners_from_pred": winners_from_pred,
                "winner_cells": winner_cells,
                "winners_per_column_mean": winners_per_col_mean,
                "distal_bias_winner_mean": winner_bias_mean,
                "distal_bias_nonwinner_mean": nonwin_mean,
                "columns_with_bias": columns_with_bias,
                "nonzero_bias_cells": nonzero_bias_cells,
                "predicted_not_winner": predicted_not_winner,
                "ltp_pre_ratio": pre_ratio,
                "ltp_cur_ratio": cur_ratio,
                "predicted_columns": cols_pred,
                "column_precision": col_precision,
                "column_recall": col_recall,
                "cell_precision": cell_precision,
                "cell_recall": cell_recall,
                "tp_segments": tp_segments,
                "fp_segments": fp_segments,
                "notes": "bio_v1",
            },
        )

        if compat_writer is not None:
            compat_metrics.write_compat_row(
                compat_writer,
                t,
                tok,
                t,
                active_cols.tolist(),
                bursting.tolist(),
                active_cell_ids_list,
                predicted_cells.tolist(),
                model_cfg.k_active_columns,
            )

        predicted_cols_prev = pred_set
        predicted_cells_prev = set(predicted_cells.tolist())
        active_prev = set(active_cell_ids_list)
        winner_prev = set(winners.tolist())

    if compat_writer is not None:
        compat_writer.close()
    return run_dir
