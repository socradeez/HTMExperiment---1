
import os, json
import pandas as pd
import matplotlib.pyplot as plt

METRIC_NOTES = {
    "active_cells": "|A_t|",
    "predicted_cells": "|Pred_{t-1}|",
    "tp": "|A_t ∩ Pred_{t-1}|",
    "fp": "|Pred_{t-1} \\ A_t|",
    "fn": "|A_t \\ Pred_{t-1}|",
    "precision": "tp / (tp + fp)\n(0 if denom = 0)",
    "recall": "tp / (tp + fn)\n(0 if denom = 0)",
    "f1": "2 * precision * recall\n/ (precision + recall)\n(0 if denom = 0)",
    "active_columns": "|C_t|",
    "bursting_columns": "|C_t \\ PredCols_{t-1}|,\nPredCols_{t-1} = {col(c): c ∈ Pred_{t-1}}",
    "sparsity_cells": "|A_t| / {N_cells}",
    "sparsity_columns": "|C_t| / {N_cols}",
    "stability_overlap_last": "|A_t ∩ A_last(x)|",
    "stability_jaccard_last": "J(A_t, A_last(x))",
    "stability_jaccard_ema": "J(A_t, EMA_x)",
    "stability_best_window": "max_{i∈last W} J(A_t, A_prev^i(x))",
    "stability_worst_window": "min_{i∈last W} J(A_t, A_prev^i(x))",
    "converged": "1 if J(A_t, EMA_x) ≥ τ\nfor M sightings\n(τ={tau}, M={M})",
    "overlap_last_cells": "|A_t ∩ A_last(x)|",
    "diff_last_cells": "|A_t ⊕ A_last(x)|",
    "overlap_last_cols": "|C_t ∩ C_last(x)|",
    "diff_last_cols": "|C_t ⊕ C_last(x)|",
}

def annotate(ax, metric_name, model_cfg_dict, run_cfg_dict):
    note = METRIC_NOTES.get(metric_name)
    if not note:
        return
    fmt = {}
    if model_cfg_dict:
        n_cols = model_cfg_dict.get("num_columns")
        cpc = model_cfg_dict.get("cells_per_column")
        if n_cols is not None:
            fmt["N_cols"] = n_cols
        if n_cols is not None and cpc is not None:
            fmt["N_cells"] = n_cols * cpc
    if run_cfg_dict:
        tau = run_cfg_dict.get("convergence_tau")
        M = run_cfg_dict.get("convergence_M")
        if tau is not None:
            fmt["tau"] = tau
        if M is not None:
            fmt["M"] = M
    try:
        note = note.format(**fmt)
    except Exception:
        pass
    y = 0.99 - 0.12 * len(ax.texts)
    ax.text(
        0.01,
        y,
        note,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox={"boxstyle": "round", "alpha": 0.2},
    )

def _ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def plot_single_metric_figures(
    csv_path: str,
    outdir: str,
    annotate_formulas: bool = False,
    model_cfg: dict | None = None,
    run_cfg: dict | None = None,
):
    _ensure_dir(outdir)
    df = pd.read_csv(csv_path)
    idx = df["step"] if "step" in df.columns else range(len(df))

    cfg_dir = os.path.dirname(csv_path)
    if model_cfg is None:
        try:
            with open(os.path.join(cfg_dir, "config_model.json")) as f:
                model_cfg = json.load(f)
        except Exception:
            model_cfg = None
    if run_cfg is None:
        try:
            with open(os.path.join(cfg_dir, "config_run.json")) as f:
                run_cfg = json.load(f)
        except Exception:
            run_cfg = None

    cols = [
        "active_cells",
        "predicted_cells",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
        "sparsity_cells",
        "sparsity_columns",
        "stability_jaccard_last",
        "stability_jaccard_ema",
        "stability_best_window",
        "stability_worst_window",
        "bursting_columns",
    ]
    for col in cols:
        if col not in df.columns:
            continue
        plt.figure()
        ax = plt.gca()
        ax.plot(idx, df[col])
        ax.set_xlabel("step")
        ax.set_ylabel(col)
        ax.set_title(col)
        if annotate_formulas:
            annotate(ax, col, model_cfg, run_cfg)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{col}.png"))
        plt.close()

def plot_dashboard(
    csv_path: str,
    outpath: str,
    annotate_formulas: bool = False,
    model_cfg: dict | None = None,
    run_cfg: dict | None = None,
):
    df = pd.read_csv(csv_path)
    idx = df["step"] if "step" in df.columns else range(len(df))

    cfg_dir = os.path.dirname(csv_path)
    if model_cfg is None:
        try:
            with open(os.path.join(cfg_dir, "config_model.json")) as f:
                model_cfg = json.load(f)
        except Exception:
            model_cfg = None
    if run_cfg is None:
        try:
            with open(os.path.join(cfg_dir, "config_run.json")) as f:
                run_cfg = json.load(f)
        except Exception:
            run_cfg = None

    metrics = [
        ("active vs predicted", ["active_cells", "predicted_cells", "tp"]),
        ("errors", ["fp", "fn"]),
        ("quality", ["precision", "recall", "f1"]),
        ("stability", ["stability_jaccard_last", "stability_jaccard_ema"]),
    ]
    base = os.path.splitext(outpath)[0]
    for title, cols in metrics:
        plt.figure()
        ax = plt.gca()
        for c in cols:
            if c in df.columns:
                ax.plot(idx, df[c], label=c)
                if annotate_formulas:
                    annotate(ax, c, model_cfg, run_cfg)
        ax.set_xlabel("step")
        ax.set_ylabel(title)
        ax.legend()
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(f"{base}_{title.replace(' ', '_')}.png")
        plt.close()


def plot_per_input_phasefold(csv_path: str, outdir: str, what: str):
    _ensure_dir(outdir)
    df = pd.read_csv(csv_path)
    if "input_id" not in df.columns or "step" not in df.columns:
        return
    df = df.sort_values("step")
    df["occurrence"] = df.groupby("input_id").cumcount() + 1
    if what == "cells":
        cols = {
            "overlap": "overlap_last_cells",
            "diff": "diff_last_cells",
        }
    else:
        cols = {
            "overlap": "overlap_last_cols",
            "diff": "diff_last_cols",
        }
    inputs = df["input_id"].unique()
    for kind, col in cols.items():
        if col not in df.columns:
            continue
        plt.figure()
        ax = plt.gca()
        for inp in inputs:
            sub = df[df["input_id"] == inp]
            ax.plot(sub["occurrence"], sub[col], label=inp)
        ax.set_xlabel("occurrence (per input)")
        ax.set_ylabel(kind)
        ax.set_title(f"per-input {kind} to last ({what})")
        ax.legend()
        plt.tight_layout()
        fname = f"per_input_{kind}_last_{what}.png"
        plt.savefig(os.path.join(outdir, fname))
        plt.close()
