import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_stability_global(df: pd.DataFrame, outpath: Path) -> None:
    """Global stability: mean stability_jaccard_last per cycle.
    Fallback: normalized stability_overlap_last per cycle."""
    plt.figure(figsize=(7, 4))
    if {"stability_jaccard_last", "cycle"}.issubset(df.columns):
        g = df.groupby("cycle", as_index=False)["stability_jaccard_last"].mean()
        plt.plot(g["cycle"], g["stability_jaccard_last"])
        plt.ylabel("stability_jaccard_last (mean)")
        plt.title("Encoding stability (global)")
    elif {"stability_overlap_last", "cycle", "active_cells"}.issubset(df.columns):
        tmp = df.copy()
        tmp["norm_overlap"] = np.where(
            tmp["active_cells"] > 0,
            tmp["stability_overlap_last"] / tmp["active_cells"],
            np.nan,
        )
        g = tmp.groupby("cycle", as_index=False)["norm_overlap"].mean()
        plt.plot(g["cycle"], g["norm_overlap"])
        plt.ylabel("normalized overlap")
        plt.title("Encoding stability (global, overlap fallback)")
    else:
        raise ValueError(
            "metrics must include stability_jaccard_last+cycle or "
            "stability_overlap_last+active_cells+cycle for global plot"
        )
    plt.xlabel("cycle")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_stability_per_input(df: pd.DataFrame, outpath: Path) -> bool:
    """Per-input stability: stability_jaccard_last mean per (input_id, cycle).
    Returns True if plot was made; False if required columns are missing."""
    req = {"stability_jaccard_last", "cycle", "input_id"}
    if not req.issubset(df.columns):
        return False

    plt.figure(figsize=(8, 5))
    for tok, grp in df.groupby("input_id"):
        agg = grp.groupby("cycle", as_index=False)["stability_jaccard_last"].mean()
        plt.plot(agg["cycle"], agg["stability_jaccard_last"], label=str(tok))
    plt.title("Encoding stability (per input)")
    plt.xlabel("cycle")
    plt.ylabel("stability_jaccard_last")
    plt.legend(ncol=2, fontsize=8, frameon=False)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    return True


def plot_baseline_meta(csv_path: str, outdir: str):
    df = pd.read_csv(csv_path)
    idx = df["step"] if "step" in df.columns else range(len(df))

    if "surprise_mean" in df.columns or "bursting_columns" in df.columns:
        plt.figure()
        if "surprise_mean" in df.columns:
            plt.plot(idx, df["surprise_mean"], label="mean")
        if "bursting_columns" in df.columns:
            plt.plot(idx, df["bursting_columns"], label="count")
        plt.xlabel("step")
        plt.ylabel("surprise")
        plt.title("Surprise")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "surprise.png"))
        plt.close()

    if "spread_v1" in df.columns or "spread_v2" in df.columns:
        plt.figure()
        if "spread_v1" in df.columns:
            plt.plot(idx, df["spread_v1"], label="v1")
        if "spread_v2" in df.columns:
            plt.plot(idx, df["spread_v2"], label="v2")
        plt.xlabel("step")
        plt.ylabel("spread")
        plt.title("Spread")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "spread.png"))
        plt.close()

    if "overconfident_rate" in df.columns:
        plt.figure()
        plt.plot(idx, df["overconfident_rate"])
        plt.xlabel("step")
        plt.ylabel("overconfident rate")
        plt.title("Overconfident rate")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "overconfident_rate.png"))
        plt.close()

    if "prediction_accuracy" in df.columns:
        plt.figure()
        plt.plot(idx, df["prediction_accuracy"])
        plt.xlabel("step")
        plt.ylabel("prediction accuracy")
        plt.title("Prediction accuracy")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "prediction_accuracy.png"))
        plt.close()

    if "segments" in df.columns or "synapses" in df.columns:
        plt.figure()
        if "segments" in df.columns:
            plt.plot(idx, df["segments"], label="segments")
        if "synapses" in df.columns:
            plt.plot(idx, df["synapses"], label="synapses")
        plt.xlabel("step")
        plt.ylabel("count")
        plt.title("Capacity load")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "capacity.png"))
        plt.close()

    plot_stability_global(df, Path(outdir) / "encoding_stability_global.png")
    plot_stability_per_input(df, Path(outdir) / "encoding_stability_per_input.png")


PLOTTERS = {"baseline_meta": plot_baseline_meta}


def plot_baseline_meta_sweep(csv_paths, labels, outdir):
    dfs = []
    idxs = []
    kept_labels = []
    for path, label in zip(csv_paths, labels):
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        dfs.append(df)
        idxs.append(df["step"] if "step" in df.columns else range(len(df)))
        kept_labels.append(label)
    if not dfs:
        return

    def any_col(name):
        return any(name in df.columns for df in dfs)

    if any_col("surprise_mean") or any_col("bursting_columns"):
        plt.figure()
        for df, idx, label in zip(dfs, idxs, kept_labels):
            if "surprise_mean" in df.columns:
                plt.plot(idx, df["surprise_mean"], label=f"{label}-mean")
            if "bursting_columns" in df.columns:
                plt.plot(idx, df["bursting_columns"], label=f"{label}-count")
        plt.xlabel("step")
        plt.ylabel("surprise")
        plt.title("Surprise (all runs)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "surprise.png"))
        plt.close()

    if any_col("spread_v1") or any_col("spread_v2"):
        plt.figure()
        for df, idx, label in zip(dfs, idxs, kept_labels):
            if "spread_v1" in df.columns:
                plt.plot(idx, df["spread_v1"], label=f"{label}-v1")
            if "spread_v2" in df.columns:
                plt.plot(idx, df["spread_v2"], label=f"{label}-v2")
        plt.xlabel("step")
        plt.ylabel("spread")
        plt.title("Spread (all runs)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "spread.png"))
        plt.close()

    if any_col("overconfident_rate"):
        plt.figure()
        for df, idx, label in zip(dfs, idxs, kept_labels):
            plt.plot(idx, df["overconfident_rate"], label=label)
        plt.xlabel("step")
        plt.ylabel("overconfident rate")
        plt.title("Overconfident rate (all runs)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "overconfident_rate.png"))
        plt.close()

    if any_col("prediction_accuracy"):
        plt.figure()
        for df, idx, label in zip(dfs, idxs, kept_labels):
            plt.plot(idx, df["prediction_accuracy"], label=label)
        plt.xlabel("step")
        plt.ylabel("prediction accuracy")
        plt.title("Prediction accuracy (all runs)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "prediction_accuracy.png"))
        plt.close()

    if any_col("segments") or any_col("synapses"):
        plt.figure()
        for df, idx, label in zip(dfs, idxs, kept_labels):
            if "segments" in df.columns:
                plt.plot(idx, df["segments"], label=f"{label}-segments")
            if "synapses" in df.columns:
                plt.plot(idx, df["synapses"], label=f"{label}-synapses")
        plt.xlabel("step")
        plt.ylabel("count")
        plt.title("Capacity load (all runs)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "capacity.png"))
        plt.close()

    if any_col("stability_jaccard_last") and any_col("cycle"):
        plt.figure()
        for df, label in zip(dfs, kept_labels):
            if {"stability_jaccard_last", "cycle"}.issubset(df.columns):
                g = df.groupby("cycle", as_index=False)["stability_jaccard_last"].mean()
                plt.plot(g["cycle"], g["stability_jaccard_last"], label=label)
            elif {"stability_overlap_last", "active_cells", "cycle"}.issubset(df.columns):
                tmp = df.copy()
                tmp["norm_overlap"] = np.where(
                    tmp["active_cells"] > 0,
                    tmp["stability_overlap_last"] / tmp["active_cells"],
                    np.nan,
                )
                g = tmp.groupby("cycle", as_index=False)["norm_overlap"].mean()
                plt.plot(g["cycle"], g["norm_overlap"], label=label)
        plt.xlabel("cycle")
        plt.ylabel("stability_jaccard_last (mean)")
        plt.title("Encoding stability (global, all runs)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "encoding_stability_global.png"))
        plt.close()

    if any_col("stability_jaccard_last") and any_col("input_id") and any_col("cycle"):
        plt.figure()
        for df, label in zip(dfs, kept_labels):
            if not {"stability_jaccard_last", "input_id", "cycle"}.issubset(df.columns):
                continue
            for tok, grp in df.groupby("input_id"):
                agg = grp.groupby("cycle", as_index=False)["stability_jaccard_last"].mean()
                plt.plot(agg["cycle"], agg["stability_jaccard_last"], label=f"{label}-{tok}")
        plt.title("Encoding stability (per input, all runs)")
        plt.xlabel("cycle")
        plt.ylabel("stability_jaccard_last")
        plt.legend(ncol=2, fontsize=8, frameon=False)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "encoding_stability_per_input.png"))
        plt.close()

