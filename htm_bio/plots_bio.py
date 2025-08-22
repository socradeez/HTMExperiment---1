"""Generate baseline-style plots for BIO runs."""

import os


def make_all(run_dir: str) -> None:
    """Produce starter PNGs from metrics.csv in ``run_dir``."""
    csv_path = os.path.join(run_dir, "metrics.csv")
    try:
        from plotting import plot_single_metric_figures, plot_per_input_phasefold

        plot_single_metric_figures(csv_path, run_dir)
        plot_per_input_phasefold(csv_path, run_dir, "columns")
        plot_per_input_phasefold(csv_path, run_dir, "cells")
    except Exception:
        # Minimal fallback
        import pandas as pd
        import matplotlib.pyplot as plt

        if not os.path.exists(csv_path):
            return
        df = pd.read_csv(csv_path)
        idx = df["t"] if "t" in df.columns else range(len(df))
        basic = ["precision", "recall", "f1", "sparsity_columns", "sparsity_cells", "fp", "fn"]
        for col in basic:
            if col not in df.columns:
                continue
            plt.figure()
            plt.plot(idx, df[col])
            plt.xlabel("t")
            plt.ylabel(col)
            plt.title(col)
            plt.tight_layout()
            plt.savefig(os.path.join(run_dir, f"{col}.png"))
            plt.close()
        if "input_id" in df.columns:
            df["occurrence"] = df.groupby("input_id").cumcount() + 1
            for what in ["columns", "cells"]:
                diff_col = f"diff_last_{what}"
                ov_col = f"overlap_last_{what}"
                for col in [diff_col, ov_col]:
                    if col not in df.columns:
                        continue
                    plt.figure()
                    ax = plt.gca()
                    for inp in df["input_id"].unique():
                        sub = df[df["input_id"] == inp]
                        ax.plot(sub["occurrence"], sub[col], label=inp)
                    ax.set_xlabel("occurrence (per input)")
                    ax.set_ylabel(col.split("_")[0])
                    ax.set_title(f"per-input {col.replace('_last_', ' ')}")
                    ax.legend()
                    plt.tight_layout()
                    fname = f"per_input_{col}.png"
                    plt.savefig(os.path.join(run_dir, fname))
                    plt.close()
