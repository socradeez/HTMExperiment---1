
import os
import pandas as pd
import matplotlib.pyplot as plt

def _ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def plot_single_metric_figures(csv_path: str, outdir: str):
    _ensure_dir(outdir)
    df = pd.read_csv(csv_path)
    idx = df["step"] if "step" in df.columns else range(len(df))

    for col in ["active_cells", "predicted_cells", "tp", "fp", "fn",
                "precision", "recall", "f1",
                "sparsity_cells", "sparsity_columns",
                "stability_jaccard_last", "stability_jaccard_ema",
                "stability_best_window", "stability_worst_window",
                "bursting_columns"]:
        if col not in df.columns: continue
        plt.figure()
        plt.plot(idx, df[col])
        plt.xlabel("step")
        plt.ylabel(col)
        plt.title(col)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{col}.png"))
        plt.close()

def plot_dashboard(csv_path: str, outpath: str):
    df = pd.read_csv(csv_path)
    idx = df["step"] if "step" in df.columns else range(len(df))

    metrics = [
        ("active vs predicted", ["active_cells", "predicted_cells", "tp"]),
        ("errors", ["fp", "fn"]),
        ("quality", ["precision", "recall", "f1"]),
        ("stability", ["stability_jaccard_last", "stability_jaccard_ema"]),
    ]
    base = os.path.splitext(outpath)[0]
    for title, cols in metrics:
        plt.figure()
        for c in cols:
            if c in df.columns:
                plt.plot(idx, df[c], label=c)
        plt.xlabel("step")
        plt.ylabel(title)
        plt.legend()
        plt.title(title)
        plt.tight_layout()
        plt.savefig(f"{base}_{title.replace(' ', '_')}.png")
        plt.close()
