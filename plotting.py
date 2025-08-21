import os
import pandas as pd
import matplotlib.pyplot as plt


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

    if "encoding_diff" in df.columns and "sequence_id" in df.columns:
        plt.figure()
        for seq_id, sub in df.dropna(subset=["encoding_diff"]).groupby("sequence_id"):
            plt.plot(sub["step"], sub["encoding_diff"], label=seq_id)
        plt.xlabel("step")
        plt.ylabel("encoding diff to prev")
        plt.title("Encoding similarity")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "encoding_similarity.png"))
        plt.close()


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

    if any_col("encoding_diff") and any_col("sequence_id"):
        plt.figure()
        for df, label in zip(dfs, kept_labels):
            if "encoding_diff" not in df.columns or "sequence_id" not in df.columns:
                continue
            for seq_id, sub in df.dropna(subset=["encoding_diff"]).groupby("sequence_id"):
                plt.plot(sub["step"], sub["encoding_diff"], label=f"{label}-{seq_id}")
        plt.xlabel("step")
        plt.ylabel("encoding diff to prev")
        plt.title("Encoding similarity (all runs)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "encoding_similarity.png"))
        plt.close()

