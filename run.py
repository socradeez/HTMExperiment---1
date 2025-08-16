
import os, json
import numpy as np
from typing import Dict, List, Set
from datetime import datetime
from itertools import combinations
from dataclasses import asdict

from config import ModelConfig, RunConfig, json_dumps
from metrics import MetricsCollector
from metaplasticity import MetaParams
from plotting import (
    plot_single_metric_figures,
    plot_dashboard,
    plot_per_input_phasefold,
)
from htm_core import SpatialPooler, TemporalMemory, seeded_rng, active_cells_prev_global

def make_run_dir(base: str, run_name: str) -> str:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    d = os.path.join(base, f"{ts}_{run_name}")
    os.makedirs(d, exist_ok=True)
    return d

def random_sdr_indices(rng: np.random.Generator, size: int, on_bits: int) -> np.ndarray:
    return rng.choice(size, size=on_bits, replace=False)

def flip_bits(rng: np.random.Generator, sdr: np.ndarray, size: int, flip: int) -> np.ndarray:
    if flip <= 0: return sdr
    sdr_set = set(sdr.tolist())
    offs = set(range(size)) - sdr_set
    turn_off = rng.choice(np.array(list(sdr_set)), size=min(flip, len(sdr_set)), replace=False)
    turn_on  = rng.choice(np.array(list(offs)), size=min(flip, len(offs)), replace=False)
    out = sdr_set - set(turn_off.tolist())
    out |= set(turn_on.tolist())
    return np.array(sorted(out), dtype=np.int32)

def build_inputs(rng: np.random.Generator, cfg: RunConfig, model_cfg: ModelConfig) -> Dict[str, np.ndarray]:
    tokens = cfg.sequence.split(cfg.sequence_delimiter)
    mapping = {}
    for t in tokens:
        idx = random_sdr_indices(rng, size=model_cfg.input_size, on_bits=cfg.sdr_on_bits)
        mapping[t] = idx
    return mapping

def sdr_to_dense(idx: np.ndarray, size: int) -> np.ndarray:
    arr = np.zeros(size, dtype=np.uint8); arr[idx] = 1; return arr

def main(model_cfg: ModelConfig, run_cfg: RunConfig):
    rng = seeded_rng(run_cfg.seed)
    run_name = run_cfg.run_name or "htm_np"
    outdir = make_run_dir(run_cfg.output_dir, run_name)
    with open(os.path.join(outdir, "config_model.json"), "w") as f:
        f.write(json_dumps(asdict(model_cfg)))
    with open(os.path.join(outdir, "config_run.json"), "w") as f:
        f.write(json_dumps(asdict(run_cfg)))

    sp = SpatialPooler.create(model_cfg, rng)
    tm = TemporalMemory.create(model_cfg, rng)

    token_map = build_inputs(rng, run_cfg, model_cfg)
    tokens = run_cfg.sequence.split(run_cfg.sequence_delimiter)
    sdr_sets = {t: set(map(int, token_map[t])) for t in token_map}
    sdr_similarity = {}
    for a, b in combinations(token_map.keys(), 2):
        sa, sb = sdr_sets[a], sdr_sets[b]
        overlap = len(sa & sb)
        union = len(sa | sb)
        jacc = overlap / union if union else 1.0
        sdr_similarity[f"{a}-{b}"] = {"overlap": overlap, "jaccard": jacc}
    with open(os.path.join(outdir, "input_sdr_similarity.json"), "w") as f:
        json.dump(sdr_similarity, f, indent=2)
    metrics = MetricsCollector(
        num_cells=model_cfg.num_columns * model_cfg.cells_per_column,
        cells_per_column=model_cfg.cells_per_column,
        output_dir=outdir,
        run_name=run_name,
        ema_threshold=run_cfg.ema_threshold,
        stability_window=run_cfg.stability_window,
        convergence_tau=run_cfg.convergence_tau,
        convergence_M=run_cfg.convergence_M,
    )

    predicted_prev: Set[int] = set()
    step = 0
    pos = 0
    seq_id = "seq0"

    global active_cells_prev_global
    active_cells_prev_global = set()

    while step < run_cfg.steps:
        tok = tokens[pos % len(tokens)]
        idx = token_map[tok]
        idx = flip_bits(rng, idx, model_cfg.input_size, run_cfg.input_flip_bits)
        dense_inp = sdr_to_dense(idx, model_cfg.input_size)

        overlaps = sp.compute_overlap(dense_inp)
        sorted_overlaps = np.sort(overlaps)[::-1]
        k = model_cfg.k_active_columns
        kth_overlap = float(sorted_overlaps[k-1]) if k-1 < sorted_overlaps.size else 0.0
        kplus1_overlap = float(sorted_overlaps[k]) if k < sorted_overlaps.size else 0.0
        k_margin = kth_overlap - kplus1_overlap
        active_cols = sp.k_wta(overlaps)
        active_cols_set = set(active_cols.tolist())
        sp_connected_mean = None
        sp_near_thr_frac = None
        if active_cols.size > 0:
            perms = sp.proximal_perm[active_cols]
            connected = perms >= model_cfg.perm_connected
            sp_connected_mean = float(connected.sum(axis=1).mean())
            near_thr = np.abs(perms - model_cfg.perm_connected) <= run_cfg.sp_near_threshold_eps
            sp_near_thr_frac = float(near_thr.sum() / perms.size)

        predictive_cells = tm.compute_predictive_cells(active_cells_prev_global)
        active_cells, active_segments = tm.activate_cells(active_cols, predictive_cells)

        metrics.seen_in_run[tok] += 1
        metrics.seen_global[tok] += 1
        metrics.log_step(
            step=step,
            sequence_id=seq_id,
            pos_in_seq=(pos % len(tokens)),
            inp_id=tok,
            input_seen_in_run=metrics.seen_in_run[tok],
            input_seen_global=metrics.seen_global[tok],
            active_cells=active_cells,
            active_columns=active_cols_set,
            predicted_prev=predicted_prev,
            kth_overlap=kth_overlap,
            kplus1_overlap=kplus1_overlap,
            k_margin=k_margin,
            sp_connected_mean=sp_connected_mean,
            sp_near_thr_frac=sp_near_thr_frac,
        )

        if run_cfg.learn:
            sp.learn(dense_inp, active_cols)
            tm.learn(active_cells_prev_global, active_cols, active_cells, active_segments, predictive_cells)

        predicted_prev = tm.compute_predictive_cells(active_cells)
        active_cells_prev_global = set(active_cells)

        step += 1
        pos += 1

    saved = metrics.finalize()
    plots_dir = os.path.join(outdir, "plots"); os.makedirs(plots_dir, exist_ok=True)
    if run_cfg.figure_mode == "single":
        plot_single_metric_figures(
            saved["csv"], plots_dir, annotate_formulas=run_cfg.annotate_formulas
        )
    elif run_cfg.figure_mode == "dashboard":
        plot_dashboard(
            saved["csv"],
            os.path.join(plots_dir, "dashboard.png"),
            annotate_formulas=run_cfg.annotate_formulas,
        )
    else:
        plot_single_metric_figures(
            saved["csv"], plots_dir, annotate_formulas=run_cfg.annotate_formulas
        )

    if run_cfg.per_input_plots_cells:
        plot_per_input_phasefold(saved["csv"], plots_dir, what="cells")
    if run_cfg.per_input_plots_columns:
        plot_per_input_phasefold(saved["csv"], plots_dir, what="columns")

    if run_cfg.diagnostics_print:
        try:
            import pandas as pd
        except Exception as e:
            print("Diagnostics skipped: pandas not available", e)
        else:
            df = pd.read_csv(saved["csv"])
            idx = np.load(saved["npz"], allow_pickle=True)
            act_cols = [set(arr.tolist()) for arr in idx["active_columns"]]
            reuse_prev = []
            reuse_jacc = []
            prev = set()
            for cur in act_cols:
                inter = len(cur & prev)
                reuse_prev.append(inter)
                union = len(cur | prev)
                reuse_jacc.append(inter / union if union else 0.0)
                prev = cur
            df["reuse_prev"] = reuse_prev
            df["reuse_prev_jacc"] = reuse_jacc
            token_by_pos = {i: tokens[i] for i in range(len(tokens))}

            print("=== Diagnostics Summary ===")
            print("Input SDR similarity (overlap/jaccard):")
            for pair, stats in sdr_similarity.items():
                print(f"{pair}: {stats['overlap']} / {stats['jaccard']:.3f}")

            grp = df.groupby("pos_in_seq")
            km_mean = grp["k_margin"].mean()
            km_std = grp["k_margin"].std()
            km_p10 = grp["k_margin"].quantile(0.1)
            km_med = grp["k_margin"].median()
            print("\nPer-input k-WTA margin (mean±std, p10):")
            for pos in km_mean.index:
                tok = token_by_pos.get(pos, str(pos))
                print(f"{tok}: {km_mean[pos]:.3f}±{km_std[pos]:.3f}, p10={km_p10[pos]:.3f}")
            low_p10 = km_p10.idxmin()
            low_med = km_med.idxmin()
            print(f"Lowest p10 margin: {token_by_pos.get(low_p10, low_p10)}")
            print(f"Lowest median margin: {token_by_pos.get(low_med, low_med)}")

            reuse_mean = grp["reuse_prev"].mean()
            print("\nCross-input column reuse (mean reuse count):")
            for pos in reuse_mean.index:
                tok = token_by_pos.get(pos, str(pos))
                print(f"{tok}: {reuse_mean[pos]:.3f}")

            sp_conn = grp["sp_connected_mean"].mean()
            sp_near = grp["sp_near_thr_frac"].mean()
            print("\nSP health on active columns (mean connected, near-thr frac):")
            for pos in sp_conn.index:
                tok = token_by_pos.get(pos, str(pos))
                print(f"{tok}: {sp_conn[pos]:.3f}, {sp_near[pos]:.3f}")

            pred_stats = grp[["predicted_cells", "bursting_columns", "precision", "recall"]].mean()
            print("\nPrediction health by input (means):")
            for pos, row in pred_stats.iterrows():
                tok = token_by_pos.get(pos, str(pos))
                print(
                    f"{tok}: pred={row['predicted_cells']:.3f}, burst={row['bursting_columns']:.3f}, "
                    f"prec={row['precision']:.3f}, rec={row['recall']:.3f}"
                )
    if model_cfg.meta.enabled:
        print(
            f"Metaplasticity: rungs={model_cfg.meta.rungs}, "
            f"decay_beta={model_cfg.meta.decay_beta}, decay_floor={model_cfg.meta.decay_floor}"
        )
    print("Run complete. Outputs in:", outdir)
    return outdir

if __name__ == "__main__":
    model_cfg = ModelConfig(
        input_size=1024,
        num_columns=2048,
        cells_per_column=10,
        k_active_columns=40,
        synapses_per_column=64,
        perm_connected=0.25,
        init_perm_mean=0.26,
        init_perm_sd=0.02,
        perm_inc=0.03,
        meta=MetaParams(enabled=False),
        perm_dec=0.015,
        distal_synapses_per_segment=20,
        segment_activation_threshold=10,
        new_segment_init_perm_mean=0.26,
        new_segment_init_perm_sd=0.02,
    )
    run_cfg = RunConfig(
        seed=7,
        steps=200,
        learn=True,
        figure_mode="single",
        output_dir="runs",
        sdr_on_bits=20,
        sequence="A>B>C>D",
        stability_window=50,
        ema_threshold=0.5,
        convergence_tau=0.9,
        convergence_M=3,
        input_flip_bits=0,
        run_name="starter"
    )
    main(model_cfg, run_cfg)
