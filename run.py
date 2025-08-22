
import os, json
import numpy as np
from typing import Dict, List, Set, Optional
from datetime import datetime
from itertools import combinations
from dataclasses import asdict
import time
from collections import defaultdict

from config import ModelConfig, RunConfig, json_dumps
from metrics import MetricsCollector
from htm_core import SpatialPooler, TemporalMemory, seeded_rng

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

def build_inputs(
    rng: np.random.Generator,
    cfg: RunConfig,
    model_cfg: ModelConfig,
    tokens_unique: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    tokens = tokens_unique if tokens_unique is not None else cfg.sequence.split(cfg.sequence_delimiter)
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

    tokens_unique: Optional[List[str]] = None
    if run_cfg.explicit_step_tokens is not None:
        tokens_unique = sorted(set(run_cfg.explicit_step_tokens))
        run_cfg.steps = len(run_cfg.explicit_step_tokens)

    with open(os.path.join(outdir, "config_model.json"), "w") as f:
        f.write(json_dumps(asdict(model_cfg)))
    run_dict = asdict(run_cfg)
    if run_dict.get("explicit_step_tokens") is not None:
        run_dict["explicit_step_tokens_len"] = len(run_dict["explicit_step_tokens"])
        del run_dict["explicit_step_tokens"]
    with open(os.path.join(outdir, "config_run.json"), "w") as f:
        f.write(json_dumps(run_dict))

    if run_cfg.backend == "torch":
        from torch_backend import make_sp_torch, make_tm_torch
        import torch
        sp = make_sp_torch(model_cfg, run_cfg.seed, run_cfg.device)
        device = sp.device
        tm = make_tm_torch(model_cfg, run_cfg.seed, run_cfg.device)
        print(f"Backend: torch (device={device})")
    else:
        sp = SpatialPooler.create(model_cfg, rng)
        device = None
        tm = TemporalMemory.create(model_cfg, rng)
        print("Backend: numpy")

    token_map = build_inputs(rng, run_cfg, model_cfg, tokens_unique)
    tokens = tokens_unique if tokens_unique is not None else run_cfg.sequence.split(run_cfg.sequence_delimiter)
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
        overconfident_window=run_cfg.overconfident_window,
    )

    predicted_prev: Set[int] = set()
    step = 0
    pos = 0

    t_start = time.time()
    active_cells_prev: Set[int] = set()
    prev_dense_by_token: Dict[str, np.ndarray] = {}

    while step < run_cfg.steps:
        if run_cfg.explicit_step_tokens is not None:
            tok = run_cfg.explicit_step_tokens[step]
            pos_in_seq = (
                run_cfg.token_pos_map.get(tok, -1)
                if run_cfg.token_pos_map
                else (pos % len(tokens))
            )
            seq_id = tok.split("_")[0]
        else:
            tok = tokens[pos % len(tokens)]
            pos_in_seq = pos % len(tokens)
            seq_id = "seq0"
        idx = token_map[tok]
        idx = flip_bits(rng, idx, model_cfg.input_size, run_cfg.input_flip_bits)
        dense_inp = sdr_to_dense(idx, model_cfg.input_size)
        prev_dense = prev_dense_by_token.get(tok)
        if prev_dense is not None:
            overlap = np.logical_and(prev_dense, dense_inp).sum()
            union = np.logical_or(prev_dense, dense_inp).sum()
            encoding_diff = 1 - (overlap / union) if union else 0.0
        else:
            encoding_diff = 0.0
        prev_dense_by_token[tok] = dense_inp.copy()

        if run_cfg.backend == "torch":
            x_bool = torch.from_numpy(dense_inp).to(device).bool()
            overlaps_t = sp.compute_overlap(x_bool)
            sorted_overlaps_t = torch.sort(overlaps_t, descending=True)[0]
            k = model_cfg.k_active_columns
            kth_overlap = float(sorted_overlaps_t[k-1].item()) if k-1 < sorted_overlaps_t.numel() else 0.0
            kplus1_overlap = float(sorted_overlaps_t[k].item()) if k < sorted_overlaps_t.numel() else 0.0
            k_margin = kth_overlap - kplus1_overlap
            active_cols_t = sp.k_wta(overlaps_t, model_cfg.k_active_columns)
            active_cols = active_cols_t.cpu().numpy()
            active_cols_set = set(active_cols.tolist())
            sp_connected_mean = None
            sp_near_thr_frac = None
            if active_cols_t.numel() > 0:
                perms = sp.proximal_perm[active_cols_t]
                connected = perms >= model_cfg.perm_connected
                sp_connected_mean = float(connected.sum(dim=1).float().mean().item())
                near_thr = torch.abs(perms - model_cfg.perm_connected) <= run_cfg.sp_near_threshold_eps
                sp_near_thr_frac = float(near_thr.float().sum().item() / perms.numel())
            pred_cells_t, _, active_segments_t = tm.predict_from_set(active_cells_prev)
            active_cells_t = tm.activate_cells(active_cols_t, pred_cells_t)
            active_cells = set(active_cells_t.cpu().numpy().tolist())
        else:
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
            predictive_cells = tm.compute_predictive_cells(active_cells_prev)
            active_cells, active_segments = tm.activate_cells(active_cols, predictive_cells, active_cells_prev)

        cpc = model_cfg.cells_per_column
        tp_cells = predicted_prev & active_cells
        hit_cols = {cell // cpc for cell in tp_cells}
        burst_cols = active_cols_set - hit_cols
        pred_col_sizes = defaultdict(int)
        for cell in predicted_prev:
            pred_col_sizes[cell // cpc] += 1
        narrow_cells_prev = {cell for cell in predicted_prev if pred_col_sizes[cell // cpc] <= 2}
        narrow_hit_cells = narrow_cells_prev & active_cells
        overconfident_rate = 0.0
        if narrow_cells_prev:
            overconfident_rate = (
                len(narrow_cells_prev - active_cells) / len(narrow_cells_prev)
            )
        surprise_mean = len(burst_cols) / len(active_cols_set) if active_cols_set else 0.0
        prediction_accuracy = 1.0 - surprise_mean if active_cols_set else 0.0
        spread_v1 = (
            sum(pred_col_sizes.values()) / len(pred_col_sizes)
            if pred_col_sizes
            else 0.0
        )
        active_pred_cols = set(pred_col_sizes.keys()) & active_cols_set
        spread_v2 = (
            sum(pred_col_sizes[c] for c in active_pred_cols) / len(active_pred_cols)
            if active_pred_cols
            else 0.0
        )
        segments = tm.num_segments + len(tm.pending_owner)
        synapses = tm.perm_values.numel() + len(tm.pending_perm)

        metrics.seen_in_run[tok] += 1
        metrics.seen_global[tok] += 1
        metrics.log_step(
            step=step,
            sequence_id=seq_id,
            pos_in_seq=pos_in_seq,
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
            surprise_mean=surprise_mean,
            surprise_count=len(burst_cols),
            spread_v1=spread_v1,
            spread_v2=spread_v2,
            overconfident_rate=overconfident_rate,
            prediction_accuracy=prediction_accuracy,
            segments=segments,
            synapses=synapses,
            encoding_diff=encoding_diff,
            burst_cols=burst_cols,
            predicted_col_sizes=pred_col_sizes,
            covered_cols=hit_cols,
            narrow_cells_prev=narrow_cells_prev,
            narrow_hit_cells=narrow_hit_cells,
        )

        if run_cfg.learn:
            if run_cfg.backend == "torch":
                sp.learn(x_bool, active_cols_t)
                tm.learn(active_cells_prev, active_cols_t, active_cells_t, active_segments_t, pred_cells_t)
            else:
                sp.learn(dense_inp, active_cols)
                tm.learn(active_cells_prev, active_cols, active_cells, active_segments, predictive_cells)

        if run_cfg.backend == "torch":
            pred_next_t, _, _ = tm.predict_from_set(active_cells)
            predicted_prev = set(pred_next_t.cpu().numpy().tolist())
        else:
            predicted_prev = tm.compute_predictive_cells(active_cells)
        active_cells_prev = set(active_cells)

        step += 1
        pos += 1

    if run_cfg.backend == "torch":
        tm.flush_pending()
        elapsed = time.time() - t_start
        steps_per_sec = run_cfg.steps / elapsed if elapsed > 0 else float('inf')
        print(f"Timing: {steps_per_sec:.2f} steps/sec, nnz(P)={sp.perm_values.numel()}, nnz(M)={tm.perm_values.numel()}")

    saved = metrics.finalize()

    if run_cfg.plots:
        from plotting import PLOTTERS
        plot_dir = os.path.join(outdir, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        for name in run_cfg.plots:
            fn = PLOTTERS.get(name)
            if fn:
                fn(saved["csv"], plot_dir)
            else:
                print(f"Unknown plot bundle: {name}")

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
    print("Run complete. Outputs in:", outdir)
    return outdir

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--plots", nargs="*", default=None, help="Plot bundles to generate")
    args = parser.parse_args()

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
        output_dir="runs",
        sdr_on_bits=20,
        sequence="A>B>C>D",
        stability_window=50,
        ema_threshold=0.5,
        convergence_tau=0.9,
        convergence_M=3,
        input_flip_bits=0,
        run_name="starter",
        backend="torch",
        plots=args.plots,
    )
    main(model_cfg, run_cfg)
