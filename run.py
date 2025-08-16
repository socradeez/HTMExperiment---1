
import os, json
import numpy as np
from typing import Dict, List, Set
from datetime import datetime

from config import ModelConfig, RunConfig, json_dumps
from metrics import MetricsCollector
from plotting import plot_single_metric_figures, plot_dashboard
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
        f.write(json_dumps(model_cfg.__dict__))
    with open(os.path.join(outdir, "config_run.json"), "w") as f:
        f.write(json_dumps(run_cfg.__dict__))

    sp = SpatialPooler.create(model_cfg, rng)
    tm = TemporalMemory.create(model_cfg, rng)

    token_map = build_inputs(rng, run_cfg, model_cfg)
    tokens = run_cfg.sequence.split(run_cfg.sequence_delimiter)
    metrics = MetricsCollector(
        num_cells=model_cfg.num_columns*model_cfg.cells_per_column,
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
        active_cols = sp.k_wta(overlaps)
        active_cols_set = set(active_cols.tolist())

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
        )

        if run_cfg.learn:
            sp.learn(dense_inp, active_cols)
            tm.learn(active_cells_prev_global, active_cols, active_cells, active_segments)

        predicted_prev = tm.compute_predictive_cells(active_cells)
        active_cells_prev_global = set(active_cells)

        step += 1
        pos += 1

    saved = metrics.finalize()
    plots_dir = os.path.join(outdir, "plots"); os.makedirs(plots_dir, exist_ok=True)
    if run_cfg.figure_mode == "single":
        plot_single_metric_figures(saved["csv"], plots_dir)
    elif run_cfg.figure_mode == "dashboard":
        plot_dashboard(saved["csv"], os.path.join(plots_dir, "dashboard.png"))
    else:
        plot_single_metric_figures(saved["csv"], plots_dir)

    print("Run complete. Outputs in:", outdir)
    return outdir

if __name__ == "__main__":
    model_cfg = ModelConfig(
        input_size=1024,
        num_columns=2048,
        cells_per_column=10,
        k_active_columns=40,
        synapses_per_column=32,
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
