# HTM NumPy Starter (single layer)

Quickstart:
```bash
cd htm_np
python run.py --plots baseline_meta
```

Outputs:
- `runs/<timestamp>_starter/metrics.csv` — per-step metrics
- `runs/<timestamp>_starter/indices.npz` — sparse indices (active, predicted_prev, tp/fp/fn)
- `runs/<timestamp>_starter/plots/` — baseline instrumentation charts, including
  encoding stability per sequence step (cycle vs. Jaccard distance to the prior
  occurrence of that input) and active vs. predicted neuron counts per timestep

Tune config in `config.py` / `run.py`. Change sparsity or permanence thresholds easily.
Optional: `pip install torch` and set `RunConfig(backend="torch", device="cuda")` to run the Spatial Pooler on GPU.

Sweeps:
```bash
python htm_baseline_sweep.py --repetitions 100 --num_sequences 4,8,16 --seq_length 4,8,16 --overlap 0,25,50 --plots baseline_meta --seeds 2
```
Collects runs over combinations of sequence counts, sequence lengths, and overlaps, writing a
`baseline_sweep_summary.csv` alongside the individual run directories. When
`--plots baseline_meta` is provided, a `plots/` folder is also created under the
sweep output, showing each baseline plot with lines for every run layered
together.
