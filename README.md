
# HTM NumPy Starter (single layer)

Quickstart:
```bash
cd htm_np
python run.py
```

Outputs:
- `runs/<timestamp>_starter/metrics.csv` — per-step metrics
- `runs/<timestamp>_starter/indices.npz` — sparse indices (active, predicted_prev, tp/fp/fn)
- `runs/<timestamp>_starter/plots/` — PNGs (one figure per metric by default)

Tune config in `config.py` / `run.py`. Change sparsity, permanence thresholds, or figure mode easily.
Optional: `pip install torch` and set `RunConfig(backend="torch", device="cuda")` to run the Spatial Pooler on GPU.

## Experimental
See [README_BIO.md](README_BIO.md) for the separate BIO variant (scaffold only).
