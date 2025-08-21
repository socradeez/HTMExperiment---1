
# HTM NumPy Starter (single layer)

Quickstart:
```bash
cd htm_np
python run.py
```

Outputs:
- `runs/<timestamp>_starter/metrics.csv` — per-step metrics
- `runs/<timestamp>_starter/indices.npz` — sparse indices (active, predicted_prev, tp/fp/fn)

Tune config in `config.py` / `run.py`. Change sparsity or permanence thresholds easily.
Optional: `pip install torch` and set `RunConfig(backend="torch", device="cuda")` to run the Spatial Pooler on GPU.
