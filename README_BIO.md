# BIO Variant

## What is this?
An experimental path combining subthreshold prediction with within-column inhibition. It reuses the Torch backend but lives in an isolated package so the baseline stack remains untouched.

## Status
Subthreshold distal bias lowers activation thresholds and a within-column inhibitor picks sparse winners. A Torch sparse TM manages segments and can optionally employ metaplastic gating (disabled by default).

## Run
```bash
python run_bio.py --steps 20
```
Artifacts (configs and `metrics_bio.csv`) are written under `runs/bio/`.

## Quick sweep
Run a small grid and produce baseline-style PNGs:

```bash
python -m htm_bio.sweep_bio_quick --device cuda --occurrences 40 --seeds 2 --plots
```

Runs land under `runs/bio_quick/` and include `metrics.csv` with familiar figures.

## Next steps
Refine bias/inhibition models and compare metrics against the baseline implementation.
