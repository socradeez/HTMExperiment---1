# BIO Variant

## What is this?
An experimental path combining subthreshold prediction with within-column inhibition. It reuses the Torch backend but lives in an isolated package so the baseline stack remains untouched.

## Status
Basic activation and learning loop implemented. Distal bias lowers activation thresholds and a simple inhibitory competition selects winners. Metaplastic gating is available via existing parameters but disabled by default.

## Run
```bash
python run_bio.py --steps 20
```
Run artifacts (configs and `metrics_bio.csv`) are written under `runs/bio/`.

## Next steps
Refine bias/inhibition models and compare metrics against the baseline implementation.
