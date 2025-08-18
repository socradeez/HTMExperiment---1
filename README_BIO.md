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

## Next steps
Refine bias/inhibition models and compare metrics against the baseline implementation.
