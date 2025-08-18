# BIO Variant Scaffold

## What is this?
A separate experimental path combining subthreshold prediction with within-column inhibition. This package is isolated from the main HTM stack.

## Status
Scaffold only — no activation or learning logic yet.

## Run
```bash
python run_bio.py --dry-run
```
This performs a dry run and writes configs/metrics to a fresh directory under `runs/bio/`.

## Next steps
Implement distal bias, column inhibition, then hook learning and metaplastic gating.
