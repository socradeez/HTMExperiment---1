# HTM NumPy Starter (single layer)

This project provides a small NumPy implementation of a single-layer
Hierarchical Temporal Memory (HTM) system along with several sweep scripts for
experiments.

## Quickstart

```bash
python run.py --plots baseline_meta
```

Outputs from `run.py` are written under `runs/<timestamp>_starter/`:

- `metrics.csv` — per-step metrics
- `indices.npz` — sparse indices (active, predicted_prev, tp/fp/fn)
- `plots/` — baseline instrumentation charts, including encoding stability per
  sequence step (cycle vs. Jaccard distance to the prior occurrence of that
  input) and active vs. predicted neuron counts per timestep

Tune config in `config.py` / `run.py`. Change sparsity or permanence thresholds
easily. Optional: `pip install torch` and set `RunConfig(backend="torch",
device="cuda")` to run the Spatial Pooler on GPU.

## Entry points

### `run.py`
Runs a single training sequence. CLI flags:

- `--plots` (zero or more names): plot bundles to generate.

Example:

```bash
python run.py --plots baseline_meta
```

### `htm_baseline_sweep.py`
Collects runs over combinations of sequence counts, sequence lengths, and
overlaps.

Flags:

- `--repetitions INT` (default `1`): occurrences per sequence in each run.
- `--num_sequences N1 N2 ...` (default `1`): list of sequence counts.
- `--seq_length L1 L2 ...` (default `4`): list of sequence lengths.
- `--overlap P1 P2 ...` (default `0`): percentage of overlap between sequences.
- `--plots [PLOT ...]` (optional): plot bundles to generate for each run.
- `--seeds INT` (default `1`): number of random seeds.
- `--out PATH` (default `runs/sweep`): output directory for runs and summary.
- `--backend {numpy,torch}` (default `torch`): backend for SP/TM.

Example:

```bash
python htm_baseline_sweep.py --repetitions 100 --num_sequences 4,8,16 \
  --seq_length 4,8,16 --overlap 0,25,50 --plots baseline_meta --seeds 2
```

### `sweep_capacity.py`
Explores capacity limits across sequence length, count, and overlap.

Flags:

- `--lengths L1,L2,...` (default `4,8,16,32`): comma-separated sequence lengths.
- `--seq-counts S1,S2,...` (default `1,2,4,8`): comma-separated sequence counts.
- `--overlaps O1,O2,...` (default `0,25,50`): comma-separated overlap
  percentages.
- `--schedule {blocked,interleave,blocked_then_interleave}` (default
  `interleave`): presentation schedule for sequences.
- `--occurrences INT` (default `60`): occurrences per sequence.
- `--seeds INT` (default `3`): number of random seeds.
- `--out PATH` (default `runs/capacity`): output directory.

Example:

```bash
python sweep_capacity.py --lengths 8,16 --seq-counts 2,4 --overlaps 0,25 \
  --schedule blocked --occurrences 50 --seeds 2
```

### `continual_sweep.py`
Generates streams with randomly interleaved noise and sequences to test
continual learning.

Flags:

- `--vocab INT` (default `100`): vocabulary size for noise tokens.
- `--steps INT` (default `1000`): total timesteps in each run.
- `--gap_means G1,G2,...` (default `2,6,12,24`): Poisson means for noise gaps.
- `--seq_counts K1,K2,...` (default `1,2,4`): list of sequence counts.
- `--seq_lengths L1,L2,...` (default `4,8,16`): list of sequence lengths.
- `--noise_types TYPE1,TYPE2,...` (default `in_dist`): noise vocab types.
- `--plots [PLOT ...]` (optional): plot bundles to generate for each run.
- `--seeds INT` (default `1`): number of random seeds.
- `--out PATH` (default `runs/continual`): output directory.

Example:

```bash
python continual_sweep.py --vocab 50 --steps 500 --gap_means 6 --seq_counts 2 \
  --seq_lengths 8 --noise_types in_dist,ood --plots baseline_meta \
  --seeds 3 --out runs/continual
```

Each sweep script writes a summary CSV in its output directory alongside the
individual run folders.

