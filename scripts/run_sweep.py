import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import argparse
from htm.sweeps import run_hardening_sweep

def _parse_float_list(value: str):
    return [float(v) for v in value.split(',') if v]

def _parse_int_list(value: str):
    return [int(v) for v in value.split(',') if v]

def main():
    parser = argparse.ArgumentParser(description='Run hardening parameter sweep')
    parser.add_argument('--rates', type=str, default='0.0,0.02,0.05,0.1,0.2', help='Comma-separated hardening rates')
    parser.add_argument('--thresholds', type=str, default='0.55,0.6,0.65,0.7', help='Comma-separated hardening thresholds')
    parser.add_argument('--seeds', type=str, default='0,1,2', help='Comma-separated random seeds')
    parser.add_argument('--epochs-per-phase', type=int, default=25, help='Training epochs for each phase')
    parser.add_argument('--outdir', type=str, default='sweep_results', help='Output directory')
    args = parser.parse_args()

    rates = _parse_float_list(args.rates)
    thresholds = _parse_float_list(args.thresholds)
    seeds = _parse_int_list(args.seeds)

    run_hardening_sweep(rates=rates, thresholds=thresholds, seeds=seeds, epochs_per_phase=args.epochs_per_phase, outdir=args.outdir)

if __name__ == '__main__':
    main()
