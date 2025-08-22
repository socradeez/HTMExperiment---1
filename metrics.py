
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Set, Tuple, Optional
import csv, os, json
from collections import defaultdict, Counter, deque

@dataclass
class StabilityState:
    last_sdr: Dict[str, Set[int]] = field(default_factory=dict)
    last_cols: Dict[str, Set[int]] = field(default_factory=dict)
    counts: Dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    totals: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    history: Dict[str, deque] = field(default_factory=dict)

@dataclass
class MetricsCollector:
    num_cells: int
    cells_per_column: int
    output_dir: str
    run_name: str
    ema_threshold: float = 0.5
    stability_window: int = 50
    convergence_tau: float = 0.9
    convergence_M: int = 3

    overconfident_window: int = 2

    rows: List[Dict] = field(default_factory=list)
    stability: StabilityState = field(default_factory=StabilityState)
    sparse_indices: Dict[str, List[np.ndarray]] = field(default_factory=lambda: defaultdict(list))
    seen_in_run: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    seen_global: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    consecutive_hits: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    uniq_cells_ever: Set[int] = field(default_factory=set)
    uniq_cols_ever: Set[int] = field(default_factory=set)
    narrow_miss_streak: Dict[int, int] = field(default_factory=lambda: defaultdict(int))

    def __post_init__(self):
        self.num_cols = self.num_cells // self.cells_per_column

    def _proto_from_counts(self, inp_id: str) -> Set[int]:
        c = self.stability.counts[inp_id]
        tot = max(1, self.stability.totals[inp_id])
        return {cell for cell, cnt in c.items() if (cnt / tot) >= self.ema_threshold}

    @staticmethod
    def jaccard(a: Set[int], b: Set[int]) -> float:
        if not a and not b:
            return 1.0
        inter = len(a & b)
        union = len(a | b)
        return inter / union if union else 1.0

    def _hist_best_worst(self, inp_id: str, current: Set[int]) -> Tuple[float, float]:
        dq = self.stability.history.get(inp_id, deque(maxlen=self.stability_window))
        if not dq:
            return (0.0, 0.0)
        vals = [self.jaccard(current, prev) for prev in dq]
        return (max(vals), min(vals))

    def log_step(self,
                 step: int,
                 sequence_id: str,
                 pos_in_seq: int,
                 inp_id: str,
                 input_seen_in_run: int,
                 input_seen_global: int,
                 active_cells: Set[int],
                 active_columns: Set[int],
                 predicted_prev: Set[int],
                 kth_overlap: Optional[float] = None,
                 kplus1_overlap: Optional[float] = None,
                 k_margin: Optional[float] = None,
                 sp_connected_mean: Optional[float] = None,
                 sp_near_thr_frac: Optional[float] = None,
                 surprise_mean: Optional[float] = None,
                 surprise_count: Optional[int] = None,
                 spread_v1: Optional[float] = None,
                 spread_v2: Optional[float] = None,
                 overconfident_rate: Optional[float] = None,
                 prediction_accuracy: Optional[float] = None,
                 segments: Optional[int] = None,
                 synapses: Optional[int] = None,
                 encoding_diff: Optional[float] = None,
                 burst_cols: Optional[Set[int]] = None,
                 predicted_col_sizes: Optional[Dict[int, int]] = None,
                 covered_cols: Optional[Set[int]] = None,
                 narrow_cells_prev: Optional[Set[int]] = None,
                 narrow_hit_cells: Optional[Set[int]] = None):
        tp = active_cells & predicted_prev
        fp = predicted_prev - active_cells
        fn = active_cells - predicted_prev

        if burst_cols is None or covered_cols is None or predicted_col_sizes is None:
            col_counts: Dict[int, int] = defaultdict(int)
            for cell in predicted_prev:
                col_counts[cell // self.cells_per_column] += 1
            if predicted_col_sizes is None:
                predicted_col_sizes = col_counts
            hit_cols = {cell // self.cells_per_column for cell in tp}
            if burst_cols is None:
                burst_cols = set(active_columns) - hit_cols
            if covered_cols is None:
                covered_cols = hit_cols
        if surprise_count is None and burst_cols is not None:
            surprise_count = len(burst_cols)

        precision = len(tp) / (len(tp) + len(fp)) if (len(tp)+len(fp))>0 else 0.0
        recall    = len(tp) / (len(tp) + len(fn)) if (len(tp)+len(fn))>0 else 0.0
        f1        = (2*precision*recall)/(precision+recall) if (precision+recall)>0 else 0.0

        last_cells = self.stability.last_sdr.get(inp_id, set())
        last_cols = self.stability.last_cols.get(inp_id, set())

        jac_last = self.jaccard(active_cells, last_cells) if last_cells else 0.0
        overlap_last_cells = len(active_cells & last_cells) if last_cells else 0
        diff_last_cells = len(active_cells ^ last_cells)
        overlap_last_cols = len(active_columns & last_cols) if last_cols else 0
        diff_last_cols = len(active_columns ^ last_cols)
        proto = self._proto_from_counts(inp_id)
        jac_ema = self.jaccard(active_cells, proto) if proto else 0.0
        best_hist, worst_hist = self._hist_best_worst(inp_id, active_cells)

        if jac_ema >= self.convergence_tau:
            self.consecutive_hits[inp_id] += 1
        else:
            self.consecutive_hits[inp_id] = 0
        converged = 1 if self.consecutive_hits[inp_id] >= self.convergence_M else 0

        self.uniq_cells_ever.update(active_cells)
        self.uniq_cols_ever.update(active_columns)

        row = {
            "step": step,
            "sequence_id": sequence_id,
            "pos_in_seq": pos_in_seq,
            "input_id": inp_id,
            "seen_in_run": input_seen_in_run,
            "seen_global": input_seen_global,
            "active_cells": len(active_cells),
            "predicted_cells": len(predicted_prev),
            "tp": len(tp),
            "fp": len(fp),
            "fn": len(fn),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "active_columns": len(active_columns),
            "bursting_columns": surprise_count if surprise_count is not None else 0,
            "sparsity_cells": len(active_cells) / self.num_cells,
            "sparsity_columns": len(active_columns) / self.num_cols,
            "stability_overlap_last": overlap_last_cells,
            "stability_jaccard_last": jac_last,
            "stability_jaccard_ema": jac_ema,
            "stability_best_window": best_hist,
            "stability_worst_window": worst_hist,
            "converged": converged,
            "unique_active_cells_ever": len(self.uniq_cells_ever),
            "unique_active_columns_ever": len(self.uniq_cols_ever),
            "overlap_last_cells": overlap_last_cells,
            "diff_last_cells": diff_last_cells,
            "overlap_last_cols": overlap_last_cols,
            "diff_last_cols": diff_last_cols,
            "kth_overlap": kth_overlap,
            "kplus1_overlap": kplus1_overlap,
            "k_margin": k_margin,
            "sp_connected_mean": sp_connected_mean,
            "sp_near_thr_frac": sp_near_thr_frac,
        }
        if surprise_mean is not None:
            row["surprise_mean"] = surprise_mean
        if spread_v1 is not None:
            row["spread_v1"] = spread_v1
        if spread_v2 is not None:
            row["spread_v2"] = spread_v2
        if overconfident_rate is not None:
            row["overconfident_rate"] = overconfident_rate
        if prediction_accuracy is not None:
            row["prediction_accuracy"] = prediction_accuracy
        if segments is not None:
            row["segments"] = segments
        if synapses is not None:
            row["synapses"] = synapses
        if encoding_diff is not None:
            row["encoding_diff"] = encoding_diff
        self.rows.append(row)

        self.stability.last_sdr[inp_id] = set(active_cells)
        self.stability.last_cols[inp_id] = set(active_columns)
        self.stability.totals[inp_id] += 1
        for c in active_cells:
            self.stability.counts[inp_id][c] += 1
        if inp_id not in self.stability.history:
            self.stability.history[inp_id] = deque(maxlen=self.stability_window)
        self.stability.history[inp_id].append(set(active_cells))

        self.sparse_indices["active_cells"].append(np.fromiter(active_cells, dtype=np.int32))
        self.sparse_indices["active_columns"].append(np.fromiter(active_columns, dtype=np.int32))
        self.sparse_indices["predicted_prev"].append(np.fromiter(predicted_prev, dtype=np.int32))
        self.sparse_indices["tp"].append(np.fromiter(tp, dtype=np.int32))
        self.sparse_indices["fp"].append(np.fromiter(fp, dtype=np.int32))
        self.sparse_indices["fn"].append(np.fromiter(fn, dtype=np.int32))
        if burst_cols is not None:
            self.sparse_indices["burst_columns"].append(np.fromiter(burst_cols, dtype=np.int32))
        if predicted_col_sizes is not None:
            self.sparse_indices["predicted_cols_prev"].append(
                np.fromiter(predicted_col_sizes.keys(), dtype=np.int32)
            )
            self.sparse_indices["predicted_cols_size"].append(
                np.fromiter(predicted_col_sizes.values(), dtype=np.int32)
            )
        if covered_cols is not None:
            self.sparse_indices["covered_columns"].append(np.fromiter(covered_cols, dtype=np.int32))
        if narrow_cells_prev is not None:
            self.sparse_indices["narrow_cells_prev"].append(
                np.fromiter(narrow_cells_prev, dtype=np.int32)
            )
        if narrow_hit_cells is not None:
            self.sparse_indices["narrow_hit_cells"].append(
                np.fromiter(narrow_hit_cells, dtype=np.int32)
            )
        if narrow_cells_prev is not None:
            for cell in narrow_cells_prev:
                if narrow_hit_cells and cell in narrow_hit_cells:
                    self.narrow_miss_streak[cell] = 0
                else:
                    self.narrow_miss_streak[cell] = min(
                        self.overconfident_window,
                        self.narrow_miss_streak.get(cell, 0) + 1,
                    )

    def finalize(self):
        os.makedirs(self.output_dir, exist_ok=True)
        csv_path = os.path.join(self.output_dir, "metrics.csv")
        if self.rows:
            with open(csv_path, "w", newline="") as f:
                import csv
                fieldnames = sorted({k for row in self.rows for k in row.keys()})
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.rows)
        npz_path = os.path.join(self.output_dir, "indices.npz")
        np.savez_compressed(npz_path, **{k: np.array(v, dtype=object) for k, v in self.sparse_indices.items()})
        with open(os.path.join(self.output_dir, "summary.json"), "w") as f:
            json.dump({
                "total_steps": len(self.rows),
                "unique_active_cells_ever": len(self.uniq_cells_ever),
                "unique_active_columns_ever": len(self.uniq_cols_ever),
            }, f, indent=2)
        return {"csv": csv_path, "npz": npz_path}
