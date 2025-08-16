
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

    rows: List[Dict] = field(default_factory=list)
    stability: StabilityState = field(default_factory=StabilityState)
    sparse_indices: Dict[str, List[np.ndarray]] = field(default_factory=lambda: defaultdict(list))
    seen_in_run: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    seen_global: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    consecutive_hits: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    uniq_cells_ever: Set[int] = field(default_factory=set)
    uniq_cols_ever: Set[int] = field(default_factory=set)

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
                 predicted_prev: Set[int]):
        tp = active_cells & predicted_prev
        fp = predicted_prev - active_cells
        fn = active_cells - predicted_prev

        # Bursting columns: active but with no predicted_prev cells in that column
        predicted_cols_prev = {c // self.cells_per_column for c in predicted_prev}
        bursting_cols = set(active_columns) - predicted_cols_prev

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
            "bursting_columns": len(bursting_cols),
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
        }
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
        self.sparse_indices["predicted_prev"].append(np.fromiter(predicted_prev, dtype=np.int32))
        self.sparse_indices["tp"].append(np.fromiter(tp, dtype=np.int32))
        self.sparse_indices["fp"].append(np.fromiter(fp, dtype=np.int32))
        self.sparse_indices["fn"].append(np.fromiter(fn, dtype=np.int32))

    def finalize(self):
        os.makedirs(self.output_dir, exist_ok=True)
        csv_path = os.path.join(self.output_dir, "metrics.csv")
        if self.rows:
            with open(csv_path, "w", newline="") as f:
                import csv
                writer = csv.DictWriter(f, fieldnames=list(self.rows[0].keys()))
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
