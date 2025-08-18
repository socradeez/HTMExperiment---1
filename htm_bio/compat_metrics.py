import csv
import os
from typing import Iterable, Set, Dict, Any


class _CompatWriter:
    """Writer holding state for baseline-compatible metrics."""

    def __init__(self, run_dir: str, num_cells: int, num_columns: int):
        self.path = os.path.join(run_dir, "metrics.csv")
        self.num_cells = num_cells
        self.num_columns = num_columns
        self.prev_predicted: Set[int] = set()
        self.last_cells: Dict[str, Set[int]] = {}
        self.last_cols: Dict[str, Set[int]] = {}
        self._file = open(self.path, "w", newline="")
        self._writer = csv.DictWriter(
            self._file,
            fieldnames=[
                "t",
                "input_id",
                "pos_in_seq",
                "active_columns",
                "bursting_columns",
                "active_cells",
                "predicted_cells",
                "precision",
                "recall",
                "f1",
                "sparsity_columns",
                "sparsity_cells",
                "stability_jaccard_last",
                "stability_jaccard_last_cols",
                "overlap_last_cells",
                "diff_last_cells",
                "overlap_last_cols",
                "diff_last_cols",
                "fp",
                "fn",
            ],
        )
        self._writer.writeheader()

    def close(self) -> None:
        self._file.close()

    # internal helpers -------------------------------------------------
    @staticmethod
    def _jaccard(a: Set[int], b: Set[int]) -> float:
        if not a and not b:
            return 1.0
        inter = len(a & b)
        union = len(a | b)
        return inter / union if union else 1.0

    def write(
        self,
        t: int,
        input_id: str,
        pos_in_seq: int,
        active_columns: Iterable[int],
        bursting_columns: Iterable[int],
        active_cells_ids: Iterable[int],
        predicted_cells_ids: Iterable[int],
        k_active_columns: int,
    ) -> None:
        act_cols = set(active_columns)
        burst_cols = set(bursting_columns)
        act_cells = set(active_cells_ids)
        pred_cells = set(predicted_cells_ids)

        tp = len(self.prev_predicted & act_cells)
        fp = len(self.prev_predicted - act_cells)
        fn = len(act_cells - self.prev_predicted)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        sparsity_cols = len(act_cols) / self.num_columns if self.num_columns else 0.0
        sparsity_cells = len(act_cells) / self.num_cells if self.num_cells else 0.0

        last_cells = self.last_cells.get(input_id, set())
        last_cols = self.last_cols.get(input_id, set())
        overlap_last_cells = len(act_cells & last_cells)
        diff_last_cells = len(act_cells ^ last_cells)
        overlap_last_cols = len(act_cols & last_cols)
        diff_last_cols = len(act_cols ^ last_cols)
        jac_cells = self._jaccard(act_cells, last_cells)
        jac_cols = self._jaccard(act_cols, last_cols)

        row = {
            "t": t,
            "input_id": input_id,
            "pos_in_seq": pos_in_seq,
            "active_columns": len(act_cols),
            "bursting_columns": len(burst_cols),
            "active_cells": len(act_cells),
            "predicted_cells": len(self.prev_predicted),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "sparsity_columns": sparsity_cols,
            "sparsity_cells": sparsity_cells,
            "stability_jaccard_last": jac_cells,
            "stability_jaccard_last_cols": jac_cols,
            "overlap_last_cells": overlap_last_cells,
            "diff_last_cells": diff_last_cells,
            "overlap_last_cols": overlap_last_cols,
            "diff_last_cols": diff_last_cols,
            "fp": fp,
            "fn": fn,
        }
        self._writer.writerow(row)

        # update state
        self.prev_predicted = pred_cells
        self.last_cells[input_id] = act_cells
        self.last_cols[input_id] = act_cols


def open_writer(run_dir: str, num_cells: int, num_columns: int) -> _CompatWriter:
    return _CompatWriter(run_dir, num_cells, num_columns)


def write_compat_row(
    writer: _CompatWriter,
    t: int,
    input_id: str,
    pos_in_seq: int,
    active_columns: Iterable[int],
    bursting_columns: Iterable[int],
    active_cells_ids: Iterable[int],
    predicted_cells_ids: Iterable[int],
    k_active_columns: int,
) -> None:
    writer.write(
        t,
        input_id,
        pos_in_seq,
        active_columns,
        bursting_columns,
        active_cells_ids,
        predicted_cells_ids,
        k_active_columns,
    )
