"""Network orchestration for HTM components."""
import numpy as np

from .sp import SpatialPooler
from .tm import TemporalMemory
from .confidence_tm import ConfidenceModulatedTM


class HTMNetwork:
    """Standard HTM network consisting of SP and TM."""

    def __init__(self, input_size=100, sp_params=None, tm_params=None, seed=None):
        sp_params = dict(sp_params or {})
        tm_params = dict(tm_params or {})

        if seed is not None:
            sp_params.setdefault("seed", seed)
            tm_params.setdefault("seed", seed)
        elif "seed" in sp_params:
            tm_params.setdefault("seed", sp_params["seed"])
        elif "seed" in tm_params:
            sp_params.setdefault("seed", tm_params["seed"])

        sp_defaults = {"column_count": 100, "sparsity": 0.1, "boost_strength": 0.0}
        for k, v in sp_defaults.items():
            sp_params.setdefault(k, v)

        self.input_size = input_size
        self.sp = SpatialPooler(input_size=input_size, **sp_params)

        tm_params.setdefault("column_count", self.sp.column_count)
        self.tm = self._build_tm(tm_params)

        self.prediction_accuracy = []
        self.anomaly_scores = []

    def _build_tm(self, tm_params):
        return TemporalMemory(**tm_params)

    def compute(self, input_vector, learn=True):
        active_columns = self.sp.compute(input_vector, learn=learn)
        predicted_columns = self._get_predicted_columns()
        correctly_predicted = np.sum(active_columns & predicted_columns)
        active_cells, predictive_cells = self.tm.compute(active_columns, learn=learn)

        if np.sum(active_columns) > 0:
            accuracy = correctly_predicted / np.sum(active_columns)
            self.prediction_accuracy.append(accuracy)
            anomaly = 1.0 - accuracy
            self.anomaly_scores.append(anomaly)

        return {
            "active_columns": active_columns,
            "active_cells": active_cells,
            "predictive_cells": predictive_cells,
            "anomaly_score": self.anomaly_scores[-1] if self.anomaly_scores else 1.0,
            "system_confidence": getattr(self.tm, "current_system_confidence", None),
        }

    def _get_predicted_columns(self):
        predicted_columns = np.zeros(self.sp.column_count, dtype=bool)
        for cell in self.tm.predictive_cells:
            col = cell // self.tm.cells_per_column
            predicted_columns[col] = True
        return predicted_columns

    def reset_sequence(self):
        self.tm.reset()


class ConfidenceHTMNetwork(HTMNetwork):
    """HTM network using confidence-modulated temporal memory."""

    def __init__(self, input_size=100, sp_params=None, tm_params=None, seed=None):
        self.system_confidence_history = []
        super().__init__(input_size=input_size, sp_params=sp_params, tm_params=tm_params, seed=seed)

    def _build_tm(self, tm_params):
        return ConfidenceModulatedTM(**tm_params)

    def compute(self, input_vector, learn=True):
        res = super().compute(input_vector, learn=learn)
        if res["system_confidence"] is not None:
            self.system_confidence_history.append(res["system_confidence"])
        return res
