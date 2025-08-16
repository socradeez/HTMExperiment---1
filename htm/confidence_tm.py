"""Confidence-modulated Temporal Memory implementation."""

from collections import defaultdict, deque
from typing import Dict
import logging

import numpy as np

from .tm import TemporalMemory

log = logging.getLogger(__name__)


class ConfidenceModulatedTM(TemporalMemory):
    """Temporal Memory with confidence-based learning rate modulation."""

    def __init__(
        self,
        confidence_window: int = 100,
        base_learning_rate: float = 0.1,
        exploration_bonus: float = 2.0,
        confidence_threshold: float = 0.7,
        hardening_rate: float = 0.03,
        hardening_threshold: float = 0.7,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        # Confidence parameters
        self.confidence_window = confidence_window
        self.base_learning_rate = base_learning_rate
        self.exploration_bonus = exploration_bonus
        self.confidence_threshold = confidence_threshold
        self.hardening_rate = hardening_rate
        self.hardening_threshold = hardening_threshold

        # Confidence tracking
        self.cell_confidence = defaultdict(lambda: deque(maxlen=confidence_window))
        self.system_confidence = deque(maxlen=confidence_window)
        self.current_system_confidence = 0.5
        self.current_cell_confidences: Dict[int, float] = {}

        # Synapse hardening
        self.synapse_hardness = defaultdict(lambda: defaultdict(float))

        # Instrumentation fields
        self._hardening_updates = 0
        self._hardness_decays = 0
        self._hardness_sum = 0.0
        self._hardness_count = 0
        self._conf_over_thr_steps = 0
        self._total_steps = 0

        self.version_tag = "CONF-TM v2 (no-base-learn; hardened adapt)"
        log.info(self.version_tag)

        # Step counter
        self.timestep = 0

    # ------------------------------------------------------------------
    def compute(self, active_columns, learn: bool = True):
        """Process input columns and update confidence metrics."""
        prev_active = self.active_cells.copy()
        prev_predictive = self.predictive_cells.copy()

        # Core TM computation without learning
        active_cells, predictive_cells = super().compute(active_columns, learn=False)

        if self.timestep > 0:
            self._update_confidence_metrics(active_columns, prev_predictive)
            self._total_steps += 1
            if self.current_system_confidence >= self.hardening_threshold:
                self._conf_over_thr_steps += 1

        if learn:
            self._apply_confidence_modulation()
            self._confidence_modulated_learning(prev_active)

        self.timestep += 1
        return active_cells, predictive_cells

    # ------------------------------------------------------------------
    def _apply_confidence_modulation(self) -> None:
        """Apply exploration boost when system confidence is low."""
        if self.current_system_confidence >= self.confidence_threshold:
            return

        exploration_boost = self.exploration_bonus - 1.0
        for cell_id in list(self.winner_cells):
            for segment in self.segments.get(cell_id, []):
                for i, (target_cell, perm) in enumerate(segment["synapses"]):
                    if target_cell in self.active_cells:
                        new_perm = min(1.0, perm + exploration_boost * 0.01)
                        segment["synapses"][i] = (target_cell, float(new_perm))

    # ------------------------------------------------------------------
    def _update_confidence_metrics(self, active_columns, prev_predictive):
        """Update cell and system confidence based on prediction success."""
        active_cols_set = set(np.where(active_columns)[0])
        predicted_cols_set = {cell // self.cells_per_column for cell in prev_predictive}

        if active_cols_set:
            predicted_active = len(active_cols_set & predicted_cols_set)
            precision = predicted_active / len(active_cols_set)
            system_conf = precision
        else:
            system_conf = self.current_system_confidence

        self.system_confidence.append(system_conf)
        if self.system_confidence:
            self.current_system_confidence = float(np.mean(self.system_confidence))

        for cell in self.active_cells:
            self.cell_confidence[cell].append(1.0 if cell in prev_predictive else 0.25)

        self.current_cell_confidences = {
            cell: float(np.mean(hist))
            for cell, hist in self.cell_confidence.items()
            if hist
        }

    # ------------------------------------------------------------------
    def _confidence_modulated_learning(self, prev_active) -> None:
        if not prev_active:
            return

        for cell_id in list(self.winner_cells):
            if self.current_system_confidence < self.confidence_threshold:
                learning_rate = self.base_learning_rate * self.exploration_bonus
            else:
                learning_rate = self.base_learning_rate

            best_segment = None
            best_score = 0
            for seg_idx, segment in enumerate(self.segments.get(cell_id, [])):
                score = self._count_active_synapses(segment, prev_active)
                if score >= self.learning_threshold and score > best_score:
                    best_segment = (seg_idx, segment)
                    best_score = score

            if best_segment is not None:
                self._adapt_segment(best_segment[1], prev_active, permanence_inc=learning_rate)
            elif len(prev_active) >= self.learning_threshold:
                self._grow_segment(cell_id, prev_active)

        wrongly_predictive = self.predictive_cells - self.active_cells
        for cell_id in wrongly_predictive:
            for seg_idx, segment in enumerate(self.segments.get(cell_id, [])):
                if (cell_id, seg_idx) in self.active_segments:
                    self._adapt_segment(
                        segment, self.active_cells, permanence_inc=-self.predicted_decrement
                    )

    # ------------------------------------------------------------------
    def _grow_segment(self, cell_id, source_cells):
        segment = super()._grow_segment(cell_id, source_cells)
        if segment is not None:
            seg_idx = len(self.segments.get(cell_id, [])) - 1
            segment["owner"] = cell_id
            segment["seg_idx"] = seg_idx
        return segment

    # ------------------------------------------------------------------
    def _adapt_segment(self, segment, active_cells, permanence_inc):
        if "owner" in segment and "seg_idx" in segment:
            cell_id = segment["owner"]
            seg_idx = segment["seg_idx"]
            self._adapt_segment_with_hardening(
                cell_id, seg_idx, segment, active_cells, permanence_inc
            )
        else:
            super()._adapt_segment(segment, active_cells, permanence_inc)

    # ------------------------------------------------------------------
    def _adapt_segment_with_hardening(
        self, cell_id, seg_idx, segment, active_cells, base_rate
    ) -> None:
        conf = self.current_system_confidence
        for i, (target_cell, perm) in enumerate(list(segment["synapses"])):
            hardness = self.synapse_hardness[cell_id].get((seg_idx, i), 0.0)

            if conf >= self.hardening_threshold:
                delta_h = self.hardening_rate * max(0.0, conf - self.hardening_threshold)
            else:
                delta_h = -0.5 * self.hardening_rate
                if abs(delta_h) > 0:
                    self._hardness_decays += 1

            new_hardness = float(np.clip(hardness + delta_h, 0.0, 1.0))
            if abs(new_hardness - hardness) > 1e-9:
                self._hardening_updates += 1
            self.synapse_hardness[cell_id][(seg_idx, i)] = new_hardness
            self._hardness_sum += new_hardness
            self._hardness_count += 1

            if target_cell in active_cells:
                delta = base_rate * (1.0 - new_hardness)
            else:
                delta = -self.permanence_decrement * (1.0 - new_hardness)

            new_perm = float(np.clip(perm + delta, 0.0, 1.0))
            segment["synapses"][i] = (target_cell, new_perm)
