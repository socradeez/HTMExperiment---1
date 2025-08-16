"""Spatial pooler implementation."""

import numpy as np


class SpatialPooler:
    """Spatial Pooler creates sparse distributed representations from input."""

    def __init__(self,
                 input_size=2048,
                 column_count=2048,
                 sparsity=0.02,
                 potential_radius=None,
                 permanence_inc=0.1,
                 permanence_dec=0.01,
                 connected_threshold=0.5,
                 min_overlap=5,
                 boost_strength=2.0,
                 duty_cycle_period=1000,
                 seed=42):

        np.random.seed(seed)

        # Core parameters
        self.input_size = input_size
        self.column_count = column_count
        self.sparsity = sparsity
        self.active_columns_count = max(1, int(column_count * sparsity))
        self.potential_radius = potential_radius or input_size
        self.permanence_inc = permanence_inc
        self.permanence_dec = permanence_dec
        self.connected_threshold = connected_threshold
        self.min_overlap = min_overlap
        self.boost_strength = boost_strength
        self.duty_cycle_period = duty_cycle_period

        # Initialize proximal synapses with random permanences
        self.permanences = np.random.uniform(
            connected_threshold - 0.2,
            connected_threshold + 0.2,
            (column_count, input_size)
        )

        # Randomly connect only a subset of inputs to each column
        connectivity = np.random.random((column_count, input_size)) < 0.5
        self.permanences *= connectivity

        # Duty cycles and boosting
        self.overlap_duty_cycles = np.zeros(column_count)
        self.active_duty_cycles = np.zeros(column_count)
        self.boost_factors = np.ones(column_count)
        self.iteration = 0

    def compute(self, input_vector, learn=True):
        """Main SP computation: create SDR from input."""

        # Calculate overlap scores
        connected = self.permanences >= self.connected_threshold
        overlap = np.sum(connected * input_vector, axis=1)

        # Apply minimum overlap and boosting
        overlap[overlap < self.min_overlap] = 0
        boosted_overlap = overlap * self.boost_factors

        # Inhibition - select top k columns
        active_columns = np.zeros(self.column_count, dtype=bool)
        if np.any(boosted_overlap > 0):
            # Choose at most active_columns_count winners with positive boosted overlap
            candidates = np.where(boosted_overlap > 0)[0]
            if len(candidates) <= self.active_columns_count:
                winners = candidates
            else:
                # partial sort
                winners = candidates[np.argpartition(boosted_overlap[candidates], -self.active_columns_count)[-self.active_columns_count:]]
            active_columns[winners] = True

        if learn:
            self._update_permanences(input_vector, active_columns)
            self._update_duty_cycles(active_columns, overlap > 0)
            self._update_boost_factors()

        self.iteration += 1
        return active_columns

    def _update_permanences(self, input_vector, active_columns):
        """Hebbian learning for active columns."""
        active_idx = np.where(active_columns)[0]
        if active_idx.size == 0:
            return
        active_inputs = input_vector.astype(bool)
        for col in active_idx:
            self.permanences[col, active_inputs] += self.permanence_inc
            self.permanences[col, ~active_inputs] -= self.permanence_dec
            self.permanences[col] = np.clip(self.permanences[col], 0, 1)

    def _update_duty_cycles(self, active_columns, overlap_mask):
        """Track activity statistics for boosting."""
        period = min(self.iteration + 1, self.duty_cycle_period)
        self.active_duty_cycles = (
            (self.active_duty_cycles * (period - 1) + active_columns) / period
        )
        self.overlap_duty_cycles = (
            (self.overlap_duty_cycles * (period - 1) + overlap_mask) / period
        )

    def _update_boost_factors(self):
        """Boost columns that don't activate enough."""
        if self.iteration < self.duty_cycle_period:
            return

        target_density = self.sparsity
        for col in range(self.column_count):
            if self.active_duty_cycles[col] < target_density:
                boost = 1 + self.boost_strength * (target_density - self.active_duty_cycles[col])
                self.boost_factors[col] = boost
            else:
                self.boost_factors[col] = 1.0
