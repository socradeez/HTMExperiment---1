"""Temporal memory implementation."""

import numpy as np
from typing import Dict, List


class TemporalMemory:
    """Temporal Memory learns sequences and makes predictions."""

    def __init__(self,
                 column_count=2048,
                 cells_per_column=32,
                 activation_threshold=13,
                 learning_threshold=10,
                 initial_permanence=0.21,
                 permanence_increment=0.1,
                 permanence_decrement=0.1,
                 predicted_decrement=0.01,
                 connection_threshold=0.5,  # explicit threshold
                 max_segments_per_cell=128,
                 max_synapses_per_segment=32,  # Reduced from 128
                 seed=42):

        np.random.seed(seed)

        self.column_count = column_count
        self.cells_per_column = cells_per_column
        self.total_cells = column_count * cells_per_column
        self.activation_threshold = activation_threshold
        self.learning_threshold = learning_threshold
        self.initial_permanence = initial_permanence
        self.permanence_increment = permanence_increment
        self.permanence_decrement = permanence_decrement
        self.predicted_decrement = predicted_decrement
        self.connection_threshold = connection_threshold
        self.max_segments_per_cell = max_segments_per_cell
        self.max_synapses_per_segment = max_synapses_per_segment

        self.active_cells = set()
        self.predictive_cells = set()
        self.winner_cells = set()
        self.active_segments = set()
        self.matching_segments = set()

        # Use regular dict instead of defaultdict to avoid accidental key creation
        self.segments: Dict[int, List[dict]] = {}
        self.segment_counter = 0

    def compute(self, active_columns, learn=True):
        """Main TM computation: process active columns and make predictions."""

        prev_active = self.active_cells.copy()
        prev_winner = self.winner_cells.copy()

        self.active_cells = set()
        self.winner_cells = set()

        # Phase 1: Activate cells based on current input and previous predictions
        for col in np.where(active_columns)[0]:
            column_cells = self._get_column_cells(col)
            predicted_cells = self.predictive_cells & column_cells

            if predicted_cells:
                self.active_cells.update(predicted_cells)
                self.winner_cells.update(predicted_cells)
            else:
                # Bursting - all cells become active
                self.active_cells.update(column_cells)
                # Choose best matching cell as winner
                winner = self._get_best_matching_cell(col, prev_active)
                self.winner_cells.add(winner)

        # Phase 2: Learning - perform learning on winner cells
        if learn and len(prev_active) > 0:
            for cell_id in self.winner_cells:
                # Check if any segments were matching
                best_segment = None
                best_score = 0

                for seg_idx, segment in enumerate(self.segments.get(cell_id, [])):
                    score = self._count_active_synapses(segment, prev_active)
                    if score >= self.learning_threshold and score > best_score:
                        best_segment = (seg_idx, segment)
                        best_score = score

                if best_segment is not None:
                    # Reinforce the best matching segment
                    self._adapt_segment(best_segment[1], prev_active, permanence_inc=self.permanence_increment)
                elif len(prev_active) >= self.learning_threshold:
                    # No good segment exists, grow a new one
                    self._grow_segment(cell_id, prev_active)

        # Phase 3: Calculate predictions for next timestep based on current active cells
        self.predictive_cells = set()
        self.active_segments = set()
        self.matching_segments = set()

        # Only iterate over cells that actually have segments
        for cell_id, cell_segments in self.segments.items():
            for seg_idx, segment in enumerate(cell_segments):
                active_synapses = self._count_active_synapses(segment, self.active_cells)

                if active_synapses >= self.activation_threshold:
                    self.predictive_cells.add(cell_id)
                    self.active_segments.add((cell_id, seg_idx))
                elif active_synapses >= self.learning_threshold:
                    self.matching_segments.add((cell_id, seg_idx))

        # Phase 4: Punish incorrectly predictive segments (optional, only if learning)
        if learn:
            wrongly_predictive = self.predictive_cells - self.active_cells
            for cell_id in wrongly_predictive:
                for seg_idx, segment in enumerate(self.segments.get(cell_id, [])):
                    if (cell_id, seg_idx) in self.active_segments:
                        # Punish based on the CURRENT active cells that drove prediction
                        self._adapt_segment(segment, self.active_cells,
                                           permanence_inc=-self.predicted_decrement)

        return self.active_cells, self.predictive_cells

    def _get_column_cells(self, column):
        """Get all cell indices for a column."""
        start = column * self.cells_per_column
        return set(range(start, start + self.cells_per_column))

    def _get_best_matching_cell(self, column, active_cells):
        """Choose the best cell to represent this column."""
        column_cells = self._get_column_cells(column)

        best_cell = None
        best_score = -1

        for cell in column_cells:
            for segment in self.segments.get(cell, []):
                score = self._count_active_synapses(segment, active_cells)
                if score > best_score:
                    best_score = score
                    best_cell = cell

        if best_cell is None or best_score < self.learning_threshold:
            min_segments = float('inf')
            for cell in column_cells:
                num_segments = len(self.segments.get(cell, []))
                if num_segments < min_segments:
                    min_segments = num_segments
                    best_cell = cell

        return best_cell

    def _count_active_synapses(self, segment, active_cells):
        """Count synapses connected to active cells."""
        count = 0
        for cell_id, permanence in segment['synapses']:
            if cell_id in active_cells and permanence >= self.connection_threshold:
                count += 1
        return count

    def _adapt_segment(self, segment, active_cells, permanence_inc):
        """Update permanences based on active cells - used by base TM."""
        for i, (cell_id, perm) in enumerate(segment['synapses']):
            if cell_id in active_cells:
                segment['synapses'][i] = (cell_id, float(np.clip(perm + permanence_inc, 0, 1)))
            else:
                segment['synapses'][i] = (cell_id, float(np.clip(perm - self.permanence_decrement, 0, 1)))

    def _grow_segment(self, cell_id, source_cells):
        """Create new segment with synapses to source cells."""
        if len(self.segments.get(cell_id, [])) >= self.max_segments_per_cell:
            return None

        # Convert set to list if necessary
        if isinstance(source_cells, set):
            source_cells = list(source_cells)
        else:
            source_cells = list(source_cells)

        # Limit synapses per segment
        sample_size = min(len(source_cells), self.max_synapses_per_segment)
        if sample_size < self.learning_threshold:
            return None

        sampled_cells = np.random.choice(source_cells, sample_size, replace=False)

        new_segment = {
            'synapses': [(int(cell), float(self.initial_permanence)) for cell in sampled_cells]
        }

        self.segments.setdefault(cell_id, []).append(new_segment)
        return new_segment

    def reset(self):
        """Clear all cell states."""
        self.active_cells = set()
        self.predictive_cells = set()
        self.winner_cells = set()
        self.active_segments = set()
        self.matching_segments = set()
