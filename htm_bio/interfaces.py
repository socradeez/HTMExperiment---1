"""Abstract interfaces for the BIO scaffold."""

from typing import Protocol, Tuple


class PredictiveBiasComputer(Protocol):
    def compute_bias(self, tm, active_cells_prev) -> Tuple["Tensor", "Tensor"]:
        """Compute per-cell bias by querying ``tm`` with previous active cells.

        Args:
            tm: temporal memory instance providing segment data.
            active_cells_prev: set or tensor of previously active cells.
        Returns:
            Tuple of ``(bias_per_cell, segment_counts)`` tensors.
        """
        ...


class InhibitionModel(Protocol):
    def select_winners(
        self,
        active_columns,
        bias,
        ff_threshold,
        cells_per_column,
        winners_per_column,
    ) -> Tuple["Tensor", "Tensor"]:
        """Select winners and bursting columns given per-cell ``bias``.

        Returns a tuple ``(winner_cell_ids, bursting_column_ids)``.
        """
        ...


class LearningRule(Protocol):
    """Placeholder for future learning rule interface."""
    ...
