"""Abstract interfaces for the BIO scaffold."""

from typing import Protocol


class PredictiveBiasComputer(Protocol):
    def compute_bias(self, distal_evidence):
        """Compute per-cell threshold reductions from distal input.

        Args:
            distal_evidence: tensor of distal activity per cell.
        Returns:
            Tensor of the same shape containing threshold reductions.
        """
        ...


class InhibitionModel(Protocol):
    def select_winners(
        self, active_columns, distal_bias, cells_per_column, winners_per_column
    ):
        """Select active cells within ``active_columns`` using ``distal_bias``."""
        ...


class LearningRule(Protocol):
    """Placeholder for future learning rule interface."""
    ...
