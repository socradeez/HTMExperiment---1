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
    def select_winners(self, ff_drive, distal_bias, cells_per_column, winners_per_column):
        """Select active cells within each column.

        Args:
            ff_drive: feed-forward drive per cell.
            distal_bias: bias per cell.
            cells_per_column: number of cells per column.
            winners_per_column: target winners per column.
        Returns:
            1D tensor or list of active cell indices.
        """
        ...


class LearningRule(Protocol):
    """Placeholder for future learning rule interface."""
    ...
