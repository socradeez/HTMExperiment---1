"""Wrapper around BIO TM's bias computation."""

from .interfaces import PredictiveBiasComputer


class SubthresholdPredictor(PredictiveBiasComputer):
    """Delegate bias computation to the temporal memory instance."""

    def compute_bias(self, tm, active_cells_prev):
        """Return ``(bias_per_cell, segment_counts)`` from ``tm``.

        Parameters
        ----------
        tm : BioTM
            Temporal memory object providing ``compute_bias``.
        active_cells_prev : set[int] | Tensor
            Previously active cells at ``t-1``.
        """
        return tm.compute_bias(active_cells_prev)
