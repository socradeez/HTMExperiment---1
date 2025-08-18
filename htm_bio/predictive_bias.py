"""Subthreshold predictive bias computation."""

import torch

from .interfaces import PredictiveBiasComputer


class SubthresholdPredictor(PredictiveBiasComputer):
    """Map distal evidence to a threshold-reducing bias."""

    def __init__(self, bias_gain: float, bias_cap: float, segment_activation_threshold: int):
        self.bias_gain = bias_gain
        self.bias_cap = bias_cap
        self.segment_activation_threshold = segment_activation_threshold

    def compute_bias(self, distal_evidence: torch.Tensor) -> torch.Tensor:
        """Return per-cell bias given distal evidence.

        ``distal_evidence`` is the number of active connected synapses per cell.
        It is normalized by ``segment_activation_threshold`` then scaled by
        ``bias_gain`` and finally capped at ``bias_cap``.
        """
        norm = distal_evidence / float(self.segment_activation_threshold)
        bias = norm * self.bias_gain
        return torch.clamp(bias, max=self.bias_cap)
