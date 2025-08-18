"""Subthreshold predictive bias stubs."""

import torch

from .interfaces import PredictiveBiasComputer


class SubthresholdPredictor(PredictiveBiasComputer):
    """Placeholder predictive bias module.

    Currently returns zeros and logs a stub message when run in dry-run mode.
    """

    def __init__(self, bias_gain: float, bias_cap: float, segment_activation_threshold: int, dry_run: bool = True):
        self.bias_gain = bias_gain
        self.bias_cap = bias_cap
        self.segment_activation_threshold = segment_activation_threshold
        self.dry_run = dry_run

    def compute_bias(self, distal_evidence):
        if self.dry_run:
            print("BIO stub: bias=0")
        return torch.zeros_like(distal_evidence)
