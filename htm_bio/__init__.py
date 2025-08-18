"""Scaffolding for the BIO variant with subthreshold prediction and
within-column inhibition. Currently a dry-run only."""

from .config import BioModelConfig, BioRunConfig
from .runner import main

__all__ = ["BioModelConfig", "BioRunConfig", "main"]
