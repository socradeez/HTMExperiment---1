from dataclasses import dataclass
from typing import Optional, List, Dict

from dataclasses import field
from config import MetaParams


@dataclass
class BioModelConfig:
    """Model hyperparameters for the BIO scaffold."""
    input_size: int = 1024
    num_columns: int = 2048
    cells_per_column: int = 10
    # ~2% of columns by default
    k_active_columns: int = 40

    # Feed-forward & thresholds
    ff_threshold: float = 1.0

    # Distal / predictive bias
    segment_activation_threshold: int = 10
    bias_gain: float = 1.0
    bias_cap: float = 1.0

    # Inhibition (within column)
    inhibition_strength: float = 1.0
    winners_per_column: int = 1

    # Metaplasticity
    meta: MetaParams = field(default_factory=MetaParams)

    # Torch device/runtime
    backend: str = "torch"
    device: str = "cuda"


@dataclass
class BioRunConfig:
    """Runtime configuration for a BIO run."""
    sequence: str = "A>B>C>D"
    explicit_step_tokens: Optional[List[str]] = None
    token_pos_map: Optional[Dict[str, int]] = None
    steps: Optional[int] = None

    outdir: str = "runs"
    seed: int = 7
    figure_mode: str = "single"
    schedule_name: Optional[str] = None

    dry_run: bool = True
