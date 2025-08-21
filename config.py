
from dataclasses import dataclass
from typing import Optional, List, Dict

@dataclass
class ModelConfig:
    # Inputs / columns / cells
    input_size: int = 1024
    num_columns: int = 2048
    cells_per_column: int = 10

    # k-WTA
    k_active_columns: int = 40  # ~2%

    # Proximal (Spatial Pooling)
    synapses_per_column: int = 32
    perm_connected: float = 0.25
    init_perm_mean: float = 0.26
    init_perm_sd: float = 0.02
    perm_inc: float = 0.03
    perm_dec: float = 0.015

    # Distal / Temporal Memory
    distal_synapses_per_segment: int = 20
    segment_activation_threshold: int = 10
    new_segment_init_perm_mean: float = 0.26
    new_segment_init_perm_sd: float = 0.02

@dataclass
class RunConfig:
    seed: int = 7
    steps: int = 400
    learn: bool = True
    diagnostics_print: bool = True
    sp_near_threshold_eps: float = 0.01
    output_dir: str = "runs"
    run_name: Optional[str] = None
    backend: str = "numpy"  # choices: "numpy", "torch"
    device: str = "cuda"    # "cuda" or "cpu"

    # Input generation
    sdr_on_bits: int = 20   # ~2%
    sequence: str = "A>B>C>D"
    sequence_delimiter: str = ">"

    # Stability
    stability_window: int = 50
    ema_threshold: float = 0.5
    convergence_tau: float = 0.9
    convergence_M: int = 3

    # Noise
    input_flip_bits: int = 0

    # Optional explicit scheduling
    explicit_step_tokens: Optional[List[str]] = None
    token_pos_map: Optional[Dict[str, int]] = None
    schedule_name: Optional[str] = None
    plots: Optional[List[str]] = None
    overconfident_window: int = 2

def json_dumps(d) -> str:
    import json
    return json.dumps(d, indent=2, sort_keys=True)
