import warnings
import numpy as np
import torch

from config import ModelConfig
from htm_core import seeded_rng

_ver = tuple(int(x) for x in torch.__version__.split(".")[:2])
assert _ver >= (2, 0), "Torch >=2.0 required"


class TorchSP:
    def __init__(self, cfg: ModelConfig, rng: np.random.Generator, device: str):
        self.cfg = cfg
        dev = torch.device(device)
        if dev.type == "cuda" and not torch.cuda.is_available():
            warnings.warn("CUDA not available, falling back to CPU")
            dev = torch.device("cpu")
        self.device = dev

        idx = rng.integers(0, cfg.input_size, size=(cfg.num_columns, cfg.synapses_per_column), endpoint=False)
        perm = rng.normal(loc=cfg.init_perm_mean, scale=cfg.init_perm_sd, size=(cfg.num_columns, cfg.synapses_per_column))
        perm = np.clip(perm, 0.0, 1.0).astype(np.float32)

        row_ptr = np.arange(0, (cfg.num_columns + 1) * cfg.synapses_per_column, cfg.synapses_per_column, dtype=np.int64)
        self.crow_indices = torch.from_numpy(row_ptr).to(self.device)
        self.col_indices = torch.from_numpy(idx.reshape(-1).astype(np.int64)).to(self.device)
        self.perm_values = torch.from_numpy(perm.reshape(-1)).to(self.device)

    @property
    def proximal_perm(self) -> torch.Tensor:
        return self.perm_values.view(self.cfg.num_columns, self.cfg.synapses_per_column)

    def compute_overlap(self, x_dense_bool: torch.Tensor) -> torch.Tensor:
        x_dense_float = x_dense_bool.to(torch.float32)
        conn_values = (self.perm_values >= self.cfg.perm_connected).to(torch.float32)
        P_conn = torch.sparse_csr_tensor(
            self.crow_indices, self.col_indices, conn_values,
            size=(self.cfg.num_columns, self.cfg.input_size), device=self.device
        )
        overlaps = torch.sparse.mv(P_conn, x_dense_float)
        return overlaps

    def k_wta(self, overlaps: torch.Tensor, k: int) -> torch.Tensor:
        if overlaps.numel() == 0:
            return torch.empty(0, dtype=torch.int64, device=overlaps.device)
        k = min(k, overlaps.numel())
        return torch.topk(overlaps, k).indices

    def learn(self, x_dense_bool: torch.Tensor, active_cols: torch.Tensor):
        if active_cols.numel() == 0:
            return
        for col in active_cols.tolist():
            start = int(self.crow_indices[col])
            end = int(self.crow_indices[col + 1])
            syn_idx = self.col_indices[start:end]
            active = x_dense_bool[syn_idx]
            self.perm_values[start:end] += self.cfg.perm_inc * active.float()
            self.perm_values[start:end] -= self.cfg.perm_dec * (~active).float()
            torch.clamp_(self.perm_values[start:end], 0.0, 1.0)


class TorchTM:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("TorchTM is not implemented yet")


def make_sp_torch(model_cfg: ModelConfig, seed: int, device: str) -> TorchSP:
    rng = seeded_rng(seed)
    return TorchSP(model_cfg, rng, device)

