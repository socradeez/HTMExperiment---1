import warnings
import numpy as np
import torch
from typing import Set, List

from config import ModelConfig
from htm_core import seeded_rng
from metaplasticity import apply_gates, effective_dec

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
            size=(self.cfg.num_columns, self.cfg.input_size), device=self.device,
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
    def __init__(self, cfg: ModelConfig, rng: np.random.Generator, device: str):
        self.cfg = cfg
        self.rng = rng
        dev = torch.device(device)
        if dev.type == "cuda" and not torch.cuda.is_available():
            warnings.warn("CUDA not available, falling back to CPU")
            dev = torch.device("cpu")
        self.device = dev
        self.num_cells = cfg.num_columns * cfg.cells_per_column
        self.crow_indices = torch.tensor([0], dtype=torch.int64, device=self.device)
        self.col_indices = torch.tensor([], dtype=torch.int64, device=self.device)
        self.perm_values = torch.tensor([], dtype=torch.float32, device=self.device)
        self.seg_owner_cell = torch.tensor([], dtype=torch.int64, device=self.device)
        self.num_segments = 0
        self.pending_row_idx: List[int] = []
        self.pending_col_idx: List[int] = []
        self.pending_perm: List[float] = []
        self.pending_owner: List[int] = []
        self.flush_interval = 10
        self.pending_threshold = 1000
        self.steps_since_flush = 0

    def predict_from_set(self, active_cells_prev: Set[int]):
        if self.num_segments == 0 or not active_cells_prev:
            return (
                torch.empty(0, dtype=torch.int64, device=self.device),
                torch.empty(0, dtype=torch.int64, device=self.device),
                torch.empty(0, dtype=torch.int64, device=self.device),
            )
        a_prev = torch.zeros(self.num_cells, dtype=torch.float32, device=self.device)
        idx = torch.tensor(list(active_cells_prev), dtype=torch.int64, device=self.device)
        a_prev[idx] = 1.0
        conn = (self.perm_values >= self.cfg.perm_connected).to(torch.float32)
        M_conn = torch.sparse_csr_tensor(
            self.crow_indices, self.col_indices, conn,
            size=(self.num_segments, self.num_cells), device=self.device,
        )
        s = torch.sparse.mv(M_conn, a_prev)
        active_segments = torch.nonzero(
            s >= self.cfg.segment_activation_threshold, as_tuple=False
        ).squeeze(1)
        if active_segments.numel() == 0:
            return (
                torch.empty(0, dtype=torch.int64, device=self.device),
                torch.empty(0, dtype=torch.int64, device=self.device),
                active_segments,
            )
        predictive_cells = self.seg_owner_cell[active_segments]
        predictive_cells = torch.unique(predictive_cells)
        predictive_columns = torch.unique(predictive_cells // self.cfg.cells_per_column)
        return predictive_cells, predictive_columns, active_segments

    def activate_cells(self, active_columns: torch.Tensor, predictive_prev: torch.Tensor) -> torch.Tensor:
        active_cells: List[int] = []
        pred_cols = predictive_prev // self.cfg.cells_per_column
        for col in active_columns.tolist():
            mask = pred_cols == col
            preds = predictive_prev[mask]
            if preds.numel() > 0:
                active_cells.extend(preds.tolist())
            else:
                base = col * self.cfg.cells_per_column
                active_cells.extend(list(range(base, base + self.cfg.cells_per_column)))
        if not active_cells:
            return torch.empty(0, dtype=torch.int64, device=self.device)
        return torch.tensor(active_cells, dtype=torch.int64, device=self.device)

    def learn(
        self,
        active_cells_prev: Set[int],
        active_columns: torch.Tensor,
        active_cells: torch.Tensor,
        active_segments: torch.Tensor,
        predictive_prev: torch.Tensor,
    ):
        if active_segments.numel() > 0 and active_cells_prev:
            a_prev = torch.zeros(self.num_cells, dtype=torch.bool, device=self.device)
            idx_prev = torch.tensor(list(active_cells_prev), dtype=torch.int64, device=self.device)
            a_prev[idx_prev] = True
            pred_prev_cols = predictive_prev // self.cfg.cells_per_column
            for seg in active_segments.tolist():
                start = int(self.crow_indices[seg])
                end = int(self.crow_indices[seg + 1])
                cols = self.col_indices[start:end]
                perms = self.perm_values[start:end]
                is_prev = a_prev[cols]
                conn = perms >= self.cfg.perm_connected
                seg_active = int((is_prev & conn).sum().item())
                margin = seg_active - self.cfg.segment_activation_threshold
                owner = int(self.seg_owner_cell[seg].item())
                col = owner // self.cfg.cells_per_column
                entropy = int((pred_prev_cols == col).sum().item())
                if self.cfg.meta.enabled:
                    inc = np.full(is_prev.sum().item(), self.cfg.perm_inc, dtype=np.float32)
                    inc = apply_gates(perms[is_prev].detach().cpu().numpy(), inc, margin, entropy, self.cfg.meta)
                    dec = effective_dec(perms[~is_prev].detach().cpu().numpy(), self.cfg.perm_dec, self.cfg.meta)
                    perms[is_prev] += torch.from_numpy(inc).to(self.device)
                    perms[~is_prev] -= torch.from_numpy(dec).to(self.device)
                else:
                    perms[is_prev] += self.cfg.perm_inc
                    perms[~is_prev] -= self.cfg.perm_dec
                torch.clamp_(perms, 0.0, 1.0)

        predicted_cols = set((predictive_prev // self.cfg.cells_per_column).tolist())
        bursting = set(active_columns.tolist()) - predicted_cols
        if bursting and active_cells_prev:
            a_prev_vec = torch.zeros(self.num_cells, dtype=torch.bool, device=self.device)
            idx_prev = torch.tensor(list(active_cells_prev), dtype=torch.int64, device=self.device)
            a_prev_vec[idx_prev] = True
            for col in bursting:
                winner = self._best_matching_cell(col, a_prev_vec)
                self._grow_new_segment(winner, active_cells_prev)

        self.steps_since_flush += 1
        if self.steps_since_flush >= self.flush_interval or len(self.pending_perm) > self.pending_threshold:
            self.flush_pending()

    def _best_matching_cell(self, col: int, a_prev_vec: torch.Tensor) -> int:
        best_cell = None
        best_score = -1
        base = col * self.cfg.cells_per_column
        for i in range(self.cfg.cells_per_column):
            cell_id = base + i
            segs = torch.nonzero(self.seg_owner_cell == cell_id, as_tuple=False).squeeze(1)
            score = 0
            for seg in segs.tolist():
                start = int(self.crow_indices[seg])
                end = int(self.crow_indices[seg + 1])
                cols = self.col_indices[start:end]
                if cols.numel() == 0:
                    continue
                perms = self.perm_values[start:end]
                conn = perms >= self.cfg.perm_connected
                overlap = int((a_prev_vec[cols] & conn).sum().item())
                if overlap > score:
                    score = overlap
            if score > best_score:
                best_score = score
                best_cell = cell_id
        if best_cell is None:
            best_cell = base + int(self.rng.integers(0, self.cfg.cells_per_column))
        return best_cell

    def _grow_new_segment(self, cell_id: int, active_cells_prev: Set[int]):
        pres = np.array(list(active_cells_prev), dtype=np.int64)
        if pres.size == 0:
            return
        if pres.size > self.cfg.distal_synapses_per_segment:
            pres = self.rng.choice(pres, size=self.cfg.distal_synapses_per_segment, replace=False)
        perms = self.rng.normal(
            self.cfg.new_segment_init_perm_mean,
            self.cfg.new_segment_init_perm_sd,
            size=pres.shape[0],
        )
        perms = np.clip(perms, 0.0, 1.0).astype(np.float32)
        row_id = self.num_segments + len(self.pending_owner)
        self.pending_owner.append(cell_id)
        self.pending_row_idx.extend([row_id] * len(pres))
        self.pending_col_idx.extend(pres.tolist())
        self.pending_perm.extend(perms.tolist())

    def flush_pending(self):
        if not self.pending_owner:
            return
        pending_row = torch.tensor(self.pending_row_idx, dtype=torch.int64, device=self.device)
        pending_col = torch.tensor(self.pending_col_idx, dtype=torch.int64, device=self.device)
        pending_perm = torch.tensor(self.pending_perm, dtype=torch.float32, device=self.device)
        new_owner = torch.tensor(self.pending_owner, dtype=torch.int64, device=self.device)
        if self.num_segments > 0:
            existing_row = torch.repeat_interleave(
                torch.arange(self.num_segments, device=self.device, dtype=torch.int64),
                self.crow_indices[1:] - self.crow_indices[:-1],
            )
            row = torch.cat([existing_row, pending_row])
            col = torch.cat([self.col_indices, pending_col])
            val = torch.cat([self.perm_values, pending_perm])
            owner = torch.cat([self.seg_owner_cell, new_owner])
        else:
            row, col, val, owner = pending_row, pending_col, pending_perm, new_owner
        num_segments_new = self.num_segments + len(self.pending_owner)
        M = torch.sparse_coo_tensor(
            torch.stack([row, col]), val, (num_segments_new, self.num_cells), device=self.device,
        )
        M = M.coalesce().to_sparse_csr()
        self.crow_indices = M.crow_indices()
        self.col_indices = M.col_indices()
        self.perm_values = M.values()
        self.seg_owner_cell = owner
        self.num_segments = num_segments_new
        self.pending_row_idx.clear()
        self.pending_col_idx.clear()
        self.pending_perm.clear()
        self.pending_owner.clear()
        self.steps_since_flush = 0


def make_sp_torch(model_cfg: ModelConfig, seed: int, device: str) -> TorchSP:
    rng = seeded_rng(seed)
    return TorchSP(model_cfg, rng, device)


def make_tm_torch(model_cfg: ModelConfig, seed: int, device: str) -> TorchTM:
    rng = seeded_rng(seed)
    try:
        return TorchTM(model_cfg, rng, device)
    except RuntimeError as e:
        if "out of memory" in str(e).lower() and str(device).startswith("cuda"):
            warnings.warn("CUDA OOM, falling back to CPU")
            return TorchTM(model_cfg, rng, "cpu")
        raise

