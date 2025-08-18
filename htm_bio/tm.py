"""Torch sparse BIO temporal memory."""

from typing import Set, Tuple

import numpy as np
import torch

from .utils_torch import get_device, set_to_bool_vec
from .metaplasticity_bio import apply_gates, effective_dec, MetaParams
from .interfaces import InhibitionModel


class BioTM:
    def __init__(self, cfg, rng: np.random.Generator, device: str):
        self.cfg = cfg
        self.rng = rng
        self.device = get_device(device)
        self.num_cells = cfg.num_columns * cfg.cells_per_column
        # CSR representation of segments
        self.crow_indices = torch.tensor([0], dtype=torch.int64, device=self.device)
        self.col_indices = torch.tensor([], dtype=torch.int64, device=self.device)
        self.perm_values = torch.tensor([], dtype=torch.float32, device=self.device)
        self.seg_owner_cell = torch.tensor([], dtype=torch.int64, device=self.device)
        self.num_segments = 0
        # pending additions
        self.pending_row_idx: list[int] = []
        self.pending_col_idx: list[int] = []
        self.pending_perm: list[float] = []
        self.pending_owner: list[int] = []
        self.flush_interval = 10
        self.pending_threshold = 1000
        self.steps_since_flush = 0

    # ------------------------------------------------------------------
    def score_segments(self, active_prev: Set[int]) -> torch.Tensor:
        """Return overlap counts for all segments."""
        if self.num_segments == 0 or not active_prev:
            return torch.zeros(0, dtype=torch.float32, device=self.device)
        a_prev = set_to_bool_vec(active_prev, self.num_cells, self.device).to(torch.float32)
        conn = (self.perm_values >= self.cfg.perm_connected).to(torch.float32)
        M_conn = torch.sparse_csr_tensor(
            self.crow_indices, self.col_indices, conn,
            size=(self.num_segments, self.num_cells), device=self.device,
        )
        counts = torch.sparse.mm(M_conn, a_prev.unsqueeze(1)).squeeze(1)
        return counts

    # ------------------------------------------------------------------
    def compute_bias(self, active_prev: Set[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute per-cell bias and return segment counts."""
        counts = self.score_segments(active_prev)
        margins = torch.clamp(counts - self.cfg.segment_activation_threshold, min=0.0)
        bias = torch.zeros(self.num_cells, dtype=torch.float32, device=self.device)
        if margins.numel() > 0:
            contrib = margins * self.cfg.bias_gain
            per_cell = torch.zeros_like(bias)
            per_cell.index_add_(0, self.seg_owner_cell, contrib)
            bias = torch.clamp(per_cell, max=self.cfg.bias_cap)
        return bias, counts

    # ------------------------------------------------------------------
    def activate_cells(
        self, active_columns: torch.Tensor, bias: torch.Tensor, inhibition: InhibitionModel
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select active cells given feed-forward columns and bias."""
        winners, bursting = inhibition.select_winners(
            active_columns,
            bias,
            self.cfg.ff_threshold,
            self.cfg.cells_per_column,
            self.cfg.winners_per_column,
        )
        active = torch.zeros(self.num_cells, dtype=torch.bool, device=self.device)
        if winners.numel() > 0:
            active[winners] = True
        for col in bursting.tolist():
            start = col * self.cfg.cells_per_column
            end = start + self.cfg.cells_per_column
            active[start:end] = True
        return active, bursting

    # ------------------------------------------------------------------
    def learn(
        self,
        active_prev: Set[int],
        active_cells: torch.Tensor,
        bursting_columns: torch.Tensor,
        seg_counts: torch.Tensor,
        meta: MetaParams,
    ) -> None:
        if self.num_segments > 0 and active_prev:
            a_prev_vec = set_to_bool_vec(active_prev, self.num_cells, self.device)
            active_mask = torch.zeros(self.num_segments, dtype=torch.bool, device=self.device)
            active_mask[: seg_counts.shape[0]] = seg_counts > 0
            winner_counts = active_cells.view(self.cfg.num_columns, self.cfg.cells_per_column).sum(dim=1)
            for seg in torch.nonzero(active_mask, as_tuple=False).squeeze(1).tolist():
                owner = int(self.seg_owner_cell[seg].item())
                if not active_cells[owner]:
                    continue
                start = int(self.crow_indices[seg])
                end = int(self.crow_indices[seg + 1])
                cols = self.col_indices[start:end]
                perms = self.perm_values[start:end]
                if cols.numel() == 0:
                    continue
                is_prev = a_prev_vec[cols]
                margin = int(seg_counts[seg].item() - self.cfg.segment_activation_threshold)
                col = owner // self.cfg.cells_per_column
                entropy = int(winner_counts[col].item())
                if meta.enabled:
                    inc = np.full(is_prev.sum().item(), self.cfg.perm_inc, dtype=np.float32)
                    inc = apply_gates(perms[is_prev].detach().cpu().numpy(), inc, margin, entropy, meta)
                    dec = effective_dec(perms[~is_prev].detach().cpu().numpy(), self.cfg.perm_dec, meta)
                    perms[is_prev] += torch.from_numpy(inc).to(self.device)
                    perms[~is_prev] -= torch.from_numpy(dec).to(self.device)
                else:
                    perms[is_prev] += self.cfg.perm_inc
                    perms[~is_prev] -= self.cfg.perm_dec
                torch.clamp_(perms, 0.0, 1.0)
        # bursting growth
        if bursting_columns.numel() > 0 and active_prev:
            a_prev_vec = set_to_bool_vec(active_prev, self.num_cells, self.device)
            for col in bursting_columns.tolist():
                winner = self._best_matching_cell(col, a_prev_vec)
                self._grow_new_segment(winner, active_prev)
        self.steps_since_flush += 1
        if self.steps_since_flush >= self.flush_interval or len(self.pending_perm) > self.pending_threshold:
            self.flush_pending()

    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    def _grow_new_segment(self, cell_id: int, active_prev: Set[int]) -> None:
        pres = np.array(list(active_prev), dtype=np.int64)
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

    # ------------------------------------------------------------------
    def flush_pending(self) -> None:
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
