"""Column-level inhibition."""

import torch

from .interfaces import InhibitionModel


class ColumnInhibition(InhibitionModel):
    """Deterministic top-k selection per column."""

    def __init__(self, inhibition_strength: float, winners_per_column: int = 1):
        self.inhibition_strength = inhibition_strength
        self.winners_per_column = winners_per_column

    def select_winners(
        self,
        active_columns: torch.Tensor,
        bias: torch.Tensor,
        cells_per_column: int,
        winners_per_column: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        winners = []
        bursting = []
        bias_eps = 1e-9
        for col in active_columns.tolist():
            start = col * cells_per_column
            end = start + cells_per_column
            b = bias[start:end]
            if b.numel() == 0 or b.max() <= bias_eps:
                bursting.append(col)
                continue
            pos_idx = torch.nonzero(b > bias_eps, as_tuple=False).squeeze(1)
            if pos_idx.numel() == 0:
                bursting.append(col)
                continue
            k = min(winners_per_column, pos_idx.numel())
            topk_local = torch.topk(b[pos_idx], k).indices
            winners.extend((pos_idx[topk_local] + start).tolist())
        device = bias.device
        win_tensor = torch.tensor(winners, dtype=torch.int64, device=device)
        burst_tensor = torch.tensor(bursting, dtype=torch.int64, device=device)
        return win_tensor, burst_tensor
