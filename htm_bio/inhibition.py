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
        ff_threshold: float,
        cells_per_column: int,
        winners_per_column: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        winners = []
        bursting = []
        eps = 1e-6
        for col in active_columns.tolist():
            start = col * cells_per_column
            end = start + cells_per_column
            net = bias[start:end]
            if net.numel() == 0 or net.max() <= 0.0 + eps:
                bursting.append(col)
                continue
            pos_mask = net > 0
            pos_idx = torch.nonzero(pos_mask, as_tuple=False).squeeze(1)
            if pos_idx.numel() == 0:
                bursting.append(col)
                continue
            k = min(winners_per_column, pos_idx.numel())
            topk_local = torch.topk(net[pos_idx], k).indices
            winners.extend((pos_idx[topk_local] + start).tolist())
        device = bias.device
        win_tensor = torch.tensor(winners, dtype=torch.int64, device=device)
        burst_tensor = torch.tensor(bursting, dtype=torch.int64, device=device)
        return win_tensor, burst_tensor
