"""Column-level inhibition."""

from typing import Optional

import torch

from .interfaces import InhibitionModel


class ColumnInhibition(InhibitionModel):
    """Select winners within active columns using distal bias."""

    def __init__(self, inhibition_strength: float, winners_per_column: int = 1):
        self.inhibition_strength = inhibition_strength
        self.winners_per_column = winners_per_column

    def select_winners(
        self,
        active_columns: torch.Tensor,
        distal_bias: torch.Tensor,
        cells_per_column: int,
        winners_per_column: Optional[int] = None,
    ) -> torch.Tensor:
        if winners_per_column is None:
            winners_per_column = self.winners_per_column
        winners = []
        for col in active_columns.tolist():
            start = col * cells_per_column
            end = start + cells_per_column
            biases = distal_bias[start:end]
            k = min(winners_per_column, biases.numel())
            if k == 0:
                continue
            topk = torch.topk(biases, k).indices + start
            winners.extend(topk.tolist())
        if not winners:
            return torch.empty(0, dtype=torch.int64, device=distal_bias.device)
        return torch.tensor(winners, dtype=torch.int64, device=distal_bias.device)
