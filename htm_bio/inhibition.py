"""Column-level inhibition stubs."""

from .interfaces import InhibitionModel


class ColumnInhibition(InhibitionModel):
    """Placeholder inhibition model.

    In dry-run mode, this does nothing and logs a message.
    """

    def __init__(self, inhibition_strength: float, winners_per_column: int = 1, dry_run: bool = True):
        self.inhibition_strength = inhibition_strength
        self.winners_per_column = winners_per_column
        self.dry_run = dry_run

    def select_winners(self, ff_drive, distal_bias, cells_per_column, winners_per_column=None):
        if winners_per_column is None:
            winners_per_column = self.winners_per_column
        if self.dry_run:
            print("BIO stub: inhibition no-op")
            return []
        return []
