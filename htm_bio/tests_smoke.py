"""Tiny smoke test for the BIO runner."""

from htm_bio.config import BioModelConfig, BioRunConfig
from htm_bio import runner


def test_smoke() -> None:
    mc = BioModelConfig()
    rc = BioRunConfig(steps=4, dry_run=False)
    runner.main(mc, rc)


if __name__ == "__main__":
    test_smoke()
