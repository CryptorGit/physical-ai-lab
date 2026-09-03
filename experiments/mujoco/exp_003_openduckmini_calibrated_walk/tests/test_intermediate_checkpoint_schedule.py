from pathlib import Path
import sys

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT))

from training.checkpointing import checkpoint_name, crossed_thresholds


def test_threshold_crossing_records_requested_and_actual():
    thresholds = (0, 50_000, 100_000, 250_000, 500_000, 1_000_000)
    assert crossed_thresholds(0, 50_000, thresholds) == (50_000,)
    assert crossed_thresholds(50_000, 100_000, thresholds) == (100_000,)
    assert crossed_thresholds(230_000, 280_000, thresholds) == (250_000,)
    assert (
        checkpoint_name(250_000, 280_000)
        == "requested_0000250000_actual_0000280000"
    )

