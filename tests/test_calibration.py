import numpy as np

from scripts.train_models_calibrated import (
    calibration_diagnostics,
    select_stable_threshold,
)


def test_calibration_diagnostics_reports_threshold_grid():
    probs = np.array([0.20, 0.40, 0.60, 0.70, 0.80, 0.90])
    truth = np.array([0, 0, 1, 1, 0, 1])

    diagnostics = calibration_diagnostics(
        probs,
        truth,
        thresholds=[0.50, 0.70, 0.80],
    )

    assert [row["threshold"] for row in diagnostics] == [0.50, 0.70, 0.80]
    assert all("precision" in row for row in diagnostics)
    assert all("recall" in row for row in diagnostics)
    assert all("f1" in row for row in diagnostics)
    assert all("predicted_positive_rate" in row for row in diagnostics)


def test_stable_threshold_uses_validation_windows_only():
    windows = [
        (
            np.array([0.20, 0.60, 0.80, 0.90]),
            np.array([0, 1, 1, 0]),
        ),
        (
            np.array([0.30, 0.65, 0.82, 0.95]),
            np.array([0, 1, 1, 0]),
        ),
    ]

    threshold, score = select_stable_threshold(
        windows,
        minimum=0.50,
        maximum=0.90,
        steps=9,
        minimum_precision=0.45,
        minimum_coverage=0.0,
    )

    assert 0.50 <= threshold <= 0.90
    assert score >= 0.0


def test_stable_threshold_rejects_empty_windows():
    try:
        select_stable_threshold([])
    except ValueError as exc:
        assert "validation windows" in str(exc)
    else:
        raise AssertionError("Expected ValueError for empty validation windows")
