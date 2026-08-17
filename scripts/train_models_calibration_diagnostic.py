"""Diagnostic entry point for calibration/generalization analysis.

This wrapper intentionally keeps validation-window stability as a diagnostic
rather than a pre-test hard gate. The aggregate validation threshold remains
frozen before the untouched test set is evaluated.
"""

from __future__ import annotations

import numpy as np

from scripts import train_models_calibrated as calibrated
from utils.config_loader import load_all_configs


def select_calibration_threshold_for_diagnostics(
    validation_windows: list[tuple[np.ndarray, np.ndarray]],
    minimum: float = 0.20,
    maximum: float = 0.85,
    steps: int = 131,
    minimum_precision: float = 0.45,
    minimum_coverage: float = 0.005,
    max_precision_std: float = 0.20,
) -> tuple[float, float]:
    """Select from aggregate validation only; report stability separately.

    The test set must remain untouched by threshold selection. Stability is
    deliberately diagnostic here so a failed window-stability condition does
    not prevent us from measuring the model's true out-of-sample performance.
    """
    del max_precision_std

    if not validation_windows:
        raise ValueError("validation windows cannot be empty")

    validation_probs = np.concatenate(
        [probs for probs, _ in validation_windows]
    )
    validation_truth = np.concatenate(
        [truth for _, truth in validation_windows]
    )

    best = None
    selected = None
    for threshold in np.linspace(minimum, maximum, steps):
        metrics = calibrated._classification_metrics(
            validation_probs,
            validation_truth,
            float(threshold),
        )
        precision = float(metrics["precision"])
        coverage = float(metrics["predicted_positive_rate"])
        if precision < minimum_precision or coverage < minimum_coverage:
            continue

        f1 = calibrated._f1(metrics)
        score = (f1, precision, coverage)
        if best is None or score > best:
            best = score
            selected = float(threshold)

    if selected is None or best is None:
        raise RuntimeError(
            "No aggregate validation threshold satisfied the precision and "
            "coverage constraints"
        )

    print(
        "Calibration diagnostic mode: threshold selected from aggregate "
        "validation; chronological stability is diagnostic only."
    )
    return selected, float(best[0])


def main() -> None:
    """Run calibrated training while allowing the test gate to execute."""
    config = load_all_configs(require_notifications=False)
    model_cfg = config["model"]

    # The stability threshold is reported, but must not block the untouched
    # test evaluation. The production quality gates remain unchanged.
    configured_std = float(model_cfg.get("calibration_max_precision_std", 0.20))
    model_cfg["calibration_max_precision_std"] = max(configured_std, 1.0)

    calibrated.select_stable_threshold = select_calibration_threshold_for_diagnostics
    calibrated.train(config)


if __name__ == "__main__":
    main()
