import numpy as np
import torch

from scripts.train_models import (
    _classification_metrics,
    select_decision_threshold,
)


def test_classification_metrics_uses_supplied_threshold():
    # At threshold 0.75 only 0.81 is predicted positive. The corresponding
    # truth value must therefore be 1 for precision=1.0 and recall=0.5.
    probs = np.array([0.51, 0.55, 0.81, 0.74])
    truth = np.array([1, 0, 1, 0])

    metrics = _classification_metrics(
        probs,
        truth,
        threshold=0.75,
    )

    assert metrics["threshold"] == 0.75
    assert metrics["predicted_positive_rate"] == 0.25
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 0.5


def test_validation_threshold_is_frozen_before_test_gate():
    validation_probs = np.array([
        0.51,
        0.52,
        0.74,
        0.76,
        0.78,
        0.82,
    ])
    validation_truth = np.array([1, 0, 1, 1, 0, 0])

    threshold, f1 = select_decision_threshold(
        validation_probs,
        validation_truth,
        minimum=0.50,
        maximum=0.80,
        steps=31,
    )

    assert 0.50 <= threshold <= 0.80
    assert f1 >= 0.0

    test_probs = np.array([
        0.49,
        0.61,
        0.79,
        0.83,
    ])
    test_truth = np.array([0, 1, 1, 0])

    test_metrics = _classification_metrics(
        test_probs,
        test_truth,
        threshold=threshold,
    )

    assert test_metrics["threshold"] == threshold
    assert test_metrics["samples"] == len(test_truth)
    assert np.isfinite(test_metrics["roc_auc"])


def test_threshold_search_does_not_consume_test_data():
    validation_probs = np.array([0.55, 0.60, 0.70, 0.80])
    validation_truth = np.array([1, 0, 1, 0])

    threshold_before, f1_before = select_decision_threshold(
        validation_probs,
        validation_truth,
    )

    # Test observations are deliberately unrelated. The selected threshold
    # must depend only on validation probabilities and labels.
    _ = _classification_metrics(
        np.array([0.20, 0.40, 0.90, 0.95]),
        np.array([0, 1, 1, 0]),
        threshold=threshold_before,
    )

    threshold_after, f1_after = select_decision_threshold(
        validation_probs,
        validation_truth,
    )

    assert threshold_after == threshold_before
    assert f1_after == f1_before
