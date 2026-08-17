"""Calibrated GRU training entrypoint.

This module is intentionally separate from the existing trainer so the
original implementation remains recoverable while the new calibration path
is exercised in CI. It uses class-weighted BCE during fitting and selects the
operating threshold from validation data only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from scripts.train_models import (
    _classification_metrics,
    build_dataset,
    build_labeled_dataset,
    make_sequences,
    sha256_file,
)
from models.feature_engineering import FEATURE_COLUMNS, FEATURE_VERSION, FeatureEngineer
from models.gru import SignalValidatorGRU
from utils.config_loader import load_all_configs


def select_calibrated_threshold(
    probs: np.ndarray,
    truth: np.ndarray,
    minimum: float = 0.20,
    maximum: float = 0.85,
    steps: int = 131,
    minimum_precision: float = 0.45,
) -> tuple[float, float]:
    """Select a validation-only threshold.

    The primary objective is F1 among thresholds meeting the configured
    precision floor. If no threshold meets the floor, the best-F1 threshold is
    returned and the caller rejects the model before test promotion.
    """
    probs = np.asarray(probs, dtype=float).ravel()
    truth = np.asarray(truth, dtype=int).ravel()
    if len(probs) != len(truth) or len(probs) == 0:
        raise ValueError("Validation probabilities and labels are invalid")
    if steps < 2 or not 0.0 < minimum < maximum < 1.0:
        raise ValueError("Invalid threshold search configuration")

    best = None
    fallback = (-1.0, float(minimum))
    for threshold in np.linspace(minimum, maximum, steps):
        metrics = _classification_metrics(probs, truth, float(threshold))
        f1 = float(2.0 * metrics["precision"] * metrics["recall"] /
                   max(metrics["precision"] + metrics["recall"], 1e-12))
        if f1 > fallback[0]:
            fallback = (f1, float(threshold))
        if metrics["precision"] >= minimum_precision and (best is None or f1 > best[0]):
            best = (f1, float(threshold))

    if best is None:
        return fallback[1], fallback[0]
    return best[1], best[0]


def _build_partitions(config: dict):
    """Build chronological train/validation/test tensors and scaler."""
    frames = build_dataset(config)
    sequence_length = int(config["model"]["sequence_length"])
    datasets = []
    train_feature_frames = []
    statistics = []
    total_labeled = 0

    for frame in frames:
        features, labels = build_labeled_dataset(frame, config)
        split1 = int(len(labels) * 0.70)
        split2 = int(len(labels) * 0.85)
        if split1 < 100 or split2 - split1 < 40 or len(labels) - split2 < 40:
            raise RuntimeError(f"Chronological split too small: {len(labels)}")
        datasets.append((features, labels, split1, split2))
        train_end = int(labels.iloc[split1 - 1]["candidate_index"]) + 1
        train_feature_frames.append(features.iloc[:train_end])
        symbol = frame["symbol"].iloc[0] if "symbol" in frame.columns else "UNKNOWN"
        statistics.append({
            "symbol": symbol,
            "labeled_candidates": int(len(labels)),
            "positive_labels": int(labels["label"].sum()),
            "negative_labels": int((labels["label"] == 0).sum()),
            "positive_rate": float(labels["label"].mean()),
            "train_candidates": split1,
            "validation_candidates": split2 - split1,
            "test_candidates": len(labels) - split2,
        })
        total_labeled += len(labels)

    if not train_feature_frames:
        raise RuntimeError("No training feature frames available")

    engineer = FeatureEngineer(sequence_length)
    engineer.scaler.fit(np.concatenate([x.to_numpy() for x in train_feature_frames], axis=0))

    train_x, train_y, val_x, val_y, test_x, test_y = [], [], [], [], [], []
    for features, labels, split1, split2 in datasets:
        for destination_x, destination_y, selected in (
            (train_x, train_y, labels.iloc[:split1]),
            (val_x, val_y, labels.iloc[split1:split2]),
            (test_x, test_y, labels.iloc[split2:]),
        ):
            x, y = make_sequences(
                features.to_numpy(),
                selected,
                selected["candidate_index"].to_numpy(),
                engineer,
            )
            destination_x.append(x)
            destination_y.append(y)

    return (
        torch.cat(train_x), torch.cat(train_y),
        torch.cat(val_x), torch.cat(val_y),
        torch.cat(test_x), torch.cat(test_y),
        engineer, statistics, total_labeled,
    )


def train(config: dict) -> None:
    """Train with class weighting and validation-only threshold calibration."""
    (
        X_train, y_train, X_val, y_val, X_test, y_test,
        engineer, statistics, total_labeled,
    ) = _build_partitions(config)

    positive = float(y_train.sum().item())
    negative = float(y_train.numel() - positive)
    if positive <= 0 or negative <= 0:
        raise RuntimeError("Training partition must contain both classes")
    positive_weight = negative / positive

    print(
        f"Dataset sizes: train={len(X_train)} validation={len(X_val)} test={len(X_test)}"
    )
    print(
        f"Training class balance: positive_rate={positive / (positive + negative):.4f} "
        f"positive_weight={positive_weight:.4f}"
    )

    model = SignalValidatorGRU(input_dim=len(FEATURE_COLUMNS))
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.BCELoss(reduction="none")

    best_state = None
    best_val_auc = -1.0
    for epoch in range(20):
        model.train()
        optimizer.zero_grad()
        predictions = model(X_train)
        if not torch.isfinite(predictions).all():
            raise RuntimeError("Model produced non-finite training predictions")
        sample_weights = torch.where(
            y_train > 0.5,
            torch.full_like(y_train, positive_weight),
            torch.ones_like(y_train),
        )
        loss = (criterion(predictions, y_train) * sample_weights).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("Training loss became non-finite")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_probs = model(X_val).cpu().numpy().ravel()
        val_truth = y_val.cpu().numpy().ravel().astype(int)
        base = _classification_metrics(val_probs, val_truth, 0.5)
        print(
            f"epoch={epoch + 1} loss={loss.item():.5f} "
            f"val_auc={base['roc_auc']:.4f} "
            f"val_positive_rate={base['positive_rate']:.4f}"
        )
        if base["roc_auc"] > best_val_auc:
            best_val_auc = base["roc_auc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training produced no valid model state")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_probs = model(X_val).cpu().numpy().ravel()
        test_probs = model(X_test).cpu().numpy().ravel()
    val_truth = y_val.cpu().numpy().ravel().astype(int)
    test_truth = y_test.cpu().numpy().ravel().astype(int)

    minimum_precision = float(config["model"].get("minimum_test_precision", 0.45))
    threshold, validation_f1 = select_calibrated_threshold(
        val_probs,
        val_truth,
        minimum=float(config["model"].get("threshold_minimum", 0.20)),
        maximum=float(config["model"].get("threshold_maximum", 0.85)),
        steps=int(config["model"].get("threshold_steps", 131)),
        minimum_precision=minimum_precision,
    )

    val_metrics = _classification_metrics(val_probs, val_truth, threshold)
    test_metrics = _classification_metrics(test_probs, test_truth, threshold)

    print(
        f"Decision threshold selected from validation: threshold={threshold:.4f} "
        f"validation_f1={validation_f1:.4f}"
    )
    print(
        f"Validation metrics at frozen threshold: precision={val_metrics['precision']:.4f} "
        f"recall={val_metrics['recall']:.4f} f1={validation_f1:.4f}"
    )
    print(
        f"Test metrics at frozen validation threshold: roc_auc={test_metrics['roc_auc']:.4f} "
        f"precision={test_metrics['precision']:.4f} recall={test_metrics['recall']:.4f} "
        f"predicted_positive_rate={test_metrics['predicted_positive_rate']:.4f}"
    )

    if val_metrics["precision"] < minimum_precision or val_metrics["predicted_positive_rate"] <= 0.0:
        raise RuntimeError(
            f"Model rejected before test promotion: validation precision={val_metrics['precision']:.4f} "
            f"predicted_positive_rate={val_metrics['predicted_positive_rate']:.4f}"
        )

    minimum_auc = float(config["model"].get("minimum_test_auc", 0.55))
    failures = []
    if test_metrics["roc_auc"] < minimum_auc:
        failures.append(f"ROC-AUC {test_metrics['roc_auc']:.4f} < {minimum_auc:.4f}")
    if test_metrics["precision"] < minimum_precision:
        failures.append(f"precision {test_metrics['precision']:.4f} < {minimum_precision:.4f}")
    if failures:
        raise RuntimeError(
            f"Model rejected at frozen validation threshold {threshold:.4f}: " + "; ".join(failures)
        )

    weights = Path(config["model"]["path"])
    scaler_path = Path(config["model"]["scaler_path"])
    metadata_path = Path(config["model"]["metadata_path"])
    weights.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), weights)
    engineer.save_scaler(str(scaler_path))

    metadata = {
        "model_version": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).strftime("gru-%Y%m%dT%H%M%SZ"),
        "feature_version": FEATURE_VERSION,
        "feature_columns": FEATURE_COLUMNS,
        "sequence_length": int(config["model"]["sequence_length"]),
        "decision_threshold": threshold,
        "validation": val_metrics,
        "validation_threshold_selection_f1": validation_f1,
        "test": test_metrics,
        "trained_symbols": config["trading"]["symbols"],
        "training_candles_per_symbol": config["market_data"]["training_history_candles"],
        "total_labeled_candidates": total_labeled,
        "symbol_statistics": statistics,
        "candidate_policy": config["training_events"],
        "training_positive_rate": positive / (positive + negative),
        "training_positive_weight": positive_weight,
        "threshold_search": {
            "minimum": float(config["model"].get("threshold_minimum", 0.20)),
            "maximum": float(config["model"].get("threshold_maximum", 0.85)),
            "steps": int(config["model"].get("threshold_steps", 131)),
            "minimum_precision": minimum_precision,
        },
        "label": "TP-before-SL on causal SMC event and continuation candidates",
        "model_sha256": sha256_file(weights),
        "scaler_sha256": sha256_file(scaler_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    train(load_all_configs(require_notifications=False))
