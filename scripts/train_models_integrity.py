"""Train the GRU with direction/event-aware, integrity-preserving labels."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.candidate_context import CONTEXT_COLUMNS, CONTEXT_VERSION, augment_sequence, context_vector, validate_candidate_context
from models.feature_engineering import FEATURE_COLUMNS, FEATURE_VERSION, FeatureEngineer
from models.gru import SignalValidatorGRU
from scripts.train_models import _classification_metrics, build_dataset, build_labeled_dataset, sha256_file
from utils.config_loader import load_all_configs


def _f1(metrics: dict) -> float:
    precision = float(metrics["precision"])
    recall = float(metrics["recall"])
    return 0.0 if precision + recall <= 0 else 2 * precision * recall / (precision + recall)


def _make_sequences(features: np.ndarray, labels, engineer: FeatureEngineer):
    """Create sequences keyed by (candidate_index, direction), never index alone."""
    validate_candidate_context(labels)
    scaled = engineer.scaler.transform(features)
    if not np.isfinite(scaled).all():
        raise RuntimeError("Feature scaler produced non-finite values")
    sequences, contexts, targets = [], [], []
    for row in labels.itertuples(index=False):
        index = int(row.candidate_index)
        start = index - engineer.sequence_length + 1
        if start < 0:
            raise RuntimeError(f"Candidate index {index} lacks a complete causal sequence")
        sequence = scaled[start : index + 1]
        if len(sequence) != engineer.sequence_length:
            raise RuntimeError(f"Invalid sequence length for candidate {index}: {len(sequence)}")
        sequences.append(augment_sequence(sequence, row.direction, row.event_type))
        contexts.append(context_vector(row.direction, row.event_type))
        targets.append(float(row.label))
    return (
        torch.tensor(np.stack(sequences), dtype=torch.float32),
        torch.tensor(np.stack(contexts), dtype=torch.float32),
        torch.tensor(np.asarray(targets)[:, None], dtype=torch.float32),
    )


def _build_partitions(config: dict):
    frames = build_dataset(config)
    sequence_length = int(config["model"]["sequence_length"])
    datasets = []
    train_feature_frames = []
    statistics = []
    for frame in frames:
        features, labels = build_labeled_dataset(frame, config)
        validate_candidate_context(labels)
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
            "direction_counts": {d: int((labels["direction"] == d).sum()) for d in ("LONG", "SHORT")},
            "event_type_counts": {str(e): int((labels["event_type"] == e).sum()) for e in sorted(labels["event_type"].unique())},
        })
    engineer = FeatureEngineer(sequence_length)
    engineer.scaler.fit(np.concatenate([x.to_numpy() for x in train_feature_frames], axis=0))
    partition_lists = {"train": [[], [], []], "validation": [[], [], []], "test": [[], [], []]}
    for features, labels, split1, split2 in datasets:
        selections = {"train": labels.iloc[:split1], "validation": labels.iloc[split1:split2], "test": labels.iloc[split2:]}
        for name, selected in selections.items():
            x, c, y = _make_sequences(features.to_numpy(), selected, engineer)
            partition_lists[name][0].append(x)
            partition_lists[name][1].append(c)
            partition_lists[name][2].append(y)
    result = []
    for name in ("train", "validation", "test"):
        groups = partition_lists[name]
        if any(not items for items in groups):
            raise RuntimeError(f"Empty {name} partition")
        result.extend(torch.cat(items) for items in groups)
    return (*result, engineer, statistics)


def _select_threshold(probs: np.ndarray, truth: np.ndarray, minimum_precision: float, minimum_coverage: float):
    best = None
    for threshold in np.linspace(0.20, 0.85, 131):
        metrics = _classification_metrics(probs, truth, float(threshold))
        if metrics["precision"] < minimum_precision or metrics["predicted_positive_rate"] < minimum_coverage:
            continue
        score = (_f1(metrics), float(metrics["precision"]))
        if best is None or score > best[0]:
            best = (score, float(threshold))
    if best is None:
        raise RuntimeError("No validation threshold satisfied precision and coverage constraints")
    return best[1], best[0][0]


def train(config: dict) -> None:
    X_train, C_train, y_train, X_val, C_val, y_val, X_test, C_test, y_test, engineer, statistics = _build_partitions(config)
    positive = float(y_train.sum())
    negative = float(y_train.numel() - positive)
    if positive <= 0 or negative <= 0:
        raise RuntimeError("Training partition must contain both classes")
    positive_weight = negative / positive
    print(f"Dataset sizes: train={len(X_train)} validation={len(X_val)} test={len(X_test)}")
    print(f"Training class balance: positive_rate={positive / (positive + negative):.4f} positive_weight={positive_weight:.4f}")
    print(f"Candidate context: version={CONTEXT_VERSION} columns={list(CONTEXT_COLUMNS)}")
    model = SignalValidatorGRU(input_dim=len(FEATURE_COLUMNS), context_dim=len(CONTEXT_COLUMNS))
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.BCELoss(reduction="none")
    best_state = None
    best_val_auc = -1.0
    for epoch in range(20):
        model.train()
        optimizer.zero_grad()
        predictions = model(X_train, event_context=C_train)
        weights = torch.where(y_train > 0.5, torch.full_like(y_train, positive_weight), torch.ones_like(y_train))
        loss = (criterion(predictions, y_train) * weights).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("Training loss became non-finite")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_probs_epoch = model(X_val, event_context=C_val).cpu().numpy().ravel()
        val_truth = y_val.cpu().numpy().ravel().astype(int)
        metrics = _classification_metrics(val_probs_epoch, val_truth, 0.5)
        print(f"epoch={epoch + 1} loss={loss.item():.5f} val_auc={metrics['roc_auc']:.4f} val_positive_rate={metrics['positive_rate']:.4f}")
        if metrics["roc_auc"] > best_val_auc:
            best_val_auc = metrics["roc_auc"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("Training produced no valid model state")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_probs = model(X_val, event_context=C_val).cpu().numpy().ravel()
        test_probs = model(X_test, event_context=C_test).cpu().numpy().ravel()
    val_truth = y_val.cpu().numpy().ravel().astype(int)
    test_truth = y_test.cpu().numpy().ravel().astype(int)
    cfg = config["model"]
    minimum_precision = float(cfg.get("minimum_test_precision", 0.45))
    minimum_coverage = float(cfg.get("calibration_minimum_coverage", 0.005))
    threshold, validation_f1 = _select_threshold(val_probs, val_truth, minimum_precision, minimum_coverage)
    val_metrics = _classification_metrics(val_probs, val_truth, threshold)
    test_metrics = _classification_metrics(test_probs, test_truth, threshold)
    windows = np.array_split(np.arange(len(val_probs)), int(cfg.get("calibration_validation_windows", 3)))
    window_diagnostics = []
    for number, indices in enumerate(windows, 1):
        metrics = _classification_metrics(val_probs[indices], val_truth[indices], threshold)
        window_diagnostics.append({"window": number, "samples": int(metrics["samples"]), "positive_rate": float(metrics["positive_rate"]), "precision": float(metrics["precision"]), "recall": float(metrics["recall"]), "f1": _f1(metrics), "predicted_positive_rate": float(metrics["predicted_positive_rate"])})
    print(f"Decision threshold selected from aggregate validation: threshold={threshold:.4f} validation_f1={validation_f1:.4f}")
    for row in window_diagnostics:
        print(f"Validation window {row['window']}: precision={row['precision']:.4f} recall={row['recall']:.4f} f1={row['f1']:.4f} coverage={row['predicted_positive_rate']:.4f}")
    print(f"Validation metrics at frozen threshold: precision={val_metrics['precision']:.4f} recall={val_metrics['recall']:.4f} f1={_f1(val_metrics):.4f} coverage={val_metrics['predicted_positive_rate']:.4f}")
    print(f"Test metrics at frozen validation threshold: roc_auc={test_metrics['roc_auc']:.4f} precision={test_metrics['precision']:.4f} recall={test_metrics['recall']:.4f} predicted_positive_rate={test_metrics['predicted_positive_rate']:.4f}")
    minimum_auc = float(cfg.get("minimum_test_auc", 0.55))
    if test_metrics["roc_auc"] < minimum_auc or test_metrics["precision"] < minimum_precision:
        raise RuntimeError(f"Model rejected at frozen validation threshold {threshold:.4f}: roc_auc={test_metrics['roc_auc']:.4f} required>={minimum_auc:.4f}; precision={test_metrics['precision']:.4f} required>={minimum_precision:.4f}")
    weights_path = Path(cfg["path"])
    scaler_path = Path(cfg["scaler_path"])
    metadata_path = Path(cfg["metadata_path"])
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), weights_path)
    engineer.save_scaler(str(scaler_path))
    metadata = {
        "model_version": dt.datetime.now(dt.timezone.utc).strftime("gru-%Y%m%dT%H%M%SZ"),
        "feature_version": FEATURE_VERSION,
        "feature_columns": FEATURE_COLUMNS,
        "candidate_context_version": CONTEXT_VERSION,
        "candidate_context_columns": list(CONTEXT_COLUMNS),
        "sequence_length": int(cfg["sequence_length"]),
        "decision_threshold": threshold,
        "validation": {**val_metrics, "f1": _f1(val_metrics)},
        "test": {**test_metrics, "f1": _f1(test_metrics)},
        "validation_windows": window_diagnostics,
        "validation_precision_std": float(np.std([row["precision"] for row in window_diagnostics])),
        "threshold_search": {"minimum": 0.20, "maximum": 0.85, "steps": 131, "minimum_coverage": minimum_coverage},
        "trained_symbols": [item["symbol"] for item in statistics],
        "total_labeled_candidates": int(sum(item["labeled_candidates"] for item in statistics)),
        "symbol_statistics": statistics,
        "training_positive_weight": positive_weight,
        "training_positive_rate": positive / (positive + negative),
    }
    metadata["model_sha256"] = sha256_file(weights_path)
    metadata["scaler_sha256"] = sha256_file(scaler_path)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Validated directional model written: {weights_path}")


if __name__ == "__main__":
    train(load_all_configs(require_notifications=False))
