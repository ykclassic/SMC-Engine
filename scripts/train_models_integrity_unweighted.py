"""Train the integrity GRU without aggressive positive-class reweighting.

The production precision gate remains unchanged. This entry point reuses the
canonical dataset, causal partitioning, threshold diagnostics, and OOS gates
from train_models_integrity.py, but trains with ordinary BCE so probabilities
remain usable for precision-oriented threshold calibration.

When ``--experiment-mode`` is supplied, the exact same training, validation,
and test loops are executed, but a failing production gate is recorded in an
experiment-results JSON artifact instead of terminating the process. Experiment
artifacts are written beside that JSON output and never overwrite production
model paths.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.candidate_context import CONTEXT_COLUMNS, CONTEXT_VERSION
from models.feature_engineering import FEATURE_COLUMNS, FEATURE_VERSION
from models.gru import SignalValidatorGRU
from scripts.train_models import _classification_metrics, sha256_file
from scripts.train_models_integrity import _build_partitions, _f1, _select_threshold
from utils.config_loader import load_all_configs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the integrity GRU."
    )
    parser.add_argument(
        "--experiment-mode",
        action="store_true",
        help=(
            "Run the normal training/validation/test loops but do not fail "
            "the process when the production test gate rejects the model. "
            "Evaluation metrics are written to --output."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Alias for --experiment-mode.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/ablation_results.json"),
        help="Path for the experiment evaluation JSON artifact.",
    )
    return parser.parse_args()


def _write_experiment_results(
    output_path: Path,
    *,
    gate_passed: bool,
    rejection_reason: str | None,
    threshold: float,
    validation_f1: float,
    val_metrics: dict,
    test_metrics: dict,
    window_diagnostics: list[dict],
    minimum_auc: float,
    minimum_precision: float,
    minimum_coverage: float,
    positive_weight: float,
    training_positive_rate: float,
    statistics: list[dict],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "integrity-training-experiment-v1",
        "experiment_mode": True,
        "gate_passed": gate_passed,
        "rejection_reason": rejection_reason,
        "decision_threshold": float(threshold),
        "validation_f1_at_threshold": float(validation_f1),
        "validation": {
            **val_metrics,
            "f1": _f1(val_metrics),
        },
        "test": {
            **test_metrics,
            "f1": _f1(test_metrics),
        },
        "validation_windows": window_diagnostics,
        "gates": {
            "minimum_test_auc": minimum_auc,
            "minimum_test_precision": minimum_precision,
            "minimum_coverage": minimum_coverage,
        },
        "training": {
            "positive_weight": positive_weight,
            "positive_rate": training_positive_rate,
        },
        "feature_version": FEATURE_VERSION,
        "feature_columns": FEATURE_COLUMNS,
        "candidate_context_version": CONTEXT_VERSION,
        "candidate_context_columns": list(CONTEXT_COLUMNS),
        "trained_symbols": [item["symbol"] for item in statistics],
        "total_labeled_candidates": int(
            sum(item["labeled_candidates"] for item in statistics)
        ),
        "symbol_statistics": statistics,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Experiment evaluation written: {output_path}")


def train(
    config: dict,
    *,
    experiment_mode: bool = False,
    output_path: Path | None = None,
) -> None:
    (
        X_train,
        C_train,
        y_train,
        X_val,
        C_val,
        y_val,
        X_test,
        C_test,
        y_test,
        engineer,
        statistics,
    ) = _build_partitions(config)

    positive = float(y_train.sum())
    negative = float(y_train.numel() - positive)
    if positive <= 0 or negative <= 0:
        raise RuntimeError("Training partition must contain both classes")

    cfg = config["model"]
    minimum_precision = float(cfg.get("minimum_test_precision", 0.45))
    minimum_coverage = float(cfg.get("calibration_minimum_coverage", 0.005))
    positive_weight = float(cfg.get("positive_class_weight", 1.0))
    if not np.isfinite(positive_weight) or positive_weight <= 0:
        raise RuntimeError(
            f"positive_class_weight must be a finite positive number: {positive_weight}"
        )

    print(
        f"Dataset sizes: train={len(X_train)} validation={len(X_val)} "
        f"test={len(X_test)}"
    )
    print(
        f"Training class balance: positive_rate={positive / (positive + negative):.4f} "
        f"positive_weight={positive_weight:.4f}"
    )
    print(
        f"Candidate context: version={CONTEXT_VERSION} "
        f"columns={list(CONTEXT_COLUMNS)}"
    )

    model = SignalValidatorGRU(
        input_dim=len(FEATURE_COLUMNS),
        context_dim=len(CONTEXT_COLUMNS),
    )
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.BCELoss()
    best_state = None
    best_val_auc = -1.0

    for epoch in range(20):
        model.train()
        optimizer.zero_grad()
        predictions = model(X_train, event_context=C_train)
        if positive_weight == 1.0:
            loss = criterion(predictions, y_train)
        else:
            weights = torch.where(
                y_train > 0.5,
                torch.full_like(y_train, positive_weight),
                torch.ones_like(y_train),
            )
            loss = (
                nn.BCELoss(reduction="none")(predictions, y_train) * weights
            ).mean()

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
        print(
            f"epoch={epoch + 1} loss={loss.item():.5f} "
            f"val_auc={metrics['roc_auc']:.4f} "
            f"val_positive_rate={metrics['positive_rate']:.4f}"
        )
        if metrics["roc_auc"] > best_val_auc:
            best_val_auc = metrics["roc_auc"]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("Training produced no valid model state")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_probs = model(X_val, event_context=C_val).cpu().numpy().ravel()
        test_probs = model(X_test, event_context=C_test).cpu().numpy().ravel()

    val_truth = y_val.cpu().numpy().ravel().astype(int)
    test_truth = y_test.cpu().numpy().ravel().astype(int)
    threshold, validation_f1 = _select_threshold(
        val_probs,
        val_truth,
        minimum_precision,
        minimum_coverage,
    )

    val_metrics = _classification_metrics(val_probs, val_truth, threshold)
    test_metrics = _classification_metrics(test_probs, test_truth, threshold)

    windows = np.array_split(
        np.arange(len(val_probs)),
        int(cfg.get("calibration_validation_windows", 3)),
    )
    window_diagnostics = []
    for number, indices in enumerate(windows, 1):
        metrics = _classification_metrics(
            val_probs[indices],
            val_truth[indices],
            threshold,
        )
        window_diagnostics.append(
            {
                "window": number,
                "samples": int(metrics["samples"]),
                "positive_rate": float(metrics["positive_rate"]),
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "f1": _f1(metrics),
                "predicted_positive_rate": float(
                    metrics["predicted_positive_rate"]
                ),
            }
        )

    print(
        f"Decision threshold selected from aggregate validation: "
        f"threshold={threshold:.4f} validation_f1={validation_f1:.4f}"
    )
    for row in window_diagnostics:
        print(
            f"Validation window {row['window']}: "
            f"precision={row['precision']:.4f} "
            f"recall={row['recall']:.4f} "
            f"f1={row['f1']:.4f} "
            f"coverage={row['predicted_positive_rate']:.4f}"
        )
    print(
        f"Validation metrics at frozen threshold: "
        f"precision={val_metrics['precision']:.4f} "
        f"recall={val_metrics['recall']:.4f} "
        f"f1={_f1(val_metrics):.4f} "
        f"coverage={val_metrics['predicted_positive_rate']:.4f}"
    )
    print(
        f"Test metrics at frozen validation threshold: "
        f"roc_auc={test_metrics['roc_auc']:.4f} "
        f"precision={test_metrics['precision']:.4f} "
        f"recall={test_metrics['recall']:.4f} "
        f"predicted_positive_rate={test_metrics['predicted_positive_rate']:.4f}"
    )

    minimum_auc = float(cfg.get("minimum_test_auc", 0.55))
    gate_passed = (
        test_metrics["roc_auc"] >= minimum_auc
        and test_metrics["precision"] >= minimum_precision
    )
    rejection_reason = None
    if not gate_passed:
        rejection_reason = (
            f"Model rejected at frozen validation threshold {threshold:.4f}: "
            f"roc_auc={test_metrics['roc_auc']:.4f} "
            f"required>={minimum_auc:.4f}; "
            f"precision={test_metrics['precision']:.4f} "
            f"required>={minimum_precision:.4f}"
        )

    if experiment_mode:
        experiment_output = output_path or Path("artifacts/ablation_results.json")
        _write_experiment_results(
            experiment_output,
            gate_passed=gate_passed,
            rejection_reason=rejection_reason,
            threshold=threshold,
            validation_f1=validation_f1,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            window_diagnostics=window_diagnostics,
            minimum_auc=minimum_auc,
            minimum_precision=minimum_precision,
            minimum_coverage=minimum_coverage,
            positive_weight=positive_weight,
            training_positive_rate=positive / (positive + negative),
            statistics=statistics,
        )
        if not gate_passed:
            print(f"WARNING: {rejection_reason}")
        else:
            print("Experiment model passed the production evaluation gate.")

        weights_path = experiment_output.with_name("experiment_model.pt")
        scaler_path = experiment_output.with_name("experiment_scaler.pkl")
        metadata_path = experiment_output.with_name("experiment_model_metadata.json")
    else:
        if not gate_passed:
            raise RuntimeError(rejection_reason or "Model rejected by production gate")
        weights_path = Path(cfg["path"])
        scaler_path = Path(cfg["scaler_path"])
        metadata_path = Path(cfg["metadata_path"])

    weights_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), weights_path)
    engineer.save_scaler(str(scaler_path))

    metadata = {
        "model_version": dt.datetime.now(dt.timezone.utc).strftime(
            "gru-%Y%m%dT%H%M%SZ"
        ),
        "feature_version": FEATURE_VERSION,
        "feature_columns": FEATURE_COLUMNS,
        "candidate_context_version": CONTEXT_VERSION,
        "candidate_context_columns": list(CONTEXT_COLUMNS),
        "sequence_length": int(cfg["sequence_length"]),
        "decision_threshold": threshold,
        "validation": {**val_metrics, "f1": _f1(val_metrics)},
        "test": {**test_metrics, "f1": _f1(test_metrics)},
        "validation_windows": window_diagnostics,
        "validation_precision_std": float(
            np.std([row["precision"] for row in window_diagnostics])
        ),
        "threshold_search": {
            "minimum": 0.20,
            "maximum": 0.85,
            "steps": 131,
            "minimum_coverage": minimum_coverage,
            "validation_precision_floor": minimum_precision,
        },
        "trained_symbols": [item["symbol"] for item in statistics],
        "total_labeled_candidates": int(
            sum(item["labeled_candidates"] for item in statistics)
        ),
        "symbol_statistics": statistics,
        "training_positive_weight": positive_weight,
        "training_positive_rate": positive / (positive + negative),
        "training_loss": "bce" if positive_weight == 1.0 else "weighted_bce",
    }
    metadata["model_sha256"] = sha256_file(weights_path)
    metadata["scaler_sha256"] = sha256_file(scaler_path)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Validated directional model written: {weights_path}")


if __name__ == "__main__":
    args = _parse_args()
    train(
        load_all_configs(require_notifications=False),
        experiment_mode=args.experiment_mode or args.dry_run,
        output_path=args.output,
    )
