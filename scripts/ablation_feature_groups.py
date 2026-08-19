"""Run controlled feature-group ablation experiments for the SMC validator.

The experiment is deliberately separate from production retraining. It uses the
canonical chronological partitions and scaler from train_models_integrity.py,
keeps candidate context unchanged, and selects a feature configuration using
validation metrics only. The untouched test partition is evaluated only for the
single configuration selected by validation.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.candidate_context import CONTEXT_COLUMNS, context_vector, validate_candidate_context
from models.feature_engineering import FEATURE_COLUMNS
from models.gru import SignalValidatorGRU
from scripts.train_models import _classification_metrics
from scripts.train_models_integrity import _build_partitions, _f1, _select_threshold
from utils.config_loader import load_all_configs

BASE_FEATURES = tuple(FEATURE_COLUMNS[:16])
FEATURE_GROUPS = {
    "geometry": (
        "displacement_atr",
        "fvg_size_atr",
        "fvg_age",
        "fvg_distance_atr",
        "fvg_fill_ratio",
        "ob_size_atr",
        "ob_age",
        "ob_distance_atr",
    ),
    "liquidity": (
        "sweep_magnitude_atr",
        "liquidity_distance_atr",
    ),
    "regime": (
        "adx_norm",
        "atr_percentile",
        "structure_bias_numeric",
    ),
}

EXPERIMENTS = {
    "baseline": BASE_FEATURES,
    "baseline_plus_geometry": BASE_FEATURES + FEATURE_GROUPS["geometry"],
    "baseline_plus_geometry_liquidity": (
        BASE_FEATURES + FEATURE_GROUPS["geometry"] + FEATURE_GROUPS["liquidity"]
    ),
    "baseline_plus_geometry_liquidity_regime": (
        BASE_FEATURES
        + FEATURE_GROUPS["geometry"]
        + FEATURE_GROUPS["liquidity"]
        + FEATURE_GROUPS["regime"]
    ),
}

# Full smc-v4 is retained as a separate audit row. It is equivalent to the
# final cumulative experiment but is named explicitly for registry/reporting.
EXPERIMENTS["full_smc_v4"] = tuple(FEATURE_COLUMNS)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _feature_indices(names: tuple[str, ...]) -> list[int]:
    positions = {name: index for index, name in enumerate(FEATURE_COLUMNS)}
    return [positions[name] for name in names]


def _mask_partition(
    tensor: torch.Tensor,
    feature_names: tuple[str, ...],
) -> torch.Tensor:
    """Zero excluded standardized features; zero equals the scaler mean."""
    keep = set(_feature_indices(feature_names))
    masked = torch.zeros_like(tensor)
    selected = sorted(keep)
    masked[..., selected] = tensor[..., selected]
    return masked


def _train_one(
    X_train: torch.Tensor,
    C_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    C_val: torch.Tensor,
    y_val: torch.Tensor,
    feature_names: tuple[str, ...],
    seed: int,
    epochs: int,
) -> tuple[dict, SignalValidatorGRU]:
    _seed_everything(seed)
    train_x = _mask_partition(X_train, feature_names)
    val_x = _mask_partition(X_val, feature_names)

    positive = float(y_train.sum())
    negative = float(y_train.numel() - positive)
    if positive <= 0 or negative <= 0:
        raise RuntimeError("Training partition must contain both classes")

    model = SignalValidatorGRU(
        input_dim=len(FEATURE_COLUMNS),
        context_dim=len(CONTEXT_COLUMNS),
    )
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.BCELoss()
    best_state = None
    best_auc = -1.0

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        predictions = model(train_x, event_context=C_train)
        loss = criterion(predictions, y_train)
        if not torch.isfinite(loss):
            raise RuntimeError("Ablation training loss became non-finite")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            probabilities = model(val_x, event_context=C_val).cpu().numpy().ravel()
        truth = y_val.cpu().numpy().ravel().astype(int)
        metrics = _classification_metrics(probabilities, truth, 0.5)
        if metrics["roc_auc"] > best_auc:
            best_auc = metrics["roc_auc"]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("Ablation produced no valid model state")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probabilities = model(val_x, event_context=C_val).cpu().numpy().ravel()

    truth = y_val.cpu().numpy().ravel().astype(int)
    return {
        "roc_auc_at_0_5": float(_classification_metrics(probabilities, truth, 0.5)["roc_auc"]),
        "probabilities": probabilities,
        "truth": truth,
    }, model


def _validation_summary(
    probabilities: np.ndarray,
    truth: np.ndarray,
    minimum_precision: float,
    minimum_coverage: float,
) -> dict:
    threshold, validation_f1 = _select_threshold(
        probabilities,
        truth,
        minimum_precision,
        minimum_coverage,
    )
    metrics = _classification_metrics(probabilities, truth, threshold)
    return {
        "threshold": float(threshold),
        "roc_auc": float(metrics["roc_auc"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(_f1(metrics)),
        "coverage": float(metrics["predicted_positive_rate"]),
        "validation_f1_selected": float(validation_f1),
    }


def run(config: dict, output_path: Path, seed: int, epochs: int) -> None:
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
        _engineer,
        _statistics,
    ) = _build_partitions(config)

    cfg = config["model"]
    minimum_precision = float(cfg.get("minimum_test_precision", 0.45))
    minimum_coverage = float(cfg.get("calibration_minimum_coverage", 0.005))

    results: list[dict] = []
    for name, features in EXPERIMENTS.items():
        print(f"\n=== ABLATION {name} ({len(features)} features) ===")
        training, _model = _train_one(
            X_train,
            C_train,
            y_train,
            X_val,
            C_val,
            y_val,
            features,
            seed,
            epochs,
        )
        summary = _validation_summary(
            training["probabilities"],
            training["truth"],
            minimum_precision,
            minimum_coverage,
        )
        row = {
            "experiment": name,
            "features": list(features),
            "feature_count": len(features),
            "validation": summary,
        }
        results.append(row)
        print(json.dumps(row, indent=2, sort_keys=True))

    # Validation-only selection. Test remains untouched until the winner is
    # determined and is never used to select or tune a feature configuration.
    winner = max(
        results,
        key=lambda row: (
            row["validation"]["precision"] >= minimum_precision,
            row["validation"]["precision"],
            row["validation"]["f1"],
            row["validation"]["roc_auc"],
        ),
    )

    # Re-train the validation winner with the same deterministic seed and
    # evaluate the frozen validation threshold exactly once on the untouched
    # test partition.
    winning_features = tuple(winner["features"])
    training, model = _train_one(
        X_train,
        C_train,
        y_train,
        X_val,
        C_val,
        y_val,
        winning_features,
        seed,
        epochs,
    )
    threshold = float(winner["validation"]["threshold"])
    test_x = _mask_partition(X_test, winning_features)
    model.eval()
    with torch.no_grad():
        test_probabilities = model(test_x, event_context=C_test).cpu().numpy().ravel()
    test_truth = y_test.cpu().numpy().ravel().astype(int)
    test_metrics = _classification_metrics(test_probabilities, test_truth, threshold)

    winner["test"] = {
        "roc_auc": float(test_metrics["roc_auc"]),
        "precision": float(test_metrics["precision"]),
        "recall": float(test_metrics["recall"]),
        "f1": float(_f1(test_metrics)),
        "coverage": float(test_metrics["predicted_positive_rate"]),
    }

    payload = {
        "schema_version": "smc-feature-ablation-v1",
        "selection_policy": "validation_only_then_single_test_evaluation",
        "seed": seed,
        "epochs": epochs,
        "feature_version": "smc-v4",
        "base_feature_count": len(BASE_FEATURES),
        "feature_groups": {name: list(values) for name, values in FEATURE_GROUPS.items()},
        "production_gates": {
            "minimum_test_precision": minimum_precision,
            "minimum_test_auc": float(cfg.get("minimum_test_auc", 0.55)),
            "minimum_coverage": minimum_coverage,
        },
        "experiments": results,
        "validation_winner": winner["experiment"],
        "validation_winner_features": winner["features"],
        "selected_test_metrics": winner["test"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print("\n=== ABLATION WINNER ===")
    print(json.dumps({
        "experiment": winner["experiment"],
        "validation": winner["validation"],
        "test": winner["test"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="artifacts/feature_ablation/ablation_results.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    run(
        load_all_configs(require_notifications=False),
        Path(args.output),
        args.seed,
        args.epochs,
    )
