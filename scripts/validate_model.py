"""Validate a trained GRU artifact without requiring live exchange credentials."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import joblib
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.candidate_context import CONTEXT_COLUMNS, CONTEXT_VERSION
from models.feature_engineering import FEATURE_COLUMNS, FEATURE_VERSION
from models.gru import SignalValidatorGRU
from utils.config_loader import load_all_configs


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate() -> int:
    config = load_all_configs(require_secrets=False, require_notifications=False)
    model_cfg = config["model"]
    model_path = Path(model_cfg["path"])
    scaler_path = Path(model_cfg["scaler_path"])
    metadata_path = Path(model_cfg["metadata_path"])

    for path in (model_path, scaler_path, metadata_path):
        if not path.is_file() or path.stat().st_size == 0:
            print(f"Validation failed: missing or empty artifact {path}")
            return 1

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"Validation failed: invalid metadata JSON: {exc}")
        return 1

    if metadata.get("feature_version") != FEATURE_VERSION:
        print("Validation failed: feature version mismatch")
        return 1
    if metadata.get("feature_columns") != FEATURE_COLUMNS:
        print("Validation failed: feature schema mismatch")
        return 1
    if metadata.get("candidate_context_version") != CONTEXT_VERSION:
        print("Validation failed: candidate context version mismatch")
        return 1
    if metadata.get("candidate_context_columns") != list(CONTEXT_COLUMNS):
        print("Validation failed: candidate context schema mismatch")
        return 1

    try:
        sequence_length = int(metadata["sequence_length"])
    except (KeyError, TypeError, ValueError):
        print("Validation failed: metadata sequence_length is invalid")
        return 1
    if sequence_length != int(model_cfg["sequence_length"]):
        print("Validation failed: sequence length mismatch")
        return 1

    if metadata.get("model_sha256") != sha256_file(model_path):
        print("Validation failed: model artifact hash mismatch")
        return 1
    if metadata.get("scaler_sha256") != sha256_file(scaler_path):
        print("Validation failed: scaler artifact hash mismatch")
        return 1

    try:
        scaler = joblib.load(scaler_path)
    except Exception as exc:
        print(f"Validation failed: scaler could not be loaded: {exc}")
        return 1
    if not hasattr(scaler, "transform"):
        print("Validation failed: scaler does not implement transform()")
        return 1
    if getattr(scaler, "n_features_in_", len(FEATURE_COLUMNS)) != len(FEATURE_COLUMNS):
        print("Validation failed: scaler feature count does not match base feature schema")
        return 1

    model = SignalValidatorGRU(input_dim=len(FEATURE_COLUMNS), context_dim=len(CONTEXT_COLUMNS))
    try:
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
    except Exception as exc:
        print(f"Validation failed: model weights are incompatible: {exc}")
        return 1

    test = metadata.get("test", {})
    try:
        minimum_auc = float(model_cfg.get("minimum_test_auc", 0.55))
        minimum_precision = float(model_cfg.get("minimum_test_precision", 0.45))
        test_auc = float(test.get("roc_auc", 0.0))
        test_precision = float(test.get("precision", 0.0))
    except (TypeError, ValueError):
        print("Validation failed: test metrics are invalid")
        return 1

    if test_auc < minimum_auc or test_precision < minimum_precision:
        print(
            "Validation failed: "
            f"test ROC-AUC={test_auc:.4f} precision={test_precision:.4f}; "
            f"required ROC-AUC>={minimum_auc:.4f} precision>={minimum_precision:.4f}"
        )
        return 1

    print(
        "Validation passed: "
        f"test ROC-AUC={test_auc:.4f}, precision={test_precision:.4f}, "
        f"threshold={metadata.get('decision_threshold')}, "
        f"context={metadata.get('candidate_context_version')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(validate())
