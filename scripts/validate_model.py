from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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
    config = load_all_configs(require_notifications=False)
    model_cfg = config["model"]
    model_path = Path(model_cfg["path"]); scaler_path = Path(model_cfg["scaler_path"]); metadata_path = Path(model_cfg["metadata_path"])
    for path in (model_path, scaler_path, metadata_path):
        if not path.exists():
            print(f"Validation failed: missing artifact {path}")
            return 1
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("feature_version") != FEATURE_VERSION or metadata.get("feature_columns") != FEATURE_COLUMNS:
        print("Validation failed: feature schema mismatch")
        return 1
    if int(metadata.get("sequence_length", -1)) != int(model_cfg["sequence_length"]):
        print("Validation failed: sequence length mismatch")
        return 1
    if metadata.get("model_sha256") != sha256_file(model_path) or metadata.get("scaler_sha256") != sha256_file(scaler_path):
        print("Validation failed: artifact hash mismatch")
        return 1
    model = SignalValidatorGRU(input_dim=len(FEATURE_COLUMNS))
    try:
        model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    except Exception as exc:
        print(f"Validation failed: model weights are incompatible: {exc}")
        return 1
    test = metadata.get("test", {})
    minimum_auc = float(model_cfg.get("minimum_test_auc", 0.55)); minimum_precision = float(model_cfg.get("minimum_test_precision", 0.45))
    if float(test.get("roc_auc", 0.0)) < minimum_auc or float(test.get("precision", 0.0)) < minimum_precision:
        print(f"Validation failed: test ROC-AUC={test.get('roc_auc')} precision={test.get('precision')}")
        return 1
    print(f"Validation passed: test ROC-AUC={test.get('roc_auc'):.4f}, precision={test.get('precision'):.4f}, threshold={metadata.get('decision_threshold')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(validate())
