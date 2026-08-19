from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import torch

from models.feature_engineering import FEATURE_COLUMNS, FEATURE_VERSION, FeatureEngineer
from models.gru import SignalValidatorGRU


class ModelInference:
    def __init__(self, config: dict) -> None:
        model_config = config["model"]
        self.enabled = bool(model_config.get("enabled", True))
        self.required = bool(model_config.get("required", True))
        self.model_path = Path(model_config["path"])
        self.scaler_path = Path(model_config["scaler_path"])
        self.metadata_path = Path(model_config["metadata_path"])
        self.sequence_length = int(model_config.get("sequence_length", 32))
        self.config_threshold = float(model_config.get("min_confidence", 0.60))
        self.decision_threshold = self.config_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger("SMC-Inference")
        self.features = FeatureEngineer(self.sequence_length)
        self.available = False
        self.model = SignalValidatorGRU(input_dim=len(FEATURE_COLUMNS)).to(self.device)

        if self.enabled:
            self._load()
        else:
            self.logger.info("AI inference disabled; deterministic SMC mode is active")

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _load(self) -> None:
        for path in (self.model_path, self.scaler_path, self.metadata_path):
            if not path.exists():
                message = f"Required model artifact missing: {path}"
                if self.required:
                    raise FileNotFoundError(message)
                self.logger.warning(message)
                return

        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        if metadata.get("feature_version") != FEATURE_VERSION or metadata.get("feature_columns") != FEATURE_COLUMNS:
            raise RuntimeError("Model feature schema does not match runtime")
        if int(metadata.get("sequence_length", -1)) != self.sequence_length:
            raise RuntimeError("Model sequence length does not match runtime")
        if metadata.get("model_sha256") != self._sha256(self.model_path):
            raise RuntimeError("Model artifact hash mismatch")
        if metadata.get("scaler_sha256") != self._sha256(self.scaler_path):
            raise RuntimeError("Feature scaler hash mismatch")

        state = torch.load(self.model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.features.load_scaler(str(self.scaler_path))
        self.decision_threshold = max(self.config_threshold, float(metadata.get("decision_threshold", self.config_threshold)))
        self.model.eval()
        self.available = True
        self.logger.info("Loaded model %s with threshold %.3f", metadata.get("model_version", "unknown"), self.decision_threshold)

    def predict_confidence(self, df) -> float:
        if not self.enabled or not self.available:
            raise RuntimeError("AI model is not available")
        scaled = self.features.transform(df)
        sequence = self.features.prepare_sequence(scaled)
        tensor = torch.as_tensor(sequence, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            prediction = float(self.model(tensor).squeeze().item())
        if not np.isfinite(prediction):
            raise RuntimeError("AI model returned a non-finite probability")
        return max(0.0, min(1.0, prediction))
