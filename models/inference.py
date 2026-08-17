from __future__ import annotations

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
        self._load()

    def _load(self) -> None:
        for path in (self.model_path, self.scaler_path, self.metadata_path):
            if not path.exists():
                message = f"Required model artifact missing: {path}"
                if self.required:
                    raise FileNotFoundError(message)
                self.logger.warning(message)
                return

        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        if metadata.get("feature_version") != FEATURE_VERSION:
            raise RuntimeError("Model feature version does not match runtime feature version")
        if metadata.get("feature_columns") != FEATURE_COLUMNS:
            raise RuntimeError("Model feature columns do not match runtime feature columns")
        if int(metadata.get("sequence_length", -1)) != self.sequence_length:
            raise RuntimeError("Model sequence length does not match runtime configuration")

        state = torch.load(self.model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.features.load_scaler(str(self.scaler_path))
        self.decision_threshold = max(self.config_threshold, float(metadata.get("decision_threshold", self.config_threshold)))
        self.model.eval()
        self.available = True
        self.logger.info("Loaded model version %s with decision threshold %.3f", metadata.get("model_version", "unknown"), self.decision_threshold)

    def predict_confidence(self, df) -> float:
        if not self.available:
            raise RuntimeError("AI model is not available")
        scaled = self.features.transform(df)
        sequence = self.features.prepare_sequence(scaled)
        tensor = torch.as_tensor(sequence, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            prediction = float(self.model(tensor).squeeze().item())
        if not np.isfinite(prediction):
            raise RuntimeError("AI model returned a non-finite probability")
        return max(0.0, min(1.0, prediction))
