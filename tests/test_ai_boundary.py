from __future__ import annotations

import ast
from pathlib import Path

from core.confluence import ConfluenceEngine
from utils.config_loader import load_all_configs


def _config() -> dict:
    return {
        "confluence": {
            "minimum_score": 0.70,
            "weights": {
                "daily_bias": 0.20,
                "h4_bias": 0.20,
                "h1_setup": 0.20,
                "m15_sweep": 0.20,
                "m15_confirmation": 0.20,
            },
        },
        "model": {
            "enabled": False,
            "required": False,
            "path": "models/weights/latest_gru.pth",
            "scaler_path": "models/weights/feature_scaler.joblib",
            "metadata_path": "models/weights/model_metadata.json",
        },
        "risk_management": {
            "atr_sl_multiplier": 1.5,
            "default_tp_rr": 2.0,
        },
    }


def test_confluence_does_not_import_ai_when_disabled(monkeypatch) -> None:
    import core.confluence as confluence_module

    real_import_module = confluence_module.importlib.import_module

    def guarded_import(name: str, package: str | None = None):
        if name == "models.inference":
            raise AssertionError("AI inference was imported while AI was disabled")
        return real_import_module(name, package)

    monkeypatch.setattr(confluence_module.importlib, "import_module", guarded_import)

    engine = ConfluenceEngine(_config())

    assert engine.ai_enabled is False
    assert engine.ai_engine is None


def test_production_confluence_has_no_eager_ai_import() -> None:
    path = Path("core/confluence.py")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    forbidden = {"torch", "joblib", "models.inference"}
    violations = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in forbidden:
                    violations.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module in forbidden:
                violations.append(node.module)

    assert violations == []


def test_live_environment_can_disable_ai(monkeypatch) -> None:
    monkeypatch.setenv("SMC_AI_ENABLED", "false")

    config = load_all_configs(require_secrets=False, require_notifications=False)

    assert config["model"]["enabled"] is False
    assert config["model"]["required"] is False
