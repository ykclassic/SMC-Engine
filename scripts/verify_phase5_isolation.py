"""Verify that AI research remains isolated from the deterministic production path.

This check is intentionally static: it does not train a model, download weights,
or mutate production configuration. A passing result means the repository has a
clear production/research dependency boundary and that live signal execution
cannot require AI artifacts.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

PRODUCTION_FILES = (
    Path("main.py"),
    Path("core/engine.py"),
    Path("core/confluence.py"),
    Path("core/structure.py"),
    Path("core/zones.py"),
    Path("core/risk.py"),
    Path("data/exchange_api.py"),
    Path("alerts/discord_bot.py"),
)
FORBIDDEN_MODULES = {"torch", "joblib", "models.inference"}
MODEL_ARTIFACTS = (
    Path("models/weights/latest_gru.pth"),
    Path("models/weights/feature_scaler.joblib"),
    Path("models/weights/model_metadata.json"),
)
LIVE_WORKFLOW = Path(".github/workflows/live_signal_monitor.yml")
RESEARCH_WORKFLOW = Path(".github/workflows/smc_feature_ablation.yml")


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def _load_settings() -> dict:
    text = Path("config/settings.yaml").read_text(encoding="utf-8")
    # The repository's settings file is JSON-compatible YAML. Keeping this
    # parser dependency-free makes the verification usable in live CI.
    return json.loads(text)


def verify() -> dict:
    missing = [str(path) for path in PRODUCTION_FILES if not path.is_file()]
    if missing:
        raise SystemExit(f"Missing production files: {missing}")

    settings = _load_settings()
    model = settings.get("model", {})
    if model.get("enabled") is not False:
        raise SystemExit("Phase 5 isolation failure: production AI must be disabled")
    if model.get("required") is not False:
        raise SystemExit("Phase 5 isolation failure: production model must not be required")

    violations = []
    for path in PRODUCTION_FILES:
        for module in _imports(path):
            if module in FORBIDDEN_MODULES:
                violations.append(f"{path}: {module}")
    if violations:
        raise SystemExit("AI runtime imports detected:\n" + "\n".join(violations))

    present_artifacts = [str(path) for path in MODEL_ARTIFACTS if path.exists()]
    if present_artifacts:
        raise SystemExit(
            "AI model artifacts must not be present in the production checkout: "
            + ", ".join(present_artifacts)
        )

    live_text = LIVE_WORKFLOW.read_text(encoding="utf-8")
    research_text = RESEARCH_WORKFLOW.read_text(encoding="utf-8")
    if "smc_feature_ablation" in live_text or "ablation_feature_groups" in live_text:
        raise SystemExit("Live workflow references the AI research workflow")
    if "models/weights/latest_gru.pth" in live_text:
        raise SystemExit("Live workflow references a model artifact")
    if "SMC_AI_ENABLED: \"false\"" not in live_text:
        raise SystemExit("Live workflow does not explicitly disable AI")
    if "scripts.ablation_feature_groups" not in research_text:
        raise SystemExit("Research workflow does not invoke the isolated ablation harness")

    result = {
        "schema_version": "phase5-ai-isolation-v1",
        "production_ai_enabled": False,
        "production_model_required": False,
        "production_ai_runtime_imports": False,
        "production_model_artifacts_present": False,
        "live_workflow_references_research": False,
        "research_workflow": str(RESEARCH_WORKFLOW),
        "production_workflow": str(LIVE_WORKFLOW),
        "status": "PASS",
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    verify()
