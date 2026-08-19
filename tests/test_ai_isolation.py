from pathlib import Path


WORKFLOW_DIR = Path(__file__).parents[1] / ".github" / "workflows"
PRODUCTION_WORKFLOWS = (
    "backtest.yml",
    "deploy.yml",
    "lint.yml",
    "live_signal_monitor.yml",
    "phase4_baseline_validation.yml",
)


def test_ai_retraining_workflow_is_research_only() -> None:
    workflow = (WORKFLOW_DIR / "ai_retrain.yml").read_text(encoding="utf-8")

    assert "AI - Research Retraining" in workflow
    assert "research_only" in workflow
    assert "production_dependency" in workflow
    assert "promotion_enabled" in workflow
    assert "gh release create" not in workflow
    assert "permissions:\n      contents: write" not in workflow


def test_ai_research_workflow_never_promotes_models() -> None:
    workflow = (WORKFLOW_DIR / "ai_retrain.yml").read_text(encoding="utf-8")

    forbidden_production_actions = (
        "promote-model",
        "Create Immutable GitHub Model Release",
        "Validated Model Package",
        "model-$(cat model_version.txt)",
    )
    for marker in forbidden_production_actions:
        assert marker not in workflow


def test_production_workflows_do_not_execute_ai_training() -> None:
    forbidden = (
        "train_models.py",
        "train_models_integrity.py",
        "train_models_integrity_unweighted.py",
        "train_models_calibrated.py",
        "train_models_ablation.py",
        "ablation_feature_groups.py",
        "diagnose_candidate_label_integrity.py",
        "latest_gru.pth",
        "feature_scaler.joblib",
        "model_metadata.json",
    )

    for filename in PRODUCTION_WORKFLOWS:
        workflow = (WORKFLOW_DIR / filename).read_text(encoding="utf-8")
        for marker in forbidden:
            assert marker not in workflow, f"{filename} contains AI marker: {marker}"


def test_phase4_baseline_workflow_declares_ai_independence() -> None:
    workflow = (WORKFLOW_DIR / "phase4_baseline_validation.yml").read_text(
        encoding="utf-8"
    )

    assert "deterministic_smc" in workflow
    assert "model_artifacts_required" in workflow
    assert "ai_dependency" in workflow
