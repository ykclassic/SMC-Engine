import pytest

from scripts.evaluate_ai_candidate_gate import evaluate


def _report(auc=0.56, precision=0.46, coverage=0.10):
    return {
        "schema_version": "smc-feature-ablation-v1",
        "selection_policy": "validation_only_then_single_test_evaluation",
        "validation_winner": "baseline_plus_geometry_liquidity_regime",
        "production_gates": {
            "minimum_test_auc": 0.55,
            "minimum_test_precision": 0.45,
            "minimum_coverage": 0.005,
        },
        "selected_test_metrics": {
            "roc_auc": auc,
            "precision": precision,
            "coverage": coverage,
        },
    }


def test_candidate_passes_only_when_all_production_gates_pass():
    result = evaluate(_report())
    assert result["all_production_gates_passed"] is True
    assert result["action"] == "BEGIN_PHASE6_REINTRODUCTION"


def test_precision_failure_stops_phase6():
    result = evaluate(_report(precision=0.2784))
    assert result["checks"]["precision"]["passed"] is False
    assert result["all_production_gates_passed"] is False
    assert result["action"] == "STOP_PHASE6"


def test_auc_failure_stops_phase6():
    result = evaluate(_report(auc=0.5499))
    assert result["checks"]["roc_auc"]["passed"] is False
    assert result["action"] == "STOP_PHASE6"


def test_coverage_failure_stops_phase6():
    result = evaluate(_report(coverage=0.0049))
    assert result["checks"]["coverage"]["passed"] is False
    assert result["action"] == "STOP_PHASE6"


def test_invalid_selection_policy_is_rejected():
    report = _report()
    report["selection_policy"] = "test_set_tuned"
    with pytest.raises(SystemExit, match="test-set isolation"):
        evaluate(report)
