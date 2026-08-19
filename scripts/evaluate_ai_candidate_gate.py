"""Check an AI research report against the configured production thresholds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def evaluate(report: dict) -> dict:
    if report.get("schema_version") != "smc-feature-ablation-v1":
        raise SystemExit("Unsupported candidate artifact schema")
    if report.get("selection_policy") != "validation_only_then_single_test_evaluation":
        raise SystemExit("Candidate artifact does not prove test-set isolation")

    gates = report.get("production_gates", {})
    metrics = report.get("selected_test_metrics", {})
    for key in ("minimum_test_auc", "minimum_test_precision", "minimum_coverage"):
        if key not in gates:
            raise SystemExit(f"Candidate artifact missing gate: {key}")
    for key in ("roc_auc", "precision", "coverage"):
        if key not in metrics:
            raise SystemExit(f"Candidate artifact missing test metric: {key}")

    checks = {
        "roc_auc": {
            "actual": float(metrics["roc_auc"]),
            "required": float(gates["minimum_test_auc"]),
        },
        "precision": {
            "actual": float(metrics["precision"]),
            "required": float(gates["minimum_test_precision"]),
        },
        "coverage": {
            "actual": float(metrics["coverage"]),
            "required": float(gates["minimum_coverage"]),
        },
    }
    for check in checks.values():
        check["passed"] = check["actual"] >= check["required"]

    passed = all(check["passed"] for check in checks.values())
    return {
        "schema_version": "phase6-ai-candidate-gate-v1",
        "candidate": report.get("validation_winner"),
        "selection_policy": report["selection_policy"],
        "checks": checks,
        "all_production_gates_passed": passed,
        "action": "BEGIN_PHASE6_REINTRODUCTION" if passed else "STOP_PHASE6",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()

    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    result = evaluate(report)
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return 0 if result["all_production_gates_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
