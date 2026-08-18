"""Diagnostic-only ablation of candidate context and label integrity.

This module deliberately does not write or promote production model artifacts.
It builds one canonical labeled population, verifies its directional identity,
and trains four isolated classifiers with identical data, splits, optimizer,
seed, and training schedule.

The diagnostic report is persisted even when an experiment fails so CI can
upload the evidence needed to diagnose the failure.
"""

from __future__ import annotations

import hashlib
import json
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.candidate_context import (
    CONTEXT_COLUMNS,
    context_vector,
    validate_candidate_context,
)
from models.feature_engineering import FEATURE_COLUMNS, FeatureEngineer
from scripts.train_models import build_dataset, build_labeled_dataset
from utils.config_loader import load_all_configs

SEED = 42
EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
REPORT_PATH = Path("models/weights/label_context_ablation.json")

EVENT_COLUMNS = tuple(CONTEXT_COLUMNS[1:])
MODES = (
    "baseline",
    "direction_only",
    "event_only",
    "direction_plus_event",
)


class DiagnosticGRU(nn.Module):
    """Small diagnostic-only GRU; never used for production artifacts."""

    def __init__(self, input_dim: int, context_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.context_dim = context_dim
        self.gru = nn.GRU(
            input_size=input_dim + context_dim,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
        )
        self.fc = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected [batch, sequence, {self.input_dim}], got {tuple(x.shape)}"
            )
        if self.context_dim:
            if context is None or context.shape != (x.shape[0], self.context_dim):
                raise ValueError(
                    f"Expected context shape [{x.shape[0]}, {self.context_dim}]"
                )
            repeated = context[:, None, :].expand(-1, x.shape[1], -1)
            x = torch.cat([x, repeated], dim=-1)
        output, _ = self.gru(x)
        return self.sigmoid(self.fc(output[:, -1, :]))


def _seed() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def _f1(precision: float, recall: float) -> float:
    return (
        0.0
        if precision + recall == 0
        else 2 * precision * recall / (precision + recall)
    )


def _metrics(
    probs: np.ndarray,
    truth: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    from sklearn.metrics import precision_score, recall_score, roc_auc_score

    pred = (probs >= threshold).astype(int)
    precision = float(precision_score(truth, pred, zero_division=0))
    recall = float(recall_score(truth, pred, zero_division=0))
    return {
        "roc_auc": (
            float(roc_auc_score(truth, probs))
            if len(np.unique(truth)) > 1
            else 0.5
        ),
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
        "predicted_positive_rate": float(pred.mean()),
        "positive_rate": float(truth.mean()),
        "samples": int(len(truth)),
    }


def _context(row, mode: str) -> np.ndarray:
    full = np.asarray(
        context_vector(row.direction, row.event_type),
        dtype=np.float32,
    )
    if mode == "baseline":
        return np.empty(0, dtype=np.float32)
    if mode == "direction_only":
        return full[:1]
    if mode == "event_only":
        return full[1:]
    if mode == "direction_plus_event":
        return full
    raise ValueError(f"Unknown ablation mode: {mode}")


def _make_symbol_partitions(features, labels, engineer, mode: str):
    scaled = engineer.scaler.transform(features)
    if not np.isfinite(scaled).all():
        raise RuntimeError("Feature scaler produced non-finite values")

    split1 = int(len(labels) * 0.70)
    split2 = int(len(labels) * 0.85)
    partitions = {"train": [], "validation": [], "test": []}

    for name, selected in (
        ("train", labels.iloc[:split1]),
        ("validation", labels.iloc[split1:split2]),
        ("test", labels.iloc[split2:]),
    ):
        sequences = []
        contexts = []
        targets = []
        for row in selected.itertuples(index=False):
            index = int(row.candidate_index)
            start = index - engineer.sequence_length + 1
            if start < 0:
                raise RuntimeError(f"Candidate {index} lacks a causal sequence")
            sequence = scaled[start : index + 1]
            if len(sequence) != engineer.sequence_length:
                raise RuntimeError(f"Invalid sequence length for candidate {index}")
            sequences.append(sequence)
            contexts.append(_context(row, mode))
            targets.append(float(row.label))
        partitions[name] = [
            torch.tensor(np.stack(sequences), dtype=torch.float32),
            (
                torch.tensor(np.stack(contexts), dtype=torch.float32)
                if contexts and contexts[0].size
                else None
            ),
            torch.tensor(np.asarray(targets)[:, None], dtype=torch.float32),
        ]
    return partitions


def _fit_mode(partitions: list[dict], mode: str) -> dict:
    _seed()
    context_dim = {
        "baseline": 0,
        "direction_only": 1,
        "event_only": len(EVENT_COLUMNS),
        "direction_plus_event": len(CONTEXT_COLUMNS),
    }[mode]
    model = DiagnosticGRU(len(FEATURE_COLUMNS), context_dim)
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    criterion = nn.BCELoss(reduction="none")

    X_train = torch.cat([p["train"][0] for p in partitions])
    C_train = (
        None
        if context_dim == 0
        else torch.cat([p["train"][1] for p in partitions])
    )
    y_train = torch.cat([p["train"][2] for p in partitions])
    X_val = torch.cat([p["validation"][0] for p in partitions])
    C_val = (
        None
        if context_dim == 0
        else torch.cat([p["validation"][1] for p in partitions])
    )
    y_val = torch.cat([p["validation"][2] for p in partitions])

    positive = float(y_train.sum())
    negative = float(y_train.numel() - positive)
    if positive <= 0 or negative <= 0:
        raise RuntimeError(f"{mode}: training partition lacks both classes")
    positive_weight = negative / positive

    best_auc = -1.0
    best_epoch = 0
    best_state = None
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        probs = model(X_train, C_train)
        weights = torch.where(
            y_train > 0.5,
            torch.full_like(y_train, positive_weight),
            torch.ones_like(y_train),
        )
        loss = (criterion(probs, y_train) * weights).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_probs = model(X_val, C_val).cpu().numpy().ravel()
        val_truth = y_val.cpu().numpy().ravel().astype(int)
        auc = _metrics(val_probs, val_truth)["roc_auc"]
        if auc > best_auc:
            best_auc = auc
            best_epoch = epoch + 1
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError(f"{mode}: no valid checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_probs = model(X_val, C_val).cpu().numpy().ravel()
        test_probs = np.concatenate(
            [
                model(
                    partition["test"][0],
                    partition["test"][1] if context_dim else None,
                )
                .cpu()
                .numpy()
                .ravel()
                for partition in partitions
            ]
        )
    test_truth = np.concatenate(
        [
            partition["test"][2].cpu().numpy().ravel().astype(int)
            for partition in partitions
        ]
    )
    val_truth = y_val.cpu().numpy().ravel().astype(int)
    return {
        "mode": mode,
        "best_epoch": best_epoch,
        "best_validation_auc": best_auc,
        "validation": _metrics(val_probs, val_truth),
        "test": _metrics(test_probs, test_truth),
        "positive_weight": positive_weight,
    }


def _integrity_report(labeled_frames: list[tuple[str, object]]) -> dict:
    report = {"symbols": {}, "population_identical_across_modes": True}
    fingerprints = []
    for symbol, labels in labeled_frames:
        # The shared validator correctly rejects duplicate directional keys,
        # but this diagnostic API has a stronger, historical contract: duplicate
        # candidate identity is an integrity failure and must surface as
        # RuntimeError. Detect it before calling the generic validator so the
        # caller receives the documented exception type without weakening the
        # validator's ValueError contract for malformed context.
        duplicate_mask = labels.duplicated(
            subset=["candidate_index", "direction"],
            keep=False,
        )
        if duplicate_mask.any():
            duplicate_rows = labels.loc[
                duplicate_mask,
                ["candidate_index", "direction"],
            ].drop_duplicates()
            raise RuntimeError(
                f"{symbol}: duplicate (candidate_index, direction) keys: "
                f"{duplicate_rows.to_dict(orient='records')}"
            )

        validate_candidate_context(labels)
        keys = labels[["candidate_index", "direction"]].astype(
            {"candidate_index": int}
        ).copy()
        duplicate_keys = int(
            keys.duplicated(["candidate_index", "direction"]).sum()
        )
        candle_collisions = int(
            labels["candidate_index"].duplicated(keep=False).sum()
        )
        directional_counts = {
            d: int((labels["direction"] == d).sum()) for d in ("LONG", "SHORT")
        }
        directional_positive_rates = {
            d: (
                float(labels.loc[labels["direction"] == d, "label"].mean())
                if (labels["direction"] == d).any()
                else 0.0
            )
            for d in ("LONG", "SHORT")
        }
        event_stats = {}
        for event_type, group in labels.groupby("event_type", sort=True):
            event_stats[str(event_type)] = {
                "candidates": int(len(group)),
                "positive": int(group["label"].sum()),
                "positive_rate": float(group["label"].mean()),
            }
        canonical = labels[
            ["candidate_index", "direction", "event_type", "label"]
        ].sort_values(
            ["candidate_index", "direction", "event_type"], kind="stable"
        ).to_csv(index=False)
        fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        fingerprints.append(fingerprint)
        report["symbols"][symbol] = {
            "labeled_candidates": int(len(labels)),
            "unique_candle_indices": int(labels["candidate_index"].nunique()),
            "directional_keys": int(len(keys.drop_duplicates())),
            "duplicate_directional_keys": duplicate_keys,
            "same_candle_opposite_direction_rows": candle_collisions,
            "direction_counts": directional_counts,
            "direction_positive_rates": directional_positive_rates,
            "event_statistics": event_stats,
            "label_fingerprint": fingerprint,
            "positive_rate": float(labels["label"].mean()),
        }
        if duplicate_keys:
            raise RuntimeError(f"{symbol}: duplicate (candidate_index, direction) keys")
    report["population_fingerprints"] = fingerprints
    return report


def run(config: dict) -> dict:
    frames = build_dataset(config)
    labeled_frames = []
    engineer = FeatureEngineer(int(config["model"]["sequence_length"]))
    train_feature_frames = []
    raw_partitions = []

    for frame in frames:
        features, labels = build_labeled_dataset(frame, config)
        validate_candidate_context(labels)
        symbol = str(frame["symbol"].iloc[0])
        labeled_frames.append((symbol, labels.copy()))
        split1 = int(len(labels) * 0.70)
        train_end = int(labels.iloc[split1 - 1]["candidate_index"]) + 1
        train_feature_frames.append(features.iloc[:train_end])
        raw_partitions.append((symbol, features, labels))

    engineer.scaler.fit(
        np.concatenate([x.to_numpy() for x in train_feature_frames], axis=0)
    )
    integrity = _integrity_report(labeled_frames)

    results = []
    for mode in MODES:
        partitions = [
            _make_symbol_partitions(features.to_numpy(), labels, engineer, mode)
            for _, features, labels in raw_partitions
        ]
        results.append(_fit_mode(partitions, mode))

    return {
        "diagnostic_version": "label-context-ablation-v2",
        "status": "success",
        "seed": SEED,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "feature_columns": list(FEATURE_COLUMNS),
        "context_columns": list(CONTEXT_COLUMNS),
        "event_columns": list(EVENT_COLUMNS),
        "modes": list(MODES),
        "integrity": integrity,
        "results": results,
    }


def _write_report(report: dict) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def main() -> None:
    config = load_all_configs(require_notifications=False)
    try:
        report = run(config)
    except Exception as exc:
        report = {
            "diagnostic_version": "label-context-ablation-v2",
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        _write_report(report)
        print(f"Diagnostic report written after failure: {REPORT_PATH}")
        raise

    _write_report(report)
    print("=== Label Integrity ===")
    for symbol, data in report["integrity"]["symbols"].items():
        print(
            f"{symbol}: candidates={data['labeled_candidates']} "
            f"unique_directional_keys={data['directional_keys']} "
            f"duplicate_directional_keys={data['duplicate_directional_keys']} "
            f"opposite_direction_rows={data['same_candle_opposite_direction_rows']} "
            f"positive_rate={data['positive_rate']:.4f}"
        )
        print(f"  direction_counts={data['direction_counts']}")
        print(f"  direction_positive_rates={data['direction_positive_rates']}")
    print("=== Context Ablation ===")
    for result in report["results"]:
        print(
            f"{result['mode']}: best_epoch={result['best_epoch']} "
            f"best_val_auc={result['best_validation_auc']:.4f} "
            f"test_auc={result['test']['roc_auc']:.4f} "
            f"test_precision={result['test']['precision']:.4f} "
            f"test_recall={result['test']['recall']:.4f}"
        )
    print(f"Diagnostic report written: {REPORT_PATH}")


if __name__ == "__main__":
    main()
