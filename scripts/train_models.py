"""Train only on real exchange data and promote only validated artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.liquidity import LiquidityEngine
from core.structure import MarketStructureDetector
from core.training_events import SMCTrainingEventBuilder
from core.zones import ZoneEngine
from data.exchange_api import ExchangeInterface
from models.feature_engineering import FEATURE_COLUMNS, FEATURE_VERSION, FeatureEngineer
from models.gru import SignalValidatorGRU
from utils.config_loader import load_all_configs


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_frame(frame: pd.DataFrame, structure: MarketStructureDetector, zones: ZoneEngine, liquidity: LiquidityEngine) -> pd.DataFrame:
    frame = structure.analyze(frame)
    frame = zones.detect_fvg(frame)
    frame = zones.find_order_blocks(frame)
    frame = liquidity.identify_liquidity_pools(frame)
    return liquidity.detect_sweeps(frame)


def build_dataset(config: dict) -> list[pd.DataFrame]:
    exchange = ExchangeInterface(config)
    structure = MarketStructureDetector(config)
    liquidity = LiquidityEngine(config)
    zones = ZoneEngine(config)
    frames: list[pd.DataFrame] = []
    for symbol in config["trading"]["symbols"]:
        raw = exchange.fetch_ohlcv_history(symbol, config["market_data"]["timeframes"]["m15"], candles=int(config["market_data"].get("training_history_candles", 5000)))
        if len(raw) < int(config["market_data"]["minimum_rows"]["m15"]):
            raise RuntimeError(f"Insufficient M15 history for {symbol}: {len(raw)}")
        processed = prepare_frame(raw, structure, zones, liquidity)
        processed["symbol"] = symbol
        frames.append(processed)
    if not frames:
        raise RuntimeError("No real training data was retrieved")
    return frames


def build_labeled_dataset(frame: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    sequence_length = int(config["model"]["sequence_length"])
    fe = FeatureEngineer(sequence_length)
    features = fe.build_features(frame)
    builder = SMCTrainingEventBuilder(config)
    candidates = builder.build(frame)
    labels = builder.label_candidates(
        frame,
        candidates,
        fe._atr(frame),
        float(config["risk_management"]["atr_sl_multiplier"]),
        float(config["risk_management"]["default_tp_rr"]),
        int(config["model"].get("label_horizon", 20)),
    )
    labels = labels[labels["candidate_index"] >= sequence_length - 1].reset_index(drop=True)
    minimum = int(config["training_events"].get("minimum_labeled_candidates_per_symbol", 300))
    if len(labels) < minimum:
        raise RuntimeError(f"Not enough labeled SMC candidates: {len(labels)}; required {minimum}")
    return features, labels


def make_sequences(features: np.ndarray, labels: pd.DataFrame, indices: np.ndarray, scaler: FeatureEngineer):
    scaled = scaler.scaler.transform(features)
    X = np.stack([scaled[i - scaler.sequence_length + 1 : i + 1] for i in indices])
    label_map = labels.set_index("candidate_index")["label"]
    y = label_map.loc[indices].to_numpy()
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y[:, None], dtype=torch.float32)


def evaluate(model, X, y):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
    model.eval()
    with torch.no_grad():
        probs = model(X).cpu().numpy().ravel()
    truth = y.cpu().numpy().ravel()
    pred = (probs >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(truth, pred)),
        "precision": float(precision_score(truth, pred, zero_division=0)),
        "recall": float(recall_score(truth, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(truth, probs)) if len(np.unique(truth)) > 1 else 0.5,
        "positive_rate": float(truth.mean()),
    }
    return metrics, probs, truth


def train(config: dict) -> None:
    frames = build_dataset(config)
    sequence_length = int(config["model"]["sequence_length"])
    datasets = []
    train_feature_frames = []
    total_labeled = 0

    for frame in frames:
        features, labels = build_labeled_dataset(frame, config)
        split1 = int(len(labels) * 0.70)
        split2 = int(len(labels) * 0.85)
        if split1 < 100 or split2 - split1 < 40 or len(labels) - split2 < 40:
            raise RuntimeError(f"Chronological split is too small for {len(labels)} labeled candidates")
        datasets.append((features, labels, split1, split2))
        train_end = int(labels.iloc[split1 - 1]["candidate_index"]) + 1
        train_feature_frames.append(features.iloc[:train_end])
        total_labeled += len(labels)

    fit_engineer = FeatureEngineer(sequence_length)
    fit_engineer.scaler.fit(np.concatenate([f.to_numpy() for f in train_feature_frames], axis=0))
    train_x, train_y, val_x, val_y, test_x, test_y = [], [], [], [], [], []

    for features, labels, split1, split2 in datasets:
        for destination_x, destination_y, selected in (
            (train_x, train_y, labels.iloc[:split1]),
            (val_x, val_y, labels.iloc[split1:split2]),
            (test_x, test_y, labels.iloc[split2:]),
        ):
            x, y = make_sequences(features.to_numpy(), selected, selected["candidate_index"].to_numpy(), fit_engineer)
            destination_x.append(x)
            destination_y.append(y)

    X_train, y_train = torch.cat(train_x), torch.cat(train_y)
    X_val, y_val = torch.cat(val_x), torch.cat(val_y)
    X_test, y_test = torch.cat(test_x), torch.cat(test_y)

    model = SignalValidatorGRU(input_dim=len(FEATURE_COLUMNS))
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.BCELoss()
    best_state, best_val_auc = None, -1.0

    for epoch in range(20):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_train), y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        metrics, _, _ = evaluate(model, X_val, y_val)
        print(f"epoch={epoch + 1} loss={loss.item():.5f} val_auc={metrics['roc_auc']:.4f} val_positive_rate={metrics['positive_rate']:.4f}")
        if metrics["roc_auc"] > best_val_auc:
            best_val_auc = metrics["roc_auc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training produced no valid model state")
    model.load_state_dict(best_state)
    val_metrics, val_probs, val_truth = evaluate(model, X_val, y_val)
    test_metrics, _, _ = evaluate(model, X_test, y_test)

    best_threshold, best_f1 = 0.60, -1.0
    for threshold in np.linspace(0.50, 0.80, 31):
        pred = (val_probs >= threshold).astype(int)
        tp = ((pred == 1) & (val_truth == 1)).sum()
        fp = ((pred == 1) & (val_truth == 0)).sum()
        fn = ((pred == 0) & (val_truth == 1)).sum()
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        if f1 > best_f1:
            best_f1, best_threshold = f1, float(threshold)

    minimum_auc = float(config["model"].get("minimum_test_auc", 0.55))
    minimum_precision = float(config["model"].get("minimum_test_precision", 0.45))
    if test_metrics["roc_auc"] < minimum_auc or test_metrics["precision"] < minimum_precision:
        raise RuntimeError(f"Model rejected: test ROC-AUC={test_metrics['roc_auc']:.4f} precision={test_metrics['precision']:.4f}")

    weights = Path(config["model"]["path"])
    scaler_path = Path(config["model"]["scaler_path"])
    metadata_path = Path(config["model"]["metadata_path"])
    weights.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), weights)
    fit_engineer.save_scaler(str(scaler_path))
    metadata = {
        "model_version": datetime.now(timezone.utc).strftime("gru-%Y%m%dT%H%M%SZ"),
        "feature_version": FEATURE_VERSION,
        "feature_columns": FEATURE_COLUMNS,
        "sequence_length": sequence_length,
        "decision_threshold": best_threshold,
        "validation": val_metrics,
        "test": test_metrics,
        "trained_symbols": config["trading"]["symbols"],
        "training_candles_per_symbol": config["market_data"]["training_history_candles"],
        "total_labeled_candidates": total_labeled,
        "candidate_policy": config["training_events"],
        "label": "TP-before-SL on causal SMC event and continuation candidates",
        "model_sha256": sha256_file(weights),
        "scaler_sha256": sha256_file(scaler_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    train(load_all_configs(require_notifications=False))
