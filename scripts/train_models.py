from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.liquidity import LiquidityEngine
from core.structure import MarketStructureDetector
from core.zones import ZoneEngine
from data.exchange_api import ExchangeInterface
from models.feature_engineering import FEATURE_COLUMNS, FEATURE_VERSION, FeatureEngineer
from models.gru import SignalValidatorGRU
from utils.config_loader import load_all_configs


def build_dataset(config: dict):
    exchange = ExchangeInterface(config)
    structure = MarketStructureDetector(config)
    liquidity = LiquidityEngine(config)
    zones = ZoneEngine(config)
    symbols = config["trading"]["symbols"]
    m15_tf = config["market_data"]["timeframes"]["m15"]
    limit = int(config["market_data"].get("training_history_candles", 5000))
    frames = []
    for symbol in symbols:
        frame = exchange.fetch_ohlcv_history(symbol, m15_tf, candles=limit)
        processed = structure.analyze(frame)
        processed = zones.detect_fvg(processed)
        processed = zones.find_order_blocks(processed)
        processed = liquidity.identify_liquidity_pools(processed)
        processed = liquidity.detect_sweeps(processed)
        processed["symbol"] = symbol
        frames.append(processed)
    if not frames:
        raise RuntimeError("No real training data was retrieved")
    return frames


def build_labels(frame, config: dict) -> np.ndarray:
    """Label explicit SMC events only; no momentum-only or synthetic targets."""
    fe = FeatureEngineer()
    atr = fe._atr(frame).to_numpy()
    rr = float(config["risk_management"]["default_tp_rr"])
    sl_mult = float(config["risk_management"]["atr_sl_multiplier"])
    horizon = int(config["model"].get("label_horizon", 20))
    labels = np.full(len(frame), np.nan)

    for i in range(len(frame) - horizon):
        if not np.isfinite(atr[i]) or atr[i] <= 0:
            continue
        row = frame.iloc[i]
        event_direction = {
            "BULLISH_SWEEP": 1,
            "BULLISH_BOS": 1,
            "BULLISH_CHOCH": 1,
            "BULLISH_FVG": 1,
            "BULLISH_OB": 1,
            "BEARISH_SWEEP": -1,
            "BEARISH_BOS": -1,
            "BEARISH_CHOCH": -1,
            "BEARISH_FVG": -1,
            "BEARISH_OB": -1,
        }
        direction = next(
            (event_direction[value] for value in (
                row.get("liquidity_sweep"), row.get("bos"), row.get("choch"), row.get("fvg"), row.get("order_block")
            ) if value in event_direction),
            0,
        )
        if direction == 0:
            continue

        entry = float(row["close"])
        risk = float(atr[i]) * sl_mult
        target = entry + direction * risk * rr
        stop = entry - direction * risk
        future = frame.iloc[i + 1 : i + horizon + 1]
        hit_target = (future["high"] >= target) if direction > 0 else (future["low"] <= target)
        hit_stop = (future["low"] <= stop) if direction > 0 else (future["high"] >= stop)
        target_idx = np.flatnonzero(hit_target.to_numpy())
        stop_idx = np.flatnonzero(hit_stop.to_numpy())
        if target_idx.size == 0 and stop_idx.size == 0:
            continue
        if target_idx.size and stop_idx.size and target_idx[0] == stop_idx[0]:
            labels[i] = 0.0
        elif target_idx.size and (not stop_idx.size or target_idx[0] < stop_idx[0]):
            labels[i] = 1.0
        else:
            labels[i] = 0.0
    return labels


def sequence_dataset(frame, config: dict):
    sequence_length = int(config["model"]["sequence_length"])
    fe = FeatureEngineer(sequence_length)
    raw_features = fe.build_features(frame)
    labels = build_labels(frame, config)
    valid = np.where(np.isfinite(labels))[0]
    valid = valid[valid >= sequence_length - 1]
    if len(valid) < 100:
        raise RuntimeError(f"Not enough labeled SMC candidates: {len(valid)}")
    return raw_features, labels, valid


def make_sequences(features: np.ndarray, labels: np.ndarray, indices: np.ndarray, scaler: FeatureEngineer):
    scaled = scaler.scaler.transform(features)
    X = np.stack([scaled[i - scaler.sequence_length + 1 : i + 1] for i in indices])
    y = labels[indices]
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y[:, None], dtype=torch.float32)


def evaluate(model, X, y):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
    model.eval()
    with torch.no_grad():
        probs = model(X).cpu().numpy().ravel()
    truth = y.cpu().numpy().ravel()
    pred = (probs >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(truth, pred)),
        "precision": float(precision_score(truth, pred, zero_division=0)),
        "recall": float(recall_score(truth, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(truth, probs)) if len(np.unique(truth)) > 1 else 0.5,
    }, probs, truth


def train(config: dict) -> None:
    frames = build_dataset(config)
    sequence_length = int(config["model"]["sequence_length"])
    datasets = []
    train_feature_frames = []
    for frame in frames:
        features, labels, indices = sequence_dataset(frame, config)
        split1 = int(len(indices) * 0.70)
        split2 = int(len(indices) * 0.85)
        if split1 < 50 or split2 - split1 < 20 or len(indices) - split2 < 20:
            raise RuntimeError(f"Chronological split is too small for {len(indices)} SMC events")
        datasets.append((features, labels, indices, split1, split2))
        train_feature_frames.append(features.iloc[: indices[split1] + 1])

    fit_engineer = FeatureEngineer(sequence_length)
    fit_engineer.scaler.fit(np.concatenate([f.to_numpy() for f in train_feature_frames], axis=0))

    train_x, train_y, val_x, val_y, test_x, test_y = [], [], [], [], [], []
    for features, labels, indices, split1, split2 in datasets:
        for destination_x, destination_y, selected in (
            (train_x, train_y, indices[:split1]),
            (val_x, val_y, indices[split1:split2]),
            (test_x, test_y, indices[split2:]),
        ):
            x, y = make_sequences(features.to_numpy(), labels, selected, fit_engineer)
            destination_x.append(x)
            destination_y.append(y)

    X_train, y_train = torch.cat(train_x), torch.cat(train_y)
    X_val, y_val = torch.cat(val_x), torch.cat(val_y)
    X_test, y_test = torch.cat(test_x), torch.cat(test_y)

    model = SignalValidatorGRU(input_dim=len(FEATURE_COLUMNS))
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.BCELoss()
    best_state = None
    best_val_auc = -1.0

    for epoch in range(20):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_train), y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        metrics, _, _ = evaluate(model, X_val, y_val)
        print(f"epoch={epoch + 1} loss={loss.item():.5f} val_auc={metrics['roc_auc']:.4f}")
        if metrics["roc_auc"] > best_val_auc:
            best_val_auc = metrics["roc_auc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training produced no valid model state")
    model.load_state_dict(best_state)
    val_metrics, val_probs, val_truth = evaluate(model, X_val, y_val)
    test_metrics, _, _ = evaluate(model, X_test, y_test)

    thresholds = np.linspace(0.50, 0.80, 31)
    best_threshold, best_f1 = 0.60, -1.0
    for threshold in thresholds:
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
        raise RuntimeError(
            f"Model rejected: test ROC-AUC={test_metrics['roc_auc']:.4f} precision={test_metrics['precision']:.4f}"
        )

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
        "label": "TP-before-SL on explicit SMC BOS/CHoCH/liquidity/FVG/order-block events",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    train(load_all_configs())
