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
from models.feature_engineering import (
    FEATURE_COLUMNS,
    FEATURE_VERSION,
    FeatureEngineer,
)
from models.gru import SignalValidatorGRU
from utils.config_loader import load_all_configs


REQUIRED_CANDIDATE_COLUMNS = {
    "candidate_index",
    "direction",
    "event_type",
}

REQUIRED_LABEL_COLUMNS = {
    "candidate_index",
    "direction",
    "event_type",
    "label",
}


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def prepare_frame(
    frame: pd.DataFrame,
    structure: MarketStructureDetector,
    zones: ZoneEngine,
    liquidity: LiquidityEngine,
) -> pd.DataFrame:
    """Build the causal SMC feature/event frame."""

    frame = structure.analyze(frame)
    frame = zones.detect_fvg(frame)
    frame = zones.find_order_blocks(frame)
    frame = liquidity.identify_liquidity_pools(frame)
    frame = liquidity.detect_sweeps(frame)

    return frame


def build_dataset(config: dict) -> list[pd.DataFrame]:
    """Fetch and prepare real exchange data for every configured symbol."""

    exchange = ExchangeInterface(config)
    structure = MarketStructureDetector(config)
    liquidity = LiquidityEngine(config)
    zones = ZoneEngine(config)

    frames: list[pd.DataFrame] = []

    for symbol in config["trading"]["symbols"]:
        timeframe = config["market_data"]["timeframes"]["m15"]
        candles = int(
            config["market_data"].get(
                "training_history_candles",
                5000,
            )
        )

        raw = exchange.fetch_ohlcv_history(
            symbol,
            timeframe,
            candles=candles,
        )

        minimum_rows = int(
            config["market_data"]["minimum_rows"]["m15"]
        )

        if len(raw) < minimum_rows:
            raise RuntimeError(
                f"Insufficient M15 history for {symbol}: "
                f"{len(raw)} < {minimum_rows}"
            )

        processed = prepare_frame(
            raw,
            structure,
            zones,
            liquidity,
        )

        processed["symbol"] = symbol
        frames.append(processed)

    if not frames:
        raise RuntimeError("No real training data was retrieved")

    return frames


def validate_candidates(
    candidates: pd.DataFrame,
    frame_length: int,
) -> pd.DataFrame:
    """
    Validate the canonical SMC training-event contract.

    Candidate indices represent the candle on which the setup becomes
    causally actionable.
    """

    if candidates is None or candidates.empty:
        raise RuntimeError(
            "SMCTrainingEventBuilder produced zero candidates"
        )

    missing = REQUIRED_CANDIDATE_COLUMNS.difference(
        candidates.columns
    )

    if missing:
        raise RuntimeError(
            "SMCTrainingEventBuilder contract violation: "
            f"missing columns {sorted(missing)}; "
            f"received {sorted(candidates.columns.tolist())}"
        )

    result = candidates.copy()

    result["candidate_index"] = pd.to_numeric(
        result["candidate_index"],
        errors="coerce",
    )

    if result["candidate_index"].isna().any():
        raise RuntimeError(
            "SMC candidates contain non-numeric candidate_index values"
        )

    result["candidate_index"] = (
        result["candidate_index"].astype(int)
    )

    if (
        (result["candidate_index"] < 0)
        | (result["candidate_index"] >= frame_length)
    ).any():
        invalid = result.loc[
            (result["candidate_index"] < 0)
            | (result["candidate_index"] >= frame_length),
            "candidate_index",
        ].tolist()

        raise RuntimeError(
            "SMC candidates contain out-of-range candidate indices: "
            f"{invalid[:10]}"
        )

    result["direction"] = (
        result["direction"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    invalid_directions = set(result["direction"]) - {
        "LONG",
        "SHORT",
    }

    if invalid_directions:
        raise RuntimeError(
            "Invalid SMC candidate directions: "
            f"{sorted(invalid_directions)}"
        )

    result["event_type"] = (
        result["event_type"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    result = result.sort_values(
        ["candidate_index", "direction"],
        kind="stable",
    ).reset_index(drop=True)

    # Multiple SMC detections can legitimately occur around the same
    # candle, but the same candle/direction must not be duplicated.
    result = result.drop_duplicates(
        subset=["candidate_index", "direction"],
        keep="first",
    ).reset_index(drop=True)

    if result.empty:
        raise RuntimeError(
            "All SMC candidates were removed during validation"
        )

    return result


def build_labeled_dataset(
    frame: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate causal SMC candidates and TP-before-SL labels."""

    sequence_length = int(
        config["model"]["sequence_length"]
    )

    feature_engineer = FeatureEngineer(sequence_length)

    features = feature_engineer.build_features(frame)

    builder = SMCTrainingEventBuilder(config)

    candidates = builder.build(frame)

    candidates = validate_candidates(
        candidates,
        len(frame),
    )

    labels = builder.label_candidates(
        frame,
        candidates,
        feature_engineer._atr(frame),
        float(
            config["risk_management"]["atr_sl_multiplier"]
        ),
        float(
            config["risk_management"]["default_tp_rr"]
        ),
        int(
            config["model"].get(
                "label_horizon",
                20,
            )
        ),
    )

    if labels is None or labels.empty:
        raise RuntimeError(
            "SMCTrainingEventBuilder produced zero labels"
        )

    missing = REQUIRED_LABEL_COLUMNS.difference(
        labels.columns
    )

    if missing:
        raise RuntimeError(
            "SMCTrainingEventBuilder label contract violation: "
            f"missing columns {sorted(missing)}; "
            f"received {sorted(labels.columns.tolist())}"
        )

    labels = labels.copy()

    labels["candidate_index"] = pd.to_numeric(
        labels["candidate_index"],
        errors="coerce",
    )

    labels["label"] = pd.to_numeric(
        labels["label"],
        errors="coerce",
    )

    labels = labels.dropna(
        subset=[
            "candidate_index",
            "label",
        ]
    )

    labels["candidate_index"] = (
        labels["candidate_index"].astype(int)
    )

    labels["label"] = labels["label"].astype(float)

    invalid_labels = ~labels["label"].isin([0.0, 1.0])

    if invalid_labels.any():
        raise RuntimeError(
            "Training labels must be binary 0/1. "
            f"Invalid values: "
            f"{sorted(labels.loc[invalid_labels, 'label'].unique())}"
        )

    labels = labels[
        labels["candidate_index"] >= sequence_length - 1
    ].copy()

    labels = labels[
        labels["candidate_index"] < len(frame)
    ].copy()

    labels = labels.sort_values(
        ["candidate_index", "direction"],
        kind="stable",
    ).reset_index(drop=True)

    labels = labels.drop_duplicates(
        subset=["candidate_index", "direction"],
        keep="first",
    ).reset_index(drop=True)

    minimum = int(
        config["training_events"].get(
            "minimum_labeled_candidates_per_symbol",
            300,
        )
    )

    if len(labels) < minimum:
        raise RuntimeError(
            "Not enough labeled SMC candidates: "
            f"{len(labels)}; required {minimum}"
        )

    return features, labels


def make_sequences(
    features: np.ndarray,
    labels: pd.DataFrame,
    indices: np.ndarray,
    scaler: FeatureEngineer,
):
    """Create scaled causal sequences for the GRU."""

    if len(indices) == 0:
        raise RuntimeError(
            "Cannot create training sequences from zero indices"
        )

    scaled = scaler.scaler.transform(features)

    if not np.isfinite(scaled).all():
        raise RuntimeError(
            "Feature scaler produced non-finite values"
        )

    label_map = (
        labels
        .drop_duplicates(
            subset=["candidate_index"],
            keep="first",
        )
        .set_index("candidate_index")["label"]
    )

    missing_indices = [
        int(index)
        for index in indices
        if int(index) not in label_map.index
    ]

    if missing_indices:
        raise RuntimeError(
            "Missing labels for candidate indices: "
            f"{missing_indices[:10]}"
        )

    sequences = []

    for index in indices:
        index = int(index)

        start = (
            index
            - scaler.sequence_length
            + 1
        )

        if start < 0:
            raise RuntimeError(
                f"Candidate index {index} does not have "
                f"a complete sequence of "
                f"{scaler.sequence_length} candles"
            )

        sequence = scaled[start : index + 1]

        if len(sequence) != scaler.sequence_length:
            raise RuntimeError(
                f"Invalid sequence length for candidate "
                f"{index}: {len(sequence)}"
            )

        sequences.append(sequence)

    X = np.stack(sequences)

    y = label_map.loc[
        [int(index) for index in indices]
    ].to_numpy()

    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(
            y[:, None],
            dtype=torch.float32,
        ),
    )


def evaluate(model, X, y):
    """Evaluate model probabilities and classification metrics."""

    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    model.eval()

    with torch.no_grad():
        probs = (
            model(X)
            .detach()
            .cpu()
            .numpy()
            .ravel()
        )

    truth = (
        y.detach()
        .cpu()
        .numpy()
        .ravel()
    )

    if not np.isfinite(probs).all():
        raise RuntimeError(
            "Model produced non-finite predictions"
        )

    pred = (
        probs >= 0.5
    ).astype(int)

    metrics = {
        "accuracy": float(
            accuracy_score(truth, pred)
        ),
        "precision": float(
            precision_score(
                truth,
                pred,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                truth,
                pred,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                truth,
                probs,
            )
        )
        if len(np.unique(truth)) > 1
        else 0.5,
        "positive_rate": float(
            truth.mean()
        ),
        "samples": int(len(truth)),
    }

    return metrics, probs, truth


def train(config: dict) -> None:
    """Train and promote a validated GRU model."""

    frames = build_dataset(config)

    sequence_length = int(
        config["model"]["sequence_length"]
    )

    datasets = []
    train_feature_frames = []
    total_labeled = 0
    symbol_statistics = []

    for frame in frames:
        features, labels = build_labeled_dataset(
            frame,
            config,
        )

        split1 = int(
            len(labels) * 0.70
        )

        split2 = int(
            len(labels) * 0.85
        )

        train_count = split1
        validation_count = split2 - split1
        test_count = len(labels) - split2

        if (
            train_count < 100
            or validation_count < 40
            or test_count < 40
        ):
            raise RuntimeError(
                "Chronological split is too small for "
                f"{len(labels)} labeled candidates: "
                f"train={train_count}, "
                f"validation={validation_count}, "
                f"test={test_count}"
            )

        datasets.append(
            (
                features,
                labels,
                split1,
                split2,
            )
        )

        train_end = (
            int(
                labels.iloc[
                    split1 - 1
                ]["candidate_index"]
            )
            + 1
        )

        train_feature_frames.append(
            features.iloc[:train_end]
        )

        total_labeled += len(labels)

        symbol = (
            frame["symbol"].iloc[0]
            if "symbol" in frame.columns
            else "UNKNOWN"
        )

        positive_rate = float(
            labels["label"].mean()
        )

        symbol_statistics.append(
            {
                "symbol": symbol,
                "labeled_candidates": int(
                    len(labels)
                ),
                "positive_labels": int(
                    labels["label"].sum()
                ),
                "negative_labels": int(
                    (labels["label"] == 0).sum()
                ),
                "positive_rate": positive_rate,
                "train_candidates": train_count,
                "validation_candidates": validation_count,
                "test_candidates": test_count,
            }
        )

        print(
            f"{symbol}: "
            f"candidates={len(labels)} "
            f"positive_rate={positive_rate:.4f}"
        )

    fit_engineer = FeatureEngineer(
        sequence_length
    )

    training_arrays = [
        dataframe.to_numpy()
        for dataframe in train_feature_frames
    ]

    if not training_arrays:
        raise RuntimeError(
            "No training feature frames available"
        )

    fit_engineer.scaler.fit(
        np.concatenate(
            training_arrays,
            axis=0,
        )
    )

    train_x = []
    train_y = []
    val_x = []
    val_y = []
    test_x = []
    test_y = []

    for (
        features,
        labels,
        split1,
        split2,
    ) in datasets:

        train_labels = labels.iloc[
            :split1
        ]

        validation_labels = labels.iloc[
            split1:split2
        ]

        test_labels = labels.iloc[
            split2:
        ]

        for (
            destination_x,
            destination_y,
            selected,
        ) in (
            (
                train_x,
                train_y,
                train_labels,
            ),
            (
                val_x,
                val_y,
                validation_labels,
            ),
            (
                test_x,
                test_y,
                test_labels,
            ),
        ):
            x, y = make_sequences(
                features.to_numpy(),
                selected,
                selected[
                    "candidate_index"
                ].to_numpy(),
                fit_engineer,
            )

            destination_x.append(x)
            destination_y.append(y)

    if not train_x or not val_x or not test_x:
        raise RuntimeError(
            "One or more training partitions are empty"
        )

    X_train = torch.cat(train_x)
    y_train = torch.cat(train_y)

    X_val = torch.cat(val_x)
    y_val = torch.cat(val_y)

    X_test = torch.cat(test_x)
    y_test = torch.cat(test_y)

    print(
        "Dataset sizes: "
        f"train={len(X_train)} "
        f"validation={len(X_val)} "
        f"test={len(X_test)}"
    )

    model = SignalValidatorGRU(
        input_dim=len(FEATURE_COLUMNS)
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-5,
    )

    criterion = nn.BCELoss()

    best_state = None
    best_val_auc = -1.0

    for epoch in range(20):
        model.train()

        optimizer.zero_grad()

        predictions = model(X_train)

        if not torch.isfinite(
            predictions
        ).all():
            raise RuntimeError(
                "Model produced non-finite training predictions"
            )

        loss = criterion(
            predictions,
            y_train,
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                "Training loss became non-finite"
            )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0,
        )

        optimizer.step()

        metrics, _, _ = evaluate(
            model,
            X_val,
            y_val,
        )

        print(
            f"epoch={epoch + 1} "
            f"loss={loss.item():.5f} "
            f"val_auc={metrics['roc_auc']:.4f} "
            f"val_positive_rate="
            f"{metrics['positive_rate']:.4f}"
        )

        if metrics["roc_auc"] > best_val_auc:
            best_val_auc = metrics["roc_auc"]

            best_state = {
                key: value.detach()
                .cpu()
                .clone()
                for key, value
                in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError(
            "Training produced no valid model state"
        )

    model.load_state_dict(
        best_state
    )

    val_metrics, val_probs, val_truth = evaluate(
        model,
        X_val,
        y_val,
    )

    test_metrics, _, _ = evaluate(
        model,
        X_test,
        y_test,
    )

    best_threshold = 0.60
    best_f1 = -1.0

    for threshold in np.linspace(
        0.50,
        0.80,
        31,
    ):
        pred = (
            val_probs >= threshold
        ).astype(int)

        tp = (
            (pred == 1)
            & (val_truth == 1)
        ).sum()

        fp = (
            (pred == 1)
            & (val_truth == 0)
        ).sum()

        fn = (
            (pred == 0)
            & (val_truth == 1)
        ).sum()

        precision = (
            tp / max(tp + fp, 1)
        )

        recall = (
            tp / max(tp + fn, 1)
        )

        f1 = (
            2
            * precision
            * recall
            / max(
                precision + recall,
                1e-12,
            )
        )

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(
                threshold
            )

    minimum_auc = float(
        config["model"].get(
            "minimum_test_auc",
            0.55,
        )
    )

    minimum_precision = float(
        config["model"].get(
            "minimum_test_precision",
            0.45,
        )
    )

    if (
        test_metrics["roc_auc"]
        < minimum_auc
        or test_metrics["precision"]
        < minimum_precision
    ):
        raise RuntimeError(
            "Model rejected: "
            f"test ROC-AUC="
            f"{test_metrics['roc_auc']:.4f} "
            f"precision="
            f"{test_metrics['precision']:.4f}"
        )

    weights = Path(
        config["model"]["path"]
    )

    scaler_path = Path(
        config["model"]["scaler_path"]
    )

    metadata_path = Path(
        config["model"]["metadata_path"]
    )

    weights.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        model.state_dict(),
        weights,
    )

    fit_engineer.save_scaler(
        str(scaler_path)
    )

    metadata = {
        "model_version": datetime.now(
            timezone.utc
        ).strftime(
            "gru-%Y%m%dT%H%M%SZ"
        ),
        "feature_version": FEATURE_VERSION,
        "feature_columns": FEATURE_COLUMNS,
        "sequence_length": sequence_length,
        "decision_threshold": best_threshold,
        "validation": val_metrics,
        "test": test_metrics,
        "trained_symbols": config[
            "trading"
        ]["symbols"],
        "training_candles_per_symbol": config[
            "market_data"
        ]["training_history_candles"],
        "total_labeled_candidates": total_labeled,
        "symbol_statistics": symbol_statistics,
        "candidate_policy": config[
            "training_events"
        ],
        "label": (
            "TP-before-SL on causal SMC "
            "event and continuation candidates"
        ),
        "model_sha256": sha256_file(
            weights
        ),
        "scaler_sha256": sha256_file(
            scaler_path
        ),
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            metadata,
            indent=2,
        )
    )


if __name__ == "__main__":
    train(
        load_all_configs(
            require_notifications=False
        )
    )
