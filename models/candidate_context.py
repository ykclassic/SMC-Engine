"""Candidate-level context encoding for direction-aware SMC training."""

from __future__ import annotations

import numpy as np
import pandas as pd


EVENT_TYPES = (
    "BEARISH_BOS",
    "BEARISH_CHOCH",
    "BEARISH_FVG",
    "BEARISH_OB",
    "BEARISH_SWEEP",
    "BULLISH_BOS",
    "BULLISH_CHOCH",
    "BULLISH_FVG",
    "BULLISH_OB",
    "BULLISH_SWEEP",
)

CONTEXT_COLUMNS = ("candidate_direction", *EVENT_TYPES)
CONTEXT_VERSION = "candidate-context-v1"


def _normalise_event_type(value: object) -> str:
    return str(value).upper().strip()


def validate_candidate_context(labels: pd.DataFrame) -> None:
    """Reject ambiguous or malformed candidate context before sequence creation."""
    required = {"candidate_index", "direction", "event_type", "label"}
    missing = required.difference(labels.columns)
    if missing:
        raise ValueError(
            "Candidate context is missing required columns: "
            + ", ".join(sorted(missing))
        )

    directions = (
        labels["direction"].astype(str).str.upper().str.strip()
    )
    invalid = set(directions) - {"LONG", "SHORT"}
    if invalid:
        raise ValueError(f"Invalid candidate directions: {sorted(invalid)}")

    events = labels["event_type"].map(_normalise_event_type)
    invalid_events = set(events) - set(EVENT_TYPES)
    if invalid_events:
        raise ValueError(
            "Unsupported candidate event types: "
            + ", ".join(sorted(invalid_events))
        )

    duplicate_keys = labels.duplicated(
        subset=["candidate_index", "direction"],
        keep=False,
    )
    if duplicate_keys.any():
        raise ValueError(
            "Duplicate candidate direction keys remain after label generation"
        )


def context_vector(direction: str, event_type: str) -> np.ndarray:
    """Return [direction, one-hot event type] candidate context."""
    direction = str(direction).upper().strip()
    event_type = _normalise_event_type(event_type)
    if direction not in {"LONG", "SHORT"}:
        raise ValueError(f"Unsupported direction: {direction!r}")
    if event_type not in EVENT_TYPES:
        raise ValueError(f"Unsupported event type: {event_type!r}")

    vector = np.zeros(len(CONTEXT_COLUMNS), dtype=np.float32)
    vector[0] = 1.0 if direction == "LONG" else -1.0
    vector[1 + EVENT_TYPES.index(event_type)] = 1.0
    return vector


def augment_sequence(sequence: np.ndarray, direction: str, event_type: str) -> np.ndarray:
    """Append candidate context to every timestep of a causal sequence."""
    sequence = np.asarray(sequence, dtype=np.float32)
    if sequence.ndim != 2:
        raise ValueError("Sequence must have shape [timesteps, features]")
    context = context_vector(direction, event_type)
    repeated = np.repeat(context[None, :], sequence.shape[0], axis=0)
    return np.concatenate([sequence, repeated], axis=1)
