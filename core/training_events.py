from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TrainingCandidate:
    index: int
    direction: int
    event_type: str
    distance_from_event: int


EVENT_DIRECTIONS = {
    "BULLISH_BOS": 1,
    "BULLISH_CHOCH": 1,
    "BULLISH_SWEEP": 1,
    "BULLISH_FVG": 1,
    "BULLISH_OB": 1,
    "BEARISH_BOS": -1,
    "BEARISH_CHOCH": -1,
    "BEARISH_SWEEP": -1,
    "BEARISH_FVG": -1,
    "BEARISH_OB": -1,
}


class SMCTrainingEventBuilder:
    """Build causal, non-overlapping-enough SMC candidates for model training."""

    def __init__(self, config: dict) -> None:
        training = config.get("training_events", {})
        self.window = max(0, int(training.get("continuation_window", 8)))
        self.min_spacing = max(1, int(training.get("minimum_candidate_spacing", 3)))
        self.max_candidates_per_event = max(1, int(training.get("max_candidates_per_event", 3)))

    @staticmethod
    def _event_at(row: pd.Series) -> tuple[str | None, int]:
        # Priority favours the most information-dense confirmation event.
        for column in ("choch", "bos", "liquidity_sweep", "fvg", "ob_event"):
            value = row.get(column)
            if value in EVENT_DIRECTIONS:
                return str(value), EVENT_DIRECTIONS[str(value)]
        return None, 0

    def build(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame(columns=["candidate_index", "direction", "event_type", "distance_from_event"])

        candidates: list[TrainingCandidate] = []
        last_selected_by_direction = {1: -10**9, -1: -10**9}
        event_indices: list[tuple[int, str, int]] = []
        for i in range(len(frame)):
            event_type, direction = self._event_at(frame.iloc[i])
            if event_type:
                event_indices.append((i, event_type, direction))

        for event_index, event_type, direction in event_indices:
            emitted = 0
            for distance in range(self.window + 1):
                candidate_index = event_index + distance
                if candidate_index >= len(frame):
                    break
                if candidate_index < last_selected_by_direction[direction] + self.min_spacing:
                    continue
                row = frame.iloc[candidate_index]
                close = float(row["close"])
                open_price = float(row["open"])
                # Continuation candidates must close in the originating
                # direction. The event candle itself is always eligible.
                if distance > 0:
                    if direction > 0 and close < open_price:
                        continue
                    if direction < 0 and close > open_price:
                        continue
                candidates.append(TrainingCandidate(candidate_index, direction, event_type, distance))
                last_selected_by_direction[direction] = candidate_index
                emitted += 1
                if emitted >= self.max_candidates_per_event:
                    break

        if not candidates:
            return pd.DataFrame(columns=["candidate_index", "direction", "event_type", "distance_from_event"])

        result = pd.DataFrame([c.__dict__ for c in candidates])
        result = result.drop_duplicates(subset=["candidate_index", "direction"], keep="first")
        return result.sort_values("candidate_index").reset_index(drop=True)

    @staticmethod
    def label_candidates(frame: pd.DataFrame, candidates: pd.DataFrame, atr: pd.Series, sl_multiplier: float, rr: float, horizon: int) -> pd.DataFrame:
        rows = []
        for candidate in candidates.itertuples(index=False):
            i = int(candidate.candidate_index)
            if i + horizon >= len(frame):
                continue
            atr_value = float(atr.iloc[i]) if np.isfinite(atr.iloc[i]) else np.nan
            if not np.isfinite(atr_value) or atr_value <= 0:
                continue
            direction = int(candidate.direction)
            entry = float(frame.iloc[i]["close"])
            risk = atr_value * sl_multiplier
            stop = entry - direction * risk
            target = entry + direction * risk * rr
            future = frame.iloc[i + 1 : i + horizon + 1]
            hit_target = (future["high"] >= target) if direction > 0 else (future["low"] <= target)
            hit_stop = (future["low"] <= stop) if direction > 0 else (future["high"] >= stop)
            target_hits = np.flatnonzero(hit_target.to_numpy())
            stop_hits = np.flatnonzero(hit_stop.to_numpy())
            if not target_hits.size and not stop_hits.size:
                continue
            target_first = target_hits.size > 0 and (not stop_hits.size or target_hits[0] < stop_hits[0])
            stop_first = stop_hits.size > 0 and (not target_hits.size or stop_hits[0] < target_hits[0])
            rows.append({
                "candidate_index": i,
                "direction": direction,
                "event_type": candidate.event_type,
                "distance_from_event": int(candidate.distance_from_event),
                "label": 1.0 if target_first else 0.0 if stop_first else 0.0,
            })
        return pd.DataFrame(rows)
