from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TrainingCandidate:
    """Canonical training candidate produced by the SMC event builder."""

    candidate_index: int
    direction: str
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
    """Build causal SMC candidates for supervised model training."""

    def __init__(self, config: dict) -> None:
        training = config.get("training_events", {})
        self.window = max(
            0,
            int(training.get("continuation_window", 8)),
        )
        self.min_spacing = max(
            1,
            int(training.get("minimum_candidate_spacing", 3)),
        )
        self.max_candidates_per_event = max(
            1,
            int(training.get("max_candidates_per_event", 3)),
        )

    @staticmethod
    def _event_at(row: pd.Series) -> tuple[str | None, int]:
        """Return the highest-priority confirmed event and its sign."""

        for column in (
            "choch",
            "bos",
            "liquidity_sweep",
            "fvg",
            "ob_event",
        ):
            value = row.get(column)
            if value in EVENT_DIRECTIONS:
                event_type = str(value)
                return event_type, EVENT_DIRECTIONS[event_type]

        return None, 0

    @staticmethod
    def _direction_name(direction: int) -> str:
        """Convert the internal event sign to the canonical public label."""

        if direction > 0:
            return "LONG"
        if direction < 0:
            return "SHORT"
        raise ValueError(f"Unsupported training direction: {direction}")

    def build(self, frame: pd.DataFrame) -> pd.DataFrame:
        """
        Build causal event/continuation candidates.

        Candidate spacing is applied independently inside each originating
        event window. The previous implementation used a single global
        spacing cursor per direction, which could suppress later SMC events
        whenever two events occurred within the spacing interval.
        """

        columns = [
            "candidate_index",
            "direction",
            "event_type",
            "distance_from_event",
        ]

        if frame.empty:
            return pd.DataFrame(columns=columns)

        candidates: list[TrainingCandidate] = []
        event_indices: list[tuple[int, str, int]] = []

        for index in range(len(frame)):
            event_type, direction = self._event_at(frame.iloc[index])
            if event_type is not None:
                event_indices.append((index, event_type, direction))

        for event_index, event_type, direction in event_indices:
            emitted = 0
            last_candidate_index = -10**9

            for distance in range(self.window + 1):
                candidate_index = event_index + distance

                if candidate_index >= len(frame):
                    break

                if candidate_index < last_candidate_index + self.min_spacing:
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

                candidates.append(
                    TrainingCandidate(
                        candidate_index=candidate_index,
                        direction=self._direction_name(direction),
                        event_type=event_type,
                        distance_from_event=distance,
                    )
                )

                last_candidate_index = candidate_index
                emitted += 1

                if emitted >= self.max_candidates_per_event:
                    break

        if not candidates:
            return pd.DataFrame(columns=columns)

        result = pd.DataFrame(
            [
                {
                    "candidate_index": candidate.candidate_index,
                    "direction": candidate.direction,
                    "event_type": candidate.event_type,
                    "distance_from_event": candidate.distance_from_event,
                }
                for candidate in candidates
            ],
            columns=columns,
        )

        return (
            result.drop_duplicates(
                subset=[
                    "candidate_index",
                    "direction",
                ],
                keep="first",
            )
            .sort_values(
                ["candidate_index", "direction"],
                kind="stable",
            )
            .reset_index(drop=True)
        )

    @staticmethod
    def label_candidates(
        frame: pd.DataFrame,
        candidates: pd.DataFrame,
        atr: pd.Series,
        sl_multiplier: float,
        rr: float,
        horizon: int,
    ) -> pd.DataFrame:
        """
        Label candidates by TP-before-SL within the forward horizon.

        A candidate that reaches neither TP nor SL during the horizon is
        retained as label 0. Dropping unresolved candidates creates selection
        bias and can collapse the training dataset when volatility is low.
        A same-candle TP/SL collision is also conservatively labelled 0.
        """

        rows: list[dict] = []

        for candidate in candidates.itertuples(index=False):
            index = int(candidate.candidate_index)

            if index + horizon >= len(frame):
                continue

            atr_value = float(atr.iloc[index])
            if not np.isfinite(atr_value) or atr_value <= 0:
                continue

            direction_name = str(candidate.direction).upper().strip()
            if direction_name == "LONG":
                direction = 1
            elif direction_name == "SHORT":
                direction = -1
            else:
                raise ValueError(
                    "Unsupported training candidate direction: "
                    f"{candidate.direction!r}"
                )

            entry = float(frame.iloc[index]["close"])
            risk = atr_value * float(sl_multiplier)
            stop = entry - direction * risk
            target = entry + direction * risk * float(rr)

            future = frame.iloc[index + 1 : index + horizon + 1]

            if direction > 0:
                hit_target = future["high"] >= target
                hit_stop = future["low"] <= stop
            else:
                hit_target = future["low"] <= target
                hit_stop = future["high"] >= stop

            target_hits = np.flatnonzero(hit_target.to_numpy())
            stop_hits = np.flatnonzero(hit_stop.to_numpy())

            target_first = (
                target_hits.size > 0
                and (
                    not stop_hits.size
                    or target_hits[0] < stop_hits[0]
                )
            )
            stop_first = (
                stop_hits.size > 0
                and (
                    not target_hits.size
                    or stop_hits[0] < target_hits[0]
                )
            )

            # No target before stop, including unresolved and same-candle
            # collisions, is conservatively represented as a negative label.
            label = 1.0 if target_first else 0.0

            rows.append(
                {
                    "candidate_index": index,
                    "direction": direction_name,
                    "event_type": str(candidate.event_type),
                    "distance_from_event": int(candidate.distance_from_event),
                    "label": label,
                }
            )

        return pd.DataFrame(rows)
