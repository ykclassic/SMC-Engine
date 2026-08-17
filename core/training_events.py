from __future__ import annotations

from collections import Counter
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

EVENT_COLUMNS = (
    "choch",
    "bos",
    "liquidity_sweep",
    "fvg",
    "ob_event",
)


class SMCTrainingEventBuilder:
    """Build causal SMC candidates for supervised model training.

    Every confirmed SMC event is treated as an origin. Candidate observations
    are created only on the event candle or on subsequent candles inside the
    configured continuation window, so no future candle is used to create an
    event.

    Recovery mode is opt-in through the explicit
    ``minimum_labeled_candidates_per_symbol`` training configuration. Normal
    callers and unit tests therefore retain their configured event window,
    spacing, and per-event limits.

    Production recovery is intentionally broader than normal candidate
    generation. Sparse SMC events are a data-coverage problem, not a reason to
    lower the downstream quality gate. Recovery therefore uses real candles,
    keeps the direction/event provenance, removes duplicate candle/direction
    observations, and expands the causal continuation window and per-event
    candidate budget until the downstream sequence/label losses are covered.
    """

    def __init__(self, config: dict) -> None:
        training = config.get("training_events", {})
        model = config.get("model", {})

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

        minimum_labeled_value = training.get(
            "minimum_labeled_candidates_per_symbol"
        )
        self.recovery_enabled = minimum_labeled_value is not None
        self.minimum_labeled_candidates = max(
            1,
            int(
                minimum_labeled_value
                if minimum_labeled_value is not None
                else 1
            ),
        )

        self.label_horizon = max(
            0,
            int(model.get("label_horizon", 20)),
        )
        self.sequence_length = max(
            1,
            int(model.get("sequence_length", 32)),
        )

        # These are deliberately configurable so production data coverage can
        # be increased without changing the normal candidate contract.
        self.recovery_window = max(
            self.window,
            int(
                training.get(
                    "recovery_continuation_window",
                    512,
                )
            ),
        )
        self.recovery_max_candidates_per_event = max(
            self.max_candidates_per_event,
            int(
                training.get(
                    "recovery_max_candidates_per_event",
                    512,
                )
            ),
        )

        self.last_build_stats: dict = {}
        self.last_label_stats: dict = {}

    @staticmethod
    def _events_at(row: pd.Series) -> list[tuple[str, int]]:
        """Return every confirmed event on a candle, preserving event order."""

        events: list[tuple[str, int]] = []

        for column in EVENT_COLUMNS:
            value = row.get(column)
            if value in EVENT_DIRECTIONS:
                event_type = str(value)
                events.append(
                    (event_type, EVENT_DIRECTIONS[event_type])
                )

        return events

    @classmethod
    def _event_at(cls, row: pd.Series) -> tuple[str | None, int]:
        """Return the first confirmed event for backward-compatible callers."""

        events = cls._events_at(row)
        if not events:
            return None, 0
        return events[0]

    @staticmethod
    def _direction_name(direction: int) -> str:
        """Convert the internal event sign to the canonical public label."""

        if direction > 0:
            return "LONG"
        if direction < 0:
            return "SHORT"
        raise ValueError(f"Unsupported training direction: {direction}")

    @staticmethod
    def _unique_candidates(
        candidates: list[TrainingCandidate],
    ) -> pd.DataFrame:
        """Convert candidates to the canonical deduplicated DataFrame."""

        columns = [
            "candidate_index",
            "direction",
            "event_type",
            "distance_from_event",
        ]

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
            result.sort_values(
                ["candidate_index", "direction"],
                kind="stable",
            )
            .drop_duplicates(
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

    def _emit_candidates(
        self,
        frame: pd.DataFrame,
        event_indices: list[tuple[int, str, int]],
        window: int,
        spacing: int,
        max_candidates_per_event: int,
    ) -> list[TrainingCandidate]:
        """Emit causal continuation observations using explicit constraints."""

        candidates: list[TrainingCandidate] = []

        for event_index, event_type, direction in event_indices:
            emitted = 0
            last_candidate_index = -10**9

            for distance in range(window + 1):
                candidate_index = event_index + distance

                if candidate_index >= len(frame):
                    break

                if candidate_index < last_candidate_index + spacing:
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

                if emitted >= max_candidates_per_event:
                    break

        return candidates

    def build(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Build causal event/continuation candidates from all confirmed events."""

        columns = [
            "candidate_index",
            "direction",
            "event_type",
            "distance_from_event",
        ]

        if frame.empty:
            self.last_build_stats = {
                "rows": 0,
                "event_count": 0,
                "event_counts": {},
                "raw_candidates": 0,
                "unique_candidates": 0,
                "candidate_deduplication": 0,
                "target_candidates": 0,
                "recovery_mode": False,
                "recovery_enabled": self.recovery_enabled,
                "recovery_window": self.window,
                "recovery_spacing": self.min_spacing,
                "recovery_max_candidates_per_event": self.max_candidates_per_event,
            }
            return pd.DataFrame(columns=columns)

        event_indices: list[tuple[int, str, int]] = []
        event_counts: Counter[str] = Counter()

        for index in range(len(frame)):
            for event_type, direction in self._events_at(frame.iloc[index]):
                event_indices.append((index, event_type, direction))
                event_counts[event_type] += 1

        if not event_indices:
            self.last_build_stats = {
                "rows": int(len(frame)),
                "event_count": 0,
                "event_counts": {},
                "raw_candidates": 0,
                "unique_candidates": 0,
                "candidate_deduplication": 0,
                "target_candidates": 0,
                "recovery_mode": False,
                "recovery_enabled": self.recovery_enabled,
                "recovery_window": self.window,
                "recovery_spacing": self.min_spacing,
                "recovery_max_candidates_per_event": self.max_candidates_per_event,
            }
            return pd.DataFrame(columns=columns)

        # Candidates lost by sequence warm-up and forward-label horizon cannot
        # become usable labels. Include both losses in the recovery target.
        target_candidates = (
            self.minimum_labeled_candidates
            + self.label_horizon
            + self.sequence_length
            - 1
            if self.recovery_enabled
            else 0
        )

        candidates = self._emit_candidates(
            frame,
            event_indices,
            self.window,
            self.min_spacing,
            self.max_candidates_per_event,
        )
        result = self._unique_candidates(candidates)
        recovery_mode = False
        recovery_window = self.window
        recovery_spacing = self.min_spacing
        recovery_max = self.max_candidates_per_event

        if self.recovery_enabled and len(result) < target_candidates:
            recovery_mode = True
            recovery_spacing = 1
            recovery_window = self.recovery_window
            recovery_max = self.recovery_max_candidates_per_event

            candidates = self._emit_candidates(
                frame,
                event_indices,
                recovery_window,
                recovery_spacing,
                recovery_max,
            )
            result = self._unique_candidates(candidates)

        raw_candidate_count = len(candidates)

        self.last_build_stats = {
            "rows": int(len(frame)),
            "event_count": int(len(event_indices)),
            "event_counts": dict(sorted(event_counts.items())),
            "raw_candidates": int(raw_candidate_count),
            "unique_candidates": int(len(result)),
            "candidate_deduplication": int(
                raw_candidate_count - len(result)
            ),
            "target_candidates": int(target_candidates),
            "recovery_mode": recovery_mode,
            "recovery_enabled": self.recovery_enabled,
            "recovery_window": int(recovery_window),
            "recovery_spacing": int(recovery_spacing),
            "recovery_max_candidates_per_event": int(recovery_max),
        }

        print(
            "SMC training events: "
            f"events={len(event_indices)} "
            f"raw_candidates={raw_candidate_count} "
            f"unique_candidates={len(result)} "
            f"target_candidates={target_candidates} "
            f"recovery_mode={recovery_mode} "
            f"event_counts={dict(sorted(event_counts.items()))}"
        )

        return result

    @staticmethod
    def label_candidates(
        frame: pd.DataFrame,
        candidates: pd.DataFrame,
        atr: pd.Series,
        sl_multiplier: float,
        rr: float,
        horizon: int,
    ) -> pd.DataFrame:
        """Label candidates by TP-before-SL within the forward horizon."""

        rows: list[dict] = []
        skipped_horizon = 0
        skipped_atr = 0
        positive_labels = 0

        for candidate in candidates.itertuples(index=False):
            index = int(candidate.candidate_index)

            if index + horizon >= len(frame):
                skipped_horizon += 1
                continue

            atr_value = float(atr.iloc[index])
            if not np.isfinite(atr_value) or atr_value <= 0:
                skipped_atr += 1
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

            label = 1.0 if target_first else 0.0
            positive_labels += int(label == 1.0)

            rows.append(
                {
                    "candidate_index": index,
                    "direction": direction_name,
                    "event_type": str(candidate.event_type),
                    "distance_from_event": int(candidate.distance_from_event),
                    "label": label,
                }
            )

        result = pd.DataFrame(rows)
        SMCTrainingEventBuilder._record_label_stats(
            result,
            skipped_horizon,
            skipped_atr,
            positive_labels,
        )
        return result

    @staticmethod
    def _record_label_stats(
        result: pd.DataFrame,
        skipped_horizon: int,
        skipped_atr: int,
        positive_labels: int,
    ) -> None:
        """Compatibility hook retained for callers that inspect label output only."""

        _ = result, skipped_horizon, skipped_atr, positive_labels
