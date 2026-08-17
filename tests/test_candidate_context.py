import numpy as np
import pandas as pd
import pytest
import torch

from models.candidate_context import CONTEXT_COLUMNS, context_vector, validate_candidate_context
from models.gru import SignalValidatorGRU


def _labels():
    return pd.DataFrame(
        {
            "candidate_index": [10, 10, 11],
            "direction": ["LONG", "SHORT", "LONG"],
            "event_type": ["BULLISH_BOS", "BEARISH_BOS", "BULLISH_FVG"],
            "label": [1.0, 0.0, 1.0],
        }
    )


def test_directional_candidates_are_distinct_contexts():
    long_context = context_vector("LONG", "BULLISH_BOS")
    short_context = context_vector("SHORT", "BEARISH_BOS")
    assert long_context.shape == (len(CONTEXT_COLUMNS),)
    assert short_context.shape == long_context.shape
    assert long_context[0] == 1.0
    assert short_context[0] == -1.0
    assert not np.array_equal(long_context, short_context)


def test_candidate_integrity_allows_same_index_with_opposite_direction():
    validate_candidate_context(_labels())


def test_duplicate_candidate_direction_is_rejected():
    labels = _labels()
    labels.loc[2, "candidate_index"] = 10
    labels.loc[2, "direction"] = "LONG"
    with pytest.raises(ValueError, match="Duplicate candidate direction"):
        validate_candidate_context(labels)


def test_gru_requires_candidate_context():
    model = SignalValidatorGRU(input_dim=16)
    x = torch.zeros((2, 32, 16))
    context = torch.zeros((2, len(CONTEXT_COLUMNS)))
    output = model(x, event_context=context)
    assert output.shape == (2, 1)
    with pytest.raises(ValueError, match="context is required"):
        model(x)
