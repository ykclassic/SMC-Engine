from __future__ import annotations

import numpy as np
import torch

from models.feature_engineering import FEATURE_COLUMNS
from scripts.ablation_feature_groups import (
    BASE_FEATURES,
    EXPERIMENTS,
    FEATURE_GROUPS,
    _mask_partition,
)


def test_ablation_groups_are_disjoint_and_known() -> None:
    assert len(BASE_FEATURES) == 16
    assert set(BASE_FEATURES).issubset(FEATURE_COLUMNS)

    groups = [set(values) for values in FEATURE_GROUPS.values()]
    assert all(group.issubset(FEATURE_COLUMNS) for group in groups)
    assert not (groups[0] & groups[1])
    assert not (groups[0] & groups[2])
    assert not (groups[1] & groups[2])


def test_ablation_matrix_is_cumulative() -> None:
    assert len(EXPERIMENTS["baseline"]) == 16
    assert len(EXPERIMENTS["baseline_plus_geometry"]) == 24
    assert len(EXPERIMENTS["baseline_plus_geometry_liquidity"]) == 26
    assert len(EXPERIMENTS["baseline_plus_geometry_liquidity_regime"]) == 29
    assert tuple(FEATURE_COLUMNS) == EXPERIMENTS["full_smc_v4"]


def test_masking_preserves_selected_features_and_zeros_excluded_features() -> None:
    tensor = torch.arange(2 * 3 * len(FEATURE_COLUMNS), dtype=torch.float32).reshape(
        2, 3, len(FEATURE_COLUMNS)
    )
    masked = _mask_partition(tensor, BASE_FEATURES)

    base_indices = [FEATURE_COLUMNS.index(name) for name in BASE_FEATURES]
    extra_indices = [
        index for index in range(len(FEATURE_COLUMNS)) if index not in base_indices
    ]

    assert torch.equal(masked[..., base_indices], tensor[..., base_indices])
    assert torch.count_nonzero(masked[..., extra_indices]).item() == 0
    assert np.isfinite(masked.numpy()).all()
