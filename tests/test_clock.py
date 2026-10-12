"""SI representation and Gregorian clock controls, with no observed labels."""

import math
from datetime import datetime, timezone

import pytest
import torch

from examples.external_field import exterior_batch, make_fixture_model
from pure_pino.clock_features import (
    AVAILABILITY_DELAY_SCALE_S,
    CLOCK_FEATURE_CONTRACT_ID,
    HISTORY_POSITION_SCALE_S,
    NATIVE_DURATION_SCALE_S,
    recover_raw_seconds,
    token_clock_features,
)
from pure_pino.utc_features import utc_cycle_features


def test_exact_model_clock_features_and_reversible_raw_seconds():
    batch = exterior_batch(nodes=4)
    snapshots = {
        name: getattr(batch, name).clone()
        for name in ("tokens", "token_start", "token_end", "token_available")
    }
    encoded = make_fixture_model(width=8).token_clock_features(batch)
    start, end, available = recover_raw_seconds(encoded)
    for actual, name in zip(
        (start, end, available), ("token_start", "token_end", "token_available")
    ):
        torch.testing.assert_close(actual, snapshots[name], rtol=1e-15, atol=1e-8)
    torch.testing.assert_close(
        encoded[..., 2] * NATIVE_DURATION_SCALE_S,
        end - start,
        rtol=1e-15,
        atol=1e-8,
    )
    assert HISTORY_POSITION_SCALE_S == 730 * 86400
    assert NATIVE_DURATION_SCALE_S == AVAILABILITY_DELAY_SCALE_S == 3600
    for name, original in snapshots.items():
        torch.testing.assert_close(getattr(batch, name), original, rtol=0, atol=0)


@pytest.mark.parametrize("mutation", ["nan", "negative_duration", "float32"])
def test_invalid_raw_clock_fails_without_clipping(mutation):
    batch = exterior_batch(nodes=4)
    if mutation == "nan":
        batch.token_available[0, 0] = float("nan")
    elif mutation == "negative_duration":
        batch.token_end[0, 0] = batch.token_start[0, 0] - 1
    else:
        batch.token_start = batch.token_start.float()
    with pytest.raises(ValueError):
        token_clock_features(
            batch.token_start,
            batch.token_end,
            batch.token_available,
            contract_id=CLOCK_FEATURE_CONTRACT_ID,
            mode="split_native_support_affine",
        )


def test_gregorian_leap_year_and_daily_phase_are_independent_of_targets():
    origin_ns = int(datetime(2020, 2, 29, tzinfo=timezone.utc).timestamp()) * 10**9
    query = torch.tensor([0.0, 43200.0], dtype=torch.float64)
    actual = utc_cycle_features(query, torch.tensor([origin_ns]))
    expected = torch.tensor(
        [
            [
                [math.sin(2 * math.pi * 59 / 366), math.cos(2 * math.pi * 59 / 366), 0.0, 1.0],
                [math.sin(2 * math.pi * 59.5 / 366), math.cos(2 * math.pi * 59.5 / 366), 0.0, -1.0],
            ]
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
