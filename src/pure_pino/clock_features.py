"""Declared SI-clock representation only; never changes raw observation support.

The new four-feature affine map is invertible. Its constants are physical
units and the predeclared 730-day context, not target/source fitted statistics.
Longer histories are not clipped or declared supported by this numerical map.
"""

import math

import torch

LEGACY_CLOCK_CONTRACT_ID = "RELATIVE_SUPPORT_WINDOW_DAYS_V7"
CLOCK_FEATURE_CONTRACT_ID = "RELATIVE_SUPPORT_SPLIT_SI_AFFINE_CLOCK_V1"
HISTORY_POSITION_SCALE_S = 730.0 * 86400.0
NATIVE_DURATION_SCALE_S = 3600.0
AVAILABILITY_DELAY_SCALE_S = 3600.0


def validate_clock_config(contract_id, mode, position_scale_s, duration_scale_s, delay_scale_s):
    permitted = {
        (LEGACY_CLOCK_CONTRACT_ID, "legacy_relative_days"),
        (CLOCK_FEATURE_CONTRACT_ID, "split_native_support_affine"),
    }
    if (contract_id, mode) not in permitted:
        raise ValueError("Explicit matching token-clock contract/mode required")
    actual = (position_scale_s, duration_scale_s, delay_scale_s)
    expected = (HISTORY_POSITION_SCALE_S, NATIVE_DURATION_SCALE_S, AVAILABILITY_DELAY_SCALE_S)
    if any(
        not isinstance(v, (int, float)) or not math.isfinite(v) or v != e
        for v, e in zip(actual, expected)
    ):
        raise ValueError("Clock representation requires predeclared 730-day/hour SI constants")


def token_clock_features(
    start_s,
    end_s,
    available_s,
    *,
    contract_id,
    mode,
    position_scale_s=HISTORY_POSITION_SCALE_S,
    duration_scale_s=NATIVE_DURATION_SCALE_S,
    delay_scale_s=AVAILABILITY_DELAY_SCALE_S,
    legacy_time_scale_s=86400.0,
):
    validate_clock_config(contract_id, mode, position_scale_s, duration_scale_s, delay_scale_s)
    if mode == "legacy_relative_days":
        # Exact old evaluation order preserves old configuration semantics.
        return torch.stack(
            (
                start_s / legacy_time_scale_s,
                end_s / legacy_time_scale_s,
                (end_s - start_s) / legacy_time_scale_s,
                available_s / legacy_time_scale_s,
            ),
            -1,
        )
    for clock in (start_s, end_s, available_s):
        if not torch.is_tensor(clock) or clock.dtype != torch.float64:
            raise ValueError("New clock representation requires raw float64 SI seconds")
        if (
            clock.shape != start_s.shape
            or clock.device != start_s.device
            or not bool(torch.isfinite(clock).all())
        ):
            raise ValueError("Finite shape/device-consistent raw SI clocks required")
    if not bool((end_s >= start_s).all()):
        raise ValueError("Nonnegative original support duration required")
    return torch.stack(
        (
            start_s / position_scale_s,
            end_s / position_scale_s,
            (end_s - start_s) / duration_scale_s,
            (available_s - end_s) / delay_scale_s,
        ),
        -1,
    )


def recover_raw_seconds(
    features, *, position_scale_s=HISTORY_POSITION_SCALE_S, delay_scale_s=AVAILABILITY_DELAY_SCALE_S
):
    """Inverse witness for the NEW representation, not a forcing conversion."""
    if features.shape[-1] != 4:
        raise ValueError("Four declared support-clock features required")
    start = features[..., 0] * position_scale_s
    end = features[..., 1] * position_scale_s
    available = end + features[..., 3] * delay_scale_s
    return start, end, available
