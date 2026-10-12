"""Metadata-only Gregorian UTC phases; no observation or physics dependency.

UTC follows the source's POSIX nanosecond clock, without leap-second insertion.
Calendar routing uses detached clocks. Within each calendar interval, phases
remain differentiable functions of the original relative-seconds tensor.
The annual phase is continuous and piecewise smooth at year boundaries: its
derivative changes when the Gregorian year length changes.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import torch

UTC_FEATURE_CONTRACT_ID = "UTC_GREGORIAN_ACTUAL_YEAR_AND_DAY_PHASE_V7"
UTC_FEATURE_NAMES = ("utc_annual_sin", "utc_annual_cos", "utc_day_sin", "utc_day_cos")
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
_SECOND_NS = 10**9
_DAY_NS = 86400 * _SECOND_NS


def _datetime_ns(value: datetime) -> int:
    delta = value - _EPOCH
    return (delta.days * 86400 + delta.seconds) * _SECOND_NS


def utc_cycle_features(relative_seconds: torch.Tensor, utc_origin_ns: torch.Tensor) -> torch.Tensor:
    """Return [B,Q,4] phases for origin UTC plus relative seconds.

    relative_seconds is float64 [Q] or [B,Q]. The origin is int64 [B],
    obtained solely from external window metadata. No calendar fallbacks are
    allowed. Tokens use their unchanged native interval-start clock; query and
    initialization use the same mapping, including negative history times.
    """
    if not isinstance(relative_seconds, torch.Tensor) or relative_seconds.dtype != torch.float64:
        raise ValueError("UTC phases require original float64 relative seconds")
    if relative_seconds.ndim not in (1, 2) or not bool(torch.isfinite(relative_seconds).all()):
        raise ValueError("Finite [Q] or [B,Q] relative seconds required")
    if (
        not isinstance(utc_origin_ns, torch.Tensor)
        or utc_origin_ns.dtype != torch.int64
        or utc_origin_ns.ndim != 1
    ):
        raise ValueError("External int64 [B] UTC origin metadata required")
    if utc_origin_ns.numel() == 0 or relative_seconds.shape[-1] == 0:
        raise ValueError("Nonempty UTC origin and query clock required")
    batch_size = len(utc_origin_ns)
    seconds = (
        relative_seconds[None].expand(batch_size, -1)
        if relative_seconds.ndim == 1
        else relative_seconds
    )
    if seconds.shape[0] != batch_size:
        raise ValueError("UTC origin/query batch dimensions differ")
    origins = utc_origin_ns.detach().cpu().tolist()
    clocks = seconds.detach().cpu().tolist()
    year_start, year_length, day_start = [], [], []
    for origin, row in zip(origins, clocks):
        ys, yl, ds = [], [], []
        for clock in row:
            # Detached integer clock chooses only the Gregorian calendar interval.
            absolute_ns = origin + round(clock * _SECOND_NS)
            stamp = _EPOCH + timedelta(seconds=absolute_ns // _SECOND_NS)
            first = datetime(stamp.year, 1, 1, tzinfo=timezone.utc)
            next_first = datetime(stamp.year + 1, 1, 1, tzinfo=timezone.utc)
            ys.append((_datetime_ns(first) - origin) / _SECOND_NS)
            yl.append(float((next_first - first).days * 86400))
            ds.append(((absolute_ns // _DAY_NS) * _DAY_NS - origin) / _SECOND_NS)
        year_start.append(ys)
        year_length.append(yl)
        day_start.append(ds)

    def make(values):
        return torch.tensor(values, dtype=torch.float64, device=seconds.device)

    annual = (seconds - make(year_start)) / make(year_length) * (2 * math.pi)
    daily = (seconds - make(day_start)) / 86400.0 * (2 * math.pi)
    return torch.stack((annual.sin(), annual.cos(), daily.sin(), daily.cos()), -1)
