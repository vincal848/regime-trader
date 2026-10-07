"""Hourly bar schema and quality checks (spec §3).

A bar frame has columns `open, high, low, close, volume` and a tz-aware
DatetimeIndex of bar *start* times inside US regular trading hours
(09:30-16:00 New York). IBKR labels RTH hourly bars 09:30, 10:00, 11:00, ...;
Yahoo labels them 09:30, 10:30, ...; both are valid, so the bar-of-day is the
bar's position within its session rather than a fixed clock grid.

Any failed check raises `BarError`. Upstream the live loop treats that
exactly like stale data: go flat.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd

COLUMNS = ("open", "high", "low", "close", "volume")
TIMEZONE = "America/New_York"
SESSION_OPEN = pd.Timedelta(hours=9, minutes=30)
SESSION_CLOSE = pd.Timedelta(hours=16)
BARS_PER_SESSION = 7  # hourly RTH bars in a full session, on either grid


class BarError(ValueError):
    """A bar frame failed a quality check."""


def validate_bars(bars: pd.DataFrame) -> None:
    """Raise `BarError` unless every bar passes every quality check."""
    if list(bars.columns[: len(COLUMNS)]) != list(COLUMNS):
        raise BarError(f"columns must start with {COLUMNS}, got {list(bars.columns)}")
    index = bars.index
    if not isinstance(index, pd.DatetimeIndex) or index.tz is None:
        raise BarError("index must be a timezone-aware DatetimeIndex")
    if not index.is_monotonic_increasing or not index.is_unique:
        raise BarError("bar timestamps must be strictly increasing")
    values = bars[list(COLUMNS)]
    if values.isna().any(axis=None):
        raise BarError("bars have missing values")
    if (bars["high"] < bars[["open", "close"]].max(axis=1)).any():
        raise BarError("high below open or close")
    if (bars["low"] > bars[["open", "close"]].min(axis=1)).any() or (bars["low"] <= 0).any():
        raise BarError("low above open or close, or not positive")
    if (bars["volume"] < 0).any():
        raise BarError("negative volume")
    if not in_regular_hours(index).all():
        raise BarError("bars outside regular trading hours (09:30-16:00 New York)")


def in_regular_hours(index: pd.DatetimeIndex) -> npt.NDArray[np.bool_]:
    """Which bars start inside the New York regular session (09:30-16:00)."""
    local = index.tz_convert(TIMEZONE)
    clock = local - local.normalize()
    inside: npt.NDArray[np.bool_] = np.asarray((clock >= SESSION_OPEN) & (clock < SESSION_CLOSE))
    return inside


def bar_of_day(index: pd.DatetimeIndex) -> npt.NDArray[np.int64]:
    """Position of each bar within its New York trading session (0 = first)."""
    session = index.tz_convert(TIMEZONE).normalize()
    counts = pd.Series(1, index=index).groupby(session).cumsum().to_numpy()
    return (counts - 1).astype(np.int64)
