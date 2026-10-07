"""The deterministic, strictly causal feature pipeline (spec §4).

Five raw features per bar t, each a function of bars <= t only:

    ret           ln(close_t / close_{t-1}); the session's first bar includes the overnight gap
    rv            standard deviation of the last 21 log returns (about three sessions)
    range         ln(high_t / low_t)
    volume_ratio  volume_t / median volume of the same bar-of-day over the previous 20 sessions
    trend         (ln close_t - its 70-bar EMA) / rv: about two weeks of trend, in volatility units

Each raw feature x is standardized with an expanding mean and standard
deviation that end at bar t-1, so z_t never uses bar t's own statistics:

    z_t = (x_t - mean(x_..t-1)) / std(x_..t-1)

Everything is a rolling, expanding or shifted operation, so changing a bar
after t cannot change any row <= t (tests/test_bars_features.py).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd

from regime_trader.bars import bar_of_day

FEATURES = ("ret", "rv", "range", "volume_ratio", "trend")
Z_FEATURES = tuple(f"z_{name}" for name in FEATURES)

VOL_WINDOW = 21
VOLUME_SESSIONS = 20
TREND_SPAN = 70
MIN_HISTORY = 140  # bars before a z-score is defined (20 sessions)


def compute_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Raw and standardized features, one row per bar (NaN during warm-up)."""
    log_close = np.log(bars["close"])
    ret = log_close.diff()
    rv = ret.rolling(VOL_WINDOW).std()
    bar_range = np.log(bars["high"] / bars["low"])

    slot = pd.Series(bar_of_day(pd.DatetimeIndex(bars.index)), index=bars.index)
    typical_volume = (
        bars["volume"]
        .groupby(slot)
        .transform(lambda volume: volume.shift(1).rolling(VOLUME_SESSIONS).median())
    )
    volume_ratio = bars["volume"] / typical_volume

    trend = (log_close - log_close.ewm(span=TREND_SPAN, adjust=False).mean()) / rv

    raw = pd.DataFrame(
        {"ret": ret, "rv": rv, "range": bar_range, "volume_ratio": volume_ratio, "trend": trend},
        index=bars.index,
    ).replace([np.inf, -np.inf], np.nan)
    past_mean = raw.expanding(min_periods=MIN_HISTORY).mean().shift(1)
    past_std = raw.expanding(min_periods=MIN_HISTORY).std().shift(1)
    z = ((raw - past_mean) / past_std).add_prefix("z_")
    return pd.concat([raw, z], axis=1)


def feature_rows(features: pd.DataFrame) -> list[dict[str, float]]:
    """Feature rows as the engine reads them: one {name: value} per bar."""
    return [{str(k): float(v) for k, v in row.items()} for row in features.to_dict("records")]


def healthy(features: pd.DataFrame) -> npt.NDArray[np.bool_]:
    """Rows where every raw and standardized feature is finite: the only
    rows a model is fitted on or a decision is made from."""
    mask: npt.NDArray[np.bool_] = np.isfinite(features[[*FEATURES, *Z_FEATURES]].to_numpy()).all(axis=1)
    return mask
