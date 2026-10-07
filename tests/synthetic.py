"""Synthetic hourly SPY-like bars for tests (IBKR RTH grid, 7 bars a day)."""

import numpy as np
import pandas as pd

IBKR_STARTS = ["09:30", "10:00", "11:00", "12:00", "13:00", "14:00", "15:00"]


def make_bars(n_days: int = 60, seed: int = 0, start: str = "2024-01-02", vol: float = 0.003) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, periods=n_days)
    index = pd.DatetimeIndex(
        [pd.Timestamp(f"{d.date()} {t}", tz="America/New_York") for d in days for t in IBKR_STARTS]
    )
    n = len(index)
    log_close = np.log(450.0) + np.cumsum(rng.normal(0.0, vol, n))
    close = np.exp(log_close)
    open_ = np.concatenate([[450.0], close[:-1]]) * np.exp(rng.normal(0.0, vol / 4, n))
    wiggle = np.abs(rng.normal(0.0, vol / 2, n))
    high = np.maximum(open_, close) * np.exp(wiggle)
    low = np.minimum(open_, close) * np.exp(-wiggle)
    hour_shape = np.tile([3.0, 1.5, 1.0, 0.8, 0.9, 1.2, 2.5], n_days)  # intraday U-shape
    volume = np.round(1e6 * hour_shape * rng.lognormal(0.0, 0.2, n))
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=index
    )
