"""Demo data: Yahoo Finance hourly bars (about two years, free, personal use).

Only for demonstrating the pipeline. Yahoo's hourly history is too short for
the spec's 3-year fit plus 2 years of walk-forward, so acceptance results
come from IBKR data. Downloaded bars stay local (the bar cache is
gitignored).
"""

from __future__ import annotations

import pandas as pd

from regime_trader.bars import COLUMNS, SESSION_CLOSE, SESSION_OPEN, TIMEZONE


def normalize_yahoo(raw: pd.DataFrame, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """yfinance frame (single or multi-level columns) -> the bar schema, RTH
    only. With `now`, a bar that has not finished an hour by then is dropped:
    Yahoo, like IB, includes the bar still forming."""
    frame = raw.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    frame.columns = [str(c).lower() for c in frame.columns]
    index = pd.DatetimeIndex(frame.index)
    frame.index = (
        index.tz_localize("UTC").tz_convert(TIMEZONE) if index.tz is None else index.tz_convert(TIMEZONE)
    )
    clock = frame.index - frame.index.normalize()
    frame = frame[(clock >= SESSION_OPEN) & (clock < SESSION_CLOSE)]
    frame = frame[list(COLUMNS)].astype(float).dropna()
    if now is not None:
        frame = frame[frame.index + pd.Timedelta(hours=1) <= now]
    return frame[frame["volume"] > 0]


def download_hourly(symbol: str, period: str = "730d") -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(symbol, period=period, interval="1h", progress=False, auto_adjust=False)
    return normalize_yahoo(raw, now=pd.Timestamp.now(tz=TIMEZONE))
