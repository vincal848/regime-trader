"""Performance statistics and the acceptance gates (spec §11).

Returns are daily: the last equity value of each New York session, so the
Sharpe ratio and t-statistic are on the conventional daily scale (Sharpe
annualized by the square root of 252). Hourly marks would overstate the
number of independent observations.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from regime_trader.bars import TIMEZONE

TRADING_DAYS = 252


def daily_returns(equity: pd.Series) -> pd.Series:
    index = pd.DatetimeIndex(equity.index)
    session_close = equity.groupby(index.tz_convert(TIMEZONE).normalize()).last()
    returns: pd.Series = session_close.pct_change().dropna()
    return returns


def sharpe(returns: pd.Series) -> float:
    sd = float(returns.std(ddof=1))
    return float(returns.mean()) / sd * float(np.sqrt(TRADING_DAYS)) if sd > 0 else 0.0


def t_statistic(returns: pd.Series) -> float:
    sd = float(returns.std(ddof=1))
    return float(returns.mean()) / (sd / float(np.sqrt(len(returns)))) if sd > 0 else 0.0


def max_drawdown(equity: pd.Series) -> float:
    return float((1.0 - equity / equity.cummax()).max())


def hit_rate(trade_pnls: Sequence[float]) -> float:
    return float(np.mean([pnl > 0 for pnl in trade_pnls])) if trade_pnls else 0.0


@dataclass(frozen=True)
class Gates:
    min_sharpe: float = 1.5
    max_drawdown: float = 0.15
    min_hit_rate: float = 0.55
    min_t_statistic: float = 2.0


@dataclass(frozen=True)
class GateResult:
    passed: bool
    checks: dict[str, bool]


def evaluate_gates(
    stats: Mapping[str, float], baseline_sharpes: Mapping[str, float], gates: Gates
) -> GateResult:
    """Every check must pass, out of sample and after costs."""
    checks = {
        "sharpe": stats["sharpe"] > gates.min_sharpe,
        "max_drawdown": stats["max_drawdown"] < gates.max_drawdown,
        "hit_rate": stats["hit_rate"] > gates.min_hit_rate,
        "t_statistic": stats["t_statistic"] > gates.min_t_statistic,
    }
    for name, baseline in baseline_sharpes.items():
        checks[f"beats {name}"] = stats["sharpe"] > baseline
    return GateResult(all(checks.values()), checks)
