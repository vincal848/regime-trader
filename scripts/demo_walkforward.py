"""The Yahoo demo walk-forward, with the detail docs/DEMO.md reports.

    regime-trader --root demo fetch --source yahoo
    python scripts/demo_walkforward.py demo 2025-01-02

A demo, not the acceptance test: Yahoo's free hourly history is too short for
the spec's fit-from-2018 walk-forward and a 12-month locked holdout.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from regime_trader.backtest import BacktestConfig, acceptance, acceptance_report
from regime_trader.store import BarCache, load_playbooks


def main(root: Path, test_start: str) -> None:
    bars = BarCache(root / "data").load("SPY")
    playbooks = load_playbooks(root / "playbooks")
    config = BacktestConfig(test_start=pd.Timestamp(test_start, tz="America/New_York"))
    result = acceptance(bars, playbooks, config)
    run = result.result
    print(f"Data: {len(bars)} hourly bars, {bars.index[0]} to {bars.index[-1]}")
    print(f"Walk-forward from {config.test_start.date()}, refit every {config.refit_days} days\n")
    print(acceptance_report(result))
    print(f"\nOrders that would have waited for manual approval (> $25,000): {run.manual_approvals}")
    print("\nRefits:")
    for refit in run.refits:
        drift = (
            "first fit"
            if refit.drift is None
            else ("DRIFT: " + "; ".join(refit.drift.reasons) if refit.drift.drifted else "no drift")
        )
        sizing = "calibrated" if refit.calibrated else "quarter-cap sizing"
        print(f"  {refit.ts:%Y-%m-%d}  K={refit.n_states}  {', '.join(refit.labels)}  ({drift}; {sizing})")
    print("\nPer active state (log return over the bars it was active):")
    print(run.per_state.round(5).to_string())
    print("\nTrades by state:")
    trades = pd.DataFrame([{"state": t.state, "pnl": t.pnl} for t in run.trades])
    if len(trades):
        print(trades.groupby("state")["pnl"].agg(["count", "sum", "mean"]).round(2).to_string())


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2])
