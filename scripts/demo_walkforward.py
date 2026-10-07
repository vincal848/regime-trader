"""The Yahoo demo walk-forward: regenerates docs/DEMO_RESULTS.md and docs/img/.

    pip install -e ".[demo]"
    regime-trader --root demo fetch --source yahoo
    python scripts/demo_walkforward.py demo 2025-01-02

Every number and figure the README and docs/DEMO.md quote comes from this
run, so the documentation cannot drift away from the code.

A demo, not the acceptance test: Yahoo's free hourly history is too short for
the spec's fit-from-2018 walk-forward and a 12-month locked holdout.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from regime_trader.backtest import (
    Acceptance,
    BacktestConfig,
    acceptance,
    acceptance_report,
    run_buy_and_hold,
)
from regime_trader.store import BarCache, load_playbooks

DOCS = Path(__file__).resolve().parents[1] / "docs"
STATE_COLOURS = {
    "CALM_UP": "#cfe8cf",
    "CHOP": "#e8e3cf",
    "STRESS": "#f3d9b8",
    "CRASH": "#f2b8b8",
    "NONE": "#eeeeee",
}
ZOOM = ("2025-03-24", "2025-05-16")  # the April 2025 tariff sell-off


def _colour(label: str) -> str:
    return STATE_COLOURS.get(label.rsplit("_", 1)[0] if label[-1:].isdigit() else label, "#eeeeee")


def _shade(axis: plt.Axes, active: pd.Series) -> None:
    """Shade the background by the active state, one span per run of bars."""
    runs = (active != active.shift()).cumsum()
    for _, run in active.groupby(runs):
        axis.axvspan(run.index[0], run.index[-1], color=_colour(str(run.iloc[0])), lw=0, zorder=0)


def _figures(result: Acceptance, hold: pd.Series, bars: pd.DataFrame) -> None:
    run = result.result
    (DOCS / "img").mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(10, 4.2))
    _shade(axis, run.active)
    axis.plot(hold.index, hold / hold.iloc[0], color="#555555", lw=1.2, label="Buy-and-hold SPY")
    axis.plot(
        run.equity.index, run.equity / run.equity.iloc[0], color="#1f5fa8", lw=1.6, label="Regime trader"
    )
    axis.set_ylabel("Growth of $1")
    axis.set_title("Walk-forward on hourly SPY (Yahoo demo data): background = active HMM state")
    axis.legend(loc="upper left", frameon=False)
    figure.tight_layout()
    figure.savefig(DOCS / "img" / "equity.png", dpi=130)
    plt.close(figure)

    start, end = (pd.Timestamp(day, tz="America/New_York") for day in ZOOM)
    window = slice(start, end)
    figure, price_axis = plt.subplots(figsize=(10, 4.2))
    _shade(price_axis, run.active[window])
    price_axis.plot(bars["close"][window], color="#333333", lw=1.1)
    price_axis.set_ylabel("SPY close")
    shares_axis = price_axis.twinx()
    shares_axis.fill_between(
        run.positions[window].index, run.positions[window], step="post", color="#1f5fa8", alpha=0.35
    )
    shares_axis.set_ylabel("Shares held")
    price_axis.set_title("April 2025 sell-off: CRASH (red) covers the drop; the trader stays flat")
    figure.tight_layout()
    figure.savefig(DOCS / "img" / "april_2025.png", dpi=130)
    plt.close(figure)


def _results_markdown(result: Acceptance, bars: pd.DataFrame, config: BacktestConfig) -> str:
    run = result.result
    lines = [
        "# Demo results (generated)",
        "",
        "<!-- Written by scripts/demo_walkforward.py; do not edit by hand. -->",
        "",
        f"Data: {len(bars):,} hourly RTH bars of SPY, {bars.index[0]:%Y-%m-%d} to {bars.index[-1]:%Y-%m-%d}. "
        f"Walk-forward from {config.test_start:%Y-%m-%d}, refit every {config.refit_days} days. No holdout.",
        "",
        "```",
        acceptance_report(result),
        "```",
        "",
        f"Orders that would have waited for manual approval (over $25,000): {run.manual_approvals}",
        "",
        "## Refits",
        "",
        "| Date | K | Labels | Drift | Sizing |",
        "|---|---|---|---|---|",
    ]
    for refit in run.refits:
        drift = "first fit" if refit.drift is None else ("**drift**" if refit.drift.drifted else "none")
        sizing = "calibrated" if refit.calibrated else "quarter cap"
        lines.append(
            f"| {refit.ts:%Y-%m-%d} | {refit.n_states} | {', '.join(refit.labels)} | {drift} | {sizing} |"
        )
    trades = pd.DataFrame([{"state": t.state, "pnl": t.pnl} for t in run.trades])
    lines += [
        "",
        "## Per state",
        "",
        "| State | Bars active | Share | Trades | P&L ($) |",
        "|---|---|---|---|---|",
    ]
    for state, row in run.per_state.iterrows():
        mine = trades[trades["state"] == state]["pnl"] if len(trades) else pd.Series(dtype=float)
        lines.append(
            f"| {state} | {int(row['bars']):,} | {row['share']:.1%} | {len(mine)} | {mine.sum():+,.0f} |"
        )
    return "\n".join(lines) + "\n"


def main(root: Path, test_start: str) -> None:
    bars = BarCache(root / "data").load("SPY")
    playbooks = load_playbooks(root / "playbooks")
    config = BacktestConfig(test_start=pd.Timestamp(test_start, tz="America/New_York"))
    result = acceptance(bars, playbooks, config)
    markdown = _results_markdown(result, bars, config)
    (DOCS / "DEMO_RESULTS.md").write_text(markdown, encoding="utf-8")
    _figures(result, run_buy_and_hold(bars, config), bars)
    print(markdown)


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2])
