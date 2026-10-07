"""The local dashboard (spec §14): `regime-trader dashboard` opens it.

`dashboard_data` gathers everything the page shows from the journal and
the live checkpoint. It is read-only, so the page can never trade.
`main` is the Streamlit page itself.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from regime_trader.bars import TIMEZONE
from regime_trader.engine import playbook_for
from regime_trader.hmm import expected_duration
from regime_trader.live import load_checkpoint
from regime_trader.nightly import round_trips
from regime_trader.playbook import Playbook
from regime_trader.refit import Fit
from regime_trader.risk import RiskLimits
from regime_trader.store import Journal


@dataclass(frozen=True)
class DashboardData:
    probabilities: pd.DataFrame  # filtered, one row per decided bar, one column per state
    current_state: str | None
    expected_remaining_bars: float  # memoryless: equal to the state's expected duration
    playbook: Playbook | None
    last_action: str
    last_reasons: tuple[str, ...]
    position_shares: int
    equity: float
    realized_pnl: float
    daily_loss_headroom: float  # fraction of equity left before the daily loss limit
    drawdown_headroom: float  # fraction of peak left before the kill switch
    events: pd.DataFrame


def dashboard_data(
    journal: Journal, fit: Fit, playbooks: Mapping[str, Playbook], state_path: Path, limits: RiskLimits
) -> DashboardData:
    decisions = journal.decisions()
    index = pd.DatetimeIndex(pd.to_datetime(decisions["ts"], utc=True)).tz_convert(TIMEZONE)
    probabilities = pd.DataFrame(
        [json.loads(p) for p in decisions["probabilities"]], index=index, columns=list(fit.model.labels)
    )
    last = decisions.iloc[-1] if len(decisions) else None
    state = str(last["active"]) if last is not None and last["active"] else None
    durations = expected_duration(fit.model.hmm)
    checkpoint = load_checkpoint(state_path)
    equity = checkpoint.last_equity if checkpoint else math.nan
    position = checkpoint.engine.position if checkpoint else None

    def headroom(limit: float, reference: float | None) -> float:
        if reference is None or not reference > 0:
            return math.nan
        used = 1.0 - equity / reference
        return max(0.0, limit - used) if math.isfinite(used) else math.nan

    labels = fit.model.labels
    return DashboardData(
        probabilities=probabilities,
        current_state=state,
        expected_remaining_bars=float(durations[labels.index(state)]) if state in labels else math.nan,
        playbook=playbook_for(playbooks, state),
        last_action="none"
        if last is None
        else f"{last['order_status']}: target {last['target_shares']} shares",
        last_reasons=() if last is None else tuple(json.loads(last["reasons"])),
        position_shares=position.shares if position else 0,
        equity=equity,
        realized_pnl=sum(t.pnl for t in round_trips(journal.fills(), decisions)),
        daily_loss_headroom=headroom(
            limits.daily_loss_limit, checkpoint.start_of_day_equity if checkpoint else None
        ),
        drawdown_headroom=headroom(limits.max_drawdown, checkpoint.peak_equity if checkpoint else None),
        events=journal.events().tail(50),
    )


def main() -> None:  # pragma: no cover -- the Streamlit page; the data above is tested
    import sys

    import streamlit as st

    from regime_trader.store import load_fit, load_playbooks

    root = Path(sys.argv[sys.argv.index("--root") + 1] if "--root" in sys.argv else ".")
    st.set_page_config(page_title="Regime trader", layout="wide")
    data = dashboard_data(
        Journal(root / "journal.db"),
        load_fit(root / "models" / "fit.json"),
        load_playbooks(root / "playbooks"),
        root / "live_state.json",
        RiskLimits(),
    )
    st.title("Regime trader (IBKR paper)")
    left, middle, right = st.columns(3)
    left.metric("State", data.current_state or "none", f"~{data.expected_remaining_bars:.0f} bars expected")
    middle.metric("Position", f"{data.position_shares} shares", f"realized {data.realized_pnl:+,.2f}")
    right.metric("Equity", f"{data.equity:,.0f}")
    st.subheader("Filtered state probabilities")
    st.area_chart(data.probabilities)
    st.subheader("Last action")
    st.write(data.last_action)
    st.write(list(data.last_reasons))
    st.subheader("Active playbook")
    st.write(data.playbook)
    st.subheader("Risk headroom")
    st.write(
        f"Daily loss: {data.daily_loss_headroom:.2%} left. "
        f"Drawdown before the kill switch: {data.drawdown_headroom:.2%} left."
    )
    st.subheader("Events")
    st.dataframe(data.events)
    st.caption("Refreshes on reload; the trader writes the journal once per bar.")


if __name__ == "__main__":  # pragma: no cover
    main()
