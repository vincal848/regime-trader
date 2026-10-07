"""The live trader: one completed bar in, at most one order out (spec §8, §10, §14).

Each bar, just after it closes:

1. **Connection.** While disconnected nothing can be sent. The time spent
   disconnected counts toward the kill switch (300 s).
2. **Reconcile.** The previous order's fills are journaled and alerted. An
   order that did not reach its target within a bar is cancelled and counts
   as a reject. The broker's position is the truth: a position the trader
   did not open is alerted and closed.
3. **Data.** Completed bars are merged into the cache (IB also returns the
   bar still forming; it is dropped). No completed bar for 15 minutes past
   its close means stale data, and stale data means flat.
4. **Account.** An equity read that is not positive, or that moves more than
   25% since the last read, is refused as a bad read (the fat-finger guard).
5. **Decide** with `engine.decide`, the same code the backtest runs.
6. **Drift.** A rolling live log-likelihood below the in-sample floor freezes
   entries until the next refit.
7. **Kill switch.** Any kill reason is written to the control folder, so the
   trader stays flat, even across restarts, until you reset it.
8. **Order.** Adding more than $25,000 of exposure waits for an approval
   window (`regime-trader approve`). Reducing risk never waits.
9. **Journal and checkpoint.**

Any exception: flatten if possible, journal, alert, and carry on at the next
bar. Alerts are best-effort: a failed alert is journaled and never stops
trading.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, replace
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from regime_trader.alerts import Alerts
from regime_trader.bars import TIMEZONE
from regime_trader.engine import (
    Decision,
    EngineConfig,
    EngineState,
    Fill,
    OpenPosition,
    open_position,
    playbook_for,
)
from regime_trader.engine import decide as engine_decide
from regime_trader.features import Z_FEATURES, compute_features
from regime_trader.hmm import filter_step
from regime_trader.ibkr import FillReport
from regime_trader.playbook import Playbook
from regime_trader.refit import DriftConfig, Fit, likelihood_alarm
from regime_trader.risk import AccountState, Approved, Vetoed, kill_reasons
from regime_trader.store import BarCache, Journal
from regime_trader.switching import INITIAL, SwitchState

BAR_CLOSE_HOURS = range(10, 17)  # IBKR RTH grid: 09:30-10:00, then hourly to 16:00
EXTERNAL = "EXTERNAL"  # the playbook of a position the trader did not open
MAX_BAR_LAG = pd.Timedelta(minutes=15)  # after a bar's close, before the data counts as stale
WAKE_DELAY = pd.Timedelta(seconds=5)  # after a bar's close, for IB to publish it
WATCHDOG_SLACK = pd.Timedelta(minutes=10)


class Broker(Protocol):
    @property
    def connected(self) -> bool: ...
    def recent(self, symbol: str, days: int) -> pd.DataFrame: ...
    def equity(self) -> float: ...
    def position(self, symbol: str) -> int: ...
    def move_to_target(self, symbol: str, current: int, target: int, reference_price: float) -> Any: ...
    def fill_report(self, handle: Any) -> FillReport: ...
    def cancel(self, handle: Any) -> None: ...


class BadAccountRead(RuntimeError):  # noqa: N818 -- reads as a sentence in the journal
    """The broker's equity figure is not plausible; nothing is sized from it."""


@dataclass(frozen=True)
class LiveConfig:
    symbol: str = "SPY"
    recent_days: int = 2  # sessions of bars requested each hour
    max_bar_lag: pd.Timedelta = MAX_BAR_LAG
    max_equity_jump: float = 0.25  # per bar, before an equity read is refused
    engine: EngineConfig = field(default_factory=EngineConfig)
    drift: DriftConfig = field(default_factory=DriftConfig)


# --- scheduling ------------------------------------------------------------------------------


def bar_end(starts: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """When each IBKR hourly RTH bar closes: 09:30 -> 10:00, 10:00 -> 11:00."""
    return starts.floor("h") + pd.Timedelta(hours=1)


def _closes_on(day: date) -> list[pd.Timestamp]:
    if day.weekday() >= 5:
        return []
    return [pd.Timestamp(f"{day} {hour:02d}:00").tz_localize(TIMEZONE) for hour in BAR_CLOSE_HOURS]


def next_bar_close(now: pd.Timestamp) -> pd.Timestamp:
    """The first scheduled bar close strictly after `now` (weekends skipped;
    on a holiday the trader wakes, finds no new bar, and stays flat)."""
    local = now.tz_convert(TIMEZONE)
    day = local.date()
    while True:
        for close in _closes_on(day):
            if close > local:
                return close
        day += timedelta(days=1)


class BarHandler(Protocol):
    def on_bar(self, now: pd.Timestamp) -> object: ...


def run(
    trader: BarHandler,
    clock: Callable[[], pd.Timestamp],
    sleep: Callable[[float], object],
    should_stop: Callable[[], bool],
    delay: pd.Timedelta = WAKE_DELAY,
) -> None:
    """Wake `delay` after each bar close and hand the trader the time."""
    while not should_stop():
        wake = next_bar_close(clock()) + delay
        sleep(max(0.0, (wake - clock()).total_seconds()))
        trader.on_bar(clock())


# --- control folder --------------------------------------------------------------------------


class Control:
    """The sticky kill switch and the approval window. Plain files, so the
    CLI (or you, by hand) can change them while the trader runs."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self._kill = directory / "KILL"
        self._approval = directory / "APPROVED_UNTIL"

    def killed(self) -> str | None:
        return self._kill.read_text(encoding="utf-8") if self._kill.exists() else None

    def kill(self, reason: str) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self._kill.write_text(reason, encoding="utf-8")

    def reset(self) -> None:
        self._kill.unlink(missing_ok=True)

    def approve(self, until: pd.Timestamp) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self._approval.write_text(until.isoformat(), encoding="utf-8")

    def approved(self, now: pd.Timestamp) -> bool:
        if not self._approval.exists():
            return False
        return bool(now < pd.Timestamp(self._approval.read_text(encoding="utf-8")))


# --- checkpoint ------------------------------------------------------------------------------


@dataclass(frozen=True)
class Checkpoint:
    """Everything the trader carries from one bar to the next, persisted so
    a restart resumes exactly where it stopped."""

    fit_id: str
    engine: EngineState
    filtered_through: pd.Timestamp  # the last bar folded into `engine.prior`
    last_run: pd.Timestamp
    session: str
    start_of_day_equity: float
    peak_equity: float
    last_equity: float  # NaN until the first accepted read
    consecutive_rejects: int
    disconnected_since: pd.Timestamp | None
    live_ll: tuple[float, ...]
    entries_frozen: bool


def fit_id(fit: Fit) -> str:
    return f"{fit.trained_through.isoformat()}|{','.join(fit.model.labels)}"


def _fresh(fit: Fit, now: pd.Timestamp, previous: Checkpoint | None) -> Checkpoint:
    """A checkpoint for a new fit. Account history and any open position carry
    over; the filter, the switching state and the drift window start again."""
    position = previous.engine.position if previous else None
    return Checkpoint(
        fit_id=fit_id(fit),
        engine=EngineState(prior=fit.prior, switch=INITIAL, position=position),
        filtered_through=fit.trained_through,
        last_run=now,
        session=previous.session if previous else "",
        start_of_day_equity=previous.start_of_day_equity if previous else math.nan,
        peak_equity=previous.peak_equity if previous else 0.0,
        last_equity=previous.last_equity if previous else math.nan,
        consecutive_rejects=previous.consecutive_rejects if previous else 0,
        disconnected_since=None,
        live_ll=(),
        entries_frozen=False,
    )


def _iso(ts: pd.Timestamp | None) -> str | None:
    return None if ts is None else ts.isoformat()


def _timestamp(text: str | None) -> pd.Timestamp | None:
    return None if text is None else pd.Timestamp(text)


def save_checkpoint(path: Path, checkpoint: Checkpoint) -> None:
    engine = checkpoint.engine
    payload = {
        "fit_id": checkpoint.fit_id,
        "prior": engine.prior.tolist(),
        "switch": asdict(engine.switch),
        "position": asdict(engine.position) if engine.position else None,
        "filtered_through": _iso(checkpoint.filtered_through),
        "last_run": _iso(checkpoint.last_run),
        "session": checkpoint.session,
        "start_of_day_equity": checkpoint.start_of_day_equity,
        "peak_equity": checkpoint.peak_equity,
        "last_equity": checkpoint.last_equity,
        "consecutive_rejects": checkpoint.consecutive_rejects,
        "disconnected_since": _iso(checkpoint.disconnected_since),
        "live_ll": list(checkpoint.live_ll),
        "entries_frozen": checkpoint.entries_frozen,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_checkpoint(path: Path) -> Checkpoint | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    position = data["position"]
    filtered_through = _timestamp(data["filtered_through"])
    last_run = _timestamp(data["last_run"])
    assert filtered_through is not None
    assert last_run is not None
    return Checkpoint(
        fit_id=data["fit_id"],
        engine=EngineState(
            prior=np.asarray(data["prior"], dtype=np.float64),
            switch=SwitchState(**data["switch"]),
            position=OpenPosition(**position) if position else None,
        ),
        filtered_through=filtered_through,
        last_run=last_run,
        session=data["session"],
        start_of_day_equity=float(data["start_of_day_equity"]),
        peak_equity=float(data["peak_equity"]),
        last_equity=float(data["last_equity"]),
        consecutive_rejects=int(data["consecutive_rejects"]),
        disconnected_since=_timestamp(data["disconnected_since"]),
        live_ll=tuple(float(v) for v in data["live_ll"]),
        entries_frozen=bool(data["entries_frozen"]),
    )


def missed_bar(state_path: Path, now: pd.Timestamp, slack: pd.Timedelta = WATCHDOG_SLACK) -> bool:
    """The watchdog's test: did the trader skip the latest scheduled bar close?
    No checkpoint at all counts as missed."""
    checkpoint = load_checkpoint(state_path)
    if checkpoint is None:
        return True
    due = now.tz_convert(TIMEZONE) - slack
    closes = [close for close in _closes_on(due.date()) if close <= due]
    return bool(closes) and checkpoint.last_run < closes[-1]


# --- the trader ------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Order:
    ts: pd.Timestamp
    decision: Decision | None  # None for an emergency flatten
    target: int
    handle: Any


class Trader:
    def __init__(
        self,
        *,
        broker: Broker,
        fit: Fit,
        playbooks: Mapping[str, Playbook],
        journal: Journal,
        cache: BarCache,
        control: Control,
        alerts: Alerts,
        state_path: Path,
        config: LiveConfig | None = None,
    ) -> None:
        self.broker = broker
        self.fit = fit
        self.playbooks = playbooks
        self.journal = journal
        self.cache = cache
        self.control = control
        self.alerts = alerts
        self.state_path = state_path
        self.config = config or LiveConfig()
        self._order: _Order | None = None
        saved = load_checkpoint(state_path)
        now = pd.Timestamp.now(tz=TIMEZONE)
        self._checkpoint = saved if saved and saved.fit_id == fit_id(fit) else _fresh(fit, now, saved)

    @property
    def engine_state(self) -> EngineState:
        return self._checkpoint.engine

    def on_bar(self, now: pd.Timestamp) -> Decision | None:
        try:
            return self._on_bar(now)
        except Exception as error:  # the last line of defence: flat, journaled, alerted
            self._fail(now, error)
            return None
        finally:
            self._checkpoint = replace(self._checkpoint, last_run=now)
            save_checkpoint(self.state_path, self._checkpoint)

    # -- the steps ----------------------------------------------------------------------------

    def _on_bar(self, now: pd.Timestamp) -> Decision | None:
        if not self.broker.connected:
            self._disconnected(now)
            return None
        self._checkpoint = replace(self._checkpoint, disconnected_since=None)
        self._reconcile(now)

        recent = self.broker.recent(self.config.symbol, self.config.recent_days)
        complete = recent[bar_end(pd.DatetimeIndex(recent.index)) <= now]
        bars = self.cache.save(self.config.symbol, complete)
        latest = pd.Timestamp(bars.index[-1])
        if now - bar_end(pd.DatetimeIndex([latest]))[0] > self.config.max_bar_lag:
            self._stale(now, latest, float(bars["close"].iloc[-1]))
            return None
        if latest <= self._checkpoint.filtered_through:
            return None  # this bar was already decided
        return self._decide(now, bars, latest)

    def _decide(self, now: pd.Timestamp, bars: pd.DataFrame, latest: pd.Timestamp) -> Decision:
        cp = self._checkpoint
        equity = self._read_equity()
        session = latest.date().isoformat()
        start_of_day = equity if session != cp.session else cp.start_of_day_equity
        peak = max(cp.peak_equity, equity)
        features = compute_features(bars)
        engine = self._catch_up(features, latest)
        shares = self.broker.position(self.config.symbol)
        account = AccountState(
            equity=equity,
            start_of_day_equity=start_of_day,
            peak_equity=peak,
            position_shares=shares,
            consecutive_rejects=cp.consecutive_rejects,
            disconnected_seconds=0.0,
            killed=self.control.killed() is not None,
        )
        price = float(bars["close"].iloc[-1])
        row = {str(k): float(v) for k, v in features.iloc[-1].items()}
        engine_config = replace(
            self.config.engine,
            entries_frozen=cp.entries_frozen,
            calibrated=self.config.engine.calibrated and self.fit.calibrated,
        )
        engine, decision = engine_decide(
            engine,
            latest,
            row,
            price,
            account,
            self.fit.model,
            self.playbooks,
            self.fit.kelly,
            engine_config,
            healthy=True,
        )
        self._checkpoint = replace(
            cp,
            engine=engine,
            filtered_through=latest,
            session=session,
            start_of_day_equity=start_of_day,
            peak_equity=peak,
            last_equity=equity,
        )
        self._watch_drift(latest, decision)
        self._watch_kill(latest, account)
        if decision.regime.switched:
            self._notify(latest, "switch", f"{cp.engine.switch.active} -> {decision.regime.active}")
        status = self._execute(decision, shares, price, now)
        self.journal.record_decision(
            ts=latest,
            labels=self.fit.model.labels,
            probabilities=decision.probabilities,
            next_state=decision.next_state,
            active=decision.regime.active,
            target_shares=decision.target_shares,
            order=status,
            reasons=decision.reasons,
        )
        return decision

    def _execute(self, decision: Decision, shares: int, price: float, now: pd.Timestamp) -> str:
        order = decision.order
        if isinstance(order, Vetoed):
            return "vetoed"
        if order is None or decision.target_shares == shares:
            return "none"
        assert isinstance(order, Approved)
        if order.needs_manual_approval and not self.control.approved(now):
            delta = decision.target_shares - shares
            self._notify(
                decision.ts,
                "approval",
                f"BUY {delta} {self.config.symbol} (${delta * price:,.0f}) needs approval: "
                "run `regime-trader approve` to allow it",
            )
            return "awaiting approval"
        self._send(decision.ts, decision, decision.target_shares, shares, price)
        return "sent"

    def _send(
        self, ts: pd.Timestamp, decision: Decision | None, target: int, shares: int, price: float
    ) -> None:
        handle = self.broker.move_to_target(self.config.symbol, shares, target, price)
        self._order = _Order(ts, decision, target, handle)

    # -- reconciliation -----------------------------------------------------------------------

    def _reconcile(self, now: pd.Timestamp) -> None:
        cp = self._checkpoint
        shares = self.broker.position(self.config.symbol)
        held = cp.engine.position.shares if cp.engine.position else 0
        rejects = cp.consecutive_rejects
        explained = 0
        price = math.nan
        decision = None
        if self._order is not None:
            order, self._order = self._order, None
            decision = order.decision
            report = self.broker.fill_report(order.handle)
            if report.filled:
                explained, price = report.filled, report.average_price
                fill = Fill(order.ts, now, report.filled, report.average_price, report.commission)
                self.journal.record_fill(fill)
                self._notify(now, "fill", f"{report.filled:+d} {self.config.symbol} @ {price:.2f}")
            if not report.done:
                self.broker.cancel(order.handle)
            if shares == order.target:
                rejects = 0
            else:
                rejects += 1
                self._notify(
                    now, "error", f"order to {order.target} shares ended at {shares}: reject {rejects}"
                )
        if shares != held + explained:
            self._notify(
                now, "error", f"position mismatch: broker holds {shares}, trader expected {held + explained}"
            )
        position = self._position_after(shares, price, decision)
        self._checkpoint = replace(
            cp, engine=replace(cp.engine, position=position), consecutive_rejects=rejects
        )

    def _position_after(self, shares: int, price: float, decision: Decision | None) -> OpenPosition | None:
        current = self._checkpoint.engine.position
        if shares == 0:
            return None
        if current is not None:
            return replace(current, shares=shares)
        playbook = playbook_for(self.playbooks, decision.regime.active) if decision else None
        if decision is not None and playbook is not None and math.isfinite(price):
            return open_position(playbook, shares, price, decision.entry_rv)
        return OpenPosition(EXTERNAL, shares, math.nan, 0.0, math.inf, 0)  # closed by the engine

    def _catch_up(self, features: pd.DataFrame, latest: pd.Timestamp) -> EngineState:
        """Fold any bars missed while down (or skipped as unhealthy) into the prior."""
        engine = self._checkpoint.engine
        index = pd.DatetimeIndex(features.index)
        missed = features[(index > self._checkpoint.filtered_through) & (index < latest)]
        prior = engine.prior
        for x in missed[list(Z_FEATURES)].to_numpy():
            if np.isfinite(x).all():
                prior = filter_step(self.fit.model.hmm, prior, x).next_state
        return replace(engine, prior=prior)

    # -- guards -------------------------------------------------------------------------------

    def _read_equity(self) -> float:
        equity = float(self.broker.equity())
        last = self._checkpoint.last_equity
        if not math.isfinite(equity) or equity <= 0:
            raise BadAccountRead(f"equity read {equity!r} refused")
        if math.isfinite(last) and abs(equity / last - 1) > self.config.max_equity_jump:
            raise BadAccountRead(f"equity read {equity:,.0f} refused: last accepted read was {last:,.0f}")
        return equity

    def _watch_drift(self, ts: pd.Timestamp, decision: Decision) -> None:
        cp = self._checkpoint
        if not math.isfinite(decision.log_likelihood):
            return
        window = self.config.drift.ll_window
        live_ll = (*cp.live_ll, decision.log_likelihood)[-window:]
        mean, floor = likelihood_alarm(np.array(live_ll), self.fit.insample_ll, self.config.drift)
        frozen = cp.entries_frozen
        if not frozen and math.isfinite(mean) and mean < floor:
            frozen = True
            self._notify(
                ts,
                "drift",
                f"live log-likelihood {mean:.2f} below the in-sample floor {floor:.2f}: "
                "entries frozen until the next refit",
            )
        self._checkpoint = replace(cp, live_ll=live_ll, entries_frozen=frozen)

    def _watch_kill(self, ts: pd.Timestamp, account: AccountState) -> None:
        reasons = kill_reasons(self.config.engine.limits, account)
        if reasons and self.control.killed() is None:
            self.control.kill("; ".join(reasons))
            self._notify(
                ts,
                "kill",
                "kill switch: "
                + "; ".join(reasons)
                + ". Flattening; run `regime-trader kill --reset` to resume",
            )

    def _disconnected(self, now: pd.Timestamp) -> None:
        cp = self._checkpoint
        since = cp.disconnected_since or now
        seconds = (now - since).total_seconds()
        if cp.disconnected_since is None:
            self._notify(now, "error", "broker disconnected")
        else:
            self.journal.record_event(now, "disconnected", f"{seconds:.0f}s")
        self._checkpoint = replace(cp, disconnected_since=since)
        if seconds > self.config.engine.limits.max_disconnect_seconds and self.control.killed() is None:
            self.control.kill(f"broker disconnected {seconds:.0f}s")
            self._notify(now, "kill", f"kill switch: broker disconnected {seconds:.0f}s")

    def _stale(self, now: pd.Timestamp, latest: pd.Timestamp, price: float) -> None:
        shares = self.broker.position(self.config.symbol)
        message = f"stale data: the last completed bar started {latest}"
        if shares:
            self._send(now, None, 0, shares, price)
            self._notify(now, "error", f"{message}; flattening {shares} shares")
        else:
            self.journal.record_event(now, "stale", message)

    def _fail(self, now: pd.Timestamp, error: Exception) -> None:
        self._notify(now, "error", f"{type(error).__name__}: {error}; flattening")
        try:
            if self.broker.connected:
                shares = self.broker.position(self.config.symbol)
                if shares:
                    price = float(self.cache.load(self.config.symbol)["close"].iloc[-1])
                    self._send(now, None, 0, shares, price)
        except Exception as flatten_error:
            self.journal.record_event(now, "error", f"flatten failed: {flatten_error}")

    def _notify(self, ts: pd.Timestamp, kind: str, text: str) -> None:
        """Journal the event, then alert. A failed alert is journaled too."""
        self.journal.record_event(ts, kind, text)
        try:
            self.alerts.send(kind, text)
        except Exception as error:
            self.journal.record_event(ts, "error", f"alert failed ({kind}): {error}")
