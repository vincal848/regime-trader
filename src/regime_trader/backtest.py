"""Walk-forward backtest with refits, costs and baselines (spec §11, §12).

The loop is the live loop with a simulated broker:

- **Refits.** The first fit, at `test_start`, chooses K (`select_states`, with
  the last `validation_days` sessions held out). Each refit then refits that
  K on all bars strictly before its timestamp, every `refit_days`, on an
  expanding window. Labels are matched to the previous fit, and a drift
  alarm freezes entries until the next refit.
- **Decisions** come from `engine.decide` at each bar's close.
- **Fills** happen at the *next* bar's open, paying slippage, half the
  spread and commission. A decision can never trade on the bar that
  produced it.
- **Kelly per state** is estimated at each refit from training data only:
  the next-bar returns of bars where that state led and its playbook's
  entry condition held.
- **Holdout.** With `holdout_start` set, bars from that time on are dropped
  unless `include_holdout=True` is passed explicitly, for a candidate you
  have approved.

The baselines are buy-and-hold, and `run_static` (one playbook traded in
every regime at its full size, with the same fills and costs).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace

import numpy as np
import numpy.typing as npt
import pandas as pd

from regime_trader.bars import TIMEZONE
from regime_trader.engine import (
    Decision,
    EngineConfig,
    EngineState,
    Fill,
    OpenPosition,
    decide,
    on_fill,
    open_position,
    price_exit,
)
from regime_trader.features import compute_features, healthy
from regime_trader.metrics import (
    GateResult,
    Gates,
    daily_returns,
    evaluate_gates,
    hit_rate,
    max_drawdown,
    sharpe,
    t_statistic,
)
from regime_trader.playbook import Playbook, evaluate_signal
from regime_trader.refit import (
    DriftConfig,
    DriftReport,
    Fit,
    FitConfig,
    adopt_fit,
    fit_regime,
    rolling_alarm,
)
from regime_trader.risk import AccountState, Approved, check_order, kill_reasons, state_cap

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Costs:
    commission_per_share: float = 0.0035  # IBKR tiered
    min_commission: float = 0.35
    slippage_bps: float = 1.0
    half_spread: float = 0.005  # half of SPY's one-cent spread


def fill_price(open_price: float, buying: bool, costs: Costs) -> float:
    slip = costs.slippage_bps / 10_000
    if buying:
        return open_price * (1 + slip) + costs.half_spread
    return open_price * (1 - slip) - costs.half_spread


def commission(shares: int, costs: Costs) -> float:
    return 0.0 if shares == 0 else max(costs.min_commission, costs.commission_per_share * abs(shares))


@dataclass(frozen=True)
class BacktestConfig:
    test_start: pd.Timestamp
    refit_days: int = 30
    candidates: tuple[int, ...] = (2, 3, 4, 5)
    restarts: int = 10
    seed: int = 20260107
    validation_days: int = 126  # sessions held out of the first fit to choose K
    costs: Costs = field(default_factory=Costs)
    initial_equity: float = 100_000.0
    engine: EngineConfig = field(default_factory=EngineConfig)
    drift: DriftConfig = field(default_factory=DriftConfig)
    holdout_start: pd.Timestamp | None = None  # None: no locked holdout

    @property
    def fit(self) -> FitConfig:
        return FitConfig(self.candidates, self.restarts, self.seed, self.validation_days)


@dataclass(frozen=True)
class Trade:
    state: str
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    shares: int  # largest size held
    entry_price: float
    exit_price: float  # share-weighted average
    pnl: float  # after commissions and slippage


@dataclass(frozen=True)
class RefitRecord:
    ts: pd.Timestamp
    n_states: int
    labels: tuple[str, ...]
    transmat: FloatArray
    drift: DriftReport | None  # None for the first fit
    calibrated: bool


@dataclass(frozen=True)
class BacktestResult:
    equity: pd.Series  # marked at each bar's close
    positions: pd.Series  # shares held during each bar
    active: pd.Series  # active regime label at each decision
    fills: tuple[Fill, ...]
    trades: tuple[Trade, ...]
    refits: tuple[RefitRecord, ...]
    per_state: pd.DataFrame
    manual_approvals: int  # orders that would have waited for you live


class _Portfolio:
    """Cash, shares and round-trip trades for a long/flat book."""

    def __init__(self, cash: float, costs: Costs) -> None:
        self.cash = cash
        self.shares = 0
        self.costs = costs
        self.trades: list[Trade] = []
        self._open: dict[str, float] = {}
        self._state = ""
        self._entry_ts = pd.Timestamp(0)

    def mark(self, price: float) -> float:
        return self.cash + self.shares * price

    def execute(
        self, target: int, open_price: float, decision_ts: pd.Timestamp, fill_ts: pd.Timestamp, state: str
    ) -> Fill:
        price = fill_price(open_price, buying=target > self.shares, costs=self.costs)
        if target > self.shares:  # no leverage: a buy is capped at what the cash can pay for, fees included
            affordable = math.floor(
                (self.cash - self.costs.min_commission) / (price + self.costs.commission_per_share)
            )
            target = self.shares + max(min(target - self.shares, affordable), 0)
        delta = target - self.shares
        fee = commission(delta, self.costs)
        self.cash -= delta * price + fee
        if self.shares == 0:
            self._open = {"cost": 0.0, "proceeds": 0.0, "bought": 0.0, "sold": 0.0, "max": 0.0}
            self._state, self._entry_ts = state, fill_ts
        book = self._open
        if delta > 0:
            book["cost"] += delta * price + fee
            book["bought"] += delta
        else:
            book["proceeds"] += -delta * price - fee
            book["sold"] += -delta
        self.shares = target
        book["max"] = max(book["max"], float(target))
        if target == 0:
            self.trades.append(
                Trade(
                    state=self._state,
                    entry_ts=self._entry_ts,
                    exit_ts=fill_ts,
                    shares=int(book["max"]),
                    entry_price=book["cost"] / book["bought"],
                    exit_price=book["proceeds"] / book["sold"],
                    pnl=book["proceeds"] - book["cost"],
                )
            )
        return Fill(decision_ts, fill_ts, delta, price, fee)


def _rows(features: pd.DataFrame) -> list[dict[str, float]]:
    return [{str(k): float(v) for k, v in row.items()} for row in features.to_dict("records")]


def _trim(bars: pd.DataFrame, config: BacktestConfig, include_holdout: bool) -> pd.DataFrame:
    if config.holdout_start is not None and not include_holdout:
        return bars[bars.index < config.holdout_start]
    return bars


def _per_state(equity: pd.Series, active: pd.Series) -> pd.DataFrame:
    returns = np.log(equity).diff().iloc[1:]
    labels = active.shift(1).iloc[1:]
    grouped = returns.groupby(labels.to_numpy())
    table = pd.DataFrame({"bars": grouped.size(), "log_return": grouped.sum(), "mean": grouped.mean()})
    table["share"] = table["bars"] / table["bars"].sum()
    return table


def run_backtest(
    bars: pd.DataFrame,
    playbooks: Mapping[str, Playbook],
    config: BacktestConfig,
    include_holdout: bool = False,
) -> BacktestResult:
    bars = _trim(bars, config, include_holdout)
    features = compute_features(bars)
    healthy_rows = healthy(features)
    rows = _rows(features)
    index = pd.DatetimeIndex(bars.index)
    opens, closes = bars["open"].to_numpy(), bars["close"].to_numpy()
    next_returns = features["ret"].shift(-1).to_numpy()
    sessions = index.tz_convert(TIMEZONE).normalize()

    portfolio = _Portfolio(config.initial_equity, config.costs)
    current: Fit | None = None
    state: EngineState | None = None
    frozen, killed = False, False
    next_refit = index[0]
    refits: list[RefitRecord] = []
    live_ll: list[float] = []
    fills: list[Fill] = []
    pending: tuple[Decision, int] | None = None
    equity: list[float] = []
    positions: list[int] = []
    active: list[str] = []
    peak = start_of_day = config.initial_equity
    manual = 0

    first = int(index.searchsorted(config.test_start))
    for i in range(first, len(index)):
        ts = index[i]
        if pending is not None and state is not None:
            decision, target = pending
            fill = portfolio.execute(target, opens[i], decision.ts, ts, decision.regime.active or "NONE")
            fills.append(fill)
            state = on_fill(state, decision, playbooks, portfolio.shares, fill.price)
            pending = None

        if ts >= next_refit:
            past = (index < ts) & healthy_rows
            new = fit_regime(features[past], next_returns[past], playbooks, config.fit, current, config.drift)
            state = adopt_fit(state, new)
            current, live_ll, frozen = new, [], new.drifted
            hmm = new.model.hmm
            refits.append(
                RefitRecord(ts, hmm.n_states, new.model.labels, hmm.transmat, new.drift, new.calibrated)
            )
            next_refit = ts + pd.Timedelta(days=config.refit_days)

        assert current is not None  # the first bar always refits
        assert state is not None
        if i > first and sessions[i] != sessions[i - 1]:
            start_of_day = equity[-1]
        mark = portfolio.mark(closes[i])
        peak = max(peak, mark)
        account = AccountState(mark, start_of_day, peak, portfolio.shares, 0, 0.0, killed)
        calibrated = config.engine.calibrated and current.calibrated
        engine_config = replace(config.engine, entries_frozen=frozen, calibrated=calibrated)
        state, decision = decide(
            state,
            ts,
            rows[i],
            float(closes[i]),
            account,
            current.model,
            playbooks,
            current.kelly,
            engine_config,
            bool(healthy_rows[i]),
        )
        killed = killed or bool(kill_reasons(config.engine.limits, account))
        if math.isfinite(decision.log_likelihood):
            live_ll.append(decision.log_likelihood)
            frozen = frozen or rolling_alarm(live_ll, current.ll_floor, config.drift.ll_window)
        if decision.target_shares != portfolio.shares and isinstance(decision.order, Approved):
            manual += int(decision.order.needs_manual_approval)
            pending = (decision, decision.target_shares)
        equity.append(mark)
        positions.append(portfolio.shares)
        active.append(decision.regime.active or "NONE")

    test_index = index[first:]
    equity_series = pd.Series(equity, index=test_index, name="equity")
    active_series = pd.Series(active, index=test_index, name="active")
    return BacktestResult(
        equity=equity_series,
        positions=pd.Series(positions, index=test_index, name="shares"),
        active=active_series,
        fills=tuple(fills),
        trades=tuple(portfolio.trades),
        refits=tuple(refits),
        per_state=_per_state(equity_series, active_series),
        manual_approvals=manual,
    )


def run_buy_and_hold(bars: pd.DataFrame, config: BacktestConfig, include_holdout: bool = False) -> pd.Series:
    """Buy at the first test bar's open with every dollar, hold to the end."""
    bars = _trim(bars, config, include_holdout)
    test = bars[bars.index >= config.test_start]
    price = fill_price(float(test["open"].iloc[0]), buying=True, costs=config.costs)
    shares = math.floor(config.initial_equity / price)
    cash = config.initial_equity - shares * price - commission(shares, config.costs)
    equity: pd.Series = cash + shares * test["close"]
    return equity.rename("equity")


def run_static(
    bars: pd.DataFrame, playbook: Playbook, config: BacktestConfig, include_holdout: bool = False
) -> tuple[pd.Series, tuple[Trade, ...]]:
    """One playbook in every regime, at its full size within its state cap,
    with the same fills, costs and hard limits as the system."""
    bars = _trim(bars, config, include_holdout)
    features = compute_features(bars)
    healthy_rows = healthy(features)
    rows = _rows(features)
    index = pd.DatetimeIndex(bars.index)
    opens, closes = bars["open"].to_numpy(), bars["close"].to_numpy()
    portfolio = _Portfolio(config.initial_equity, config.costs)
    limits = config.engine.limits
    fraction = min(playbook.max_size, state_cap(limits, playbook.state))
    position: OpenPosition | None = None
    pending: tuple[pd.Timestamp, int, float] | None = None
    equity: list[float] = []
    first = int(index.searchsorted(config.test_start))
    for i in range(first, len(index)):
        ts = index[i]
        if pending is not None:
            decision_ts, target, rv = pending
            fill = portfolio.execute(target, opens[i], decision_ts, ts, playbook.state)
            position = open_position(playbook, target, fill.price, rv) if target else None
            pending = None
        elif position is not None:
            position = replace(position, bars_held=position.bars_held + 1)
        mark = portfolio.mark(closes[i])
        equity.append(mark)
        if not healthy_rows[i]:
            target = 0
        elif position is not None:
            exit_now = price_exit(position, closes[i]) is not None
            target = (
                0
                if exit_now or evaluate_signal(playbook, rows[i], True, position.bars_held).exit
                else position.shares
            )
        else:
            entering = evaluate_signal(playbook, rows[i], False, 0).enter
            target = math.floor(fraction * mark / closes[i]) if entering else 0
        if target != portfolio.shares:
            account = AccountState(mark, mark, mark, portfolio.shares, 0, 0.0, False)
            if isinstance(check_order(limits, account, playbook.state, target, float(closes[i])), Approved):
                pending = (ts, target, rows[i]["rv"])
    return pd.Series(equity, index=index[first:], name="equity"), tuple(portfolio.trades)


def summarize(equity: pd.Series, trades: tuple[Trade, ...]) -> dict[str, float]:
    daily = daily_returns(equity)
    return {
        "sharpe": sharpe(daily),
        "max_drawdown": max_drawdown(equity),
        "hit_rate": hit_rate([t.pnl for t in trades]),
        "t_statistic": t_statistic(daily),
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1),
        "trades": float(len(trades)),
    }


@dataclass(frozen=True)
class Acceptance:
    result: BacktestResult
    stats: dict[str, float]
    baselines: dict[str, float]  # name -> out-of-sample Sharpe
    gates: GateResult


def baseline_sharpes(
    bars: pd.DataFrame,
    playbooks: Mapping[str, Playbook],
    config: BacktestConfig,
    include_holdout: bool = False,
) -> dict[str, float]:
    """Buy-and-hold, and the single static playbook that did best *before*
    the test period (spec §11), both scored out of sample."""
    baselines = {"buy-and-hold": sharpe(daily_returns(run_buy_and_hold(bars, config, include_holdout)))}
    tradable = [p for p in playbooks.values() if p.max_size > 0]
    if tradable:
        training = bars[bars.index < config.test_start]
        on_training = replace(config, test_start=training.index[0])

        def training_sharpe(playbook: Playbook) -> float:
            score = sharpe(daily_returns(run_static(training, playbook, on_training)[0]))
            return score if math.isfinite(score) else -math.inf

        best = max(tradable, key=training_sharpe)
        static, _ = run_static(bars, best, config, include_holdout)
        baselines[f"static {best.state}"] = sharpe(daily_returns(static))
    return baselines


def acceptance(
    bars: pd.DataFrame,
    playbooks: Mapping[str, Playbook],
    config: BacktestConfig,
    include_holdout: bool = False,
    baselines: dict[str, float] | None = None,
    gates: Gates | None = None,
) -> Acceptance:
    """The walk-forward run, scored against the acceptance gates (spec §11)."""
    result = run_backtest(bars, playbooks, config, include_holdout)
    stats = summarize(result.equity, result.trades)
    if baselines is None:
        baselines = baseline_sharpes(bars, playbooks, config, include_holdout)
    return Acceptance(result, stats, baselines, evaluate_gates(stats, baselines, gates or Gates()))


def acceptance_report(acceptance: Acceptance) -> str:
    stats, gates = acceptance.stats, acceptance.gates
    lines = [
        f"Sharpe {stats['sharpe']:.2f} | max drawdown {stats['max_drawdown']:.1%} | "
        f"hit rate {stats['hit_rate']:.1%} | t-statistic {stats['t_statistic']:.2f} | "
        f"total return {stats['total_return']:.1%} | trades {stats['trades']:.0f}",
        "Baselines (Sharpe): " + ", ".join(f"{name} {v:.2f}" for name, v in acceptance.baselines.items()),
        "Gates: " + ("PASSED" if gates.passed else "FAILED"),
    ]
    lines += [f"  [{'x' if ok else ' '}] {name}" for name, ok in gates.checks.items()]
    return "\n".join(lines)
