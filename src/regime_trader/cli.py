"""`regime-trader`: the command line.

    fetch      cache hourly bars (IBKR, or Yahoo for the demo)
    fit        fit the regime model on the cached bars; a refit reports drift
    backtest   walk-forward backtest scored against the acceptance gates
    live       run the paper trader (IB Gateway must be running)
    kill       engage (or --reset) the sticky kill switch
    approve    open a window in which orders over $25,000 may be sent
    nightly    daily report, then the Claude review and proposal backtests
    report     the daily report only
    watchdog   alert if the trader missed a bar (schedule every 15 minutes)
    dashboard  open the local dashboard

Everything lives under `--root` (default: the current folder). Secrets come
from the environment or `ROOT/.env`, and are never printed.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from regime_trader.alerts import Alerts, NullAlerts, TelegramAlerts
from regime_trader.backtest import BacktestConfig, acceptance, acceptance_report
from regime_trader.bars import TIMEZONE
from regime_trader.features import compute_features, healthy
from regime_trader.ibkr import IbkrBroker, PaperOnlyError, settings_from_env
from regime_trader.live import Control, LiveConfig, Trader, missed_bar, run
from regime_trader.llm import AnthropicReviewClient, NightlyReviewer, SpendLedger
from regime_trader.nightly import daily_report, run_nightly
from regime_trader.refit import DriftConfig, FitConfig, drift_report, fit_regime
from regime_trader.store import BarCache, Journal, load_fit, load_playbooks, save_fit

SYMBOL = "SPY"
DEFAULT_TEST_START = "2021-01-04"
HOLDOUT_DAYS = 365


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def fit(self) -> Path:
        return self.models / "fit.json"

    @property
    def journal(self) -> Path:
        return self.root / "journal.db"

    @property
    def playbooks(self) -> Path:
        return self.root / "playbooks"

    @property
    def proposals(self) -> Path:
        return self.root / "proposals"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    @property
    def control(self) -> Path:
        return self.root / "control"

    @property
    def state(self) -> Path:
        return self.root / "live_state.json"

    @property
    def spend(self) -> Path:
        return self.root / "llm_spend.json"


def read_env(path: Path) -> dict[str, str]:
    """KEY=VALUE lines; blank lines and # comments ignored; quotes stripped."""
    if not path.exists():
        return {}
    env = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip().strip("\"'")
    return env


def _alerts(env: Mapping[str, str]) -> Alerts:
    token, chat = env.get("TELEGRAM_BOT_TOKEN"), env.get("TELEGRAM_CHAT_ID")
    return TelegramAlerts(token, chat) if token and chat else NullAlerts()


def _now() -> pd.Timestamp:
    return pd.Timestamp.now(tz=TIMEZONE)


def _training_set(bars: pd.DataFrame) -> tuple[pd.DataFrame, npt.NDArray[np.float64]]:
    features = compute_features(bars)
    rows = healthy(features)
    next_returns = features["ret"].shift(-1).to_numpy()
    return features[rows], next_returns[rows]


def _fit_config(args: argparse.Namespace) -> FitConfig:
    return FitConfig(tuple(args.candidates), args.restarts, FitConfig.seed, args.validation_days)


# --- commands ---------------------------------------------------------------------------------


def _fetch(args: argparse.Namespace, paths: Paths, env: Mapping[str, str]) -> int:
    if args.source == "yahoo":
        from regime_trader.yahoo import download_hourly

        bars = download_hourly(SYMBOL)
    else:
        from ib_async import IB

        broker = IbkrBroker(IB(), settings_from_env(env))
        broker.connect()
        bars = broker.history(SYMBOL, args.years)
    cached = BarCache(paths.data).save(SYMBOL, bars)
    print(f"cached {len(cached)} bars, {cached.index[0]} to {cached.index[-1]}")
    return 0


def _fit(args: argparse.Namespace, paths: Paths, env: Mapping[str, str]) -> int:
    bars = BarCache(paths.data).load(SYMBOL)
    features, next_returns = _training_set(bars)
    previous = load_fit(paths.fit) if paths.fit.exists() else None
    playbooks = load_playbooks(paths.playbooks)
    fit = fit_regime(features, next_returns, playbooks, _fit_config(args), previous)
    print(f"fit {fit.model.hmm.n_states} states through {fit.trained_through}: {', '.join(fit.model.labels)}")
    print("kelly: " + ", ".join(f"{k} {v:.2f}" for k, v in fit.kelly.items()))
    sizing = "probability-weighted" if fit.calibrated else "a fixed quarter of the cap (not yet calibrated)"
    print(f"sizing: {sizing}")
    if previous is not None:
        drift = drift_report(previous.model, fit.model, np.empty(0), previous.insample_ll, DriftConfig())
        verdict = "ALARM: " + "; ".join(drift.reasons) if drift.drifted else "none"
        print(
            f"drift vs the previous fit: {verdict} "
            f"(transition shift {drift.transition_shift:.3f}, mean shift {drift.mean_shift_sd:.2f} sd)"
        )
        if drift.drifted:
            message = f"refit drift: {verdict}. The new fit is saved; review it before trading it."
            _alerts(env).send("drift", message)
        archive = paths.models / f"fit-{previous.trained_through:%Y%m%dT%H%M}.json"
        paths.fit.replace(archive)
    save_fit(paths.fit, fit)
    return 0


class UsageError(ValueError):
    """The command cannot run as asked; the message says what to change."""


def _backtest_config(args: argparse.Namespace, bars: pd.DataFrame) -> BacktestConfig:
    if bars.empty:
        raise UsageError("no cached bars: run `regime-trader fetch` first")
    test_start = pd.Timestamp(args.test_start).tz_localize(TIMEZONE)
    holdout = pd.Timestamp(bars.index[-1]) - pd.Timedelta(days=HOLDOUT_DAYS)
    if not args.no_holdout and holdout <= test_start:
        raise UsageError(
            f"the 12-month locked holdout starts {holdout.date()}, before the test start "
            f"{test_start.date()}: fetch more history, or pass --no-holdout (demo data only)"
        )
    return BacktestConfig(
        test_start=test_start,
        refit_days=args.refit_days,
        candidates=tuple(args.candidates),
        restarts=args.restarts,
        validation_days=args.validation_days,
        holdout_start=None if args.no_holdout else holdout,
    )


def _backtest(args: argparse.Namespace, paths: Paths, env: Mapping[str, str]) -> int:
    bars = BarCache(paths.data).load(SYMBOL)
    config = _backtest_config(args, bars)
    result = acceptance(bars, load_playbooks(paths.playbooks), config, include_holdout=args.include_holdout)
    if args.no_holdout:
        scope = "no locked holdout: demo only"
    else:
        scope = "INCLUDING the locked holdout" if args.include_holdout else "locked holdout excluded"
    report = f"Walk-forward from {config.test_start.date()} ({scope})\n{acceptance_report(result)}\n"
    folder = paths.runs / datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "report.md").write_text(report, encoding="utf-8")
    result.result.equity.to_csv(folder / "equity.csv")
    print(report)
    return 0 if result.gates.passed else 1


class _Session:
    """Reconnects to IB Gateway before each bar when the connection dropped."""

    def __init__(self, broker: IbkrBroker, trader: Trader) -> None:
        self.broker = broker
        self.trader = trader

    def on_bar(self, now: pd.Timestamp) -> object:
        if not self.broker.connected:
            with contextlib.suppress(OSError):  # still down: the time counts toward the kill switch
                self.broker.connect()
        return self.trader.on_bar(now)


def _live(args: argparse.Namespace, paths: Paths, env: Mapping[str, str]) -> int:
    settings = settings_from_env(env)
    from ib_async import IB

    ib = IB()
    broker = IbkrBroker(ib, settings)
    broker.connect()
    trader = Trader(
        broker=broker,
        fit=load_fit(paths.fit),
        playbooks=load_playbooks(paths.playbooks),
        journal=Journal(paths.journal),
        cache=BarCache(paths.data),
        control=Control(paths.control),
        alerts=_alerts(env),
        state_path=paths.state,
        config=LiveConfig(),
    )
    print(f"trading {SYMBOL} on paper account {settings!r}; Ctrl+C stops (positions stay open)")
    run(_Session(broker, trader), clock=_now, sleep=ib.sleep, should_stop=lambda: False)
    return 0


def _kill(args: argparse.Namespace, paths: Paths, env: Mapping[str, str]) -> int:
    control = Control(paths.control)
    if args.reset:
        control.reset()
        print("kill switch reset: the trader resumes at the next bar")
    else:
        control.kill(args.reason)
        print("kill switch engaged: the trader flattens at the next bar and stays flat until reset.")
        print("To flatten immediately, close the position in TWS or the IBKR app.")
    return 0


def _approve(args: argparse.Namespace, paths: Paths, env: Mapping[str, str]) -> int:
    until = _now() + pd.Timedelta(minutes=args.minutes)
    Control(paths.control).approve(until)
    print(f"orders over the approval threshold may be sent until {until:%Y-%m-%d %H:%M %Z}")
    return 0


def _day(args: argparse.Namespace) -> date:
    return date.fromisoformat(args.day) if args.day else _now().date()


def _report(args: argparse.Namespace, paths: Paths, env: Mapping[str, str]) -> int:
    day = _day(args)
    report = daily_report(Journal(paths.journal), BarCache(paths.data).load(SYMBOL), load_fit(paths.fit), day)
    path = paths.reports / f"{day}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    print(report)
    return 0


def _nightly(args: argparse.Namespace, paths: Paths, env: Mapping[str, str]) -> int:
    bars = BarCache(paths.data).load(SYMBOL)
    reviewer = None
    if env.get("ANTHROPIC_API_KEY") and not args.no_review:
        import anthropic

        client = AnthropicReviewClient(anthropic.Anthropic(api_key=env["ANTHROPIC_API_KEY"]))
        budget = float(env.get("LLM_MONTHLY_BUDGET_USD", "20"))
        reviewer = NightlyReviewer(client, SpendLedger(paths.spend), budget)
    result = run_nightly(
        day=_day(args),
        bars=bars,
        fit=load_fit(paths.fit),
        playbooks=load_playbooks(paths.playbooks),
        journal=Journal(paths.journal),
        alerts=_alerts(env),
        reports_dir=paths.reports,
        proposals_dir=paths.proposals,
        backtest=_backtest_config(args, bars),
        reviewer=reviewer,
        now=datetime.now(UTC),
    )
    print(f"report: {result.report_path}")
    print(f"review: {result.review_path or 'skipped'}")
    for evaluation in result.evaluations:
        print(f"  {evaluation.state}: {evaluation.error or ('passed' if evaluation.passed else 'failed')}")
    return 0


def _watchdog(args: argparse.Namespace, paths: Paths, env: Mapping[str, str]) -> int:
    if missed_bar(paths.state, _now()):
        _alerts(env).send("error", "watchdog: the trader missed a bar. Is it running, and is IB Gateway up?")
        print("missed a bar")
        return 1
    return 0


def _dashboard(args: argparse.Namespace, paths: Paths, env: Mapping[str, str]) -> int:
    from regime_trader import dashboard

    command = [sys.executable, "-m", "streamlit", "run", dashboard.__file__, "--", "--root", str(paths.root)]
    return subprocess.call(command)


# --- parsing ----------------------------------------------------------------------------------


def _fit_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--candidates", type=int, nargs="+", default=list(FitConfig.candidates))
    parser.add_argument("--restarts", type=int, default=FitConfig.restarts)
    parser.add_argument("--validation-days", type=int, default=FitConfig.validation_days)


def _backtest_arguments(parser: argparse.ArgumentParser) -> None:
    _fit_arguments(parser)
    parser.add_argument("--test-start", default=DEFAULT_TEST_START)
    parser.add_argument("--refit-days", type=int, default=BacktestConfig.refit_days)
    parser.add_argument("--no-holdout", action="store_true", help="no locked holdout (demo data only)")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="regime-trader", description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, default=Path())
    commands = parser.add_subparsers(dest="command", required=True)

    fetch = commands.add_parser("fetch", help="cache hourly bars")
    fetch.add_argument("--source", choices=("ibkr", "yahoo"), default="ibkr")
    fetch.add_argument("--years", type=int, default=6)
    fetch.set_defaults(handler=_fetch)

    fit = commands.add_parser("fit", help="fit (or refit) the regime model")
    _fit_arguments(fit)
    fit.set_defaults(handler=_fit)

    backtest = commands.add_parser("backtest", help="walk-forward backtest against the gates")
    _backtest_arguments(backtest)
    backtest.add_argument("--include-holdout", action="store_true", help="only for a candidate you approved")
    backtest.set_defaults(handler=_backtest)

    commands.add_parser("live", help="run the paper trader").set_defaults(handler=_live)

    kill = commands.add_parser("kill", help="engage or reset the kill switch")
    kill.add_argument("--reason", default="manual kill (CLI)")
    kill.add_argument("--reset", action="store_true")
    kill.set_defaults(handler=_kill)

    approve = commands.add_parser("approve", help="allow orders over the approval threshold for a while")
    approve.add_argument("--minutes", type=int, default=60)
    approve.set_defaults(handler=_approve)

    nightly = commands.add_parser("nightly", help="daily report, Claude review, proposal backtests")
    _backtest_arguments(nightly)
    nightly.add_argument("--day")
    nightly.add_argument("--no-review", action="store_true")
    nightly.set_defaults(handler=_nightly)

    report = commands.add_parser("report", help="the daily report")
    report.add_argument("--day")
    report.set_defaults(handler=_report)

    commands.add_parser("watchdog", help="alert if the trader missed a bar").set_defaults(handler=_watchdog)
    commands.add_parser("dashboard", help="open the dashboard").set_defaults(handler=_dashboard)
    return parser


def main(argv: Sequence[str] | None = None, env: Mapping[str, str] | None = None) -> int:
    args = _parser().parse_args(argv)
    paths = Paths(args.root)
    if env is None:
        env = {**read_env(paths.root / ".env"), **os.environ}
    try:
        code: int = args.handler(args, paths, env)
    except PaperOnlyError as error:
        print(f"refused: {error}. This system trades IBKR paper accounts only.", file=sys.stderr)
        return 2
    except UsageError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return code
