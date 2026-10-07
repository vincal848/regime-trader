"""The nightly loop (spec §13) and the daily report (spec §14).

After the close:

1. **Daily report.** Written to `reports/DAY.md` and sent as an alert.
2. **Review.** The day's record goes to Claude (`llm.NightlyReviewer`, under
   the monthly spend cap), framed inside `<record>` tags as data.
3. **Validation.** Each proposed playbook is parsed with the same
   whitelisted grammar as the real ones; anything else is rejected.
4. **Backtest.** Each valid proposal is run through the acceptance gates
   (§11), outside the locked holdout, with the same baselines as the current
   system.
5. **Filing.** Everything lands in `proposals/DAY/`: the review, each
   candidate file, and `evaluation.md`.

Nothing is applied. A candidate that clears the gates waits for you to copy
it into `playbooks/` as a normal commit. Every evaluated candidate
increments a running count in `proposals/trials.json`, so the
multiple-testing risk of repeated tweaking stays visible.
"""

from __future__ import annotations

import contextlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from regime_trader.alerts import Alerts
from regime_trader.backtest import BacktestConfig, acceptance, acceptance_report, baseline_sharpes
from regime_trader.bars import TIMEZONE
from regime_trader.calibration import smoothed_states, state_calibration
from regime_trader.features import Z_FEATURES, compute_features, healthy
from regime_trader.hmm import forward_filter
from regime_trader.llm import BudgetExceededError, NightlyReviewer, ReviewRefusedError
from regime_trader.playbook import Playbook, PlaybookError, parse_playbook
from regime_trader.refit import Fit
from regime_trader.store import Journal

CALIBRATION_SESSIONS = 60
PLAYBOOK_TAG = re.compile(r'<playbook state="([A-Za-z0-9_]+)">(.*?)</playbook>', re.DOTALL)


# --- the journal, by day ---------------------------------------------------------------------


def _local(column: pd.Series) -> pd.Series:
    return pd.to_datetime(column, utc=True).dt.tz_convert(TIMEZONE)


def _on(frame: pd.DataFrame, column: str, day: date) -> pd.DataFrame:
    return frame[_local(frame[column]).dt.date == day] if len(frame) else frame


@dataclass(frozen=True)
class RoundTrip:
    state: str
    opened: str
    closed: str
    pnl: float  # after commissions

    @property
    def closed_on(self) -> date:
        return pd.Timestamp(self.closed).tz_convert(TIMEZONE).date()


def round_trips(fills: pd.DataFrame, decisions: pd.DataFrame) -> list[RoundTrip]:
    """Pair fills into flat-to-flat round trips, each credited to the state
    that was active when its opening order was decided."""
    active = dict(zip(decisions["ts"], decisions["active"], strict=True)) if len(decisions) else {}
    trips = []
    position, cash, opened, state = 0, 0.0, "", ""
    ordered = fills.sort_values("fill_ts")
    columns = ("decision_ts", "fill_ts", "shares", "price", "commission")
    for decision_ts, fill_ts, shares, price, commission in zip(*(ordered[c] for c in columns), strict=True):
        if position == 0:
            opened, state = str(fill_ts), str(active.get(decision_ts) or "NONE")
        position += int(shares)
        cash -= int(shares) * float(price) + float(commission)
        if position == 0:
            trips.append(RoundTrip(state, opened, str(fill_ts), cash))
            cash = 0.0
    return trips


def _healthy_z(bars: pd.DataFrame) -> pd.DataFrame:
    features = compute_features(bars)
    return features.loc[healthy(features), list(Z_FEATURES)]


def wrong_state_calls(bars: pd.DataFrame, fit: Fit, day: date) -> pd.DataFrame:
    """Bars of `day` where the filtered leader differs from the state that
    hindsight (smoothing over everything through `day`) assigns."""
    z = _healthy_z(bars[pd.DatetimeIndex(bars.index).date <= day])
    labels = np.array(fit.model.labels)
    filtered = labels[forward_filter(fit.model.hmm, z.to_numpy()).filtered.argmax(axis=1)]
    hindsight = labels[smoothed_states(fit.model, z.to_numpy())]
    calls = pd.DataFrame({"filtered": filtered, "hindsight": hindsight}, index=z.index)
    today = calls[pd.DatetimeIndex(calls.index).date == day]
    wrong: pd.DataFrame = today[today["filtered"] != today["hindsight"]]
    return wrong


def _calibration_line(bars: pd.DataFrame, fit: Fit) -> str:
    z = _healthy_z(bars).to_numpy()[-CALIBRATION_SESSIONS * 7 :]
    report = state_calibration(fit.model, z)
    brier = np.mean([s.brier for s in report.states])
    climatology = np.mean([s.climatology_brier for s in report.states])
    verdict = "calibrated" if report.calibrated else "NOT calibrated: sizing falls back to fixed fractions"
    return (
        f"Brier {brier:.3f} vs climatology {climatology:.3f} over the last "
        f"{CALIBRATION_SESSIONS} sessions ({verdict})"
    )


def daily_report(journal: Journal, bars: pd.DataFrame, fit: Fit, day: date) -> str:
    decisions = _on(journal.decisions(), "ts", day)
    trips = [t for t in round_trips(journal.fills(), journal.decisions()) if t.closed_on == day]
    lines = [f"# Daily report {day}", ""]

    lines += ["## Current state"]
    if len(decisions):
        last = decisions.iloc[-1]
        probabilities = json.loads(last["probabilities"])
        lines.append(
            f"{last['active'] or 'none'} (filtered: "
            + ", ".join(f"{k} {v:.2f}" for k, v in probabilities.items())
            + f"); target {last['target_shares']} shares, last order {last['order_status']}"
        )
    else:
        lines.append("no decisions today")

    lines += ["", "## Time in each state"]
    counts = decisions["active"].fillna("none").value_counts() if len(decisions) else pd.Series(dtype=int)
    lines += [f"- {state}: {n} bars ({n / counts.sum():.0%})" for state, n in counts.items()] or ["- none"]

    lines += ["", "## Trades"]
    lines += [f"- {t.state}: {t.opened} -> {t.closed}, P&L {t.pnl:+,.2f}" for t in trips] or ["- none"]
    by_state: dict[str, float] = {}
    for t in trips:
        by_state[t.state] = by_state.get(t.state, 0.0) + t.pnl
    lines += ["", "## P&L per state"]
    lines += [f"- {state}: {pnl:+,.2f}" for state, pnl in by_state.items()] or ["- none"]

    wins = [t for t in trips if t.pnl > 0]
    win_rate = f"{len(wins) / len(trips):.0%} of {len(trips)} trades" if trips else "no trades"
    lines += ["", "## Win rate", win_rate]
    losses = [t.pnl for t in trips if t.pnl < 0]
    lines += ["", "## Largest loss", f"{min(losses):+,.2f}" if losses else "none"]
    lines += ["", "## Calibration", _calibration_line(bars, fit)]
    return "\n".join(lines) + "\n"


def day_record(journal: Journal, bars: pd.DataFrame, fit: Fit, day: date) -> str:
    """The day as data for the reviewer, inside <record> tags."""
    day_bars = bars[pd.DatetimeIndex(bars.index).date == day]
    decisions, fills = journal.decisions(), journal.fills()
    trips = round_trips(fills, decisions)
    sections = {
        "Bars": day_bars.to_csv(),
        "Decisions (filtered probabilities, never smoothed)": _on(decisions, "ts", day).to_csv(index=False),
        "Fills": _on(fills, "fill_ts", day).to_csv(index=False),
        "Events": _on(journal.events(), "ts", day).to_csv(index=False),
        "Round trips (all time)": "\n".join(f"{t.state},{t.opened},{t.closed},{t.pnl:.2f}" for t in trips),
        "Wrong state calls (filtered leader vs hindsight)": wrong_state_calls(bars, fit, day).to_csv(),
        "Daily report": daily_report(journal, bars, fit, day),
    }
    body = "\n\n".join(f"## {title}\n{text.strip() or '(none)'}" for title, text in sections.items())
    return f"<record>\nDay: {day}\nModel states: {', '.join(fit.model.labels)}\n\n{body}\n</record>\n"


# --- proposals -------------------------------------------------------------------------------


@dataclass(frozen=True)
class Proposal:
    state: str
    text: str
    playbook: Playbook | None
    error: str | None


def extract_playbooks(review: str) -> list[Proposal]:
    """Each <playbook state="X"> block, parsed with the playbook grammar.
    A block that fails to parse, or declares a different state, is rejected."""
    proposals = []
    for state, text in PLAYBOOK_TAG.findall(review):
        try:
            playbook = parse_playbook(text)
        except (PlaybookError, ValueError) as error:
            proposals.append(Proposal(state, text, None, str(error)))
            continue
        if playbook.state != state:
            proposals.append(Proposal(state, text, None, f"declares {playbook.state}, not {state}"))
            continue
        proposals.append(Proposal(state, text, playbook, None))
    return proposals


@dataclass(frozen=True)
class ProposalEvaluation:
    state: str
    error: str | None
    passed: bool = False
    checks: dict[str, bool] = field(default_factory=dict)
    report: str = ""


def evaluate_proposals(
    proposals: list[Proposal], bars: pd.DataFrame, playbooks: Mapping[str, Playbook], config: BacktestConfig
) -> list[ProposalEvaluation]:
    """Backtest each valid proposal (outside the holdout) against the gates."""
    valid = [p for p in proposals if p.playbook is not None]
    baselines = baseline_sharpes(bars, playbooks, config) if valid else {}
    evaluations = []
    for proposal in proposals:
        if proposal.playbook is None:
            evaluations.append(ProposalEvaluation(proposal.state, proposal.error))
            continue
        candidate = {**playbooks, proposal.state: proposal.playbook}
        result = acceptance(bars, candidate, config, baselines=baselines)
        evaluations.append(
            ProposalEvaluation(
                proposal.state, None, result.gates.passed, result.gates.checks, acceptance_report(result)
            )
        )
    return evaluations


def _count_trials(proposals_dir: Path, new: int) -> int:
    path = proposals_dir / "trials.json"
    total = (json.loads(path.read_text(encoding="utf-8"))["count"] if path.exists() else 0) + new
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"count": total}), encoding="utf-8")
    return total


def _evaluation_markdown(evaluations: list[ProposalEvaluation], trials: int) -> str:
    lines = ["# Proposal evaluation", ""]
    for e in evaluations:
        if e.error:
            lines += [f"## {e.state}: rejected", f"Not a valid playbook: {e.error}", ""]
        else:
            verdict = "cleared the gates" if e.passed else "failed the gates"
            lines += [f"## {e.state}: {verdict}", "```", e.report, "```", ""]
    lines += [
        f"Candidate configurations tested to date: {trials}. Each one is another draw against the same",
        "data, so treat a marginal pass with suspicion.",
        "",
        "Nothing ships without your approval: copy a candidate into playbooks/ as a normal commit",
        "(tests and a changelog entry), and evaluate it once on the locked holdout first.",
    ]
    return "\n".join(lines) + "\n"


# --- the nightly run ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NightlyResult:
    report_path: Path
    review_path: Path | None
    evaluations: tuple[ProposalEvaluation, ...]


def _alert(alerts: Alerts, kind: str, text: str) -> None:
    with contextlib.suppress(Exception):  # alerts are best-effort; the files are the record
        alerts.send(kind, text)


def run_nightly(
    *,
    day: date,
    bars: pd.DataFrame,
    fit: Fit,
    playbooks: Mapping[str, Playbook],
    journal: Journal,
    alerts: Alerts,
    reports_dir: Path,
    proposals_dir: Path,
    backtest: BacktestConfig,
    reviewer: NightlyReviewer | None,
    now: datetime,
) -> NightlyResult:
    report = daily_report(journal, bars, fit, day)
    report_path = reports_dir / f"{day}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    _alert(alerts, "report", report)

    if reviewer is None:
        return NightlyResult(report_path, None, ())
    try:
        review = reviewer.review(day_record(journal, bars, fit, day), now)
    except (BudgetExceededError, ReviewRefusedError) as error:
        _alert(alerts, "error", f"nightly review skipped (LLM budget or refusal): {error}")
        return NightlyResult(report_path, None, ())

    folder = proposals_dir / str(day)
    folder.mkdir(parents=True, exist_ok=True)
    review_path = folder / "review.md"
    header = f"<!-- {review.model}, ${review.cost_usd:.4f}; advice only, never applied automatically -->"
    review_path.write_text(f"{header}\n{review.text}", encoding="utf-8")
    proposals = extract_playbooks(review.text)
    for proposal in proposals:
        if proposal.playbook is not None:
            (folder / f"{proposal.state}.md").write_text(proposal.text.strip() + "\n", encoding="utf-8")
    evaluations = evaluate_proposals(proposals, bars, playbooks, backtest)
    trials = _count_trials(proposals_dir, sum(e.error is None for e in evaluations))
    (folder / "evaluation.md").write_text(_evaluation_markdown(evaluations, trials), encoding="utf-8")
    passed = [e.state for e in evaluations if e.passed]
    summary = f"nightly review filed in proposals/{day}: {len(proposals)} playbook proposals"
    if passed:
        summary += f"; {', '.join(passed)} cleared the gates and await your approval"
    _alert(alerts, "report", summary)
    return NightlyResult(report_path, review_path, tuple(evaluations))
