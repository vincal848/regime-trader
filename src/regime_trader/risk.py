"""Hard risk limits: checked before every order, and no model can change them (spec §10).

This module imports nothing from the HMM, playbooks or sizing
(tests/test_architecture.py), so the limits are only ever set in code.
Once the kill switch or the daily loss limit fires, the only orders allowed
are those that reduce the position, and a flattening order never waits for
manual approval. Everything else is vetoed with a reason that the journal
records.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

DEFAULT_STATE_CAPS: Mapping[str, float] = MappingProxyType(
    {"CALM_UP": 1.0, "CHOP": 0.5, "STRESS": 0.25, "CRASH": 0.0}
)


@dataclass(frozen=True)
class RiskLimits:
    max_position_fraction: float = 1.0  # notional / equity: no leverage
    state_caps: Mapping[str, float] = field(default_factory=lambda: DEFAULT_STATE_CAPS)
    daily_loss_limit: float = 0.02  # fraction of start-of-day equity
    max_drawdown: float = 0.10  # fraction of peak equity: the kill switch
    manual_approval_notional: float = 25_000.0
    max_consecutive_rejects: int = 3
    max_disconnect_seconds: float = 300.0


@dataclass(frozen=True)
class AccountState:
    equity: float
    start_of_day_equity: float
    peak_equity: float
    position_shares: int
    consecutive_rejects: int
    disconnected_seconds: float
    killed: bool  # the manual kill switch, or a kill that has not been reset


@dataclass(frozen=True)
class Approved:
    target_shares: int
    needs_manual_approval: bool


@dataclass(frozen=True)
class Vetoed:
    reason: str


def base_state(label: str) -> str:
    """The state a numbered label belongs to: CALM_UP_2 -> CALM_UP. Numbered
    siblings share their base state's cap and playbook. Only a trailing
    `_<digits>` is stripped, so an unmatched refit state (CALM_UP_NEW) stays
    unknown until you review it."""
    stem, _, suffix = label.rpartition("_")
    return stem if stem and suffix.isdigit() else label


def state_cap(limits: RiskLimits, label: str) -> float:
    """The cap for a state label, by its base state. Unknown states get zero."""
    return limits.state_caps.get(base_state(label), 0.0)


def kill_reasons(limits: RiskLimits, account: AccountState) -> tuple[str, ...]:
    reasons = []
    drawdown = 1.0 - account.equity / account.peak_equity if account.peak_equity > 0 else 0.0
    if drawdown >= limits.max_drawdown:
        reasons.append(f"max drawdown {drawdown:.1%} >= {limits.max_drawdown:.0%}")
    if account.consecutive_rejects >= limits.max_consecutive_rejects:
        reasons.append(f"{account.consecutive_rejects} consecutive order rejects")
    if account.disconnected_seconds > limits.max_disconnect_seconds:
        reasons.append(f"broker disconnected {account.disconnected_seconds:.0f}s")
    if account.killed:
        reasons.append("kill switch engaged")
    return tuple(reasons)


def daily_loss_breached(limits: RiskLimits, account: AccountState) -> bool:
    return account.equity / account.start_of_day_equity - 1.0 <= -limits.daily_loss_limit


def check_order(
    limits: RiskLimits, account: AccountState, label: str, target_shares: int, price: float
) -> Approved | Vetoed:
    """Approve moving the position to `target_shares`, or veto it."""
    if target_shares < 0:
        return Vetoed("long/flat only: short targets are not allowed")
    current = account.position_shares
    reducing = target_shares <= current
    kills = kill_reasons(limits, account)
    if kills and target_shares != 0:
        return Vetoed("kill switch: " + "; ".join(kills) + ": flatten only")
    if daily_loss_breached(limits, account) and not reducing:
        return Vetoed(f"daily loss limit {limits.daily_loss_limit:.0%} reached: no new risk today")
    exposure = target_shares * price / account.equity
    cap = min(limits.max_position_fraction, state_cap(limits, label))
    if exposure > cap and not reducing:
        return Vetoed(f"exposure {exposure:.1%} exceeds the {label} cap {cap:.0%}")
    if kills or reducing:
        return Approved(target_shares, needs_manual_approval=False)  # reducing risk never waits
    notional = (target_shares - current) * price
    return Approved(target_shares, needs_manual_approval=notional > limits.manual_approval_notional)
