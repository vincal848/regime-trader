"""The decision engine: one bar in, one decision out (spec §2).

The backtest and live trading both call `decide`, so the code that is
tested is the code that trades. Per bar:

1. Advance the forward filter one step (`hmm.filter_step`).
2. Update the switching state machine, which picks the active playbook.
3. Decide the target position, in this order of authority:
   - the kill switch or the daily loss limit means flat;
   - unhealthy input, no active regime, or zero size means flat;
   - a position opened by another playbook is closed;
   - a stop, take-profit, exit signal or time limit closes the position;
   - while holding, size may only shrink;
   - with no position, the active playbook's entry signal opens one, sized
     by `sizing`.
4. Check the target against the hard limits (`risk.check_order`).

The engine never places orders and never assumes a fill. The execution
layer calls `on_fill` with what actually happened.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace

import numpy as np
import numpy.typing as npt
import pandas as pd

from regime_trader.features import FEATURES, Z_FEATURES
from regime_trader.hmm import RegimeModel, filter_step, initial_prior
from regime_trader.playbook import Playbook, Signal, evaluate_signal
from regime_trader.risk import (
    AccountState,
    Approved,
    RiskLimits,
    Vetoed,
    check_order,
    daily_loss_breached,
    kill_reasons,
    state_cap,
)
from regime_trader.sizing import KELLY_FRACTION, target_fraction
from regime_trader.switching import INITIAL, Regime, SwitchConfig, SwitchState, step_switch

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class OpenPosition:
    playbook: str  # the playbook that opened it
    shares: int
    entry_price: float
    stop_price: float
    take_profit_price: float
    bars_held: int


@dataclass(frozen=True)
class EngineState:
    prior: FloatArray  # P(s_t | x_..t-1): the filter's prior for the next bar
    switch: SwitchState
    position: OpenPosition | None


@dataclass(frozen=True)
class EngineConfig:
    switch: SwitchConfig = field(default_factory=SwitchConfig)
    limits: RiskLimits = field(default_factory=RiskLimits)
    calibrated: bool = True  # False: fixed sizing of a quarter of the state cap (spec §9)


@dataclass(frozen=True)
class Decision:
    ts: pd.Timestamp
    probabilities: FloatArray
    next_state: FloatArray
    log_likelihood: float  # NaN when the bar was unhealthy
    regime: Regime
    signal: Signal | None  # None when no playbook was consulted
    target_fraction: float
    target_shares: int
    order: Approved | Vetoed | None  # None when the target equals the current position
    entry_rv: float  # the bar's realized volatility, for setting stops on a fill
    reasons: tuple[str, ...]


def start(model: RegimeModel) -> EngineState:
    return EngineState(prior=initial_prior(model.hmm), switch=INITIAL, position=None)


def _sized_shares(
    probabilities: FloatArray,
    active: str,
    model: RegimeModel,
    playbook: Playbook,
    kelly: Mapping[str, float],
    multiplier: float,
    account: AccountState,
    price: float,
    config: EngineConfig,
) -> tuple[float, int]:
    cap = state_cap(config.limits, active)
    if config.calibrated:
        index = model.labels.index(active)
        fraction = target_fraction(
            probabilities, index, kelly.get(active, 0.0), cap, playbook.max_size, multiplier
        )
    else:
        fraction = min(KELLY_FRACTION * cap, playbook.max_size) * multiplier
    return fraction, math.floor(fraction * account.equity / price)


def decide(
    state: EngineState,
    ts: pd.Timestamp,
    row: Mapping[str, float],
    price: float,
    account: AccountState,
    model: RegimeModel,
    playbooks: Mapping[str, Playbook],
    kelly: Mapping[str, float],
    config: EngineConfig,
    healthy: bool,
) -> tuple[EngineState, Decision]:
    observation = np.array([row.get(name, math.nan) for name in Z_FEATURES])
    raw = np.array([row.get(name, math.nan) for name in FEATURES])
    healthy = healthy and bool(np.isfinite(observation).all() and np.isfinite(raw).all())
    if healthy:
        step = filter_step(model.hmm, state.prior, observation)
        probabilities, next_state, log_likelihood = step.filtered, step.next_state, step.log_likelihood
    else:
        probabilities, next_state, log_likelihood = state.prior, state.prior, math.nan
    switch, regime = step_switch(
        state.switch, model.labels, probabilities, next_state, healthy, config.switch
    )

    position = state.position
    if position is not None:
        position = replace(position, bars_held=position.bars_held + 1)
    current = account.position_shares
    reasons = list(regime.reasons)
    signal: Signal | None = None
    fraction = 0.0
    target = 0
    kills = kill_reasons(config.limits, account)

    if kills or daily_loss_breached(config.limits, account):
        reasons.append("hard limit: " + ("; ".join(kills) if kills else "daily loss limit") + ": flatten")
    elif not healthy or regime.active is None or regime.size_multiplier == 0.0:
        pass  # the switching reasons already explain why size is zero
    elif regime.active not in playbooks:
        reasons.append(f"no playbook for {regime.active}: flat")
    elif position is not None and position.playbook != regime.active:
        reasons.append(f"playbook switched {position.playbook} -> {regime.active}: close")
    else:
        playbook = playbooks[regime.active]
        fraction, sized = _sized_shares(
            probabilities,
            regime.active,
            model,
            playbook,
            kelly,
            regime.size_multiplier,
            account,
            price,
            config,
        )
        if position is not None and price <= position.stop_price:
            reasons.append(f"stop {position.stop_price:.2f} hit at {price:.2f}")
        elif position is not None and price >= position.take_profit_price:
            reasons.append(f"take profit {position.take_profit_price:.2f} hit at {price:.2f}")
        else:
            signal = evaluate_signal(
                playbook, row, position is not None, position.bars_held if position else 0
            )
            reasons.append(signal.reason)
            if position is not None and not signal.exit:
                target = min(current, sized)  # while holding, size may only shrink
            elif position is None and signal.enter:
                target = sized

    order: Approved | Vetoed | None = None
    if target != current:
        order = check_order(config.limits, account, regime.active or "NONE", target, price)
        if isinstance(order, Vetoed):
            reasons.append(f"risk veto: {order.reason}")
            target = current
    decision = Decision(
        ts=ts,
        probabilities=probabilities,
        next_state=next_state,
        log_likelihood=log_likelihood,
        regime=regime,
        signal=signal,
        target_fraction=fraction,
        target_shares=target,
        order=order,
        entry_rv=float(row.get("rv", math.nan)),
        reasons=tuple(reasons),
    )
    prior = next_state if healthy else state.prior
    return EngineState(prior=prior, switch=switch, position=position), decision


def on_fill(
    state: EngineState, decision: Decision, playbooks: Mapping[str, Playbook], shares: int, fill_price: float
) -> EngineState:
    """Record the position after a fill. `shares` is the new total position."""
    if shares == 0:
        return replace(state, position=None)
    if state.position is not None:
        return replace(state, position=replace(state.position, shares=shares))
    active = decision.regime.active
    if active is None or active not in playbooks:
        raise ValueError(f"fill opened a position without an active playbook ({active})")
    playbook = playbooks[active]
    rv = decision.entry_rv
    position = OpenPosition(
        playbook=active,
        shares=shares,
        entry_price=fill_price,
        stop_price=fill_price * math.exp(-playbook.stop_loss_vol * rv),
        take_profit_price=fill_price * math.exp(playbook.take_profit_vol * rv),
        bars_held=0,
    )
    return replace(state, position=position)
