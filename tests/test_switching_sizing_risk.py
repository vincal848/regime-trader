"""Step 4: switching rules (spec §8), sizing (§9) and hard risk limits (§10)."""

from dataclasses import replace

import numpy as np
import pytest

from regime_trader.risk import (
    AccountState,
    Approved,
    RiskLimits,
    Vetoed,
    check_order,
    kill_reasons,
    state_cap,
)
from regime_trader.sizing import entropy, kelly_fraction, target_fraction
from regime_trader.switching import INITIAL, Regime, SwitchConfig, SwitchState, step_switch

LABELS = ("CALM_UP", "CHOP", "STRESS", "CRASH")
CONFIG = SwitchConfig()


def _run(
    probs: list[list[float]],
    state: SwitchState = INITIAL,
    next_state: list[float] | None = None,
    healthy: bool = True,
) -> tuple[SwitchState, list[Regime]]:
    regimes: list[Regime] = []
    for p in probs:
        nxt = np.array(next_state if next_state is not None else p)
        state, regime = step_switch(state, LABELS, np.array(p), nxt, healthy, CONFIG)
        regimes.append(regime)
    return state, regimes


CALM = [0.85, 0.10, 0.04, 0.01]
CHOPPY = [0.10, 0.85, 0.04, 0.01]


# --- switching ----------------------------------------------------------------------


def test_no_regime_until_a_state_takes_over() -> None:
    _, regimes = _run([CALM, CALM])
    assert regimes[-1].active is None
    assert regimes[-1].size_multiplier == 0.0


def test_rule_1_and_2_takeover_needs_probability_above_070_for_three_bars() -> None:
    state, regimes = _run([CALM, CALM, CALM])
    assert regimes[1].active is None
    assert regimes[2].active == "CALM_UP"
    assert regimes[2].switched


def test_rule_1_a_lead_at_or_below_the_threshold_never_takes_over() -> None:
    _, regimes = _run([[0.70, 0.20, 0.05, 0.05]] * 5)
    assert all(r.active is None for r in regimes)


def test_rule_2_an_interrupted_lead_restarts_the_count() -> None:
    _, regimes = _run([CALM, CALM, CHOPPY, CALM, CALM])
    assert regimes[-1].active is None
    _, regimes = _run([CALM, CALM, CHOPPY, CALM, CALM, CALM])
    assert regimes[-1].active == "CALM_UP"


def test_rule_3_cooldown_blocks_a_new_switch() -> None:
    state, _ = _run([CALM] * 3)
    state, regimes = _run([CHOPPY] * 9, state)
    switch_bars = [i for i, r in enumerate(regimes) if r.switched]
    # The challenger qualifies after 3 bars, but the 7-bar cooldown from the
    # first switch holds it back until the cooldown has run out.
    assert switch_bars == [6]
    assert regimes[6].active == "CHOP"


def test_rule_4_rising_stress_risk_halves_size_early() -> None:
    state, _ = _run([CALM] * 3)
    _, regimes = _run([CALM], state, next_state=[0.60, 0.10, 0.20, 0.10])  # 0.30 > 0.25 into STRESS/CRASH
    assert regimes[0].size_multiplier == 0.5
    assert any("next-bar stress" in reason for reason in regimes[0].reasons)


def test_rule_5_near_tie_between_the_top_two_means_zero_size() -> None:
    state, _ = _run([CALM] * 3)
    _, regimes = _run([[0.45, 0.40, 0.10, 0.05]], state)
    assert regimes[0].size_multiplier == 0.0
    assert any("uncertain" in reason for reason in regimes[0].reasons)


def test_rule_6_unhealthy_input_means_flat_and_resets_the_challenger() -> None:
    state, _ = _run([CALM] * 3)
    state, _ = _run([CHOPPY, CHOPPY], state)
    state, regimes = _run([CHOPPY], state, healthy=False)
    assert regimes[0].size_multiplier == 0.0
    assert state.streak == 0


def test_every_regime_carries_its_probabilities_and_reasons() -> None:
    _, regimes = _run([CALM] * 3)
    assert regimes[2].reasons
    assert "switch" in regimes[2].reasons[0]


# --- sizing --------------------------------------------------------------------------


def test_entropy_is_zero_when_certain_and_ln_k_when_uniform() -> None:
    assert entropy(np.array([1.0, 0.0, 0.0])) == pytest.approx(0.0)
    assert entropy(np.full(4, 0.25)) == pytest.approx(np.log(4))


def test_kelly_is_mean_over_variance_and_never_negative() -> None:
    returns = np.array([0.01, -0.005, 0.012, 0.003])
    assert kelly_fraction(returns) == pytest.approx(returns.mean() / returns.var(ddof=1))
    assert kelly_fraction(-returns) == 0.0


def test_size_is_quarter_kelly_times_confidence_times_certainty() -> None:
    probs = np.array([0.9, 0.05, 0.03, 0.02])
    size = target_fraction(probs, active=0, kelly=2.0, cap=1.0, playbook_max=1.0, multiplier=1.0)
    expected = 0.25 * 2.0 * 0.9 * (1 - entropy(probs) / np.log(4))
    assert size == pytest.approx(expected)


def test_size_is_capped_by_the_state_cap_and_the_playbook() -> None:
    probs = np.array([1.0, 0.0, 0.0, 0.0])
    assert target_fraction(probs, 0, kelly=40.0, cap=0.5, playbook_max=1.0, multiplier=1.0) == pytest.approx(
        0.5
    )
    assert target_fraction(probs, 0, kelly=40.0, cap=1.0, playbook_max=0.3, multiplier=1.0) == pytest.approx(
        0.3
    )


def test_size_is_zero_with_a_zero_multiplier_or_zero_cap() -> None:
    probs = np.array([1.0, 0.0, 0.0, 0.0])
    assert target_fraction(probs, 0, kelly=2.0, cap=1.0, playbook_max=1.0, multiplier=0.0) == 0.0
    assert target_fraction(probs, 0, kelly=2.0, cap=0.0, playbook_max=1.0, multiplier=1.0) == 0.0


# --- risk -------------------------------------------------------------------------------

LIMITS = RiskLimits()
PRICE = 500.0


FLAT = AccountState(
    equity=100_000.0,
    start_of_day_equity=100_000.0,
    peak_equity=100_000.0,
    position_shares=0,
    consecutive_rejects=0,
    disconnected_seconds=0.0,
    killed=False,
)


def _account(**changes: float) -> AccountState:
    return replace(FLAT, **changes)


def test_state_caps_match_by_label_prefix_and_unknown_labels_get_zero() -> None:
    assert state_cap(LIMITS, "CALM_UP_2") == 1.0
    assert state_cap(LIMITS, "CHOP") == 0.5
    assert state_cap(LIMITS, "STRESS_1") == 0.25
    assert state_cap(LIMITS, "CRASH") == 0.0
    assert state_cap(LIMITS, "SOMETHING_NEW") == 0.0


def test_an_order_inside_every_limit_is_approved() -> None:
    result = check_order(LIMITS, _account(), "CALM_UP", target_shares=40, price=PRICE)
    assert isinstance(result, Approved)
    assert not result.needs_manual_approval


def test_long_flat_only_rejects_short_targets() -> None:
    assert isinstance(check_order(LIMITS, _account(), "CALM_UP", target_shares=-10, price=PRICE), Vetoed)


def test_state_cap_vetoes_an_oversized_target() -> None:
    result = check_order(LIMITS, _account(), "STRESS", target_shares=60, price=PRICE)  # 30% > 25% cap
    assert isinstance(result, Vetoed)
    assert "cap" in result.reason


def test_large_orders_need_manual_approval() -> None:
    result = check_order(LIMITS, _account(), "CALM_UP", target_shares=60, price=PRICE)  # $30,000 > $25,000
    assert isinstance(result, Approved)
    assert result.needs_manual_approval


def test_daily_loss_limit_allows_only_reducing_orders() -> None:
    account = _account(equity=97_900.0, position_shares=100)  # -2.1% today
    assert isinstance(check_order(LIMITS, account, "CALM_UP", target_shares=120, price=PRICE), Vetoed)
    assert isinstance(check_order(LIMITS, account, "CALM_UP", target_shares=0, price=PRICE), Approved)


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"equity": 89_000.0}, "drawdown"),
        ({"consecutive_rejects": 3}, "rejects"),
        ({"disconnected_seconds": 301.0}, "disconnected"),
        ({"killed": True}, "kill switch"),
    ],
)
def test_kill_switch_triggers(changes: dict[str, float], reason: str) -> None:
    account = _account(**changes, position_shares=50)
    reasons = kill_reasons(LIMITS, account)
    assert any(reason in r for r in reasons)
    assert isinstance(check_order(LIMITS, account, "CALM_UP", target_shares=60, price=PRICE), Vetoed)
    assert isinstance(check_order(LIMITS, account, "CALM_UP", target_shares=0, price=PRICE), Approved)
