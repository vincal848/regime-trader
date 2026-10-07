"""Step 6: one decision per bar from the shared engine (spec §2, §6, §8-§10)."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from synthetic import make_bars

from regime_trader.engine import EngineConfig, EngineState, OpenPosition, decide, start
from regime_trader.features import Z_FEATURES, compute_features
from regime_trader.hmm import HmmModel, RegimeModel
from regime_trader.playbook import Playbook, parse_playbook
from regime_trader.risk import AccountState, Approved

LABELS = ("CALM_UP", "CHOP", "STRESS", "CRASH")
# Four states over the five z-features. Each state has a distinct centre, so
# a feature row can be steered into a chosen state deterministically.
MEANS = np.array(
    [
        [0.0, -1.0, 0.0, 0.0, 2.0],  # CALM_UP: low vol, strong trend
        [0.0, 0.0, 0.0, 0.0, 0.0],  # CHOP
        [0.0, 2.0, 0.0, 0.0, -1.0],  # STRESS
        [0.0, 4.0, 0.0, 0.0, -3.0],  # CRASH
    ]
)
MODEL = RegimeModel(
    hmm=HmmModel(
        startprob=np.full(4, 0.25),
        transmat=np.full((4, 4), 0.02) + np.eye(4) * 0.92,
        means=MEANS,
        covars=np.array([np.eye(5) * 0.3] * 4),
    ),
    labels=LABELS,
    return_mean=np.array([0.001, 0.0, -0.001, -0.004]),
    return_vol=np.array([0.002, 0.003, 0.006, 0.012]),
)


def _playbook(state: str, entry: str, exit_: str, max_size: float) -> Playbook:
    block = "\n".join(
        [
            "```toml",
            f'state = "{state}"',
            f'entry = "{entry}"',
            f'exit = "{exit_}"',
            "stop_loss_vol = 3.0",
            "take_profit_vol = 50.0",
            f"max_size = {max_size}",
            "max_hold_bars = 0",
            'invalidation = "test"',
            "```",
        ]
    )
    return parse_playbook(block)


PLAYBOOKS = {
    "CALM_UP": _playbook("CALM_UP", "trend > 0", "trend < -5", 1.0),
    "CHOP": _playbook("CHOP", "never", "always", 0.5),
    "STRESS": _playbook("STRESS", "trend > -100", "never", 1.0),
    "CRASH": _playbook("CRASH", "never", "always", 0.0),
}
KELLY = dict.fromkeys(LABELS, 4.0)
CONFIG = EngineConfig()
ACCOUNT = AccountState(
    equity=100_000.0,
    start_of_day_equity=100_000.0,
    peak_equity=100_000.0,
    position_shares=0,
    consecutive_rejects=0,
    disconnected_seconds=0.0,
    killed=False,
)
TS = pd.Timestamp("2024-03-01 11:00", tz="America/New_York")


def _row(state: int, trend: float = 1.0) -> dict[str, float]:
    z = dict(zip(Z_FEATURES, MEANS[state], strict=True))
    return {"ret": 0.001, "rv": 0.002, "range": 0.002, "volume_ratio": 1.0, "trend": trend, **z}


def _steer(state: EngineState, target: int, bars: int = 4) -> EngineState:
    for _ in range(bars):
        state, _ = decide(
            state, TS, _row(target), 500.0, ACCOUNT, MODEL, PLAYBOOKS, KELLY, CONFIG, healthy=True
        )
    return state


def _holding(state: EngineState, playbook: str, shares: int, stop: float = 480.0) -> EngineState:
    return replace(state, position=OpenPosition(playbook, shares, 500.0, stop, 600.0, 3))


def test_entering_a_calm_up_regime_produces_a_sized_long_order() -> None:
    state = _steer(start(MODEL), 0)
    _, decision = decide(state, TS, _row(0), 500.0, ACCOUNT, MODEL, PLAYBOOKS, KELLY, CONFIG, healthy=True)
    assert decision.regime.active == "CALM_UP"
    assert isinstance(decision.order, Approved)
    assert 0 < decision.target_shares <= 200  # at most 100% of equity at $500
    assert decision.probabilities.shape == (4,)


def test_unhealthy_input_means_flat() -> None:
    state = _holding(_steer(start(MODEL), 0), "CALM_UP", 100)
    holding = replace(ACCOUNT, position_shares=100)
    _, decision = decide(state, TS, _row(0), 500.0, holding, MODEL, PLAYBOOKS, KELLY, CONFIG, healthy=False)
    assert decision.target_shares == 0
    assert any("unhealthy" in reason for reason in decision.reasons)


def test_nan_features_are_unhealthy() -> None:
    state = _steer(start(MODEL), 0)
    row = {**_row(0), "z_ret": float("nan")}
    _, decision = decide(state, TS, row, 500.0, ACCOUNT, MODEL, PLAYBOOKS, KELLY, CONFIG, healthy=True)
    assert decision.target_shares == 0


def test_sizing_never_exceeds_the_state_cap() -> None:
    # STRESS's playbook allows max_size 1.0; the STRESS state cap is 25%.
    state = _steer(start(MODEL), 2)
    _, decision = decide(state, TS, _row(2), 500.0, ACCOUNT, MODEL, PLAYBOOKS, KELLY, CONFIG, healthy=True)
    assert decision.regime.active == "STRESS"
    assert decision.target_shares * 500.0 <= 0.25 * ACCOUNT.equity


def test_crash_regime_never_enters_and_flattens() -> None:
    state = _holding(_steer(start(MODEL), 3), "CRASH", 50)
    holding = replace(ACCOUNT, position_shares=50)
    _, decision = decide(state, TS, _row(3), 500.0, holding, MODEL, PLAYBOOKS, KELLY, CONFIG, healthy=True)
    assert decision.target_shares == 0


def test_a_playbook_switch_closes_the_old_position() -> None:
    state = _holding(_steer(start(MODEL), 1), "CALM_UP", 80)  # CHOP is now active
    holding = replace(ACCOUNT, position_shares=80)
    _, decision = decide(state, TS, _row(1), 500.0, holding, MODEL, PLAYBOOKS, KELLY, CONFIG, healthy=True)
    assert decision.target_shares == 0
    assert any("playbook" in reason for reason in decision.reasons)


def test_a_close_through_the_stop_exits() -> None:
    state = _holding(_steer(start(MODEL), 0), "CALM_UP", 80, stop=495.0)
    holding = replace(ACCOUNT, position_shares=80)
    _, decision = decide(state, TS, _row(0), 494.0, holding, MODEL, PLAYBOOKS, KELLY, CONFIG, healthy=True)
    assert decision.target_shares == 0
    assert any("stop" in reason for reason in decision.reasons)


def test_the_kill_switch_flattens_whatever_the_model_says() -> None:
    state = _holding(_steer(start(MODEL), 0), "CALM_UP", 80)
    killed = replace(ACCOUNT, position_shares=80, killed=True)
    _, decision = decide(state, TS, _row(0), 500.0, killed, MODEL, PLAYBOOKS, KELLY, CONFIG, healthy=True)
    assert decision.target_shares == 0
    assert isinstance(decision.order, Approved)


def test_uncalibrated_sizing_falls_back_to_a_quarter_of_the_cap() -> None:
    state = _steer(start(MODEL), 0)
    config = replace(CONFIG, calibrated=False)
    _, decision = decide(state, TS, _row(0), 500.0, ACCOUNT, MODEL, PLAYBOOKS, KELLY, config, healthy=True)
    assert decision.target_fraction == pytest.approx(0.25 * 1.0 * decision.regime.size_multiplier)


def test_no_decision_depends_on_a_future_bar() -> None:
    """The spec's key guarantee: perturb every bar after t, rerun the whole
    pipeline (features, filter, switching, playbook, sizing, risk), and every
    decision at or before t is identical."""
    bars = make_bars(80, seed=21)

    def run(frame: pd.DataFrame) -> list[tuple[str, int, str | None]]:
        features = compute_features(frame)
        closes = frame["close"].to_numpy(dtype=float)
        state, out = start(MODEL), []
        for i, (ts, row) in enumerate(features.iterrows()):
            values = {str(k): float(v) for k, v in row.items()}
            healthy = bool(np.isfinite(list(values.values())).all())
            price = float(closes[i])
            state, decision = decide(
                state, pd.Timestamp(str(ts)), values, price, ACCOUNT, MODEL, PLAYBOOKS, KELLY, CONFIG, healthy
            )
            out.append(
                (decision.probabilities.tobytes().hex(), decision.target_shares, decision.regime.active)
            )
        return out

    base = run(bars)
    rng = np.random.default_rng(5)
    for t in rng.integers(300, len(bars) - 5, size=4):
        values = bars.to_numpy(copy=True)
        values[t + 1 :, :4] *= np.exp(rng.normal(0.0, 0.1, len(bars) - t - 1))[:, None]
        values[t + 1 :, 4] *= 5.0
        assert run(pd.DataFrame(values, index=bars.index, columns=bars.columns))[: t + 1] == base[: t + 1]


def test_frozen_entries_block_new_positions_but_allow_exits() -> None:
    frozen = replace(CONFIG, entries_frozen=True)
    state = _steer(start(MODEL), 0)
    _, decision = decide(state, TS, _row(0), 500.0, ACCOUNT, MODEL, PLAYBOOKS, KELLY, frozen, healthy=True)
    assert decision.target_shares == 0
    assert any("frozen" in reason for reason in decision.reasons)
    holding = _holding(state, "CALM_UP", 80, stop=495.0)
    account = replace(ACCOUNT, position_shares=80)
    _, decision = decide(holding, TS, _row(0), 494.0, account, MODEL, PLAYBOOKS, KELLY, frozen, healthy=True)
    assert decision.target_shares == 0  # the stop still works
