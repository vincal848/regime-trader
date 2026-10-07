"""Step 7b: the walk-forward backtest (spec §11, §12)."""

from dataclasses import replace
from itertools import pairwise

import numpy as np
import pandas as pd
import pytest
from synthetic import make_regime_bars

from regime_trader.backtest import (
    BacktestConfig,
    BacktestResult,
    Costs,
    commission,
    fill_price,
    run_backtest,
    run_buy_and_hold,
    run_static,
    summarize,
)
from regime_trader.playbook import Playbook, parse_playbook

N_DAYS = 330
BARS = make_regime_bars(N_DAYS, seed=3)
TEST_START = BARS.index[7 * 200]  # about 9 months of training before trading starts


def _playbook(state: str, entry: str, exit_: str, max_size: float) -> Playbook:
    lines = [
        "```toml",
        f'state = "{state}"',
        f'entry = "{entry}"',
        f'exit = "{exit_}"',
        "stop_loss_vol = 4.0",
        "take_profit_vol = 8.0",
        f"max_size = {max_size}",
        "max_hold_bars = 0",
        'invalidation = "test"',
        "```",
    ]
    return parse_playbook("\n".join(lines))


PLAYBOOKS = {
    "CALM_UP": _playbook("CALM_UP", "trend > 0", "trend < -1", 1.0),
    "CHOP": _playbook("CHOP", "z_ret < -1", "z_ret > 0", 0.5),
    "STRESS": _playbook("STRESS", "never", "always", 0.25),
    "CRASH": _playbook("CRASH", "never", "always", 0.0),
}
CONFIG = BacktestConfig(test_start=TEST_START, candidates=(2, 3), restarts=2, validation_days=60)


@pytest.fixture(scope="module")
def result() -> BacktestResult:
    return run_backtest(BARS, PLAYBOOKS, CONFIG)


def test_fill_prices_pay_half_the_spread_and_slippage() -> None:
    costs = Costs(slippage_bps=1.0, half_spread=0.005)
    assert fill_price(500.0, buying=True, costs=costs) == pytest.approx(500.0 * 1.0001 + 0.005)
    assert fill_price(500.0, buying=False, costs=costs) == pytest.approx(500.0 * 0.9999 - 0.005)


def test_commission_is_per_share_with_a_minimum() -> None:
    costs = Costs(commission_per_share=0.0035, min_commission=0.35)
    assert commission(10, costs) == pytest.approx(0.35)
    assert commission(1000, costs) == pytest.approx(3.5)
    assert commission(0, costs) == 0.0


def test_the_backtest_trades_only_out_of_sample(result: BacktestResult) -> None:
    assert result.equity.index[0] >= TEST_START
    assert result.equity.index[-1] == BARS.index[-1]
    assert len(result.trades) > 0


def test_every_fill_happens_at_the_open_of_the_bar_after_its_decision(result: BacktestResult) -> None:
    position = {ts: i for i, ts in enumerate(BARS.index)}
    for fill in result.fills:
        assert position[fill.fill_ts] == position[fill.decision_ts] + 1
        assert fill.price != pytest.approx(BARS.at[fill.decision_ts, "close"], rel=0)


def test_positions_stay_long_or_flat_and_within_the_largest_cap(result: BacktestResult) -> None:
    assert (result.positions >= 0).all()
    notional = result.positions * BARS["close"].reindex(result.positions.index)
    assert (notional <= result.equity * 1.0 + 1e-6).all()


def test_sizing_waits_for_calibration(result: BacktestResult) -> None:
    first, second = result.refits[0], result.refits[1]
    assert not first.calibrated  # nothing has scored the first fit yet
    window = (result.positions.index >= first.ts) & (result.positions.index < second.ts)
    notional = (result.positions * BARS["close"].reindex(result.positions.index))[window]
    assert (notional <= 0.25 * result.equity[window] + 1e-6).all()


def test_cash_accounting_reconciles(result: BacktestResult) -> None:
    closed = sum(t.pnl for t in result.trades)
    open_pnl = result.equity.iloc[-1] - CONFIG.initial_equity - closed
    last = result.positions.iloc[-1]
    if last == 0:
        assert open_pnl == pytest.approx(0.0, abs=1e-6)
    else:
        assert np.isfinite(open_pnl)


def test_refits_are_monthly_with_stable_labels(result: BacktestResult) -> None:
    times = [refit.ts for refit in result.refits]
    assert len(times) >= 3
    assert all((b - a) >= pd.Timedelta(days=30) for a, b in pairwise(times))
    assert len({tuple(sorted(refit.labels)) for refit in result.refits}) == 1


def test_costs_reduce_the_result(result: BacktestResult) -> None:
    free = run_backtest(BARS, PLAYBOOKS, replace(CONFIG, costs=Costs(0.0, 0.0, 0.0, 0.0)))
    assert free.equity.iloc[-1] > result.equity.iloc[-1]


def test_the_holdout_is_locked_unless_explicitly_requested() -> None:
    holdout_start = BARS.index[7 * 300]
    config = replace(CONFIG, holdout_start=holdout_start)
    locked = run_backtest(BARS, PLAYBOOKS, config)
    assert locked.equity.index[-1] < holdout_start
    opened = run_backtest(BARS, PLAYBOOKS, config, include_holdout=True)
    assert opened.equity.index[-1] >= holdout_start


def test_no_refit_or_mark_depends_on_bars_after_it(result: BacktestResult) -> None:
    cut = result.refits[2].ts
    values = BARS.to_numpy(copy=True)
    later = BARS.index > cut
    rng = np.random.default_rng(9)
    values[later, :4] *= np.exp(rng.normal(0.0, 0.05, int(later.sum())))[:, None]
    perturbed = run_backtest(pd.DataFrame(values, index=BARS.index, columns=BARS.columns), PLAYBOOKS, CONFIG)
    # Equal to 1e-9, not bit-for-bit: hmmlearn's EM runs on multithreaded BLAS
    # (spread ~1e-12 between identical runs). A leak would differ by far more.
    for before, after in zip(result.refits[:3], perturbed.refits[:3], strict=True):
        np.testing.assert_allclose(before.transmat, after.transmat, atol=1e-9)
    earlier = result.equity.index < cut
    pd.testing.assert_series_equal(result.equity[earlier], perturbed.equity[earlier], rtol=1e-9)


def test_per_state_attribution_covers_every_bar(result: BacktestResult) -> None:
    assert result.per_state["bars"].sum() == len(result.equity) - 1


def test_baselines_and_summary() -> None:
    hold = run_buy_and_hold(BARS, CONFIG)
    assert hold.index[0] >= TEST_START
    assert hold.iloc[-1] / hold.iloc[0] == pytest.approx(
        BARS["close"].iloc[-1] / BARS["close"].loc[hold.index[0]], rel=0.01
    )
    static_equity, static_trades = run_static(BARS, PLAYBOOKS["CALM_UP"], CONFIG)
    assert static_equity.index[0] >= TEST_START
    stats = summarize(static_equity, static_trades)
    assert set(stats) == {"sharpe", "max_drawdown", "hit_rate", "t_statistic", "total_return", "trades"}
