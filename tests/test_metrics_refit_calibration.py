"""Step 7a: performance metrics and gates (spec §11), refit label matching and
drift (§12), and probability calibration (§9)."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from regime_trader.calibration import brier_score, reliability_table, state_calibration
from regime_trader.hmm import HmmModel, RegimeModel
from regime_trader.metrics import (
    Gates,
    daily_returns,
    evaluate_gates,
    hit_rate,
    max_drawdown,
    sharpe,
    t_statistic,
)
from regime_trader.refit import DriftConfig, drift_report, match_labels

# --- metrics ------------------------------------------------------------------------


def test_sharpe_and_t_statistic_on_a_known_series() -> None:
    returns = pd.Series([0.01, -0.005, 0.002, 0.004, -0.001])
    mean, sd = returns.mean(), returns.std(ddof=1)
    assert sharpe(returns) == pytest.approx(mean / sd * np.sqrt(252))
    assert t_statistic(returns) == pytest.approx(mean / (sd / np.sqrt(5)))


def test_max_drawdown() -> None:
    equity = pd.Series([100.0, 120.0, 90.0, 110.0, 80.0, 130.0])
    assert max_drawdown(equity) == pytest.approx(1 - 80 / 120)


def test_hit_rate() -> None:
    assert hit_rate([10.0, -5.0, 3.0, 0.0]) == pytest.approx(0.5)  # a zero is not a win
    assert hit_rate([]) == 0.0


def test_daily_returns_use_each_sessions_last_hourly_equity() -> None:
    index = pd.DatetimeIndex(
        [pd.Timestamp(f"2024-01-0{d} {t}", tz="America/New_York") for d in (2, 3) for t in ("10:00", "15:00")]
    )
    equity = pd.Series([100.0, 101.0, 99.0, 103.02], index=index)
    np.testing.assert_allclose(daily_returns(equity).to_numpy(), [103.02 / 101.0 - 1])


def test_gates_pass_only_when_every_check_passes() -> None:
    good = {"sharpe": 1.8, "max_drawdown": 0.10, "hit_rate": 0.58, "t_statistic": 2.5}
    result = evaluate_gates(good, baseline_sharpes={"buy_and_hold": 0.9, "static": 1.1}, gates=Gates())
    assert result.passed
    weak = evaluate_gates({**good, "t_statistic": 1.9}, {"buy_and_hold": 0.9, "static": 1.1}, Gates())
    assert not weak.passed
    assert weak.checks["t_statistic"] is False
    beaten = evaluate_gates(good, {"buy_and_hold": 2.0, "static": 1.1}, Gates())
    assert not beaten.passed
    assert beaten.checks["beats buy_and_hold"] is False


# --- refit ----------------------------------------------------------------------------

BASE = RegimeModel(
    hmm=HmmModel(
        startprob=np.array([0.4, 0.3, 0.3]),
        transmat=np.array([[0.9, 0.08, 0.02], [0.1, 0.85, 0.05], [0.05, 0.15, 0.8]]),
        means=np.array([[0.2, -0.5], [0.0, 0.3], [-0.6, 1.5]]),
        covars=np.array([np.eye(2) * 0.5, np.eye(2), np.eye(2) * 2.0]),
    ),
    labels=("CALM_UP", "CHOP", "CRASH"),
    return_mean=np.array([0.001, 0.0, -0.003]),
    return_vol=np.array([0.002, 0.004, 0.010]),
)


def _permuted(model: RegimeModel, order: list[int], labels: tuple[str, ...]) -> RegimeModel:
    hmm = model.hmm
    return RegimeModel(
        hmm=HmmModel(
            hmm.startprob[order], hmm.transmat[np.ix_(order, order)], hmm.means[order], hmm.covars[order]
        ),
        labels=labels,
        return_mean=model.return_mean[order],
        return_vol=model.return_vol[order],
    )


def test_match_labels_recovers_the_previous_labels_after_a_permuted_refit() -> None:
    # A refit returns the same states in a different order, and their fresh
    # labels may come out differently; matching restores the old names.
    refit = _permuted(BASE, [2, 0, 1], ("STRESS", "CHOP", "CALM_UP"))
    assert match_labels(BASE, refit).labels == ("CRASH", "CALM_UP", "CHOP")


def test_identical_refit_shows_no_drift() -> None:
    report = drift_report(BASE, BASE, live_ll=np.zeros(100), insample_ll=np.zeros(500), config=DriftConfig())
    assert not report.drifted
    assert report.reasons == ()


def test_a_shifted_transition_matrix_is_drift() -> None:
    transmat = BASE.hmm.transmat.copy()
    transmat[0] = [0.7, 0.28, 0.02]  # self-transition moves by 0.2 > 0.15
    shifted = replace(BASE, hmm=replace(BASE.hmm, transmat=transmat))
    report = drift_report(BASE, shifted, np.zeros(100), np.zeros(500), DriftConfig())
    assert report.drifted
    assert any("transition" in reason for reason in report.reasons)


def test_shifted_state_means_are_drift() -> None:
    means = BASE.hmm.means.copy()
    means[1] += [0.0, 1.5]  # 1.5 old standard deviations
    shifted = replace(BASE, hmm=replace(BASE.hmm, means=means))
    report = drift_report(BASE, shifted, np.zeros(100), np.zeros(500), DriftConfig())
    assert report.drifted
    assert any("mean" in reason for reason in report.reasons)


def test_a_drop_in_live_likelihood_is_drift() -> None:
    rng = np.random.default_rng(0)
    insample = rng.normal(0.0, 1.0, 2000)
    live = rng.normal(-2.5, 1.0, 70)
    report = drift_report(BASE, BASE, live_ll=live, insample_ll=insample, config=DriftConfig())
    assert report.drifted
    assert any("likelihood" in reason for reason in report.reasons)


# --- calibration -------------------------------------------------------------------------


def test_brier_score() -> None:
    assert brier_score(np.array([1.0, 0.0]), np.array([1, 0])) == 0.0
    assert brier_score(np.array([0.5, 0.5]), np.array([1, 0])) == pytest.approx(0.25)


def test_reliability_table_groups_predictions_into_bins() -> None:
    probs = np.array([0.05, 0.15, 0.85, 0.95, 0.9])
    outcomes = np.array([0, 0, 1, 1, 0])
    table = reliability_table(probs, outcomes, bins=10)
    top = table[table["bin_low"] == 0.9].iloc[0]
    assert top["count"] == 2
    assert top["observed"] == pytest.approx(0.5)


def test_a_true_model_is_better_calibrated_than_climatology() -> None:
    rng = np.random.default_rng(1)
    hmm = BASE.hmm
    states = [int(rng.choice(3, p=hmm.startprob))]
    for _ in range(2999):
        states.append(int(rng.choice(3, p=hmm.transmat[states[-1]])))
    x = np.array([rng.multivariate_normal(hmm.means[s], hmm.covars[s]) for s in states])
    report = state_calibration(BASE, x)
    assert report.calibrated
    assert all(row.brier < row.climatology_brier for row in report.states)


def test_a_wrong_model_is_flagged_as_uncalibrated() -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(0.0, 1.0, size=(2000, 2))  # data with no regimes at all
    confident = replace(BASE, hmm=replace(BASE.hmm, transmat=np.eye(3) * 0.999 + 0.0005))
    report = state_calibration(confident, x, reference=BASE)
    assert not report.calibrated
