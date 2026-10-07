"""Step 2: bar validation and causal features (spec §3, §4)."""

import numpy as np
import pandas as pd
import pytest
from synthetic import make_bars

from regime_trader.bars import BarError, bar_of_day, validate_bars
from regime_trader.features import FEATURES, Z_FEATURES, compute_features

# --- bars -------------------------------------------------------------------------


def test_valid_bars_pass() -> None:
    validate_bars(make_bars(5))


@pytest.mark.parametrize(
    ("corrupt", "message"),
    [
        (lambda b: b.assign(high=b["close"] * 0.99), "high"),
        (lambda b: b.assign(low=b["close"] * 1.01), "low"),
        (lambda b: b.assign(volume=-1.0), "volume"),
        (lambda b: b.iloc[[0, 2, 1, 3]], "increasing"),
        (lambda b: b.assign(close=np.nan), "missing"),
        (lambda b: b.tz_localize(None), "timezone"),
        (lambda b: b.set_axis(b.index + pd.Timedelta(hours=7)), "regular trading hours"),
    ],
)
def test_bad_bars_are_rejected(corrupt: object, message: str) -> None:
    with pytest.raises(BarError, match=message):
        validate_bars(corrupt(make_bars(3)))  # type: ignore[operator]


def test_bar_of_day_counts_position_within_each_session() -> None:
    bars = make_bars(3)
    assert list(bar_of_day(bars.index)) == [0, 1, 2, 3, 4, 5, 6] * 3


def test_bar_of_day_works_on_the_yahoo_half_hour_grid() -> None:
    index = pd.DatetimeIndex(
        [pd.Timestamp(f"2024-01-02 {t}", tz="America/New_York") for t in ("09:30", "10:30", "11:30", "15:30")]
    )
    assert list(bar_of_day(index)) == [0, 1, 2, 3]


# --- features ---------------------------------------------------------------------


def test_feature_columns() -> None:
    frame = compute_features(make_bars(40))
    assert list(frame.columns) == [*FEATURES, *Z_FEATURES]
    assert FEATURES == ("ret", "rv", "range", "volume_ratio", "trend")


def test_features_become_finite_after_the_warm_up() -> None:
    frame = compute_features(make_bars(60))
    assert frame.iloc[: 7 * 20].isna().any(axis=None)
    assert np.isfinite(frame.iloc[7 * 30 :].to_numpy()).all()


def test_no_feature_at_t_depends_on_any_later_bar() -> None:
    bars = make_bars(60, seed=1)
    base = compute_features(bars)
    rng = np.random.default_rng(2)
    for t in rng.integers(150, len(bars) - 5, size=8):
        future = bars.copy()
        scale = np.exp(rng.normal(0.0, 0.05, len(bars) - t - 1))
        for column in ("open", "high", "low", "close"):
            future.iloc[t + 1 :, future.columns.get_loc(column)] *= scale
        future.iloc[t + 1 :, future.columns.get_loc("volume")] *= 3.0
        perturbed = compute_features(future)
        pd.testing.assert_frame_equal(perturbed.iloc[: t + 1], base.iloc[: t + 1])


def test_z_scores_use_only_statistics_through_the_previous_bar() -> None:
    frame = compute_features(make_bars(60, seed=3))
    raw = frame["ret"].to_numpy()
    for t in (300, 350, 400):
        past = raw[:t]
        past = past[np.isfinite(past)]
        expected = (raw[t] - past.mean()) / past.std(ddof=1)
        assert frame["z_ret"].iloc[t] == pytest.approx(expected)


def test_volume_ratio_compares_with_the_same_bar_of_day() -> None:
    bars = make_bars(30, seed=4)
    shape = np.tile([3.0, 1.5, 1.0, 0.8, 0.9, 1.2, 2.5], 30)
    bars["volume"] = 1e6 * shape  # pure intraday shape: every bar is typical for its hour
    ratio = compute_features(bars)["volume_ratio"].iloc[7 * 21 :]
    np.testing.assert_allclose(ratio, 1.0)
