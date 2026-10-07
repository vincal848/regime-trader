"""Are the state probabilities calibrated? Offline evaluation only (spec §9).

The one-bar-ahead state predictions the engine acts on (forward filter,
next-state probabilities) are scored against the states assigned *in
hindsight* by smoothed posteriors. Smoothing uses future bars, which is
exactly why it is confined to this module (tests/test_architecture.py): it
grades decisions, and never makes them.

A state's predictions are calibrated when their Brier score beats the
climatological forecast (always predicting the state's base rate). If any
state fails, the engine's sizing falls back to a fixed quarter of the cap
(`EngineConfig.calibrated = False`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from regime_trader.hmm import RegimeModel, forward_filter

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def brier_score(probabilities: FloatArray, outcomes: npt.NDArray[np.int_]) -> float:
    return float(np.mean((probabilities - outcomes) ** 2))


def reliability_table(
    probabilities: FloatArray, outcomes: npt.NDArray[np.int_], bins: int = 10
) -> pd.DataFrame:
    """Mean predicted against observed frequency, per probability bin (empty bins omitted)."""
    index = np.clip((probabilities * bins).astype(int), 0, bins - 1)
    rows = []
    for b in np.unique(index):
        inside = index == b
        rows.append(
            {
                "bin_low": b / bins,
                "bin_high": (b + 1) / bins,
                "count": int(inside.sum()),
                "predicted": float(probabilities[inside].mean()),
                "observed": float(outcomes[inside].mean()),
            }
        )
    return pd.DataFrame(rows)


def smoothed_states(model: RegimeModel, x: FloatArray) -> IntArray:
    """Most probable state per bar given *all* bars (forward-backward).
    Evaluation only: this looks ahead."""
    from hmmlearn.hmm import GaussianHMM

    hmm = GaussianHMM(n_components=model.hmm.n_states, covariance_type="full")
    hmm.startprob_, hmm.transmat_ = model.hmm.startprob, model.hmm.transmat
    hmm.means_, hmm.covars_ = model.hmm.means, model.hmm.covars
    states: IntArray = np.asarray(hmm.predict_proba(x).argmax(axis=1), dtype=np.int64)
    return states


@dataclass(frozen=True)
class StateCalibration:
    label: str
    brier: float
    climatology_brier: float
    reliability: pd.DataFrame


@dataclass(frozen=True)
class CalibrationReport:
    states: tuple[StateCalibration, ...]
    calibrated: bool


def state_calibration(
    model: RegimeModel, x: FloatArray, reference: RegimeModel | None = None
) -> CalibrationReport:
    """Score `model`'s one-step-ahead predictions against hindsight states
    from `reference` (by default the model itself, e.g. the next refit)."""
    judge = reference if reference is not None else model
    hindsight = np.array(judge.labels)[smoothed_states(judge, x)][1:]  # matched by label, not index
    predicted = forward_filter(model.hmm, x).next_state[:-1]
    states = []
    for k, label in enumerate(model.labels):
        outcomes = (hindsight == label).astype(np.int64)
        base_rate = float(outcomes.mean())
        states.append(
            StateCalibration(
                label=label,
                brier=brier_score(predicted[:, k], outcomes),
                climatology_brier=brier_score(np.full(len(outcomes), base_rate), outcomes),
                reliability=reliability_table(predicted[:, k], outcomes),
            )
        )
    calibrated = all(s.brier < s.climatology_brier for s in states if s.climatology_brier > 0)
    return CalibrationReport(tuple(states), calibrated)
