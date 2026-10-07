"""Walk-forward refits: stable state labels and drift detection (spec §12).

**Matching.** A refit returns its states in arbitrary order. Each new state
takes the label of the previous state nearest to it in feature space (the
HMM means, in z-units), by minimum-cost assignment (Hungarian). A new state
left unmatched, when K grew, keeps its own fresh label.

**Drift** compares the matched previous and new models, plus live fit
quality:
- any transition probability between matched states moves by more than
  `max_transition_shift`;
- any matched state's mean moves by more than `max_mean_shift_sd` of that
  state's previous per-feature standard deviations;
- the mean live log-likelihood over the last `ll_window` bars falls below the
  `ll_percentile`-th percentile of in-sample rolling means.

Any of these freezes new entries and raises an alert. Exits stay allowed.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment

from regime_trader.hmm import RegimeModel

FloatArray = npt.NDArray[np.float64]


def match_labels(previous: RegimeModel, new: RegimeModel) -> RegimeModel:
    cost = np.linalg.norm(new.hmm.means[:, None, :] - previous.hmm.means[None, :, :], axis=2)
    rows, cols = linear_sum_assignment(cost)
    labels = list(new.labels)
    for new_state, old_state in zip(rows, cols, strict=True):
        labels[new_state] = previous.labels[old_state]
    taken = {previous.labels[c] for c in cols}
    for state in set(range(new.hmm.n_states)) - set(rows.tolist()):
        if labels[state] in taken:
            labels[state] = f"{labels[state]}_NEW"
    return replace(new, labels=tuple(labels))


@dataclass(frozen=True)
class DriftConfig:
    max_transition_shift: float = 0.15
    max_mean_shift_sd: float = 1.0
    ll_window: int = 35
    ll_percentile: float = 1.0


@dataclass(frozen=True)
class DriftReport:
    transition_shift: float
    mean_shift_sd: float
    live_ll_mean: float  # NaN until a full window of live bars exists
    ll_threshold: float
    drifted: bool
    reasons: tuple[str, ...]


def drift_report(
    previous: RegimeModel, new: RegimeModel, live_ll: FloatArray, insample_ll: FloatArray, config: DriftConfig
) -> DriftReport:
    """`new` must already carry matched labels (`match_labels`)."""
    common = [label for label in new.labels if label in previous.labels]
    old_index = [previous.labels.index(label) for label in common]
    new_index = [new.labels.index(label) for label in common]

    old_a = previous.hmm.transmat[np.ix_(old_index, old_index)]
    new_a = new.hmm.transmat[np.ix_(new_index, new_index)]
    transition_shift = float(np.abs(new_a - old_a).max()) if common else 0.0

    old_sd = np.sqrt(np.diagonal(previous.hmm.covars[old_index], axis1=1, axis2=2))
    mean_shift = np.abs(new.hmm.means[new_index] - previous.hmm.means[old_index]) / old_sd
    mean_shift_sd = float(mean_shift.max()) if common else 0.0

    window = config.ll_window
    rolling_insample = np.convolve(insample_ll, np.ones(window) / window, mode="valid")
    threshold = (
        float(np.percentile(rolling_insample, config.ll_percentile)) if len(rolling_insample) else -np.inf
    )
    live_mean = float(np.mean(live_ll[-window:])) if len(live_ll) >= window else float("nan")

    reasons = []
    if transition_shift > config.max_transition_shift:
        reasons.append(f"transition probability shifted by {transition_shift:.3f}")
    if mean_shift_sd > config.max_mean_shift_sd:
        reasons.append(f"state mean shifted by {mean_shift_sd:.2f} standard deviations")
    if np.isfinite(live_mean) and live_mean < threshold:
        percentile = f"p{config.ll_percentile:g}"
        reasons.append(
            f"live log-likelihood {live_mean:.3f} below the in-sample {percentile} {threshold:.3f}"
        )
    return DriftReport(transition_shift, mean_shift_sd, live_mean, threshold, bool(reasons), tuple(reasons))
