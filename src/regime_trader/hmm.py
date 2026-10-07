"""Gaussian HMM: fitting, state selection, forward filtering and labels (spec §5, §6).

Decisions use only `forward_filter`, our own forward recursion over the
fitted parameters:

    filtered_t   = P(s_t | x_1..x_t)
    next_state_t = P(s_{t+1} | x_1..x_t) = filtered_t @ A

hmmlearn is used only to *fit* parameters. Its smoothed and Viterbi methods
(`predict_proba`, `predict`, `decode`, `score_samples`) use future
observations, and tests/test_architecture.py bans them outside
`calibration`. hmmlearn's `score` is the forward algorithm, so it is
look-ahead free. Even so, likelihoods here come from `forward_filter` too,
so a single code path produces every number the system acts on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class HmmModel:
    """Parameters of a K-state Gaussian HMM with full covariances."""

    startprob: FloatArray  # (K,)
    transmat: FloatArray  # (K, K), rows sum to 1
    means: FloatArray  # (K, D)
    covars: FloatArray  # (K, D, D)

    @property
    def n_states(self) -> int:
        return int(self.startprob.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.means.shape[1])


@dataclass(frozen=True)
class RegimeModel:
    """A fitted HMM with each state's raw-return statistics and its label."""

    hmm: HmmModel
    labels: tuple[str, ...]
    return_mean: FloatArray  # per state, raw hourly log return
    return_vol: FloatArray


@dataclass(frozen=True)
class FilterResult:
    filtered: FloatArray  # (T, K) P(s_t | x_..t)
    next_state: FloatArray  # (T, K) P(s_{t+1} | x_..t)
    log_likelihood: FloatArray  # (T,) log p(x_t | x_..t-1)


def _emission_log_density(model: HmmModel, x: FloatArray) -> FloatArray:
    return np.column_stack(
        [
            multivariate_normal(model.means[k], model.covars[k], allow_singular=True).logpdf(x)
            for k in range(model.n_states)
        ]
    ).reshape(len(x), model.n_states)


@dataclass(frozen=True)
class FilterStep:
    filtered: FloatArray  # (K,) P(s_t | x_..t)
    next_state: FloatArray  # (K,) P(s_{t+1} | x_..t): the prior for the next bar
    log_likelihood: float  # log p(x_t | x_..t-1)


def initial_prior(model: HmmModel) -> FloatArray:
    """The prior for the first bar: the start probabilities."""
    prior: FloatArray = model.startprob.copy()
    return prior


def _update(model: HmmModel, prior: FloatArray, log_density: FloatArray) -> FilterStep:
    """One forward step in log space: Bayes on the emission, then one transition."""
    with np.errstate(divide="ignore"):  # a zero probability is a legitimate -inf in log space
        joint = np.log(prior) + log_density
    step_ll = float(logsumexp(joint))
    filtered = np.exp(joint - step_ll)
    return FilterStep(filtered, filtered @ model.transmat, step_ll)


def filter_step(model: HmmModel, prior: FloatArray, x: FloatArray) -> FilterStep:
    """Advance the filter by one observation `x` (D,). Feeding each step's
    `next_state` back in as `prior` reproduces `forward_filter` exactly."""
    if not np.isfinite(x).all():
        raise ValueError("filter_step needs a finite observation")
    return _update(model, prior, _emission_log_density(model, x[np.newaxis, :])[0])


def forward_filter(model: HmmModel, x: FloatArray) -> FilterResult:
    """Filtered and next-state probabilities for every row of `x` (T, D).

    Row t uses rows 0..t only. This is `filter_step` run over the rows, with
    the emission densities computed in one vectorized pass.
    """
    if not np.isfinite(x).all():
        raise ValueError("forward_filter needs finite observations; drop the warm-up rows first")
    log_density = _emission_log_density(model, x)
    filtered = np.empty((len(x), model.n_states))
    next_state = np.empty((len(x), model.n_states))
    step_ll = np.empty(len(x))
    prior = initial_prior(model)
    for t in range(len(x)):
        step = _update(model, prior, log_density[t])
        filtered[t], next_state[t], step_ll[t] = step.filtered, step.next_state, step.log_likelihood
        prior = step.next_state
    return FilterResult(filtered, next_state, step_ll)


def log_likelihood(model: HmmModel, x: FloatArray) -> float:
    return float(forward_filter(model, x).log_likelihood.sum())


def n_parameters(n_states: int, n_features: int) -> int:
    """Free parameters: start probabilities, transitions, means, full covariances."""
    covariance = n_features * (n_features + 1) // 2
    return (n_states - 1) + n_states * (n_states - 1) + n_states * n_features + n_states * covariance


def bic(log_likelihood: float, n_states: int, n_features: int, n_obs: int) -> float:
    return -2.0 * log_likelihood + n_parameters(n_states, n_features) * float(np.log(n_obs))


def fit_hmm(x: FloatArray, n_states: int, restarts: int, seed: int) -> HmmModel:
    """Best of `restarts` EM fits (by training log-likelihood), seeded."""
    from hmmlearn.hmm import GaussianHMM

    best: HmmModel | None = None
    best_ll = -np.inf
    for restart in range(restarts):
        candidate = GaussianHMM(
            n_components=n_states, covariance_type="full", n_iter=200, tol=1e-4, random_state=seed + restart
        )
        candidate.fit(x)
        model = HmmModel(
            np.asarray(candidate.startprob_, dtype=np.float64),
            np.asarray(candidate.transmat_, dtype=np.float64),
            np.asarray(candidate.means_, dtype=np.float64),
            np.asarray(candidate.covars_, dtype=np.float64),
        )
        if not np.isfinite(model.transmat).all() or np.any(model.transmat.sum(axis=1) == 0):
            continue  # a degenerate EM run (an empty state); skip it
        ll = log_likelihood(model, x)
        if ll > best_ll:
            best, best_ll = model, ll
    if best is None:
        raise RuntimeError(f"every one of {restarts} HMM restarts with {n_states} states degenerated")
    return best


@dataclass(frozen=True)
class SelectionRow:
    n_states: int
    bic: float
    oos_ll_per_bar: float
    oos_ll_se: float


@dataclass(frozen=True)
class Selection:
    n_states: int
    model: HmmModel
    table: tuple[SelectionRow, ...]


def select_states(
    train: FloatArray,
    valid: FloatArray,
    candidates: tuple[int, ...] = (2, 3, 4, 5),
    restarts: int = 50,
    seed: int = 0,
) -> Selection:
    """Choose K by out-of-sample log-likelihood with the one-standard-error
    rule: the smallest K whose validation score is within one standard error
    of the best. BIC is reported alongside it.

    The validation series is filtered after the training series, so the
    first validation bar is scored with the state belief the training data
    leaves behind, exactly as the model will run live.
    """
    fits: dict[int, HmmModel] = {}
    rows = []
    for k in candidates:
        model = fit_hmm(train, k, restarts, seed)
        fits[k] = model
        step_ll = forward_filter(model, np.vstack([train, valid])).log_likelihood[len(train) :]
        rows.append(
            SelectionRow(
                n_states=k,
                bic=bic(log_likelihood(model, train), k, train.shape[1], len(train)),
                oos_ll_per_bar=float(step_ll.mean()),
                oos_ll_se=float(step_ll.std(ddof=1) / np.sqrt(len(step_ll))),
            )
        )
    best = max(rows, key=lambda row: row.oos_ll_per_bar)
    chosen = min(row.n_states for row in rows if row.oos_ll_per_bar >= best.oos_ll_per_bar - best.oos_ll_se)
    return Selection(chosen, fits[chosen], tuple(rows))


def label_states(return_mean: FloatArray, return_vol: FloatArray) -> tuple[str, ...]:
    """Labels from each state's mean return and volatility (spec §5), in order:

    CRASH    the highest-volatility state, if its mean return is negative
    STRESS   volatility above the median (and not CRASH)
    CALM_UP  volatility at or below the median, and a positive mean
    CHOP     everything else
    Repeated labels get suffixes _1, _2, ... in state order.
    """
    median_vol = float(np.median(return_vol))
    highest = int(np.argmax(return_vol))
    names = []
    for k, (mean, vol) in enumerate(zip(return_mean, return_vol, strict=True)):
        if k == highest and mean < 0:
            names.append("CRASH")
        elif vol > median_vol:
            names.append("STRESS")
        elif mean > 0:
            names.append("CALM_UP")
        else:
            names.append("CHOP")
    repeats = {name: names.count(name) for name in names}
    seen: dict[str, int] = {}
    labelled = []
    for name in names:
        if repeats[name] > 1:
            seen[name] = seen.get(name, 0) + 1
            labelled.append(f"{name}_{seen[name]}")
        else:
            labelled.append(name)
    return tuple(labelled)


def characterize(model: HmmModel, x: FloatArray, raw_returns: FloatArray) -> RegimeModel:
    """Attach each state's raw-return mean and volatility, weighting training
    bars by their *filtered* state probabilities, plus the resulting labels."""
    weights = forward_filter(model, x).filtered
    totals = weights.sum(axis=0)
    mean = weights.T @ raw_returns / totals
    variance = (weights * (raw_returns[:, None] - mean[None, :]) ** 2).sum(axis=0) / totals
    vol = np.sqrt(variance)
    return RegimeModel(model, label_states(mean, vol), mean, vol)


def expected_duration(model: HmmModel) -> FloatArray:
    """Expected bars spent in each state per visit: 1 / (1 - A_ii)."""
    duration: FloatArray = 1.0 / (1.0 - np.diag(model.transmat))
    return duration
