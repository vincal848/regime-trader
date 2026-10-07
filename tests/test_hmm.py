"""Step 3: Gaussian HMM fitting, selection, forward filtering and labels (spec §5, §6)."""

import itertools

import numpy as np
import pytest
from scipy.stats import multivariate_normal

from regime_trader.hmm import (
    HmmModel,
    bic,
    characterize,
    expected_duration,
    fit_hmm,
    forward_filter,
    label_states,
    log_likelihood,
    select_states,
)

PLANTED = HmmModel(
    startprob=np.array([0.5, 0.3, 0.2]),
    transmat=np.array([[0.95, 0.04, 0.01], [0.05, 0.90, 0.05], [0.02, 0.08, 0.90]]),
    means=np.array([[0.5, -1.0], [0.0, 0.0], [-1.5, 2.0]]),
    covars=np.array([np.eye(2) * 0.3, np.eye(2) * 0.5, np.eye(2) * 0.8]),
)


def _sample(model: HmmModel, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    states = np.empty(n, dtype=int)
    states[0] = rng.choice(model.n_states, p=model.startprob)
    for t in range(1, n):
        states[t] = rng.choice(model.n_states, p=model.transmat[states[t - 1]])
    x = np.array([rng.multivariate_normal(model.means[s], model.covars[s]) for s in states])
    return x, states


def test_forward_filter_matches_brute_force_path_enumeration() -> None:
    small = HmmModel(
        startprob=np.array([0.6, 0.4]),
        transmat=np.array([[0.8, 0.2], [0.3, 0.7]]),
        means=np.array([[0.0], [1.5]]),
        covars=np.array([[[1.0]], [[0.5]]]),
    )
    x, _ = _sample(small, 5, seed=1)
    filtered = forward_filter(small, x).filtered
    density = np.array(
        [[multivariate_normal(small.means[k], small.covars[k]).pdf(x[t]) for k in range(2)] for t in range(5)]
    )
    for t in range(5):
        joint = np.zeros(2)
        for path in itertools.product(range(2), repeat=t + 1):
            p = small.startprob[path[0]] * density[0, path[0]]
            for u in range(1, t + 1):
                p *= small.transmat[path[u - 1], path[u]] * density[u, path[u]]
            joint[path[-1]] += p
        np.testing.assert_allclose(filtered[t], joint / joint.sum(), rtol=1e-10)


def test_filtered_rows_are_distributions_and_next_state_is_one_transition_ahead() -> None:
    x, _ = _sample(PLANTED, 300, seed=2)
    result = forward_filter(PLANTED, x)
    np.testing.assert_allclose(result.filtered.sum(axis=1), 1.0)
    np.testing.assert_allclose(result.next_state, result.filtered @ PLANTED.transmat)


def test_filtering_a_prefix_gives_the_same_rows() -> None:
    # Causality of the filter itself: rows up to t do not depend on data after t.
    x, _ = _sample(PLANTED, 200, seed=3)
    full = forward_filter(PLANTED, x).filtered
    np.testing.assert_allclose(forward_filter(PLANTED, x[:120]).filtered, full[:120])


def test_log_likelihood_matches_hmmlearn() -> None:
    from hmmlearn.hmm import GaussianHMM

    x, _ = _sample(PLANTED, 400, seed=4)
    reference = GaussianHMM(n_components=3, covariance_type="full")
    reference.startprob_, reference.transmat_ = PLANTED.startprob, PLANTED.transmat
    reference.means_, reference.covars_ = PLANTED.means, PLANTED.covars
    assert log_likelihood(PLANTED, x) == pytest.approx(reference.score(x), rel=1e-9)


def test_fit_recovers_a_planted_three_state_process() -> None:
    x, states = _sample(PLANTED, 3000, seed=5)
    model = fit_hmm(x, n_states=3, restarts=5, seed=0)
    order = [int(np.argmin(np.linalg.norm(model.means - mean, axis=1))) for mean in PLANTED.means]
    assert sorted(order) == [0, 1, 2]
    np.testing.assert_allclose(model.means[order], PLANTED.means, atol=0.15)
    decoded = forward_filter(model, x).filtered.argmax(axis=1)
    assert np.mean(np.array(order)[states] == decoded) > 0.9


def test_seeded_fits_are_reproducible() -> None:
    x, _ = _sample(PLANTED, 800, seed=6)
    first, second = fit_hmm(x, 3, restarts=3, seed=11), fit_hmm(x, 3, restarts=3, seed=11)
    np.testing.assert_array_equal(first.transmat, second.transmat)


def test_bic_counts_parameters() -> None:
    # K=2, D=2 full covariance: 1 start + 2 transition + 4 means + 6 covariance = 13 parameters.
    assert bic(log_likelihood=-100.0, n_states=2, n_features=2, n_obs=50) == pytest.approx(
        200.0 + 13 * np.log(50)
    )


def test_selection_prefers_the_simpler_model_on_a_near_tie() -> None:
    two_state = HmmModel(
        startprob=np.array([0.5, 0.5]),
        transmat=np.array([[0.97, 0.03], [0.03, 0.97]]),
        means=np.array([[0.0, 0.0], [2.0, -2.0]]),
        covars=np.array([np.eye(2), np.eye(2)]),
    )
    x, _ = _sample(two_state, 2400, seed=7)
    selection = select_states(x[:1600], x[1600:], candidates=(2, 3, 4), restarts=3, seed=0)
    assert selection.n_states == 2
    assert [row.n_states for row in selection.table] == [2, 3, 4]


def test_selection_finds_three_well_separated_states() -> None:
    x, _ = _sample(PLANTED, 3000, seed=8)
    assert select_states(x[:2000], x[2000:], candidates=(2, 3, 4), restarts=3, seed=0).n_states == 3


@pytest.mark.parametrize(
    ("means", "vols", "expected"),
    [
        ([0.001, -0.002, 0.0], [0.002, 0.009, 0.004], ("CALM_UP", "CRASH", "STRESS")),
        ([0.001, -0.0001], [0.002, 0.003], ("CALM_UP", "CHOP")),
        ([0.001, 0.0, -0.001, -0.004], [0.002, 0.003, 0.006, 0.012], ("CALM_UP", "CHOP", "STRESS", "CRASH")),
        ([0.001, 0.0005, -0.001], [0.002, 0.0021, 0.008], ("CALM_UP_1", "CALM_UP_2", "CRASH")),
        ([0.001, 0.002], [0.002, 0.009], ("CALM_UP", "STRESS")),  # highest vol but positive mean: not CRASH
    ],
)
def test_labels_follow_the_spec_rules(
    means: list[float], vols: list[float], expected: tuple[str, ...]
) -> None:
    assert label_states(np.array(means), np.array(vols)) == expected


def test_expected_duration() -> None:
    np.testing.assert_allclose(expected_duration(PLANTED), [20.0, 10.0, 10.0])


def test_characterize_attaches_return_statistics_and_labels() -> None:
    x, states = _sample(PLANTED, 3000, seed=9)
    raw_returns = np.where(states == 2, -0.004, 0.001) + np.random.default_rng(0).normal(0, 0.001, 3000)
    model = characterize(PLANTED, x, raw_returns)
    assert model.labels[2] == "CRASH"
    assert model.return_mean[2] == pytest.approx(-0.004, abs=0.0005)
    assert model.return_vol is not None
