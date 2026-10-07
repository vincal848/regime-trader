# Changelog

Each step of docs/PLAN.md is logged as its tests (written first, failing),
then its implementation, then anything the review found.

## [Unreleased]

### Step 1: skeleton, tooling, architecture test
- Tests first: `tests/test_architecture.py` enforces the layer map:
  - no upward imports;
  - `risk` depends on `bars` only;
  - layers 0–2 do no I/O;
  - hmmlearn's smoothed and Viterbi methods only in `calibration`.
- Implementation: `pyproject.toml` (core numpy/pandas/scipy/hmmlearn;
  extras `ibkr`, `llm`, `dashboard`, `demo`, `dev`), CI (ruff, mypy
  `--strict`, pytest with a 90% coverage gate), `.env.example` (names only;
  live port refused by design), `.gitignore` (data, models, journal and
  `.env` never committed).

### Step 2: bars and features
- Tests first (`tests/test_bars_features.py`):
  - seven kinds of bad bar are rejected;
  - bar-of-day works on both the IBKR and Yahoo hourly grids;
  - **no feature at t changes when any later bar changes** (8 random cut
    points, prices and volumes perturbed);
  - z-scores use only statistics through t−1;
  - the volume ratio compares like-for-like hours.
- Implementation: `bars.py` (schema, `validate_bars`, `bar_of_day`) and
  `features.py` (five raw features, expanding z-scores shifted by one bar).
- Test correction: the warm-up expectation was 30 sessions, but the volume
  ratio (20 sessions) followed by its own z-score history (140 bars) needs
  about 41.

### Step 3: Gaussian HMM
- Tests first (`tests/test_hmm.py`):
  - the forward filter equals brute-force path enumeration;
  - rows are distributions, next-state probabilities = filtered · A, and
    filtering a prefix gives the same rows;
  - the log-likelihood equals hmmlearn's forward score;
  - a planted 3-state process is recovered (means within 0.15, more than
    90% state accuracy);
  - BIC parameter count;
  - the one-standard-error rule picks K = 2 on two-state data and K = 3 on
    three-state data;
  - labelling rules, expected duration, filtered-weight characterization.
- Implementation, `hmm.py`:
  - frozen `HmmModel` / `RegimeModel`; a log-space `forward_filter`;
  - `fit_hmm` (seeded, best of N restarts, degenerate runs skipped);
  - `select_states` with BIC reported;
  - `label_states`; `characterize` (state statistics weighted by
    *filtered* probabilities, so no smoothing is needed anywhere outside
    calibration); `expected_duration`.
- Test corrections:
  - Two labelling cases expected intuition rather than the spec's rule. In
    a 3-state model the middle-volatility state is not above the median, so
    it is CHOP. In a 2-state model the higher-volatility losing state is
    CRASH, the risk-first reading. The rule is kept and the tests follow it.
  - Seeded fits agree to about 1e-13, not bit-for-bit, because of
    multithreaded BLAS. The test asserts 1e-10.

### Step 4: switching, sizing, risk
- Tests first (`tests/test_switching_sizing_risk.py`, 24 tests):
  - each switching rule 1–6 on its own, plus no regime before the first
    takeover, interrupted leads, and the cooldown deferring a qualified
    challenger;
  - entropy, Kelly, the size formula, caps and zero cases;
  - state caps by label prefix, with unknown labels capped at zero;
  - long/flat only, the state-cap veto, the manual approval threshold, and
    the daily loss limit allowing only reducing orders;
  - every kill-switch trigger (drawdown, rejects, disconnect, manual) vetoes
    new risk but approves flattening.
- Implementation:
  - `switching.py`: frozen `SwitchState`, `step_switch` returning a
    `Regime` with the reasons for every rule that fired;
  - `sizing.py`: ¼ Kelly × P(active) × (1 − H / ln K) × multiplier, capped;
  - `risk.py`: `RiskLimits`, `AccountState`, `check_order` →
    `Approved | Vetoed`, and `kill_reasons`. It imports nothing from the
    package; flattening never waits for approval.
- All 24 passed on first implementation. The only fix was typing the
  test helpers without `type: ignore`.
