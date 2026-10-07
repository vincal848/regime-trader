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

### Step 5: playbooks
- Tests first (`tests/test_playbook.py`, 21 tests):
  - typed parsing; unknown keys, unknown features, code-injection
    attempts, `==`, and an out-of-range `max_size` or stop are all
    rejected;
  - exactly one TOML block;
  - entry/exit evaluation, `and` binding tighter than `or`, z-features and
    negative numbers, `never`/`always`;
  - NaN never triggers an entry; `max_hold_bars` forces an exit;
  - all four starting playbooks parse, and CRASH never enters.
- Implementation: `playbook.py` (tomllib plus a regex grammar into a
  disjunction-of-conjunctions `Condition`; no `eval`). Starting playbooks
  `playbooks/{CALM_UP,CHOP,STRESS,CRASH}.md`, written by the research
  layer (Claude).
- Deviation from the spec: stops and take-profits are multiples of the
  existing 21-bar realized volatility (`stop_loss_vol`, `take_profit_vol`)
  instead of ATR. Same idea, no extra feature.

### Step 6: engine
- Tests first:
  - `tests/test_engine.py`: a sized long on CALM_UP entry; flat on
    unhealthy or NaN input; sizing never exceeds the state cap; CRASH never
    enters and flattens; a playbook switch closes the old position; a close
    through the stop exits; the kill switch flattens whatever the model
    says; uncalibrated sizing falls back to a quarter of the cap; **no
    decision depends on a future bar** (the full pipeline rerun after
    perturbing every later bar, at 4 random cut points; probabilities
    compared byte for byte).
  - `tests/test_hmm.py`: the incremental `filter_step` equals the batch
    filter.
- Implementation:
  - `hmm.filter_step` / `initial_prior`. `forward_filter` now runs the same
    one-step `_update`, so one code path produces every probability.
  - `engine.py`: frozen `EngineState`, `OpenPosition`, `Decision`;
    `decide` (filter → switching → playbook → sizing → risk, with hard
    limits first); `on_fill` sets stops and take-profits from the fill
    price and the entry bar's realized volatility.
- Design note: stops are evaluated at bar closes and exit at the next
  open, in backtest and live alike. A gap can go through a stop. That is
  recorded for the go-live risk list.

### Step 7a: metrics, gates, refits, calibration
- Tests first (`tests/test_metrics_refit_calibration.py`, 14 tests):
  - Sharpe, t-statistic, drawdown and hit rate on known series; daily
    returns from each session's last mark;
  - gates fail on any single check, including losing to a baseline;
  - label matching restores names after a permuted refit;
  - drift fires on a transition shift, a mean shift and a live-likelihood
    drop, and stays quiet on an identical refit;
  - Brier and reliability; a true model beats climatology; an
    overconfident model on regime-free data is flagged as uncalibrated.
- Implementation:
  - `metrics.py`: daily-scale statistics, `Gates`, `evaluate_gates`;
  - `refit.py`: Hungarian `match_labels` on z-space means; `drift_report`
    measuring shifts in old-state standard deviations, with live
    log-likelihood against an in-sample rolling percentile;
  - `calibration.py`: the only module allowed to use smoothed posteriors.
- All 14 passed on first implementation.

### Step 7b: walk-forward backtest
- Tests first:
  - `tests/test_backtest.py`, 12 tests on a synthetic two-regime market:
    - fill price and commission math;
    - trading only out of sample;
    - **every fill at the open of the bar after its decision**;
    - long/flat with notional ≤ equity;
    - cash accounting reconciles;
    - refits are spaced at least 30 days apart with stable labels;
    - costs reduce the result;
    - the holdout is locked unless explicitly requested;
    - **refits and equity marks before a cut are unchanged when every
      later bar is perturbed**;
    - per-state attribution covers every bar;
    - baselines and the summary.
  - `tests/test_engine.py`: frozen entries block new positions but allow
    exits.
- Implementation:
  - `backtest.py`: `run_backtest` (K selected once, then monthly
    expanding refits; label matching; drift freezes entries; Kelly from
    training data only), `run_buy_and_hold`, `run_static`, `summarize`, and
    a shared `_Portfolio` for fills, costs and round trips.
  - `engine.py`: `entries_frozen`; `open_position` and `price_exit` were
    extracted so the static baseline reuses the engine's stop logic instead
    of copying it.
- Found while implementing: decisions sized at the close but filled at the
  next open could cost more than the cash available, briefly leveraging
  the account. Execution now caps every buy at what the cash can pay for,
  fees included.
- Test correction: the causality test compares refit parameters to 1e-9,
  not bit-for-bit. Identical runs differ by about 1e-12 because of
  multithreaded BLAS in hmmlearn; a leak would differ by far more.
