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

### Step 8: adapters (store, IBKR, Yahoo, Telegram, nightly LLM)
- Tests first (`tests/test_adapters.py`):
  - **Bar cache:** saves merge, with the later bar winning.
  - **Journal:** round-trips decisions (probabilities by label), fills and
    events.
  - **Model:** JSON round-trip.
  - **Playbooks:** a playbook whose file name doesn't match its state is
    rejected.
  - **IBKR, paper only:**
    - non-`DU` accounts and the live ports 4001/7496 are refused;
    - the account ID is masked in repr;
    - after connecting, a session that can see a live account disconnects
      and raises;
    - hourly history arrives in New York time;
    - orders are marketable limits 5 bps through the price, rounded to the
      cent, and nothing is sent when already on target.
  - **Yahoo:** multi-level columns are normalized and bars filtered to
    regular hours.
  - **Telegram:**
    - outbound only;
    - `send` is the only public method;
    - unknown alert kinds are rejected;
    - the token is redacted from repr and from errors.
  - **Nightly LLM:**
    - priced at the Opus 5.5 rates, and unknown (fallback) models at the
      highest known rate;
    - the call is refused *before* it is made when the worst case exceeds
      the monthly budget;
    - a refusal still records its cost.
- Implementation:
  - `store.py`: Parquet bar cache and SQLite journal.
  - `ibkr.py`: typed `IbClient` protocol over ib_async; the paper guard
    runs both before and after connecting.
  - `yahoo.py`
  - `alerts.py`: stdlib urllib.
  - `llm.py`: streamed `claude-opus-5-5`, adaptive thinking, effort high,
    server-side refusal fallbacks, and a JSON spend ledger written
    atomically.
- `Fill` moved from `backtest` to `engine`: live trading produces fills too,
  and the live app must not import from the research layer.
- Review: the exceptions were renamed `BudgetExceededError` and
  `ReviewRefusedError` (N818). The camelCase lint rule is relaxed only for
  the IB protocol and its test fake, which mirror IB's API names.

### Step 9: apps (live, nightly, dashboard, CLI) and the fit bundle
- Tests first (`tests/test_apps.py`, plus additions to `test_adapters.py`
  and `test_switching_sizing_risk.py`):
  - **Live trader**, against a fake broker and an injected clock:
    - each bar is decided and journaled exactly once;
    - stale data flattens a position and alerts, and while flat it is only
      journaled;
    - an exception flattens and alerts;
    - the drawdown kill switch flattens, and the trader stays flat after
      equity recovers;
    - a manual kill flattens;
    - orders over $25,000 wait until an approval window is open;
    - more than 300 s disconnected kills;
    - unfilled orders are cancelled, and three rejects kill;
    - fills are journaled and alerted;
    - a restart resumes from the checkpoint without deciding a bar twice;
    - failed alerts never stop trading;
    - an implausible equity read is refused;
    - live-likelihood drift freezes entries;
    - the watchdog spots a missed bar, but only during the session.
  - **Scheduling:** the IBKR bar-close grid and the wake loop.
  - **Nightly:**
    - proposed playbooks are parsed with the real grammar, so an injected
      expression or a mislabelled state is rejected;
    - valid proposals are backtested against the gates and filed under
      `proposals/DAY/`;
    - `playbooks/` is byte-for-byte unchanged afterwards;
    - an exhausted budget skips the review and alerts;
    - the daily report has every section in spec §14;
    - the record handed to Claude is wrapped in `<record>` tags as data.
  - **Dashboard:** its data comes from the journal and the checkpoint.
  - **CLI:**
    - kill and reset;
    - an approval window;
    - fit, then a refit with a drift line and the previous fit archived;
    - a backtest that prints the gates;
    - a clear error when the holdout leaves no test period;
    - the daily report;
    - a live account is refused with exit code 2;
    - `.env` is parsed without echoing it.
- Implementation:
  - `refit.fit_regime` builds the **fit bundle** (model, per-state Kelly,
    in-sample log-likelihoods, next-bar prior), which the backtest now
    uses too, so live and research fit identically. `likelihood_alarm` is
    split out of `drift_report` for the live check.
  - `backtest.acceptance`, `baseline_sharpes` and `acceptance_report`: one
    scoring path for the CLI and the nightly loop. The static baseline is
    chosen on the training period only (spec §11).
  - `features.healthy`: one definition of a usable row, replacing three
    copies.
  - `live.py`: `Trader`, the `Control` folder (sticky `KILL`,
    `APPROVED_UNTIL`), the JSON checkpoint, `next_bar_close`, `run` and
    `missed_bar`.
  - `nightly.py`, `dashboard.py` (Streamlit, read-only), and `cli.py`.
  - Adapters:
    - `ibkr.recent`, `fill_report` and `cancel`;
    - `store.save_fit` and `load_fit`;
    - an `approval` alert kind;
    - the reviewer's prompt now asks for proposals in tagged, grammar-only
      form.
- Fix (root cause, in `risk.check_order`): reducing orders no longer need
  manual approval. Before, an exit over $25,000 could have waited for you.
  The test was added alongside the fix.
- Test harness correction: IB returns the bar that is still forming, so the
  trader uses only completed bars and runs at each bar's close. The rig now
  steps on bar closes rather than 30 minutes into each bar.

### Fix: numbered states never traded (found by the demo run)
- **Symptom.** The first Yahoo demo walk-forward made zero trades in 21
  months.
- **Root cause.** When the model finds two states of the same kind, the
  labeller numbers them (`CALM_UP_1`, `CALM_UP_2`; spec §5). State caps
  already matched on the base name, but playbook lookups, the per-state
  Kelly estimate, the live trader and the engine's "same playbook?" check
  all used the exact label. So every numbered state was flat, and a switch
  between siblings would have closed the position. The synthetic tests
  never produced numbered states, so nothing caught it.
- Tests first:
  - a numbered state trades its base playbook;
  - a switch between sibling states keeps the position;
  - Kelly for a numbered state uses its base playbook;
  - `base_state` strips only the numbering, so `CALM_UP_NEW`, an unmatched
    state after a refit, stays unknown and capped at zero.
- Fix: one rule, `risk.base_state`, and one lookup, `engine.playbook_for`,
  used everywhere a label meets a playbook or a cap.

### Fix: Yahoo and IB both include the bar still forming
- `yahoo.normalize_yahoo(raw, now=...)` drops a bar that has not finished
  its hour. The test was written first and failed first. The live trader
  already used only completed bars.

### Calibration gates sizing (spec §9)
- **Gap.** Spec §9 says sizing uses the model's probabilities only once
  they are shown to be calibrated, and otherwise falls back to a fixed
  quarter of the state cap. Nothing wired this: the backtest and the live
  trader always assumed the model was calibrated.
- Tests first:
  - calibration matches hindsight states by label, not index, so a refit
    that orders states differently is scored correctly (before, this
    mis-scored silently);
  - a first fit is not calibrated, and a refit scores it;
  - an uncalibrated fit sizes at no more than a quarter of the cap, live
    and in the backtest (the first walk-forward window);
  - the fit bundle round-trips its verdict.
- Implementation:
  - `Fit.calibrated`;
  - `fit_regime(..., previous: Fit)` scores the previous fit's one-bar-ahead
    predictions over the bars since it was trained, against the new fit's
    hindsight states, and keeps the previous verdict when fewer than ten
    sessions are new;
  - the backtest and live AND the verdict into `EngineConfig.calibrated`;
  - `regime-trader fit` prints which sizing applies.

### Step 10: demo run and docs
- **Demo walk-forward** on Yahoo hourly SPY, 2025-01 to 2026-10 (no
  holdout; a demo, not acceptance). Sharpe −1.16 against buy-and-hold's
  1.08: the gates fail, honestly reported in `docs/DEMO.md`. It is
  reproducible with `scripts/demo_walkforward.py`. The first demo run is
  also what exposed the numbered-state defect fixed above.
- `docs/GO_LIVE.md`:
  - the spec §15 checklist with current evidence; the verdict is "not
    ready";
  - **WHAT COULD BLOW UP THIS ACCOUNT?**: each failure mode with its
    mitigation in code and the residual risk.
- `docs/DEPLOY_WINDOWS.md`: IB Gateway under IBC, the NSSM service, Task
  Scheduler jobs (nightly, watchdog, monthly refit), power, update and
  auto-login settings, and the daily commands.
- README expanded; `docs/ARCHITECTURE.md` brought in line with the build:
  the engine signature, the fit bundle, the live loop, and the files on
  disk.
- The CLI backtest header now says "no locked holdout: demo only" for
  `--no-holdout` runs.
