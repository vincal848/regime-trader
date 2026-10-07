# Regime Trader: Specification (v0.2, decisions recorded; awaiting approval)

An hourly SPY strategy that trades a different playbook for each hidden
market regime. A Gaussian Hidden Markov Model reads the market state, Claude
(Opus 5.5) researches and writes the playbooks, and deterministic code makes
every decision and enforces every limit.

**Status:** draft for approval. Nothing is built until this spec, then
`docs/ARCHITECTURE.md`, then `docs/PLAN.md` are each approved. Values marked
*default* are proposals you can still change; decisions you have made are
marked *decided*. Paper trading only: this build is to see the full system
working, and live money stays out of scope.

---

## 1. Scope

| Item | Choice |
|---|---|
| Venue | Interactive Brokers **paper** account, through IB Gateway and the official TWS API (`ib_async`) |
| Instrument | SPY (ARCA ETF), regular trading hours only (09:30–16:00 ET) |
| Candles | 1 hour, RTH-aligned: 09:30–10:00 (a half bar), then 10:00 … 15:00–16:00, so 7 bars a day |
| Decision time | At each bar's close, using only that bar and earlier ones |
| Execution | The order goes out at the next bar's open, so a fill can never use the bar that produced the signal |
| Direction | Long/flat only (decided). CRASH and STRESS mean flat or reduced, never short. Shorts would be a later, separately gated change |
| Live money | Out of scope until every check in §13 is clean and you approve it explicitly |

## 2. Three layers that never overlap

| Layer | Who | Runs | Owns | Never does |
|---|---|---|---|---|
| 1. Research | Claude (Opus 5.5) | Nightly, offline | Feature ideas, `playbooks/<STATE>.md`, `strategy.md` drafts, loss post-mortems | Places orders, changes limits, grades its own output, ships without the gates |
| 2. State | Gaussian HMM | Every bar | Filtered state probabilities P(s_t), next-state probabilities P(s_{t+1}) | Picks positions or sizes |
| 3. Decision | Deterministic code | Every bar | Thresholds, switching, sizing, every risk veto, every order | Defers a limit to a model |

The models advise; the code decides. Playbooks are parsed by code into
typed, range-checked parameters (§7). A playbook that asks for more than a
hard limit is clipped to the limit, and the clip is logged.

## 3. Data

- **Source.** IBKR historical bars (`TRADES`, RTH only, 1 hour), cached
  locally as Parquet. The first download respects IBKR's pacing limits;
  later runs append to the cache.
- **History.** From 2018-01-02, so the data covers several regimes: the
  2018 Q4 sell-off, the 2020 crash, the 2022 bear market and the 2025
  drawdown. The fitting window starts with at least 3 years.
- **Live.** 1-hour bars, built from IBKR real-time bars and checked against
  the historical bar once it finalizes. This needs a US equity real-time
  data subscription (a few dollars a month).
- **Staleness.** If no bar has arrived 10 minutes after a bar's scheduled
  close during RTH, the data is *stale*, and the system goes flat (§8 rule 6).
- **Quality checks** on every bar: high ≥ max(open, close), low ≤ min,
  volume ≥ 0, and a timestamp that is strictly increasing and on the RTH
  grid. A failed check counts as stale.

## 4. Features (deterministic, past-only)

Five numbers per bar t, each computed only from bars ≤ t:

| # | Feature | Definition |
|---|---|---|
| 1 | Log return | ln(close_t / close_{t-1}) (the first bar of the day includes the overnight gap) |
| 2 | Realized volatility | standard deviation of the last 21 log returns (about 3 days) |
| 3 | Range | ln(high_t / low_t) |
| 4 | Volume ratio | volume_t / median volume of the *same bar of the day* over the previous 20 days (removes the intraday U-shape) |
| 5 | Trend | (ln close_t − its 70-bar EMA) / realized volatility (about a 10-day trend, in volatility units) |

**Standardization.** Each feature becomes a z-score using the mean and
standard deviation of an expanding window that ends at bar t−1. The
statistics never include bar t itself.

**Test (written first).** Changing any bar after t changes no feature, z-score
or decision at or before t (§6).

## 5. Hidden Markov Model

- **Model.** `hmmlearn.GaussianHMM` with full covariance, over the five
  z-scored features.
- **Number of states, 2–5.** Each K gets BIC on the fitting window and an
  out-of-sample log-likelihood per bar on the following 6 months. Choose
  the smallest K whose out-of-sample score is within one standard error of
  the best (*the simpler model when scores are close*).
- **Fitting.** A fixed base seed (`20260107`) and 50 random restarts per K.
  Keep the restart with the highest training log-likelihood.
- **Report per fit:** the transition matrix; each state's mean return and
  volatility (from the raw, un-standardized feature 1) and its expected
  duration, 1 / (1 − A_ii) bars.
- **Automatic labels** from each state's return mean μ and volatility σ:

  | Label | Rule (applied in this order) |
  |---|---|
  | CRASH | highest σ, and μ < 0 |
  | STRESS | σ above the median σ (and not CRASH) |
  | CALM_UP | σ at or below the median, and μ > 0 |
  | CHOP | anything else |

  With K < 4 some labels go unused. Two states with the same label get a
  numeric suffix (CHOP_1, CHOP_2).

## 6. No look-ahead (the critical rule)

- **Decisions use filtered probabilities only:** P(s_t | x_1..x_t), from
  our own forward recursion over the fitted parameters. The next-state
  probabilities are P(s_{t+1} | x_1..x_t) = P(s_t | ·) · A.
- **Never Viterbi, never smoothed posteriors.** Do *not* use hmmlearn's
  `predict` (Viterbi) or `predict_proba` for decisions. `predict_proba`
  returns **smoothed** posteriors computed with forward-backward, which
  uses future bars. They are allowed only in offline evaluation
  (calibration, §9), and an architecture test forbids importing them in
  decision code.
- **Tests written first:**
  1. The forward filter matches a brute-force computation on a small
     example.
  2. **The future-perturbation test.** For many random t, randomize every
     bar after t, rerun the whole pipeline, and assert that every feature,
     probability, playbook switch, size and order at or before t is
     bit-for-bit unchanged.
  3. The backtest fills orders at the open of the bar after the signal,
     never at or before the signal bar's close.

## 7. Playbooks (one per state)

`playbooks/<STATE>.md` is written by Claude. It holds prose (rationale and
invalidation) plus one fenced block of parameters that the code parses and
validates:

```toml
[entry]        # e.g. trend > 0.5 and return > 0
[exit]         # e.g. trend < 0
stop_loss_atr = 2.0          # stop, in multiples of the 21-bar ATR
take_profit_atr = 4.0
max_size = 0.8               # fraction of equity; clipped to the state cap (§10)
invalidation = "..."         # the exact condition that ends this playbook's validity
```

Starting playbooks:

| State | Playbook |
|---|---|
| CALM_UP | Trend-following: long while trend > threshold; exit on a trend cross-down, the stop or the take-profit |
| CHOP | Mean reversion: long when the return z-score < −k; exit back at the mean, the stop, or after N bars |
| STRESS | Reduced size (half the cap) on CALM_UP-style entries only, or stand aside (chosen by backtest) |
| CRASH | Flat; no entries; existing positions closed at the next open |

Entry and exit conditions are restricted to a small, whitelisted expression
grammar over the five features and the position state. No code from a
playbook is ever executed. Claude writes and revises playbooks; the backtest
gates (§11), run by the harness, grade them.

## 8. Switching rules (code)

| # | Rule | *Default* |
|---|---|---|
| 1 | A new state takes over only above this filtered probability | 0.70 |
| 2 | … and only after keeping the lead this many consecutive bars | 3 |
| 3 | Cooldown after any switch | 7 bars (one trading day) |
| 4 | If P(next bar in STRESS or CRASH) exceeds this, cut size early | > 0.25, cut to half |
| 5 | If the top two filtered probabilities are this close, treat as uncertain: size zero | within 0.15 |
| 6 | On a model error, NaN, stale data or failed quality check | go flat |

Every switch and every rule firing is logged with the full probability
vector, the rule and the reason.

## 9. Sizing and calibration

- **Size** = min(state leverage cap, ¼ Kelly) × P(active state) ×
  (1 − H / ln K), where H is the entropy of the filtered probability vector.
  It is zero in CRASH and in uncertain states.
- **Kelly** per playbook, from the mean and variance of its training-window
  bar returns. It is re-estimated at each refit and never uses test data.
- **Calibration before sizing goes live.** For each state, compare the
  one-bar-ahead predicted probabilities with the states assigned in
  hindsight by the next refit's smoothed posterior. This is offline
  evaluation only, so the look-ahead is legitimate here. Report a Brier
  score and a reliability curve per state. In paper trading, do the same
  on your own fills: the predicted trade win probability against the
  realized outcome. If calibration is poor (Brier worse than the
  climatological baseline), sizing falls back to a fixed ¼ of the cap.

## 10. Hard risk limits (code; checked before every order; no model can change them)

| Limit | *Default* |
|---|---|
| Max position (notional / equity) | 1.0, no leverage |
| State leverage caps | CALM_UP 1.0 · CHOP 0.5 · STRESS 0.25 · CRASH 0 |
| Daily loss limit | −2% of start-of-day equity: flatten and stop entries until the next session |
| Max drawdown | −10% from the equity peak: **kill switch** |
| Kill switch | Flatten everything, cancel all orders, halt trading until you reset it by hand. Triggered by the drawdown, three consecutive order rejects, broker disconnect over 5 minutes in RTH, or the local `regime-trader kill` command |
| Manual approval | Order notional above $25,000 (decided) waits for your confirmation |
| Account guard | Refuses to start unless the account ID starts with `DU` (an IBKR paper account) and the API port is the paper port |
| Order types | Market-on-open style orders at the next bar's open, or marketable limit orders capped at 5 bps through the touch |

Limits live in one module (`risk.py`) with no imports from the model or
playbook layers; the architecture test enforces that. Market data, news
and headlines are treated strictly as data, never as instructions. The
system never asks for or enters a password or 2FA code: you log into IB
Gateway yourself (or through IBC on the server).

## 11. Backtest and acceptance gates

- **Walk-forward.** Fit on 2018-01 to 2020-12, refit every 30 days on an
  expanding window, and trade out of sample from 2021-01 to the present.
  That spans more than two years and several regimes.
- **Costs.** IBKR tiered commission ($0.0035/share, minimum $0.35), plus
  slippage of half the spread (1 cent) plus 1 bp.
- **Reported** per state and for the full system; daily returns
  annualized ×√252.
- **Baselines:** buy-and-hold SPY, and the best single static strategy
  (trend-only, or mean-reversion-only, chosen on training data only).
- **Gates, all out of sample and after costs:**
  - Sharpe above 1.5;
  - maximum drawdown below 15%;
  - hit rate above 55% (per trade);
  - t-statistic of the mean daily return above 2.0;
  - beats both baselines.

  The winner is written into `strategy.md`.
- **Locked holdout.** The most recent 12 months are kept out of every
  development and self-improvement loop, and evaluated only for a
  candidate you approve. *This deviates from your prompt on purpose:*
  re-running the same backtest after every nightly tweak turns the gates
  into a fitting target, and the reported Sharpe stops meaning anything.
  The holdout is the defence. Each result also reports how many candidate
  configurations were tried, so the multiple-testing risk is visible.

**Honest prior.** Few hourly regime strategies on SPY clear Sharpe 1.5 after
costs. If nothing passes, the system stays on paper and the report says so;
the gates are not loosened.

## 12. Refits, label stability and drift

- **Refit** every 30 days on an expanding window. The refit happens
  offline (overnight or at the weekend) and goes live at the next open.
- **Stable labels.** New states are matched to old ones by minimum-cost
  assignment (Hungarian) on standardized (mean return, volatility) pairs.
  A state that matches nothing gets a new label and alerts you.
- **Drift alarm,** which freezes new entries (exits stay allowed) and
  alerts you, when any of these holds (*defaults*):
  - any transition probability moves by more than 0.15;
  - any matched state's mean return or volatility moves by more than 1
    of the previous fit's standard deviations;
  - the 35-bar rolling live log-likelihood per bar falls below the 1st
    percentile of the in-sample distribution.
- Claude proposes responses; the harness validates them against the gates
  and the holdout; you approve before anything ships.

## 13. Self-improvement loop (nightly)

1. The harness assembles the day's record: bars, filtered probabilities,
   switches, orders, fills, P&L, and every wrong state call (judged in
   hindsight by smoothed posteriors) and every losing trade.
2. Claude (Opus 5.5) reads it and writes a post-mortem: root cause, one
   new rule per loss, and small proposed edits to features, playbooks or
   `strategy.md`. These go into `proposals/YYYY-MM-DD/`. Claude does not
   score its own proposals.
3. The harness backtests each proposal against the same gates (§11) and
   records the result next to the proposal.
4. Nothing ships automatically. A proposal that clears the gates waits for
   your approval, then lands as a normal commit with tests and a changelog
   entry. Rollback is `git revert`.

Claude runs nightly through the Anthropic API (decided). The
`ANTHROPIC_API_KEY` lives in `.env`, and code enforces a monthly spend cap
(*default*: $20): when the cap is reached, the nightly review is skipped and
you get an alert. Trading itself never depends on the API being up.

## 14. Operations

- **Dashboard.** A local web page (Streamlit) showing:
  - the state probabilities over time;
  - the current state and its expected remaining duration;
  - the active playbook;
  - the last action and its reason;
  - position and P&L;
  - the risk-limit headroom.

  It updates every bar.
- **Alerts.** Outbound-only Telegram bot (token in `.env`) for every fill,
  error, state switch, drift warning and kill-switch event. It takes no
  inbound commands, so a chat message can never trade.
- **Daily report** (Telegram and file):
  - the current state;
  - time spent in each state;
  - trades, and P&L per state;
  - win rate and the largest loss;
  - calibration score.
- **Secrets.** `.env` only (`.env.example` lists the names): the IB
  Gateway host, port and account ID, the Telegram token, and the
  Anthropic key. The file is gitignored and never logged.
- **Deployment: your Windows PC** (decided). IB Gateway runs under IBC,
  which handles the daily restart and re-login and has Windows scripts. The
  trader and the dashboard run as Windows services with restart-on-failure
  (via NSSM, or Task Scheduler "restart if the task fails"). Windows-specific
  requirements:
  - sleep and hibernate disabled;
  - Windows Update restarts confined to active hours outside 09:00–16:30
    ET;
  - an auto-login account for IBC;
  - a watchdog that sends an alert when the trader misses a bar.

  A home PC adds power and internet outages to the risk list in §15. The
  system must be flat whenever it cannot see the market (§8 rule 6). Vercel
  cannot work: it is serverless and cannot keep IB Gateway running.

## 15. Go-live checklist (all must be clean; otherwise no live money)

1. Do paper results match the backtest? Same-period slippage and fills
   within tolerance, and P&L inside the backtest's confidence band.
2. Is every decision on filtered probabilities only? Confirmed by the
   future-perturbation test (§6) and an import ban on smoothing and Viterbi
   in decision code.
3. Did the kill switch fire in testing? Each trigger in §10 is exercised
   in tests and in a paper drill.
4. Is any hard limit delegated to a model? The architecture test proves
   `risk.py` imports neither the model nor the playbook layers.
5. Does the refit keep labels stable across every walk-forward refit?
6. What market would break this? Answered under the heading **WHAT COULD
   BLOW UP THIS ACCOUNT?**, covering at least:
   - overnight and weekend gaps, which a stop cannot catch;
   - halts and limit-up/down events;
   - a regime the training data never saw;
   - a slow drift that stays under the alarm;
   - IB Gateway or data outages while in a position;
   - correlated failure of the model and the risk inputs;
   - overfitting through repeated self-improvement;
   - fat-finger sizing from a bad equity read.

## 16. Build process

The same discipline as MarketMicrostructure:

- three approval stops (this spec, the architecture, the plan);
- then, for every module, a failing test first, the implementation, and a
  review against this spec;
- a `CHANGELOG.md` entry for every change;
- CI (ruff, mypy `--strict`, pytest) green before `main` moves;
- rollback through git.

No AgenKit (decided).
