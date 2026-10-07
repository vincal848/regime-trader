# Architecture

## Principles

The design rules are enforced by `tests/test_architecture.py`, so the
diagram below cannot drift from the code.

1. **One decision engine.** The backtest and live trading call the same
   pure `engine.decide(...)`. Training/serving skew, where the backtest
   tests different code from what trades, cannot happen.
2. **Side effects live at the edges.** Layers 0–2 do no I/O: no files, no
   network, no clock, no logging. The broker, data downloads, Telegram, the
   Anthropic API, the journal and the dashboard are adapters in layer 3,
   used only by the apps in layer 4.
3. **Code decides, models advise.** `risk.py` imports nothing from the HMM
   or playbook modules. A hard limit cannot be reached, changed or bypassed
   through a model.
4. **No look-ahead by construction.**
   - Features are causal (rolling or expanding windows, standardized with
     statistics through t−1).
   - State probabilities come from our own forward filter.
   - hmmlearn's smoothed and Viterbi methods (`predict_proba`, `predict`,
     `decode`, `score_samples`) may be called only in `calibration.py`,
     which is offline evaluation.
   - A future-perturbation test proves it end to end.
5. **Direct code.** Plain functions and frozen dataclasses. No plugin
   systems, no ORMs, no `eval`. Playbook rules are a tiny parsed grammar.

## Layers

```
4  apps        cli · live · nightly · dashboard                  orchestration, the clock
3  adapters    store · ibkr · yahoo · alerts · llm               all I/O
2  research    backtest · refit · calibration                    walk-forward, labels, drift, evaluation
1  engine      engine                                            one bar in, one decision out
0  core        bars · features · hmm · switching · playbook ·    pure, typed, no I/O
               sizing · risk · metrics
```

A module may import only from its own layer or lower ones. `risk` imports
only from `bars`.

## Core modules (layer 0)

| Module | Contract |
|---|---|
| `bars` | Hourly bar frame schema (`open, high, low, close, volume`, tz-aware index on the RTH grid); `validate_bars` raises on any quality failure (spec §3) |
| `features` | `compute_features(bars) -> DataFrame`: the five raw features and their z-scores. Row t depends only on bars ≤ t, and the z-scores use statistics ≤ t−1 |
| `hmm` | `fit_hmm`, `select_states` (BIC, plus out-of-sample log-likelihood with a one-standard-error rule), `forward_filter` (filtered probabilities and next-state probabilities), `label_states`, `expected_duration`. `HmmModel` is a frozen dataclass of parameters, so decisions never touch the hmmlearn object |
| `switching` | `SwitchState`, `step_switch(state, probs, next_probs, healthy, config)`: hysteresis, cooldown, early stress cut, uncertainty, flat on error (spec §8) |
| `playbook` | `parse_playbook(markdown) -> Playbook`: one TOML block, with conditions in a whitelisted grammar (`feature op number`, `and` / `or`); `evaluate_signal(playbook, row, position)` |
| `sizing` | `target_fraction(...)`: ¼ Kelly × P(state) × (1 − H/ln K), clipped to the state cap (spec §9) |
| `risk` | `RiskLimits`, `AccountState`, `check_order(...) -> Approved | Vetoed`, `kill_reasons(...)`. Pure; checked before every order |
| `metrics` | Sharpe, maximum drawdown, hit rate, t-statistic, and the acceptance-gate evaluation |

## Engine (layer 1)

```
decide(state, ts, row, price, account, model, playbooks, kelly, config, healthy)
   -> (EngineState, Decision(ts, probabilities, next_state, log_likelihood, regime,
                             signal, target_fraction, target_shares, order, entry_rv, reasons))
```

`EngineState` carries the filter's prior, the switching state and the open
position from bar to bar. `decide` advances the forward filter one step,
then runs switching, the playbook signal, sizing and the risk check. The
`Decision` it returns is everything the journal, dashboard and alerts need.
The engine never places orders; `on_fill` records what the broker actually
did. `playbook_for` maps a state to its playbook, and numbered siblings
(`CALM_UP_1`, `CALM_UP_2`) share their base state's playbook.

## Research (layer 2)

- **`backtest`.** A walk-forward loop over bars:
  - the model is refit every 30 days on an expanding window;
  - labels are matched to the previous fit;
  - a fill happens at the next bar's open, with commission and slippage;
  - stops and take-profits are checked against the next bars' high/low
    (a gap fills at the open);
  - it reports per state and in total, against buy-and-hold and the best
    static strategy, with the acceptance gates and a locked holdout;
  - `acceptance` is the one scoring path, used by the CLI and the nightly
    loop.
- **`refit`.**
  - `fit_regime` builds the **fit bundle** shared by the backtest and live
    trading: the labelled model, per-state Kelly, in-sample
    log-likelihoods, the next-bar prior, and the calibration verdict
    (each refit scores the previous fit out of sample).
  - `match_labels` (Hungarian assignment on state statistics).
  - `drift_report` (transition and mean shifts, stored on the fit) and `rolling_alarm`
    (the live log-likelihood check).
- **`calibration`.** The only module allowed to use smoothed posteriors.
  Brier score and reliability per state.

## Adapters (layer 3)

| Adapter | Side effect | Tested with |
|---|---|---|
| `store` | Parquet bar cache and SQLite journal (decisions, orders, fills, alerts) | temp directories |
| `ibkr` | IB Gateway through `ib_async`: historical bars, live bars, orders, account. Refuses non-paper accounts | a fake IB client |
| `yahoo` | Demo history download | recorded frames |
| `alerts` | Outbound-only Telegram | a fake HTTP sender |
| `llm` | Anthropic API nightly reviewer with a monthly spend cap | a fake client |

## Apps (layer 4)

- **`live`.** Once per bar close:
  - reconcile the last order with the broker;
  - take only completed bars, and treat stale ones as flat;
  - refuse an implausible equity read;
  - run `engine.decide`;
  - watch drift and the kill switch;
  - send the order, or hold it for approval above $25,000;
  - write the journal and the checkpoint; send alerts.

  Any exception means flat and an alert. `Control` is a folder holding the
  sticky `KILL` file and the approval window, so the CLI can change both
  while the trader runs.
- **`nightly`.** Assembles the day's record, asks the LLM for proposals,
  backtests each proposal against the gates, and writes `proposals/`.
  Nothing ships without your approval.
- **`dashboard`.** Streamlit, reading the journal only.
- **`cli`.** `fetch`, `fit`, `backtest`, `live`, `kill`, `approve`,
  `nightly`, `report`, `watchdog`, `dashboard`.

## State on disk

```
data/            bar cache (gitignored)
journal.db       decisions, fills, events (gitignored)
models/          fit.json (the live fit bundle) and archived fit-*.json (gitignored)
live_state.json  the trader's checkpoint (gitignored)
control/         KILL and APPROVED_UNTIL (gitignored)
reports/         daily reports (gitignored)
proposals/       nightly reviews, candidate playbooks, gate results, trials.json (gitignored)
llm_spend.json   the monthly LLM spend ledger (gitignored)
playbooks/       <STATE>.md (versioned): a proposal ships only by being copied here in a commit
strategy.md      the accepted configuration, written by you when one passes the gates
                 and the holdout (versioned; none yet)
.env             secrets (gitignored; .env.example lists names)
```
