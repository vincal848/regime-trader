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
decide(history_bars, model, playbooks, switch_state, position, account, config)
   -> Decision(ts, probs, next_probs, regime, playbook, signal, target_fraction, order, veto, reasons)
```

It runs features, the forward filter, switching, the playbook signal,
sizing and the risk check, and returns a `Decision`, which is everything the
journal, dashboard and alerts need, plus the new switching state. The engine
never places orders.

## Research (layer 2)

- **`backtest`.** A walk-forward loop over bars:
  - the model is refit every 30 days on an expanding window;
  - labels are matched to the previous fit;
  - a fill happens at the next bar's open, with commission and slippage;
  - stops and take-profits are checked against the next bars' high/low
    (a gap fills at the open);
  - it reports per state and in total, against buy-and-hold and the best
    static strategy, with the acceptance gates and a locked holdout.
- **`refit`.** `match_labels` (Hungarian assignment on state statistics)
  and `drift_report` (transition and mean shifts, live log-likelihood
  percentile).
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

- **`live`.** Once per bar close: get the bar, check staleness, run
  `engine.decide`, run the risk check, send the order through the broker,
  write the journal, send alerts. Any exception means flat and an alert.
- **`nightly`.** Assembles the day's record, asks the LLM for proposals,
  backtests each proposal against the gates, and writes `proposals/`.
  Nothing ships without your approval.
- **`dashboard`.** Streamlit, reading the journal only.
- **`cli`.** `fetch`, `fit`, `backtest`, `live`, `kill`, `nightly`,
  `report`.

## State on disk

```
data/        bar cache (gitignored)
journal.db   decisions, orders, fills, switches, alerts (gitignored)
models/      fitted HmmModel JSON per refit, with label map (gitignored)
playbooks/   <STATE>.md (versioned)
proposals/   nightly LLM proposals and their gate results (versioned)
strategy.md  the accepted configuration (versioned)
.env         secrets (gitignored; .env.example lists names)
```
