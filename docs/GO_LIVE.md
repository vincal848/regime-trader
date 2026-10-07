# Go-live checklist

**Verdict: not ready for live money.** Several checks below are open, and
live money is out of scope for this build in any case (spec §1). This file
says what "clean" would mean for each check, and where each one stands
today.

Each check from spec §15 is answered here with its evidence. The rule is
simple: until every check is clean, and you explicitly approve, no live
account. The code backs this up. `ibkr.settings_from_env` and
`IbkrBroker.connect` refuse any account that does not start with `DU`, and
the live ports 4001 and 7496. Going live would mean deliberately changing
tested code, not flipping a setting.

## Status

| # | Check | Status | Evidence |
|---|---|---|---|
| 1 | Paper results match the backtest | **Open** | Needs weeks of paper trading. Compare `journal.db` fills with backtest fills over the same period, and check that slippage stays inside the cost model (1 bp plus half a cent) and P&L inside the backtest's band |
| 2 | Decisions use filtered probabilities only | **Clean** | `test_engine.py` future-perturbation test (changing bars after t never changes the decision at t); `test_architecture.py` bans `predict_proba`, `predict`, `decode` and `score_samples` outside `calibration.py` |
| 3 | The kill switch fired in testing | **Clean in tests; paper drill open** | `test_apps.py`: the drawdown kill, manual kill, disconnect kill and reject kill each flatten and halt. You still need to run a paper drill: `regime-trader kill --reason "paper drill"` while in a position, confirm the flatten, then `regime-trader kill --reset` |
| 4 | No hard limit is delegated to a model | **Clean** | `test_architecture.py`: `risk.py` imports nothing but `bars`. Playbook sizes are clipped to the state caps in code, and the LLM's proposals are parsed by a whitelisted grammar and never applied automatically |
| 5 | Labels stay stable across refits | **Clean on synthetic data; open on real data** | `test_metrics_refit_calibration.py`: Hungarian matching survives permuted refits. On real data, check each `regime-trader fit` drift line and the archived `models/fit-*.json` files |
| 6 | What market would break this? | Answered below | Each failure mode below has a mitigation and a residual risk |

## WHAT COULD BLOW UP THIS ACCOUNT?

Each entry gives what happens, what the code does about it, and what is
left over.

### Overnight and weekend gaps
- **What happens.** SPY opens far through the stop. Stops here are checked
  on hourly closes and filled at the next open, so a gap fills wherever the
  market opens.
- **Mitigation.**
  - Long/flat only, with no leverage: buys are capped at cash, and the
    largest position is 100% of equity.
  - The STRESS state is capped at 25% and CRASH at 0%.
  - The daily loss limit (2%) blocks new risk after a bad open.
- **Residual.** A gap the size of 1987's (−20%) on a full CALM_UP position
  loses about 20% of equity. Nothing hourly can stop that. The only defence
  is size, so if that is unacceptable, lower the CALM_UP cap in
  `risk.DEFAULT_STATE_CAPS`.

### Trading halts and limit-up/limit-down
- **What happens.** Orders don't fill. The trader's marketable limit sits,
  is cancelled after a bar, and counts as a reject.
- **Mitigation.** Three consecutive rejects fire the kill switch, and every
  reject is alerted. Limit orders are capped 5 bps through the reference
  price, so a reopening auction cannot fill far away.
- **Residual.** While SPY is halted you can't exit. A market-wide halt can
  last the rest of the day.

### A regime the training data never saw
- **What happens.** The HMM assigns new behaviour to the "nearest" known
  state, possibly a confident CALM_UP.
- **Mitigation.**
  - The live log-likelihood alarm freezes entries when the rolling fit
    drops below the in-sample 1st percentile.
  - The top-two probability gap rule zeroes size when the model is torn.
  - Entropy shrinks size whenever the model is unsure.
- **Residual.** A novel regime that looks like a known one in these five
  features passes every check. Fit on history that includes 2020 and 2022.

### Slow drift that stays under the alarm
- **What happens.** State means and transition probabilities creep, so each
  refit is "within tolerance" while the model slowly stops matching reality.
- **Mitigation.**
  - Each refit compares against the previous fit, not the original.
  - Archived fits (`models/fit-*.json`) let you compare across many
    months.
  - The calibration line in the daily report shows the Brier score against
    climatology; when it loses to climatology, sizing should fall back to
    fixed fractions.
- **Residual.** Detecting this is a human job: read the daily report.
  Calibration that silently degrades is the most likely slow failure.

### IB Gateway, internet or power outage while in a position
- **What happens.** The trader can't see the market or send orders. The
  position stays open at the broker.
- **Mitigation.**
  - After 300 s disconnected, the kill switch engages; the trader
    flattens as soon as it reconnects, and stays flat.
  - Stale data (no completed bar 15 minutes after a close) flattens.
  - The watchdog alerts on a missed bar.
  - IBC restarts IB Gateway.
- **Residual.** A home PC on home internet can be down for hours. During
  that time the position is unmanaged. You can always close it from the
  IBKR mobile app. Consider a UPS and a phone hotspot.

### Correlated failure of the model and the risk inputs
- **What happens.** The same bad data feed drives both the model and the
  equity read (for example a stale price that makes everything look calm).
- **Mitigation.**
  - Equity comes from IB's account summary (NetLiquidation), not from the
    bar feed.
  - An equity read that is not positive, or moves more than 25% in one
    bar, is refused, and refusal means flatten.
  - The bar staleness check is independent of the model.
- **Residual.** A plausible-but-wrong equity figure within 25% of the last
  one is accepted.

### Overfitting through repeated self-improvement
- **What happens.** Each nightly tweak is tuned to recent data; the gates
  become a fitting target.
- **Mitigation.**
  - Claude never grades its own proposals.
  - Every proposal is backtested outside the 12-month locked holdout.
  - `proposals/trials.json` counts every candidate tested.
  - Nothing ships without your approval and a commit.
- **Residual.** You are the last line of defence. Evaluate an approved
  candidate on the holdout **once**. If it fails there, it does not ship,
  and there is no second try on the same holdout.

### Fat-finger sizing from a bad equity read
- **What happens.** A wrong equity figure produces a huge order.
- **Mitigation.**
  - The equity plausibility check (above).
  - The state caps, and the 100%-of-equity maximum position.
  - Any order adding more than $25,000 of exposure waits for
    `regime-trader approve`.
- **Residual.** With approval windows open, a bad read within the 25% band
  can size up to 125% of true equity for one bar. IB's own margin checks
  would reject cash-account buys beyond buying power.

### Other risks worth naming
- **Your own approvals.** An approval window lets every large order through
  until it ends. Keep windows short.
- **Software change.** Every change goes through tests and CI, and the
  future-perturbation and architecture tests guard the two properties that
  matter most.
- **API keys.** IBKR has none (you log in to IB Gateway). The Telegram
  token and Anthropic key live in `.env` only, and are never logged or
  shown in a repr.
- **Prompt injection.** Journal text and headlines reach Claude only as
  data inside `<record>` tags, and anything Claude writes back is parsed by
  a whitelisted grammar or filed for a human. No path leads from a model's
  output to an order.

## Before anyone considers live money

1. All six checks clean, with paper results (check 1) over at least three
   months that include a volatile stretch.
2. The acceptance gates passed on IBKR data (not Yahoo) out of sample, then
   once on the locked holdout.
3. A paper kill drill and an outage drill (pull the network cable while in
   a position) both done and journaled.
4. A deliberate, reviewed code change to the paper-only guard. It is
   intentionally not a configuration flag.
