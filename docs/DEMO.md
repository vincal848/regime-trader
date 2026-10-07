# Demo walk-forward (Yahoo data)

**This is a demo, not the acceptance test.** Yahoo's free hourly history
(about 2.9 years) is too short for spec §11, which needs a fit from 2018,
walk-forward from 2021, and a 12-month locked holdout. This run has no
holdout at all. It shows that the pipeline runs end to end on real SPY
bars, and what the starting playbooks do. It is not evidence for or
against the idea.

**Reproduce it:**

```powershell
regime-trader --root demo fetch --source yahoo
python scripts/demo_walkforward.py demo 2025-01-02
```

- Data: 5,066 hourly RTH bars of SPY, 2023-11-08 to 2026-10-07.
- Walk-forward: from 2025-01-02, refitting every 30 days on an expanding
  window.
- Model: K chosen once by out-of-sample likelihood, 10 restarts.
- Costs: IBKR tiered commission, 1 bp of slippage, and half a cent of
  spread.
- Run on 2026-10-07, after the step 11 fixes (switching carries across
  refits; the likelihood alarm runs every bar). Before those fixes the
  same run gave a Sharpe of −1.16 over 124 trades.

## Result: the gates fail

| | System | Buy-and-hold | Best static (CALM_UP, chosen on training data) | Gate |
|---|---|---|---|---|
| Sharpe (daily, ×√252) | **−0.96** | 1.08 | 0.11 | > 1.5 |
| Max drawdown | 6.0% | | | < 15% ✓ |
| Hit rate (per trade) | 35.6% | | | > 55% |
| t-statistic | −1.28 | | | > 2.0 |
| Total return | −4.7% | | | |
| Trades | 118 | | | |

Only the drawdown gate passes, and it passes because the system is small
and often flat. As the spec warned (§11, "Honest prior"), the system stays
on paper and the gates are not loosened.

## What the run shows

1. **The model found calm and crash, nothing in between.** Every refit
   chose K = 3 with labels `CALM_UP_1`, `CALM_UP_2` and `CRASH`. The CHOP
   and STRESS playbooks never traded. Over this window, SPY's hourly bars
   split into two shades of calm plus a rare, violent state (6.5% of
   bars).
2. **The CALM_UP entry rule loses after costs.** Both calm states traded
   the CALM_UP playbook (enter on `trend > 0.5 and ret > 0`, stop 3 × rv,
   target 6 × rv). It won 36% of the time, and the losses outweighed the
   wins. That is what a short-horizon momentum entry looks like on hourly
   SPY once each round trip pays about 2 bps. Buy-and-hold earned more by
   simply sitting through the same calm periods.
3. **Calibration flips between refits.** Only 6 of the 22 refits judged
   the previous fit calibrated, so sizing ran at a quarter of the cap most
   of the time. The probabilities are not reliable enough to size from,
   which is why the fallback exists.
4. **No drift alarms fired.** Labels stayed stable through every refit (the
   Hungarian matching worked, even though the state order changed nearly
   every month).
5. **Manual approval would have mattered.** 18 orders exceeded $25,000
   and would have waited for `regime-trader approve`.

## What would be worth testing next (through the normal loop)

These are hypotheses for the nightly proposal loop or a human edit. Each
would go through the gates and the holdout, on IBKR data from 2018, before
anything changes:

- A CALM_UP playbook that holds through the calm state instead of trading
  momentum bursts. The regime label may carry the information, while the
  entry trigger adds costs.
- Fewer, longer trades: a higher entry threshold, or a minimum hold.
- A cleaner calm/stress split, for example a volatility-only feature set,
  so that STRESS and CHOP can appear at all.

Every one of these is another draw against the same data, and
`proposals/trials.json` counts them.
