# Demo results (generated)

<!-- Written by scripts/demo_walkforward.py; do not edit by hand. -->

Data: 5,066 hourly RTH bars of SPY, 2023-11-08 to 2026-10-07. Walk-forward from 2025-01-02, refit every 30 days. No holdout.

```
Sharpe -0.96 | max drawdown 6.0% | hit rate 35.6% | t-statistic -1.28 | total return -4.7% | trades 118
Baselines (Sharpe): buy-and-hold 1.08, static CALM_UP 0.11
Gates: FAILED
  [ ] sharpe
  [x] max_drawdown
  [ ] hit_rate
  [ ] t_statistic
  [ ] beats buy-and-hold
  [ ] beats static CALM_UP
```

Orders that would have waited for manual approval (over $25,000): 18

## Refits

| Date | K | Labels | Drift | Sizing |
|---|---|---|---|---|
| 2025-01-02 | 3 | CALM_UP_1, CALM_UP_2, CRASH | first fit | quarter cap |
| 2025-02-03 | 3 | CALM_UP_2, CRASH, CALM_UP_1 | none | quarter cap |
| 2025-03-05 | 3 | CALM_UP_1, CALM_UP_2, CRASH | none | calibrated |
| 2025-04-04 | 3 | CRASH, CALM_UP_2, CALM_UP_1 | none | calibrated |
| 2025-05-05 | 3 | CALM_UP_1, CALM_UP_2, CRASH | none | calibrated |
| 2025-06-04 | 3 | CRASH, CALM_UP_2, CALM_UP_1 | none | quarter cap |
| 2025-07-07 | 3 | CALM_UP_2, CRASH, CALM_UP_1 | none | quarter cap |
| 2025-08-06 | 3 | CALM_UP_2, CALM_UP_1, CRASH | none | calibrated |
| 2025-09-05 | 3 | CRASH, CALM_UP_2, CALM_UP_1 | none | quarter cap |
| 2025-10-06 | 3 | CALM_UP_1, CRASH, CALM_UP_2 | none | quarter cap |
| 2025-11-05 | 3 | CALM_UP_2, CRASH, CALM_UP_1 | none | quarter cap |
| 2025-12-05 | 3 | CALM_UP_1, CALM_UP_2, CRASH | none | calibrated |
| 2026-01-05 | 3 | CALM_UP_2, CALM_UP_1, CRASH | none | quarter cap |
| 2026-02-04 | 3 | CALM_UP_2, CALM_UP_1, CRASH | none | quarter cap |
| 2026-03-06 | 3 | CALM_UP_2, CALM_UP_1, CRASH | none | quarter cap |
| 2026-04-06 | 3 | CALM_UP_1, CALM_UP_2, CRASH | none | quarter cap |
| 2026-05-06 | 3 | CALM_UP_1, CALM_UP_2, CRASH | none | quarter cap |
| 2026-06-05 | 3 | CALM_UP_1, CALM_UP_2, CRASH | none | quarter cap |
| 2026-07-06 | 3 | CALM_UP_2, CALM_UP_1, CRASH | none | calibrated |
| 2026-08-05 | 3 | CALM_UP_1, CALM_UP_2, CRASH | none | quarter cap |
| 2026-09-04 | 3 | CALM_UP_1, CALM_UP_2, CRASH | none | quarter cap |
| 2026-10-05 | 3 | CALM_UP_1, CALM_UP_2, CRASH | none | quarter cap |

## Per state

| State | Bars active | Share | Trades | P&L ($) |
|---|---|---|---|---|
| CALM_UP_1 | 1,329 | 43.3% | 53 | -1,283 |
| CALM_UP_2 | 1,533 | 50.0% | 65 | -3,425 |
| CRASH | 200 | 6.5% | 0 | +0 |
| NONE | 6 | 0.2% | 0 | +0 |
