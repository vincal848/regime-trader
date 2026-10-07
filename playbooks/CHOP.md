# CHOP: mean reversion

**Thesis.** With no drift and moderate volatility, sharp hourly drops tend
to partly reverse as liquidity providers lean against them. Buy unusually
weak bars and take the bounce quickly.

- **Entry:** the last bar's return is more than 1.5 standard deviations
  below normal (`z_ret < -1.5`), on no worse than ordinary volume (a
  high-volume drop is information, not noise).
- **Exit:** the return normalises (`z_ret > 0`), or a stop or take-profit
  is hit, or after one session.
- **Invalidation:** three consecutive down bars after entry. Weakness that
  persists is a trend, and this playbook is fading it.

```toml
state = "CHOP"
entry = "z_ret < -1.5 and volume_ratio < 1.5"
exit = "z_ret > 0"
stop_loss_vol = 2.0
take_profit_vol = 2.0
max_size = 0.5
max_hold_bars = 7
invalidation = "Three consecutive down bars after entry."
```
