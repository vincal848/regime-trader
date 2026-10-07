# CALM_UP: trend-following

**Thesis.** In a low-volatility, positive-drift regime, hourly returns are
mildly autocorrelated through flows that build over days. Ride the trend
while it holds, and step aside as soon as price falls back through its
two-week average.

- **Entry:** the trend is more than half a volatility unit above its 70-bar
  EMA, and the last bar was up.
- **Exit:** the trend turns negative, or a stop or take-profit is hit, or
  after five sessions.
- **Invalidation:** two consecutive closes below the 70-bar EMA while the
  HMM still says CALM_UP. If that happens, the regime call is wrong, not
  just the trade.

```toml
state = "CALM_UP"
entry = "trend > 0.5 and ret > 0"
exit = "trend < 0"
stop_loss_vol = 3.0
take_profit_vol = 6.0
max_size = 1.0
max_hold_bars = 35
invalidation = "Two consecutive closes below the 70-bar EMA while the filtered state is CALM_UP."
```
