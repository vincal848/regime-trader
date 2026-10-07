# CRASH: flat

**Thesis.** Highest volatility with a negative drift. No entry is worth the
gap risk. Any open position is closed at the next open.

```toml
state = "CRASH"
entry = "never"
exit = "always"
stop_loss_vol = 1.0
take_profit_vol = 1.0
max_size = 0.0
max_hold_bars = 0
invalidation = "None: this playbook is the stand-aside default."
```
