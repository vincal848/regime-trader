# STRESS: reduced trend exposure

**Thesis.** High volatility without a clear crash: moves are large in both
directions. Only the strongest, clearest uptrends are worth small exposure.
Everything else stands aside.

- **Entry:** a strong trend (more than 1.5 volatility units) and an up bar.
- **Exit:** any loss of trend, or a stop, or after half a session.
- **Size:** at most 25% of equity (the state cap enforces it regardless).
- **Invalidation:** volatility keeps rising for a full session while
  holding.

```toml
state = "STRESS"
entry = "trend > 1.5 and ret > 0"
exit = "trend < 0.5"
stop_loss_vol = 2.0
take_profit_vol = 3.0
max_size = 0.25
max_hold_bars = 4
invalidation = "Realized volatility rises for seven consecutive bars while holding."
```
