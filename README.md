# Regime Trader

Hourly SPY regime trader for an Interactive Brokers **paper** account: a
Gaussian HMM detects the market state, Claude writes one playbook per state,
and deterministic code makes every decision and enforces every limit.

See [docs/SPEC.md](docs/SPEC.md), [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
and [docs/PLAN.md](docs/PLAN.md).
