# Regime Trader

An hourly SPY strategy for an Interactive Brokers **paper** account. A
different playbook trades in each hidden market regime:

- **A Gaussian Hidden Markov Model** reads the market state each hour.
- **Claude (Opus 5.5)** researches and proposes the playbooks overnight.
- **Deterministic code** makes every decision and enforces every limit.

The models advise; the code decides.

> **Status: paper only.** The build works end to end against fakes and
> demo data, but no configuration has yet passed the acceptance gates.
> Read [docs/GO_LIVE.md](docs/GO_LIVE.md): it lists exactly what is still
> open, and answers the question "WHAT COULD BLOW UP THIS ACCOUNT?". The
> code refuses live accounts and live ports.

## Why it is built this way

The prompt this project started from asked for an AI-run regime trader.
The design keeps what is useful in that idea and removes the ways it fails:

| Risk in the idea | What the code does |
|---|---|
| A model that sees the future in a backtest | Decisions use only the **forward filter**: P(state now, given bars so far). Smoothed and Viterbi inference are confined to the offline calibration module, and an architecture test enforces it. A future-perturbation test checks that changing any later bar never changes an earlier decision |
| A model that talks itself past its limits | Hard limits live in `risk.py`, which imports nothing but the bar schema. Playbook sizes are clipped to state caps in code. Claude's proposals are parsed by a whitelisted grammar and never applied automatically |
| Overfitting through nightly self-improvement | Every proposal is backtested outside a 12-month **locked holdout**, the number of candidates tried is counted, and nothing ships without your approval and a commit |
| Over-sizing on overconfident probabilities | Size = min(¼ Kelly, state cap) × P(state) × (1 − entropy). Until a refit has shown the probabilities to be **calibrated** out of sample, sizing falls back to a fixed quarter of the cap |
| Unattended failure | Stale data, exceptions, disconnections, repeated rejects, the drawdown limit and bad equity reads all end flat, with an alert. The kill switch is sticky across restarts |

## How a bar flows

```
IB Gateway ─► completed hourly bar ─► features (past-only, z-scored on t−1 statistics)
          ─► HMM forward filter ─► switching rules (hysteresis, cooldown, early cut)
          ─► playbook for the active state ─► size (Kelly × probability × certainty)
          ─► risk.check_order (caps, daily loss, kill switch, $25k approval)
          ─► marketable limit order (±5 bps) ─► journal + Telegram
```

The backtest and the live trader call the same `engine.decide`, so the
code that is tested is the code that trades.

## Layout

| Layer | Modules | Rule |
|---|---|---|
| 0 core | `bars`, `features`, `hmm`, `switching`, `playbook`, `sizing`, `risk`, `metrics` | Pure functions, no I/O |
| 1 engine | `engine` | One bar in, one decision out |
| 2 research | `refit`, `backtest`, `calibration` | Walk-forward, gates, drift; no I/O |
| 3 adapters | `store`, `ibkr`, `yahoo`, `alerts`, `llm` | All I/O lives here |
| 4 apps | `live`, `nightly`, `dashboard`, `cli` | Wiring |

`tests/test_architecture.py` enforces the layering. Details are in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Quick start (demo data, no broker)

```powershell
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install -e ".[demo,dev]"
mkdir demo; xcopy playbooks demo\playbooks\ /E
.venv\Scripts\regime-trader --root demo fetch --source yahoo
.venv\Scripts\regime-trader --root demo backtest --test-start 2025-01-02 --no-holdout
```

Yahoo's free hourly history covers about two to three years. That is
enough to watch the pipeline work, but not enough for the real acceptance
test (spec §11), which runs on IBKR history from 2018. The demo's results
are in [docs/DEMO.md](docs/DEMO.md).

## Paper trading

[docs/DEPLOY_WINDOWS.md](docs/DEPLOY_WINDOWS.md) covers the full Windows
setup: IB Gateway under IBC, the trader as an NSSM service, the nightly
review, the watchdog and the monthly refit as scheduled tasks, and the
power and update settings a trading PC needs.

| Command | What it does |
|---|---|
| `regime-trader fetch` | Cache hourly bars (IBKR, or Yahoo for the demo) |
| `regime-trader fit` | Fit, or refit, the regime model; reports drift and calibration |
| `regime-trader backtest` | Walk-forward backtest against the gates; exit code 1 if they fail |
| `regime-trader live` | Run the paper trader |
| `regime-trader kill [--reset]` | The sticky kill switch |
| `regime-trader approve --minutes N` | Allow orders over $25,000 for N minutes |
| `regime-trader nightly` | Daily report, then the Claude review and proposal backtests |
| `regime-trader dashboard` | The local dashboard |

Secrets (the Telegram token and Anthropic key) live in `.env` (see
`.env.example`). They are never committed, logged or shown in a repr.
IBKR needs no API key: you log in to IB Gateway yourself.

## Development

```powershell
.venv\Scripts\python -m pip install -e ".[ibkr,llm,dashboard,demo,dev]"
.venv\Scripts\python -m ruff check src tests
.venv\Scripts\python -m mypy src tests
.venv\Scripts\python -m pytest --cov=regime_trader
```

Every change follows the same order: failing tests, the implementation,
then a [CHANGELOG.md](CHANGELOG.md) entry. Rollback is `git revert`. The
spec is [docs/SPEC.md](docs/SPEC.md) and the build plan
[docs/PLAN.md](docs/PLAN.md).

## License

MIT. Not investment advice. Trading on these signals can lose money.
