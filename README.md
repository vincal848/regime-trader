# Regime Trader

[![ci](https://github.com/vincal848/regime-trader/actions/workflows/ci.yml/badge.svg)](https://github.com/vincal848/regime-trader/actions/workflows/ci.yml)

**An hourly SPY regime trader for an Interactive Brokers paper account: a
Gaussian HMM reads the market state, a different playbook trades each
state, Claude proposes changes overnight, and deterministic code makes
every decision and enforces every limit.**

This started as a viral prompt for an "AI that trades market regimes". The
idea has real content. Markets do move through persistent regimes, and a
hidden Markov model is the textbook way to infer them. But taken literally,
the prompt has four ways to fail:

- a backtest that peeks at the future;
- a model that sizes from overconfident probabilities;
- a language model allowed to touch risk limits;
- a nightly "self-improvement" loop that overfits the backtest.

This build keeps the idea and closes each of those failure modes in code:

1. **No look-ahead.** Decisions use only the HMM's *forward filter*
   (P(state now | bars so far)). Smoothed and Viterbi inference is confined
   to one offline calibration module, an architecture test enforces that,
   and a future-perturbation test proves that changing later bars never
   changes an earlier decision.
2. **Sizing earns its inputs.** Size = min(¼ Kelly, state cap) × P(state) ×
   (1 − entropy). Until a refit has shown the probabilities to be
   *calibrated* out of sample, sizing falls back to a fixed quarter of the
   cap.
3. **The models advise; the code decides.** Hard limits live in `risk.py`,
   which imports nothing but the bar schema: a daily loss cap, a sticky
   kill switch, state caps, long/flat only, and manual approval above
   $25,000. The account guard refuses anything but an IBKR paper account.
4. **Self-improvement behind gates.** Claude's nightly proposals are parsed
   by a whitelisted grammar, backtested outside a 12-month locked holdout,
   counted against a multiple-testing tally, and never applied
   automatically.

The full specification is **[docs/SPEC.md](docs/SPEC.md)**, the design
**[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**, the build plan
**[docs/PLAN.md](docs/PLAN.md)**, the demo results
**[docs/DEMO.md](docs/DEMO.md)**, and every change, tests first,
**[CHANGELOG.md](CHANGELOG.md)**.

## Status

All 11 plan steps are implemented, test-first: 245 tests at 95% coverage,
`mypy --strict` and ruff. CI enforces the layer architecture. **It is paper
only, and no configuration has passed the acceptance gates.**
[docs/GO_LIVE.md](docs/GO_LIVE.md) answers "WHAT COULD BLOW UP THIS
ACCOUNT?" and lists what is still open.

| Area | Acceptance | Result |
|---|---|---|
| No look-ahead | A decision at bar t is unchanged when any later bar changes; smoothing is banned outside calibration | **met** (future-perturbation and AST architecture tests) |
| Shared engine | The backtest and live trading call the same `engine.decide` and install refits the same way | **met**: one `decide`, one `adopt_fit` |
| Hard limits | Every kill trigger flattens and halts; no limit is reachable from a model | **met** in tests (drawdown, manual, disconnect, reject and stale-data paths); paper drill open |
| Calibration | Sizing uses probabilities only after a refit beats climatology out of sample | **met**; on real data only 6 of 22 refits qualified |
| Nightly loop | Proposals are validated, backtested against the gates, and never applied | **met** (prompt-injection and grammar-rejection tests) |
| Acceptance gates | Sharpe > 1.5, max drawdown < 15%, hit rate > 55%, t > 2, beats buy-and-hold and the best static strategy | **not met** on the Yahoo demo: Sharpe −0.96 against buy-and-hold's 1.08 ([DEMO.md](docs/DEMO.md)). The real test needs IBKR history from 2018 |

## Quick start

```bash
pip install -e ".[dev]"     # add ,ibkr,llm,dashboard,demo as needed
pytest -q                   # 245 tests
ruff check . && mypy        # lint, strict types
```

The demo walk-forward needs no broker; Yahoo data stays local in `demo/`:

```bash
pip install -e ".[demo]"
mkdir demo && cp -r playbooks demo/
regime-trader --root demo fetch --source yahoo
regime-trader --root demo backtest --test-start 2025-01-02 --no-holdout
python scripts/demo_walkforward.py demo 2025-01-02     # per-state and per-refit detail
```

Paper trading, with the full Windows setup in
[docs/DEPLOY_WINDOWS.md](docs/DEPLOY_WINDOWS.md):

```bash
regime-trader fetch          # hourly bars from IB Gateway
regime-trader fit            # fit or refit; reports drift and calibration
regime-trader backtest       # walk-forward against the gates (exit 1 when they fail)
regime-trader live           # the paper trader
regime-trader kill [--reset] # sticky kill switch
regime-trader approve --minutes 60
regime-trader nightly        # daily report, Claude review, proposal backtests
regime-trader dashboard
```

## How a bar flows

```
IB Gateway ─► completed hourly bar ─► causal features (z-scored on t−1 statistics)
          ─► HMM forward filter ─► switching (hysteresis, cooldown, early stress cut)
          ─► the active state's playbook ─► size (¼ Kelly × probability × certainty)
          ─► risk.check_order (caps, daily loss, kill switch, $25k approval)
          ─► marketable limit order (±5 bps) ─► journal + Telegram
```

## Repository guide

| Path | Contents |
|---|---|
| `src/regime_trader/` | The package: 21 modules in 5 layers (core, engine, research, adapters, apps). See [ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| `tests/` | Tests per layer, plus architecture, future-perturbation and fake-broker live tests |
| `playbooks/` | One Markdown playbook per state, each with a TOML block in a whitelisted condition grammar |
| `scripts/demo_walkforward.py` | The reproducible demo run behind DEMO.md |
| `docs/SPEC.md` | What the system must do, the decisions made, and the deviations from the original prompt |
| `docs/ARCHITECTURE.md` | Layers, module contracts, state on disk |
| `docs/PLAN.md` | The 11-step build plan, tests first |
| `docs/DEMO.md` | The Yahoo demo walk-forward and what it shows |
| `docs/GO_LIVE.md` | The go-live checklist and "WHAT COULD BLOW UP THIS ACCOUNT?" |
| `docs/DEPLOY_WINDOWS.md` | IB Gateway under IBC, NSSM service, scheduled jobs, PC settings |
| `CHANGELOG.md` | Every step: tests first, then implementation, then the defects found |

## Data and secrets

No market data is committed. Bars, the journal, fitted models and
proposals live in gitignored folders.

- **IBKR (TWS API via `ib_async`).** Hourly RTH TRADES bars: the real
  source for fitting, backtesting and trading.
- **Yahoo Finance.** About three years of free hourly bars, enough for the
  demo but not for acceptance.

IBKR needs no API key: you log in to IB Gateway yourself. The Telegram
token and Anthropic key live in `.env` (`.env.example` lists the names).
They are never committed, logged or shown in a repr. Telegram is outbound
only, so a chat message can never trade.

## License

MIT © 2026 Caleb Vinson. Not investment advice.
