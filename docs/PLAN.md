# Plan

Every step lands in this order:

1. failing tests;
2. the implementation;
3. a review against `SPEC.md` and the architecture rules;
4. a `CHANGELOG.md` entry.

`ruff`, `mypy --strict` and `pytest` must be green before the next step
starts. Rollback is `git revert` of a step's commits.

| Step | Builds | Tests written first (abridged) |
|---|---|---|
| 1 | Package skeleton, tooling, CI, architecture test | Layer imports enforced; `risk` imports only `bars`; smoothing and Viterbi calls only in `calibration` |
| 2 | `bars`, `features` | Quality checks reject bad bars; every feature at t is unchanged when bars after t change; z-scores use only statistics ≤ t−1; same-bar-of-day volume ratio |
| 3 | `hmm` | Forward filter equals a brute-force enumeration; filtered rows sum to 1; next-state probabilities = filtered · A; recovers a planted 3-state process; K selection prefers the simpler model on near ties; labels follow the spec's rules; expected duration = 1/(1 − A_ii); seeded fits reproduce |
| 4 | `switching`, `sizing`, `risk` | Every switching rule in spec §8 as its own test; size formula, entropy shrink, zero in CRASH and uncertain states, state caps; every risk limit and kill trigger; a vetoed order never passes |
| 5 | `playbook` | The TOML block parses; unknown keys and features are rejected; the grammar refuses anything but comparisons with and/or; `max_size` is clipped to the cap; the four starting playbooks parse |
| 6 | `engine` | One decision per bar; flat on unhealthy input; the future-perturbation test over the full engine; the risk veto wins over any playbook |
| 7 | `refit`, `backtest`, `metrics`, `calibration` | Fills at the next open, never the signal bar; costs applied; stops gap to the open; label matching survives permuted refits; drift alarm fires on shifted parameters; gates computed exactly on known series; baselines; a holdout that the development API cannot read |
| 8 | Adapters: `store`, `ibkr`, `yahoo`, `alerts`, `llm` | Round-trips; the paper-only account guard; secrets never appear in logs or reprs; Telegram has no inbound path; the LLM spend cap blocks calls past budget |
| 9 | Apps: `live`, `nightly`, `dashboard`, `cli` | The live loop with a fake broker and clock: stale data means flat, an exception means flat and an alert, the kill switch flattens and halts, manual approval above $25,000; nightly proposals are never applied automatically |
| 10 | Demo run and docs | A Yahoo-data walk-forward demo (labelled a demo, not the acceptance test); `docs/DEPLOY_WINDOWS.md`; `docs/GO_LIVE.md` with "WHAT COULD BLOW UP THIS ACCOUNT?" |
| 11 | `/simplify` review | Four parallel review agents (reuse, simplification, efficiency, altitude) over the full diff; fixes applied and logged |

**Known external dependencies the build cannot complete alone:**
- logging into the IBKR paper account in IB Gateway (you);
- the market-data subscription (you);
- the Telegram bot token and Anthropic key in `.env` (you).

Everything that touches them is built and tested against fakes, and runs as
soon as they are provided.
