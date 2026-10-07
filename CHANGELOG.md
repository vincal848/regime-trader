# Changelog

Each step of docs/PLAN.md is logged as its tests (written first, failing),
then its implementation, then anything the review found.

## [Unreleased]

### Step 1: skeleton, tooling, architecture test
- Tests first: `tests/test_architecture.py` enforces the layer map:
  - no upward imports;
  - `risk` depends on `bars` only;
  - layers 0–2 do no I/O;
  - hmmlearn's smoothed and Viterbi methods only in `calibration`.
- Implementation: `pyproject.toml` (core numpy/pandas/scipy/hmmlearn;
  extras `ibkr`, `llm`, `dashboard`, `demo`, `dev`), CI (ruff, mypy
  `--strict`, pytest with a 90% coverage gate), `.env.example` (names only;
  live port refused by design), `.gitignore` (data, models, journal and
  `.env` never committed).
