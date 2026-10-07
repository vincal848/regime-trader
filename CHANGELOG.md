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

### Step 2: bars and features
- Tests first (`tests/test_bars_features.py`):
  - seven kinds of bad bar are rejected;
  - bar-of-day works on both the IBKR and Yahoo hourly grids;
  - **no feature at t changes when any later bar changes** (8 random cut
    points, prices and volumes perturbed);
  - z-scores use only statistics through t−1;
  - the volume ratio compares like-for-like hours.
- Implementation: `bars.py` (schema, `validate_bars`, `bar_of_day`) and
  `features.py` (five raw features, expanding z-scores shifted by one bar).
- Test correction: the warm-up expectation was 30 sessions, but the volume
  ratio (20 sessions) followed by its own z-score history (140 bars) needs
  about 41.
