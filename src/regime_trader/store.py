"""Local persistence: the bar cache, the journal, fitted models and playbooks.

- **Bar cache.** One Parquet file per symbol. A save merges new bars into
  the existing ones; a revised bar replaces the cached one.
- **Journal.** SQLite. Every decision (with its full probability vector and
  reasons), every fill, and every event (switches, alerts, kill-switch
  firings, drift warnings). The dashboard and the nightly review read it;
  nothing else writes it.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

from regime_trader.bars import COLUMNS, validate_bars
from regime_trader.engine import Fill
from regime_trader.hmm import FloatArray, HmmModel, RegimeModel
from regime_trader.playbook import Playbook, parse_playbook
from regime_trader.refit import Fit


class BarCache:
    def __init__(self, root: Path) -> None:
        self.root = root

    def _path(self, symbol: str) -> Path:
        return self.root / f"{symbol}_1h.parquet"

    def load(self, symbol: str) -> pd.DataFrame:
        path = self._path(symbol)
        if not path.exists():
            return pd.DataFrame(columns=list(COLUMNS), index=pd.DatetimeIndex([], tz="America/New_York"))
        return pd.read_parquet(path)

    def save(self, symbol: str, bars: pd.DataFrame) -> pd.DataFrame:
        merged = pd.concat([self.load(symbol), bars[list(COLUMNS)]])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        validate_bars(merged)
        self.root.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(self._path(symbol))
        return merged


def load_playbooks(directory: Path) -> dict[str, Playbook]:
    """Every `<STATE>.md` in `directory`, keyed by the state it declares.
    A file must be named after its declared state."""
    playbooks = {}
    for path in sorted(directory.glob("*.md")):
        playbook = parse_playbook(path.read_text(encoding="utf-8"))
        if playbook.state != path.stem:
            raise ValueError(f"{path.name} declares state {playbook.state}; rename the file or fix the state")
        playbooks[playbook.state] = playbook
    return playbooks


def _model_payload(model: RegimeModel) -> dict[str, object]:
    return {
        "startprob": model.hmm.startprob.tolist(),
        "transmat": model.hmm.transmat.tolist(),
        "means": model.hmm.means.tolist(),
        "covars": model.hmm.covars.tolist(),
        "labels": list(model.labels),
        "return_mean": model.return_mean.tolist(),
        "return_vol": model.return_vol.tolist(),
    }


def _array(data: dict[str, object], key: str) -> FloatArray:
    return np.asarray(data[key], dtype=np.float64)


def _model_from(data: dict[str, object]) -> RegimeModel:
    hmm = HmmModel(
        _array(data, "startprob"), _array(data, "transmat"), _array(data, "means"), _array(data, "covars")
    )
    labels = data["labels"]
    assert isinstance(labels, list)
    return RegimeModel(
        hmm, tuple(str(label) for label in labels), _array(data, "return_mean"), _array(data, "return_vol")
    )


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_model(path: Path, model: RegimeModel) -> None:
    _write_json(path, _model_payload(model))


def load_model(path: Path) -> RegimeModel:
    return _model_from(json.loads(path.read_text(encoding="utf-8")))


def save_fit(path: Path, fit: Fit) -> None:
    payload = _model_payload(fit.model)
    payload |= {
        "kelly": fit.kelly,
        "insample_ll": fit.insample_ll.tolist(),
        "prior": fit.prior.tolist(),
        "trained_through": fit.trained_through.isoformat(),
    }
    _write_json(path, payload)


def load_fit(path: Path) -> Fit:
    data = json.loads(path.read_text(encoding="utf-8"))
    return Fit(
        model=_model_from(data),
        kelly={str(k): float(v) for k, v in data["kelly"].items()},
        insample_ll=_array(data, "insample_ll"),
        prior=_array(data, "prior"),
        trained_through=pd.Timestamp(data["trained_through"]),
    )


_SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    ts TEXT NOT NULL, active TEXT, probabilities TEXT NOT NULL, next_state TEXT NOT NULL,
    target_shares INTEGER NOT NULL, order_status TEXT NOT NULL, reasons TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS fills (
    decision_ts TEXT NOT NULL, fill_ts TEXT NOT NULL, shares INTEGER NOT NULL,
    price REAL NOT NULL, commission REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS events (ts TEXT NOT NULL, kind TEXT NOT NULL, message TEXT NOT NULL);
"""


class Journal:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path)
        try:
            with db:
                yield db
        finally:
            db.close()

    def record_decision(
        self,
        *,
        ts: pd.Timestamp,
        labels: Sequence[str],
        probabilities: FloatArray,
        next_state: FloatArray,
        active: str | None,
        target_shares: int,
        order: str,
        reasons: Sequence[str],
    ) -> None:
        def by_label(values: FloatArray) -> str:
            return json.dumps({label: round(float(p), 6) for label, p in zip(labels, values, strict=True)})

        with self._connect() as db:
            db.execute(
                "INSERT INTO decisions VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    ts.isoformat(),
                    active,
                    by_label(probabilities),
                    by_label(next_state),
                    target_shares,
                    order,
                    json.dumps(list(reasons)),
                ),
            )

    def record_fill(self, fill: Fill) -> None:
        with self._connect() as db:
            db.execute(
                "INSERT INTO fills VALUES (?, ?, ?, ?, ?)",
                (
                    fill.decision_ts.isoformat(),
                    fill.fill_ts.isoformat(),
                    fill.shares,
                    fill.price,
                    fill.commission,
                ),
            )

    def record_event(self, ts: pd.Timestamp, kind: str, message: str) -> None:
        with self._connect() as db:
            db.execute("INSERT INTO events VALUES (?, ?, ?)", (ts.isoformat(), kind, message))

    def _read(self, table: str) -> pd.DataFrame:
        with self._connect() as db:
            return pd.read_sql_query(f"SELECT * FROM {table}", db)

    def decisions(self) -> pd.DataFrame:
        return self._read("decisions")

    def fills(self) -> pd.DataFrame:
        return self._read("fills")

    def events(self) -> pd.DataFrame:
        return self._read("events")
