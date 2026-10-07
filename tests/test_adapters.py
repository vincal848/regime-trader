"""Step 8: adapters, the only modules that do I/O, tested against fakes."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from synthetic import make_bars

from regime_trader.alerts import AlertError, NullAlerts, TelegramAlerts
from regime_trader.engine import Fill
from regime_trader.hmm import HmmModel, RegimeModel
from regime_trader.ibkr import IbkrBroker, PaperOnlyError, settings_from_env
from regime_trader.llm import (
    OPUS_5_5,
    BudgetExceededError,
    NightlyReviewer,
    RawReview,
    ReviewRefusedError,
    SpendLedger,
    Usage,
    cost_usd,
)
from regime_trader.store import BarCache, Journal, load_model, load_playbooks, save_model
from regime_trader.yahoo import normalize_yahoo

REPO = Path(__file__).resolve().parents[1]

# --- store ----------------------------------------------------------------------------


def test_bar_cache_merges_new_bars_and_keeps_the_latest_values(tmp_path: Path) -> None:
    cache = BarCache(tmp_path)
    bars = make_bars(5)
    cache.save("SPY", bars.iloc[:20])
    revised = bars.iloc[15:].copy()
    revised.loc[revised.index[0], "volume"] = 123.0
    cache.save("SPY", revised)
    loaded = cache.load("SPY")
    assert len(loaded) == len(bars)
    assert loaded.loc[revised.index[0], "volume"] == 123.0
    assert cache.load("QQQ").empty


def test_playbooks_load_by_state_and_reject_a_misnamed_file(tmp_path: Path) -> None:
    playbooks = load_playbooks(REPO / "playbooks")
    assert set(playbooks) == {"CALM_UP", "CHOP", "STRESS", "CRASH"}
    text = (REPO / "playbooks" / "CHOP.md").read_text(encoding="utf-8")
    (tmp_path / "CALM_UP.md").write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="CHOP"):
        load_playbooks(tmp_path)


def test_models_round_trip_through_json(tmp_path: Path) -> None:
    model = RegimeModel(
        hmm=HmmModel(
            np.array([0.5, 0.5]),
            np.array([[0.9, 0.1], [0.2, 0.8]]),
            np.zeros((2, 5)),
            np.array([np.eye(5)] * 2),
        ),
        labels=("CALM_UP", "CRASH"),
        return_mean=np.array([0.001, -0.002]),
        return_vol=np.array([0.002, 0.01]),
    )
    save_model(tmp_path / "model.json", model)
    loaded = load_model(tmp_path / "model.json")
    assert loaded.labels == model.labels
    np.testing.assert_array_equal(loaded.hmm.covars, model.hmm.covars)


def test_journal_records_decisions_fills_and_events(tmp_path: Path) -> None:
    journal = Journal(tmp_path / "journal.db")
    ts = pd.Timestamp("2026-01-05 11:00", tz="America/New_York")
    journal.record_decision(
        ts=ts,
        labels=("CALM_UP", "CRASH"),
        probabilities=np.array([0.9, 0.1]),
        next_state=np.array([0.85, 0.15]),
        active="CALM_UP",
        target_shares=40,
        order="approved",
        reasons=("switch None -> CALM_UP",),
    )
    journal.record_fill(Fill(ts, ts + pd.Timedelta(hours=1), 40, 500.05, 0.35))
    journal.record_event(ts, "switch", "None -> CALM_UP")
    decisions = journal.decisions()
    assert decisions.iloc[0]["active"] == "CALM_UP"
    assert json.loads(decisions.iloc[0]["probabilities"]) == {"CALM_UP": 0.9, "CRASH": 0.1}
    assert journal.fills().iloc[0]["shares"] == 40
    assert journal.events().iloc[0]["kind"] == "switch"


# --- ibkr ------------------------------------------------------------------------------

PAPER_ENV = {"IB_HOST": "127.0.0.1", "IB_PORT": "4002", "IB_CLIENT_ID": "17", "IB_ACCOUNT": "DU1234567"}


def test_settings_accept_only_a_paper_account_on_a_paper_port() -> None:
    settings = settings_from_env(PAPER_ENV)
    assert (settings.port, settings.account) == (4002, "DU1234567")
    with pytest.raises(PaperOnlyError, match="paper"):
        settings_from_env({**PAPER_ENV, "IB_ACCOUNT": "U7654321"})
    with pytest.raises(PaperOnlyError, match="port"):
        settings_from_env({**PAPER_ENV, "IB_PORT": "4001"})


def test_settings_repr_masks_the_account() -> None:
    assert "1234567" not in repr(settings_from_env(PAPER_ENV))


@dataclass
class FakeBar:
    date: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class FakeIb:
    accounts: list[str] = field(default_factory=lambda: ["DU1234567"])
    orders: list[tuple[Any, Any]] = field(default_factory=list)
    connected: bool = False

    def connect(self, host: str, port: int, clientId: int) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def isConnected(self) -> bool:
        return self.connected

    def managedAccounts(self) -> list[str]:
        return self.accounts

    def reqHistoricalData(
        self,
        contract: Any,
        endDateTime: Any,
        durationStr: str,
        barSizeSetting: str,
        whatToShow: str,
        useRTH: bool,
        formatDate: int,
    ) -> list[FakeBar]:
        start = datetime(2026, 1, 5, 14, 30, tzinfo=UTC)
        return [FakeBar(start + pd.Timedelta(hours=h), 500.0, 501.0, 499.0, 500.5, 1e6) for h in range(3)]

    def accountSummary(self, account: str) -> list[Any]:
        @dataclass
        class Item:
            tag: str
            value: str
            currency: str

        return [Item("NetLiquidation", "100250.5", "USD"), Item("BuyingPower", "400000", "USD")]

    def positions(self) -> list[Any]:
        @dataclass
        class Contract:
            symbol: str

        @dataclass
        class Position:
            account: str
            contract: Contract
            position: float

        return [Position("DU1234567", Contract("SPY"), 40.0)]

    def placeOrder(self, contract: Any, order: Any) -> Any:
        self.orders.append((contract, order))
        return order


@dataclass
class FakeOrder:
    action: str
    totalQuantity: int
    lmtPrice: float


def _broker(fake: FakeIb) -> IbkrBroker:
    return IbkrBroker(
        fake,
        settings_from_env(PAPER_ENV),
        contract=lambda symbol: symbol,
        limit_order=lambda action, quantity, price: FakeOrder(action, quantity, price),
    )


def test_connecting_to_a_live_account_is_refused() -> None:
    fake = FakeIb(accounts=["U7654321"])
    with pytest.raises(PaperOnlyError):
        _broker(fake).connect()
    assert not fake.connected


def test_broker_reads_equity_position_and_history() -> None:
    broker = _broker(FakeIb())
    broker.connect()
    assert broker.equity() == pytest.approx(100250.5)
    assert broker.position("SPY") == 40
    history = broker.history("SPY", years=1)
    assert list(history.columns) == ["open", "high", "low", "close", "volume"]
    assert str(pd.DatetimeIndex(history.index).tz) == "America/New_York"
    assert history.index[0].hour == 9
    assert history.index[0].minute == 30


def test_orders_are_marketable_limits_capped_at_5_bps() -> None:
    fake = FakeIb()
    broker = _broker(fake)
    broker.connect()
    broker.move_to_target("SPY", current=40, target=100, reference_price=500.0)
    broker.move_to_target("SPY", current=100, target=0, reference_price=500.0)
    (_, buy), (_, sell) = fake.orders
    assert (buy.action, buy.totalQuantity, buy.lmtPrice) == ("BUY", 60, 500.25)
    assert (sell.action, sell.totalQuantity, sell.lmtPrice) == ("SELL", 100, 499.75)


def test_no_order_when_already_on_target() -> None:
    fake = FakeIb()
    broker = _broker(fake)
    broker.connect()
    assert broker.move_to_target("SPY", current=40, target=40, reference_price=500.0) is None
    assert fake.orders == []


# --- yahoo -------------------------------------------------------------------------------


def test_yahoo_frames_are_normalized_to_the_bar_schema() -> None:
    index = pd.DatetimeIndex(["2026-01-05 09:30", "2026-01-05 10:30", "2026-01-05 16:30"]).tz_localize(
        "America/New_York"
    )
    columns = pd.MultiIndex.from_product([["Open", "High", "Low", "Close", "Volume"], ["SPY"]])
    raw = pd.DataFrame(np.tile([500.0, 501.0, 499.0, 500.5, 1e6], (3, 1)), index=index, columns=columns)
    bars = normalize_yahoo(raw)
    assert list(bars.columns) == ["open", "high", "low", "close", "volume"]
    assert len(bars) == 2  # the 16:30 bar is outside regular hours


# --- alerts --------------------------------------------------------------------------------

TOKEN = "123456:SECRET-TOKEN"


def test_telegram_alerts_post_a_tagged_message() -> None:
    sent: list[tuple[str, dict[str, str]]] = []

    def sender(url: str, payload: dict[str, str]) -> int:
        sent.append((url, payload))
        return 200

    alerts = TelegramAlerts(TOKEN, "42", sender=sender)
    assert alerts.send("switch", "CHOP -> CALM_UP (p=0.91)")
    url, payload = sent[0]
    assert url.endswith("/sendMessage")
    assert payload == {"chat_id": "42", "text": "[SWITCH] CHOP -> CALM_UP (p=0.91)"}


def test_the_bot_token_never_leaks() -> None:
    def failing(url: str, payload: dict[str, str]) -> int:
        raise OSError(f"connection to {url} refused")

    alerts = TelegramAlerts(TOKEN, "42", sender=failing)
    assert TOKEN not in repr(alerts)
    with pytest.raises(AlertError) as raised:
        alerts.send("error", "boom")
    assert "SECRET" not in str(raised.value)


def test_alerts_are_outbound_only_and_reject_unknown_kinds() -> None:
    alerts = TelegramAlerts(TOKEN, "42", sender=lambda url, payload: 200)
    public = {name for name in dir(alerts) if not name.startswith("_")}
    assert public == {"send"}
    with pytest.raises(ValueError, match="kind"):
        alerts.send("buy now", "please")


def test_null_alerts_send_nothing() -> None:
    assert NullAlerts().send("fill", "x") is False


# --- llm ------------------------------------------------------------------------------------


def test_cost_uses_per_token_type_prices() -> None:
    usage = Usage(
        input_tokens=1_000_000,
        output_tokens=100_000,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=1_000_000,
    )
    assert cost_usd(usage, "claude-opus-5-5") == pytest.approx(4.00 + 2.00 + 0.20)


def test_an_unexpected_fallback_model_is_priced_at_the_highest_known_rate() -> None:
    usage = Usage(1_000_000, 0, 0, 0)
    assert cost_usd(usage, "some-other-model") == pytest.approx(10.00)


@dataclass
class FakeClient:
    responses: Sequence[RawReview]
    calls: int = 0

    def complete(self, system: str, prompt: str, max_tokens: int) -> RawReview:
        self.calls += 1
        return self.responses[self.calls - 1]


def _raw(stop_reason: str = "end_turn") -> RawReview:
    return RawReview("post-mortem and proposals", "claude-opus-5-5", stop_reason, Usage(20_000, 4_000, 0, 0))


def test_the_ledger_persists_spend_per_month(tmp_path: Path) -> None:
    SpendLedger(tmp_path / "spend.json").add("2026-10", 1.25)
    ledger = SpendLedger(tmp_path / "spend.json")
    ledger.add("2026-10", 0.75)
    assert ledger.spent("2026-10") == pytest.approx(2.0)
    assert ledger.spent("2026-11") == 0.0


def test_a_review_records_its_actual_cost(tmp_path: Path) -> None:
    ledger = SpendLedger(tmp_path / "spend.json")
    reviewer = NightlyReviewer(FakeClient([_raw()]), ledger, monthly_budget_usd=20.0)
    review = reviewer.review("today's record", now=datetime(2026, 10, 7, tzinfo=UTC))
    assert review.text == "post-mortem and proposals"
    assert review.cost_usd == pytest.approx(20_000 * 4e-6 + 4_000 * 20e-6)
    assert ledger.spent("2026-10") == pytest.approx(review.cost_usd)


def test_the_budget_blocks_a_call_before_it_is_made(tmp_path: Path) -> None:
    ledger = SpendLedger(tmp_path / "spend.json")
    ledger.add("2026-10", 19.50)
    client = FakeClient([_raw()])
    reviewer = NightlyReviewer(client, ledger, monthly_budget_usd=20.0, max_tokens=32_000)
    with pytest.raises(BudgetExceededError):
        reviewer.review("today's record", now=datetime(2026, 10, 7, tzinfo=UTC))
    assert client.calls == 0


def test_a_refusal_is_raised_and_still_billed(tmp_path: Path) -> None:
    ledger = SpendLedger(tmp_path / "spend.json")
    reviewer = NightlyReviewer(FakeClient([_raw("refusal")]), ledger, monthly_budget_usd=20.0)
    with pytest.raises(ReviewRefusedError):
        reviewer.review("today's record", now=datetime(2026, 10, 7, tzinfo=UTC))
    assert ledger.spent("2026-10") > 0


def test_opus_pricing_constants() -> None:
    assert (OPUS_5_5.input, OPUS_5_5.output, OPUS_5_5.cache_read) == (4.00, 20.00, 0.20)
