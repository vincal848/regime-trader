"""Interactive Brokers through IB Gateway (the official TWS API, via `ib_async`).

**Paper only, by construction:**
- the account ID must start with `DU` (IBKR's paper accounts);
- the API port must not be a live one (4001 for IB Gateway, 7496 for TWS);
- after connecting, every account the session can see must be a paper
  account, or the broker disconnects and refuses to run.

There are no API keys. You log in to IB Gateway yourself, or through IBC on
the server, and this code never sees a password or 2FA code. The TWS API
cannot withdraw funds.

**Orders** are marketable limit orders capped 5 bps through the reference
price, so a thin moment cannot fill far from the market. The `IbClient`
protocol lists exactly the IB methods used, so tests run against a fake.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import pandas as pd

from regime_trader.bars import COLUMNS, SESSION_CLOSE, SESSION_OPEN, TIMEZONE

LIVE_PORTS = frozenset({4001, 7496})
MAX_SLIPPAGE_BPS = 5.0


class PaperOnlyError(RuntimeError):
    """The connection would reach a live (non-paper) account."""


@dataclass(frozen=True)
class IbkrSettings:
    host: str
    port: int
    client_id: int
    account: str

    def __repr__(self) -> str:
        masked = f"{self.account[:2]}*****"
        return (
            f"IbkrSettings(host={self.host!r}, port={self.port}, "
            f"client_id={self.client_id}, account={masked!r})"
        )


def settings_from_env(env: Mapping[str, str]) -> IbkrSettings:
    settings = IbkrSettings(
        host=env.get("IB_HOST", "127.0.0.1"),
        port=int(env.get("IB_PORT", "4002")),
        client_id=int(env.get("IB_CLIENT_ID", "17")),
        account=env.get("IB_ACCOUNT", ""),
    )
    if settings.port in LIVE_PORTS:
        raise PaperOnlyError(f"port {settings.port} is a live-trading port; use the paper port (4002)")
    if not settings.account.startswith("DU"):
        raise PaperOnlyError("IB_ACCOUNT is not an IBKR paper account (paper IDs start with DU)")
    return settings


class IbClient(Protocol):
    def connect(self, host: str, port: int, clientId: int) -> Any: ...
    def disconnect(self) -> Any: ...
    def isConnected(self) -> bool: ...
    def managedAccounts(self) -> list[str]: ...
    def reqHistoricalData(
        self,
        contract: Any,
        endDateTime: Any,
        durationStr: str,
        barSizeSetting: str,
        whatToShow: str,
        useRTH: bool,
        formatDate: int,
    ) -> Any: ...
    def accountSummary(self, account: str) -> Any: ...
    def positions(self) -> Any: ...
    def placeOrder(self, contract: Any, order: Any) -> Any: ...


def _stock(symbol: str) -> Any:
    from ib_async import Stock

    return Stock(symbol, "SMART", "USD")


def _limit_order(action: str, quantity: int, price: float) -> Any:
    from ib_async import LimitOrder

    return LimitOrder(action, quantity, price)


class IbkrBroker:
    def __init__(
        self,
        client: IbClient,
        settings: IbkrSettings,
        contract: Callable[[str], Any] = _stock,
        limit_order: Callable[[str, int, float], Any] = _limit_order,
    ) -> None:
        self._client = client
        self.settings = settings
        self._contract = contract
        self._limit_order = limit_order

    def connect(self) -> None:
        self._client.connect(self.settings.host, self.settings.port, clientId=self.settings.client_id)
        accounts = list(self._client.managedAccounts())
        if self.settings.account not in accounts or not all(a.startswith("DU") for a in accounts):
            self._client.disconnect()
            raise PaperOnlyError("the session can see a non-paper account; refusing to trade")

    @property
    def connected(self) -> bool:
        return bool(self._client.isConnected())

    def history(self, symbol: str, years: int) -> pd.DataFrame:
        """Hourly RTH TRADES bars for the last `years` years, one year per request."""
        contract = self._contract(symbol)
        frames = []
        end: Any = ""
        for _ in range(years):
            bars = self._client.reqHistoricalData(contract, end, "1 Y", "1 hour", "TRADES", True, 2)
            if not bars:
                break
            frame = pd.DataFrame(
                {
                    "open": [b.open for b in bars],
                    "high": [b.high for b in bars],
                    "low": [b.low for b in bars],
                    "close": [b.close for b in bars],
                    "volume": [float(b.volume) for b in bars],
                },
                index=pd.DatetimeIndex(pd.to_datetime([b.date for b in bars], utc=True)).tz_convert(TIMEZONE),
            )
            frames.append(frame)
            end = bars[0].date
        history = pd.concat(frames).sort_index()
        history = history[~history.index.duplicated(keep="last")]
        index = pd.DatetimeIndex(history.index)
        clock = index - index.normalize()
        rth: pd.DataFrame = history[(clock >= SESSION_OPEN) & (clock < SESSION_CLOSE)][list(COLUMNS)]
        return rth

    def equity(self) -> float:
        for item in self._client.accountSummary(self.settings.account):
            if item.tag == "NetLiquidation" and item.currency == "USD":
                return float(item.value)
        raise RuntimeError("NetLiquidation missing from the account summary")

    def position(self, symbol: str) -> int:
        return int(
            sum(
                p.position
                for p in self._client.positions()
                if p.account == self.settings.account and p.contract.symbol == symbol
            )
        )

    def move_to_target(self, symbol: str, current: int, target: int, reference_price: float) -> Any:
        """Send one marketable limit order to move from `current` to `target`
        shares, or nothing if already there. Returns the IB trade handle."""
        delta = target - current
        if delta == 0:
            return None
        buying = delta > 0
        offset = MAX_SLIPPAGE_BPS / 10_000
        price = round(reference_price * (1 + offset if buying else 1 - offset), 2)
        order = self._limit_order("BUY" if buying else "SELL", abs(delta), price)
        return self._client.placeOrder(self._contract(symbol), order)
