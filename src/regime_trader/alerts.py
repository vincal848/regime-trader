"""Outbound-only Telegram alerts (spec §14).

The only public method is `send`. Nothing here polls, receives updates or
exposes a webhook, so no chat message can ever reach the trading code. The
bot token is never shown in a repr, and never appears in an error message
or a chained traceback.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from collections.abc import Callable
from typing import Protocol

KINDS = frozenset({"fill", "error", "switch", "drift", "kill", "approval", "report"})
Sender = Callable[[str, dict[str, str]], int]


class AlertError(RuntimeError):
    """An alert could not be delivered (message redacted)."""


class Alerts(Protocol):
    def send(self, kind: str, text: str) -> bool: ...


def _post(url: str, payload: dict[str, str]) -> int:
    request = urllib.request.Request(url, data=urllib.parse.urlencode(payload).encode(), method="POST")
    with urllib.request.urlopen(request, timeout=10) as response:
        json.load(response)
        return int(response.status)


class TelegramAlerts:
    def __init__(self, token: str, chat_id: str, sender: Sender = _post) -> None:
        self._token = token
        self._chat_id = chat_id
        self._sender = sender

    def __repr__(self) -> str:
        return f"TelegramAlerts(chat_id={self._chat_id!r}, token=<redacted>)"

    def send(self, kind: str, text: str) -> bool:
        if kind not in KINDS:
            raise ValueError(f"unknown alert kind {kind!r}; expected one of {sorted(KINDS)}")
        url = f"https://api.telegram.org/bot{self._token}/sendMessage"
        payload = {"chat_id": self._chat_id, "text": f"[{kind.upper()}] {text}"}
        try:
            status = self._sender(url, payload)
        except Exception as error:
            message = str(error).replace(self._token, "<redacted>")
            raise AlertError(f"telegram alert failed: {message}") from None
        return status == 200


class NullAlerts:
    """Used when Telegram is not configured: alerts still go to the journal."""

    def send(self, kind: str, text: str) -> bool:
        return False
