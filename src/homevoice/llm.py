"""The model side: a small client interface and an Ollama client.

The broker never depends on which model or server answers. Today the client
points at Ollama; later it points at the fleet's LLM gateway with a token.
Both speak the same chat-with-tools shape, so only the client changes.
"""

from __future__ import annotations

import ipaddress
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict[str, Any]


@dataclass(frozen=True)
class Reply:
    text: str
    tool_calls: tuple[ToolCall, ...] = ()
    seconds: float = 0.0


class LLMClient(Protocol):
    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Reply: ...


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, f"refused redirect to {newurl}", headers, fp)


# No proxy (the environment's http_proxy must not see the conversation) and
# no redirects (a 302 must not carry a token or the conversation elsewhere).
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())


def check_base_url(url: str) -> None:
    """Plain http only to loopback; anything else must be https (the gateway)."""
    u = urllib.parse.urlsplit(url)
    if u.scheme == "https" and u.hostname:
        return
    if u.scheme == "http" and u.hostname:
        host = u.hostname
        if host == "localhost":
            return
        try:
            if ipaddress.ip_address(host).is_loopback:
                return
        except ValueError:
            pass
    raise ValueError(f"model URL must be http on loopback or https: {url!r}")


@dataclass
class OllamaClient:
    """Ollama's /api/chat, with tools, no streaming, thinking off for voice."""

    model: str
    base_url: str = "http://127.0.0.1:11434"
    timeout: float = 30.0
    think: bool = False
    headers: dict[str, str] = field(default_factory=dict)  # e.g. a gateway bearer token later

    def __post_init__(self) -> None:
        check_base_url(self.base_url)

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Reply:
        body = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "stream": False,
            "think": self.think,
            "options": {"temperature": 0},
        }
        req = urllib.request.Request(
            self.base_url.rstrip("/") + "/api/chat",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", **self.headers},
            method="POST",
        )
        start = time.monotonic()
        with _OPENER.open(req, timeout=self.timeout) as resp:
            data = json.load(resp)
        seconds = time.monotonic() - start
        msg = data.get("message") or {}
        calls = []
        for c in msg.get("tool_calls") or []:
            fn = c.get("function") or {}
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"_unparsed": args}
            if not isinstance(args, dict):
                args = {"_unparsed": args}
            calls.append(ToolCall(str(fn.get("name", "")), args))
        return Reply(str(msg.get("content") or ""), tuple(calls), seconds)
