"""The model side: a small client interface and an Ollama client.

The broker never depends on which model or server answers. Today the client
points at Ollama; later it points at the fleet's LLM gateway with a token.
Both speak the same chat-with-tools shape, so only the client changes.
"""

from __future__ import annotations

import json
import time
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


@dataclass
class OllamaClient:
    """Ollama's /api/chat, with tools, no streaming, thinking off for voice."""

    model: str
    base_url: str = "http://127.0.0.1:11434"
    timeout: float = 30.0
    think: bool = False
    headers: dict[str, str] = field(default_factory=dict)  # e.g. a gateway bearer token later

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
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310 - fixed URL scheme from config
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
