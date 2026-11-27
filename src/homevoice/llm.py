"""The model side: a small client interface and an Ollama client.

The broker never depends on which model or server answers. Today the client
points at Ollama; later it points at the fleet's LLM gateway with a token.
Both speak the same chat-with-tools shape, so only the client changes.
"""

from __future__ import annotations

import ipaddress
import json
import os
import stat
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


PRIORITY_TIERS = (0, 1, 2)  # batch, chat, voice
MAX_KEY_LEN = 4096


class Secret:
    """Holds a key so that repr, str, asdict and vars never show it."""

    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        self._value = value

    def reveal(self) -> str:
        return self._value

    def __repr__(self) -> str:
        return "Secret(***)"

    __str__ = __repr__

    def __deepcopy__(self, memo: dict) -> "Secret":
        return self

    def __reduce__(self):
        raise TypeError("a Secret cannot be pickled")


def read_priority_key(path: str) -> str:
    """The key that lets the server honor a priority tier.

    It must be a regular file that only its owner can read, so a mistake in
    deployment shows up here instead of as a key every user on the box holds.
    """
    # O_NONBLOCK so a FIFO at the path cannot hang us before the check below.
    flags = os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NOCTTY", 0)
    fd = os.open(path, flags)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise ValueError(f"priority key {path!r} is not a regular file")
        if st.st_mode & 0o077:
            raise ValueError(f"priority key {path!r} is readable by others; chmod 600 it")
        if st.st_uid != os.geteuid():
            raise ValueError(f"priority key {path!r} is not owned by this user")
        if st.st_size > MAX_KEY_LEN + 2:
            raise ValueError(f"priority key {path!r} is too long")
        with os.fdopen(fd, "rb") as f:
            fd = -1
            raw = f.read(MAX_KEY_LEN + 3)
    finally:
        if fd >= 0:
            os.close(fd)
    try:
        key = raw.decode("utf-8").strip()
    except UnicodeDecodeError:
        raise ValueError(f"priority key {path!r} is not UTF-8") from None
    if not key or len(key) > MAX_KEY_LEN or any(c.isspace() for c in key):
        raise ValueError(f"priority key {path!r} is empty, too long or has spaces")
    return key


@dataclass(frozen=True)
class OllamaClient:
    """Ollama's /api/chat, with tools, no streaming, thinking off for voice.

    ``priority`` asks the server for a scheduling tier (2 is voice). A server
    that supports tiers honors one only with the matching key, so the tier is
    granted by holding the key file, never by asking. The client is frozen, so
    its URL cannot change after the check, and the key is held as a Secret.
    """

    model: str
    base_url: str = "http://127.0.0.1:11434"
    timeout: float = 30.0
    think: bool = False
    headers: dict[str, str] = field(default_factory=dict, repr=False)  # e.g. a gateway bearer token later
    priority: int | None = None
    priority_key_file: str | None = None
    _priority_key: Secret | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        check_base_url(self.base_url)
        if self.priority is None or self.priority == 0:
            if self.priority_key_file is not None:
                raise ValueError("priority_key_file needs a priority above 0")
            if self.priority is None:
                return
        if type(self.priority) is not int or self.priority not in PRIORITY_TIERS:
            raise ValueError(f"priority must be one of {PRIORITY_TIERS}")
        if self.priority > 0:
            if self.priority_key_file is None:
                raise ValueError("a priority above 0 needs priority_key_file")
            object.__setattr__(self, "_priority_key", Secret(read_priority_key(self.priority_key_file)))

    def _options(self) -> dict[str, Any]:
        options: dict[str, Any] = {"temperature": 0}
        if self.priority is not None:
            options["priority"] = self.priority
        if self._priority_key is not None:
            options["priority_key"] = self._priority_key.reveal()
        return options

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Reply:
        check_base_url(self.base_url)  # again, in case of object.__setattr__
        body = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "stream": False,
            "think": self.think,
            "options": self._options(),
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
