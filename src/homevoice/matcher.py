"""The intent matcher: fixed sentences that run without the model.

Common household commands ("timer 10 minutes", "turn off the kitchen lamp",
"add milk to the list") match here and go straight to the broker. That is
faster than a model round trip, and it removes model mishears for them.

The matcher only ever proposes actions that the policy puts in class 0 or 1.
It has no sentence for anything above that, so "unlock the door" can never
match here; it goes to the model, and the broker decides.
"""

from __future__ import annotations

import re
from typing import Callable

from .broker import Proposal, Source
from .policy import HOUSEHOLD, Policy

_NUMBERS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "fifteen": 15,
    "twenty": 20, "thirty": 30, "forty": 40, "forty five": 45, "sixty": 60,
}


def _num(s: str) -> int | None:
    s = s.strip().lower()
    if s.isdigit():
        return int(s)
    return _NUMBERS.get(s)


def _clean(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"^(please|hey|ok|okay)\s+", "", text)
    text = re.sub(r"\s+please$", "", text)
    return " ".join(text.split())


Rule = Callable[[re.Match[str], Policy], Proposal | None]


def _timer(m: re.Match[str], p: Policy) -> Proposal | None:
    n = _num(m.group("n"))
    if n is None:
        return None
    minutes = n * 60 if m.group("unit").startswith("hour") else n
    return Proposal("timer.start", {"minutes": minutes}, Source.MATCHER)


def _light(m: re.Match[str], p: Policy) -> Proposal | None:
    spoken = m.group("dev")
    d = p.device_by_spoken_name(spoken, kind="light")
    if d is None:
        d = p.device_by_spoken_name(re.sub(r" lights?$", "", spoken), kind="light")
    if d is None:
        return None
    return Proposal("light.set", {"device": d.name, "state": m.group("state")}, Source.MATCHER)


def _list_add(m: re.Match[str], p: Policy) -> Proposal | None:
    return Proposal("list.add", {"item": m.group("item")}, Source.MATCHER)


def _lock(m: re.Match[str], p: Policy) -> Proposal | None:
    d = p.device_by_spoken_name(m.group("dev"), kind="lock")
    if d is None:
        return None
    return Proposal("lock.lock", {"device": d.name}, Source.MATCHER)


def _undo(m: re.Match[str], p: Policy) -> Proposal | None:
    return Proposal("undo", {}, Source.MATCHER)


_RULES: list[tuple[re.Pattern[str], Rule]] = [
    (re.compile(r"^(?:set (?:a |the )?)?timer (?:for )?(?P<n>\w+(?: five)?) (?P<unit>minutes?|hours?)$"), _timer),
    (re.compile(r"^(?:set (?:a |an )?)?(?P<n>\w+(?: five)?) (?P<unit>minutes?|hours?) timer$"), _timer),
    (re.compile(r"^turn (?P<state>on|off) (?:the )?(?P<dev>[\w' ]+)$"), _light),
    (re.compile(r"^turn (?:the )?(?P<dev>[\w' ]+?) (?P<state>on|off)$"), _light),
    (re.compile(r"^(?:lights? )?(?P<state>on|off) (?:in )?(?:the )?(?P<dev>[\w' ]+)$"), _light),
    (re.compile(r"^add (?P<item>[\w' ]{1,60}?) to (?:the |my )?(?:shopping |grocery )?list$"), _list_add),
    (re.compile(r"^lock (?:the )?(?P<dev>[\w' ]+)$"), _lock),
    (re.compile(r"^undo(?: that)?$"), _undo),
]


def match(text: str, policy: Policy) -> Proposal | None:
    """Return a proposal for a fixed sentence, or None to hand it to the model.

    A match is dropped when the policy does not have the action, or has it
    above class 1. The broker checks again either way.
    """
    t = _clean(text)
    for pattern, rule in _RULES:
        m = pattern.match(t)
        if not m:
            continue
        prop = rule(m, policy)
        if prop is None:
            continue
        action = policy.actions.get(prop.action)
        if action is None or action.cls > HOUSEHOLD:
            continue
        if any(policy.devices[v].cls is not None and policy.devices[v].cls > HOUSEHOLD
               for k, v in prop.args.items() if k == "device" and v in policy.devices):
            continue
        return prop
    return None
