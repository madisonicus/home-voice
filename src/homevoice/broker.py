"""The action broker: the only place that decides what voice may do.

The model never holds a tool. It, or the intent matcher, proposes an action;
`Broker.decide` checks it against the policy and the session, and returns a
Decision. Only an `ALLOW` may be carried out here. `HANDOFF` means: send the
rendered fields to the person's phone, where they finish it with their own
app or passkey. The broker never holds the rights a handoff needs.

Rules, in order (see the design doc, "What it may do" and "How the rules are
enforced"):

1. Mode `off` refuses everything; `answer-only` allows class 0 only.
2. An unknown action, or arguments that do not match the policy, is refused.
3. The class is the highest of the action's and its devices'.
4. Class 4 is refused, with where to do it instead. Then the room caps the class.
5. Class 3 is a handoff to the person with an open personal window.
6. Class 2 needs an open personal window and no web content in the session.
7. Class 1 from the model is refused when the session is tainted. The intent
   matcher acts on the person's own words, so it is not.
8. Class 0 is allowed. Outbound class 0 (web search) is refused while the
   session holds personal data.

Class 2 and 3 also need the identity provider: no Rauthy, no approvals, and
never a spoken yes instead.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from .mode import Mode, read_mode
from .policy import ANSWER, HANDOFF, HOUSEHOLD, NEVER, PERSONAL, Action, Policy


class Source(str, Enum):
    MATCHER = "matcher"  # fixed sentences, from the person's own words
    MODEL = "model"  # proposed by the language model


class Verdict(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    APPROVAL_NEEDED = "approval_needed"  # ask the person's phone for a personal window
    HANDOFF = "handoff"  # send the rendered fields to the person's phone


@dataclass(frozen=True)
class Proposal:
    action: str
    args: dict[str, Any] = field(default_factory=dict)
    source: Source = Source.MODEL


@dataclass(frozen=True)
class Decision:
    verdict: Verdict
    action: str
    cls: int | None
    reason: str  # for the decision log
    say: str  # one line for the speaker; never empty on a refusal
    args: dict[str, Any] = field(default_factory=dict)  # typed, checked values
    person: str | None = None  # for HANDOFF and class 2

    @property
    def allowed(self) -> bool:
        return self.verdict is Verdict.ALLOW


@dataclass
class Session:
    """One conversation in one room. Taint lasts until reset()."""

    room: str
    tainted: bool = False  # untrusted text reached the model's context
    personal_data: bool = False  # class 2 output reached the model's context
    web: bool = False  # outbound web content reached the model's context
    person: str | None = None  # whose personal window is open
    window_last_speech: float | None = None

    def reset(self) -> None:
        self.tainted = False
        self.personal_data = False
        self.web = False
        self.end_window()

    def end_window(self) -> None:
        self.person = None
        self.window_last_speech = None


@dataclass
class _Done:
    at: float
    room: str
    action: str
    args: dict[str, Any]


class Broker:
    def __init__(
        self,
        policy: Policy,
        mode_path: str | Path | None,
        idp_up: Callable[[], bool] = lambda: False,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.policy = policy
        self.mode_path = mode_path
        self.idp_up = idp_up
        self.clock = clock
        self._sessions: list[Session] = []
        self._last_mode: Mode | None = None
        self._done: list[_Done] = []

    # -- sessions and windows -------------------------------------------------

    def new_session(self, room: str) -> Session:
        if room not in self.policy.rooms:
            raise ValueError(f"unknown room {room!r}")
        s = Session(room=room)
        self._sessions.append(s)
        return s

    def open_window(self, session: Session, person: str) -> None:
        """Called only after the person approved on their phone (passkey)."""
        session.person = person
        session.window_last_speech = self.clock()

    def heard_speech(self, session: Session) -> None:
        if session.person is not None:
            session.window_last_speech = self.clock()

    def mute_pressed(self, room: str) -> None:
        for s in self._sessions:
            if s.room == room:
                s.end_window()

    def _window_open(self, session: Session) -> bool:
        if session.person is None or session.window_last_speech is None:
            return False
        if self.clock() - session.window_last_speech > self.policy.personal_idle_seconds:
            session.end_window()
            return False
        return True

    def _check_mode(self) -> Mode:
        mode = read_mode(self.mode_path)
        if self._last_mode is not None and mode != self._last_mode:
            # A flip of the switch ends every personal window.
            for s in self._sessions:
                s.end_window()
        self._last_mode = mode
        return mode

    # -- the decision ---------------------------------------------------------

    def decide(self, session: Session, proposal: Proposal) -> Decision:
        mode = self._check_mode()
        name = proposal.action

        if mode is Mode.OFF:
            return self._deny(name, None, "mode off", "Voice is switched off.")

        action = self.policy.actions.get(name)
        if action is None:
            return self._deny(name, None, "unknown action", "I can't do that.")

        ok, args, why, devices = self._check_args(action, proposal.args)
        if not ok:
            return self._deny(name, action.cls, f"bad arguments: {why}", "I didn't get that. Please say it again.")

        cls = max([action.cls] + [d.cls for d in devices if d.cls is not None])

        if mode is Mode.ANSWER_ONLY and cls != ANSWER:
            return self._deny(name, cls, "mode answer-only", "I can only answer questions right now.")

        if cls == NEVER:
            where = action.instead or "Slack or Claude"
            return self._deny(name, cls, "class 4", f"I don't do that by voice. Use {where}.")

        cap = self.policy.rooms[session.room].max_class
        if cls > cap:
            return self._deny(name, cls, f"room cap {cap}", "I can't do that from this room.")

        if cls == HANDOFF:
            if not self.idp_up():
                return self._deny(name, cls, "identity provider down", "I can't reach your phone right now.")
            if not self._window_open(session):
                return Decision(Verdict.APPROVAL_NEEDED, name, cls, "class 3 needs a person",
                                "Who is this for? Check your phone.", args)
            return Decision(Verdict.HANDOFF, name, cls, "class 3 handoff",
                            f"It's on your phone, {session.person}.", args, session.person)

        if cls == PERSONAL:
            if not self.idp_up():
                return self._deny(name, cls, "identity provider down", "I can't reach your phone right now.")
            if session.web:
                return self._deny(name, cls, "personal data and web content never meet",
                                  "Say the wake word again to start fresh, then ask me.")
            if not self._window_open(session):
                return Decision(Verdict.APPROVAL_NEEDED, name, cls, "class 2 needs a personal window",
                                "Check your phone to approve.", args)
            return self._allow(session, action, cls, args, f"For {session.person}:")

        if cls == HOUSEHOLD:
            if proposal.source is Source.MODEL and session.tainted:
                return self._deny(name, cls, "tainted session, model proposal",
                                  "Say that again in your own words and I'll do it.")
            return self._allow(session, action, cls, args, "")

        # Class 0.
        if action.outbound and session.personal_data:
            return self._deny(name, cls, "personal data and web content never meet",
                              "I can't search while your personal data is open. Say the wake word again first.")
        return self._allow(session, action, cls, args, "")

    def _allow(self, session: Session, action: Action, cls: int, args: dict[str, Any], say: str) -> Decision:
        if action.untrusted_output:
            session.tainted = True
        if action.personal_output:
            session.personal_data = True
        if action.outbound:
            session.web = True
            session.tainted = True
        if cls == HOUSEHOLD:
            self._done.append(_Done(self.clock(), session.room, action.name, dict(args)))
        return Decision(Verdict.ALLOW, action.name, cls, "allowed", say, args, session.person)

    @staticmethod
    def _deny(name: str, cls: int | None, reason: str, say: str) -> Decision:
        return Decision(Verdict.DENY, name, cls, reason, say)

    def _check_args(self, action: Action, given: dict[str, Any]):
        specs = {s.name: s for s in action.args}
        if set(given) - set(specs):
            return False, {}, f"unexpected {sorted(set(given) - set(specs))}", []
        out: dict[str, Any] = {}
        devices = []
        for spec in action.args:
            if spec.name not in given:
                return False, {}, f"missing {spec.name}", []
            v = given[spec.name]
            if spec.type == "int":
                if isinstance(v, bool) or not isinstance(v, int) or not (spec.min <= v <= spec.max):
                    return False, {}, f"{spec.name} out of range", []
            elif spec.type == "str":
                if not isinstance(v, str) or not v.strip() or len(v) > spec.max_len:
                    return False, {}, f"{spec.name} not a short string", []
                v = v.strip()
            elif spec.type == "enum":
                if v not in spec.values:
                    return False, {}, f"{spec.name} not in {spec.values}", []
            elif spec.type == "device":
                d = self.policy.devices.get(v) if isinstance(v, str) else None
                if d is None or d.kind != spec.kind:
                    return False, {}, f"{spec.name} not a known {spec.kind}", []
                devices.append(d)
            out[spec.name] = v
        return True, out, "", devices

    # -- undo -----------------------------------------------------------------

    def recent_household_actions(self, room: str) -> list[tuple[str, dict[str, Any]]]:
        """Class 1 actions in this room in the last undo window, newest first.

        The caller reverts them all: one injected "undo" can then not leave
        some of several injected actions in place.
        """
        cutoff = self.clock() - self.policy.undo_seconds
        self._done = [d for d in self._done if d.at >= cutoff]
        return [(d.action, d.args) for d in reversed(self._done) if d.room == room]
