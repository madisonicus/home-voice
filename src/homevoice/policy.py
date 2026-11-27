"""Load and check the voice policy file.

The policy says what each action is, which class it has, which arguments it
takes, which devices exist and which rooms cap the class. It is the only
source of what voice may do. A policy that does not parse, or that refers to
something that does not exist, is refused: the broker does not start.

The real policy file is not in this repo. It describes the house, so it lives
in the private fleet repo (identity/voice-actions.yml). examples/policy.yml
shows the format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Classes, from the design doc.
ANSWER = 0  # speak an answer, no side effects
HOUSEHOLD = 1  # anyone in the room, undoable
PERSONAL = 2  # the person, after a phone approval
HANDOFF = 3  # prepared here, finished on the person's phone
NEVER = 4  # never by voice

CLASSES = (ANSWER, HOUSEHOLD, PERSONAL, HANDOFF, NEVER)

ARG_TYPES = ("int", "str", "enum", "device")


class PolicyError(ValueError):
    """The policy file is wrong. The broker must not start."""


@dataclass(frozen=True)
class ArgSpec:
    name: str
    type: str
    min: int | None = None
    max: int | None = None
    max_len: int | None = None
    values: tuple[str, ...] = ()
    kind: str | None = None  # device kind, for type: device


@dataclass(frozen=True)
class Action:
    name: str
    cls: int
    args: tuple[ArgSpec, ...] = ()
    # True: the action pulls text that someone other than the speaker wrote
    # (web pages, messages, invites, shared lists) into the model's context.
    untrusted_output: bool = False
    # True: the action's output is personal data (class 2 reads).
    personal_output: bool = False
    # True: the action sends data out of the house (web search queries).
    outbound: bool = False
    # Spoken when the action is refused as class 4: where to do it instead.
    instead: str = ""


@dataclass(frozen=True)
class Device:
    name: str
    kind: str
    room: str | None = None
    cls: int | None = None  # raises the class of any action on it
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class Room:
    name: str
    max_class: int


@dataclass(frozen=True)
class Policy:
    actions: dict[str, Action]
    devices: dict[str, Device]
    rooms: dict[str, Room]
    personal_idle_seconds: int = 120
    undo_seconds: int = 300
    source: str = field(default="", compare=False)

    def device_by_spoken_name(self, spoken: str, kind: str | None = None) -> Device | None:
        """Find a device by its name or an alias. Exact match only, after case folding."""
        want = _norm(spoken)
        hits = [
            d
            for d in self.devices.values()
            if (kind is None or d.kind == kind)
            and (want == _norm(d.name) or want in {_norm(a) for a in d.aliases})
        ]
        return hits[0] if len(hits) == 1 else None


def _norm(s: str) -> str:
    return " ".join(s.lower().replace("_", " ").split())


def _req(d: dict[str, Any], key: str, where: str) -> Any:
    if key not in d:
        raise PolicyError(f"{where}: missing '{key}'")
    return d[key]


def _cls(v: Any, where: str) -> int:
    if not isinstance(v, int) or isinstance(v, bool) or v not in CLASSES:
        raise PolicyError(f"{where}: class must be one of {CLASSES}, got {v!r}")
    return v


def _check_keys(d: dict[str, Any], allowed: set[str], where: str) -> None:
    extra = set(d) - allowed
    if extra:
        # An unknown key is a typo that would silently do nothing. Refuse it.
        raise PolicyError(f"{where}: unknown keys {sorted(extra)}")


def _arg(name: str, spec: Any, where: str) -> ArgSpec:
    if not isinstance(spec, dict):
        raise PolicyError(f"{where}: must be a mapping")
    _check_keys(spec, {"type", "min", "max", "max_len", "values", "kind"}, where)
    t = _req(spec, "type", where)
    if t not in ARG_TYPES:
        raise PolicyError(f"{where}: type must be one of {ARG_TYPES}, got {t!r}")
    if t == "str" and "max_len" not in spec:
        raise PolicyError(f"{where}: a str argument needs max_len")
    if t == "int" and ("min" not in spec or "max" not in spec):
        raise PolicyError(f"{where}: an int argument needs min and max")
    if t == "enum" and not spec.get("values"):
        raise PolicyError(f"{where}: an enum argument needs values")
    if t == "device" and not spec.get("kind"):
        raise PolicyError(f"{where}: a device argument needs kind")
    return ArgSpec(
        name=name,
        type=t,
        min=spec.get("min"),
        max=spec.get("max"),
        max_len=spec.get("max_len"),
        values=tuple(str(v) for v in spec.get("values", ())),
        kind=spec.get("kind"),
    )


def parse_policy(raw: Any, source: str = "") -> Policy:
    if not isinstance(raw, dict):
        raise PolicyError("policy: top level must be a mapping")
    _check_keys(raw, {"version", "settings", "rooms", "devices", "actions"}, "policy")
    if raw.get("version") != 1:
        raise PolicyError("policy: version must be 1")

    settings = raw.get("settings") or {}
    _check_keys(settings, {"personal_idle_seconds", "undo_seconds"}, "settings")

    rooms: dict[str, Room] = {}
    for name, r in (_req(raw, "rooms", "policy") or {}).items():
        where = f"rooms.{name}"
        r = r or {}
        _check_keys(r, {"max_class"}, where)
        rooms[name] = Room(name=name, max_class=_cls(_req(r, "max_class", where), where))
    if not rooms:
        raise PolicyError("policy: at least one room is needed")

    devices: dict[str, Device] = {}
    for name, d in (raw.get("devices") or {}).items():
        where = f"devices.{name}"
        if not isinstance(d, dict):
            raise PolicyError(f"{where}: must be a mapping")
        _check_keys(d, {"kind", "room", "class", "aliases"}, where)
        room = d.get("room")
        if room is not None and room not in rooms:
            raise PolicyError(f"{where}: room {room!r} is not in rooms")
        devices[name] = Device(
            name=name,
            kind=str(_req(d, "kind", where)),
            room=room,
            cls=_cls(d["class"], where) if "class" in d else None,
            aliases=tuple(str(a) for a in d.get("aliases", ())),
        )

    actions: dict[str, Action] = {}
    for name, a in (_req(raw, "actions", "policy") or {}).items():
        where = f"actions.{name}"
        if not isinstance(a, dict):
            raise PolicyError(f"{where}: must be a mapping")
        _check_keys(
            a,
            {"class", "args", "untrusted_output", "personal_output", "outbound", "instead"},
            where,
        )
        cls = _cls(_req(a, "class", where), where)
        args = tuple(_arg(k, v, f"{where}.args.{k}") for k, v in (a.get("args") or {}).items())
        for spec in args:
            if spec.type == "device" and not any(d.kind == spec.kind for d in devices.values()):
                raise PolicyError(f"{where}.args.{spec.name}: no device of kind {spec.kind!r}")
        if cls == HANDOFF:
            for spec in args:
                if spec.type == "str" and (spec.max_len or 0) > 1000:
                    raise PolicyError(f"{where}.args.{spec.name}: too long to show on a phone")
        personal = bool(a.get("personal_output", False))
        if personal and cls < PERSONAL:
            raise PolicyError(f"{where}: personal_output needs class 2 or higher")
        outbound = bool(a.get("outbound", False))
        if outbound and cls != ANSWER and cls < HANDOFF:
            # Class 1 and 2 actions must not send data out of the house.
            raise PolicyError(f"{where}: outbound is allowed only for class 0 (web search) or 3+")
        actions[name] = Action(
            name=name,
            cls=cls,
            args=args,
            untrusted_output=bool(a.get("untrusted_output", False)) or personal,
            personal_output=personal,
            outbound=outbound,
            instead=str(a.get("instead", "")),
        )
    if not actions:
        raise PolicyError("policy: at least one action is needed")

    return Policy(
        actions=actions,
        devices=devices,
        rooms=rooms,
        personal_idle_seconds=int(settings.get("personal_idle_seconds", 120)),
        undo_seconds=int(settings.get("undo_seconds", 300)),
        source=source,
    )


def load_policy(path: str | Path) -> Policy:
    p = Path(path)
    try:
        raw = yaml.safe_load(p.read_text())
    except (OSError, yaml.YAMLError) as e:
        raise PolicyError(f"{p}: cannot read: {e}") from e
    return parse_policy(raw, source=str(p))
