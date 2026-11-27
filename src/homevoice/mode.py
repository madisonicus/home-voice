"""The mode file: the software off switch.

One word in a file: `full`, `answer-only` or `off`. The broker reads it on
every request. Anything else, a missing file or a file it cannot read means
`answer-only`: the switch fails closed. The hardware mute on each speaker is
the real kill switch; this file is the one for the whole house.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path


class Mode(str, Enum):
    FULL = "full"
    ANSWER_ONLY = "answer-only"
    OFF = "off"


def read_mode(path: str | Path | None) -> Mode:
    if path is None:
        return Mode.ANSWER_ONLY
    try:
        word = Path(path).read_text(encoding="utf-8").strip().lower()
    except (OSError, UnicodeDecodeError):
        return Mode.ANSWER_ONLY
    try:
        return Mode(word)
    except ValueError:
        return Mode.ANSWER_ONLY
