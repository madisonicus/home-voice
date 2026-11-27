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


def mode_signature(path: str | Path | None) -> tuple | None:
    """What the switch looks like now: its mode plus the file's identity.

    A flip and a flip back between two requests leaves the same mode but a
    new mtime, so a window opened before the flips can still tell.
    """
    if path is None:
        return None
    try:
        st = Path(path).stat()
    except OSError:
        return (read_mode(path),)
    return (read_mode(path), st.st_ino, st.st_mtime_ns, st.st_size)


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
