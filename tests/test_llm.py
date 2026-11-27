"""The model client: what it sends, and that the priority key stays secret."""

from __future__ import annotations

import io
import json
import os

import pytest

from homevoice import llm
from homevoice.llm import OllamaClient, read_priority_key


class FakeOpener:
    def __init__(self) -> None:
        self.bodies: list[dict] = []

    def open(self, req, timeout=None):
        self.bodies.append(json.loads(req.data))
        return io.BytesIO(json.dumps({"message": {"content": "ok"}}).encode())


@pytest.fixture
def opener(monkeypatch):
    fake = FakeOpener()
    monkeypatch.setattr(llm, "_OPENER", fake)
    return fake


@pytest.fixture
def key_file(tmp_path):
    p = tmp_path / "priority.key"
    p.write_text("s3cret-key\n")
    os.chmod(p, 0o600)
    return str(p)


def test_no_priority_sends_no_priority(opener):
    OllamaClient("m").chat([], [])
    assert opener.bodies[0]["options"] == {"temperature": 0}


def test_voice_tier_sends_tier_and_key(opener, key_file):
    OllamaClient("m", priority=2, priority_key_file=key_file).chat([], [])
    assert opener.bodies[0]["options"] == {"temperature": 0, "priority": 2, "priority_key": "s3cret-key"}


def test_tier_zero_needs_and_sends_no_key(opener):
    OllamaClient("m", priority=0).chat([], [])
    assert "priority_key" not in opener.bodies[0]["options"]


def test_tier_above_zero_needs_a_key_file():
    with pytest.raises(ValueError):
        OllamaClient("m", priority=2)


def test_key_file_without_priority_is_refused(key_file):
    with pytest.raises(ValueError):
        OllamaClient("m", priority_key_file=key_file)


@pytest.mark.parametrize("bad", [3, -1, True, "2", 2.0])
def test_unknown_tier_is_refused(bad, key_file):
    with pytest.raises(ValueError):
        OllamaClient("m", priority=bad, priority_key_file=key_file)


def test_key_never_in_repr(key_file):
    c = OllamaClient("m", priority=2, priority_key_file=key_file, headers={"Authorization": "Bearer tok"})
    assert "s3cret" not in repr(c)
    assert "tok" not in repr(c)


def test_key_readable_by_others_is_refused(key_file):
    os.chmod(key_file, 0o640)
    with pytest.raises(ValueError, match="readable by others"):
        read_priority_key(key_file)


def test_missing_key_file_is_refused(tmp_path):
    with pytest.raises(OSError):
        OllamaClient("m", priority=2, priority_key_file=str(tmp_path / "nope"))


def test_symlinked_key_is_refused(tmp_path, key_file):
    link = tmp_path / "link.key"
    link.symlink_to(key_file)
    with pytest.raises(OSError):
        read_priority_key(str(link))


@pytest.mark.parametrize("content", ["", "   \n", "two words", "x" * 5000])
def test_bad_key_content_is_refused(tmp_path, content):
    p = tmp_path / "k"
    p.write_text(content)
    os.chmod(p, 0o600)
    with pytest.raises(ValueError):
        read_priority_key(str(p))


def test_key_only_goes_to_loopback_or_https(key_file):
    with pytest.raises(ValueError):
        OllamaClient("m", base_url="http://10.0.0.5:11434", priority=2, priority_key_file=key_file)
