"""Policy tests: what the broker allows, refuses and hands off.

These are deterministic. They prove the safety rules without a model: the
model can propose anything, and these show what the broker does with it.
"""

from pathlib import Path

import pytest

from homevoice.broker import Broker, Proposal, Source, Verdict
from homevoice.policy import load_policy

EXAMPLE = Path(__file__).parent.parent / "examples" / "policy.yml"


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def policy():
    return load_policy(EXAMPLE)


@pytest.fixture
def mode(tmp_path):
    p = tmp_path / "mode"
    p.write_text("full\n")
    return p


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def broker(policy, mode, clock):
    return Broker(policy, mode, idp_up=lambda: True, clock=clock)


def model(action, **args):
    return Proposal(action, args, Source.MODEL)


def matcher(action, **args):
    return Proposal(action, args, Source.MATCHER)


# -- class 4 and class 3 never run from the broker ---------------------------


def test_class4_always_refused_with_where_to_go(broker):
    s = broker.new_session("kitchen")
    broker.open_window(s, "ben")
    d = broker.decide(s, model("fleet.change"))
    assert d.verdict is Verdict.DENY
    assert "Slack" in d.say


def test_class4_device_raises_class(broker):
    s = broker.new_session("kitchen")
    d = broker.decide(s, matcher("light.set", device="space_heater", state="on"))
    assert d.verdict is Verdict.DENY and d.cls == 4


@pytest.mark.parametrize("action,args", [
    ("lock.unlock", {"device": "front_door"}),
    ("message.send", {"to": "David", "text": "hi"}),
])
def test_class3_is_never_allowed_only_handed_off(broker, action, args):
    s = broker.new_session("kitchen")
    d = broker.decide(s, model(action, **args))
    assert d.verdict is Verdict.APPROVAL_NEEDED
    broker.open_window(s, "ben")
    d = broker.decide(s, model(action, **args))
    assert d.verdict is Verdict.HANDOFF
    assert d.person == "ben"
    assert d.args == args


def test_class3_handoff_allowed_when_tainted(broker):
    """The phone shows the exact fields, so taint does not block a handoff."""
    s = broker.new_session("kitchen")
    broker.decide(s, model("web.search", query="pizza near me"))
    assert s.tainted
    broker.open_window(s, "ben")
    d = broker.decide(s, model("message.send", to="David", text="Pizza at 7?"))
    assert d.verdict is Verdict.HANDOFF


# -- class 2 ------------------------------------------------------------------


def test_class2_needs_window(broker):
    s = broker.new_session("kitchen")
    assert broker.decide(s, model("calendar.read")).verdict is Verdict.APPROVAL_NEEDED
    broker.open_window(s, "ben")
    d = broker.decide(s, model("calendar.read"))
    assert d.allowed and d.person == "ben"


def test_window_ends_after_idle(broker, clock):
    s = broker.new_session("kitchen")
    broker.open_window(s, "ben")
    clock.t += 121
    assert broker.decide(s, model("calendar.read")).verdict is Verdict.APPROVAL_NEEDED


def test_speech_keeps_window_open(broker, clock):
    s = broker.new_session("kitchen")
    broker.open_window(s, "ben")
    for _ in range(5):
        clock.t += 100
        broker.heard_speech(s)
    assert broker.decide(s, model("calendar.read")).allowed


def test_mute_ends_window(broker):
    s = broker.new_session("kitchen")
    broker.open_window(s, "ben")
    broker.mute_pressed("kitchen")
    assert broker.decide(s, model("calendar.read")).verdict is Verdict.APPROVAL_NEEDED


def test_mode_flip_ends_window(broker, mode):
    s = broker.new_session("kitchen")
    broker.open_window(s, "ben")
    broker.decide(s, model("answer"))
    mode.write_text("answer-only")
    broker.decide(s, model("answer"))
    mode.write_text("full")
    assert broker.decide(s, model("calendar.read")).verdict is Verdict.APPROVAL_NEEDED


# -- personal data and the web never meet ------------------------------------


def test_no_web_search_after_personal_read(broker):
    s = broker.new_session("kitchen")
    broker.open_window(s, "ben")
    assert broker.decide(s, model("calendar.read")).allowed
    d = broker.decide(s, model("web.search", query="ben's dentist appointment"))
    assert d.verdict is Verdict.DENY


def test_no_personal_read_after_web_search(broker):
    s = broker.new_session("kitchen")
    assert broker.decide(s, model("web.search", query="weather")).allowed
    broker.open_window(s, "ben")
    assert broker.decide(s, model("calendar.read")).verdict is Verdict.DENY


def test_reset_clears_taint(broker):
    s = broker.new_session("kitchen")
    broker.decide(s, model("web.search", query="weather"))
    s.reset()
    broker.open_window(s, "ben")
    assert broker.decide(s, model("calendar.read")).allowed


# -- taint and class 1 --------------------------------------------------------


@pytest.mark.parametrize("taint_with", [
    ("web.search", {"query": "x"}),
    ("list.read", {}),
])
def test_tainted_session_refuses_model_class1(broker, taint_with):
    s = broker.new_session("kitchen")
    action, args = taint_with
    assert broker.decide(s, model(action, **args)).allowed
    d = broker.decide(s, model("light.set", device="kitchen_lamp", state="off"))
    assert d.verdict is Verdict.DENY
    assert d.say  # refusals are spoken


def test_personal_read_taints(broker):
    s = broker.new_session("kitchen")
    broker.open_window(s, "ben")
    broker.decide(s, model("calendar.read"))
    assert broker.decide(s, model("timer.start", minutes=5)).verdict is Verdict.DENY


def test_tainted_session_still_allows_matcher_class1(broker):
    s = broker.new_session("kitchen")
    broker.decide(s, model("web.search", query="x"))
    assert broker.decide(s, matcher("light.set", device="kitchen_lamp", state="off")).allowed


def test_clean_session_allows_model_class1(broker):
    s = broker.new_session("kitchen")
    assert broker.decide(s, model("timer.start", minutes=10)).allowed


# -- arguments ----------------------------------------------------------------


@pytest.mark.parametrize("args", [
    {},
    {"minutes": 0},
    {"minutes": 100000},
    {"minutes": "10"},
    {"minutes": True},
    {"minutes": 5, "extra": 1},
])
def test_bad_args_refused(broker, args):
    s = broker.new_session("kitchen")
    assert broker.decide(s, model("timer.start", **args)).verdict is Verdict.DENY


def test_unknown_device_refused(broker):
    s = broker.new_session("kitchen")
    d = broker.decide(s, model("light.set", device="front_door", state="on"))
    assert d.verdict is Verdict.DENY


def test_unknown_action_refused(broker):
    s = broker.new_session("kitchen")
    assert broker.decide(s, model("shell.exec", cmd="rm -rf /")).verdict is Verdict.DENY


def test_long_string_refused(broker):
    s = broker.new_session("kitchen")
    assert broker.decide(s, model("list.add", item="x" * 61)).verdict is Verdict.DENY


# -- room caps ----------------------------------------------------------------


def test_room_cap(broker):
    s = broker.new_session("guest_room")
    broker.open_window(s, "ben")
    assert broker.decide(s, model("calendar.read")).verdict is Verdict.DENY
    assert broker.decide(s, model("lock.unlock", device="front_door")).verdict is Verdict.DENY
    assert broker.decide(s, matcher("timer.start", minutes=5)).allowed


def test_unknown_room_rejected(broker):
    with pytest.raises(ValueError):
        broker.new_session("garage")


# -- mode file fails closed ---------------------------------------------------


def test_missing_mode_file_is_answer_only(policy, tmp_path):
    b = Broker(policy, tmp_path / "nope", idp_up=lambda: True)
    s = b.new_session("kitchen")
    assert b.decide(s, model("answer")).allowed
    assert b.decide(s, matcher("timer.start", minutes=5)).verdict is Verdict.DENY


@pytest.mark.parametrize("content", ["", "FULLL", "full please", "\x00"])
def test_garbled_mode_file_is_answer_only(policy, tmp_path, content):
    p = tmp_path / "mode"
    p.write_text(content)
    b = Broker(policy, p, idp_up=lambda: True)
    s = b.new_session("kitchen")
    assert b.decide(s, matcher("timer.start", minutes=5)).verdict is Verdict.DENY


def test_mode_off_refuses_everything(broker, mode):
    mode.write_text("off")
    s = broker.new_session("kitchen")
    assert broker.decide(s, model("answer")).verdict is Verdict.DENY


def test_mode_read_on_every_request(broker, mode):
    s = broker.new_session("kitchen")
    assert broker.decide(s, matcher("timer.start", minutes=5)).allowed
    mode.write_text("answer-only")
    assert broker.decide(s, matcher("timer.start", minutes=5)).verdict is Verdict.DENY


# -- identity provider down ---------------------------------------------------


def test_idp_down_refuses_class2_and_3(policy, mode):
    b = Broker(policy, mode, idp_up=lambda: False)
    s = b.new_session("kitchen")
    b.open_window(s, "ben")
    assert b.decide(s, model("calendar.read")).verdict is Verdict.DENY
    assert b.decide(s, model("lock.unlock", device="front_door")).verdict is Verdict.DENY


def test_default_idp_is_down(policy, mode):
    """A broker built without an identity check never approves anything personal."""
    b = Broker(policy, mode)
    s = b.new_session("kitchen")
    b.open_window(s, "ben")
    assert b.decide(s, model("calendar.read")).verdict is Verdict.DENY


# -- undo ---------------------------------------------------------------------


def test_undo_lists_all_recent_room_actions(broker, clock):
    k = broker.new_session("kitchen")
    lr = broker.new_session("living_room")
    broker.decide(k, matcher("list.add", item="milk"))
    clock.t += 400
    broker.decide(k, matcher("list.add", item="eggs"))
    broker.decide(k, matcher("list.add", item="bread"))
    broker.decide(lr, matcher("timer.start", minutes=3))
    recent = broker.recent_household_actions("kitchen")
    assert [a[1]["item"] for a in recent] == ["bread", "eggs"]


def test_refused_actions_are_not_recorded(broker):
    s = broker.new_session("kitchen")
    broker.decide(s, model("web.search", query="x"))
    broker.decide(s, model("list.add", item="spam"))
    assert broker.recent_household_actions("kitchen") == []


# -- regressions from the adversarial review of PR #1 -------------------------


def test_undo_never_lists_lock(broker):
    """The revert of "lock" is "unlock", so lock must never be undoable."""
    s = broker.new_session("kitchen")
    assert broker.decide(s, matcher("lock.lock", device="front_door")).allowed
    assert broker.decide(s, matcher("undo")).allowed
    assert broker.recent_household_actions("kitchen") == []


def test_undo_counts_for_the_device_room(broker):
    lr = broker.new_session("living_room")
    broker.decide(lr, matcher("light.set", device="kitchen_lamp", state="on"))
    assert broker.recent_household_actions("kitchen") == [
        ("light.set", {"device": "kitchen_lamp", "state": "on"})
    ]


def test_speech_does_not_revive_expired_window(broker, clock):
    s = broker.new_session("kitchen")
    broker.open_window(s, "ben")
    clock.t += 3600
    broker.heard_speech(s)
    assert broker.decide(s, model("calendar.read")).verdict is Verdict.APPROVAL_NEEDED


def test_window_has_a_hard_cap_even_with_speech(broker, clock):
    """A TV talking all evening must not hold a window open."""
    s = broker.new_session("kitchen")
    broker.open_window(s, "ben")
    for _ in range(7):
        clock.t += 100
        broker.heard_speech(s)
    assert broker.decide(s, model("calendar.read")).verdict is Verdict.APPROVAL_NEEDED


def test_flip_and_back_between_requests_ends_window(broker, mode):
    s = broker.new_session("kitchen")
    broker.decide(s, model("answer"))
    broker.open_window(s, "ben")
    mode.write_text("off")
    mode.write_text("full")
    assert broker.decide(s, model("calendar.read")).verdict is Verdict.APPROVAL_NEEDED


def test_window_cannot_open_unless_full(broker, mode):
    mode.write_text("off")
    s = broker.new_session("kitchen")
    assert broker.open_window(s, "ben") is False
    mode.write_text("full")
    assert broker.decide(s, model("calendar.read")).verdict is Verdict.APPROVAL_NEEDED


def test_tainted_session_refuses_model_personal_write(broker):
    s = broker.new_session("kitchen")
    broker.decide(s, model("list.read"))
    broker.open_window(s, "ben")
    d = broker.decide(s, model("calendar.add", title="Send keys to stranger", when="now"))
    assert d.verdict is Verdict.DENY


def test_personal_read_then_model_write_refused(broker):
    """An invite's text must not be able to write to the calendar."""
    s = broker.new_session("kitchen")
    broker.open_window(s, "ben")
    assert broker.decide(s, model("calendar.read")).allowed
    assert broker.decide(s, model("calendar.add", title="x", when="now")).verdict is Verdict.DENY


def test_clean_session_allows_personal_write(broker):
    s = broker.new_session("kitchen")
    broker.open_window(s, "ben")
    assert broker.decide(s, model("calendar.add", title="Lunch", when="tomorrow noon")).allowed


@pytest.mark.parametrize("bad", [
    "David‮",  # right-to-left override
    "​",  # zero-width space only
    "Da​vid",
    "line\nbreak",
    "\x07bell",
])
def test_control_and_format_chars_refused(broker, bad):
    s = broker.new_session("kitchen")
    broker.open_window(s, "ben")
    assert broker.decide(s, model("message.send", to=bad, text="hi")).verdict is Verdict.DENY


def test_text_is_normalized(broker):
    s = broker.new_session("kitchen")
    d = broker.decide(s, model("list.add", item="café"))
    assert d.allowed and d.args["item"] == "café"


# -- regressions from the second adversarial pass ------------------------------


def test_own_reach_room_cannot_control_other_rooms(broker):
    g = broker.new_session("guest_room")
    assert broker.decide(g, matcher("light.set", device="kitchen_lamp", state="off")).verdict is Verdict.DENY
    assert broker.decide(g, matcher("lock.lock", device="front_door")).verdict is Verdict.DENY  # house-wide device
    assert broker.decide(g, matcher("timer.start", minutes=5)).allowed


def test_house_reach_room_can_control_other_rooms(broker):
    lr = broker.new_session("living_room")
    assert broker.decide(lr, matcher("light.set", device="kitchen_lamp", state="off")).allowed


def test_matcher_label_above_class1_refused(broker):
    s = broker.new_session("kitchen")
    broker.open_window(s, "ben")
    assert broker.decide(s, matcher("calendar.add", title="x", when="now")).verdict is Verdict.DENY
    assert broker.decide(s, matcher("lock.unlock", device="front_door")).verdict is Verdict.DENY


@pytest.mark.parametrize("bad", [
    "Daㅤvid", "ᅟ", "⠀⠀", "a b", "a b",
    "e" + "́" * 5,
])
def test_invisible_and_separator_text_refused(broker, bad):
    s = broker.new_session("kitchen")
    assert broker.decide(s, model("list.add", item=bad)).verdict is Verdict.DENY
