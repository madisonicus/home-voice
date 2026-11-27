"""A wrong policy file stops the broker. It never runs with a guess."""

import copy
from pathlib import Path

import pytest
import yaml

from homevoice.policy import PolicyError, load_policy, parse_policy

EXAMPLE = Path(__file__).parent.parent / "examples" / "policy.yml"


@pytest.fixture
def raw():
    return yaml.safe_load(EXAMPLE.read_text())


def test_example_loads():
    p = load_policy(EXAMPLE)
    assert p.actions["lock.unlock"].cls == 3


def test_missing_file(tmp_path):
    with pytest.raises(PolicyError):
        load_policy(tmp_path / "nope.yml")


def test_bad_yaml(tmp_path):
    f = tmp_path / "p.yml"
    f.write_text("version: [1\n")
    with pytest.raises(PolicyError):
        load_policy(f)


def mutate(raw, fn):
    r = copy.deepcopy(raw)
    fn(r)
    return r


@pytest.mark.parametrize("fn", [
    lambda r: r.update(version=2),
    lambda r: r.update(typo_key=1),
    lambda r: r.update(rooms={}),
    lambda r: r.update(actions={}),
    lambda r: r["rooms"]["kitchen"].update(max_class=5),
    lambda r: r["rooms"]["kitchen"].update(max_class="3"),
    lambda r: r["actions"]["timer.start"].update(clas=1),
    lambda r: r["actions"]["timer.start"].update({"class": True}),
    lambda r: r["actions"]["timer.start"]["args"]["minutes"].pop("max"),
    lambda r: r["actions"]["list.add"]["args"]["item"].pop("max_len"),
    lambda r: r["actions"]["light.set"]["args"]["state"].update(values=[]),
    lambda r: r["actions"]["light.set"]["args"]["device"].update(kind="toaster"),
    lambda r: r["devices"]["kitchen_lamp"].update(room="attic"),
    lambda r: r["actions"]["calendar.read"].update({"class": 1}),  # personal output below class 2
    lambda r: r["actions"]["list.add"].update(outbound=True),  # class 1 must not send out
    lambda r: r["actions"]["message.send"]["args"]["text"].update(max_len=5000),
])
def test_bad_policy_refused(raw, fn):
    with pytest.raises(PolicyError):
        parse_policy(mutate(raw, fn))


def test_personal_output_implies_untrusted(raw):
    p = parse_policy(raw)
    assert p.actions["calendar.read"].untrusted_output


@pytest.mark.parametrize("fn", [
    lambda r: r["settings"].update(personal_idle_seconds=10**9),
    lambda r: r["settings"].update(personal_idle_seconds=True),
    lambda r: r["settings"].update(personal_idle_seconds=-1),
    lambda r: r["settings"].update(personal_idle_seconds="120"),
    lambda r: r["settings"].update(personal_max_seconds=10**9),
    lambda r: r["actions"]["timer.start"]["args"]["minutes"].update(max=float("inf")),
    lambda r: r["actions"]["timer.start"]["args"]["minutes"].update(min=10, max=5),
    lambda r: r["actions"]["list.add"]["args"]["item"].update(max_len=float("inf")),
    lambda r: r["actions"]["list.add"]["args"]["item"].update(max_len=0),
    lambda r: r["actions"]["list.add"]["args"]["item"].update(max_len=1001),
    lambda r: r["actions"]["lock.unlock"].update(undoable=True),
    lambda r: r["actions"]["list.add"].update(undoable="yes"),
])
def test_bad_values_refused(raw, fn):
    with pytest.raises(PolicyError):
        parse_policy(mutate(raw, fn))


def test_duplicate_key_refused(tmp_path):
    f = tmp_path / "p.yml"
    f.write_text(EXAMPLE.read_text() + "\n  lock.unlock: {class: 1, args: {device: {type: device, kind: lock}}}\n")
    with pytest.raises(PolicyError, match="duplicate"):
        load_policy(f)


@pytest.mark.parametrize("fn", [
    lambda r: r["actions"]["lock.lock"].update(undoable=True),  # lock's revert is unlock
    lambda r: r["rooms"]["guest_room"].update(reach="everywhere"),
    lambda r: r["actions"]["calendar.add"].update(read_only="yes"),
])
def test_second_pass_bad_policy_refused(raw, fn):
    with pytest.raises(PolicyError):
        parse_policy(mutate(raw, fn))


def test_personal_output_write_still_tainted(raw):
    """Only read_only exempts a class 2 action from the taint rule."""
    p = parse_policy(mutate(raw, lambda r: r["actions"]["calendar.add"].update(personal_output=True)))
    assert not p.actions["calendar.add"].read_only


def test_non_string_key_is_a_policy_error(tmp_path):
    f = tmp_path / "p.yml"
    f.write_text(EXAMPLE.read_text().replace("rooms:\n", "rooms:\n  ? [a]\n  : {max_class: 1}\n", 1))
    with pytest.raises(PolicyError):
        load_policy(f)
