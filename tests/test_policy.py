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
