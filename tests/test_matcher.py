"""The matcher handles common household sentences and nothing riskier."""

from pathlib import Path

import pytest

from homevoice.broker import Source
from homevoice.matcher import match
from homevoice.policy import HOUSEHOLD, load_policy

EXAMPLE = Path(__file__).parent.parent / "examples" / "policy.yml"


@pytest.fixture
def policy():
    return load_policy(EXAMPLE)


@pytest.mark.parametrize("text,action,args", [
    ("set a timer for 10 minutes", "timer.start", {"minutes": 10}),
    ("Timer five minutes.", "timer.start", {"minutes": 5}),
    ("two hour timer", "timer.start", {"minutes": 120}),
    ("turn off the kitchen lamp", "light.set", {"device": "kitchen_lamp", "state": "off"}),
    ("turn on the sofa light", "light.set", {"device": "sofa_lamp", "state": "on"}),
    ("turn the kitchen light off", "light.set", {"device": "kitchen_lamp", "state": "off"}),
    ("lights off in the kitchen", "light.set", {"device": "kitchen_lamp", "state": "off"}),
    ("add milk to the shopping list", "list.add", {"item": "milk"}),
    ("please lock the front door", "lock.lock", {"device": "front_door"}),
    ("undo", "undo", {}),
])
def test_matches(policy, text, action, args):
    p = match(text, policy)
    assert p is not None
    assert (p.action, p.args, p.source) == (action, args, Source.MATCHER)


@pytest.mark.parametrize("text", [
    "unlock the front door",
    "unlock the door",
    "open the garage",
    "send David a message",
    "read my calendar",
    "turn on the space heater",  # a class 4 device
    "turn off the toaster",  # not in the policy
    "what's the weather",
    "locks off",
])
def test_no_match_goes_to_model(policy, text):
    assert match(text, policy) is None


def test_matcher_never_proposes_above_household(policy):
    sentences = [
        "set a timer for 10 minutes", "turn off the kitchen lamp", "add milk to the list",
        "lock the front door", "undo", "turn on the space heater", "unlock the front door",
    ]
    for t in sentences:
        p = match(t, policy)
        if p is not None:
            assert policy.actions[p.action].cls <= HOUSEHOLD
