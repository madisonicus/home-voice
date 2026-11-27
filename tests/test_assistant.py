"""The model loop: the broker still decides, whatever the model does."""

from pathlib import Path

import pytest

from homevoice.assistant import Assistant, tool_name
from homevoice.broker import Broker, Verdict
from homevoice.demo import DemoExecutor
from homevoice.llm import Reply, ToolCall
from homevoice.policy import load_policy

EXAMPLE = Path(__file__).parent.parent / "examples" / "policy.yml"


class FakeLLM:
    """Plays back scripted replies and records what it was offered."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.offered = []
        self.messages = []

    def chat(self, messages, tools):
        self.offered.append({t["function"]["name"] for t in tools})
        self.messages.append(list(messages))
        return self.replies.pop(0) if self.replies else Reply("ok")


def call(action, **args):
    return Reply("", (ToolCall(tool_name(action), args),))


@pytest.fixture
def setup(tmp_path):
    mode = tmp_path / "mode"
    mode.write_text("full")
    policy = load_policy(EXAMPLE)

    def make(*replies, idp=True):
        broker = Broker(policy, mode, idp_up=lambda: idp)
        llm = FakeLLM(*replies)
        ex = DemoExecutor()
        return Assistant(broker, llm, ex), broker, llm, ex

    return make


def test_matcher_runs_without_model(setup):
    asst, broker, llm, ex = setup()
    s = broker.new_session("kitchen")
    turn = asst.handle(s, "set a timer for 10 minutes")
    assert not turn.used_model and ex.timers == [10] and llm.offered == []


def test_model_tool_call_goes_through_broker(setup):
    asst, broker, llm, ex = setup(call("light.set", device="kitchen_lamp", state="on"), Reply("Done."))
    s = broker.new_session("kitchen")
    turn = asst.handle(s, "it's dark in here")
    assert turn.decisions[0].allowed and ex.lights == {"kitchen_lamp": "on"} and turn.say == "Done."


def test_model_cannot_call_unoffered_tool(setup):
    asst, broker, llm, ex = setup(call("fleet.change"))
    s = broker.new_session("kitchen")
    turn = asst.handle(s, "reboot the server")
    assert turn.decisions[0].verdict is Verdict.DENY
    assert tool_name("fleet.change") not in llm.offered[0]
    assert "Slack" in turn.say


def test_unknown_tool_name_refused(setup):
    asst, broker, llm, ex = setup(Reply("", (ToolCall("shell", {"cmd": "id"}),)))
    s = broker.new_session("kitchen")
    assert asst.handle(s, "x").decisions[0].verdict is Verdict.DENY
    assert ex.log == []


def test_injection_after_web_search_is_refused(setup):
    """The model reads a page that says to unlock things, then tries a class 1 action."""
    asst, broker, llm, ex = setup(
        call("web.search", query="hardware store hours"),
        call("light.set", device="kitchen_lamp", state="off"),
    )
    s = broker.new_session("kitchen")
    turn = asst.handle(s, "when does the hardware store open")
    assert [d.verdict for d in turn.decisions] == [Verdict.ALLOW, Verdict.DENY]
    assert ex.lights == {}
    # After the search the model was no longer offered household tools.
    assert tool_name("light.set") not in llm.offered[1]


def test_tainted_tools_not_offered_on_next_turn(setup):
    asst, broker, llm, ex = setup(call("list.read"), Reply("milk"), Reply("ok"))
    s = broker.new_session("kitchen")
    asst.handle(s, "what's on the list")
    asst.handle(s, "anything else?")
    assert tool_name("timer.start") not in llm.offered[-1]
    assert tool_name("web.search") in llm.offered[-1]


def test_reset_clears_history_and_taint(setup):
    asst, broker, llm, ex = setup(call("list.read"), Reply("milk"), Reply("ok"))
    s = broker.new_session("kitchen")
    asst.handle(s, "what's on the list")
    asst.reset(s)
    asst.handle(s, "hello")
    assert tool_name("timer.start") in llm.offered[-1]
    assert [m["role"] for m in llm.messages[-1]] == ["system", "user"]


def test_handoff_says_broker_line_and_stops(setup):
    asst, broker, llm, ex = setup(call("lock.unlock", device="front_door"), Reply("I unlocked it!"))
    s = broker.new_session("kitchen")
    broker.open_window(s, "ben")
    turn = asst.handle(s, "unlock the door")
    assert turn.needs_phone is not None and turn.needs_phone.verdict is Verdict.HANDOFF
    assert "phone" in turn.say and "unlocked" not in turn.say
    assert len(llm.offered) == 1  # the model did not get to speak after the handoff


def test_guest_room_offered_only_own_devices(setup):
    asst, broker, llm, ex = setup(Reply("hi"))
    s = broker.new_session("guest_room")
    asst.handle(s, "hello")
    offered = llm.offered[0]
    assert tool_name("light.set") not in offered  # no light lives in the guest room
    assert tool_name("lock.unlock") not in offered and tool_name("calendar.read") not in offered


def test_answer_only_mode_offers_class0_only(setup, tmp_path):
    asst, broker, llm, ex = setup(Reply("hi"))
    (tmp_path / "mode").write_text("answer-only")
    s = broker.new_session("kitchen")
    asst.handle(s, "hello")
    assert llm.offered[0] == {tool_name("answer"), tool_name("web.search")}


def test_device_enum_excludes_class4_device(setup):
    asst, broker, llm, ex = setup()
    s = broker.new_session("kitchen")
    light = next(t for t in asst.tools(s) if t["function"]["name"] == tool_name("light.set"))
    assert "space_heater" not in light["function"]["parameters"]["properties"]["device"]["enum"]


def test_step_limit(setup):
    asst, broker, llm, ex = setup(*[call("timer.start", minutes=1)] * 10)
    s = broker.new_session("kitchen")
    turn = asst.handle(s, "x")
    assert turn.model_calls == 4 and "too many" in turn.say


def test_system_prompt_names_room(setup):
    asst, broker, llm, ex = setup(Reply("hi"))
    s = broker.new_session("living_room")
    asst.handle(s, "hello")
    assert "living room" in llm.messages[0][0]["content"]


# -- regressions from the adversarial review of PR #4 -------------------------


def test_personal_data_dropped_when_window_ends(setup):
    asst, broker, llm, ex = setup(call("calendar.read"), Reply("Dentist Tuesday."), Reply("ok"))
    s = broker.new_session("kitchen")
    broker.open_window(s, "ben")
    asst.handle(s, "what's on my calendar")
    broker.mute_pressed("kitchen")
    asst.handle(s, "what was on it again?")
    assert not any("Dentist" in str(m.get("content")) for m in llm.messages[-1])


def test_personal_data_dropped_when_window_expires(setup):
    asst, broker, llm, ex = setup(call("calendar.read"), Reply("Dentist Tuesday."), Reply("ok"))
    t = [1000.0]
    broker.clock = lambda: t[0]
    s = broker.new_session("kitchen")
    broker.open_window(s, "ben")
    asst.handle(s, "what's on my calendar")
    t[0] += 3600
    asst.handle(s, "what was on it again?")
    assert not any("Dentist" in str(m.get("content")) for m in llm.messages[-1])


def test_every_call_gets_a_result_and_say_reports_what_ran(setup):
    asst, broker, llm, ex = setup(Reply("", (
        ToolCall(tool_name("lock.lock"), {"device": "front_door"}),
        ToolCall(tool_name("lock.unlock"), {"device": "front_door"}),
        ToolCall(tool_name("timer.start"), {"minutes": 5}),
    )))
    s = broker.new_session("kitchen")
    turn = asst.handle(s, "x")
    assert "front_door is locked" in turn.say and "phone" in turn.say
    tool_msgs = [m for m in s.history if m["role"] == "tool"]
    assert len(tool_msgs) == 3
    assert "not done" in tool_msgs[1]["content"] and tool_msgs[2]["content"] == "not run"
    assert ex.timers == []


def test_model_cannot_undo(setup):
    asst, broker, llm, ex = setup(call("undo"))
    s = broker.new_session("kitchen")
    assert asst.handle(s, "undo whatever").decisions[0].verdict is Verdict.DENY


def test_session_reset_drops_history(setup):
    asst, broker, llm, ex = setup(call("web.search", query="x"), Reply("r"), call("light.set", device="kitchen_lamp", state="on"), Reply("ok"))
    s = broker.new_session("kitchen")
    asst.handle(s, "search")
    s.reset()  # the broker's own API, not Assistant.reset
    assert s.history == []


def test_calls_per_turn_capped(setup):
    asst, broker, llm, ex = setup(Reply("", tuple(ToolCall(tool_name("timer.start"), {"minutes": 1}) for _ in range(200))))
    s = broker.new_session("kitchen")
    turn = asst.handle(s, "x")
    assert len(ex.timers) == 6 and "more than" in turn.say


@pytest.mark.parametrize("url", ["http://10.0.0.5:11434", "http://atlas:11434", "ftp://127.0.0.1", "http://", "file:///etc/passwd"])
def test_client_refuses_non_loopback_http(url):
    from homevoice.llm import OllamaClient
    with pytest.raises(ValueError):
        OllamaClient(model="m", base_url=url)


@pytest.mark.parametrize("url", ["http://127.0.0.1:11434", "http://localhost:11434", "http://[::1]:11434", "https://gw.example/"])
def test_client_accepts_loopback_or_https(url):
    from homevoice.llm import OllamaClient
    OllamaClient(model="m", base_url=url)


def test_client_ignores_proxy_and_refuses_redirect(monkeypatch):
    import http.server
    import threading
    from homevoice.llm import OllamaClient

    class H(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            self.send_response(302)
            self.send_header("Location", "http://127.0.0.2:9/steal")
            self.end_headers()

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("http_proxy", "http://127.0.0.3:1")
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.3:1")
    client = OllamaClient(model="m", base_url=f"http://127.0.0.1:{srv.server_port}")
    import urllib.error
    with pytest.raises(urllib.error.HTTPError, match="refused redirect"):
        client.chat([], [])
    srv.shutdown()
