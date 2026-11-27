# home-voice

A local voice assistant for a home, built security-first. It runs on a home
server with a local LLM (Ollama), and nothing leaves the house.

**The model never holds power.** The hard problem with an LLM voice assistant
is not speech or speed. It is control: anything in the room can talk to it.
A TV, a podcast, a guest's joke, a web page it reads, a calendar invite from a
stranger. Any of these can tell a model to "unlock the front door". So in this
project the model only proposes. A small deterministic program, the **action
broker**, decides, from a policy file, in code.

Status: early. The broker, the intent matcher, the model loop and the policy
tests exist. The speech hub and the phone handoff come next.

## How it decides

Every action has a class, set in the policy:

| Class | Examples | Who | What happens |
| --- | --- | --- | --- |
| 0 Answer | questions, stories, weather | anyone in the room | Speaks the answer. |
| 1 Household | timers, lights, shopping list, lock a door | anyone in the room | Does it. "Undo" reverts the room's last few minutes. |
| 2 Personal | "my calendar", "remind me at 5" | the person, after a passkey approval on their phone | Reads or makes undoable changes to that person's data. |
| 3 Handoff | send a message, unlock, open the garage | the person, on their phone | Prepared here, finished on the phone. The assistant never holds these rights. |
| 4 Never | admin, money, secrets, changing the policy | nobody | Refused, with where to do it instead. |

The rules that make it hold:

- **Rules are code, not prompt.** The broker re-checks every call the model
  makes: class, room, arguments, session state. Security never depends on the
  model refusing, so any model can drive it, abliterated ones included.
- **A voice is not identity.** Anyone in the room gets class 0 and 1. Personal
  data needs a passkey approval on the person's phone, and the approval ends
  after 2 minutes of quiet, 10 minutes in all, a press of the mute button, or
  any flip of the mode switch.
- **Untrusted text cannot act.** Web results, messages, shared list items and
  anything a personal read returns taint the session. A tainted session gets
  no model-proposed household actions and no model-proposed personal writes. This breaks the
  ["lethal trifecta"](https://simonwillison.net/2025/Jun/16/the-lethal-trifecta/)
  of private data, untrusted content and a way to act.
- **Personal data and the web never meet.** A session that read your calendar
  cannot search the web, so the model cannot leak it in a query.
- **Irreversible means handoff.** Unlocking and sending are finished on the
  phone, which shows the exact fields. A spoken "yes" is never approval,
  because the TV can say yes.
- **Common commands skip the model.** A fixed-sentence matcher handles timers,
  lights and lists. It has no sentence for anything above class 1, and the
  broker refuses a "matcher" proposal above class 1 anyway.
- **Rooms can be fenced.** A room with `reach: own` (a guest room) controls
  only its own devices.
- **Text shown on a phone is plain.** Bidi overrides, zero-width and invisible
  characters, line separators and stacked accents are refused, so the phone
  shows what will really be sent.
- **Undo only what is safe to revert.** The revert of "lock" is "unlock", so
  the policy marks each action `undoable` or not, and the default is not.
- **Personal data leaves with the window.** When a personal window ends, the
  conversation that saw the personal data is dropped too.
- **Nothing leaves the box by accident.** The model client talks plain http
  only to loopback, ignores proxy settings and refuses redirects.
- **Fail closed.** A missing or garbled mode file means answer-only. A policy
  file that does not parse, has a duplicate key or an unbounded value, or
  names something that does not exist, stops the broker. No identity provider means no personal requests, never a fallback to
  a spoken yes.

The design follows
[Design Patterns for Securing LLM Agents against Prompt Injections](https://arxiv.org/abs/2506.08837)
(Beurer-Kellner et al., 2025) and CaMeL's principle that a deterministic
engine, not the model, approves actions. [SECURITY.md](SECURITY.md) has the
threat model.

## Layout

| Path | What |
| --- | --- |
| `src/homevoice/broker.py` | The action broker: every decision |
| `src/homevoice/policy.py` | Loads and checks the policy file |
| `src/homevoice/matcher.py` | Fixed sentences that run without the model |
| `src/homevoice/mode.py` | The `full` / `answer-only` / `off` switch |
| `src/homevoice/assistant.py` | One spoken turn: matcher, then the model, the broker deciding each call |
| `src/homevoice/llm.py` | The model client (Ollama today, the fleet gateway later) |
| `src/homevoice/smoke.py`, `smoke/cases.yml` | Tool choice and latency of a real model, with a demo executor |
| `examples/policy.yml` | The policy format, with made-up devices |
| `tests/` | The policy tests: they prove the rules without a model |

The real policy describes a house, so it is not in this repo. Point the broker
at your own file in the format of `examples/policy.yml`.

## Run the tests

```bash
python3 -m pip install -e '.[test]'
python3 -m pytest
```

## Try a model

```bash
python3 -m homevoice.smoke --model <ollama model> --base-url http://127.0.0.1:11434
```

It measures whether the model picks the right tool and how fast. It touches
nothing real. Safety does not depend on its score: the policy tests prove the
rules without a model.

## License

MIT
