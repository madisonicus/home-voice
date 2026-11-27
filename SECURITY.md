# Security

Security is the main design goal of this project, not a feature of it.

## Reporting a problem

Use GitHub's private vulnerability reporting: the **Security** tab, then
**Report a vulnerability**. Do not open a public issue. A way to make the
broker allow something its policy refuses is exactly what we want to hear
about.

## Threat model

Ranked by likelihood times harm, for a home with a few people and occasional
guests.

1. **A mistake by the model, or by whoever changes the code.** The model
   mishears "lights off" as "locks off", or proposes a wrong argument. The
   answer is small action classes, typed arguments, undo, and a phone handoff
   for anything that cannot be undone.
2. **Words from something that is not a person in the house.** TV and radio
   audio, a phone on speaker, a guest. Text the assistant reads: web pages,
   message bodies, calendar invites, shared lists, device names. This is
   prompt injection, and it needs no attacker skill. The answer is that the
   broker decides, and untrusted text taints the session.
3. **Privacy.** Always-on microphones and false wakes. Wake word detection runs
   on the speaker, no audio is stored, and transcripts are kept briefly.
4. **A stolen or opened speaker.** Speakers hold no secret.

Out of scope, because nobody has a reason to aim them at a home: targeted
inaudible (ultrasound) commands, laser injection, voice cloning by a stranger.
The phone handoff for class 3 defeats them anyway.

## What is and is not a wall

- **The broker is the only wall.** It must hold no rights it does not need.
  Class 3 rights (unlock, send) stay on the person's phone.
- **A priority marker is not a wall.** Voice requests get top priority on the
  model server, but a header or option any caller can set proves nothing.
  The gateway or an allowlist on the server grants the tier from the
  caller's identity, and anything else strips the marker.
- **The model's system prompt is not a wall.** Nothing here depends on it.
- **Smart-home hubs' own "exposed entity" lists are not a wall** for the
  broker's token. Keep locks and alarms out of reach of any token the broker
  holds.

## Rules for contributors (people and agents)

- Every change to `broker.py`, `policy.py` or `matcher.py` keeps or adds a
  test in `tests/` that proves the rule it touches.
- Never add a fallback that turns a refusal into an allow: no "spoken yes"
  when the identity provider is down, no permissive default for a missing
  file.
- GitHub Actions are pinned to full commit SHAs, with the version in a
  comment. A tag can be moved to point at other code; a SHA cannot.
  Dependabot proposes SHA updates weekly.
- CI installs Python dependencies from `requirements-ci.txt`, pinned by
  hash. Regenerate it with the command at its top after checking each new
  release on its project page. Dependabot waits 7 days before proposing a
  new release, because compromised releases are usually pulled within days.
- Workflows hold no secrets and run with read-only `contents` permission, so
  a pull request from a stranger has nothing to steal.
- No house details in this repo: no device names, rooms, people, addresses,
  logs or secrets. They belong in the deployment's own policy file.
