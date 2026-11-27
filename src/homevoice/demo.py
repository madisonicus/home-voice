"""An in-memory executor for tests and the smoke set. It touches nothing real."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DemoExecutor:
    timers: list[int] = field(default_factory=list)
    shopping: list[str] = field(default_factory=list)
    lights: dict[str, str] = field(default_factory=dict)
    locked: set[str] = field(default_factory=set)
    calendar: list[str] = field(default_factory=lambda: ["Dentist, Tuesday 3 PM"])
    log: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def execute(self, action: str, args: dict[str, Any]) -> str:
        self.log.append((action, dict(args)))
        if action == "timer.start":
            self.timers.append(args["minutes"])
            return f"Timer set for {args['minutes']} minutes."
        if action == "list.add":
            self.shopping.append(args["item"])
            return f"Added {args['item']}."
        if action == "list.read":
            return "Shopping list: " + (", ".join(self.shopping) or "empty")
        if action == "light.set":
            self.lights[args["device"]] = args["state"]
            return f"{args['device']} is {args['state']}."
        if action == "lock.lock":
            self.locked.add(args["device"])
            return f"{args['device']} is locked."
        if action == "calendar.read":
            return "Calendar: " + "; ".join(self.calendar)
        if action == "calendar.add":
            self.calendar.append(f"{args['title']}, {args['when']}")
            return "Added to the calendar."
        if action == "web.search":
            return f"(demo) No live search. Query was: {args['query']}"
        if action == "answer":
            return ""
        return f"(demo) {action} done."

    def undo(self, actions: list[tuple[str, dict[str, Any]]]) -> str:
        for action, args in actions:
            if action == "timer.start" and args["minutes"] in self.timers:
                self.timers.remove(args["minutes"])
            elif action == "list.add" and args["item"] in self.shopping:
                self.shopping.remove(args["item"])
            elif action == "light.set":
                self.lights[args["device"]] = "off" if args["state"] == "on" else "on"
        return f"Undid {len(actions)} action{'s' if len(actions) != 1 else ''}."
