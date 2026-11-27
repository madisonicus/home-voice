"""Run the smoke set against a real model: tool choice and latency.

    python -m homevoice.smoke --model qwen3.8:27b-nvfp4 [--base-url http://127.0.0.1:11434]

Each case runs in a fresh kitchen session with an open personal window, the
example policy, a demo executor (nothing real is touched) and the matcher
off, so every sentence goes to the model.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import tempfile
from pathlib import Path

import yaml

from .assistant import Assistant
from .broker import Broker
from .demo import DemoExecutor
from .llm import OllamaClient
from .policy import load_policy

ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="model name as the server knows it")
    ap.add_argument("--base-url", default="http://127.0.0.1:11434", help="Ollama or gateway base URL")
    ap.add_argument("--policy", default=str(ROOT / "examples" / "policy.yml"))
    ap.add_argument("--cases", default=str(ROOT / "smoke" / "cases.yml"))
    ap.add_argument("--think", action="store_true", help="turn model thinking on (off for voice)")
    ap.add_argument("--priority", type=int, choices=(0, 1, 2), help="server scheduling tier (2 is voice); above 0 needs --priority-key-file")
    ap.add_argument("--priority-key-file", help="owner-only file holding the server's priority key")
    args = ap.parse_args(argv)

    policy = load_policy(args.policy)
    cases = yaml.safe_load(Path(args.cases).read_text())["cases"]
    client = OllamaClient(model=args.model, base_url=args.base_url, think=args.think, timeout=120,
                          priority=args.priority, priority_key_file=args.priority_key_file)

    with tempfile.TemporaryDirectory() as tmp:
        mode = Path(tmp) / "mode"
        mode.write_text("full")
        passed, firsts = 0, []
        for c in cases:
            broker = Broker(policy, mode, idp_up=lambda: True)
            asst = Assistant(broker, client, DemoExecutor(), use_matcher=False)
            s = broker.new_session("kitchen")
            broker.open_window(s, "tester")
            turn = asst.handle(s, c["text"])
            got = turn.decisions[0].action if turn.decisions else "none"
            ok = got == c["expect"]
            if ok and c.get("args") and turn.decisions:
                ok = all(turn.decisions[0].args.get(k) == v for k, v in c["args"].items())
            passed += ok
            first = turn.model_seconds / max(turn.model_calls, 1)
            firsts.append(first)
            verdict = turn.decisions[0].verdict.value if turn.decisions else "-"
            print(f"{'PASS' if ok else 'FAIL'}  {first:5.2f}s  want={c['expect']:<14} got={got:<14} {verdict:<16} {c['text']}")
    n = len(cases)
    firsts.sort()
    print(f"\n{args.model}: {passed}/{n} right; seconds per model call p50 {statistics.median(firsts):.2f}, "
          f"p90 {firsts[int(0.9 * (n - 1))]:.2f}")
    return 0 if passed == n else 1


if __name__ == "__main__":
    sys.exit(main())
