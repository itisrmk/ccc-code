#!/usr/bin/env python3
"""Live end-to-end tests: real Claude Code, through ccc, against the real API.

Spends a small amount of Claude quota (about $0.02 on Haiku by default). Every scenario
runs with strict thinking enforcement, so any edit to sent history is a hard failure.

    scripts/live_test.py                  run everything
    scripts/live_test.py headless redact  run the named scenarios
    scripts/live_test.py --list           show scenario names

Exit status is non-zero if any scenario fails. Secrets used here are fake.
"""

import json
import os
import random
import re
import shlex
import shutil
import string
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CCC = ROOT / "target" / "release" / "ccc"
MODEL = os.environ.get("CCC_LIVE_MODEL", "haiku")
PLACEHOLDER_LENGTH = 19


class Failed(Exception):
    pass


def check(condition, message):
    if not condition:
        raise Failed(message)


class Sandbox:
    """A throwaway working directory, state directory and traffic dump."""

    def __init__(self, base):
        self.base = Path(tempfile.mkdtemp(dir=base))
        self.work = self.base / "work"
        self.state = self.base / "state"
        self.dump = self.base / "dump"
        self.work.mkdir()
        subprocess.run(["git", "init", "-q", "."], cwd=self.work, check=True)

    def write_test_script(self, line_count):
        lines = [f"test module::case_{i} ... ok" for i in range(line_count)]
        lines[line_count // 2] = "test module::flaky_parser ... FAILED"
        lines.append(f"test result: FAILED. {line_count - 1} passed; 1 failed")
        script = 'print("""' + "\n".join(lines) + '""")\n'
        (self.work / "run_tests.py").write_text(script)
        (self.work / "Makefile").write_text("test:\n\t@python3 run_tests.py\n")

    def write_secret(self):
        rng = random.Random(7)
        body = "".join(rng.choices(string.ascii_lowercase + string.digits, k=24))
        key = "sk-ant-" + "fake" + body
        (self.work / "app.env").write_text(f"SERVICE_API_KEY={key}\n")
        return key

    def exchanges(self):
        """Every Messages API exchange captured, oldest first."""
        found = []
        for body_path in sorted(self.dump.glob("*.req-body.json")):
            stem = str(body_path)[: -len(".req-body.json")]
            raw = body_path.read_bytes()
            try:
                original = json.loads(raw)
            except ValueError:
                continue
            if not isinstance(original, dict) or "messages" not in original:
                continue
            compiled_path = Path(stem + ".req-compiled.json")
            sent_bytes = compiled_path.read_bytes() if compiled_path.exists() else raw
            head = json.loads(Path(stem + ".res-head.json").read_text())
            trace_path = Path(stem + ".trace.json")
            trace = json.loads(trace_path.read_text()) if trace_path.exists() else []
            rules = [edit["rule"] for stage in trace for edit in stage["edits"]]
            found.append(
                {
                    "original_bytes": raw,
                    "sent_bytes": sent_bytes,
                    "sent": json.loads(sent_bytes),
                    "status": head["status"],
                    "rules": rules,
                    "response": Path(stem + ".res-body").read_bytes(),
                }
            )
        return found


def ccc_args(box, *extra):
    return [
        str(CCC),
        "run",
        "--stats",
        "--enforce-thinking-check",
        "--state-dir",
        str(box.state),
        "--dump",
        str(box.dump),
        *extra,
        "--",
    ]


def run_headless(box, prompt, *claude_args, ccc_extra=(), model=MODEL):
    command = ccc_args(box, *ccc_extra) + [
        "claude",
        "-p",
        prompt,
        "--model",
        model,
        "--allowedTools",
        "Bash",
        "--output-format",
        "json",
        *claude_args,
    ]
    done = subprocess.run(
        command,
        cwd=box.work,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=300,
    )
    check(done.returncode == 0, f"exit {done.returncode}: {done.stderr[-400:]}")
    return json.loads(done.stdout)


def thinking_blocks(exchange):
    return sum(
        1
        for message in exchange["sent"]["messages"]
        if isinstance(message["content"], list)
        for block in message["content"]
        if block.get("type") == "thinking"
    )


def is_quota_probe(exchange):
    """Claude Code's startup probe, which the API can answer with a 429 on its own."""
    return exchange["sent"].get("max_tokens") == 1


def assert_all_accepted(exchanges, require_thinking=True):
    check(exchanges, "no Messages API traffic was captured")
    for index, exchange in enumerate(exchanges):
        check(
            exchange["status"] == 200 or is_quota_probe(exchange),
            f"request {index} got {exchange['status']}: {exchange['response'][:300]!r}",
        )
    check(
        not require_thinking or any(thinking_blocks(exchange) for exchange in exchanges),
        "no thinking block was replayed, so enforcement proved nothing",
    )


def scenario_headless(box):
    """Prune rules fire on a real task, and the model can still answer."""
    box.write_test_script(600)
    result = run_headless(
        box,
        "One tool call at a time, with Bash: 1) run 'python3 run_tests.py'. "
        "2) run 'make test'. 3) run 'make test' again. "
        "4) Reply with only the failing test's name.",
    )
    check(not result["is_error"], f"task errored: {result.get('result')}")
    check("flaky_parser" in result["result"], f"lost the failure: {result['result']!r}")
    exchanges = box.exchanges()
    assert_all_accepted(exchanges)
    fired = {rule for exchange in exchanges for rule in exchange["rules"]}
    check({"generic", "log"} <= fired, f"expected prune rules to fire, saw {fired}")
    check("duplicate" in fired, f"expected a duplicate stub, saw {fired}")
    last = exchanges[-1]
    check(
        len(last["sent_bytes"]) < len(last["original_bytes"]),
        "pruning did not shrink the request",
    )
    return f"{len(exchanges)} requests accepted, rules {sorted(fired)}"


def scenario_redact(box):
    """The key never leaves, the real value comes back, and a resume stays valid."""
    key = box.write_secret()
    prompt = (
        "One tool call at a time with Bash. 1) Run 'cat app.env'. 2) Run: "
        "printf %s 'VALUE' | wc -c  where VALUE is exactly the SERVICE_API_KEY value you "
        "saw, copied literally whatever it looks like. 3) Reply with only the number."
    )
    first = run_headless(box, prompt)
    check(
        str(len(key)) in first["result"],
        f"expected the real key's length {len(key)}, not the placeholder's "
        f"{PLACEHOLDER_LENGTH}: {first['result']!r}",
    )
    second = run_headless(
        box,
        "Run 'echo resumed' with Bash and reply with its output.",
        "--resume",
        first["session_id"],
    )
    check("resumed" in second["result"], f"resume failed: {second['result']!r}")

    exchanges = box.exchanges()
    assert_all_accepted(exchanges)
    for index, exchange in enumerate(exchanges):
        check(
            key.encode() not in exchange["sent_bytes"],
            f"request {index} sent the key upstream",
        )
    check(
        any(key.encode() in exchange["original_bytes"] for exchange in exchanges),
        "the key never appeared in a request, so this test proved nothing",
    )
    return f"{len(exchanges)} requests accepted, key stayed local, wc saw {len(key)}"


def scenario_enforcement(box):
    """Control: an edit to sent history must be rejected, or the other passes mean nothing."""
    key = box.write_secret()
    first = run_headless(
        box, "Run 'cat app.env' with Bash, then reply with only the word done."
    )
    shutil.rmtree(box.state / "conversations")
    box_dump_before = len(box.exchanges())
    run_headless(box, "Say ok.", "--resume", first["session_id"])
    later = box.exchanges()[box_dump_before:]
    check(later, "the resume made no requests")
    check(
        later[0]["status"] == 400,
        f"sending edited history should get a 400, got {later[0]['status']}",
    )
    check(
        b"different conversation" in later[0]["response"],
        f"unexpected rejection: {later[0]['response'][:300]!r}",
    )
    check(key.encode() in later[0]["sent_bytes"], "the control did not edit history")
    return "edited history rejected with the expected 400"


def scenario_budget(box):
    """The ledger matches Claude Code's own cost, and a notice survives a resume unseen."""
    first = run_headless(
        box,
        "Think carefully and work out 37*43 by hand. Run 'echo RESULT' with Bash using your "
        "answer, then reply with only the word done.",
        ccc_extra=("--budget-usd", "0.000001"),
    )
    listing = subprocess.run(
        [str(CCC), "stats", "--state-dir", str(box.state)],
        capture_output=True, text=True, check=True,
    ).stdout
    recorded = re.search(re.escape(first["session_id"]) + r".*\$([0-9.]+)", listing)
    check(recorded, f"session missing from ccc stats:\n{listing}")
    reported = first["total_cost_usd"]
    check(
        abs(float(recorded.group(1)) - reported) <= max(0.25 * reported, 0.0005),
        f"ledger says {recorded.group(1)}, Claude Code says {reported}",
    )
    check("done" in first["result"].lower(), f"the answer was lost: {first['result']!r}")

    second = run_headless(
        box,
        "Think carefully and work out 41*47 by hand. Run 'echo RESULT' with Bash using your "
        "answer, then 'echo again' with Bash, then reply with only the word finished.",
        "--resume",
        first["session_id"],
        ccc_extra=("--budget-usd", "0.000001"),
    )
    check("finished" in second["result"].lower(), f"the answer was lost: {second['result']!r}")
    exchanges = box.exchanges()
    assert_all_accepted(exchanges)
    check(
        any(b"[ccc budget]" in exchange["original_bytes"] for exchange in exchanges),
        "Claude Code never stored a notice, so it wasn't shown",
    )
    for index, exchange in enumerate(exchanges):
        check(b"[ccc budget]" not in exchange["sent_bytes"], f"request {index} sent the notice")
    return (
        f"ledger ${recorded.group(1)} vs reported ${reported:.4f}, notice shown then stripped, "
        f"{len(exchanges)} requests accepted"
    )


def scenario_downshift(box):
    """An explicit budget downshift really changes the model, with no history error."""
    first = run_headless(
        box,
        "Run 'echo one' with Bash, then run 'echo two' with Bash, then reply with only done.",
        ccc_extra=("--budget-usd", "0.000001", "--budget-downshift", "claude-haiku-5-5"),
        model="claude-sonnet-5-5",
    )
    check(not first["is_error"], f"task errored: {first.get('result')}")
    exchanges = box.exchanges()
    assert_all_accepted(exchanges, require_thinking=False)
    models = [exchange["sent"].get("model") for exchange in exchanges]
    check(models[0] == "claude-sonnet-5-5", f"started on the wrong model: {models}")
    check(models[-1] == "claude-haiku-5-5", f"never moved down: {models}")
    return f"models over the session: {models}"


class Terminal:
    """A real terminal session in a private tmux server, read as a rendered screen."""

    def __init__(self, command, cwd):
        self.socket = f"ccc-live-{os.getpid()}-{time.time_ns()}"
        self._tmux(
            "new-session", "-d", "-x", "140", "-y", "50", "-c", str(cwd),
            " ".join(shlex.quote(part) for part in command) + "; sleep 600",
        )

    def _tmux(self, *arguments):
        done = subprocess.run(
            ["tmux", "-L", self.socket, *arguments], capture_output=True, text=True
        )
        return done.stdout

    def screen(self):
        return self._tmux("capture-pane", "-p", "-S", "-200")

    def type(self, text):
        self._tmux("send-keys", "-l", text)

    def press(self, key):
        self._tmux("send-keys", key)

    def wait_for(self, pattern, seconds):
        deadline = time.time() + seconds
        while time.time() < deadline:
            if re.search(pattern, self.screen()):
                return True
            time.sleep(0.5)
        return False

    def close(self):
        self._tmux("kill-server")


def scenario_interactive(box):
    """The real full-screen TUI in a terminal, through ccc."""
    box.write_test_script(600)
    command = ccc_args(box) + ["claude", "--model", MODEL, "--allowedTools", "Bash"]
    terminal = Terminal(command, box.work)
    try:
        # A new directory asks whether to trust it, and the default answer is no.
        if terminal.wait_for(r"Yes, I trust this folder", 12):
            terminal.press("Down")
            time.sleep(0.5)
            terminal.press("Enter")
            time.sleep(3)
        time.sleep(3)
        terminal.type("Run 'make test' with Bash, then reply with the failing test's name.")
        time.sleep(1)
        terminal.press("Enter")
        answered = terminal.wait_for(r"(?m)^.*⏺.*flaky_parser", 150)
        check(answered, "the TUI never showed an answer:\n" + terminal.screen()[-1500:])
        time.sleep(2)
        terminal.type("/exit")
        time.sleep(1)
        terminal.press("Enter")
        check(
            terminal.wait_for(r"ccc: \d+ requests compiled", 20),
            "ccc did not exit cleanly with the TUI:\n" + terminal.screen()[-800:],
        )
    finally:
        terminal.close()

    if os.environ.get("CCC_LIVE_KEEP_DUMP"):
        shutil.copytree(box.dump, os.environ["CCC_LIVE_KEEP_DUMP"], dirs_exist_ok=True)
    exchanges = box.exchanges()
    check(exchanges, "no traffic went through ccc from the TUI")
    for index, exchange in enumerate(exchanges):
        check(
            exchange["status"] == 200 or is_quota_probe(exchange),
            f"request {index} got {exchange['status']}",
        )
    fired = {rule for exchange in exchanges for rule in exchange["rules"]}
    return f"TUI answered through ccc, {len(exchanges)} requests accepted, rules {sorted(fired)}"


def main():
    scenarios = {
        name[len("scenario_") :]: value
        for name, value in globals().items()
        if name.startswith("scenario_")
    }
    arguments = sys.argv[1:]
    if arguments == ["--list"]:
        print("\n".join(scenarios))
        return 0
    check_binary = CCC.exists()
    if not check_binary:
        print(f"{CCC} is missing. Run: cargo build --release -p ccc", file=sys.stderr)
        return 2
    chosen = arguments or list(scenarios)
    base = tempfile.mkdtemp(prefix="ccc-live-")
    failures = 0
    for name in chosen:
        started = time.time()
        try:
            detail = scenarios[name](Sandbox(base))
            print(f"PASS {name}: {detail} ({time.time() - started:.0f}s)")
        except (Failed, subprocess.TimeoutExpired, ValueError, KeyError) as err:
            failures += 1
            print(f"FAIL {name}: {err}")
    shutil.rmtree(base, ignore_errors=True)
    print(f"{len(chosen) - failures} of {len(chosen)} scenarios passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
