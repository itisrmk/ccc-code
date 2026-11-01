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
import socket
import string
import subprocess
import sys
import tempfile
import time
import urllib.request
from datetime import datetime, timedelta, timezone
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


def scenario_enrich(box):
    """Hints reach the model, stay byte-stable across turns, and survive a resume."""
    (box.work / "src").mkdir()
    (box.work / "src" / "tokens.py").write_text(
        "def verify_token(value):\n    return value == 'ok'\n"
    )
    (box.work / "src" / "billing.py").write_text("def charge_card():\n    return 1\n")
    first = run_headless(
        box,
        "Think carefully and work out 37*43 by hand. Run 'echo RESULT' with Bash using your "
        "answer. Then, without opening any files, tell me which source file you were told may "
        "relate to verifying a token. Reply with only that path.",
        ccc_extra=("--enrich",),
    )
    check("tokens.py" in first["result"], f"the hint didn't arrive: {first['result']!r}")
    second = run_headless(
        box,
        "Think carefully and work out 41*47 by hand. Run 'echo RESULT' with Bash using your "
        "answer, then 'echo again' with Bash, then reply with only the word finished.",
        "--resume",
        first["session_id"],
        ccc_extra=("--enrich",),
    )
    check("finished" in second["result"].lower(), f"resume failed: {second['result']!r}")
    exchanges = box.exchanges()
    assert_all_accepted(exchanges)
    fired = {rule for exchange in exchanges for rule in exchange["rules"]}
    check("enrich" in fired, f"enrich never fired: {fired}")
    return f"{len(exchanges)} requests accepted under enforcement, hint reached the model"


def scenario_hooks(box):
    """Hooks registered through settings really fire, and log the session's cost."""
    settings = subprocess.run(
        [str(CCC), "hook-settings", "--state-dir", str(box.state)],
        capture_output=True, text=True, check=True,
    ).stdout
    settings_file = box.base / "settings.json"
    settings_file.write_text(settings)
    result = run_headless(
        box, "Reply with only the word done.", "--settings", str(settings_file)
    )
    check(not result["is_error"], f"task errored: {result.get('result')}")
    log = box.state / "hooks.log"
    check(log.exists(), "no hook ever ran")
    events = [json.loads(line) for line in log.read_text().splitlines()]
    seen = {event["event"] for event in events}
    check({"session-start", "stop"} <= seen, f"expected start and stop hooks, saw {seen}")
    deadline = time.time() + 10
    index_dir = box.state / "index"
    while time.time() < deadline and not any(index_dir.glob("*.json")):
        time.sleep(0.25)
    check(any(index_dir.glob("*.json")), "the start hook didn't build an index")
    turns = (box.state / "turns.log").read_text()
    check(result["session_id"] in turns, f"the stop hook logged no cost: {turns}")
    return f"hooks fired: {sorted(seen)}, index built, cost logged"


def scenario_names(box):
    """Names only a model can spot stay local and come back real."""
    model_dir = Path.home() / ".ccc" / "models" / "gliner"
    built = subprocess.run([str(CCC), "--help"], capture_output=True, text=True)
    if "ner-bench" not in built.stdout or not model_dir.exists():
        return "SKIPPED: needs --features ner and the GLiNER model"
    name = "Maria Gonzalez"
    result = run_headless(
        box,
        f"Our customer {name} at Northwind Traders needs a reply. Run 'echo noted' with Bash, "
        "then reply with only the customer's full name, exactly as I wrote it.",
        ccc_extra=("--ner-model", str(model_dir)),
    )
    check(name in result["result"], f"the real name didn't come back: {result['result']!r}")
    exchanges = box.exchanges()
    assert_all_accepted(exchanges, require_thinking=False)
    for index, exchange in enumerate(exchanges):
        check(name.encode() not in exchange["sent_bytes"], f"request {index} sent the name")
    return f"{len(exchanges)} requests, name stayed local and came back real"


def scenario_learned_router(box):
    """A trained router really moves an easy conversation down and keeps it there."""
    router = box.base / "router.json"
    trained = subprocess.run(
        [str(CCC), "router-train", str(ROOT / "data" / "router-labels.jsonl"), "--out", str(router)],
        capture_output=True, text=True, check=True,
    ).stdout
    result = run_headless(
        box,
        "Fix the typo: run 'echo hello' with Bash, then reply with only the word done.",
        ccc_extra=(
            "--learned-router", str(router),
            "--router-cheap-model", "claude-haiku-5-5",
            "--router-confidence", "0.5",
        ),
        model="claude-sonnet-5-5",
    )
    check(not result["is_error"], f"task errored: {result.get('result')}")
    exchanges = box.exchanges()
    assert_all_accepted(exchanges, require_thinking=False)
    agent_calls = [e for e in exchanges if e["sent"].get("tools")]
    models = {exchange["sent"]["model"] for exchange in agent_calls}
    check(models == {"claude-haiku-5-5"}, f"expected every agent call on Haiku, saw {models}")
    return f"{len(agent_calls)} agent calls all on Haiku; {trained.strip()}"


def scenario_policy(box):
    """A team policy refuses a run that breaks it, before any traffic moves."""
    policy = box.base / "policy.toml"
    policy.write_text('require_redact = true\nredact_rules = ["Project Bluebird"]\n')
    refused = subprocess.run(
        [str(CCC), "run", "--policy", str(policy), "--no-redact", "--state-dir", str(box.state),
         "--", "true"],
        capture_output=True, text=True,
    )
    check(refused.returncode != 0, "a run breaking the policy was allowed")
    check("--no-redact" in refused.stderr, f"unclear refusal: {refused.stderr[-300:]}")

    result = run_headless(
        box,
        "Our codename is Project Bluebird. Reply with only that codename, exactly as written.",
        ccc_extra=("--policy", str(policy)),
    )
    check("Project Bluebird" in result["result"], f"not restored: {result['result']!r}")
    for index, exchange in enumerate(box.exchanges()):
        check(b"Project Bluebird" not in exchange["sent_bytes"], f"request {index} sent the codename")
    return "refused --no-redact; policy rule redacted a codename and restored it"


def scenario_dashboard(box):
    """The dashboard serves the ledger's sessions on loopback."""
    result = run_headless(box, "Reply with only the word done.")
    port = 34000 + (os.getpid() % 1000)
    server = subprocess.Popen(
        [str(CCC), "dashboard", "--listen", f"127.0.0.1:{port}", "--state-dir", str(box.state)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        page = ""
        deadline = time.time() + 10
        while time.time() < deadline and not page:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=2) as response:
                    page = response.read().decode()
            except OSError:
                time.sleep(0.2)
    finally:
        server.terminate()
        server.wait()
    check(result["session_id"] in page, f"session missing from dashboard: {page[:300]}")
    return "served the session just recorded"


def scenario_codex(box):
    """Codex answers through the proxy. Opt in: it spends ChatGPT quota, not Claude's."""
    if not os.environ.get("CCC_LIVE_CODEX"):
        return "SKIPPED: set CCC_LIVE_CODEX=1 to spend a little ChatGPT quota"
    done = subprocess.run(
        [str(CCC), "run", "--state-dir", str(box.state), "--dump", str(box.dump),
         "--upstream", "https://chatgpt.com/backend-api/codex", "--",
         "codex", "exec", "--skip-git-repo-check", "Reply with only the word ok"],
        cwd=box.work, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=180,
    )
    check(done.returncode == 0, f"exit {done.returncode}: {done.stderr[-300:]}")
    check("ok" in done.stdout.lower(), f"no answer: {done.stdout[-200:]}")
    check("unchanged" in done.stderr, "no notice that traffic isn't compiled")
    check(any(box.dump.glob("*.req-head.json")), "no traffic went through ccc")
    return "Codex answered through the proxy, with the forwarding notice shown"


MODELS = Path.home() / ".ccc" / "models"
LABEL_MODEL = MODELS / "Qwen3-4B-Q4_K_M.gguf"


def has_local_model():
    """Whether this binary was built with local models and the weights are downloaded."""
    built = subprocess.run([str(CCC), "--help"], capture_output=True, text=True)
    return "model-bench" in built.stdout and LABEL_MODEL.exists()


def scenario_local_model(box):
    """Real weights on the request path. Kept lines must be verbatim, and history valid."""
    if not has_local_model():
        return "SKIPPED: needs --features local-model and the 4B weights"
    source = ""
    for i in range(60):
        source += f"/// Computes item {i}.\npub fn item_{i}(value: u32) -> u32 {{\n"
        for j in range(5):
            source += f"    let step_{j} = value + {i * j};\n"
        source += "    step_4\n}\n"
    (box.work / "big.rs").write_text(source)
    result = run_headless(
        box,
        "One tool call at a time, with Bash. 1) Run 'cat big.rs'. 2) Think carefully and work "
        "out 37*43 by hand. 3) Run 'echo RESULT' using your answer. 4) Reply with only done.",
        ccc_extra=(
            "--classify-model", str(LABEL_MODEL), "--classify-cap-ms", "5000",
            "--prune-model", str(LABEL_MODEL), "--prune-model-cap-ms", "120000",
        ),
    )
    check(not result["is_error"], f"task errored: {result.get('result')}")
    exchanges = box.exchanges()
    assert_all_accepted(exchanges)
    fired = {rule for exchange in exchanges for rule in exchange["rules"]}
    decisions = [path.read_text()[:300] for path in (box.state / "model-decisions").glob("*.json")]
    check("model" in fired, f"the model step made no edit: {fired}, decisions {decisions}")
    original = set(source.splitlines())
    for message in exchanges[-1]["sent"]["messages"]:
        if not isinstance(message["content"], list):
            continue
        for block in message["content"]:
            content = block.get("content")
            if block.get("type") == "tool_result" and isinstance(content, str) and "fn item_" in content:
                for line in content.splitlines():
                    check(
                        line in original or line.startswith("[ccc:"),
                        f"the model paraphrased a line: {line!r}",
                    )
    return f"{len(exchanges)} requests accepted under enforcement, kept lines verbatim"


def scenario_slow_model_floor(box):
    """At the default cap this machine's model is too slow, so the stage must turn itself off."""
    if not has_local_model():
        return "SKIPPED: needs --features local-model and the 4B weights"
    source = "".join(f"fn item_{i}(value: u32) -> u32 {{ value + {i} }}\n" for i in range(420))
    (box.work / "big.rs").write_text(source)
    started = time.time()
    result = run_headless(
        box,
        "Run 'cat big.rs' with Bash, then reply with only done.",
        ccc_extra=("--prune-model", str(LABEL_MODEL)),
    )
    check(not result["is_error"], f"task errored: {result.get('result')}")
    exchanges = box.exchanges()
    fired = {rule for exchange in exchanges for rule in exchange["rules"]}
    check("model" not in fired, f"a too-slow model step ran anyway: {fired}")
    return f"stage stayed off, task finished in {time.time() - started:.0f}s"


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


def advice_settings(box, *flags):
    settings = subprocess.run(
        [str(CCC), "hook-settings", "--state-dir", str(box.state), "--advise",
         "--advise-min-context", "1000", *flags],
        capture_output=True, text=True, check=True,
    ).stdout
    return json.loads(settings)


def age_transcript(box, session_id, hours=3):
    """Moves a throwaway session's records into the past, so its cache looks expired without
    waiting an hour. Nothing else about the session changes."""
    marker = re.sub(r"[^A-Za-z0-9]", "-", os.path.realpath(box.work))
    path = Path.home() / ".claude" / "projects" / marker / f"{session_id}.jsonl"
    check(path.exists(), f"no transcript at {path}")
    aged = datetime.now(timezone.utc) - timedelta(hours=hours)
    stamp = aged.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    text = re.sub(r'"timestamp":"[^"]+"', f'"timestamp":"{stamp}"', path.read_text())
    path.write_text(text)


def scenario_advise(box):
    """Resuming after the cache expired warns the person, and the model never sees the warning."""
    settings_file = box.base / "settings.json"
    settings_file.write_text(json.dumps(advice_settings(box)))
    first = run_headless(box, "Reply with only the word done.", "--settings", str(settings_file))
    check(not first["is_error"], f"task errored: {first.get('result')}")
    age_transcript(box, first["session_id"])
    before = len(box.exchanges())
    command = ccc_args(box) + [
        "claude", "-p", "Reply with only the word ok.", "--model", MODEL,
        "--resume", first["session_id"], "--settings", str(settings_file),
        "--output-format", "stream-json", "--verbose",
    ]
    done = subprocess.run(
        command, cwd=box.work, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=300
    )
    check(done.returncode == 0, f"exit {done.returncode}: {done.stderr[-400:]}")
    check("prompt cache expired" in done.stdout, f"no warning shown: {done.stdout[-800:]}")
    later = box.exchanges()[before:]
    assert_all_accepted(later, require_thinking=False)
    for exchange in later:
        check(
            b"prompt cache expired" not in exchange["sent_bytes"],
            "the warning reached the model",
        )
    return "the warning was shown on resume, stayed out of the request, and history stayed valid"


def scenario_advise_block(box):
    """Blocking holds back the first prompt after the cache expired, and sending it again works."""
    settings = advice_settings(box, "--advise-block")
    # Leaves only the prompt hook, which is what an idle open session goes through.
    settings["hooks"] = {"UserPromptSubmit": settings["hooks"]["UserPromptSubmit"]}
    settings_file = box.base / "settings.json"
    settings_file.write_text(json.dumps({"hooks": {}}))
    first = run_headless(box, "Reply with only the word done.", "--settings", str(settings_file))
    age_transcript(box, first["session_id"])
    settings_file.write_text(json.dumps(settings))
    before = len(box.exchanges())

    def resume():
        return subprocess.run(
            ccc_args(box) + [
                "claude", "-p", "Reply with only the word ok.", "--model", MODEL,
                "--resume", first["session_id"], "--settings", str(settings_file),
                "--output-format", "stream-json", "--verbose",
            ],
            cwd=box.work, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=300,
        )

    held = resume()
    check("prompt cache expired" in held.stdout + held.stderr, f"no warning: {held.stdout[-800:]}")
    sent_while_held = [
        exchange for exchange in box.exchanges()[before:] if not is_quota_probe(exchange)
        and b"Reply with only the word ok." in exchange["sent_bytes"]
    ]
    check(not sent_while_held, "the blocked prompt was sent anyway")
    # Each Claude Code run ages nothing: the transcript's last call is still the old one.
    allowed = resume()
    check(allowed.returncode == 0, f"exit {allowed.returncode}: {allowed.stderr[-400:]}")
    sent = [
        exchange for exchange in box.exchanges()[before:]
        if b"Reply with only the word ok." in exchange["sent_bytes"]
    ]
    check(sent, "sending the prompt again didn't go through")
    assert_all_accepted(sent, require_thinking=False)
    return "the first prompt was held back with the price, and the second went through"


def scenario_init(box):
    """`ccc init` writes hooks that really warn, and removes them again, leaving other settings."""
    settings_file = box.base / "settings.json"
    settings_file.write_text(json.dumps({"permissions": {"allow": []}}))
    subprocess.run(
        [str(CCC), "init", "--settings-file", str(settings_file), "--state-dir", str(box.state)],
        check=True, capture_output=True,
    )
    # Warnings stay quiet below 100K tokens, too large for a cheap test, so this lowers the bar
    # the way a person could with the same flag.
    text = settings_file.read_text().replace(
        "--state-dir", "--advise --advise-min-context 1000 --state-dir"
    )
    settings_file.write_text(text)
    first = run_headless(box, "Reply with only the word done.", "--settings", str(settings_file))
    check(not first["is_error"], f"task errored: {first.get('result')}")
    age_transcript(box, first["session_id"])
    done = subprocess.run(
        ccc_args(box) + [
            "claude", "-p", "Reply with only the word ok.", "--model", MODEL,
            "--resume", first["session_id"], "--settings", str(settings_file),
            "--output-format", "stream-json", "--verbose",
        ],
        cwd=box.work, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=300,
    )
    check("prompt cache expired" in done.stdout, f"no warning shown: {done.stdout[-800:]}")
    check("ccc handoff --start" in done.stdout, "the warning doesn't name the way out")
    subprocess.run(
        [str(CCC), "init", "--settings-file", str(settings_file), "--uninstall"],
        check=True, capture_output=True,
    )
    left = json.loads(settings_file.read_text())
    check(left == {"permissions": {"allow": []}}, f"uninstall left {left}")
    return "installed, warned on a real resume, and uninstalled cleanly"


def scenario_handoff(box):
    """A fresh session given only `ccc handoff` knows what the first session was told."""
    codename = "Project " + "Larkspur"
    first = run_headless(
        box,
        f"This work is called {codename}. Use Bash to create a file notes.txt holding the word "
        "ready, then reply with only the word done.",
    )
    check(not first["is_error"], f"task errored: {first.get('result')}")
    marker = re.sub(r"[^A-Za-z0-9]", "-", os.path.realpath(box.work))
    transcript = Path.home() / ".claude" / "projects" / marker / f"{first['session_id']}.jsonl"
    handoff = subprocess.run(
        [str(CCC), "handoff", "--transcript", str(transcript)],
        capture_output=True, text=True, timeout=60, check=True,
    ).stdout
    check(codename in handoff, f"the handoff lost the first request: {handoff[:500]}")
    second = run_headless(
        box, f"{handoff}\n\nWhat is this work called? Reply with only its name."
    )
    check(second["session_id"] != first["session_id"], "that wasn't a fresh session")
    check("Larkspur" in second["result"], f"fresh session didn't know: {second['result']}")
    return f"a fresh session recalled it from {len(handoff)} characters of handoff"


def has_model(name):
    return (Path.home() / ".ccc" / "models" / name).exists()


LOCAL_EMBEDDING = "embeddinggemma-300M-Q8_0.gguf"
LOCAL_GEMMA = "gemma-4-E2B-it-Q4_0.gguf"
SKIP_NO_LOCAL_WORK = "SKIPPED: needs the local-model build and scripts/fetch-models.sh embedding gemma"


def builds_local_work():
    help_text = subprocess.run([str(CCC), "--help"], capture_output=True, text=True).stdout
    return " ask " in help_text and has_model(LOCAL_EMBEDDING) and has_model(LOCAL_GEMMA)


def write_auth_repo(box):
    (box.work / "gatekeeper.py").write_text(
        "def verify_session_token(token):\n    return token.startswith('ok-')\n"
    )
    (box.work / "unrelated.py").write_text("def add(a, b):\n    return a + b\n")


def scenario_enhance(box):
    """The prompt hook adds files as context, leaves the person's words alone, and stays valid."""
    write_auth_repo(box)
    settings_file = box.base / "settings.json"
    settings_file.write_text("{}")
    subprocess.run(
        [str(CCC), "init", "--enhance", "--settings-file", str(settings_file),
         "--state-dir", str(box.state)],
        check=True, capture_output=True,
    )
    prompt = "Does verify session token accept empty tokens? Reply with only yes or no."
    first = run_headless(box, prompt, "--settings", str(settings_file))
    check(not first["is_error"], f"task errored: {first.get('result')}")
    second = run_headless(
        box, "And does verify session token accept ok-1? Reply yes or no.",
        "--resume", first["session_id"], "--settings", str(settings_file),
    )
    check(not second["is_error"], f"resume errored: {second.get('result')}")
    exchanges = box.exchanges()
    assert_all_accepted(exchanges)
    sent = exchanges[-1]["sent_bytes"].decode(errors="ignore")
    check("gatekeeper.py" in sent, "the suggested file never reached the model")
    check("verify_session_token" in sent, "the symbol never reached the model")
    check(prompt in sent, "the person's words were changed")
    return f"context reached the model across a resume, {len(exchanges)} requests accepted"


def scenario_ask(box):
    """`ccc ask` answers from local code using real weights, with no network service involved."""
    if not builds_local_work():
        return SKIP_NO_LOCAL_WORK
    write_auth_repo(box)
    done = subprocess.run(
        [str(CCC), "ask", "Which file checks session tokens?",
         "--state-dir", str(Path.home() / ".ccc")],
        cwd=box.work, capture_output=True, text=True, timeout=300,
        # A dead proxy address: any attempt to reach a service would fail.
        env={**os.environ, "ANTHROPIC_BASE_URL": "http://127.0.0.1:9", "ANTHROPIC_API_KEY": "x"},
    )
    check(done.returncode == 0, f"exit {done.returncode}: {done.stderr[-400:]}")
    # The list of what it looked at always names the file, so only the answer itself counts.
    answer = done.stdout.split("Looked at:")[0]
    check("gatekeeper" in answer, f"wrong answer: {done.stdout[-500:]}")
    return "answered from local files with real weights, offline"


def scenario_mcp(box):
    """Claude really calls the local search tool, and uses what it finds."""
    if not builds_local_work():
        return SKIP_NO_LOCAL_WORK
    write_auth_repo(box)
    config = box.base / "mcp.json"
    config.write_text(json.dumps({"mcpServers": {"ccc": {
        "type": "stdio", "command": str(CCC),
        "args": ["mcp", "--state-dir", str(Path.home() / ".ccc")],
    }}}))
    done = subprocess.run(
        ccc_args(box) + [
            "claude", "-p",
            "Use the mcp__ccc__search_code tool with the query 'checking session tokens'. "
            "Reply with only the name of the best matching file.",
            "--model", MODEL, "--mcp-config", str(config), "--strict-mcp-config",
            "--allowedTools", "mcp__ccc__search_code", "--output-format", "json",
        ],
        cwd=box.work, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=300,
    )
    check(done.returncode == 0, f"exit {done.returncode}: {done.stderr[-400:]}")
    result = json.loads(done.stdout)
    check("gatekeeper" in result["result"], f"wrong answer: {result['result']}")
    sent = "".join(exchange["sent_bytes"].decode(errors="ignore") for exchange in box.exchanges())
    check("gatekeeper.py:1" in sent, "the tool's result never reached the model")
    return "Claude called local search and answered from its result"


def scenario_setup(box):
    """A clean install under a throwaway home: settings, status line and full removal."""
    home = box.base / "home"
    (home / ".claude").mkdir(parents=True)
    settings = home / ".claude" / "settings.json"
    original = {"permissions": {"allow": []}}
    settings.write_text(json.dumps(original))
    env = {**os.environ, "HOME": str(home)}
    # A server that isn't there, so this needs no download and no real model.
    done = subprocess.run(
        [str(CCC), "setup", "--server", "http://127.0.0.1:9/v1", "--embedding-model", "x", "--no-mcp"],
        env=env, cwd=box.work, capture_output=True, text=True, timeout=60,
    )
    check(done.returncode == 0, f"setup failed: {done.stderr[-400:]}")
    wired = json.loads(settings.read_text())
    check(set(wired["hooks"]) == {"SessionStart", "UserPromptSubmit"}, f"hooks: {wired['hooks']}")
    check("statusline" in wired["statusLine"]["command"], f"status line: {wired}")
    doctor = subprocess.run(
        [str(CCC), "doctor", "--no-start"], env=env, cwd=box.work,
        capture_output=True, text=True, timeout=60,
    )
    check("Cost warnings are on" in doctor.stdout, f"doctor: {doctor.stdout}")
    sample = json.dumps({"rate_limits": {"seven_day": {"used_percentage": 62}}})
    line = subprocess.run(
        [str(CCC), "statusline"], input=sample, env={**env, "NO_COLOR": "1"},
        capture_output=True, text=True, timeout=10,
    )
    check(line.stdout.strip() == "week limit 62%", f"status line said {line.stdout!r}")
    removed = subprocess.run(
        [str(CCC), "uninstall", "--purge", "--no-mcp"], env=env, cwd=box.work,
        capture_output=True, text=True, timeout=60,
    )
    check(removed.returncode == 0, f"uninstall failed: {removed.stderr[-400:]}")
    check(json.loads(settings.read_text()) == original, "settings weren't restored")
    check(not (home / ".ccc").exists(), "the state directory is still there")
    check(not any(home.joinpath(".claude").glob("*ccc*")), "a leftover file remains")
    return "installed, reported healthy, and left nothing behind"


def scenario_status_line(box):
    """The real TUI shows our status line, built from what Claude Code really sends."""
    captured = box.base / "status-input.jsonl"
    settings_file = box.base / "settings.json"
    settings_file.write_text(json.dumps({"statusLine": {
        "type": "command",
        "command": (
            f"tee -a {shlex.quote(str(captured))} | NO_COLOR=1 "
            f"{shlex.quote(str(CCC))} statusline --state-dir {shlex.quote(str(box.state))}"
        ),
    }}))
    terminal = Terminal(["claude", "--model", MODEL, "--settings", str(settings_file)], box.work)
    try:
        # The trust question can take a while on a busy machine, and typing into it does nothing.
        terminal.wait_for(r"Yes, I trust this folder|shift\+tab", 30)
        for _ in range(5):
            if "Yes, I trust this folder" not in terminal.screen():
                break
            terminal.press("Down")
            time.sleep(0.5)
            terminal.press("Enter")
            time.sleep(3)
        time.sleep(3)
        terminal.type("Reply with only the word ok.")
        time.sleep(1)
        terminal.press("Enter")
        shown = terminal.wait_for(r"context \d+", 45)
        screen = terminal.screen()
    finally:
        terminal.close()
    check(captured.exists(), f"Claude Code never ran the status line:\n{screen[-900:]}")
    last = json.loads(captured.read_text().splitlines()[-1])
    check(shown, f"not on screen. It sent {sorted(last)}, usage {last.get('context_window')}:\n{screen[-500:]}")
    line = next(line.strip() for line in screen.splitlines() if re.search(r"context \d+", line))
    return f"on screen: {line[:80]}"


def scenario_service_recovers(box):
    """Killing the service never hurts a prompt: the hook still answers, quickly."""
    write_auth_repo(box)
    state = box.state
    state.mkdir(exist_ok=True)
    config = state / "config.toml"
    config.write_text('server_url = "http://127.0.0.1:9/v1"\nembedding = "x"\n')
    event = json.dumps({"session_id": "s", "cwd": str(box.work), "prompt": "verify session token rules"})

    def hook():
        started = time.time()
        done = subprocess.run(
            [str(CCC), "hook", "enhance-prompt", "--state-dir", str(state)],
            input=event, capture_output=True, text=True, timeout=30,
        )
        check(done.returncode == 0, f"a hook failed a prompt: {done.stderr[-300:]}")
        return done.stdout, time.time() - started

    first, _ = hook()
    check("gatekeeper.py" in first, f"no context: {first}")
    subprocess.run(["pkill", "-9", "-f", f"daemon --state-dir {state}"], capture_output=True)
    second, seconds = hook()
    check("gatekeeper.py" in second, f"no context after the kill: {second}")
    check(seconds < 5, f"the hook took {seconds:.1f}s")
    subprocess.run(["pkill", "-f", f"daemon --state-dir {state}"], capture_output=True)
    return f"still answered {seconds:.2f}s after the service was killed"


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def wait_for_port(port, deadline_seconds=20):
    deadline = time.time() + deadline_seconds
    while time.time() < deadline:
        with socket.socket() as probe:
            probe.settimeout(0.25)
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.1)
    raise Failed(f"nothing listening on port {port}")


def scenario_verify(box):
    """`ccc verify` passes ccc's own proxy with its default stages, for free."""
    port = free_port()
    spawn = (
        f"{shlex.quote(str(CCC))} serve --upstream {{upstream}} --listen 127.0.0.1:{port} "
        f"--state-dir {shlex.quote(str(box.state))}"
    )
    done = subprocess.run(
        [str(CCC), "verify", "--gateway", f"http://127.0.0.1:{port}", "--spawn", spawn, "--json"],
        cwd=box.work, capture_output=True, text=True, timeout=120,
    )
    check(done.returncode == 0, f"exit {done.returncode}: {done.stdout[-800:]} {done.stderr[-300:]}")
    report = json.loads(done.stdout)
    check(report["passed"], f"verify failed: {report}")
    return f"{len(report['checks'])} checks, none failed"


def scenario_gateway_chain(box):
    """The real server's verdict: Claude Code through a gateway to an enforcing ccc to the API.
    This is how to get a real answer for any gateway that passes `ccc verify`."""
    enforcing_port, gateway_port = free_port(), free_port()
    enforcing_state, gateway_state = box.base / "enforcing", box.base / "gateway"
    servers = [
        subprocess.Popen(
            [str(CCC), "serve", "--no-prune", "--no-redact", "--enforce-thinking-check",
             "--listen", f"127.0.0.1:{enforcing_port}", "--state-dir", str(enforcing_state),
             "--dump", str(box.dump), "--upstream", "https://api.anthropic.com"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ),
        subprocess.Popen(
            [str(CCC), "serve", "--listen", f"127.0.0.1:{gateway_port}",
             "--state-dir", str(gateway_state), "--upstream", f"http://127.0.0.1:{enforcing_port}"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ),
    ]
    try:
        wait_for_port(enforcing_port)
        wait_for_port(gateway_port)
        environment = {**os.environ, "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{gateway_port}"}
        box.write_test_script(800)

        def claude(prompt, *extra):
            done = subprocess.run(
                ["claude", "-p", prompt, "--model", MODEL, "--allowedTools", "Bash",
                 "--output-format", "json", *extra],
                cwd=box.work, env=environment, stdin=subprocess.DEVNULL,
                capture_output=True, text=True, timeout=300,
            )
            check(done.returncode == 0, f"exit {done.returncode}: {done.stderr[-400:]}")
            return json.loads(done.stdout)

        first = claude("Run 'make test' with Bash, then name the failing test in one line.")
        check(not first["is_error"], f"task errored: {first.get('result')}")
        second = claude("Say ok.", "--resume", first["session_id"])
        check(not second["is_error"], f"resume errored: {second.get('result')}")
    finally:
        for server in servers:
            server.terminate()
            server.wait(timeout=10)
    exchanges = box.exchanges()
    assert_all_accepted(exchanges)
    return f"{len(exchanges)} requests accepted by the real API under strict enforcement"


def scenario_eval(box):
    """`ccc eval` mines a real commit and scores a real agent by running the tests."""
    repo = box.base / "repo"
    repo.mkdir()
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false"]
    subprocess.run(git + ["init", "-q", "."], cwd=repo, check=True)
    (repo / "tests").mkdir()
    (repo / "pager.py").write_text("def page_count(total, size):\n    return total // size\n")
    (repo / "tests" / "check.py").write_text(
        "import sys\nsys.path.insert(0, '.')\nfrom pager import page_count\n"
        "assert page_count(4, 2) == 2\n"
    )
    subprocess.run(git + ["add", "-A"], cwd=repo, check=True)
    subprocess.run(git + ["commit", "-qm", "Start"], cwd=repo, check=True)
    (repo / "pager.py").write_text(
        "def page_count(total, size):\n    return -(-total // size)\n"
    )
    with open(repo / "tests" / "check.py", "a") as test:
        test.write("assert page_count(5, 2) == 3\n")
    subprocess.run(git + ["add", "-A"], cwd=repo, check=True)
    subprocess.run(
        git + ["commit", "-qm", "Count a partial page as a page"], cwd=repo, check=True
    )
    tasks, results = box.base / "tasks.json", box.base / "results.jsonl"
    mined = subprocess.run(
        [str(CCC), "eval", "mine", str(repo), "--test", "python3 tests/check.py",
         "--out", str(tasks)],
        capture_output=True, text=True, timeout=120,
    )
    check(mined.returncode == 0, f"mining failed: {mined.stderr[-400:]}")
    check(json.loads(tasks.read_text())["tasks"], f"no task found: {mined.stdout}")
    empty = box.base / "empty"
    empty.mkdir()
    ran = subprocess.run(
        [str(CCC), "eval", "run", str(tasks), "--setup", "a", "--setup", f"b={empty}",
         "--out", str(results), "--max-usd", "1", "--reps", "1", "--model", MODEL],
        capture_output=True, text=True, timeout=600,
    )
    check(ran.returncode == 0, f"run failed: {ran.stderr[-400:]}")
    records = [json.loads(line) for line in results.read_text().splitlines()]
    check(len(records) == 2, f"expected 2 runs: {ran.stdout}")
    check(all(record["passed"] for record in records), f"a real fix wasn't scored: {records}")
    check(all(record["cost_usd"] > 0 for record in records), f"no cost recorded: {records}")
    check("too few tasks differ" in ran.stdout, f"one task should be too few: {ran.stdout}")
    return f"{len(records)} real runs scored by the tests, ${sum(r['cost_usd'] for r in records):.3f}"


def forget_transcripts(base):
    """Removes transcripts Claude Code wrote for this run's throwaway directories, so test
    sessions don't end up in the user's history or in `ccc bench` results."""
    projects = Path.home() / ".claude" / "projects"
    marker = re.sub(r"[^A-Za-z0-9]", "-", os.path.realpath(base))
    if not projects.is_dir():
        return
    for project in projects.iterdir():
        if marker in project.name:
            shutil.rmtree(project, ignore_errors=True)


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
    # A session killed with its terminal can still be writing its transcript for a moment.
    time.sleep(2)
    forget_transcripts(base)
    shutil.rmtree(base, ignore_errors=True)
    print(f"{len(chosen) - failures} of {len(chosen)} scenarios passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
