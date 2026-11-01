#!/usr/bin/env python3
"""Does restarting a long session with `ccc handoff` really save money, and what does it lose?

`ccc whatif` can only simulate this. Here real Claude Code works through 12 tasks in one repository
two ways: in one continuing session, and restarting every few tasks with a handoff. A rule stated
only in the first prompt checks whether the handoff carries what matters. The score comes from
running each module's tests and reading the log, never from anything the model says.

    scripts/restart_test.py --model claude-haiku-5-5 --reps 3 --max-usd 5 --out results.jsonl
    scripts/restart_test.py --large --restart-every 6 --max-usd 15 --out results.jsonl

--large gives each module long design notes that must be read, so sessions grow past 300K tokens,
the size at which `ccc whatif` simulates savings. A ticket number at the end of each note has to
reach the log, which checks the notes really were read.

Everything it runs is written in this repository. Spends Claude quota.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e2e import RUN_TESTS, TASKS  # noqa: E402
from eval_fixture import FIXES  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CCC = Path(os.environ.get("CCC_BIN", ROOT / "target" / "release" / "ccc"))
# Long enough that test output fills context the way real builds do, but under the size at which
# Claude Code moves output to a file.
NOISE_CASES = 800
RULE = (
    "Project rule for all of this work: every fix must add one line to FIXES.log in the repository "
    "root naming the module file you fixed. "
)
# About 20K tokens each, just under what Claude Code's Read tool takes in one go.
NOTE_LINES = 1400
NOTES_PER_TASK = 2
WORDS = (
    "the module keeps its state in memory because callers read it far more often than they write "
    "and earlier versions tried a file which turned out slower under load so review this before "
    "changing how values are stored or passed between functions in later releases"
).split()
LARGE_RULE = (
    "Project rule for all of this work: every fix must add one line to FIXES.log in the repository "
    "root naming the module file you fixed and the ticket numbers from the end of its notes. "
)
FIRST_FILE = {task: next(name for name in fixed) for task, (_, fixed) in FIXES.items()}


def ticket(task, number):
    return f"T-{sum(map(ord, task)) * 7 + number}"


def write_notes(work):
    (work / "notes").mkdir()
    for task in FIXES:
        for number in range(NOTES_PER_TASK):
            lines = []
            for line in range(NOTE_LINES):
                start = (line * 7 + number) % len(WORDS)
                words = (WORDS + WORDS)[start:start + 12]
                lines.append(f"{task} {number} {line}: {' '.join(words)}")
            lines.append(f"Ticket: {ticket(task, number)}")
            (work / "notes" / f"{task}-{number}.md").write_text("\n".join(lines) + "\n")


def set_up(base, large=False):
    work = Path(tempfile.mkdtemp(dir=base, prefix="work-"))
    work = work.resolve()
    subprocess.run(["git", "init", "-q", "."], cwd=work, check=True)
    (work / "run_tests.py").write_text(RUN_TESTS.replace("range(300)", f"range({NOISE_CASES})"))
    for task in FIXES:
        for name, content in TASKS[task].items():
            (work / name).write_text(content)
    if large:
        write_notes(work)
    return work


def task_prompt(task, first, large=False):
    message = FIXES[task][0]
    rule = RULE if first else ""
    notes = ""
    if large:
        rule = LARGE_RULE if first else ""
        names = " and ".join(f"notes/{task}-{number}.md" for number in range(NOTES_PER_TASK))
        notes = (
            f"Before changing anything, read all of {names} with the Read tool, without using "
            "grep, head or tail on them. "
        )
    return (
        f"{rule}{notes}Work on {FIRST_FILE[task]} only for now: {message}. Tests for every module run "
        "with 'python3 run_tests.py', and tests for modules not reached yet will still fail. Don't "
        "change test files or run_tests.py. Stop when this module's tests pass."
    )


def claude(work, prompt, model, session=None, budget=6.0):
    command = [
        "claude", "-p", prompt, "--model", model,
        "--allowedTools", "Bash,Read,Edit,Write,Grep,Glob",
        "--output-format", "json", "--setting-sources", "project", "--strict-mcp-config",
        "--max-budget-usd", str(budget),
    ]
    if session:
        command += ["--resume", session]
    done = subprocess.run(
        command, cwd=work, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=900
    )
    try:
        return json.loads(done.stdout)
    except ValueError:
        return {"is_error": True, "subtype": "no report", "total_cost_usd": 0, "session_id": session}


def project_dir(work):
    return Path.home() / ".claude" / "projects" / re.sub(r"[^A-Za-z0-9]", "-", str(work))


def module_passes(work, task):
    tests = [name for name in TASKS[task] if name.startswith("test_")]
    script = (
        "import importlib, sys\nsys.path.insert(0, '.')\n"
        f"module = importlib.import_module({tests[0][:-3]!r})\n"
        "for name in dir(module):\n"
        "    if name.startswith('test_'):\n"
        "        getattr(module, name)()\n"
    )
    return subprocess.run(
        [sys.executable, "-c", script], cwd=work, capture_output=True, timeout=60
    ).returncode == 0


def tests_intact(work):
    return all(
        (work / name).read_text() == content
        for task in FIXES
        for name, content in TASKS[task].items()
        if name.startswith("test_")
    )


def recorded_cost(folder, base):
    """Cost from recorded usage. `claude -p` may report a running total on a resume, so what it
    says per run isn't summed for the score."""
    export = Path(tempfile.mkstemp(dir=base, suffix=".json")[1])
    done = subprocess.run(
        [str(CCC), "whatif", str(folder), "--export", str(export)],
        capture_output=True, text=True, timeout=120,
    )
    if done.returncode != 0:
        raise RuntimeError(done.stderr[-300:])
    return json.loads(export.read_text())["total_usd"]


def peak_context(folder):
    peak = 0
    for path in folder.glob("*.jsonl"):
        for line in path.read_text(errors="ignore").splitlines():
            try:
                record = json.loads(line)
            except ValueError:
                continue
            usage = (record.get("message") or {}).get("usage") if record.get("type") == "assistant" else None
            if isinstance(usage, dict):
                peak = max(peak, sum(usage.get(key, 0) for key in (
                    "input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")))
    return peak


def run_arm(job):
    arm, repetition, model, restart_every, base, large = job
    work = set_up(base, large)
    tasks = list(FIXES)
    reported, restarts, errors, session = 0.0, 0, [], None
    for index, task in enumerate(tasks):
        prompt = task_prompt(task, index == 0, large)
        if arm == "restart" and index and index % restart_every == 0:
            transcript = project_dir(work) / f"{session}.jsonl"
            handoff = subprocess.run(
                [str(CCC), "handoff", "--transcript", str(transcript)],
                capture_output=True, text=True, timeout=60,
            )
            if handoff.returncode != 0:
                errors.append(f"handoff: {handoff.stderr[-200:]}")
            else:
                prompt = f"{handoff.stdout}\n\n## What to do now\n\n{prompt}"
                session = None
                restarts += 1
        result = claude(work, prompt, model, session)
        reported += result.get("total_cost_usd") or 0
        session = result.get("session_id") or session
        if result.get("is_error"):
            errors.append(result.get("subtype") or "error")
    passed = [task for task in tasks if module_passes(work, task)]
    log = (work / "FIXES.log").read_text() if (work / "FIXES.log").exists() else ""
    logged = [task for task in tasks if FIRST_FILE[task] in log or FIRST_FILE[task][:-3] in log]
    tickets = sum(
        ticket(task, number) in log for task in tasks for number in range(NOTES_PER_TASK)
    )
    record = {
        "arm": arm, "repetition": repetition, "model": model, "restart_every": restart_every,
        "cost_usd": recorded_cost(project_dir(work), base), "reported_sum_usd": reported,
        "restarts": restarts, "modules_passed": len(passed),
        "rule_followed": len(logged), "tickets_logged": tickets if large else None,
        "large": large, "modules": len(tasks), "tests_intact": tests_intact(work),
        "peak_context": peak_context(project_dir(work)), "errors": errors,
    }
    shutil.rmtree(project_dir(work), ignore_errors=True)
    shutil.rmtree(work, ignore_errors=True)
    return record


def main():
    global NOTES_PER_TASK
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="claude-haiku-5-5")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--restart-every", type=int, default=4)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--large", action="store_true", help="Grow sessions past 300K tokens")
    parser.add_argument("--notes-per-task", type=int, default=NOTES_PER_TASK)
    parser.add_argument("--max-usd", type=float, required=True, help="Start no new run past this")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    NOTES_PER_TASK = args.notes_per_task
    base = Path(tempfile.mkdtemp(prefix="ccc-restart-")).resolve()
    jobs = [
        (arm, repetition, args.model, args.restart_every, base, args.large)
        for repetition in range(args.reps)
        for arm in ("continuous", "restart")
    ]
    spent, records = 0.0, []
    with open(args.out, "a") as out:
        for start in range(0, len(jobs), args.jobs):
            if spent >= args.max_usd:
                print(f"Stopped at ${spent:.2f}, the limit.")
                break
            with ThreadPoolExecutor(max_workers=args.jobs) as pool:
                for record in pool.map(run_arm, jobs[start:start + args.jobs]):
                    records.append(record)
                    spent += record["cost_usd"]
                    out.write(json.dumps(record) + "\n")
                    out.flush()
                    print(json.dumps(record), flush=True)
    shutil.rmtree(base, ignore_errors=True)
    print(f"\n{args.model}, restarting every {args.restart_every} tasks")
    print(f"{'arm':<12}{'runs':>6}{'mean cost':>12}{'peak context':>14}{'modules':>10}{'rule':>8}")
    for arm in ("continuous", "restart"):
        rows = [row for row in records if row["arm"] == arm]
        if not rows:
            continue
        mean = lambda key: sum(row[key] for row in rows) / len(rows)
        print(
            f"{arm:<12}{len(rows):>6}{'$' + format(mean('cost_usd'), '.4f'):>12}"
            f"{mean('peak_context'):>14.0f}{mean('modules_passed'):>7.1f}/12{mean('rule_followed'):>5.1f}/12"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
