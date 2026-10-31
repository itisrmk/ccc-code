#!/usr/bin/env python3
"""End-to-end task success: real Claude Code fixing real bugs, with and without ccc.

The score is whether the tests really pass afterward, checked by this script and not by
anything the model says. Every test run prints hundreds of passing lines, so tool output
that a prune rule could damage is part of every task.

    scripts/e2e.py                                 Haiku, every configuration, 2 repetitions
    scripts/e2e.py --reps 3 --model claude-sonnet-5-5
    scripts/e2e.py --configs baseline ccc --tasks interval_merge

Writes one JSON line per run to --out and prints a comparison. Spends Claude quota: about
$0.02 per run on Haiku and $0.15 on Sonnet at list prices.
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CCC = ROOT / "target" / "release" / "ccc"
NOISE_CASES = 300
PROMPT = (
    "The tests in this directory are failing. Run them with 'python3 run_tests.py', find the "
    "bugs, and fix them in the source files. Don't change run_tests.py or the test files. "
    "Stop once they pass."
)

RUN_TESTS = """\
import importlib, sys, traceback, glob

failed = 0
for i in range(%d):
    print(f"test infrastructure::smoke_{i} ... ok")
sys.path.insert(0, ".")
for path in sorted(glob.glob("test_*.py")):
    module = importlib.import_module(path[:-3])
    for name in sorted(dir(module)):
        if not name.startswith("test_"):
            continue
        try:
            getattr(module, name)()
            print(f"test {path}::{name} ... ok")
        except Exception:
            failed += 1
            print(f"test {path}::{name} ... FAILED")
            traceback.print_exc(file=sys.stdout)
print(f"test result: {'FAILED' if failed else 'ok'}. {failed} failed")
sys.exit(1 if failed else 0)
""" % NOISE_CASES

TASKS = {
    "pagination": {
        "pager.py": """\
def page(items, number, size):
    start = number * size
    return items[start:start + size]

def page_count(total, size):
    return total // size
""",
        "test_pager.py": """\
from pager import page, page_count

def test_first_page():
    assert page([1, 2, 3, 4, 5], 1, 2) == [1, 2]

def test_last_page():
    assert page([1, 2, 3, 4, 5], 3, 2) == [5]

def test_counts_a_partial_page():
    assert page_count(5, 2) == 3

def test_counts_exact_pages():
    assert page_count(4, 2) == 2
    assert page_count(0, 2) == 0
""",
    },
    "interval_merge": {
        "intervals.py": """\
def merge(intervals):
    merged = []
    for start, end in intervals:
        if merged and start < merged[-1][1]:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged
""",
        "test_intervals.py": """\
from intervals import merge

def test_overlap():
    assert merge([(1, 3), (2, 4)]) == [(1, 4)]

def test_touching_intervals_join():
    assert merge([(1, 2), (2, 3)]) == [(1, 3)]

def test_contained_interval():
    assert merge([(1, 10), (2, 3)]) == [(1, 10)]

def test_unsorted_input():
    assert merge([(5, 6), (1, 2)]) == [(1, 2), (5, 6)]
""",
    },
    "csv_quotes": {
        "csvline.py": """\
def split(line):
    fields, current, quoted = [], "", False
    for character in line:
        if character == '"':
            quoted = not quoted
        elif character == "," and not quoted:
            fields.append(current)
            current = ""
        else:
            current += character
    return fields
""",
        "test_csvline.py": """\
from csvline import split

def test_plain():
    assert split("a,b,c") == ["a", "b", "c"]

def test_quoted_comma():
    assert split('a,"b,c",d') == ["a", "b,c", "d"]

def test_escaped_quote():
    assert split('"say ""hi""",x') == ['say "hi"', "x"]

def test_empty_fields():
    assert split("a,,") == ["a", "", ""]
""",
    },
    "lru_cache": {
        "lru.py": """\
class Cache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.items = {}

    def get(self, key):
        return self.items.get(key)

    def put(self, key, value):
        self.items[key] = value
        if len(self.items) > self.capacity:
            oldest = next(iter(self.items))
            del self.items[oldest]
""",
        "test_lru.py": """\
from lru import Cache

def test_evicts_when_full():
    cache = Cache(2)
    cache.put("a", 1); cache.put("b", 2); cache.put("c", 3)
    assert cache.get("a") is None

def test_reading_keeps_an_item_alive():
    cache = Cache(2)
    cache.put("a", 1); cache.put("b", 2)
    cache.get("a")
    cache.put("c", 3)
    assert cache.get("a") == 1
    assert cache.get("b") is None

def test_overwriting_keeps_an_item_alive():
    cache = Cache(2)
    cache.put("a", 1); cache.put("b", 2)
    cache.put("a", 9)
    cache.put("c", 3)
    assert cache.get("a") == 9
    assert cache.get("b") is None
""",
    },
    "money": {
        "money.py": """\
def split_bill(total_cents, people):
    share = round(total_cents / people)
    return [share] * people

def add_tax(cents, percent):
    return round(cents * (1 + percent / 100))
""",
        "test_money.py": """\
from money import split_bill, add_tax

def test_shares_add_up():
    shares = split_bill(100, 3)
    assert sum(shares) == 100
    assert max(shares) - min(shares) <= 1

def test_earlier_people_pay_the_extra_cents():
    assert split_bill(101, 3) == [34, 34, 33]

def test_tax_rounds_half_up():
    assert add_tax(250, 10) == 275
    assert add_tax(5, 10) == 6
""",
    },
    "toposort": {
        "toposort.py": """\
def order(graph):
    seen, result = set(), []
    def visit(node):
        if node in seen:
            return
        seen.add(node)
        for dependency in graph.get(node, []):
            visit(dependency)
        result.append(node)
    for node in graph:
        visit(node)
    return result
""",
        "test_toposort.py": """\
from toposort import order, CycleError

def test_dependencies_come_first():
    result = order({"a": ["b"], "b": ["c"], "c": []})
    assert result.index("c") < result.index("b") < result.index("a")

def test_detects_cycles():
    try:
        order({"a": ["b"], "b": ["a"]})
    except CycleError:
        return
    raise AssertionError("expected CycleError")

def test_diamond_is_not_a_cycle():
    result = order({"a": ["b", "c"], "b": ["d"], "c": ["d"], "d": []})
    assert result.index("d") < result.index("b")
""",
    },
    "flatten": {
        "flatten.py": """\
def flatten(value, prefix=""):
    out = {}
    for key, item in value.items():
        name = prefix + "." + key
        if isinstance(item, dict):
            out.update(flatten(item, name))
        else:
            out[name] = item
    return out
""",
        "test_flatten.py": """\
from flatten import flatten

def test_nested():
    assert flatten({"a": {"b": 1}}) == {"a.b": 1}

def test_lists_use_indexes():
    assert flatten({"a": [1, {"b": 2}]}) == {"a.0": 1, "a.1.b": 2}

def test_empty_containers_are_kept():
    assert flatten({"a": {}, "b": []}) == {"a": {}, "b": []}
""",
    },
    "rate_limit": {
        "limiter.py": """\
class Limiter:
    def __init__(self, limit, window):
        self.limit, self.window = limit, window
        self.count, self.started = 0, 0

    def allow(self, now):
        if now - self.started > self.window:
            self.started, self.count = now, 0
        self.count += 1
        return self.count < self.limit
""",
        "test_limiter.py": """\
from limiter import Limiter

def test_allows_up_to_the_limit():
    limiter = Limiter(2, 10)
    assert limiter.allow(0)
    assert limiter.allow(1)
    assert not limiter.allow(2)

def test_a_new_window_starts_exactly_at_its_boundary():
    limiter = Limiter(1, 10)
    assert limiter.allow(0)
    assert not limiter.allow(9)
    assert limiter.allow(10)

def test_a_refused_call_does_not_use_up_the_next_window():
    limiter = Limiter(1, 10)
    limiter.allow(0)
    limiter.allow(5)
    assert limiter.allow(10)
    assert not limiter.allow(11)
""",
    },
    "config_layers": {
        "defaults.py": 'VALUES = {"port": 80, "host": "localhost", "tls": {"enabled": False, "port": 443}}\n',
        "loader.py": """\
from defaults import VALUES

def load(environment, overrides):
    merged = VALUES
    merged.update(overrides)
    merged.update(environment)
    return merged
""",
        "test_loader.py": """\
from loader import load

def test_environment_beats_overrides_beats_defaults():
    assert load({"port": 1}, {"port": 2})["port"] == 1
    assert load({}, {"port": 2})["port"] == 2

def test_calls_do_not_leak_into_each_other():
    load({"port": 1}, {})
    assert load({}, {})["port"] == 80

def test_nested_values_merge():
    merged = load({"tls": {"enabled": True}}, {})
    assert merged["tls"] == {"enabled": True, "port": 443}
""",
    },
    "path_clean": {
        "paths.py": """\
def clean(path):
    parts = []
    for part in path.split("/"):
        if part == "..":
            parts.pop()
        elif part != ".":
            parts.append(part)
    return "/".join(parts)
""",
        "test_paths.py": """\
from paths import clean

def test_dots():
    assert clean("a/./b/../c") == "a/c"

def test_repeated_slashes():
    assert clean("a//b") == "a/b"

def test_absolute_stays_absolute_and_cannot_escape_root():
    assert clean("/a/../..") == "/"
    assert clean("/a/b") == "/a/b"

def test_relative_keeps_leading_parent():
    assert clean("../a") == "../a"
    assert clean("a/../../b") == "../b"
""",
    },
    "template": {
        "render.py": """\
import re

def render(template, values):
    def replace(match):
        return str(values[match.group(1)])
    return re.sub(r"\\{(\\w+)\\}", replace, template)
""",
        "test_render.py": """\
from render import render

def test_substitutes():
    assert render("hi {name}", {"name": "a"}) == "hi a"

def test_doubled_braces_are_literal():
    assert render("{{x}} {y}", {"y": 1}) == "{x} 1"

def test_a_value_containing_braces_is_not_expanded_again():
    assert render("{a}", {"a": "{b}", "b": 2}) == "{b}"

def test_missing_values_raise_a_clear_error():
    try:
        render("{gone}", {})
    except KeyError as err:
        assert "gone" in str(err)
        return
    raise AssertionError("expected KeyError")
""",
    },
    "scheduler": {
        "jobs.py": """\
import heapq

class Queue:
    def __init__(self):
        self.heap = []

    def add(self, priority, name):
        heapq.heappush(self.heap, (priority, name))

    def next(self):
        return heapq.heappop(self.heap)[1]

    def cancel(self, name):
        self.heap = [entry for entry in self.heap if entry[1] != name]
""",
        "test_jobs.py": """\
from jobs import Queue

def test_higher_priority_first():
    queue = Queue()
    queue.add(1, "low"); queue.add(5, "high")
    assert queue.next() == "high"

def test_equal_priorities_run_in_the_order_added():
    queue = Queue()
    queue.add(1, "zebra"); queue.add(1, "apple")
    assert queue.next() == "zebra"

def test_cancel_keeps_the_rest_in_order():
    queue = Queue()
    for priority in [9, 1, 8, 2, 7, 3]:
        queue.add(priority, str(priority))
    queue.cancel("9")
    assert queue.next() == "8"
    assert queue.next() == "7"
""",
    },
}

CONFIGS = {
    "baseline": None,
    "ccc": [],
    "ccc+enrich": ["--enrich"],
}


def set_up(task, base):
    work = Path(tempfile.mkdtemp(dir=base, prefix=task + "-"))
    subprocess.run(["git", "init", "-q", "."], cwd=work, check=True)
    (work / "run_tests.py").write_text(RUN_TESTS)
    for name, content in TASKS[task].items():
        (work / name).write_text(content)
    return work


def protected_files_intact(task, work):
    """Passing doesn't count if the tests or their runner were changed."""
    if (work / "run_tests.py").read_text() != RUN_TESTS:
        return False
    return all(
        (work / name).read_text() == content
        for name, content in TASKS[task].items()
        if name.startswith("test_")
    )


def run_once(job):
    task, config, repetition, model, base = job
    work = set_up(task, base)
    state = Path(tempfile.mkdtemp(dir=base, prefix="state-"))
    claude = [
        "claude", "-p", PROMPT, "--model", model,
        "--allowedTools", "Bash,Read,Edit,Write",
        "--output-format", "json",
    ]
    extra = CONFIGS[config]
    command = (
        claude
        if extra is None
        else [str(CCC), "run", "--state-dir", str(state), *extra, "--", *claude]
    )
    started = time.time()
    record = {"task": task, "config": config, "repetition": repetition, "model": model}
    try:
        done = subprocess.run(
            command, cwd=work, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=420
        )
        output = json.loads(done.stdout)
        usage = output.get("usage", {})
        record.update(
            is_error=bool(output.get("is_error")),
            turns=output.get("num_turns"),
            cost_usd=output.get("total_cost_usd"),
            cache_read=usage.get("cache_read_input_tokens"),
            cache_write=usage.get("cache_creation_input_tokens"),
        )
    except (subprocess.TimeoutExpired, ValueError) as err:
        record.update(is_error=True, error=type(err).__name__)
    check = subprocess.run(
        [sys.executable, "run_tests.py"], cwd=work, capture_output=True, text=True, timeout=60
    )
    record["passed"] = check.returncode == 0 and protected_files_intact(task, work)
    record["seconds"] = round(time.time() - started, 1)
    shutil.rmtree(work, ignore_errors=True)
    shutil.rmtree(state, ignore_errors=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="claude-haiku-5-5")
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    parser.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    parser.add_argument("--out", default="e2e-results.jsonl")
    parser.add_argument("--check-tasks", action="store_true", help="Only confirm each task starts out failing")
    args = parser.parse_args()

    base = tempfile.mkdtemp(prefix="ccc-e2e-")
    if args.check_tasks:
        for task in args.tasks:
            work = set_up(task, base)
            check = subprocess.run([sys.executable, "run_tests.py"], cwd=work, capture_output=True, text=True)
            print(f"{task}: {'fails as intended' if check.returncode else 'ALREADY PASSES'}")
        shutil.rmtree(base, ignore_errors=True)
        return 0

    jobs = [
        (task, config, repetition, args.model, base)
        for repetition in range(args.reps)
        for task in args.tasks
        for config in args.configs
    ]
    results = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool, open(args.out, "a") as out:
        for record in pool.map(run_once, jobs):
            results.append(record)
            out.write(json.dumps(record) + "\n")
            out.flush()
            print(
                f"{record['task']:<16}{record['config']:<12}"
                f"{'pass' if record['passed'] else 'FAIL':<6}"
                f"turns {record.get('turns')}  ${record.get('cost_usd') or 0:.4f}",
                flush=True,
            )
    shutil.rmtree(base, ignore_errors=True)

    print(f"\n{args.model}, {args.reps} repetitions of {len(args.tasks)} tasks")
    print(f"{'config':<14}{'passed':>10}{'rate':>9}{'mean turns':>12}{'mean cost':>11}{'mean cache read':>17}")
    baseline_rate = None
    for config in args.configs:
        rows = [row for row in results if row["config"] == config]
        passed = sum(row["passed"] for row in rows)
        rate = 100 * passed / len(rows)
        mean = lambda key: sum(row.get(key) or 0 for row in rows) / len(rows)
        if config == "baseline":
            baseline_rate = rate
        delta = "" if baseline_rate is None or config == "baseline" else f"  ({rate - baseline_rate:+.1f} pts)"
        print(
            f"{config:<14}{passed:>6}/{len(rows):<3}{rate:>8.1f}%{mean('turns'):>12.1f}"
            f"{'$' + format(mean('cost_usd'), '.4f'):>11}{mean('cache_read'):>17.0f}{delta}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
