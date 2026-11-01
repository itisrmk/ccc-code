#!/usr/bin/env python3
"""Builds a small git repository for `ccc eval` from this project's own synthetic bugs.

Each commit fixes one module and adds its tests, so each makes a task. Everything in it is
written here, so running it means running no outside code.

    scripts/eval_fixture.py DIR
    ccc eval mine DIR --test "python3 run_tests.py" --out tasks.json
"""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e2e import RUN_TESTS, TASKS  # noqa: E402

FIXES = {
    "pagination": ("Fix pagination: pages count from 1 and a partial page counts as a page", {
        "pager.py": """\
def page(items, number, size):
    start = (number - 1) * size
    return items[start:start + size]

def page_count(total, size):
    return -(-total // size)
""",
    }),
    "interval_merge": ("Merge intervals in any order, including touching and contained ones", {
        "intervals.py": """\
def merge(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged
""",
    }),
    "csv_quotes": ("Keep the last CSV field and read doubled quotes as one quote", {
        "csvline.py": """\
def split(line):
    fields, current, quoted, index = [], "", False, 0
    while index < len(line):
        character = line[index]
        if character == '"' and quoted and line[index + 1:index + 2] == '"':
            current += '"'
            index += 1
        elif character == '"':
            quoted = not quoted
        elif character == "," and not quoted:
            fields.append(current)
            current = ""
        else:
            current += character
        index += 1
    fields.append(current)
    return fields
""",
    }),
    "lru_cache": ("Make the cache evict the least recently used item", {
        "lru.py": """\
class Cache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.items = {}

    def get(self, key):
        if key not in self.items:
            return None
        self.items[key] = self.items.pop(key)
        return self.items[key]

    def put(self, key, value):
        self.items.pop(key, None)
        self.items[key] = value
        if len(self.items) > self.capacity:
            oldest = next(iter(self.items))
            del self.items[oldest]
""",
    }),
    "money": ("Split bills exactly, and round tax half up", {
        "money.py": """\
def split_bill(total_cents, people):
    share, extra = divmod(total_cents, people)
    return [share + 1] * extra + [share] * (people - extra)

def add_tax(cents, percent):
    return (cents * (100 + percent) + 50) // 100
""",
    }),
    "toposort": ("Report dependency cycles with CycleError", {
        "toposort.py": """\
class CycleError(Exception):
    pass

def order(graph):
    done, active, result = set(), set(), []
    def visit(node):
        if node in done:
            return
        if node in active:
            raise CycleError(node)
        active.add(node)
        for dependency in graph.get(node, []):
            visit(dependency)
        active.discard(node)
        done.add(node)
        result.append(node)
    for node in graph:
        visit(node)
    return result
""",
    }),
    "flatten": ("Flatten lists by index, keep empty containers, and drop the leading dot", {
        "flatten.py": """\
def flatten(value, prefix=""):
    out = {}
    items = value.items() if isinstance(value, dict) else enumerate(value)
    for key, item in items:
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, (dict, list)) and item:
            out.update(flatten(item, name))
        else:
            out[name] = item
    return out
""",
    }),
    "rate_limit": ("Fix the limiter's limit, its window boundary and what a refusal uses up", {
        "limiter.py": """\
class Limiter:
    def __init__(self, limit, window):
        self.limit, self.window = limit, window
        self.count, self.started = 0, 0

    def allow(self, now):
        if now - self.started >= self.window:
            self.started, self.count = now, 0
        if self.count >= self.limit:
            return False
        self.count += 1
        return True
""",
    }),
    "config_layers": ("Stop config loads leaking into each other, and merge nested values", {
        "loader.py": """\
import copy

from defaults import VALUES

def _merge(base, extra):
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value

def load(environment, overrides):
    merged = copy.deepcopy(VALUES)
    _merge(merged, overrides)
    _merge(merged, environment)
    return merged
""",
    }),
    "path_clean": ("Clean paths correctly: empty parts, absolute roots and leading parents", {
        "paths.py": """\
def clean(path):
    absolute = path.startswith("/")
    parts = []
    for part in path.split("/"):
        if part in ("", "."):
            continue
        if part != "..":
            parts.append(part)
        elif parts and parts[-1] != "..":
            parts.pop()
        elif not absolute:
            parts.append(part)
    joined = "/".join(parts)
    return "/" + joined if absolute else joined
""",
    }),
    "template": ("Read doubled braces as literal braces in templates", {
        "render.py": """\
import re

def render(template, values):
    def replace(match):
        if match.group(0) == "{{":
            return "{"
        if match.group(0) == "}}":
            return "}"
        return str(values[match.group(1)])
    return re.sub(r"\\{\\{|\\}\\}|\\{(\\w+)\\}", replace, template)
""",
    }),
    "scheduler": ("Run the highest priority job first, in the order added on ties", {
        "jobs.py": """\
import heapq

class Queue:
    def __init__(self):
        self.heap = []
        self.added = 0

    def add(self, priority, name):
        heapq.heappush(self.heap, (-priority, self.added, name))
        self.added += 1

    def next(self):
        return heapq.heappop(self.heap)[2]

    def cancel(self, name):
        self.heap = [entry for entry in self.heap if entry[2] != name]
        heapq.heapify(self.heap)
""",
    }),
}


def git(repo, *args):
    subprocess.run(
        ["git", "-c", "user.name=ccc", "-c", "user.email=ccc@localhost",
         "-c", "commit.gpgsign=false", *args],
        cwd=repo, check=True,
    )


def main():
    repo = Path(sys.argv[1])
    repo.mkdir(parents=True)
    git(repo, "init", "-q", ".")
    (repo / "run_tests.py").write_text(RUN_TESTS)
    for task in FIXES:
        for name, content in TASKS[task].items():
            if not name.startswith("test_"):
                (repo / name).write_text(content)
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "Add the modules")
    for task, (message, fixed) in FIXES.items():
        for name, content in TASKS[task].items():
            if name.startswith("test_"):
                (repo / name).write_text(content)
        for name, content in fixed.items():
            (repo / name).write_text(content)
        git(repo, "add", "-A")
        git(repo, "commit", "-qm", message)
    return 0


if __name__ == "__main__":
    sys.exit(main())
