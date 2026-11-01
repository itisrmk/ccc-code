#!/usr/bin/env python3
"""Draws docs/demo.svg: a terminal playing back real `ccc` output on a sample history.

The transcripts are generated here, so the demo shows no one's real work and its numbers are
labelled as a sample. The output itself comes from running the commands, not from typing it in.
"""

import json
import re
import subprocess
import tempfile
from datetime import datetime, timedelta, timezone
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CCC = ROOT / "target" / "release" / "ccc"
OUT = ROOT / "docs" / "demo.svg"
LINE_HEIGHT = 20
WIDTH = 900
SECONDS_PER_LINE = 0.28
PAUSE_AFTER_COMMAND = 0.9
WANTED = re.compile(r"^\s*(cache reads|cache writes|output|rewrites after idle|from\s+\d+K|300K|above)|^Where|^Cost|^Restarting")


def write_history(folder):
    """A week of sample work: two long sessions and some short ones, with idle gaps."""
    start = datetime(2026, 10, 1, 9, tzinfo=timezone.utc)
    for session, (calls, step) in enumerate([(320, 2400), (210, 3000), (40, 1500), (25, 1200)]):
        lines, context, now = [], 30_000, start + timedelta(days=session)
        for call in range(calls):
            idle = call % 90 == 89
            now += timedelta(hours=3) if idle else timedelta(seconds=40)
            written = context + step if idle else step
            read = 0 if idle else context
            context += step
            if call % 6 == 0:
                lines.append(json.dumps({"type": "user", "message": {"content": "next step"}}))
            lines.append(json.dumps({
                "type": "assistant", "requestId": f"r{session}-{call}", "sessionId": f"s{session}",
                "timestamp": now.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                "message": {"model": "claude-opus-5-5", "usage": {
                    "input_tokens": 3, "cache_read_input_tokens": read,
                    "cache_creation_input_tokens": written,
                    "cache_creation": {"ephemeral_1h_input_tokens": written},
                    "output_tokens": 250,
                }},
            }))
        (folder / f"s{session}.jsonl").write_text("\n".join(lines) + "\n")


def output(*arguments):
    return subprocess.run(
        [str(CCC), *arguments], capture_output=True, text=True, check=True
    ).stdout


def wrap(text, width=78):
    lines, line = [], ""
    for word in text.split():
        if line and len(line) + len(word) + 1 > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    return lines + [line] if line else lines


def main():
    with tempfile.TemporaryDirectory() as scratch:
        folder = Path(scratch)
        write_history(folder)
        share = output("whatif", str(folder), "--share").strip()
        report = output("whatif", str(folder)).splitlines()
    kept = [line for line in report if WANTED.match(line)]
    script = [("command", "ccc whatif --share")]
    script += [("output", line) for line in wrap(share)]
    script += [("output", ""), ("command", "ccc whatif")]
    script += [("output", line) for line in kept]
    script += [("output", ""), ("note", "Sample history. Restart figures are upper limits.")]

    colors = {"command": "#7ee787", "output": "#e6edf3", "note": "#8b949e"}
    rows, clock = [], 0.6
    for index, (kind, text) in enumerate(script):
        shown = f"$ {text}" if kind == "command" else text
        y = 58 + index * LINE_HEIGHT
        rows.append(
            f'<text x="20" y="{y}" fill="{colors[kind]}" xml:space="preserve" '
            f'style="animation-delay:{clock:.2f}s">{escape(shown)}</text>'
        )
        clock += PAUSE_AFTER_COMMAND if kind == "command" else SECONDS_PER_LINE
    height = 58 + len(script) * LINE_HEIGHT + 10
    OUT.write_text(f"""<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" viewBox="0 0 {WIDTH} {height}" role="img" aria-label="ccc whatif output on a sample history">
<style>
text {{ font: 14px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; opacity: 0; animation: show .01s linear forwards; }}
@keyframes show {{ to {{ opacity: 1; }} }}
@media (prefers-reduced-motion: reduce) {{ text {{ animation: none; opacity: 1; }} }}
</style>
<rect width="{WIDTH}" height="{height}" rx="10" fill="#0d1117"/>
<circle cx="22" cy="20" r="6" fill="#ff5f56"/><circle cx="42" cy="20" r="6" fill="#ffbd2e"/><circle cx="62" cy="20" r="6" fill="#27c93f"/>
{chr(10).join(rows)}
</svg>
""")
    print(f"wrote {OUT}, {len(script)} lines, {clock:.0f} seconds")
    print(share)


if __name__ == "__main__":
    main()
