#!/usr/bin/env python3
"""Tests the npm wrapper against a local stand-in for a release: it downloads, checks, caches and
runs the binary, and refuses one whose checksum doesn't match. Free and offline."""

import functools
import hashlib
import http.server
import json
import platform
import subprocess
import sys
import tarfile
import tempfile
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WRAPPER = ROOT / "packaging" / "npm" / "bin" / "ccc.js"
VERSION = json.loads((ROOT / "packaging" / "npm" / "package.json").read_text())["version"]
TARGET = {
    ("Darwin", "arm64"): "aarch64-apple-darwin",
    ("Darwin", "x86_64"): "x86_64-apple-darwin",
    ("Linux", "x86_64"): "x86_64-unknown-linux-gnu",
}[(platform.system(), platform.machine())]


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_):
        pass


def publish(directory, tamper):
    name = f"ccc-v{VERSION}-{TARGET}.tar.gz"
    with tarfile.open(directory / name, "w:gz") as archive:
        archive.add(ROOT / "target" / "release" / "ccc", arcname="ccc")
    digest = hashlib.sha256((directory / name).read_bytes()).hexdigest()
    if tamper:
        digest = "0" * 64
    (directory / f"{name}.sha256").write_text(f"{digest}  {name}\n")


def run(base, install_dir):
    return subprocess.run(
        ["node", str(WRAPPER), "--version"],
        env={"PATH": "/usr/bin:/bin:/opt/homebrew/bin", "CCC_DOWNLOAD_BASE": base,
             "CCC_INSTALL_DIR": str(install_dir), "HOME": str(install_dir)},
        capture_output=True, text=True, timeout=120,
    )


def main():
    with tempfile.TemporaryDirectory() as scratch:
        scratch = Path(scratch)
        site = scratch / "site"
        site.mkdir()
        handler = functools.partial(Quiet, directory=str(site))
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{server.server_address[1]}"

        publish(site, tamper=True)
        refused = run(base, scratch / "bad")
        assert refused.returncode != 0, refused.stdout
        assert "checksum mismatch" in refused.stderr, refused.stderr
        assert not (scratch / "bad" / VERSION / "ccc").exists(), "a refused binary was kept"
        print("ok: a wrong checksum is refused and nothing is installed")

        publish(site, tamper=False)
        first = run(base, scratch / "good")
        assert first.returncode == 0, first.stderr
        assert first.stdout.startswith("ccc "), first.stdout
        print(f"ok: downloaded, checked and ran: {first.stdout.strip()}")

        server.shutdown()
        cached = run("http://127.0.0.1:9", scratch / "good")
        assert cached.returncode == 0, cached.stderr
        print("ok: the second run used the cached binary, with the server gone")
    return 0


if __name__ == "__main__":
    sys.exit(main())
