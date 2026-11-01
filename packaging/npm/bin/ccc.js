#!/usr/bin/env node
// Runs the ccc binary for this platform, downloading it on first use. The download is checked
// against the checksum published with the release, and nothing runs if they differ.
"use strict";

const { spawnSync } = require("node:child_process");
const crypto = require("node:crypto");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");

const VERSION = require("../package.json").version;
const REPOSITORY = "itisrmk/ccc-code";
const TARGETS = {
  "darwin arm64": "aarch64-apple-darwin",
  "darwin x64": "x86_64-apple-darwin",
  "linux x64": "x86_64-unknown-linux-gnu",
};

function fail(message) {
  process.stderr.write(`ccc: ${message}\n`);
  process.exit(1);
}

async function fetchBytes(url) {
  const response = await fetch(url, { redirect: "follow" });
  if (!response.ok) {
    fail(`downloading ${url} failed with ${response.status}`);
  }
  return Buffer.from(await response.arrayBuffer());
}

async function install(target, destination) {
  const name = `ccc-v${VERSION}-${target}.tar.gz`;
  const base =
    process.env.CCC_DOWNLOAD_BASE ||
    `https://github.com/${REPOSITORY}/releases/download/v${VERSION}`;
  const archive = await fetchBytes(`${base}/${name}`);
  const published = (await fetchBytes(`${base}/${name}.sha256`)).toString().trim().split(/\s+/)[0];
  const actual = crypto.createHash("sha256").update(archive).digest("hex");
  if (published !== actual) {
    fail(`checksum mismatch for ${name}: expected ${published}, got ${actual}. Not running it.`);
  }
  const scratch = fs.mkdtempSync(path.join(os.tmpdir(), "ccc-install-"));
  try {
    const saved = path.join(scratch, name);
    fs.writeFileSync(saved, archive);
    const unpacked = spawnSync("tar", ["-xzf", saved, "-C", scratch], { stdio: "inherit" });
    if (unpacked.status !== 0) {
      fail(`unpacking ${name} failed`);
    }
    fs.mkdirSync(path.dirname(destination), { recursive: true });
    fs.chmodSync(path.join(scratch, "ccc"), 0o755);
    // Copied then renamed, so a stopped download can't leave a broken binary in place.
    const partial = `${destination}.${process.pid}`;
    fs.copyFileSync(path.join(scratch, "ccc"), partial);
    fs.chmodSync(partial, 0o755);
    fs.renameSync(partial, destination);
  } finally {
    fs.rmSync(scratch, { recursive: true, force: true });
  }
}

async function main() {
  const target = TARGETS[`${process.platform} ${process.arch}`];
  if (!target) {
    fail(`no build for ${process.platform} ${process.arch} yet`);
  }
  const home = process.env.CCC_INSTALL_DIR || path.join(os.homedir(), ".ccc", "bin");
  const binary = path.join(home, VERSION, "ccc");
  if (!fs.existsSync(binary)) {
    await install(target, binary);
  }
  const done = spawnSync(binary, process.argv.slice(2), { stdio: "inherit" });
  process.exit(done.status ?? 1);
}

main().catch((err) => fail(err.message));
