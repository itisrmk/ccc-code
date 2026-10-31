#!/bin/sh
# Offline checks: formatting, lints, unit and integration tests, docs and proxy overhead.
# Costs nothing and sends nothing anywhere. Run before every commit.
set -eu

cd "$(dirname "$0")/.."
export PATH="/opt/homebrew/opt/rustup/bin:$PATH"

cargo fmt --all --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features
cargo test --release -p ccc-proxy --test overhead -- --ignored --nocapture --test-threads=1
echo "check.sh: all offline checks passed"
