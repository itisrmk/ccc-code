#!/bin/sh
# Installs ccc for this platform, after checking it against the release's published checksum.
#   curl -fsSL https://raw.githubusercontent.com/itisrmk/ccc-code/main/install.sh | sh
# Then run: ccc setup
set -eu

repository="itisrmk/ccc-code"
destination="${CCC_INSTALL_DIR:-$HOME/.local/bin}"

case "$(uname -s) $(uname -m)" in
    "Darwin arm64") target="aarch64-apple-darwin" ;;
    "Darwin x86_64") target="x86_64-apple-darwin" ;;
    "Linux x86_64") target="x86_64-unknown-linux-gnu" ;;
    *)
        echo "ccc: no build for $(uname -s) $(uname -m) yet" >&2
        exit 1
        ;;
esac

if [ -n "${CCC_DOWNLOAD_BASE:-}" ]; then
    base="$CCC_DOWNLOAD_BASE"
    version="${CCC_VERSION:?set CCC_VERSION with CCC_DOWNLOAD_BASE}"
else
    version="$(curl -fsSL "https://api.github.com/repos/$repository/releases/latest" |
        sed -n 's/.*"tag_name": *"\(v[^"]*\)".*/\1/p' | head -n 1)"
    if [ -z "$version" ]; then
        echo "ccc: couldn't find the latest release" >&2
        exit 1
    fi
    base="https://github.com/$repository/releases/download/$version"
fi

name="ccc-$version-$target.tar.gz"
scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT

curl -fsSL -o "$scratch/$name" "$base/$name"
curl -fsSL -o "$scratch/$name.sha256" "$base/$name.sha256"
expected="$(cut -d' ' -f1 "$scratch/$name.sha256")"
actual="$(shasum -a 256 "$scratch/$name" | cut -d' ' -f1)"
if [ "$expected" != "$actual" ]; then
    echo "ccc: checksum mismatch for $name: expected $expected, got $actual. Not installing it." >&2
    exit 1
fi

tar -xzf "$scratch/$name" -C "$scratch"
mkdir -p "$destination"
cp "$scratch/ccc" "$destination/ccc.new"
chmod 755 "$destination/ccc.new"
mv "$destination/ccc.new" "$destination/ccc"

echo "Installed ccc $version to $destination/ccc"
case ":$PATH:" in
    *":$destination:"*) echo "Next, run: ccc setup" ;;
    *) echo "Next, run: $destination/ccc setup    ($destination isn't on your PATH)" ;;
esac
