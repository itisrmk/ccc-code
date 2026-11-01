# Not yet usable: fill in the checksum once there's a tagged release. Builds from source without
# the optional local model and name detection features.
class Ccc < Formula
  desc "See what Claude Code really costs, and test what helps"
  homepage "https://github.com/itisrmk/ccc-code"
  url "https://github.com/itisrmk/ccc-code/archive/refs/tags/v0.1.0.tar.gz"
  sha256 "REPLACE_WITH_RELEASE_SHA256"
  license "MIT"
  head "https://github.com/itisrmk/ccc-code.git", branch: "main"

  depends_on "rust" => :build

  def install
    system "cargo", "install", *std_cargo_args(path: "crates/ccc")
  end

  test do
    assert_match "ccc", shell_output("#{bin}/ccc --version")
  end
end
