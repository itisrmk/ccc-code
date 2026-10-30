//! Works out what kind of output a shell command produces.

/// How a command's output should be treated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Kind {
    /// Reading source or diffs: the content is the point, so it stays whole.
    Verbatim,
    /// Build and test runs: failures and the summary matter, passing noise doesn't.
    Log,
    /// Searches and file listings: the first hits and the count matter.
    Listing,
    /// Anything else: keep both ends.
    Generic,
}

/// Commands whose output is source code or a precise view of it.
const VERBATIM: &[&str] = &[
    "cat", "sed", "nl", "head", "tail", "awk", "less", "more", "bat", "diff", "jq",
];
const LISTING: &[&str] = &[
    "grep", "egrep", "fgrep", "rg", "ag", "ack", "find", "fd", "ls", "tree", "du",
];
/// Commands that always build, test or lint.
const LOG: &[&str] = &[
    "pytest",
    "jest",
    "vitest",
    "tsc",
    "eslint",
    "ruff",
    "mypy",
    "make",
    "gradle",
    "gradlew",
    "mvn",
    "xcodebuild",
    "ctest",
    "ninja",
    "cmake",
    "playwright",
    "rspec",
];
/// Tools with only some subcommands that build or test.
const LOG_SUBCOMMANDS: &[(&str, &[&str])] = &[
    (
        "cargo",
        &[
            "test", "build", "check", "clippy", "nextest", "run", "bench", "doc", "install",
        ],
    ),
    ("npm", &["test", "run", "install", "ci", "i", "build", "t"]),
    (
        "pnpm",
        &["test", "run", "install", "i", "build", "t", "exec"],
    ),
    ("yarn", &["test", "run", "install", "build"]),
    ("bun", &["test", "run", "install", "i"]),
    ("npx", &["jest", "vitest", "tsc", "eslint", "playwright"]),
    ("go", &["test", "build", "vet", "install"]),
    ("swift", &["build", "test"]),
    ("docker", &["build"]),
    ("pip", &["install"]),
    ("pip3", &["install"]),
    ("brew", &["install", "upgrade"]),
];
const GIT_VERBATIM: &[&str] = &["diff", "show", "blame"];
const GIT_LISTING: &[&str] = &["grep", "ls-files", "log"];

/// Classifies a shell command line by the first segment that says anything about its
/// output. A pipeline ending in a reader like `tail` is treated as that reader, since the
/// author already bounded the output and chose what they wanted to see.
pub(crate) fn classify(command: &str) -> Kind {
    // A heredoc body is data rather than commands, so it says nothing about the output.
    let script = command.split("<<").next().unwrap_or(command);
    let segments: Vec<&str> = script
        .split(['\n', ';', '|', '&'])
        .map(str::trim)
        .filter(|segment| !segment.is_empty())
        .collect();

    if let Some(last) = segments.last() {
        let words = command_words(last);
        if words
            .first()
            .is_some_and(|word| matches!(*word, "head" | "tail"))
        {
            return Kind::Verbatim;
        }
    }
    let mut found = Kind::Generic;
    for segment in segments {
        match classify_segment(segment) {
            Some(Kind::Log) => return Kind::Log,
            Some(kind) if found == Kind::Generic => found = kind,
            _ => {}
        }
    }
    found
}

fn classify_segment(segment: &str) -> Option<Kind> {
    let words = command_words(segment);
    let (program, rest) = words.split_first()?;
    let program = program.rsplit('/').next().unwrap_or(program);
    let first_argument = rest.iter().find(|word| !word.starts_with('-')).copied();

    if program == "git" {
        let subcommand = first_argument?;
        return if GIT_VERBATIM.contains(&subcommand) {
            Some(Kind::Verbatim)
        } else if GIT_LISTING.contains(&subcommand) {
            Some(Kind::Listing)
        } else {
            None
        };
    }
    if matches!(program, "python" | "python3")
        && rest.windows(2).any(|pair| pair == ["-m", "pytest"])
    {
        return Some(Kind::Log);
    }
    if LOG.contains(&program) {
        return Some(Kind::Log);
    }
    if let Some((_, subcommands)) = LOG_SUBCOMMANDS.iter().find(|(tool, _)| *tool == program) {
        return first_argument
            .filter(|argument| subcommands.contains(argument))
            .map(|_| Kind::Log);
    }
    if VERBATIM.contains(&program) {
        return Some(Kind::Verbatim);
    }
    if LISTING.contains(&program) {
        return Some(Kind::Listing);
    }
    None
}

/// Splits a segment into words, skipping leading `VAR=value` assignments and wrappers.
fn command_words(segment: &str) -> Vec<&str> {
    segment
        .split_whitespace()
        .skip_while(|word| {
            matches!(*word, "time" | "env" | "sudo" | "exec" | "(" | "{")
                || word
                    .split_once('=')
                    .is_some_and(|(name, _)| !name.is_empty() && !name.starts_with('-'))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_common_commands() {
        let cases = [
            ("cargo test --all", Kind::Log),
            ("cd crates && cargo clippy -- -D warnings", Kind::Log),
            (
                "export PATH=/x:$PATH; RUST_LOG=debug cargo build",
                Kind::Log,
            ),
            ("npm run build", Kind::Log),
            ("python3 -m pytest tests/", Kind::Log),
            ("/usr/local/bin/pytest -x", Kind::Log),
            ("cargo tree", Kind::Generic),
            ("rg -n foo src", Kind::Listing),
            ("ls -la", Kind::Listing),
            ("git log --oneline", Kind::Listing),
            ("git grep foo", Kind::Listing),
            ("cat src/main.rs", Kind::Verbatim),
            ("sed -n '1,40p' a.rs", Kind::Verbatim),
            ("git diff HEAD~1", Kind::Verbatim),
            ("git status", Kind::Generic),
            ("curl -s https://x.test", Kind::Generic),
            ("python3 script.py", Kind::Generic),
        ];
        for (command, want) in cases {
            assert_eq!(classify(command), want, "{command}");
        }
    }

    #[test]
    fn a_trailing_head_or_tail_means_the_author_already_bounded_it() {
        assert_eq!(classify("cargo test 2>&1 | tail -20"), Kind::Verbatim);
        assert_eq!(classify("rg foo | head -5"), Kind::Verbatim);
    }

    #[test]
    fn a_build_anywhere_in_the_line_makes_it_a_log() {
        assert_eq!(classify("ls && cargo test"), Kind::Log);
        assert_eq!(
            classify("cargo fmt; cargo test 2>&1 | grep -E 'ok|FAILED'"),
            Kind::Log
        );
    }

    #[test]
    fn heredoc_bodies_are_not_commands() {
        let command = "python3 - <<'EOF'\ncargo test\nEOF";
        assert_eq!(classify(command), Kind::Generic);
    }

    #[test]
    fn empty_input_is_generic() {
        assert_eq!(classify(""), Kind::Generic);
    }
}
