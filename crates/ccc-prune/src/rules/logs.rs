//! Build and test logs: keep failures, frames in the repo and the summary.

use std::collections::BTreeSet;
use std::sync::LazyLock;

use regex::Regex;

use super::Limits;

/// Lines around a failure that often explain it.
const CONTEXT_LINES: usize = 2;
/// Lines kept from the start, which usually say what ran.
const HEAD_LINES: usize = 5;
/// Lines kept from the end, which usually hold the summary.
const TAIL_LINES: usize = 15;
/// A log can be nothing but failures. Past this, keeping more repeats the same news.
const MAX_SIGNAL_LINES: usize = 150;

static SIGNAL: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(concat!(
        r"(?i)\b(error|errors|fail|failed|failure|failures|panic|panicked|exception|traceback|",
        r"fatal|assert|assertion|warning|undefined|cannot|denied|timeout|timed out|not found)\b",
        r"|^\s*-->|^\s+at\s|^\s*File "
    ))
    .expect("signal pattern is valid")
});

/// Stack frames from dependencies or the toolchain, which rarely explain a failure.
static THIRD_PARTY_FRAME: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^\s*(-->|at\s|File ).*(node_modules|/\.cargo/|site-packages|/rustc/|/rustlib/)")
        .expect("frame pattern is valid")
});

/// Trims a log to its signal, or returns `None` if it is short enough to keep.
pub(super) fn trim(limits: &Limits, text: &str) -> Option<String> {
    if text.len() < limits.log_min_bytes {
        return None;
    }
    let lines: Vec<&str> = text.lines().collect();
    let mut keep: BTreeSet<usize> = BTreeSet::new();
    keep.extend(0..HEAD_LINES.min(lines.len()));
    keep.extend(lines.len().saturating_sub(TAIL_LINES)..lines.len());

    keep.extend(signal_lines(&lines));
    Some(render(&lines, &keep))
}

/// The lines reporting failures, with the lines around them that often explain them.
pub(super) fn signal_lines(lines: &[&str]) -> BTreeSet<usize> {
    let mut keep = BTreeSet::new();
    let signals = lines
        .iter()
        .enumerate()
        .filter(|(_, line)| SIGNAL.is_match(line) && !THIRD_PARTY_FRAME.is_match(line))
        .map(|(index, _)| index);
    for index in signals.take(MAX_SIGNAL_LINES) {
        let end = (index + CONTEXT_LINES + 1).min(lines.len());
        keep.extend(index.saturating_sub(CONTEXT_LINES)..end);
    }
    keep
}

/// Renders the kept lines in order, marking each gap with what was dropped.
fn render(lines: &[&str], keep: &BTreeSet<usize>) -> String {
    let mut out = String::new();
    let mut next = 0;
    for &index in keep {
        if index > next {
            out.push_str(&format!("[ccc: {} lines omitted]\n", index - next));
        }
        out.push_str(lines[index]);
        out.push('\n');
        next = index + 1;
    }
    if next < lines.len() {
        out.push_str(&format!("[ccc: {} lines omitted]\n", lines.len() - next));
    }
    out.push_str(&format!(
        "[ccc: kept {} of {} log lines. Re-run with a filter to see the rest.]\n",
        keep.len(),
        lines.len()
    ));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn numbered(count: usize) -> Vec<String> {
        (0..count)
            .map(|i| format!("compiling crate_{i} v1.0.0"))
            .collect()
    }

    #[test]
    fn short_logs_stay_whole() {
        assert_eq!(trim(&Limits::default(), "error: nope\n"), None);
    }

    #[test]
    fn keeps_failures_with_context_plus_both_ends() {
        let mut lines = numbered(1000);
        lines[500] = "error[E0308]: mismatched types".to_owned();
        let log = lines.join("\n");

        let kept = trim(&Limits::default(), &log).unwrap();

        for line in [
            "crate_0 ",
            "crate_4 ",
            "crate_498 ",
            "crate_499 ",
            "error[E0308]",
            "crate_501 ",
            "crate_999 ",
        ] {
            assert!(kept.contains(line), "missing {line:?}");
        }
        assert!(!kept.contains("crate_250 "), "{kept}");
        assert!(kept.contains("lines omitted"));
        assert!(kept.contains("of 1000 log lines"));
    }

    #[test]
    fn every_kept_line_is_verbatim_and_in_order() {
        let mut lines = numbered(500);
        lines[100] = "thread 'x' panicked at src/lib.rs:1:1".to_owned();
        lines[300] = "  --> src/main.rs:9:9".to_owned();
        let log = lines.join("\n");

        let kept = trim(&Limits::default(), &log).unwrap();

        let mut remaining = lines.iter();
        for line in kept.lines().filter(|line| !line.starts_with("[ccc:")) {
            assert!(
                remaining.any(|original| original == line),
                "{line:?} isn't an in-order line of the input"
            );
        }
    }

    #[test]
    fn repo_frames_stay_and_dependency_frames_go() {
        let mut lines = numbered(600);
        lines[200] = "    at handler (src/app.ts:10:2)".to_owned();
        lines[400] = "    at run (node_modules/x/index.js:1:1)".to_owned();
        let log = lines.join("\n");
        let kept = trim(&Limits::default(), &log).unwrap();
        assert!(kept.contains("src/app.ts"));
        assert!(!kept.contains("node_modules"));
    }

    #[test]
    fn a_log_of_only_failures_is_capped() {
        let log: String = (0..5000)
            .map(|i| format!("error: failure number {i}\n"))
            .collect();
        let kept = trim(&Limits::default(), &log).unwrap();
        assert!(kept.lines().count() < 250, "{}", kept.lines().count());
    }
}
