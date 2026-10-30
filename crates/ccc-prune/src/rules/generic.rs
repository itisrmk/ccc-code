//! Other command output: keep both ends, since output usually starts with what ran and
//! ends with what happened.

use super::Limits;
use super::logs::signal_lines;

/// Trims long output to its two ends and any failures between them, or returns `None`
/// if it is short enough to keep. Failures stay because any script can be a test runner,
/// and the command line can't always tell.
pub(super) fn trim(limits: &Limits, text: &str) -> Option<String> {
    if text.len() < limits.generic_min_bytes {
        return None;
    }
    let head_end = line_end_near(text, limits.generic_head_bytes);
    let tail_start = line_start_near(text, text.len().saturating_sub(limits.generic_tail_bytes));
    if tail_start <= head_end {
        return None;
    }
    let middle: Vec<&str> = text[head_end..tail_start].lines().collect();
    let keep = signal_lines(&middle);

    let mut out = text[..head_end].trim_end_matches('\n').to_owned();
    out.push('\n');
    let mut next = 0;
    for &index in &keep {
        if index > next {
            out.push_str(&format!("{}\n", omitted_marker(index - next)));
        }
        out.push_str(middle[index]);
        out.push('\n');
        next = index + 1;
    }
    if next < middle.len() {
        out.push_str(&format!("{}\n", omitted_marker(middle.len() - next)));
    }
    out.push_str(&text[tail_start..]);
    Some(out)
}

fn omitted_marker(lines: usize) -> String {
    format!("[ccc: {lines} lines omitted. Re-run with a filter to see them.]")
}

/// The end of the line containing `at`, or `at` itself on a line too long to keep whole.
fn line_end_near(text: &str, at: usize) -> usize {
    let at = floor_char_boundary(text, at);
    match text[at..].find('\n') {
        Some(offset) if offset <= at / 4 => at + offset + 1,
        _ => at,
    }
}

/// The start of the line after `at`, or `at` itself on a line too long to keep whole.
fn line_start_near(text: &str, at: usize) -> usize {
    let at = floor_char_boundary(text, at);
    match text[at..].find('\n') {
        Some(offset) if offset <= (text.len() - at) / 4 => at + offset + 1,
        _ => at,
    }
}

fn floor_char_boundary(text: &str, at: usize) -> usize {
    (0..=at.min(text.len()))
        .rev()
        .find(|index| text.is_char_boundary(*index))
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_both_ends_on_line_boundaries() {
        let text: String = (0..2000).map(|i| format!("line number {i}\n")).collect();
        let kept = trim(&Limits::default(), &text).unwrap();
        assert!(kept.starts_with("line number 0\n"));
        assert!(kept.ends_with("line number 1999\n"));
        assert!(kept.contains("lines omitted"));
        assert!(kept.len() < 6_500, "{}", kept.len());
        for line in kept.lines().filter(|line| !line.starts_with("[ccc:")) {
            assert!(line.starts_with("line number "), "cut mid-line: {line:?}");
        }
    }

    /// Found by a live run: a test script not recognized as one lost its only failure, and
    /// the model couldn't name the failing test.
    #[test]
    fn a_failure_in_the_omitted_middle_is_kept() {
        let mut lines: Vec<String> = (0..600).map(|i| format!("test case_{i} ... ok")).collect();
        lines[300] = "test flaky_parser ... FAILED".to_owned();
        let text = lines.join("\n");

        let kept = trim(&Limits::default(), &text).unwrap();

        assert!(kept.contains("test flaky_parser ... FAILED"), "{kept}");
        assert!(
            kept.contains("test case_299 ... ok"),
            "context before: {kept}"
        );
        assert!(
            kept.contains("test case_301 ... ok"),
            "context after: {kept}"
        );
        assert!(!kept.contains("test case_250 ... ok"), "{kept}");
        assert!(kept.len() < text.len());
    }

    #[test]
    fn short_output_stays_whole() {
        assert_eq!(trim(&Limits::default(), "short"), None);
    }

    #[test]
    fn one_enormous_line_is_cut_without_splitting_a_character() {
        let text = "é".repeat(20_000);
        let kept = trim(&Limits::default(), &text).unwrap();
        assert!(kept.len() < text.len());
        assert!(kept.starts_with('é'));
        assert!(kept.ends_with('é'));
    }
}
