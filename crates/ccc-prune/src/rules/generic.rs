//! Other command output: keep both ends, since output usually starts with what ran and
//! ends with what happened.

use super::Limits;

/// Trims long output to its two ends, or returns `None` if it is short enough to keep.
pub(super) fn trim(limits: &Limits, text: &str) -> Option<String> {
    if text.len() < limits.generic_min_bytes {
        return None;
    }
    let head_end = line_end_near(text, limits.generic_head_bytes);
    let tail_start = line_start_near(text, text.len().saturating_sub(limits.generic_tail_bytes));
    if tail_start <= head_end {
        return None;
    }
    let omitted = &text[head_end..tail_start];
    Some(format!(
        "{}\n[ccc: {} bytes in {} lines omitted. Re-run with a filter to see them.]\n{}",
        text[..head_end].trim_end_matches('\n'),
        omitted.len(),
        omitted.lines().count(),
        &text[tail_start..],
    ))
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
        assert!(kept.contains("bytes in"));
        assert!(kept.len() < 6_500, "{}", kept.len());
        for line in kept.lines().filter(|line| !line.starts_with("[ccc:")) {
            assert!(line.starts_with("line number "), "cut mid-line: {line:?}");
        }
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
