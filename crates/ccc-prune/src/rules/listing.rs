//! Search results and file listings: keep the first hits and say how many more exist.

use super::Limits;

/// Trailing lines kept so the end of a listing is visible too.
const TAIL_LINES: usize = 5;

/// Folds a long listing, or returns `None` if it is short enough to keep.
pub(super) fn trim(limits: &Limits, text: &str) -> Option<String> {
    if text.len() < limits.listing_min_bytes {
        return None;
    }
    let lines: Vec<&str> = text.lines().collect();
    if lines.len() <= limits.listing_keep_lines + TAIL_LINES {
        return None;
    }
    let tail_start = lines.len() - TAIL_LINES;
    let omitted = tail_start - limits.listing_keep_lines;
    let mut out = String::new();
    for line in &lines[..limits.listing_keep_lines] {
        out.push_str(line);
        out.push('\n');
    }
    out.push_str(&format!(
        "[ccc: {omitted} of {} lines omitted. Narrow the search to see them.]\n",
        lines.len()
    ));
    for line in &lines[tail_start..] {
        out.push_str(line);
        out.push('\n');
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn folds_long_listings_and_states_the_count() {
        let listing: String = (0..1000)
            .map(|i| format!("src/file_{i}.rs:1: hit\n"))
            .collect();
        let kept = trim(&Limits::default(), &listing).unwrap();
        assert!(kept.contains("file_0.rs"));
        assert!(kept.contains("file_49.rs"));
        assert!(!kept.contains("file_50.rs"));
        assert!(kept.contains("file_999.rs"));
        assert!(kept.contains("945 of 1000 lines omitted"), "{kept}");
    }

    #[test]
    fn few_long_lines_stay_whole() {
        let listing = format!("{}\n", "x".repeat(10_000));
        assert_eq!(trim(&Limits::default(), &listing), None);
    }
}
