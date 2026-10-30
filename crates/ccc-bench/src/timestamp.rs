//! Parses the fixed `YYYY-MM-DDTHH:MM:SS.mmmZ` timestamps transcripts use.

/// Converts a transcript timestamp to milliseconds since the Unix epoch.
pub(crate) fn parse_millis(text: &str) -> Option<i64> {
    let field = |range: std::ops::Range<usize>| text.get(range)?.parse::<i64>().ok();
    let (year, month, day) = (field(0..4)?, field(5..7)?, field(8..10)?);
    let (hour, minute, second) = (field(11..13)?, field(14..16)?, field(17..19)?);
    let millis = field(20..23).unwrap_or(0);
    let days = days_from_civil(year, month, day);
    Some(((days * 24 + hour) * 60 + minute) * 60_000 + second * 1000 + millis)
}

/// Days since 1970-01-01 in the proleptic Gregorian calendar (Howard Hinnant's algorithm).
fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let year = if month <= 2 { year - 1 } else { year };
    let era = year.div_euclid(400);
    let year_of_era = year.rem_euclid(400);
    let shifted_month = (month + 9) % 12;
    let day_of_year = (153 * shifted_month + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era * 146_097 + day_of_era - 719_468
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_known_instants() {
        assert_eq!(parse_millis("1970-01-01T00:00:00.000Z"), Some(0));
        assert_eq!(
            parse_millis("2026-10-30T12:00:00.250Z"),
            Some(1793361600250)
        );
    }

    #[test]
    fn crosses_leap_day() {
        let before = parse_millis("2028-02-28T00:00:00.000Z").unwrap();
        let after = parse_millis("2028-03-01T00:00:00.000Z").unwrap();
        assert_eq!(after - before, 2 * 24 * 3_600_000);
    }

    #[test]
    fn rejects_garbage() {
        assert_eq!(parse_millis("yesterday"), None);
    }
}
