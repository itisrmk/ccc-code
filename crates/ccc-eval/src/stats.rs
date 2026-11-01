//! Comparing two setups task by task.
//!
//! Repetitions of one task aren't independent evidence about a setup, so differences are averaged
//! per task and each task counts once. The test makes no assumption about their distribution:
//! if a setup made no difference, each task's difference was as likely to have the other sign.

use std::collections::BTreeMap;

use crate::run::Record;

/// A comparison looks at three measures, so each is held to a third of the usual 5% error rate.
pub const SIGNIFICANCE: f64 = 0.05 / 3.0;
/// Enumerate every sign pattern up to this many differing tasks, and sample beyond it.
const EXACT_LIMIT: usize = 20;
const SAMPLED_FLIPS: u64 = 1 << 16;
const BOOTSTRAP_ROUNDS: usize = 10_000;
/// Fixed so a report comes out the same every time it's made.
const RANDOM_SEED: u64 = 0x9E37_79B9_7F4A_7C15;

/// What's being measured.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    /// Share of runs whose tests passed.
    PassRate,
    /// Dollars per run at list prices.
    Cost,
    /// Agent turns per run.
    Turns,
}

impl Metric {
    fn value(self, record: &Record) -> f64 {
        match self {
            Self::PassRate => f64::from(u8::from(record.passed)),
            Self::Cost => record.cost_usd,
            Self::Turns => record.turns as f64,
        }
    }

    const fn higher_is_better(self) -> bool {
        matches!(self, Self::PassRate)
    }

    /// The name for reports.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::PassRate => "pass rate",
            Self::Cost => "cost",
            Self::Turns => "turns",
        }
    }
}

/// What a paired test found.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PairedTest {
    /// The mean difference across tasks.
    pub mean: f64,
    /// A 95% bootstrap interval around it.
    pub interval: (f64, f64),
    /// How likely a difference this large would be if the setups were the same.
    pub p_value: f64,
    /// The smallest `p_value` these tasks could ever produce. Above [`SIGNIFICANCE`], nothing
    /// could have been detected, whatever the truth.
    pub smallest_p: f64,
}

/// Tests whether `differences`, one per task, are centred on zero.
#[must_use]
pub fn paired_test(differences: &[f64]) -> PairedTest {
    let count = differences.len().max(1) as f64;
    let mean = differences.iter().sum::<f64>() / count;
    let nonzero: Vec<f64> = differences
        .iter()
        .copied()
        .filter(|value| *value != 0.0)
        .collect();
    let smallest_p = if nonzero.is_empty() {
        1.0
    } else {
        (2.0 * 0.5_f64.powi(nonzero.len().min(1023) as i32)).min(1.0)
    };
    PairedTest {
        mean,
        interval: bootstrap_interval(differences),
        p_value: sign_flip_p(&nonzero),
        smallest_p,
    }
}

fn sign_flip_p(nonzero: &[f64]) -> f64 {
    if nonzero.is_empty() {
        return 1.0;
    }
    let observed: f64 = nonzero.iter().sum::<f64>().abs();
    // Rounding must not stop a pattern with an equal total from counting.
    let reaches = |total: f64| total.abs() >= observed - 1e-12;
    if nonzero.len() <= EXACT_LIMIT {
        let patterns = 1_u64 << nonzero.len();
        let hits = (0..patterns)
            .filter(|pattern| {
                let total: f64 = nonzero
                    .iter()
                    .enumerate()
                    .map(|(bit, value)| {
                        if pattern >> bit & 1 == 1 {
                            -value
                        } else {
                            *value
                        }
                    })
                    .sum();
                reaches(total)
            })
            .count();
        return hits as f64 / patterns as f64;
    }
    let mut random = Random(RANDOM_SEED);
    let hits = (0..SAMPLED_FLIPS)
        .filter(|_| {
            let total: f64 = nonzero
                .iter()
                .map(|value| {
                    if random.next() & 1 == 1 {
                        -value
                    } else {
                        *value
                    }
                })
                .sum();
            reaches(total)
        })
        .count();
    (hits + 1) as f64 / (SAMPLED_FLIPS + 1) as f64
}

fn bootstrap_interval(differences: &[f64]) -> (f64, f64) {
    if differences.is_empty() {
        return (0.0, 0.0);
    }
    let mut random = Random(RANDOM_SEED);
    let mut means: Vec<f64> = (0..BOOTSTRAP_ROUNDS)
        .map(|_| {
            let total: f64 = (0..differences.len())
                .map(|_| differences[random.next() as usize % differences.len()])
                .sum();
            total / differences.len() as f64
        })
        .collect();
    means.sort_by(f64::total_cmp);
    (
        means[BOOTSTRAP_ROUNDS * 25 / 1000],
        means[BOOTSTRAP_ROUNDS * 975 / 1000 - 1],
    )
}

/// A small generator, so reports are reproducible with no dependency.
struct Random(u64);

impl Random {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// What a test means for the setup being tried.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// Better than the baseline, beyond chance.
    Better,
    /// Worse than the baseline, beyond chance.
    Worse,
    /// No difference beyond chance. Not proof that there's none.
    NotDetected,
    /// These tasks couldn't have shown a difference either way.
    TooFewDiffering,
}

/// One measure compared.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MetricResult {
    /// What's measured.
    pub metric: Metric,
    /// The baseline's mean.
    pub baseline_mean: f64,
    /// The other setup's mean.
    pub other_mean: f64,
    /// The test of their difference, other minus baseline.
    pub test: PairedTest,
    /// What it means.
    pub verdict: Verdict,
}

/// One setup against the baseline.
#[derive(Debug, Clone, PartialEq)]
pub struct Comparison {
    /// The baseline's name.
    pub baseline: String,
    /// The other setup's name.
    pub other: String,
    /// Tasks both setups ran.
    pub tasks: usize,
    /// Runs paired across both setups.
    pub pairs: usize,
    /// Runs under the other setup that changed a test file.
    pub tests_touched: usize,
    /// Runs under the other setup that stopped on an error.
    pub errors: usize,
    /// Each measure.
    pub metrics: Vec<MetricResult>,
}

fn verdict(metric: Metric, test: &PairedTest) -> Verdict {
    if test.smallest_p >= SIGNIFICANCE && test.p_value >= SIGNIFICANCE {
        return Verdict::TooFewDiffering;
    }
    if test.p_value >= SIGNIFICANCE {
        return Verdict::NotDetected;
    }
    if (test.mean > 0.0) == metric.higher_is_better() {
        Verdict::Better
    } else {
        Verdict::Worse
    }
}

/// Compares `other` to `baseline`, using only repetitions both completed.
#[must_use]
pub fn compare(records: &[Record], baseline: &str, other: &str) -> Comparison {
    type Key<'a> = (&'a str, usize);
    let index = |setup: &str| -> BTreeMap<Key<'_>, &Record> {
        records
            .iter()
            .filter(|record| record.setup == setup)
            .map(|record| ((record.task.as_str(), record.repetition), record))
            .collect()
    };
    let (first, second) = (index(baseline), index(other));
    let mut by_task: BTreeMap<&str, Vec<(&Record, &Record)>> = BTreeMap::new();
    for (key, record) in &first {
        if let Some(paired) = second.get(key) {
            by_task.entry(key.0).or_default().push((record, paired));
        }
    }
    let mean = |values: &[f64]| values.iter().sum::<f64>() / values.len().max(1) as f64;
    let metrics = [Metric::PassRate, Metric::Cost, Metric::Turns]
        .into_iter()
        .map(|metric| {
            let per_task: Vec<(f64, f64)> = by_task
                .values()
                .map(|pairs| {
                    let a: Vec<f64> = pairs.iter().map(|(a, _)| metric.value(a)).collect();
                    let b: Vec<f64> = pairs.iter().map(|(_, b)| metric.value(b)).collect();
                    (mean(&a), mean(&b))
                })
                .collect();
            let differences: Vec<f64> = per_task.iter().map(|(a, b)| b - a).collect();
            let test = paired_test(&differences);
            MetricResult {
                metric,
                baseline_mean: mean(&per_task.iter().map(|pair| pair.0).collect::<Vec<_>>()),
                other_mean: mean(&per_task.iter().map(|pair| pair.1).collect::<Vec<_>>()),
                test,
                verdict: verdict(metric, &test),
            }
        })
        .collect();
    let paired: Vec<&(&Record, &Record)> = by_task.values().flatten().collect();
    Comparison {
        baseline: baseline.to_owned(),
        other: other.to_owned(),
        tasks: by_task.len(),
        pairs: paired.len(),
        tests_touched: paired.iter().filter(|(_, b)| b.tests_touched).count(),
        errors: paired.iter().filter(|(_, b)| b.error.is_some()).count(),
        metrics,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(task: &str, setup: &str, repetition: usize, passed: bool, cost: f64) -> Record {
        Record {
            task: task.to_owned(),
            setup: setup.to_owned(),
            repetition,
            model: "m".to_owned(),
            passed,
            tests_touched: false,
            cost_usd: cost,
            turns: 3,
            seconds: 1.0,
            error: None,
        }
    }

    #[test]
    fn the_exact_test_matches_a_hand_count() {
        // All 8 same-sign tasks: only the two all-same patterns reach the total. 2 of 256.
        let result = paired_test(&[1.0; 8]);
        assert!((result.p_value - 2.0 / 256.0).abs() < 1e-12);
        assert!((result.smallest_p - 2.0 / 256.0).abs() < 1e-12);
        assert!((result.mean - 1.0).abs() < 1e-12);

        // 1, 1, -1 sums to 1. Patterns reaching |total| >= 1 are all but none: 8 of 8.
        let mixed = paired_test(&[1.0, 1.0, -1.0]);
        assert!((mixed.p_value - 1.0).abs() < 1e-12);
    }

    #[test]
    fn no_differences_at_all_mean_nothing_to_find() {
        let result = paired_test(&[0.0; 10]);
        assert_eq!(result.p_value, 1.0);
        assert_eq!(result.smallest_p, 1.0);
        assert_eq!(result.interval, (0.0, 0.0));
    }

    #[test]
    fn many_tasks_use_the_sampled_test_and_agree_with_the_exact_one() {
        let mut values = vec![1.0; 25];
        values.extend([-1.0; 5]);
        let result = paired_test(&values);
        // 25 of 30 going one way: a two-sided binomial gives about 0.0003.
        assert!(result.p_value < 0.002, "{result:?}");

        let balanced: Vec<f64> = (0..30)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        assert!(paired_test(&balanced).p_value > 0.9);
    }

    #[test]
    fn the_interval_brackets_the_mean() {
        let result = paired_test(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.2, 0.3, 0.1]);
        assert!(result.interval.0 <= result.mean && result.mean <= result.interval.1);
        assert!(result.interval.0 > 0.0);
    }

    #[test]
    fn a_clearly_worse_setup_is_called_worse_and_a_cheaper_one_better() {
        let mut records = Vec::new();
        for index in 0..8 {
            let task = format!("t{index}");
            records.push(record(&task, "plain", 0, true, 0.20 + index as f64 / 100.0));
            records.push(record(&task, "bad", 0, false, 0.50 + index as f64 / 100.0));
            records.push(record(&task, "lean", 0, true, 0.10 + index as f64 / 100.0));
        }
        let bad = compare(&records, "plain", "bad");
        assert_eq!((bad.tasks, bad.pairs), (8, 8));
        assert_eq!(bad.metrics[0].verdict, Verdict::Worse);
        assert_eq!(bad.metrics[1].verdict, Verdict::Worse);
        assert_eq!(bad.metrics[0].baseline_mean, 1.0);
        assert_eq!(bad.metrics[0].other_mean, 0.0);

        let lean = compare(&records, "plain", "lean");
        assert_eq!(lean.metrics[1].verdict, Verdict::Better);
        assert_eq!(lean.metrics[0].verdict, Verdict::TooFewDiffering);
    }

    #[test]
    fn too_few_tasks_are_reported_as_unable_to_tell() {
        let mut records = Vec::new();
        for index in 0..4 {
            let task = format!("t{index}");
            records.push(record(&task, "a", 0, true, 0.2));
            records.push(record(&task, "b", 0, false, 0.5));
        }
        let result = compare(&records, "a", "b");
        assert_eq!(result.metrics[0].verdict, Verdict::TooFewDiffering);
    }

    #[test]
    fn repetitions_of_a_task_count_as_one_task() {
        let mut records = Vec::new();
        for repetition in 0..50 {
            records.push(record("only", "a", repetition, true, 0.2));
            records.push(record("only", "b", repetition, false, 0.5));
        }
        let result = compare(&records, "a", "b");
        assert_eq!(result.tasks, 1);
        assert_eq!(result.pairs, 50);
        assert_eq!(result.metrics[0].verdict, Verdict::TooFewDiffering);
    }

    #[test]
    fn unpaired_runs_are_left_out() {
        let records = vec![
            record("t", "a", 0, true, 0.2),
            record("t", "b", 0, true, 0.2),
            record("t", "a", 1, false, 9.0),
        ];
        let result = compare(&records, "a", "b");
        assert_eq!(result.pairs, 1);
        assert_eq!(result.metrics[1].baseline_mean, 0.2);
    }
}
