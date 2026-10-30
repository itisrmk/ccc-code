//! `ccc bench`: replay local transcripts and print the results.

use std::io::Write;
use std::path::Path;

use anyhow::Context;

pub(crate) fn run(transcripts: &Path) -> anyhow::Result<()> {
    let (sessions, summary) = ccc_bench::load_dir(transcripts)
        .with_context(|| format!("loading transcripts from {}", transcripts.display()))?;
    let report = ccc_bench::replay(&sessions, summary, &[]);
    writeln!(std::io::stdout(), "{report}").context("writing report")
}
