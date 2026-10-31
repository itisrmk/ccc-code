//! Loads the optional local model stages.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context as _;
use ccc_pipeline::Stage;
use ccc_slm::{
    BlobPrune, BlobPruneConfig, Classify, LlamaConfig, LlamaModel, TextModel, reads_fast_enough,
};

use crate::ProxyArgs;

/// About 4K tokens at the rate fitted from real transcripts.
const PRUNE_MIN_BYTES: usize = 11_500;
/// About as much as fits in the model's context with room for its answer.
const PRUNE_MAX_BYTES: usize = 22_000;

/// The optional stages asked for on the command line.
pub(crate) struct LocalStages {
    pub(crate) classify: Option<Arc<dyn Stage>>,
    pub(crate) prune: Option<Arc<dyn Stage>>,
}

pub(crate) fn load(args: &ProxyArgs, state_dir: Option<&Path>) -> anyhow::Result<LocalStages> {
    // Both stages can share one model rather than holding the weights in memory twice.
    let mut loaded: HashMap<PathBuf, Arc<LlamaModel>> = HashMap::new();
    let mut model_at = |path: &PathBuf| -> anyhow::Result<Arc<dyn TextModel>> {
        let model = match loaded.get(path) {
            Some(model) => Arc::clone(model),
            None => {
                let model = Arc::new(
                    LlamaModel::load(path, LlamaConfig::default())
                        .with_context(|| format!("loading {}", path.display()))?,
                );
                loaded.insert(path.clone(), Arc::clone(&model));
                model
            }
        };
        Ok(model)
    };

    let classify = match &args.classify_model {
        Some(path) => {
            let cap = Duration::from_millis(args.classify_cap_ms);
            Some(Arc::new(Classify::new(model_at(path)?, cap)) as Arc<dyn Stage>)
        }
        None => None,
    };

    let prune = match &args.prune_model {
        Some(path) => {
            let model = model_at(path)?;
            let cap = Duration::from_millis(args.prune_model_cap_ms);
            let fast_enough = reads_fast_enough(model.as_ref(), PRUNE_MIN_BYTES, cap)
                .context("measuring the prune model")?;
            let dir = state_dir.context("--prune-model needs a state directory")?;
            if fast_enough {
                let config = BlobPruneConfig {
                    decisions_dir: dir.join("model-decisions"),
                    min_bytes: PRUNE_MIN_BYTES,
                    max_bytes: PRUNE_MAX_BYTES,
                    time_cap: cap,
                };
                Some(Arc::new(BlobPrune::new(model, config)) as Arc<dyn Stage>)
            } else {
                tracing::warn!(?cap, "the prune model is too slow for its cap, so it's off");
                None
            }
        }
        None => None,
    };
    Ok(LocalStages { classify, prune })
}
