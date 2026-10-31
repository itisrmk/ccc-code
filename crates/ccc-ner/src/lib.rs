//! Finds names of people and organizations with a GLiNER model on ONNX Runtime.
//!
//! GLiNER scores every short run of words against labels given at run time, so what counts
//! as a name is an input and not something baked into the model. It only reads and labels
//! text; nothing is generated.

mod model;
mod spans;

pub use model::{Entity, LoadNerError, NerError, NerModel};
pub use spans::{Span, Word, choose_spans, split_words};
