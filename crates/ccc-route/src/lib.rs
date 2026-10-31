//! Classifies calls by shape, picks a model by rule, and manages budget notices.
//!
//! Prompt caches are per model, so moving a long conversation to a cheaper model re-writes
//! its whole context and usually costs more than it saves. Routing therefore only touches
//! calls with no meaningful cached prefix, unless a budget downshift is explicitly enabled.

mod classify;
mod notice;
mod route;

pub use classify::{CallKind, classify};
pub use notice::{NoticeInjector, StripNotices, budget_notice};
pub use route::{LOCAL_PREFIX, Route, RouteRules};
