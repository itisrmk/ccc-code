//! What each session cost: token usage, list prices and a local SQLite record.
//!
//! On a subscription the dollar figures are notional, what the same traffic would cost at
//! API list prices, but they still track how quickly a session uses up its limits.

mod price;
mod store;
mod usage;

pub use price::cost_usd;
pub use store::{Ledger, OpenLedgerError, SessionTotal};
pub use usage::Usage;
