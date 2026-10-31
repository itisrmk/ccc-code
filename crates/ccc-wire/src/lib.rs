//! A model of Anthropic Messages API requests that round-trips byte for byte.
//!
//! Claude Code sends roles and block types the public API docs don't list, and the
//! prompt cache keys on exact bytes. So a [`Request`] stays a JSON tree that keeps key
//! order, number spelling and every unknown field, with accessors for what stages need.

mod request;
pub mod sse;

pub use request::{ParseRequestError, Request, text_of_tool_result};
