//! OpenAI and Anthropic API, tool calls, structured output, auth, quotas and routing (ADR 0028).
//!
//! [`engine::Engine`] is the contract with the inference engine; [`gateway::Gateway`] is the
//! API-neutral core; [`server::router`] serves both APIs over it.

pub mod anthropic;
pub mod chat;
pub mod constraint;
pub mod decode;
pub mod embeddings;
pub mod engine;
pub mod error;
pub mod fake;
pub mod gateway;
pub mod native;
pub mod openai;
pub mod reqlog;
pub mod responses;
pub mod sampling;
pub mod server;
pub mod settings;
pub mod slots;
pub mod template;
pub mod toolcall;
