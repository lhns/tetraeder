//! OpenAI and Anthropic API, tool calls, structured output, auth, quotas and routing (ADR 0028).
//!
//! [`engine::Engine`] is the contract with the inference engine; [`gateway::Gateway`] is the
//! API-neutral core; [`server::router`] serves both APIs over it.

pub mod anthropic;
pub mod chat;
pub mod constraint;
pub mod decode;
pub mod engine;
pub mod error;
pub mod fake;
pub mod gateway;
pub mod openai;
pub mod server;
pub mod template;
pub mod toolcall;
