//! USI protocol parsing and engine-session control.

#![forbid(unsafe_code)]

mod parser;
mod session;

pub use parser::{GoParameters, UsiCommand, UsiParseError, parse_command};
pub use session::{ModelKind, ProtocolSink, UsiOptions, UsiSession, run_stdio};

use open_shogi_core::EngineIdentity;

/// Name of the protocol implemented by this adapter.
pub const PROTOCOL_NAME: &str = "USI";

/// Returns the USI identity line for the current build.
#[must_use]
pub fn engine_id_line() -> String {
    let identity = EngineIdentity::current();
    format!("id name {} {}", identity.name, identity.version)
}
