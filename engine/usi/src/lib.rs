//! USI protocol parsing and engine-session control.

#![forbid(unsafe_code)]

#[cfg(all(feature = "handcrafted", feature = "pure-only"))]
compile_error!("handcrafted and pure-only are mutually exclusive");
#[cfg(not(any(feature = "handcrafted", feature = "pure-only")))]
compile_error!("select exactly one build class: handcrafted or pure-only");

mod parser;
#[cfg(feature = "pure-only")]
mod pure;
#[cfg(feature = "handcrafted")]
mod session;
#[cfg(feature = "pure-only")]
pub use pure::run_pure_stdio;

pub use parser::{GoParameters, UsiCommand, UsiParseError, parse_command};
#[cfg(feature = "handcrafted")]
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
