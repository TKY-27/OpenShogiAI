//! Compile-time separated browser runtimes.
#![forbid(unsafe_code)]

#[cfg(all(feature = "handcrafted", feature = "pure-only"))]
compile_error!("handcrafted and pure-only are mutually exclusive");
#[cfg(not(any(feature = "handcrafted", feature = "pure-only")))]
compile_error!("select exactly one build class: handcrafted or pure-only");
#[cfg(feature = "handcrafted")]
mod full;
#[cfg(feature = "handcrafted")]
pub use full::*;
#[cfg(feature = "pure-only")]
mod pure;
#[cfg(feature = "pure-only")]
pub use pure::*;

#[cfg(feature = "pure-only")]
mod pure_browser;
#[cfg(feature = "pure-only")]
pub use pure_browser::*;
