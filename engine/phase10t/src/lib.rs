#![deny(unsafe_code)]

// Diagnostics and production deliberately share one inference implementation.
pub use open_shogi_core::{
    PHASE10T_ACCUMULATOR_WIDTH, PHASE10T_FEATURE_COUNT, PHASE10T_HEAD_COUNT, PHASE10T_HIDDEN_WIDTH,
    PHASE10T_MODEL_MAGIC, Phase10TError, Phase10TEvaluator, Phase10TIdentity, Phase10TInference,
};
