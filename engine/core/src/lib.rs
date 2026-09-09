//! Independently implemented shogi rules and shared engine types.
//!
//! The core crate owns deterministic board state, legal move generation, notation, game
//! history, hashing, and traversal. It deliberately has no UI, network, Python, browser, or
//! third-party engine dependency.

#![deny(unsafe_code)]

#[cfg(all(feature = "handcrafted", feature = "pure-only"))]
compile_error!("handcrafted and pure-only are mutually exclusive");
#[cfg(not(any(feature = "handcrafted", feature = "pure-only")))]
compile_error!("select exactly one of handcrafted or pure-only");

mod analysis;
mod computation;
pub use computation::{
    COMPUTATION_FEATURES, COMPUTATION_SCHEMA, ComputationModel, ComputeControlSummary,
    computation_features,
};
#[cfg(feature = "handcrafted")]
mod champion;
#[cfg(feature = "handcrafted")]
mod evaluation;
mod game;
#[cfg(feature = "handcrafted")]
mod neural;
mod notation;
#[cfg(feature = "handcrafted")]
mod opening;
mod perft;
mod phase10r;
mod phase10t;
mod phase10v;
pub use phase10v::{
    PHASE10V_ACCUMULATOR_WIDTH, PHASE10V_FEATURE_COUNT, PHASE10V_HEAD_COUNT, PHASE10V_HIDDEN_WIDTH,
    PHASE10V_MODEL_MAGIC, Phase10VAccumulator, Phase10VError, Phase10VEvaluator, Phase10VIdentity,
    Phase10VInference,
};
mod pure_playing;
pub use pure_playing::PurePlayingEvaluator;
mod position;
mod resource;
mod runtime_profile;
mod search;
#[cfg(any(
    target_os = "macos",
    target_os = "ios",
    target_os = "tvos",
    target_os = "watchos",
    target_os = "visionos",
    target_os = "linux",
    target_os = "android"
))]
#[allow(unsafe_code)]
mod secure_file;
#[cfg(not(any(
    target_os = "macos",
    target_os = "ios",
    target_os = "tvos",
    target_os = "watchos",
    target_os = "visionos",
    target_os = "linux",
    target_os = "android"
)))]
mod secure_file_portable;
mod time_control;
mod types;

pub use analysis::{
    ANALYSIS_SCHEMA, AnalysisCacheEntry, AnalysisCacheKey, AnalysisLine, AnalysisService,
    AnalysisState, AnalysisStep, AnalysisUpdate, AnalysisUpdateSource,
    DEFAULT_ANALYSIS_CACHE_ENTRIES, MAX_ANALYSIS_MULTI_PV,
};
#[cfg(feature = "handcrafted")]
pub use champion::{
    ChampionScope, NEURAL_LINEAGE_CHAMPION_ID, OVERALL_CHAMPION_ID, overall_champion_evaluation,
};
#[cfg(feature = "handcrafted")]
pub use evaluation::{EvaluationBreakdown, EvaluationConfig, evaluate, evaluate_breakdown};
pub use game::{
    EnteringKingDeclaration, EnteringKingRule, Game, GameEnd, GameRecord, ImpasseCondition,
    ImpasseOutcome, RepetitionOutcome, repetition_outcome_from_moves,
};
#[cfg(feature = "handcrafted")]
pub use neural::{
    FEATURE_ATTACK_MAPS, FEATURE_BOARD_PIECES, FEATURE_HAND_COUNTS, FEATURE_KING_COORDINATES,
    FEATURE_SIDE_TO_MOVE, MAX_NEURAL_MODEL_BYTES, MAX_NEURAL_SCORE_CP, NeuralActivation,
    NeuralEvaluator, NeuralModelError, NeuralModelIdentity, NeuralQuantization,
    encode_neural_features, neural_feature_dimension,
};
pub use notation::{
    CsaGame, CsaParseError, CsaResultValidation, CsaSpecialMove, NotationError, parse_csa_game,
    parse_csa_move, parse_sfen, parse_usi_move, to_csa_game, to_csa_move, to_sfen, to_usi_move,
};
#[cfg(feature = "handcrafted")]
pub use opening::{
    OPENING_BOOK_SCHEMA, OpeningBookChoice, OpeningBookV2, OpeningPolicy, OpeningProfile,
};
pub use perft::{PerftError, PerftResult, perft, perft_divide};
pub use phase10r::{
    MAX_OSAVAL02_MODEL_BYTES, Osaval02Error, Osaval02Evaluator, Osaval02History, Osaval02Identity,
    Osaval02Inference, Osaval02Quantization, Osaval02SearchAdapter, Osaval02Variant,
    encode_osaval02_move,
};
pub use phase10t::{
    PHASE10T_ACCUMULATOR_WIDTH, PHASE10T_FEATURE_COUNT, PHASE10T_HEAD_COUNT, PHASE10T_HIDDEN_WIDTH,
    PHASE10T_MODEL_MAGIC, Phase10TAccumulator, Phase10TError, Phase10TEvaluator, Phase10TIdentity,
    Phase10TInference,
};
pub use position::{IllegalMove, Position, PositionError, Undo};
pub use resource::{RESOURCE_BUDGET_SCHEMA, ResourceBudget, ResourceCoordinator};
pub use runtime_profile::{
    PHASE10T_PROFILE_SCHEMA, PHASE10T_PROFILE_SCHEMA_SHA256, PHASE10V_PROFILE_SCHEMA,
    PHASE10V_PROFILE_SCHEMA_SHA256, PURE_LEARNED_PROFILE_NAME, PURE_LEARNED_PROFILE_SCHEMA,
    PURE_LEARNED_PROFILE_SCHEMA_SHA256, RuntimeProfile, RuntimeProofCounters,
};
pub use search::{
    CancellationToken, MATE_SCORE, MATE_THRESHOLD, MateSearchResult, MonotonicClock,
    RandomMoveSelector, RootMoveStat, SearchConfig, SearchEngine, SearchInfo, SearchLimits,
    SearchOutcome, SearchResult, SearchStats, SearchTermination, SystemMonotonicClock,
    is_mate_score,
};
#[doc(hidden)]
#[cfg(any(
    target_os = "macos",
    target_os = "ios",
    target_os = "tvos",
    target_os = "watchos",
    target_os = "visionos",
    target_os = "linux",
    target_os = "android"
))]
pub use secure_file::{
    AnchoredDir, AnchoredFile, DirectoryEntry, EntryKind, MAX_SECURE_CLEANUP_QUARANTINE_ENTRIES,
    StableDirectoryIdentity, StableFileIdentity,
};
#[doc(hidden)]
#[cfg(not(any(
    target_os = "macos",
    target_os = "ios",
    target_os = "tvos",
    target_os = "watchos",
    target_os = "visionos",
    target_os = "linux",
    target_os = "android"
)))]
pub use secure_file_portable::{AnchoredFile, StableFileIdentity};
pub use time_control::{
    CASUAL_HARD_MAX_MS, MAX_CLOCK_MS, MAX_MOVE_TIME_MS, MAX_SAFETY_MARGIN_MS,
    MAX_TIME_CONTROL_DEPTH, MAX_TIME_CONTROL_NODES, StabilityPolicy, TIME_CONTROL_SCHEMA,
    TimeControl, TimeControlMode, TimeManager, TimeManagerConfig, TimePlan,
};
pub use types::{
    BOARD_SQUARES, HAND_KIND_COUNT, Hand, HandPiece, Move, Piece, PieceKind, Side, Square,
};

/// Human-readable engine name used by protocol and UI adapters.
pub const ENGINE_NAME: &str = "OpenShogiAI";

/// Current source package version.
pub const ENGINE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Metadata exposed to adapters without coupling the core to a presentation layer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EngineIdentity {
    /// Stable human-readable project name.
    pub name: &'static str,
    /// Semantic-version package version compiled into this build.
    pub version: &'static str,
}

impl EngineIdentity {
    /// Returns the identity of the current engine build.
    #[must_use]
    pub const fn current() -> Self {
        Self {
            name: ENGINE_NAME,
            version: ENGINE_VERSION,
        }
    }
}

/// A seed with a stable byte representation for experiment manifests.
///
/// This small foundation type prevents host endianness from leaking into future recorded
/// experiment and self-play seeds.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ReproducibilitySeed(u64);

impl ReproducibilitySeed {
    /// Wraps an integer seed without modifying it.
    #[must_use]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Returns the original integer seed.
    #[must_use]
    pub const fn value(self) -> u64 {
        self.0
    }

    /// Encodes the seed in the canonical little-endian manifest representation.
    #[must_use]
    pub const fn to_le_bytes(self) -> [u8; 8] {
        self.0.to_le_bytes()
    }

    /// Decodes a seed from the canonical little-endian manifest representation.
    #[must_use]
    pub const fn from_le_bytes(bytes: [u8; 8]) -> Self {
        Self(u64::from_le_bytes(bytes))
    }
}

#[cfg(feature = "handcrafted")]
pub use search::NeuralEvaluationMode;

/// Evaluator implementations compiled into this core build.
pub const COMPILED_EVALUATORS: &[&str] = if cfg!(feature = "handcrafted") {
    &[
        "handcrafted",
        "neural",
        "residual",
        "composite",
        "osaval02",
        "phase10t-a1",
        "phase10v",
    ]
} else {
    &["osaval02", "phase10t-a1", "phase10v"]
};

#[cfg(test)]
mod tests {
    use super::{ENGINE_NAME, EngineIdentity, Position};

    #[test]
    fn current_identity_is_nonempty() {
        let identity = EngineIdentity::current();

        assert_eq!(identity.name, ENGINE_NAME);
        assert!(!identity.name.is_empty());
        assert!(!identity.version.is_empty());
    }

    #[test]
    fn start_position_is_available() {
        let position = Position::startpos();
        assert_eq!(position.legal_moves().len(), 30);
    }
}
