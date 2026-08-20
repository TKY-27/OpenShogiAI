//! Independently implemented shogi rules and shared engine types.
//!
//! The core crate owns deterministic board state, legal move generation, notation, game
//! history, hashing, and traversal. It deliberately has no UI, network, Python, browser, or
//! third-party engine dependency.

#![deny(unsafe_code)]

mod evaluation;
mod game;
mod neural;
mod notation;
mod perft;
mod position;
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
mod types;

pub use evaluation::{EvaluationBreakdown, EvaluationConfig, evaluate, evaluate_breakdown};
pub use game::{
    EnteringKingDeclaration, EnteringKingRule, Game, GameEnd, GameRecord, ImpasseCondition,
    ImpasseOutcome, RepetitionOutcome, repetition_outcome_from_moves,
};
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
pub use perft::{PerftError, PerftResult, perft, perft_divide};
pub use position::{IllegalMove, Position, PositionError, Undo};
pub use search::{
    CancellationToken, MATE_SCORE, MATE_THRESHOLD, MateSearchResult, RandomMoveSelector,
    SearchConfig, SearchEngine, SearchInfo, SearchLimits, SearchResult, SearchStats,
    SearchTermination, is_mate_score,
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
