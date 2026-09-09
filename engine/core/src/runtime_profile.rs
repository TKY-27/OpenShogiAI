//! Versioned runtime profiles and proof counters for evidence-bearing play.

use serde::Serialize;

/// Stable wire name of the pure learned runtime profile.
pub const PURE_LEARNED_PROFILE_NAME: &str = "pure_learned";
/// Closed profile contract implemented by native and Wasm runtimes.
pub const PURE_LEARNED_PROFILE_SCHEMA: &str = "open_shogiai_pure_learned_profile/v1";
/// SHA-256 of `configs/runtime/pure_learned-v1.json`.
pub const PURE_LEARNED_PROFILE_SCHEMA_SHA256: &str =
    "291cccea2056bee039fe4c84185304b3681ad6361df2ca1ba821a80d3645698c";

/// Frozen a1 profile contract, distinct from legacy OSAVAL02.
pub const PHASE10T_PROFILE_SCHEMA: &str = "open_shogiai_pure_learned_a1_profile/v1";
/// SHA-256 of the committed a1 profile contract.
pub const PHASE10T_PROFILE_SCHEMA_SHA256: &str =
    "b8561b90d92d1c7d356e43db66888a5b525375fa8e76158849bbfb26fd42c6a9";

pub const PHASE10V_PROFILE_SCHEMA: &str = "open_shogiai_pure_learned_v3_profile/v1";
pub const PHASE10V_PROFILE_SCHEMA_SHA256: &str =
    "d2eec27887926ccc8a076552815cd54e34b85d6d23e65732ddba4989bf59c1e7";

/// Runtime policy selected before a game starts.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum RuntimeProfile {
    /// Existing unrestricted runtime behavior.
    #[default]
    Standard,
    /// Validated learned-only evaluation with every prohibited source disabled.
    PureLearned,
}

impl RuntimeProfile {
    /// Parse the closed external profile name.
    ///
    /// # Errors
    ///
    /// Returns an error naming the two accepted profiles for anything else.
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "standard" => Ok(Self::Standard),
            PURE_LEARNED_PROFILE_NAME => Ok(Self::PureLearned),
            _ => Err(format!(
                "runtime profile must be `standard` or `{PURE_LEARNED_PROFILE_NAME}`"
            )),
        }
    }

    /// Stable external name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Standard => "standard",
            Self::PureLearned => PURE_LEARNED_PROFILE_NAME,
        }
    }
}

/// Complete proof counters required for one pure learned search or game.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct RuntimeProofCounters {
    pub profile: &'static str,
    pub profile_schema: &'static str,
    pub learned_eval_calls: u64,
    pub accumulator_updates: u64,
    pub accumulator_refreshes: u64,
    pub handcrafted_eval_calls: u64,
    pub residual_eval_calls: u64,
    pub composite_eval_calls: u64,
    pub book_hits: u64,
    pub teacher_calls: u64,
    pub fallback_count: u64,
    pub model_sha256: String,
    pub evaluator_profile_schema_hash: &'static str,
}

impl RuntimeProofCounters {
    /// Whether the evidence satisfies the pure learned invariant.
    #[must_use]
    pub fn valid_pure_learned(&self) -> bool {
        self.valid_pure_runtime() && self.learned_eval_calls > 0
    }

    /// Validate measured evidence against the explicit result of one search.
    /// The legacy positive-inference proof stays strict. Rule-terminal and pre-evaluation
    /// interruption results instead require exact zero-work shapes and their declared reason.
    #[must_use]
    pub fn valid_pure_search(&self, result: &crate::SearchResult) -> bool {
        use crate::{MATE_SCORE, SearchOutcome, SearchTermination};

        if !self.valid_pure_runtime()
            || self.learned_eval_calls != result.stats.learned_eval_calls
            || self.handcrafted_eval_calls != result.stats.handcrafted_eval_calls
            || self.residual_eval_calls != result.stats.residual_eval_calls
            || self.composite_eval_calls != result.stats.composite_eval_calls
            || self.fallback_count != result.stats.fallback_count
            || result.termination == SearchTermination::EvaluationError
        {
            return false;
        }
        if result.outcome == SearchOutcome::Evaluated {
            return self.learned_eval_calls > 0 && result.best_move.is_some();
        }
        if self.learned_eval_calls != 0
            || result.stats.neural_inference_calls != 0
            || self.accumulator_updates != 0
            || self.accumulator_refreshes != 0
            || result.nodes != 0
            || result.depth != 0
            || result.seldepth != 0
            || !result.root_moves.is_empty()
        {
            return false;
        }
        if result.outcome.is_terminal() {
            return result.termination == SearchTermination::Completed
                && result.best_move.is_none()
                && result.pv.is_empty()
                && match result.outcome {
                    SearchOutcome::Checkmate | SearchOutcome::NoLegalMoves => {
                        result.score == -MATE_SCORE
                    }
                    SearchOutcome::Repetition => result.score == 0,
                    SearchOutcome::PerpetualCheck => {
                        result.score.unsigned_abs() == MATE_SCORE.unsigned_abs()
                    }
                    _ => false,
                };
        }
        result.best_move.is_some()
            && result.pv == result.best_move.into_iter().collect::<Vec<_>>()
            && result.score == 0
            && matches!(
                (result.outcome, result.termination),
                (
                    SearchOutcome::CancelledBeforeEvaluation,
                    SearchTermination::Cancelled
                ) | (
                    SearchOutcome::NodeLimitBeforeEvaluation,
                    SearchTermination::NodeLimit
                ) | (
                    SearchOutcome::TimeLimitBeforeEvaluation,
                    SearchTermination::TimeLimit
                )
            )
    }

    fn valid_pure_runtime(&self) -> bool {
        self.profile == PURE_LEARNED_PROFILE_NAME
            && ((self.profile_schema == PURE_LEARNED_PROFILE_SCHEMA
                && self.evaluator_profile_schema_hash == PURE_LEARNED_PROFILE_SCHEMA_SHA256)
                || (self.profile_schema == PHASE10T_PROFILE_SCHEMA
                    && self.evaluator_profile_schema_hash == PHASE10T_PROFILE_SCHEMA_SHA256)
                || (self.profile_schema == PHASE10V_PROFILE_SCHEMA
                    && self.evaluator_profile_schema_hash == PHASE10V_PROFILE_SCHEMA_SHA256))
            && self.handcrafted_eval_calls == 0
            && self.residual_eval_calls == 0
            && self.composite_eval_calls == 0
            && self.book_hits == 0
            && self.teacher_calls == 0
            && self.fallback_count == 0
            && self.model_sha256.len() == 64
            && self
                .model_sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    }
}

#[cfg(test)]
mod tests {
    use sha2::{Digest, Sha256};

    use super::*;

    #[test]
    fn pure_profile_requires_every_prohibited_counter_to_be_zero() {
        let mut proof = RuntimeProofCounters {
            profile: PURE_LEARNED_PROFILE_NAME,
            profile_schema: PURE_LEARNED_PROFILE_SCHEMA,
            learned_eval_calls: 1,
            accumulator_updates: 0,
            accumulator_refreshes: 0,
            handcrafted_eval_calls: 0,
            residual_eval_calls: 0,
            composite_eval_calls: 0,
            book_hits: 0,
            teacher_calls: 0,
            fallback_count: 0,
            model_sha256: "a".repeat(64),
            evaluator_profile_schema_hash: PURE_LEARNED_PROFILE_SCHEMA_SHA256,
        };
        assert!(proof.valid_pure_learned());
        proof.handcrafted_eval_calls = 1;
        assert!(!proof.valid_pure_learned());
        proof.handcrafted_eval_calls = 0;
        proof.fallback_count = 1;
        assert!(!proof.valid_pure_learned());
    }

    #[test]
    fn profile_name_is_closed_and_versioned() {
        assert_eq!(
            RuntimeProfile::parse("pure_learned"),
            Ok(RuntimeProfile::PureLearned)
        );
        assert!(RuntimeProfile::parse("pure-learned").is_err());
        assert_eq!(
            PURE_LEARNED_PROFILE_SCHEMA,
            "open_shogiai_pure_learned_profile/v1"
        );
    }

    #[test]
    fn compiled_profile_hash_matches_the_committed_contract() {
        let bytes = include_bytes!("../../../configs/runtime/pure_learned-v1.json");
        assert_eq!(
            format!("{:x}", Sha256::digest(bytes)),
            PURE_LEARNED_PROFILE_SCHEMA_SHA256
        );
        assert_eq!(
            format!(
                "{:x}",
                Sha256::digest(include_bytes!(
                    "../../../configs/runtime/pure_learned-v3.json"
                ))
            ),
            PHASE10V_PROFILE_SCHEMA_SHA256
        );
        let a1_bytes = include_bytes!("../../../configs/runtime/pure_learned-a1-v1.json");
        assert_eq!(
            format!("{:x}", Sha256::digest(a1_bytes)),
            PHASE10T_PROFILE_SCHEMA_SHA256
        );
    }
}
