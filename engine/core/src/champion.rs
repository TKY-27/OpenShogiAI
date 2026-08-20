//! Engine-wide champion identities and default evaluator semantics.

use crate::EvaluationConfig;

/// Stable identifier for the strongest evaluator supported by authoritative overall evidence.
pub const OVERALL_CHAMPION_ID: &str = "handcrafted-experimental";

/// Stable identifier for the latest model promoted only inside the neural lineage.
pub const NEURAL_LINEAGE_CHAMPION_ID: &str = "value-v0-f32-g1r4";

/// The two champion scopes must never be conflated by adapters or registries.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChampionScope {
    Overall,
    NeuralLineage,
}

/// Returns the built-in evaluator configuration for the authoritative overall champion.
#[must_use]
pub const fn overall_champion_evaluation() -> EvaluationConfig {
    EvaluationConfig::handcrafted_experimental()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn champion_scopes_have_distinct_identities() {
        assert_ne!(OVERALL_CHAMPION_ID, NEURAL_LINEAGE_CHAMPION_ID);
        assert_eq!(
            overall_champion_evaluation(),
            EvaluationConfig::handcrafted_experimental()
        );
    }
}
