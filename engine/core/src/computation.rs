//! Learned next-iteration decision risk, shared by native and Wasm search.
//!
//! Scores are noisy search observations, including alpha-beta bounds. Predictions only
//! change root ordering and optional time spending; they never prune a legal candidate,
//! replace a leaf score, certify mate, or increase the independently allocated hard limit.
use crate::{Move, Position, RootMoveStat, SearchInfo, TimePlan};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const COMPUTATION_SCHEMA: &str = "open_shogiai_computation/v1";
pub const COMPUTATION_FEATURES: [&str; 10] = [
    "incumbent",
    "gap_cp_1000",
    "absolute_cp_1000",
    "delta_cp_1000",
    "node_share",
    "depth_8",
    "log_candidates",
    "in_check",
    "prior_incumbent",
    "root_volatility_cp_1000",
];

/// A small logistic predictor trained on disjoint source games.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Artifact {
    schema: String,
    leaf_model_sha256: String,
    features: [String; 10],
    mean: [f64; 10],
    scale: [f64; 10],
    weights: [f64; 10],
    bias: f64,
    reference_risk: f64,
    training_positions: u32,
    updates: u32,
}

/// Immutable, hash-bound computation policy. No built-in learned weights exist.
#[derive(Clone, Debug)]
pub struct ComputationModel {
    artifact: Artifact,
    sha256: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ComputeControlSummary {
    pub model_sha256: String,
    pub enabled: bool,
    pub decisions: u64,
    pub predicted_risk: f64,
    pub target_ms: f64,
    pub reordered_moves: u64,
}

impl ComputationModel {
    /// Validate schema, numeric bounds, training evidence and both artifact identities.
    /// # Errors
    /// Rejects mismatches and malformed artifacts; never retries another model.
    pub fn from_bytes(bytes: &[u8], expected: &str, leaf: &str) -> Result<Self, String> {
        if bytes.len() > 16_384 || bytes.is_empty() {
            return Err("computation model must contain 1..=16384 bytes".into());
        }
        let sha256 = format!("{:x}", Sha256::digest(bytes));
        if expected != sha256 {
            return Err("computation model SHA-256 mismatch".into());
        }
        let artifact: Artifact = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
        if artifact.schema != COMPUTATION_SCHEMA
            || artifact.leaf_model_sha256 != leaf
            || leaf.len() != 64
            || !leaf
                .bytes()
                .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
            || artifact
                .features
                .iter()
                .map(String::as_str)
                .ne(COMPUTATION_FEATURES)
            || artifact.training_positions == 0
            || artifact.updates == 0
            || artifact.updates > 10_000
            || !artifact.bias.is_finite()
            || artifact.bias.abs() > 100.0
            || !(0.01..=0.99).contains(&artifact.reference_risk)
            || artifact
                .mean
                .iter()
                .any(|x| !x.is_finite() || x.abs() > 100.0)
            || artifact
                .scale
                .iter()
                .any(|x| !x.is_finite() || !(0.001..=100.0).contains(x))
            || artifact
                .weights
                .iter()
                .any(|x| !x.is_finite() || x.abs() > 100.0)
        {
            return Err(
                "invalid computation schema, numeric range, leaf identity or training evidence"
                    .into(),
            );
        }
        Ok(Self { artifact, sha256 })
    }

    #[must_use]
    pub fn sha256(&self) -> &str {
        &self.sha256
    }

    #[must_use]
    pub fn leaf_sha256(&self) -> &str {
        &self.artifact.leaf_model_sha256
    }

    /// Probability of the candidate's decision status changing at the next completed depth.
    #[must_use]
    pub fn predict(&self, features: [f64; 10]) -> f64 {
        let z = features
            .iter()
            .enumerate()
            .fold(self.artifact.bias, |sum, (i, x)| {
                sum + self.artifact.weights[i]
                    * ((x - self.artifact.mean[i]) / self.artifact.scale[i]).clamp(-8.0, 8.0)
            });
        1.0 / (1.0 + (-z.clamp(-30.0, 30.0)).exp())
    }

    /// Candidate priorities and the incumbent's risk use the same learned predictor.
    #[must_use]
    pub fn assess(
        &self,
        position: &Position,
        info: &SearchInfo,
        previous: Option<&SearchInfo>,
    ) -> (f64, Vec<(Move, f64)>) {
        let mut risk = self.artifact.reference_risk;
        let mut priorities = Vec::new();
        for candidate in &info.root_moves {
            let features = computation_features(position, info, previous, candidate);
            let probability = self.predict(features);
            if Some(candidate.movement) == info.best_move {
                risk = probability;
            }
            // Keep the proven incumbent first to establish alpha. Challenger ordering is
            // expected decision correction per approximate search cost, with no pruning.
            let priority = if Some(candidate.movement) == info.best_move {
                10.0
            } else {
                probability / (0.05 + features[4]).sqrt()
            };
            priorities.push((candidate.movement, priority));
        }
        priorities.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        (risk, priorities)
    }

    /// Trade present uncertainty against the future per-move clock share. This target is
    /// advisory and always bounded by the pre-existing independent hard deadline.
    #[must_use]
    pub fn target_ms(&self, risk: f64, plan: TimePlan) -> Option<f64> {
        let soft = plan.soft_limit?.as_secs_f64() * 1000.0;
        let hard = plan.hard_limit?.as_secs_f64() * 1000.0;
        Some((soft * (risk / self.artifact.reference_risk).clamp(0.25, 2.5)).min(hard))
    }
}

/// Feature values before train-only standardization. Evaluation scores remain in the
/// root side-to-move perspective. Mate-coded scores are bounded observations, not labels.
#[must_use]
pub fn computation_features(
    position: &Position,
    info: &SearchInfo,
    previous: Option<&SearchInfo>,
    candidate: &RootMoveStat,
) -> [f64; 10] {
    let old = previous.and_then(|p| {
        p.root_moves
            .iter()
            .find(|r| r.movement == candidate.movement)
    });
    let total: f64 = info.root_moves.iter().map(|r| count_as_f64(r.nodes)).sum();
    [
        f64::from(Some(candidate.movement) == info.best_move),
        f64::from(
            info.score
                .saturating_sub(candidate.score)
                .clamp(-20_000, 20_000),
        ) / 1000.0,
        f64::from(candidate.score.clamp(-20_000, 20_000).abs()) / 1000.0,
        old.map_or(0.0, |o| {
            f64::from(
                candidate
                    .score
                    .saturating_sub(o.score)
                    .clamp(-20_000, 20_000),
            ) / 1000.0
        }),
        count_as_f64(candidate.nodes) / total.max(1.0),
        f64::from(info.depth) / 8.0,
        f64::from(u32::try_from(info.root_moves.len()).unwrap_or(u32::MAX)).ln_1p(),
        f64::from(position.is_in_check(position.side_to_move())),
        f64::from(previous.is_some_and(|p| p.best_move == Some(candidate.movement))),
        previous.map_or(0.0, |p| {
            f64::from(
                info.score
                    .saturating_sub(p.score)
                    .clamp(-20_000, 20_000)
                    .abs(),
            ) / 1000.0
        }),
    ]
}

fn count_as_f64(value: u64) -> f64 {
    f64::from(u32::try_from(value).unwrap_or(u32::MAX))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn artifact() -> Vec<u8> {
        serde_json::to_vec(&serde_json::json!({
            "schema": COMPUTATION_SCHEMA, "leaf_model_sha256": "a".repeat(64),
            "features": COMPUTATION_FEATURES, "mean": vec![0.0;10], "scale": vec![1.0;10],
            "weights": vec![0.1;10], "bias": -1.0, "reference_risk": 0.3,
            "training_positions": 20, "updates": 1
        }))
        .unwrap()
    }
    #[test]
    fn artifacts_bind_leaf_and_reject_untrained_or_nonfinite_configuration() {
        let bytes = artifact();
        let hash = format!("{:x}", Sha256::digest(&bytes));
        assert!(ComputationModel::from_bytes(&bytes, &hash, &"a".repeat(64)).is_ok());
        assert!(ComputationModel::from_bytes(&bytes, &hash, &"b".repeat(64)).is_err());
        assert!(ComputationModel::from_bytes(&bytes, &"0".repeat(64), &"a".repeat(64)).is_err());
        let mut value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        for (key, bad) in [
            ("updates", serde_json::json!(0)),
            ("scale", serde_json::json!(vec![0.0; 10])),
        ] {
            let original = value[key].clone();
            value[key] = bad;
            let bad_bytes = serde_json::to_vec(&value).unwrap();
            let bad_hash = format!("{:x}", Sha256::digest(&bad_bytes));
            assert!(ComputationModel::from_bytes(&bad_bytes, &bad_hash, &"a".repeat(64)).is_err());
            value[key] = original;
        }
    }
    #[test]
    fn learned_targets_vary_with_risk_but_never_exceed_clock_ceiling() {
        let bytes = artifact();
        let hash = format!("{:x}", Sha256::digest(&bytes));
        let model = ComputationModel::from_bytes(&bytes, &hash, &"a".repeat(64)).unwrap();
        let plan = crate::TimeManager::default()
            .plan(
                crate::Side::Black,
                crate::TimeControl {
                    black_time_ms: Some(180_000),
                    white_time_ms: Some(180_000),
                    casual: false,
                    ..crate::TimeControl::casual()
                },
                64,
            )
            .unwrap();
        let low = model.target_ms(0.01, plan).unwrap();
        let high = model.target_ms(0.9, plan).unwrap();
        assert!(low < high);
        assert!(high <= plan.hard_limit.unwrap().as_secs_f64() * 1000.0);
    }
}
