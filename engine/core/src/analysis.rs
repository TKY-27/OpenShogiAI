//! Persistent analysis state, compatible-cache identity, and renewed iterative deepening.

use std::collections::BTreeMap;

use crate::{
    CancellationToken, ENGINE_VERSION, Position, RootMoveStat, SearchEngine, SearchInfo,
    SearchResult, TimePlan, is_mate_score, to_sfen,
};

/// Protocol/cache schema implemented by the core analysis service.
pub const ANALYSIS_SCHEMA: &str = "open_shogi_analysis/v1";
/// Defensive maximum number of principal variations retained per position.
pub const MAX_ANALYSIS_MULTI_PV: u8 = 10;
/// Default bounded in-memory cache size.
pub const DEFAULT_ANALYSIS_CACHE_ENTRIES: usize = 1_024;

/// Full compatibility identity for cached analysis.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct AnalysisCacheKey {
    pub canonical_position: String,
    pub position_hash: u64,
    pub model_hash: String,
    pub evaluator_config_hash: String,
    pub feature_schema_hash: String,
    pub evaluation_semantics_hash: String,
    pub search_options_hash: String,
    pub opening_profile_hash: String,
    pub multi_pv: u8,
}

impl AnalysisCacheKey {
    /// Builds and validates a key from a canonical engine position and immutable identities.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid hashes or a `MultiPV` value outside the protocol bounds.
    #[expect(
        clippy::too_many_arguments,
        reason = "the cache key deliberately exposes every independent compatibility identity"
    )]
    pub fn new(
        position: &Position,
        model_hash: impl Into<String>,
        evaluator_config_hash: impl Into<String>,
        feature_schema_hash: impl Into<String>,
        evaluation_semantics_hash: impl Into<String>,
        search_options_hash: impl Into<String>,
        opening_profile_hash: impl Into<String>,
        multi_pv: u8,
    ) -> Result<Self, String> {
        if !(1..=MAX_ANALYSIS_MULTI_PV).contains(&multi_pv) {
            return Err(format!("multiPV must be 1..={MAX_ANALYSIS_MULTI_PV}"));
        }
        let key = Self {
            canonical_position: to_sfen(position),
            position_hash: position.zobrist_hash(),
            model_hash: model_hash.into(),
            evaluator_config_hash: evaluator_config_hash.into(),
            feature_schema_hash: feature_schema_hash.into(),
            evaluation_semantics_hash: evaluation_semantics_hash.into(),
            search_options_hash: search_options_hash.into(),
            opening_profile_hash: opening_profile_hash.into(),
            multi_pv,
        };
        for (name, value) in [
            ("modelHash", key.model_hash.as_str()),
            ("evaluatorConfigHash", key.evaluator_config_hash.as_str()),
            ("featureSchemaHash", key.feature_schema_hash.as_str()),
            (
                "evaluationSemanticsHash",
                key.evaluation_semantics_hash.as_str(),
            ),
            ("searchOptionsHash", key.search_options_hash.as_str()),
            ("openingProfileHash", key.opening_profile_hash.as_str()),
        ] {
            validate_identity_hash(name, value)?;
        }
        Ok(key)
    }

    fn search_compatible_with(&self, other: &Self) -> bool {
        self.model_hash == other.model_hash
            && self.evaluator_config_hash == other.evaluator_config_hash
            && self.feature_schema_hash == other.feature_schema_hash
            && self.evaluation_semantics_hash == other.evaluation_semantics_hash
            && self.search_options_hash == other.search_options_hash
    }
}

/// One completed principal-variation line.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AnalysisLine {
    pub rank: u8,
    pub score: i32,
    pub mate_score: Option<i32>,
    pub depth: u8,
    pub nodes: u64,
    pub pv: Vec<crate::Move>,
}

/// Latest fully completed depth retained for one compatible cache key.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AnalysisCacheEntry {
    pub completed_depth: u8,
    pub nodes: u64,
    pub nps: u64,
    pub score: i32,
    pub mate_score: Option<i32>,
    pub lines: Vec<AnalysisLine>,
    pub root_move_statistics: Vec<RootMoveStat>,
    pub updated_at_ms: u64,
    pub engine_version: String,
    pub model_hash: String,
}

impl AnalysisCacheEntry {
    fn from_info(key: &AnalysisCacheKey, info: &SearchInfo, updated_at_ms: u64) -> Self {
        let mut lines = info
            .root_moves
            .iter()
            .take(usize::from(key.multi_pv))
            .enumerate()
            .map(|(index, root)| AnalysisLine {
                rank: u8::try_from(index + 1).unwrap_or(u8::MAX),
                score: root.score,
                mate_score: is_mate_score(root.score).then_some(root.score),
                depth: root.depth,
                nodes: root.nodes,
                pv: root.pv.clone(),
            })
            .collect::<Vec<_>>();
        if lines.is_empty() && !info.pv.is_empty() {
            lines.push(AnalysisLine {
                rank: 1,
                score: info.score,
                mate_score: is_mate_score(info.score).then_some(info.score),
                depth: info.depth,
                nodes: info.nodes,
                pv: info.pv.clone(),
            });
        }
        Self {
            completed_depth: info.depth,
            nodes: info.nodes,
            nps: info.nps,
            score: info.score,
            mate_score: is_mate_score(info.score).then_some(info.score),
            lines,
            root_move_statistics: info.root_moves.clone(),
            updated_at_ms,
            engine_version: ENGINE_VERSION.to_owned(),
            model_hash: key.model_hash.clone(),
        }
    }
}

/// Origin of an incremental update.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AnalysisUpdateSource {
    Cache,
    Search,
}

/// Cache display or newly completed iterative-deepening update.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AnalysisUpdate {
    pub source: AnalysisUpdateSource,
    pub key: AnalysisCacheKey,
    pub entry: AnalysisCacheEntry,
}

/// Current lifecycle of the logical analysis instance.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AnalysisState {
    Idle,
    Active,
    Stopped,
    WorkerFailed,
}

/// Result of one bounded analysis work slice.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AnalysisStep {
    pub updates: Vec<AnalysisUpdate>,
    pub result: SearchResult,
}

/// Stateful analysis instance, separate from any play `SearchEngine`.
pub struct AnalysisService {
    engine: SearchEngine,
    cache: BTreeMap<AnalysisCacheKey, AnalysisCacheEntry>,
    maximum_cache_entries: usize,
    active_position: Option<Position>,
    active_key: Option<AnalysisCacheKey>,
    cancellation: CancellationToken,
    state: AnalysisState,
    worker_failures: u64,
}

impl AnalysisService {
    /// Creates a service with a bounded cache. The supplied engine must be dedicated to analysis.
    ///
    /// # Errors
    ///
    /// Returns an error when the cache capacity is zero.
    pub fn new(engine: SearchEngine, maximum_cache_entries: usize) -> Result<Self, String> {
        if maximum_cache_entries == 0 {
            return Err("analysis cache must retain at least one entry".to_owned());
        }
        Ok(Self {
            engine,
            cache: BTreeMap::new(),
            maximum_cache_entries,
            active_position: None,
            active_key: None,
            cancellation: CancellationToken::new(),
            state: AnalysisState::Idle,
            worker_failures: 0,
        })
    }

    /// Starts or changes the canonical root. Cached output is returned immediately when present.
    ///
    /// # Errors
    ///
    /// Returns an error when the key does not identify the supplied position exactly.
    pub fn start(
        &mut self,
        position: Position,
        key: AnalysisCacheKey,
    ) -> Result<Option<AnalysisUpdate>, String> {
        if key.canonical_position != to_sfen(&position)
            || key.position_hash != position.zobrist_hash()
        {
            return Err("analysis cache key does not identify the supplied position".to_owned());
        }
        self.cancellation.cancel();
        if self
            .active_key
            .as_ref()
            .is_some_and(|active| !active.search_compatible_with(&key))
        {
            self.engine.clear_transpositions();
        }
        self.cancellation = CancellationToken::new();
        self.active_position = Some(position);
        self.active_key = Some(key.clone());
        self.state = AnalysisState::Active;
        Ok(self.cache.get(&key).cloned().map(|entry| AnalysisUpdate {
            source: AnalysisUpdateSource::Cache,
            key,
            entry,
        }))
    }

    /// Cancels the active root. Completed cache entries and compatible TT entries are retained.
    pub fn stop(&mut self) {
        self.cancellation.cancel();
        self.state = AnalysisState::Stopped;
    }

    /// Performs one bounded renewed iterative-deepening slice with compatible TT reuse.
    ///
    /// # Errors
    ///
    /// Returns an error unless analysis has an active position and cache key.
    pub fn step(&mut self, plan: TimePlan, updated_at_ms: u64) -> Result<AnalysisStep, String> {
        if self.state != AnalysisState::Active {
            return Err("analysis is not active".to_owned());
        }
        let position = self
            .active_position
            .as_ref()
            .ok_or_else(|| "analysis has no active position".to_owned())?;
        let key = self
            .active_key
            .clone()
            .ok_or_else(|| "analysis has no active cache key".to_owned())?;
        let mut completed = Vec::new();
        let result = self.engine.search_reusing_transpositions_with_callback(
            position,
            plan,
            &self.cancellation,
            |info| completed.push(info.clone()),
        );
        let mut updates = Vec::with_capacity(completed.len());
        for info in completed {
            let entry = AnalysisCacheEntry::from_info(&key, &info, updated_at_ms);
            self.insert_cache(key.clone(), entry.clone());
            updates.push(AnalysisUpdate {
                source: AnalysisUpdateSource::Search,
                key: key.clone(),
                entry,
            });
        }
        Ok(AnalysisStep { updates, result })
    }

    /// Marks a host worker failure. The last completed cache entry remains displayable.
    pub fn record_worker_failure(&mut self) -> Option<AnalysisUpdate> {
        self.cancellation.cancel();
        self.worker_failures = self.worker_failures.saturating_add(1);
        self.state = AnalysisState::WorkerFailed;
        self.cached_active_update()
    }

    /// Restarts the current root after a host worker failure with a fresh cancellation token.
    ///
    /// # Errors
    ///
    /// Returns an error unless a worker failure was previously recorded.
    pub fn restart_after_worker_failure(&mut self) -> Result<Option<AnalysisUpdate>, String> {
        if self.state != AnalysisState::WorkerFailed {
            return Err("analysis worker has not failed".to_owned());
        }
        self.cancellation = CancellationToken::new();
        self.state = AnalysisState::Active;
        Ok(self.cached_active_update())
    }

    /// Removes cached entries incompatible with the supplied immutable analysis identity.
    pub fn invalidate_incompatible(&mut self, compatible: &AnalysisCacheKey) -> usize {
        let before = self.cache.len();
        self.cache
            .retain(|key, _| key.search_compatible_with(compatible));
        if self
            .active_key
            .as_ref()
            .is_some_and(|active| !active.search_compatible_with(compatible))
        {
            self.engine.clear_transpositions();
        }
        before.saturating_sub(self.cache.len())
    }

    #[must_use]
    pub const fn state(&self) -> AnalysisState {
        self.state
    }

    #[must_use]
    pub const fn worker_failures(&self) -> u64 {
        self.worker_failures
    }

    #[must_use]
    pub fn cache_len(&self) -> usize {
        self.cache.len()
    }

    fn cached_active_update(&self) -> Option<AnalysisUpdate> {
        let key = self.active_key.as_ref()?;
        self.cache.get(key).cloned().map(|entry| AnalysisUpdate {
            source: AnalysisUpdateSource::Cache,
            key: key.clone(),
            entry,
        })
    }

    fn insert_cache(&mut self, key: AnalysisCacheKey, entry: AnalysisCacheEntry) {
        if !self.cache.contains_key(&key) && self.cache.len() >= self.maximum_cache_entries {
            let oldest = self
                .cache
                .iter()
                .min_by_key(|(_, entry)| entry.updated_at_ms)
                .map(|(key, _)| key.clone());
            if let Some(oldest) = oldest {
                self.cache.remove(&oldest);
            }
        }
        self.cache.insert(key, entry);
    }
}

fn validate_identity_hash(name: &str, value: &str) -> Result<(), String> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(format!("{name} must be lowercase SHA-256 hexadecimal"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{SearchConfig, Side, TimeControl, TimeManager};

    fn hash(digit: char) -> String {
        digit.to_string().repeat(64)
    }

    fn key(position: &Position, multi_pv: u8) -> AnalysisCacheKey {
        AnalysisCacheKey::new(
            position,
            hash('1'),
            hash('2'),
            hash('3'),
            hash('4'),
            hash('5'),
            hash('6'),
            multi_pv,
        )
        .unwrap()
    }

    fn node_plan(nodes: u64, depth: u8) -> TimePlan {
        TimeManager::default()
            .plan(
                Side::Black,
                TimeControl {
                    nodes: Some(nodes),
                    depth: Some(depth),
                    casual: false,
                    ..TimeControl::casual()
                },
                depth,
            )
            .unwrap()
    }

    #[test]
    fn position_switch_cancels_prior_root_and_publishes_cached_result_immediately() {
        let first = Position::startpos();
        let first_key = key(&first, 2);
        let mut service = AnalysisService::new(
            SearchEngine::new(SearchConfig::default()),
            DEFAULT_ANALYSIS_CACHE_ENTRIES,
        )
        .unwrap();
        assert!(
            service
                .start(first.clone(), first_key.clone())
                .unwrap()
                .is_none()
        );
        let step = service.step(node_plan(2_000, 3), 10).unwrap();
        assert!(!step.updates.is_empty());

        let mut second = first.clone();
        second.make_move(second.legal_moves()[0]).unwrap();
        let second_key = key(&second, 2);
        assert!(service.start(second, second_key).unwrap().is_none());
        let cached = service.start(first, first_key).unwrap().unwrap();
        assert_eq!(cached.source, AnalysisUpdateSource::Cache);
        assert!(cached.entry.completed_depth > 0);
    }

    #[test]
    fn cancellation_failure_restart_and_invalidation_are_explicit() {
        let position = Position::startpos();
        let original = key(&position, 1);
        let mut service = AnalysisService::new(
            SearchEngine::new(SearchConfig::default()),
            DEFAULT_ANALYSIS_CACHE_ENTRIES,
        )
        .unwrap();
        service.start(position.clone(), original.clone()).unwrap();
        service.step(node_plan(1_000, 2), 20).unwrap();
        assert!(service.record_worker_failure().is_some());
        assert_eq!(service.state(), AnalysisState::WorkerFailed);
        assert!(service.restart_after_worker_failure().unwrap().is_some());
        service.stop();
        assert!(service.step(node_plan(10, 1), 21).is_err());

        let mut incompatible = key(&position, 1);
        incompatible.model_hash = hash('a');
        assert_eq!(service.invalidate_incompatible(&incompatible), 1);
        assert_eq!(service.cache_len(), 0);
    }

    #[test]
    fn cache_key_rejects_identity_and_position_mismatch() {
        let position = Position::startpos();
        assert!(
            AnalysisCacheKey::new(
                &position,
                "bad",
                hash('2'),
                hash('3'),
                hash('4'),
                hash('5'),
                hash('6'),
                1,
            )
            .is_err()
        );
        let mut wrong = key(&position, 1);
        wrong.position_hash ^= 1;
        let mut service =
            AnalysisService::new(SearchEngine::new(SearchConfig::default()), 1).unwrap();
        assert!(service.start(position, wrong).is_err());
    }

    #[test]
    fn renewed_iterative_deepening_reuses_interior_transpositions() {
        let position = Position::startpos();
        let cache_key = key(&position, 3);
        let mut service = AnalysisService::new(
            SearchEngine::new(SearchConfig::default()),
            DEFAULT_ANALYSIS_CACHE_ENTRIES,
        )
        .unwrap();
        service.start(position, cache_key).unwrap();
        let first = service.step(node_plan(50_000, 4), 1).unwrap();
        let second = service.step(node_plan(50_000, 4), 2).unwrap();

        eprintln!(
            "analysis transposition reuse: first_nodes={} second_nodes={} second_tt_hits={} second_tt_probes={}",
            first.result.nodes,
            second.result.nodes,
            second.result.stats.tt_hits,
            second.result.stats.tt_probes,
        );

        assert!(second.result.stats.tt_hits > 0);
        assert!(second.result.nodes <= first.result.nodes);
        assert!(!second.result.root_moves.is_empty());
        assert!(
            second
                .updates
                .last()
                .is_some_and(|update| update.entry.lines.len() == 3)
        );
    }
}
