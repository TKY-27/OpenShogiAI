//! Browser-facing, single-worker adapter for the independently implemented engine core.
//!
//! The public JavaScript boundary accepts only bounded strings, model bytes, and closed profile
//! names. Search remains synchronous inside a dedicated Web Worker; the page cancels work by
//! terminating that worker, so this crate does not require shared memory or browser threads.

#![forbid(unsafe_code)]

use std::{fmt::Write as _, sync::Arc, time::Duration};

use open_shogi_core::{
    ANALYSIS_SCHEMA, AnalysisCacheKey, AnalysisService, AnalysisUpdate, AnalysisUpdateSource,
    CancellationToken, EngineIdentity, EnteringKingDeclaration, Game, GameEnd, HandPiece,
    ImpasseOutcome, Move, NeuralActivation, NeuralEvaluationMode, NeuralEvaluator,
    NeuralQuantization, OpeningBookChoice, OpeningBookV2, OpeningPolicy, OpeningProfile,
    Osaval02Evaluator, Osaval02History, PieceKind, Position, RepetitionOutcome, SearchConfig,
    SearchEngine, SearchLimits, SearchResult, SearchStats, SearchTermination, Side, Square,
    TimeControl, TimeControlMode, TimeManager, parse_sfen, parse_usi_move, to_sfen, to_usi_move,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

const SNAPSHOT_SCHEMA: &str = "open_shogi_browser_snapshot/v1";
const SEARCH_SCHEMA: &str = "open_shogi_browser_search/v1";
const MODEL_SCHEMA: &str = "open_shogi_browser_model/v1";
const OPENING_BOOK_SUMMARY_SCHEMA: &str = "open_shogi_browser_opening_book/v1";
const MAX_BROWSER_MODEL_BYTES: usize = 16 * 1024 * 1024;
const MAX_BROWSER_OPENING_BOOK_BYTES: usize = 64 * 1024 * 1024;
const MAX_RESTORE_JSON_BYTES: usize = 16 * 1024;
const MAX_RESTORE_MOVES: usize = 512;
const MAX_USI_MOVE_BYTES: usize = 8;

fn default_safety_margin_ms() -> u64 {
    50
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct BrowserTimeControl {
    schema: String,
    #[serde(default)]
    black_time_ms: Option<u64>,
    #[serde(default)]
    white_time_ms: Option<u64>,
    #[serde(default)]
    byoyomi_ms: Option<u64>,
    #[serde(default)]
    black_increment_ms: Option<u64>,
    #[serde(default)]
    white_increment_ms: Option<u64>,
    #[serde(default)]
    movetime_ms: Option<u64>,
    #[serde(default)]
    nodes: Option<u64>,
    #[serde(default)]
    depth: Option<u8>,
    #[serde(default)]
    infinite: bool,
    #[serde(default)]
    casual: bool,
    #[serde(default = "default_safety_margin_ms")]
    safety_margin_ms: u64,
}

impl BrowserTimeControl {
    fn into_core(self) -> Result<TimeControl, String> {
        if self.schema != open_shogi_core::TIME_CONTROL_SCHEMA {
            return Err(format!(
                "time-control schema must be {}",
                open_shogi_core::TIME_CONTROL_SCHEMA
            ));
        }
        if self.infinite {
            return Err(
                "infinite analysis must use the analysis service, not play search".to_owned(),
            );
        }
        Ok(TimeControl {
            black_time_ms: self.black_time_ms,
            white_time_ms: self.white_time_ms,
            byoyomi_ms: self.byoyomi_ms,
            black_increment_ms: self.black_increment_ms,
            white_increment_ms: self.white_increment_ms,
            movetime_ms: self.movetime_ms,
            nodes: self.nodes,
            depth: self.depth,
            infinite: self.infinite,
            casual: self.casual,
            safety_margin_ms: self.safety_margin_ms,
        })
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct BrowserAnalysisStart {
    schema: String,
    position_sfen: String,
    model_hash: String,
    evaluator_config_hash: String,
    feature_schema_hash: String,
    evaluation_semantics_hash: String,
    search_options_hash: String,
    opening_profile_hash: String,
    multi_pv: u8,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct BrowserAnalysisStep {
    schema: String,
    nodes: u64,
    max_depth: u8,
    timestamp_ms: u64,
}

/// Returns the engine version for host-side smoke tests and the browser worker handshake.
#[must_use]
pub const fn engine_version() -> &'static str {
    EngineIdentity::current().version
}

#[derive(Clone, Copy)]
enum BrowserSearchProfile {
    Eco,
    Balanced,
    Quality,
}

impl BrowserSearchProfile {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "eco" => Ok(Self::Eco),
            "balanced" => Ok(Self::Balanced),
            "quality" => Ok(Self::Quality),
            _ => Err("search profile must be eco, balanced, or quality".to_owned()),
        }
    }

    const fn name(self) -> &'static str {
        match self {
            Self::Eco => "eco",
            Self::Balanced => "balanced",
            Self::Quality => "quality",
        }
    }

    const fn max_depth(self) -> u8 {
        match self {
            Self::Eco => 5,
            Self::Balanced => 7,
            Self::Quality => 9,
        }
    }

    const fn max_nodes(self) -> u64 {
        match self {
            Self::Eco => 1_500,
            Self::Balanced => 4_000,
            Self::Quality => 12_000,
        }
    }

    const fn transposition_megabytes(self) -> usize {
        match self {
            Self::Eco => 2,
            Self::Balanced => 4,
            Self::Quality => 8,
        }
    }

    const fn quiescence_depth(self) -> u8 {
        match self {
            Self::Eco => 4,
            Self::Balanced => 6,
            Self::Quality => 8,
        }
    }
}

#[derive(Clone, Copy)]
enum EvaluatorChoice {
    OverallChampion,
    Model,
    ModelResidual,
    ModelComposite,
}

impl EvaluatorChoice {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "overall-champion" | "handcrafted" => Ok(Self::OverallChampion),
            "model" => Ok(Self::Model),
            "model-residual" => Ok(Self::ModelResidual),
            "model-composite" => Ok(Self::ModelComposite),
            _ => Err(
                "evaluator must be overall-champion, model, model-residual, or model-composite"
                    .to_owned(),
            ),
        }
    }

    const fn name(self) -> &'static str {
        match self {
            Self::OverallChampion => "overall-champion",
            Self::Model => "model",
            Self::ModelResidual => "model-residual",
            Self::ModelComposite => "model-composite-50-50",
        }
    }
}

struct LoadedModel {
    evaluator: Arc<NeuralEvaluator>,
    summary: ModelSummary,
}

/// Stateful rules, model, and search adapter owned by exactly one browser worker.
pub struct BrowserEngine {
    initial_sfen: String,
    game: Game,
    model: Option<LoadedModel>,
    opening_book: Option<OpeningBookV2>,
    opening_book_summary: Option<OpeningBookSummary>,
    opening_policy: OpeningPolicy,
    opening_max_plies: u32,
    analysis: Option<AnalysisService>,
    analysis_identity: Option<(String, String)>,
    analysis_side: Option<Side>,
    analysis_profile: Option<BrowserSearchProfile>,
}

impl Default for BrowserEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl BrowserEngine {
    /// Starts at the standard position with the overall champion selected.
    #[must_use]
    pub fn new() -> Self {
        let position = Position::startpos();
        Self {
            initial_sfen: to_sfen(&position),
            game: Game::new(position),
            model: None,
            opening_book: None,
            opening_book_summary: None,
            opening_policy: OpeningPolicy::default(),
            opening_max_plies: 40,
            analysis: None,
            analysis_identity: None,
            analysis_side: None,
            analysis_profile: None,
        }
    }

    /// Returns the complete closed browser-state document.
    ///
    /// # Errors
    ///
    /// Returns an error if the internal bounded snapshot cannot be serialized.
    pub fn snapshot_json(&self) -> Result<String, String> {
        serde_json::to_string(&self.snapshot()).map_err(|error| error.to_string())
    }

    /// Resets to startpos or to one canonical SFEN position.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid or noncanonical SFEN or serialization failure.
    pub fn reset(&mut self, sfen: Option<&str>) -> Result<String, String> {
        let position = match sfen {
            Some(value) => parse_canonical_sfen(value)?,
            None => Position::startpos(),
        };
        self.initial_sfen = to_sfen(&position);
        self.game = Game::new(position);
        self.snapshot_json()
    }

    /// Restores an initial position and exact USI move history after worker cancellation.
    ///
    /// # Errors
    ///
    /// Returns an error for an oversized or malformed history, a noncanonical initial SFEN,
    /// an illegal move, or serialization failure.
    pub fn restore(&mut self, initial_sfen: &str, moves_json: &str) -> Result<String, String> {
        if moves_json.len() > MAX_RESTORE_JSON_BYTES {
            return Err("restore move history exceeds the browser byte limit".to_owned());
        }
        let moves: Vec<String> = serde_json::from_str(moves_json)
            .map_err(|_| "restore move history must be a JSON string array".to_owned())?;
        if moves.len() > MAX_RESTORE_MOVES {
            return Err("restore move history exceeds 512 moves".to_owned());
        }

        let position = parse_canonical_sfen(initial_sfen)?;
        let mut game = Game::new(position);
        for movement in moves {
            let parsed = parse_bounded_usi_move(&movement)?;
            game.play(parsed)
                .map_err(|error| format!("restore move is not legal: {error}"))?;
        }
        initial_sfen.clone_into(&mut self.initial_sfen);
        self.game = game;
        self.snapshot_json()
    }

    /// Parses a bounded OSAVAL artifact and records both whole-file and payload identities.
    ///
    /// # Errors
    ///
    /// Returns an error for an oversized, malformed, corrupt, unsupported, or unexpectedly
    /// hashed model artifact, or if its summary cannot be serialized.
    pub fn load_model(
        &mut self,
        bytes: &[u8],
        expected_artifact_sha256: Option<&str>,
    ) -> Result<String, String> {
        if bytes.len() > MAX_BROWSER_MODEL_BYTES {
            return Err(format!(
                "browser model is {} bytes; maximum is {MAX_BROWSER_MODEL_BYTES}",
                bytes.len()
            ));
        }
        let artifact_sha256 = sha256_hex(bytes);
        if let Some(expected) = expected_artifact_sha256 {
            validate_sha256(expected)?;
            if expected != artifact_sha256 {
                return Err("model artifact SHA-256 does not match the expected value".to_owned());
            }
        }

        let evaluator = NeuralEvaluator::from_bytes(bytes)
            .map_err(|error| format!("model validation failed: {error}"))?;
        let summary = ModelSummary::from_evaluator(&evaluator, artifact_sha256, bytes.len());
        self.model = Some(LoadedModel {
            evaluator: Arc::new(evaluator),
            summary,
        });
        self.analysis = None;
        self.analysis_identity = None;
        serde_json::to_string(&self.model.as_ref().map(|model| &model.summary))
            .map_err(|error| error.to_string())
    }

    /// Removes model weights from the worker-owned state.
    ///
    /// # Errors
    ///
    /// Returns an error if the resulting bounded snapshot cannot be serialized.
    pub fn unload_model(&mut self) -> Result<String, String> {
        self.model = None;
        self.analysis = None;
        self.analysis_identity = None;
        self.snapshot_json()
    }

    /// Loads and fully validates one provenance-bound `OpenShogiAI` opening-book v2 snapshot.
    ///
    /// # Errors
    ///
    /// Returns an error for oversized, corrupt, incompatible, or hash-mismatched bytes.
    pub fn load_opening_book(
        &mut self,
        bytes: &[u8],
        expected_artifact_sha256: Option<&str>,
    ) -> Result<String, String> {
        if bytes.len() > MAX_BROWSER_OPENING_BOOK_BYTES {
            return Err(format!(
                "browser opening book is {} bytes; maximum is {MAX_BROWSER_OPENING_BOOK_BYTES}",
                bytes.len()
            ));
        }
        let artifact_sha256 = sha256_hex(bytes);
        if let Some(expected) = expected_artifact_sha256 {
            validate_sha256(expected)?;
            if expected != artifact_sha256 {
                return Err("opening-book SHA-256 does not match the expected value".to_owned());
            }
        }
        let book = OpeningBookV2::from_compressed_bytes(bytes)
            .map_err(|error| format!("opening-book validation failed: {error}"))?;
        let summary = OpeningBookSummary {
            schema: OPENING_BOOK_SUMMARY_SCHEMA,
            artifact_sha256,
            artifact_size: bytes.len(),
            positions: book.positions(),
            candidates: book.candidates(),
        };
        self.opening_book = Some(book);
        self.opening_book_summary = Some(summary.clone());
        serde_json::to_string(&summary).map_err(|error| error.to_string())
    }

    /// Removes the browser-owned opening book without affecting game or analysis state.
    ///
    /// # Errors
    ///
    /// Returns an error if the resulting snapshot cannot be serialized.
    pub fn unload_opening_book(&mut self) -> Result<String, String> {
        self.opening_book = None;
        self.opening_book_summary = None;
        self.snapshot_json()
    }

    /// Configures the opening-only style policy. Legal move generation is never changed.
    ///
    /// # Errors
    ///
    /// Returns an error for an unknown profile or values outside the closed policy bounds.
    pub fn configure_opening(
        &mut self,
        profile: &str,
        max_plies: u32,
        minimum_sample_count: u64,
        maximum_teacher_loss_cp: i32,
    ) -> Result<String, String> {
        if !(1..=40).contains(&max_plies) {
            return Err("opening max plies must be between 1 and 40".to_owned());
        }
        if minimum_sample_count == 0 || maximum_teacher_loss_cp < 0 {
            return Err(
                "opening minimum samples must be positive and teacher loss nonnegative".to_owned(),
            );
        }
        self.opening_policy = OpeningPolicy {
            profile: OpeningProfile::parse(profile)?,
            minimum_sample_count,
            maximum_teacher_loss_cp,
        };
        self.opening_max_plies = max_plies;
        serde_json::to_string(&OpeningPolicySummary {
            profile: self.opening_policy.profile.name(),
            max_plies: self.opening_max_plies,
            minimum_sample_count: self.opening_policy.minimum_sample_count,
            maximum_teacher_loss_cp: self.opening_policy.maximum_teacher_loss_cp,
        })
        .map_err(|error| error.to_string())
    }

    /// Applies one legal USI move and returns the resulting state.
    ///
    /// # Errors
    ///
    /// Returns an error if the game has ended, the move is malformed or illegal, or the
    /// resulting state cannot be serialized.
    pub fn play_move(&mut self, movement: &str) -> Result<String, String> {
        if self.game.end().is_some() {
            return Err("the game has already ended".to_owned());
        }
        let parsed = parse_bounded_usi_move(movement)?;
        self.game
            .play(parsed)
            .map_err(|error| format!("move is not legal: {error}"))?;
        self.snapshot_json()
    }

    /// Runs one fixed, device-profile-bounded search without changing the position.
    ///
    /// # Errors
    ///
    /// Returns an error for a terminal game, unknown profile or evaluator, a missing selected
    /// model, or serialization failure.
    pub fn search_json(
        &self,
        profile: &str,
        evaluator: &str,
        multi_pv: u8,
    ) -> Result<String, String> {
        if self.game.end().is_some() {
            return Err("the game has already ended".to_owned());
        }
        if !(1..=3).contains(&multi_pv) {
            return Err("multi-PV count must be between 1 and 3".to_owned());
        }
        let profile = BrowserSearchProfile::parse(profile)?;
        let evaluator = EvaluatorChoice::parse(evaluator)?;
        if let Some(choice) = self.book_choice() {
            return serde_json::to_string(&SearchResponse::book(
                profile,
                evaluator,
                self.game.position().side_to_move(),
                choice,
                "profile-nodes",
            ))
            .map_err(|error| error.to_string());
        }
        let config = SearchConfig {
            transposition_entries: SearchEngine::transposition_entries_for_megabytes(
                profile.transposition_megabytes(),
            ),
            quiescence_depth: profile.quiescence_depth(),
            ..SearchConfig::default()
        };
        let mut engine = self.search_engine(config, evaluator)?;
        let result = engine.search(
            self.game.position(),
            SearchLimits {
                max_depth: profile.max_depth(),
                max_nodes: Some(profile.max_nodes()),
                movetime: None,
            },
            &CancellationToken::new(),
        );
        let lines = Self::multi_pv_lines(&result, multi_pv);
        let response = SearchResponse::new(
            profile,
            evaluator,
            self.game.position().side_to_move(),
            result,
            lines,
            "profile-nodes",
        );
        serde_json::to_string(&response).map_err(|error| error.to_string())
    }

    /// Runs a play search using the shared versioned time-control request.
    ///
    /// Infinite work belongs to the separate analysis service and is rejected here.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid state, request, limits, evaluator, or serialization.
    pub fn search_time_control_json(
        &self,
        profile: &str,
        evaluator: &str,
        multi_pv: u8,
        time_control_json: &str,
    ) -> Result<String, String> {
        if self.game.end().is_some() {
            return Err("the game has already ended".to_owned());
        }
        if !(1..=3).contains(&multi_pv) {
            return Err("multi-PV count must be between 1 and 3".to_owned());
        }
        if time_control_json.len() > 4_096 {
            return Err("time-control request exceeds 4096 bytes".to_owned());
        }
        let profile = BrowserSearchProfile::parse(profile)?;
        let evaluator = EvaluatorChoice::parse(evaluator)?;
        let request: BrowserTimeControl =
            serde_json::from_str(time_control_json).map_err(|error| error.to_string())?;
        let request = request.into_core()?;
        if request
            .nodes
            .is_some_and(|nodes| nodes > profile.max_nodes())
        {
            return Err(format!(
                "nodes exceeds the selected browser profile limit {}",
                profile.max_nodes()
            ));
        }
        let plan = TimeManager::default().plan(
            self.game.position().side_to_move(),
            request,
            profile.max_depth(),
        )?;
        let mode = time_control_mode_name(plan.mode);
        if let Some(choice) = self.book_choice() {
            return serde_json::to_string(&SearchResponse::book(
                profile,
                evaluator,
                self.game.position().side_to_move(),
                choice,
                mode,
            ))
            .map_err(|error| error.to_string());
        }
        let config = SearchConfig {
            transposition_entries: SearchEngine::transposition_entries_for_megabytes(
                profile.transposition_megabytes(),
            ),
            quiescence_depth: profile.quiescence_depth(),
            ..SearchConfig::default()
        };
        let mut engine = self.search_engine(config, evaluator)?;
        let result = engine.search_managed(self.game.position(), plan, &CancellationToken::new());
        let lines = Self::multi_pv_lines(&result, multi_pv);
        let response = SearchResponse::new(
            profile,
            evaluator,
            self.game.position().side_to_move(),
            result,
            lines,
            mode,
        );
        serde_json::to_string(&response).map_err(|error| error.to_string())
    }

    /// Starts infinite logical analysis for a canonical root.
    ///
    /// The browser host repeatedly calls `analysis_step_json`; cached results are returned by
    /// this call immediately. The analysis engine and its TT are distinct from play search.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid protocol input, identities, position, evaluator, or limits.
    pub fn analysis_start_json(
        &mut self,
        profile: &str,
        evaluator: &str,
        request_json: &str,
    ) -> Result<String, String> {
        if request_json.len() > 16 * 1024 {
            return Err("analysis start request exceeds 16384 bytes".to_owned());
        }
        let profile = BrowserSearchProfile::parse(profile)?;
        let evaluator = EvaluatorChoice::parse(evaluator)?;
        let request: BrowserAnalysisStart =
            serde_json::from_str(request_json).map_err(|error| error.to_string())?;
        if request.schema != ANALYSIS_SCHEMA {
            return Err(format!("analysis schema must be {ANALYSIS_SCHEMA}"));
        }
        let position = parse_canonical_sfen(&request.position_sfen)?;
        let key = AnalysisCacheKey::new(
            &position,
            request.model_hash.clone(),
            request.evaluator_config_hash,
            request.feature_schema_hash,
            request.evaluation_semantics_hash,
            request.search_options_hash,
            request.opening_profile_hash,
            request.multi_pv,
        )?;
        let runtime_identity = (
            format!("{}:{}", profile.name(), evaluator.name()),
            request.model_hash,
        );
        if self.analysis_identity.as_ref() != Some(&runtime_identity) {
            let config = SearchConfig {
                transposition_entries: SearchEngine::transposition_entries_for_megabytes(
                    profile.transposition_megabytes(),
                ),
                quiescence_depth: profile.quiescence_depth(),
                ..SearchConfig::default()
            };
            let engine = self.search_engine(config, evaluator)?;
            self.analysis = Some(AnalysisService::new(engine, 256)?);
            self.analysis_identity = Some(runtime_identity);
        }
        let cached = self
            .analysis
            .as_mut()
            .ok_or_else(|| "analysis service was not created".to_owned())?
            .start(position.clone(), key)?;
        self.analysis_side = Some(position.side_to_move());
        self.analysis_profile = Some(profile);
        analysis_response("started", cached.into_iter().collect(), None)
    }

    /// Performs one cooperative analysis slice and returns all completed-depth updates.
    ///
    /// # Errors
    ///
    /// Returns an error unless analysis is active and the bounded request is valid.
    pub fn analysis_step_json(&mut self, request_json: &str) -> Result<String, String> {
        if request_json.len() > 4_096 {
            return Err("analysis step request exceeds 4096 bytes".to_owned());
        }
        let request: BrowserAnalysisStep =
            serde_json::from_str(request_json).map_err(|error| error.to_string())?;
        if request.schema != ANALYSIS_SCHEMA {
            return Err(format!("analysis schema must be {ANALYSIS_SCHEMA}"));
        }
        let profile = self
            .analysis_profile
            .ok_or_else(|| "analysis has not been started".to_owned())?;
        if request.nodes == 0 || request.nodes > profile.max_nodes() {
            return Err(format!(
                "analysis step nodes must be 1..={}",
                profile.max_nodes()
            ));
        }
        let side = self
            .analysis_side
            .ok_or_else(|| "analysis has not been started".to_owned())?;
        let plan = TimeManager::default().plan(
            side,
            TimeControl {
                nodes: Some(request.nodes),
                depth: Some(request.max_depth),
                casual: false,
                ..TimeControl::casual()
            },
            profile.max_depth(),
        )?;
        let step = self
            .analysis
            .as_mut()
            .ok_or_else(|| "analysis has not been started".to_owned())?
            .step(plan, request.timestamp_ms)?;
        analysis_response("updates", step.updates, Some(&step.result))
    }

    /// Stops analysis while retaining completed compatible cache and TT evidence.
    ///
    /// # Errors
    ///
    /// Returns an error when analysis has not been started or serialization fails.
    pub fn analysis_stop_json(&mut self) -> Result<String, String> {
        self.analysis
            .as_mut()
            .ok_or_else(|| "analysis has not been started".to_owned())?
            .stop();
        analysis_response("stopped", Vec::new(), None)
    }

    /// Records worker failure and returns the last completed cached result when available.
    ///
    /// # Errors
    ///
    /// Returns an error when analysis has not been started or serialization fails.
    pub fn analysis_worker_failed_json(&mut self) -> Result<String, String> {
        let cached = self
            .analysis
            .as_mut()
            .ok_or_else(|| "analysis has not been started".to_owned())?
            .record_worker_failure();
        analysis_response("worker-failed", cached.into_iter().collect(), None)
    }

    /// Restarts after a recorded worker failure and immediately republishes cached output.
    ///
    /// # Errors
    ///
    /// Returns an error unless a failure was recorded or serialization fails.
    pub fn analysis_restart_json(&mut self) -> Result<String, String> {
        let cached = self
            .analysis
            .as_mut()
            .ok_or_else(|| "analysis has not been started".to_owned())?
            .restart_after_worker_failure()?;
        analysis_response("restarted", cached.into_iter().collect(), None)
    }

    fn search_engine(
        &self,
        config: SearchConfig,
        evaluator: EvaluatorChoice,
    ) -> Result<SearchEngine, String> {
        match evaluator {
            EvaluatorChoice::OverallChampion => Ok(SearchEngine::new(config)),
            EvaluatorChoice::Model
            | EvaluatorChoice::ModelResidual
            | EvaluatorChoice::ModelComposite => {
                let model = self
                    .model
                    .as_ref()
                    .ok_or_else(|| "model evaluator selected without a loaded model".to_owned())?;
                let mode = match evaluator {
                    EvaluatorChoice::Model => NeuralEvaluationMode::PureValue,
                    EvaluatorChoice::ModelResidual => NeuralEvaluationMode::Residual,
                    EvaluatorChoice::ModelComposite => NeuralEvaluationMode::Composite,
                    EvaluatorChoice::OverallChampion => unreachable!(),
                };
                Ok(SearchEngine::with_neural_mode(
                    config,
                    Arc::clone(&model.evaluator),
                    mode,
                ))
            }
        }
    }

    fn book_choice(&self) -> Option<OpeningBookChoice> {
        if self.game.position().move_number() > self.opening_max_plies {
            return None;
        }
        self.opening_book
            .as_ref()?
            .select(self.game.position(), self.opening_policy)
    }

    fn multi_pv_lines(principal: &SearchResult, multi_pv: u8) -> Vec<SearchLineSummary> {
        let mut lines = principal
            .root_moves
            .iter()
            .take(usize::from(multi_pv))
            .map(SearchLineSummary::from_root)
            .collect::<Vec<_>>();
        if lines.is_empty() && principal.best_move.is_some() {
            lines.push(SearchLineSummary::principal(principal));
        }
        for (index, line) in lines.iter_mut().enumerate() {
            line.rank = u8::try_from(index + 1).unwrap_or(u8::MAX);
        }
        lines
    }

    fn snapshot(&self) -> BrowserSnapshot {
        let position = self.game.position();
        let legal_moves = if self.game.end().is_some() {
            Vec::new()
        } else {
            position
                .legal_moves()
                .into_iter()
                .map(MoveSummary::from)
                .collect()
        };
        let board = Square::all()
            .map(|square| {
                position
                    .piece_at(square)
                    .map(|piece| BoardPiece::new(square, piece.side, piece.kind))
            })
            .collect();
        BrowserSnapshot {
            schema: SNAPSHOT_SCHEMA,
            engine: EngineSummary {
                name: EngineIdentity::current().name,
                version: EngineIdentity::current().version,
            },
            initial_sfen: self.initial_sfen.clone(),
            sfen: to_sfen(position),
            side_to_move: side_name(position.side_to_move()),
            move_number: position.move_number(),
            board,
            hands: HandsSummary::from_position(position),
            legal_moves,
            moves: self.game.moves().iter().copied().map(to_usi_move).collect(),
            terminal: self.game.end().map(TerminalSummary::from),
            evaluator: EvaluatorSummary {
                kind: if self.model.is_some() {
                    "model-available"
                } else {
                    "handcrafted-only"
                },
                model: self.model.as_ref().map(|model| model.summary.clone()),
            },
            opening_book: self.opening_book_summary.clone(),
            opening_policy: OpeningPolicySummary {
                profile: self.opening_policy.profile.name(),
                max_plies: self.opening_max_plies,
                minimum_sample_count: self.opening_policy.minimum_sample_count,
                maximum_teacher_loss_cp: self.opening_policy.maximum_teacher_loss_cp,
            },
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct BrowserSnapshot {
    schema: &'static str,
    engine: EngineSummary,
    initial_sfen: String,
    sfen: String,
    side_to_move: &'static str,
    move_number: u32,
    board: Vec<Option<BoardPiece>>,
    hands: HandsSummary,
    legal_moves: Vec<MoveSummary>,
    moves: Vec<String>,
    terminal: Option<TerminalSummary>,
    evaluator: EvaluatorSummary,
    opening_book: Option<OpeningBookSummary>,
    opening_policy: OpeningPolicySummary,
}

#[derive(Serialize)]
struct EngineSummary {
    name: &'static str,
    version: &'static str,
}

#[derive(Serialize)]
struct BoardPiece {
    square: SquareSummary,
    side: &'static str,
    kind: &'static str,
}

impl BoardPiece {
    const fn new(square: Square, side: Side, kind: PieceKind) -> Self {
        Self {
            square: SquareSummary::from_square(square),
            side: side_name(side),
            kind: piece_kind_name(kind),
        }
    }
}

#[derive(Clone, Copy, Serialize)]
struct SquareSummary {
    file: u8,
    rank: u8,
}

impl SquareSummary {
    const fn from_square(square: Square) -> Self {
        Self {
            file: square.file(),
            rank: square.rank(),
        }
    }
}

#[derive(Serialize)]
struct HandsSummary {
    black: Vec<HandEntry>,
    white: Vec<HandEntry>,
}

impl HandsSummary {
    fn from_position(position: &Position) -> Self {
        Self {
            black: hand_entries(position, Side::Black),
            white: hand_entries(position, Side::White),
        }
    }
}

#[derive(Serialize)]
struct HandEntry {
    piece: &'static str,
    count: u8,
}

#[derive(Serialize)]
struct MoveSummary {
    usi: String,
    from: Option<SquareSummary>,
    to: SquareSummary,
    drop: Option<&'static str>,
    promote: bool,
}

impl From<Move> for MoveSummary {
    fn from(movement: Move) -> Self {
        match movement {
            Move::Normal { from, to, promote } => Self {
                usi: to_usi_move(movement),
                from: Some(SquareSummary::from_square(from)),
                to: SquareSummary::from_square(to),
                drop: None,
                promote,
            },
            Move::Drop { piece, to } => Self {
                usi: to_usi_move(movement),
                from: None,
                to: SquareSummary::from_square(to),
                drop: Some(hand_piece_name(piece)),
                promote: false,
            },
        }
    }
}

#[derive(Serialize)]
struct TerminalSummary {
    kind: &'static str,
    winner: Option<&'static str>,
    loser: Option<&'static str>,
}

impl From<GameEnd> for TerminalSummary {
    fn from(end: GameEnd) -> Self {
        match end {
            GameEnd::Checkmate { winner } => Self::decisive("checkmate", winner),
            GameEnd::Resignation { loser } => Self::decisive("resignation", loser.opposite()),
            GameEnd::Repetition(RepetitionOutcome::NoContest) => Self::neutral("repetition"),
            GameEnd::Repetition(RepetitionOutcome::PerpetualCheckLoss(loser)) => {
                Self::decisive("perpetual-check", loser.opposite())
            }
            GameEnd::Impasse(ImpasseOutcome::NoContest { .. }) => Self::neutral("impasse"),
            GameEnd::Impasse(ImpasseOutcome::Loss { loser, .. }) => {
                Self::decisive("impasse-loss", loser.opposite())
            }
            GameEnd::Impasse(ImpasseOutcome::InvalidMaterial { .. }) => {
                Self::neutral("invalid-material")
            }
            GameEnd::EnteringKing(EnteringKingDeclaration::Win { side, .. }) => {
                Self::decisive("entering-king", side)
            }
            GameEnd::EnteringKing(EnteringKingDeclaration::InvalidLoss { side }) => {
                Self::decisive("entering-king-loss", side.opposite())
            }
            GameEnd::EnteringKing(
                EnteringKingDeclaration::Disabled
                | EnteringKingDeclaration::Unavailable { .. }
                | EnteringKingDeclaration::NoContest { .. },
            ) => Self::neutral("entering-king-no-contest"),
        }
    }
}

impl TerminalSummary {
    const fn decisive(kind: &'static str, winner: Side) -> Self {
        Self {
            kind,
            winner: Some(side_name(winner)),
            loser: Some(side_name(winner.opposite())),
        }
    }

    const fn neutral(kind: &'static str) -> Self {
        Self {
            kind,
            winner: None,
            loser: None,
        }
    }
}

#[derive(Serialize)]
struct EvaluatorSummary {
    kind: &'static str,
    model: Option<ModelSummary>,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct OpeningBookSummary {
    schema: &'static str,
    artifact_sha256: String,
    artifact_size: usize,
    positions: usize,
    candidates: usize,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct OpeningPolicySummary {
    profile: &'static str,
    max_plies: u32,
    minimum_sample_count: u64,
    maximum_teacher_loss_cp: i32,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct ModelSummary {
    schema: &'static str,
    artifact_sha256: String,
    payload_sha256: String,
    artifact_size: usize,
    format_version: u32,
    architecture_version: u32,
    feature_schema_version: u32,
    feature_flags: u32,
    input_dimension: usize,
    hidden_layers: usize,
    hidden_dimension: usize,
    activation: &'static str,
    quantization: &'static str,
    layer_count: usize,
    output_scale_cp: f32,
}

impl ModelSummary {
    fn from_evaluator(
        evaluator: &NeuralEvaluator,
        artifact_sha256: String,
        artifact_size: usize,
    ) -> Self {
        let identity = evaluator.identity();
        Self {
            schema: MODEL_SCHEMA,
            artifact_sha256,
            payload_sha256: identity.sha256_hex(),
            artifact_size,
            format_version: identity.format_version,
            architecture_version: identity.architecture_version,
            feature_schema_version: identity.feature_schema_version,
            feature_flags: identity.feature_flags,
            input_dimension: identity.input_dimension,
            hidden_layers: identity.hidden_layers,
            hidden_dimension: identity.hidden_dimension,
            activation: match identity.activation {
                NeuralActivation::Relu => "relu",
            },
            quantization: match identity.quantization {
                NeuralQuantization::Float32 => "float32",
                NeuralQuantization::Int8 => "int8",
            },
            layer_count: identity.layer_count,
            output_scale_cp: identity.output_scale_cp,
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct SearchResponse {
    schema: &'static str,
    time_control_schema: &'static str,
    time_control_mode: &'static str,
    profile: &'static str,
    evaluator: &'static str,
    perspective: &'static str,
    source: &'static str,
    best_move: Option<String>,
    score_cp: i32,
    depth: u8,
    seldepth: u8,
    nodes: u64,
    elapsed_ns: u64,
    nps: u64,
    pv: Vec<String>,
    termination: &'static str,
    lines: Vec<SearchLineSummary>,
    stats: SearchStatsSummary,
    #[serde(skip_serializing_if = "Option::is_none")]
    opening_book_move: Option<OpeningBookMoveSummary>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct OpeningBookMoveSummary {
    sample_count: u64,
    teacher_score_cp: i32,
    teacher_depth: u8,
    teacher_nodes: u64,
    opening_classification: String,
    provenance_references: Vec<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct AnalysisResponse {
    schema: &'static str,
    event: &'static str,
    updates: Vec<AnalysisUpdateSummary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    slice: Option<AnalysisSliceSummary>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct AnalysisUpdateSummary {
    source: &'static str,
    canonical_position: String,
    position_hash: String,
    model_hash: String,
    evaluator_config_hash: String,
    feature_schema_hash: String,
    evaluation_semantics_hash: String,
    search_options_hash: String,
    opening_profile_hash: String,
    multi_pv: u8,
    depth: u8,
    nodes: u64,
    nps: u64,
    score: i32,
    mate_score: Option<i32>,
    lines: Vec<AnalysisLineSummary>,
    root_move_statistics: Vec<AnalysisRootMoveSummary>,
    timestamp_ms: u64,
    engine_version: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct AnalysisLineSummary {
    rank: u8,
    score: i32,
    mate_score: Option<i32>,
    depth: u8,
    nodes: u64,
    pv: Vec<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct AnalysisRootMoveSummary {
    movement: String,
    score: i32,
    depth: u8,
    nodes: u64,
    pv: Vec<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct AnalysisSliceSummary {
    termination: &'static str,
    depth: u8,
    nodes: u64,
    elapsed_ns: u64,
}

impl From<AnalysisUpdate> for AnalysisUpdateSummary {
    fn from(update: AnalysisUpdate) -> Self {
        Self {
            source: match update.source {
                AnalysisUpdateSource::Cache => "cache",
                AnalysisUpdateSource::Search => "search",
            },
            canonical_position: update.key.canonical_position,
            position_hash: format!("{:016x}", update.key.position_hash),
            model_hash: update.key.model_hash,
            evaluator_config_hash: update.key.evaluator_config_hash,
            feature_schema_hash: update.key.feature_schema_hash,
            evaluation_semantics_hash: update.key.evaluation_semantics_hash,
            search_options_hash: update.key.search_options_hash,
            opening_profile_hash: update.key.opening_profile_hash,
            multi_pv: update.key.multi_pv,
            depth: update.entry.completed_depth,
            nodes: update.entry.nodes,
            nps: update.entry.nps,
            score: update.entry.score,
            mate_score: update.entry.mate_score,
            lines: update
                .entry
                .lines
                .into_iter()
                .map(|line| AnalysisLineSummary {
                    rank: line.rank,
                    score: line.score,
                    mate_score: line.mate_score,
                    depth: line.depth,
                    nodes: line.nodes,
                    pv: line.pv.into_iter().map(to_usi_move).collect(),
                })
                .collect(),
            root_move_statistics: update
                .entry
                .root_move_statistics
                .into_iter()
                .map(|root| AnalysisRootMoveSummary {
                    movement: to_usi_move(root.movement),
                    score: root.score,
                    depth: root.depth,
                    nodes: root.nodes,
                    pv: root.pv.into_iter().map(to_usi_move).collect(),
                })
                .collect(),
            timestamp_ms: update.entry.updated_at_ms,
            engine_version: update.entry.engine_version,
        }
    }
}

fn analysis_response(
    event: &'static str,
    updates: Vec<AnalysisUpdate>,
    result: Option<&SearchResult>,
) -> Result<String, String> {
    serde_json::to_string(&AnalysisResponse {
        schema: ANALYSIS_SCHEMA,
        event,
        updates: updates.into_iter().map(Into::into).collect(),
        slice: result.map(|result| AnalysisSliceSummary {
            termination: termination_name(result.termination),
            depth: result.depth,
            nodes: result.nodes,
            elapsed_ns: duration_ns(result.elapsed),
        }),
    })
    .map_err(|error| error.to_string())
}

impl SearchResponse {
    fn new(
        profile: BrowserSearchProfile,
        evaluator: EvaluatorChoice,
        perspective: Side,
        result: open_shogi_core::SearchResult,
        lines: Vec<SearchLineSummary>,
        time_control_mode: &'static str,
    ) -> Self {
        Self {
            schema: SEARCH_SCHEMA,
            time_control_schema: open_shogi_core::TIME_CONTROL_SCHEMA,
            time_control_mode,
            profile: profile.name(),
            evaluator: evaluator.name(),
            perspective: side_name(perspective),
            source: "search",
            best_move: result.best_move.map(to_usi_move),
            score_cp: result.score,
            depth: result.depth,
            seldepth: result.seldepth,
            nodes: result.nodes,
            elapsed_ns: duration_ns(result.elapsed),
            nps: result.nps,
            pv: result.pv.into_iter().map(to_usi_move).collect(),
            termination: termination_name(result.termination),
            lines,
            stats: SearchStatsSummary::from(result.stats),
            opening_book_move: None,
        }
    }

    fn book(
        profile: BrowserSearchProfile,
        evaluator: EvaluatorChoice,
        perspective: Side,
        choice: OpeningBookChoice,
        time_control_mode: &'static str,
    ) -> Self {
        let best_move = to_usi_move(choice.movement);
        let score = choice.teacher_score_cp;
        let depth = choice.teacher_depth;
        Self {
            schema: SEARCH_SCHEMA,
            time_control_schema: open_shogi_core::TIME_CONTROL_SCHEMA,
            time_control_mode,
            profile: profile.name(),
            evaluator: evaluator.name(),
            perspective: side_name(perspective),
            source: "book",
            best_move: Some(best_move.clone()),
            score_cp: score,
            depth,
            seldepth: 0,
            nodes: 0,
            elapsed_ns: 0,
            nps: 0,
            pv: vec![best_move.clone()],
            termination: "book",
            lines: vec![SearchLineSummary {
                rank: 1,
                best_move: best_move.clone(),
                score_cp: score,
                depth,
                seldepth: 0,
                nodes: 0,
                pv: vec![best_move],
            }],
            stats: SearchStatsSummary::from(SearchStats::default()),
            opening_book_move: Some(OpeningBookMoveSummary {
                sample_count: choice.sample_count,
                teacher_score_cp: score,
                teacher_depth: depth,
                teacher_nodes: choice.teacher_nodes,
                opening_classification: choice.opening_classification,
                provenance_references: choice.provenance_references,
            }),
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct SearchLineSummary {
    rank: u8,
    best_move: String,
    score_cp: i32,
    depth: u8,
    seldepth: u8,
    nodes: u64,
    pv: Vec<String>,
}

impl SearchLineSummary {
    fn principal(result: &SearchResult) -> Self {
        Self {
            rank: 1,
            best_move: result
                .best_move
                .map_or_else(|| "none".to_owned(), to_usi_move),
            score_cp: result.score,
            depth: result.depth,
            seldepth: result.seldepth,
            nodes: result.nodes,
            pv: result.pv.iter().copied().map(to_usi_move).collect(),
        }
    }

    fn from_root(root: &open_shogi_core::RootMoveStat) -> Self {
        Self {
            rank: 0,
            best_move: to_usi_move(root.movement),
            score_cp: root.score,
            depth: root.depth,
            seldepth: u8::try_from(root.pv.len()).unwrap_or(u8::MAX),
            nodes: root.nodes,
            pv: root.pv.iter().copied().map(to_usi_move).collect(),
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct SearchStatsSummary {
    tt_probes: u64,
    tt_hits: u64,
    tt_collisions: u64,
    beta_cutoffs: u64,
    candidate_moves: u64,
    pruned_moves: u64,
    qnodes: u64,
    neural_inference_calls: u64,
    neural_inference_time_ns: u64,
}

impl From<SearchStats> for SearchStatsSummary {
    fn from(stats: SearchStats) -> Self {
        Self {
            tt_probes: stats.tt_probes,
            tt_hits: stats.tt_hits,
            tt_collisions: stats.tt_collisions,
            beta_cutoffs: stats.beta_cutoffs,
            candidate_moves: stats.candidate_moves,
            pruned_moves: stats.pruned_moves,
            qnodes: stats.qnodes,
            neural_inference_calls: stats.neural_inference_calls,
            neural_inference_time_ns: duration_ns(stats.neural_inference_time),
        }
    }
}

fn parse_canonical_sfen(value: &str) -> Result<Position, String> {
    if value.is_empty() || value.len() > 512 {
        return Err("SFEN must contain between 1 and 512 bytes".to_owned());
    }
    let position = parse_sfen(value).map_err(|error| format!("invalid SFEN: {error}"))?;
    if to_sfen(&position) != value {
        return Err("SFEN must use canonical spelling".to_owned());
    }
    Ok(position)
}

fn parse_bounded_usi_move(value: &str) -> Result<Move, String> {
    if value.is_empty() || value.len() > MAX_USI_MOVE_BYTES || !value.is_ascii() {
        return Err("USI move exceeds the browser boundary".to_owned());
    }
    parse_usi_move(value).map_err(|error| format!("invalid USI move: {error}"))
}

fn validate_sha256(value: &str) -> Result<(), String> {
    if value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        Ok(())
    } else {
        Err("expected artifact SHA-256 must be 64 lowercase hexadecimal characters".to_owned())
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut encoded = String::with_capacity(64);
    for byte in digest {
        write!(&mut encoded, "{byte:02x}").expect("writing to a String cannot fail");
    }
    encoded
}

fn hand_entries(position: &Position, side: Side) -> Vec<HandEntry> {
    HandPiece::DISPLAY_ORDER
        .into_iter()
        .map(|piece| HandEntry {
            piece: hand_piece_name(piece),
            count: position.hand(side).count(piece),
        })
        .collect()
}

const fn side_name(side: Side) -> &'static str {
    match side {
        Side::Black => "black",
        Side::White => "white",
    }
}

const fn piece_kind_name(kind: PieceKind) -> &'static str {
    match kind {
        PieceKind::Pawn => "pawn",
        PieceKind::Lance => "lance",
        PieceKind::Knight => "knight",
        PieceKind::Silver => "silver",
        PieceKind::Gold => "gold",
        PieceKind::Bishop => "bishop",
        PieceKind::Rook => "rook",
        PieceKind::King => "king",
        PieceKind::PromotedPawn => "promoted-pawn",
        PieceKind::PromotedLance => "promoted-lance",
        PieceKind::PromotedKnight => "promoted-knight",
        PieceKind::PromotedSilver => "promoted-silver",
        PieceKind::Horse => "horse",
        PieceKind::Dragon => "dragon",
    }
}

const fn hand_piece_name(piece: HandPiece) -> &'static str {
    match piece {
        HandPiece::Pawn => "pawn",
        HandPiece::Lance => "lance",
        HandPiece::Knight => "knight",
        HandPiece::Silver => "silver",
        HandPiece::Gold => "gold",
        HandPiece::Bishop => "bishop",
        HandPiece::Rook => "rook",
    }
}

const fn termination_name(termination: SearchTermination) -> &'static str {
    match termination {
        SearchTermination::Completed => "completed",
        SearchTermination::Stable => "stable",
        SearchTermination::NodeLimit => "node-limit",
        SearchTermination::TimeLimit => "time-limit",
        SearchTermination::Cancelled => "cancelled",
    }
}

const fn time_control_mode_name(mode: TimeControlMode) -> &'static str {
    match mode {
        TimeControlMode::Casual => "casual",
        TimeControlMode::MoveTime => "movetime",
        TimeControlMode::Clock => "clock",
        TimeControlMode::Nodes => "nodes",
        TimeControlMode::Depth => "depth",
        TimeControlMode::Infinite => "infinite",
    }
}

fn duration_ns(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

/// Browser-safe adapter around the shared native OSAVAL02 evaluator.
pub struct BrowserOsaval02Model {
    evaluator: Osaval02Evaluator,
}

impl BrowserOsaval02Model {
    /// Validate and retain one complete model byte string.
    ///
    /// # Errors
    ///
    /// Returns the shared strict parser's error for an invalid or incompatible artifact.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        Osaval02Evaluator::from_bytes(bytes)
            .map(|evaluator| Self { evaluator })
            .map_err(|error| error.to_string())
    }

    /// Serialize the complete validated model identity.
    ///
    /// # Errors
    ///
    /// Returns an error if the already bounded identity cannot be serialized.
    pub fn identity_json(&self) -> Result<String, String> {
        serde_json::to_string(self.evaluator.identity()).map_err(|error| error.to_string())
    }

    /// Evaluate an SFEN using only the closed, bounded history input.
    ///
    /// # Errors
    ///
    /// Returns an error for malformed SFEN/history, invalid history facts, or non-finite output.
    pub fn infer_json(&self, sfen: &str, history_json: Option<&str>) -> Result<String, String> {
        let history = match history_json {
            None | Some("") => Osaval02History::default(),
            Some(encoded) if encoded.len() <= MAX_RESTORE_JSON_BYTES => {
                serde_json::from_str(encoded)
                    .map_err(|error| format!("invalid history JSON: {error}"))?
            }
            Some(_) => return Err("history JSON exceeds the 16 KiB boundary".to_owned()),
        };
        let position = parse_sfen(sfen).map_err(|error| error.to_string())?;
        let inference = self
            .evaluator
            .infer(&position, history)
            .map_err(|error| error.to_string())?;
        serde_json::to_string(&inference).map_err(|error| error.to_string())
    }

    /// Run the same inference twice and return it only if serialization is identical.
    ///
    /// # Errors
    ///
    /// Returns an inference error or an explicit deterministic-repeat failure.
    pub fn deterministic_test_json(
        &self,
        sfen: &str,
        history_json: Option<&str>,
    ) -> Result<String, String> {
        let first = self.infer_json(sfen, history_json)?;
        let second = self.infer_json(sfen, history_json)?;
        if first != second {
            return Err("OSAVAL02 repeated inference is not deterministic".to_owned());
        }
        Ok(first)
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct WasmOsaval02Model {
    inner: BrowserOsaval02Model,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl WasmOsaval02Model {
    #[wasm_bindgen(constructor)]
    pub fn new(bytes: &[u8]) -> Result<Self, JsError> {
        BrowserOsaval02Model::from_bytes(bytes)
            .map(|inner| Self { inner })
            .map_err(|error| JsError::new(&error))
    }

    pub fn identity(&self) -> Result<String, JsError> {
        self.inner
            .identity_json()
            .map_err(|error| JsError::new(&error))
    }

    pub fn infer(&self, sfen: &str, history_json: Option<String>) -> Result<String, JsError> {
        self.inner
            .infer_json(sfen, history_json.as_deref())
            .map_err(|error| JsError::new(&error))
    }

    #[wasm_bindgen(js_name = deterministicTest)]
    pub fn deterministic_test(
        &self,
        sfen: &str,
        history_json: Option<String>,
    ) -> Result<String, JsError> {
        self.inner
            .deterministic_test_json(sfen, history_json.as_deref())
            .map_err(|error| JsError::new(&error))
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct WasmBrowserEngine {
    inner: BrowserEngine,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl WasmBrowserEngine {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: BrowserEngine::new(),
        }
    }

    pub fn snapshot(&self) -> Result<String, JsError> {
        self.inner
            .snapshot_json()
            .map_err(|error| JsError::new(&error))
    }

    pub fn reset(&mut self, sfen: Option<String>) -> Result<String, JsError> {
        self.inner
            .reset(sfen.as_deref())
            .map_err(|error| JsError::new(&error))
    }

    pub fn restore(&mut self, initial_sfen: &str, moves_json: &str) -> Result<String, JsError> {
        self.inner
            .restore(initial_sfen, moves_json)
            .map_err(|error| JsError::new(&error))
    }

    #[wasm_bindgen(js_name = loadModel)]
    pub fn load_model(
        &mut self,
        bytes: &[u8],
        expected_artifact_sha256: Option<String>,
    ) -> Result<String, JsError> {
        self.inner
            .load_model(bytes, expected_artifact_sha256.as_deref())
            .map_err(|error| JsError::new(&error))
    }

    #[wasm_bindgen(js_name = unloadModel)]
    pub fn unload_model(&mut self) -> Result<String, JsError> {
        self.inner
            .unload_model()
            .map_err(|error| JsError::new(&error))
    }

    #[wasm_bindgen(js_name = loadOpeningBook)]
    pub fn load_opening_book(
        &mut self,
        bytes: &[u8],
        expected_artifact_sha256: Option<String>,
    ) -> Result<String, JsError> {
        self.inner
            .load_opening_book(bytes, expected_artifact_sha256.as_deref())
            .map_err(|error| JsError::new(&error))
    }

    #[wasm_bindgen(js_name = unloadOpeningBook)]
    pub fn unload_opening_book(&mut self) -> Result<String, JsError> {
        self.inner
            .unload_opening_book()
            .map_err(|error| JsError::new(&error))
    }

    #[wasm_bindgen(js_name = configureOpening)]
    pub fn configure_opening(
        &mut self,
        profile: &str,
        max_plies: u32,
        minimum_sample_count: u64,
        maximum_teacher_loss_cp: i32,
    ) -> Result<String, JsError> {
        self.inner
            .configure_opening(
                profile,
                max_plies,
                minimum_sample_count,
                maximum_teacher_loss_cp,
            )
            .map_err(|error| JsError::new(&error))
    }

    #[wasm_bindgen(js_name = playMove)]
    pub fn play_move(&mut self, movement: &str) -> Result<String, JsError> {
        self.inner
            .play_move(movement)
            .map_err(|error| JsError::new(&error))
    }

    pub fn search(&self, profile: &str, evaluator: &str, multi_pv: u8) -> Result<String, JsError> {
        self.inner
            .search_json(profile, evaluator, multi_pv)
            .map_err(|error| JsError::new(&error))
    }

    #[wasm_bindgen(js_name = searchWithTimeControl)]
    pub fn search_with_time_control(
        &self,
        profile: &str,
        evaluator: &str,
        multi_pv: u8,
        time_control_json: &str,
    ) -> Result<String, JsError> {
        self.inner
            .search_time_control_json(profile, evaluator, multi_pv, time_control_json)
            .map_err(|error| JsError::new(&error))
    }

    #[wasm_bindgen(js_name = analysisStart)]
    pub fn analysis_start(
        &mut self,
        profile: &str,
        evaluator: &str,
        request_json: &str,
    ) -> Result<String, JsError> {
        self.inner
            .analysis_start_json(profile, evaluator, request_json)
            .map_err(|error| JsError::new(&error))
    }

    #[wasm_bindgen(js_name = analysisStep)]
    pub fn analysis_step(&mut self, request_json: &str) -> Result<String, JsError> {
        self.inner
            .analysis_step_json(request_json)
            .map_err(|error| JsError::new(&error))
    }

    #[wasm_bindgen(js_name = analysisStop)]
    pub fn analysis_stop(&mut self) -> Result<String, JsError> {
        self.inner
            .analysis_stop_json()
            .map_err(|error| JsError::new(&error))
    }

    #[wasm_bindgen(js_name = analysisWorkerFailed)]
    pub fn analysis_worker_failed(&mut self) -> Result<String, JsError> {
        self.inner
            .analysis_worker_failed_json()
            .map_err(|error| JsError::new(&error))
    }

    #[wasm_bindgen(js_name = analysisRestart)]
    pub fn analysis_restart(&mut self) -> Result<String, JsError> {
        self.inner
            .analysis_restart_json()
            .map_err(|error| JsError::new(&error))
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write as _;

    use flate2::{Compression, write::GzEncoder};
    use serde_json::Value;
    use sha2::{Digest, Sha256};

    use super::{BrowserEngine, MAX_BROWSER_MODEL_BYTES, engine_version};

    fn opening_book_bytes() -> Vec<u8> {
        let provenance = ["1".repeat(64), "2".repeat(64)];
        let mut record = serde_json::json!({
            "schema": open_shogi_core::OPENING_BOOK_SCHEMA,
            "stateKey": "eb5bc2ef917ec96fe2172f96d7060ec4f39322caf929fc1177d2c9fc8b937ebc",
            "stateSfen": "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b -",
            "ruleProfile": "standard-shogi/v1",
            "buildVersion": "wasm-fixture",
            "provenanceReferences": provenance,
            "candidates": [{
                "moveUsi": "2g2f", "sampleCount": 2,
                "sourceDistribution": {"aobazero-no-noise": 2},
                "blackResults": {"wins": 1, "losses": 1, "draws": 0, "unknown": 0},
                "whiteResults": {"wins": 0, "losses": 0, "draws": 0, "unknown": 0},
                "teacherScoreCp": 20, "scoreUncertaintyCp": null,
                "teacherDepth": 8, "teacherNodes": 25000,
                "openingClassification": "ibisha-vs-furibisha",
                "provenanceReferences": provenance
            }]
        });
        let checksum = format!("{:x}", Sha256::digest(serde_json::to_vec(&record).unwrap()));
        record
            .as_object_mut()
            .unwrap()
            .insert("recordChecksum".to_owned(), serde_json::json!(checksum));
        let mut encoder = GzEncoder::new(Vec::new(), Compression::fast());
        writeln!(encoder, "{}", serde_json::to_string(&record).unwrap()).unwrap();
        encoder.finish().unwrap()
    }

    #[test]
    fn engine_version_is_available() {
        assert!(!engine_version().is_empty());
    }

    #[test]
    fn start_position_snapshot_is_closed_and_replayable() {
        let engine = BrowserEngine::new();
        let snapshot: Value = serde_json::from_str(&engine.snapshot_json().unwrap()).unwrap();
        let board = snapshot["board"].as_array().unwrap();

        assert_eq!(snapshot["schema"], "open_shogi_browser_snapshot/v1");
        assert_eq!(snapshot["sideToMove"], "black");
        assert_eq!(board.len(), 81);
        assert_eq!(board.iter().filter(|piece| !piece.is_null()).count(), 40);
        assert_eq!(board[10]["kind"], "rook");
        assert_eq!(board[10]["side"], "white");
        assert_eq!(board[16]["kind"], "bishop");
        assert_eq!(board[16]["side"], "white");
        assert!(
            board[18..27]
                .iter()
                .all(|piece| { piece["kind"] == "pawn" && piece["side"] == "white" })
        );
        assert!(
            board[54..63]
                .iter()
                .all(|piece| { piece["kind"] == "pawn" && piece["side"] == "black" })
        );
        assert_eq!(board[64]["kind"], "bishop");
        assert_eq!(board[64]["side"], "black");
        assert_eq!(board[70]["kind"], "rook");
        assert_eq!(board[70]["side"], "black");
        assert_eq!(snapshot["legalMoves"].as_array().unwrap().len(), 30);
    }

    #[test]
    fn legal_move_and_restore_produce_the_same_position() {
        let mut played = BrowserEngine::new();
        let played_json = played.play_move("7g7f").unwrap();
        let played_snapshot: Value = serde_json::from_str(&played_json).unwrap();

        let mut restored = BrowserEngine::new();
        let restored_json = restored
            .restore(
                played_snapshot["initialSfen"].as_str().unwrap(),
                r#"["7g7f"]"#,
            )
            .unwrap();
        let restored_snapshot: Value = serde_json::from_str(&restored_json).unwrap();

        assert_eq!(played_snapshot["sfen"], restored_snapshot["sfen"]);
        assert_eq!(restored_snapshot["moves"], serde_json::json!(["7g7f"]));
    }

    #[test]
    fn noncanonical_sfen_and_illegal_moves_fail_closed() {
        let mut engine = BrowserEngine::new();

        assert!(engine.reset(Some("startpos")).is_err());
        assert!(engine.play_move("7g7e").is_err());
        assert!(
            engine
                .restore(&engine.initial_sfen.clone(), r#"["bad"]"#)
                .is_err()
        );
    }

    #[test]
    fn browser_search_uses_the_fixed_profile_budget() {
        let engine = BrowserEngine::new();
        let response: Value =
            serde_json::from_str(&engine.search_json("eco", "handcrafted", 3).unwrap()).unwrap();

        assert_eq!(response["schema"], "open_shogi_browser_search/v1");
        assert_eq!(response["profile"], "eco");
        assert!(response["nodes"].as_u64().unwrap() <= 1_500);
        assert!(response["bestMove"].as_str().is_some());
        assert_eq!(response["lines"].as_array().unwrap().len(), 3);
        assert!(engine.search_json("unbounded", "handcrafted", 1).is_err());
        assert!(engine.search_json("eco", "model", 1).is_err());
        assert!(engine.search_json("eco", "handcrafted", 4).is_err());
    }

    #[test]
    fn shared_time_control_schema_conforms_at_the_wasm_boundary() {
        let engine = BrowserEngine::new();
        let request = serde_json::json!({
            "schema": open_shogi_core::TIME_CONTROL_SCHEMA,
            "nodes": 750,
            "safetyMarginMs": 25,
        });
        let response: Value = serde_json::from_str(
            &engine
                .search_time_control_json("eco", "overall-champion", 2, &request.to_string())
                .unwrap(),
        )
        .unwrap();

        assert_eq!(
            response["timeControlSchema"],
            open_shogi_core::TIME_CONTROL_SCHEMA
        );
        assert_eq!(response["timeControlMode"], "nodes");
        assert_eq!(response["evaluator"], "overall-champion");
        assert!(response["nodes"].as_u64().unwrap() <= 750);

        let wrong_schema = serde_json::json!({"schema": "unknown/v1", "nodes": 1});
        assert!(
            engine
                .search_time_control_json("eco", "overall-champion", 1, &wrong_schema.to_string(),)
                .is_err()
        );
        let infinite = serde_json::json!({
            "schema": open_shogi_core::TIME_CONTROL_SCHEMA,
            "infinite": true,
        });
        assert!(
            engine
                .search_time_control_json("eco", "overall-champion", 1, &infinite.to_string(),)
                .is_err()
        );
    }

    #[test]
    fn wasm_book_hit_is_immediate_and_profile_controlled() {
        let mut engine = BrowserEngine::new();
        let bytes = opening_book_bytes();
        let expected = format!("{:x}", Sha256::digest(&bytes));
        let loaded: Value =
            serde_json::from_str(&engine.load_opening_book(&bytes, Some(&expected)).unwrap())
                .unwrap();
        assert_eq!(loaded["positions"], 1);
        assert_eq!(loaded["candidates"], 1);

        engine
            .configure_opening("ibisha_strict", 40, 2, 80)
            .unwrap();
        let response: Value =
            serde_json::from_str(&engine.search_json("eco", "overall-champion", 3).unwrap())
                .unwrap();
        assert_eq!(response["source"], "book");
        assert_eq!(response["bestMove"], "2g2f");
        assert_eq!(response["nodes"], 0);
        assert_eq!(response["elapsedNs"], 0);
        assert_eq!(response["openingBookMove"]["teacherNodes"], 25_000);
        assert_eq!(
            response["openingBookMove"]["openingClassification"],
            "ibisha-vs-furibisha"
        );

        engine.configure_opening("unrestricted", 40, 3, 80).unwrap();
        let fallback: Value =
            serde_json::from_str(&engine.search_json("eco", "overall-champion", 1).unwrap())
                .unwrap();
        assert_eq!(fallback["source"], "search");
        assert!(fallback["nodes"].as_u64().unwrap() > 0);
    }

    #[test]
    fn wasm_analysis_protocol_switches_positions_reuses_cache_and_restarts() {
        let mut engine = BrowserEngine::new();
        let hash = |digit: char| digit.to_string().repeat(64);
        let start_request = |position_sfen: String| {
            serde_json::json!({
                "schema": open_shogi_core::ANALYSIS_SCHEMA,
                "positionSfen": position_sfen,
                "modelHash": hash('1'),
                "evaluatorConfigHash": hash('2'),
                "featureSchemaHash": hash('3'),
                "evaluationSemanticsHash": hash('4'),
                "searchOptionsHash": hash('5'),
                "openingProfileHash": hash('6'),
                "multiPv": 2,
            })
        };
        let first = open_shogi_core::Position::startpos();
        let first_sfen = open_shogi_core::to_sfen(&first);
        let started: Value = serde_json::from_str(
            &engine
                .analysis_start_json(
                    "eco",
                    "overall-champion",
                    &start_request(first_sfen.clone()).to_string(),
                )
                .unwrap(),
        )
        .unwrap();
        assert_eq!(started["event"], "started");
        assert_eq!(started["updates"].as_array().unwrap().len(), 0);

        let step = serde_json::json!({
            "schema": open_shogi_core::ANALYSIS_SCHEMA,
            "nodes": 1_500,
            "maxDepth": 3,
            "timestampMs": 9,
        });
        let updates: Value =
            serde_json::from_str(&engine.analysis_step_json(&step.to_string()).unwrap()).unwrap();
        assert_eq!(updates["event"], "updates");
        assert!(!updates["updates"].as_array().unwrap().is_empty());
        assert_eq!(updates["updates"][0]["modelHash"], hash('1'));

        let mut second = first;
        second.make_move(second.legal_moves()[0]).unwrap();
        engine
            .analysis_start_json(
                "eco",
                "overall-champion",
                &start_request(open_shogi_core::to_sfen(&second)).to_string(),
            )
            .unwrap();
        let cached: Value = serde_json::from_str(
            &engine
                .analysis_start_json(
                    "eco",
                    "overall-champion",
                    &start_request(first_sfen).to_string(),
                )
                .unwrap(),
        )
        .unwrap();
        assert_eq!(cached["updates"][0]["source"], "cache");

        assert!(
            engine
                .analysis_worker_failed_json()
                .unwrap()
                .contains("worker-failed")
        );
        assert!(
            engine
                .analysis_restart_json()
                .unwrap()
                .contains("restarted")
        );
        assert!(engine.analysis_stop_json().unwrap().contains("stopped"));
        assert!(engine.analysis_step_json(&step.to_string()).is_err());
    }

    #[test]
    fn model_loader_rejects_oversized_and_corrupt_artifacts() {
        let mut engine = BrowserEngine::new();
        let oversized = vec![0; MAX_BROWSER_MODEL_BYTES + 1];

        assert!(engine.load_model(&oversized, None).is_err());
        assert!(engine.load_model(b"not-an-osaval-model", None).is_err());
        assert!(
            engine
                .load_model(b"not-an-osaval-model", Some("bad"))
                .is_err()
        );
        assert!(engine.unload_model().is_ok());
    }
}
