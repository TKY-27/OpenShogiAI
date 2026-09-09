//! Phase 10V browser contract. Rules and protocol serializers mirror the development adapter,
//! while evaluation is constructed exclusively from an explicitly hash-bound OSAVAL03 model.
use open_shogi_core::{
    ANALYSIS_SCHEMA, AnalysisCacheKey, AnalysisService, AnalysisUpdate, AnalysisUpdateSource,
    CancellationToken, EngineIdentity, EnteringKingDeclaration, Game, GameEnd, HandPiece,
    ImpasseOutcome, Move, PieceKind, Position, PurePlayingEvaluator, RepetitionOutcome,
    RuntimeProofCounters, SearchConfig, SearchEngine, SearchLimits, SearchResult, SearchStats,
    SearchTermination, Side, Square, TimeControl, TimeControlMode, TimeManager, parse_sfen,
    parse_usi_move, to_sfen, to_usi_move,
};
use serde::{Deserialize, Serialize};
use std::time::Duration;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
const SNAPSHOT_SCHEMA: &str = "open_shogi_browser_snapshot/v1";
const SEARCH_SCHEMA: &str = "open_shogi_browser_search/v1";
const MAX_RESTORE_JSON_BYTES: usize = 16 * 1024;
const MAX_RESTORE_MOVES: usize = 512;
const MAX_USI_MOVE_BYTES: usize = 8;
const MAX_BROWSER_MODEL_BYTES: usize = 64 * 1024 * 1024;
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
    PureLearned,
}
impl EvaluatorChoice {
    fn parse(value: &str) -> Result<Self, String> {
        if value == "pure_learned" {
            Ok(Self::PureLearned)
        } else {
            Err("pure-only browser requires pure_learned".into())
        }
    }
    const fn name(self) -> &'static str {
        match self {
            Self::PureLearned => "pure_learned",
        }
    }
}
/// Browser state may be inspected before model loading; all evaluation fails closed until loading.
pub struct BrowserEngine {
    initial_sfen: String,
    game: Game,
    model: Option<PurePlayingEvaluator>,
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
    /// Starts in an unloaded state at the standard position; search requires a model.
    #[must_use]
    pub fn new() -> Self {
        let position = Position::startpos();
        Self {
            initial_sfen: to_sfen(&position),
            game: Game::new(position),
            model: None,
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
        self.clear_analysis();
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
        self.clear_analysis();
        self.snapshot_json()
    }

    /// Loads only OSAVAL03 and requires the expected whole-file SHA-256.
    /// # Errors
    /// Rejects missing hashes, incompatible formats, malformed weights and hash mismatches.
    pub fn load_model(
        &mut self,
        bytes: &[u8],
        expected_artifact_sha256: Option<&str>,
    ) -> Result<String, String> {
        // An unsuccessful replacement must not leave old weights available for accidental play.
        self.model = None;
        self.clear_analysis();
        let hash = expected_artifact_sha256
            .ok_or("pure-only browser requires an explicit model SHA-256")?;
        if bytes.len() > MAX_BROWSER_MODEL_BYTES {
            return Err("browser model exceeds 64 MiB".into());
        }
        let model = PurePlayingEvaluator::from_bytes("OSAVAL03", bytes, hash)?;
        self.model = Some(model);
        Ok(self.model_summary().to_string())
    }
    /// Removes model weights and invalidates all active analysis.
    /// # Errors
    /// Returns a snapshot serialization error.
    pub fn unload_model(&mut self) -> Result<String, String> {
        self.model = None;
        self.clear_analysis();
        self.snapshot_json()
    }
    /// Opening books are unavailable in the pure-only build.
    /// # Errors
    /// Always rejects because this build has no opening-book implementation.
    pub fn load_opening_book(
        &mut self,
        _bytes: &[u8],
        _hash: Option<&str>,
    ) -> Result<String, String> {
        Err("opening books are unavailable in pure-only builds".into())
    }
    /// Opening books are unavailable in the pure-only build.
    /// # Errors
    /// Always rejects because this build has no opening-book implementation.
    pub fn unload_opening_book(&mut self) -> Result<String, String> {
        Err("opening books are unavailable in pure-only builds".into())
    }
    /// Opening configuration is unavailable in the pure-only build.
    /// # Errors
    /// Always rejects because this build has no opening policy implementation.
    pub fn configure_opening(
        &mut self,
        _profile: &str,
        _max_plies: u32,
        _minimum_sample_count: u64,
        _maximum_teacher_loss_cp: i32,
    ) -> Result<String, String> {
        Err("opening policy is unavailable in pure-only builds".into())
    }
    /// Applies one legal move and invalidates analysis for the old position.
    /// # Errors
    /// Rejects malformed, illegal and post-terminal moves.
    pub fn play_move(&mut self, movement: &str) -> Result<String, String> {
        if self.game.end().is_some() {
            return Err("the game has already ended".to_owned());
        }
        let parsed = parse_bounded_usi_move(movement)?;
        self.game
            .play(parsed)
            .map_err(|error| format!("move is not legal: {error}"))?;
        self.clear_analysis();
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
        ensure_search_success(&result)?;
        let runtime_proof = self.runtime_proof(&engine, evaluator, result.stats);
        let lines = Self::multi_pv_lines(&result, multi_pv);
        let response = SearchResponse::new(
            profile,
            evaluator,
            self.game.position().side_to_move(),
            result,
            lines,
            "profile-nodes",
            runtime_proof,
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
        let plan = browser_time_plan(self.game.position().side_to_move(), request, profile)?;
        let mode = time_control_mode_name(plan.mode);

        let config = SearchConfig {
            transposition_entries: SearchEngine::transposition_entries_for_megabytes(
                profile.transposition_megabytes(),
            ),
            quiescence_depth: profile.quiescence_depth(),
            ..SearchConfig::default()
        };
        let mut engine = self.search_engine(config, evaluator)?;
        let result = engine.search_managed(self.game.position(), plan, &CancellationToken::new());
        ensure_search_success(&result)?;
        let runtime_proof = self.runtime_proof(&engine, evaluator, result.stats);
        let lines = Self::multi_pv_lines(&result, multi_pv);
        let response = SearchResponse::new(
            profile,
            evaluator,
            self.game.position().side_to_move(),
            result,
            lines,
            mode,
            runtime_proof,
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
        let model = self
            .model
            .as_ref()
            .ok_or("pure-only browser has no loaded model")?;
        if request.model_hash != model.artifact_sha256() {
            return Err("analysis model hash does not match loaded model".into());
        }
        if request.position_sfen != to_sfen(self.game.position()) {
            return Err("analysis root must match the restored game position".into());
        }
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
        if request.max_depth > profile.max_depth() {
            return Err("analysis depth exceeds the selected browser profile limit".into());
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
        ensure_search_success(&step.result)?;
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
        _evaluator: EvaluatorChoice,
    ) -> Result<SearchEngine, String> {
        let model = self
            .model
            .as_ref()
            .ok_or("pure-only browser has no loaded model")?;
        let mut engine = model.search_engine(config, model.artifact_sha256())?;
        engine.set_pure_history(
            &parse_canonical_sfen(&self.initial_sfen)?,
            self.game.moves(),
        )?;
        Ok(engine)
    }
    fn runtime_proof(
        &self,
        engine: &SearchEngine,
        _evaluator: EvaluatorChoice,
        stats: SearchStats,
    ) -> Option<RuntimeProofCounters> {
        Some(engine.runtime_proof(stats, self.model.as_ref()?.artifact_sha256().to_owned()))
    }
    fn clear_analysis(&mut self) {
        self.analysis = None;
        self.analysis_identity = None;
        self.analysis_side = None;
        self.analysis_profile = None;
    }
    fn model_summary(&self) -> serde_json::Value {
        self.model
            .as_ref()
            .map_or(serde_json::Value::Null, |model| {
                serde_json::json!({
                  "schema":"open_shogi_browser_model/v1", "modelFormat":model.format(),
                  "artifactSha256":model.artifact_sha256(), "expectedHashVerified":true,
                  "buildClass":"pure-only", "evaluationMode":"pure-value"
                })
            })
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
                    "model-required"
                },
                model: self.model_summary(),
            },
            opening_book: None,
            opening_policy: serde_json::json!({"profile":"disabled", "maxPlies":0, "minimumSampleCount":0, "maximumTeacherLossCp":0}),
            build_class: "pure-only",
            compiled_evaluators: open_shogi_core::COMPILED_EVALUATORS,
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
    opening_book: Option<serde_json::Value>,
    opening_policy: serde_json::Value,
    build_class: &'static str,
    compiled_evaluators: &'static [&'static str],
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
    model: serde_json::Value,
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
    runtime_proof: Option<RuntimeProofCounters>,
    #[serde(skip_serializing_if = "Option::is_none")]
    opening_book_move: Option<serde_json::Value>,
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
        runtime_proof: Option<RuntimeProofCounters>,
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
            runtime_proof,
            opening_book_move: None,
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
    osaval02_inference_errors: u64,
    learned_eval_calls: u64,
    handcrafted_eval_calls: u64,
    residual_eval_calls: u64,
    composite_eval_calls: u64,
    fallback_count: u64,
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
            osaval02_inference_errors: stats.osaval02_inference_errors,
            learned_eval_calls: stats.learned_eval_calls,
            handcrafted_eval_calls: stats.handcrafted_eval_calls,
            residual_eval_calls: stats.residual_eval_calls,
            composite_eval_calls: stats.composite_eval_calls,
            fallback_count: stats.fallback_count,
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
        SearchTermination::EvaluationError => "evaluation-error",
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

fn browser_time_plan(
    side: Side,
    request: TimeControl,
    profile: BrowserSearchProfile,
) -> Result<open_shogi_core::TimePlan, String> {
    if request
        .nodes
        .is_some_and(|nodes| nodes > profile.max_nodes())
    {
        return Err(format!(
            "nodes exceeds the selected browser profile limit {}",
            profile.max_nodes()
        ));
    }
    if request
        .depth
        .is_some_and(|depth| depth > profile.max_depth())
    {
        return Err(format!(
            "depth exceeds the selected browser profile limit {}",
            profile.max_depth()
        ));
    }
    let mut plan = TimeManager::default().plan(side, request, profile.max_depth())?;
    // Depth-only requests otherwise have neither a deadline nor a node ceiling.
    if plan.hard_limit.is_none() && plan.max_nodes.is_none() {
        plan.max_nodes = Some(profile.max_nodes());
    }
    Ok(plan)
}

fn ensure_search_success(result: &SearchResult) -> Result<(), String> {
    if result.termination == SearchTermination::EvaluationError {
        Err("pure-only inference failed; no result is available".into())
    } else {
        Ok(())
    }
}
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
#[derive(Default)]
pub struct WasmBrowserEngine {
    inner: BrowserEngine,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
// Optional owned strings are required by the wasm-bindgen ABI.
#[allow(clippy::needless_pass_by_value)]
impl WasmBrowserEngine {
    #[must_use]
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: BrowserEngine::new(),
        }
    }

    /// Forwards to the validated browser-state boundary.
    /// # Errors
    /// Returns the underlying protocol, state, model or search validation error.
    pub fn snapshot(&self) -> Result<String, JsError> {
        self.inner
            .snapshot_json()
            .map_err(|error| JsError::new(&error))
    }

    /// Forwards to the validated browser-state boundary.
    /// # Errors
    /// Returns the underlying protocol, state, model or search validation error.
    pub fn reset(&mut self, sfen: Option<String>) -> Result<String, JsError> {
        self.inner
            .reset(sfen.as_deref())
            .map_err(|error| JsError::new(&error))
    }

    /// Forwards to the validated browser-state boundary.
    /// # Errors
    /// Returns the underlying protocol, state, model or search validation error.
    pub fn restore(&mut self, initial_sfen: &str, moves_json: &str) -> Result<String, JsError> {
        self.inner
            .restore(initial_sfen, moves_json)
            .map_err(|error| JsError::new(&error))
    }

    #[wasm_bindgen(js_name = loadModel)]
    /// Forwards to the validated browser-state boundary.
    /// # Errors
    /// Returns the underlying protocol, state, model or search validation error.
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
    /// Forwards to the validated browser-state boundary.
    /// # Errors
    /// Returns the underlying protocol, state, model or search validation error.
    pub fn unload_model(&mut self) -> Result<String, JsError> {
        self.inner
            .unload_model()
            .map_err(|error| JsError::new(&error))
    }

    #[wasm_bindgen(js_name = loadOpeningBook)]
    /// Forwards to the validated browser-state boundary.
    /// # Errors
    /// Returns the underlying protocol, state, model or search validation error.
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
    /// Forwards to the validated browser-state boundary.
    /// # Errors
    /// Returns the underlying protocol, state, model or search validation error.
    pub fn unload_opening_book(&mut self) -> Result<String, JsError> {
        self.inner
            .unload_opening_book()
            .map_err(|error| JsError::new(&error))
    }

    #[wasm_bindgen(js_name = configureOpening)]
    /// Forwards to the validated browser-state boundary.
    /// # Errors
    /// Returns the underlying protocol, state, model or search validation error.
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
    /// Forwards to the validated browser-state boundary.
    /// # Errors
    /// Returns the underlying protocol, state, model or search validation error.
    pub fn play_move(&mut self, movement: &str) -> Result<String, JsError> {
        self.inner
            .play_move(movement)
            .map_err(|error| JsError::new(&error))
    }

    /// Forwards to the validated browser-state boundary.
    /// # Errors
    /// Returns the underlying protocol, state, model or search validation error.
    pub fn search(&self, profile: &str, evaluator: &str, multi_pv: u8) -> Result<String, JsError> {
        self.inner
            .search_json(profile, evaluator, multi_pv)
            .map_err(|error| JsError::new(&error))
    }

    #[wasm_bindgen(js_name = searchWithTimeControl)]
    /// Forwards to the validated browser-state boundary.
    /// # Errors
    /// Returns the underlying protocol, state, model or search validation error.
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
    /// Forwards to the validated browser-state boundary.
    /// # Errors
    /// Returns the underlying protocol, state, model or search validation error.
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
    /// Forwards to the validated browser-state boundary.
    /// # Errors
    /// Returns the underlying protocol, state, model or search validation error.
    pub fn analysis_step(&mut self, request_json: &str) -> Result<String, JsError> {
        self.inner
            .analysis_step_json(request_json)
            .map_err(|error| JsError::new(&error))
    }

    #[wasm_bindgen(js_name = analysisStop)]
    /// Forwards to the validated browser-state boundary.
    /// # Errors
    /// Returns the underlying protocol, state, model or search validation error.
    pub fn analysis_stop(&mut self) -> Result<String, JsError> {
        self.inner
            .analysis_stop_json()
            .map_err(|error| JsError::new(&error))
    }

    #[wasm_bindgen(js_name = analysisWorkerFailed)]
    /// Forwards to the validated browser-state boundary.
    /// # Errors
    /// Returns the underlying protocol, state, model or search validation error.
    pub fn analysis_worker_failed(&mut self) -> Result<String, JsError> {
        self.inner
            .analysis_worker_failed_json()
            .map_err(|error| JsError::new(&error))
    }

    #[wasm_bindgen(js_name = analysisRestart)]
    /// Forwards to the validated browser-state boundary.
    /// # Errors
    /// Returns the underlying protocol, state, model or search validation error.
    pub fn analysis_restart(&mut self) -> Result<String, JsError> {
        self.inner
            .analysis_restart_json()
            .map_err(|error| JsError::new(&error))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};
    use sha2::{Digest, Sha256};
    use std::sync::OnceLock;

    fn fixture() -> &'static (Vec<u8>, String) {
        static MODEL: OnceLock<(Vec<u8>, String)> = OnceLock::new();
        MODEL.get_or_init(|| {
            let mut bytes = b"OSAVAL03".to_vec();
            for value in [3_u32, 1, 8427, 256, 16, 4] {
                bytes.extend(value.to_le_bytes());
            }
            bytes.extend(17_u64.to_le_bytes());
            bytes.extend(1.0_f32.to_le_bytes());
            let parameters = 8427 * 256 + 256 + 3 * 256 * 16 + 16 + 16 * 4 + 4;
            bytes.resize(44 + parameters * 4, 0);
            bytes.extend_from_slice(&Sha256::digest(&bytes));
            let hash = format!("{:x}", Sha256::digest(&bytes));
            (bytes, hash)
        })
    }

    fn loaded() -> BrowserEngine {
        let mut browser = BrowserEngine::new();
        let (bytes, hash) = fixture();
        browser.load_model(bytes, Some(hash)).unwrap();
        browser
    }

    fn snapshot(browser: &BrowserEngine) -> Value {
        serde_json::from_str(&browser.snapshot_json().unwrap()).unwrap()
    }

    #[test]
    fn browser_profiles_bound_depth_only_requests() {
        let request = TimeControl {
            depth: Some(5),
            casual: false,
            ..TimeControl::casual()
        };
        let plan = browser_time_plan(Side::Black, request, BrowserSearchProfile::Eco).unwrap();
        assert_eq!(plan.max_nodes, Some(1_500));
        assert_eq!(plan.max_depth, 5);
        assert!(plan.hard_limit.is_none());
        let excessive = TimeControl {
            depth: Some(64),
            ..request
        };
        assert!(browser_time_plan(Side::Black, excessive, BrowserSearchProfile::Eco).is_err());
    }

    #[test]
    fn unloaded_contract_is_inspectable_but_cannot_evaluate() {
        let mut browser = BrowserEngine::new();
        let state = snapshot(&browser);
        assert_eq!(state["schema"], "open_shogi_browser_snapshot/v1");
        assert_eq!(state["board"].as_array().unwrap().len(), 81);
        assert_eq!(state["legalMoves"].as_array().unwrap().len(), 30);
        assert_eq!(state["buildClass"], "pure-only");
        assert_eq!(state["evaluator"]["kind"], "model-required");
        assert!(browser.search_json("eco", "pure_learned", 1).is_err());
        assert!(browser.load_model(&[], None).is_err());
        assert!(browser.load_opening_book(&[], None).is_err());
        assert!(browser.unload_opening_book().is_err());
        assert!(browser.configure_opening("off", 0, 0, 0).is_err());
        assert!(browser.analysis_stop_json().is_err());
    }

    #[test]
    fn loading_is_hash_bound_and_failed_replacement_closes_search() {
        let mut browser = loaded();
        let (_, hash) = fixture();
        assert_eq!(
            snapshot(&browser)["evaluator"]["model"]["artifactSha256"],
            *hash
        );
        assert!(browser.search_json("eco", "overall-champion", 1).is_err());
        assert!(
            browser
                .load_model(&fixture().0, Some(&"0".repeat(64)))
                .is_err()
        );
        assert!(browser.search_json("eco", "pure_learned", 1).is_err());
        browser.load_model(&fixture().0, Some(hash)).unwrap();
        browser.unload_model().unwrap();
        assert!(browser.search_json("eco", "pure_learned", 1).is_err());
    }

    #[test]
    fn legal_history_search_and_atomic_restore_preserve_contract() {
        let mut browser = loaded();
        let start = snapshot(&browser)["sfen"].as_str().unwrap().to_owned();
        browser.play_move("7g7f").unwrap();
        let moved = snapshot(&browser);
        assert_eq!(moved["sideToMove"], "white");
        assert_eq!(moved["moves"], json!(["7g7f"]));
        assert!(browser.restore(&start, r#"["7g7f","7g7f"]"#).is_err());
        assert_eq!(snapshot(&browser), moved);
        browser.restore(&start, r#"["7g7f"]"#).unwrap();
        let result: Value = serde_json::from_str(
            &browser
                .search_time_control_json(
                    "eco",
                    "pure_learned",
                    3,
                    &json!({"schema":open_shogi_core::TIME_CONTROL_SCHEMA,"nodes":100,"depth":1})
                        .to_string(),
                )
                .unwrap(),
        )
        .unwrap();
        assert_eq!(result["schema"], "open_shogi_browser_search/v1");
        assert_eq!(result["evaluator"], "pure_learned");
        assert_eq!(result["perspective"], "white");
        let movement = result["bestMove"].as_str().unwrap();
        assert!(
            moved["legalMoves"]
                .as_array()
                .unwrap()
                .iter()
                .any(|legal| legal["usi"] == movement)
        );
        assert_eq!(result["stats"]["handcraftedEvalCalls"], 0);
        assert_eq!(result["stats"]["fallbackCount"], 0);
        assert!(result["runtimeProof"].is_object());
    }

    #[test]
    fn analysis_identity_lifecycle_and_position_invalidation() {
        let mut browser = loaded();
        let root = snapshot(&browser)["sfen"].as_str().unwrap().to_owned();
        let hash = &fixture().1;
        let mut request = json!({"schema":ANALYSIS_SCHEMA,"positionSfen":root,
            "modelHash":hash,"evaluatorConfigHash":hash, "featureSchemaHash":hash,
            "evaluationSemanticsHash":hash, "searchOptionsHash":hash, "openingProfileHash":hash, "multiPv":1});
        request["modelHash"] = json!("wrong");
        assert!(
            browser
                .analysis_start_json("eco", "pure_learned", &request.to_string())
                .is_err()
        );
        request["modelHash"] = json!(hash);
        browser
            .analysis_start_json("eco", "pure_learned", &request.to_string())
            .unwrap();
        assert!(browser.analysis_restart_json().is_err());
        assert!(
            browser
                .analysis_step_json(
                    &json!({"schema":ANALYSIS_SCHEMA,"nodes":100,"maxDepth":64,"timestampMs":10})
                        .to_string()
                )
                .is_err()
        );
        browser
            .analysis_step_json(
                &json!({"schema":ANALYSIS_SCHEMA,"nodes":100,"maxDepth":1,"timestampMs":10})
                    .to_string(),
            )
            .unwrap();
        browser.analysis_worker_failed_json().unwrap();
        browser.analysis_restart_json().unwrap();
        browser.analysis_stop_json().unwrap();
        browser.play_move("7g7f").unwrap();
        assert!(browser.analysis_step_json("{}").is_err());
        assert!(
            browser
                .analysis_start_json("eco", "pure_learned", &request.to_string())
                .is_err()
        );
    }
}
