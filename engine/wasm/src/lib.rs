//! Browser-facing, single-worker adapter for the independently implemented engine core.
//!
//! The public JavaScript boundary accepts only bounded strings, model bytes, and closed profile
//! names. Search remains synchronous inside a dedicated Web Worker; the page cancels work by
//! terminating that worker, so this crate does not require shared memory or browser threads.

#![forbid(unsafe_code)]

use std::{fmt::Write as _, sync::Arc, time::Duration};

use open_shogi_core::{
    CancellationToken, EngineIdentity, EnteringKingDeclaration, Game, GameEnd, HandPiece,
    ImpasseOutcome, Move, NeuralActivation, NeuralEvaluator, NeuralQuantization, PieceKind,
    Position, RepetitionOutcome, SearchConfig, SearchEngine, SearchLimits, SearchResult,
    SearchStats, SearchTermination, Side, Square, parse_sfen, parse_usi_move, to_sfen, to_usi_move,
};
use serde::Serialize;
use sha2::{Digest, Sha256};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

const SNAPSHOT_SCHEMA: &str = "open_shogi_browser_snapshot/v1";
const SEARCH_SCHEMA: &str = "open_shogi_browser_search/v1";
const MODEL_SCHEMA: &str = "open_shogi_browser_model/v1";
const MAX_BROWSER_MODEL_BYTES: usize = 16 * 1024 * 1024;
const MAX_RESTORE_JSON_BYTES: usize = 16 * 1024;
const MAX_RESTORE_MOVES: usize = 512;
const MAX_USI_MOVE_BYTES: usize = 8;

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
    Handcrafted,
    Model,
}

impl EvaluatorChoice {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "handcrafted" => Ok(Self::Handcrafted),
            "model" => Ok(Self::Model),
            _ => Err("evaluator must be handcrafted or model".to_owned()),
        }
    }

    const fn name(self) -> &'static str {
        match self {
            Self::Handcrafted => "handcrafted",
            Self::Model => "model",
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
}

impl Default for BrowserEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl BrowserEngine {
    /// Starts at the standard position with the handcrafted evaluator selected.
    #[must_use]
    pub fn new() -> Self {
        let position = Position::startpos();
        Self {
            initial_sfen: to_sfen(&position),
            game: Game::new(position),
            model: None,
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
        self.snapshot_json()
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
        let lines = self.multi_pv_lines(profile, evaluator, config, &result, multi_pv)?;
        let response = SearchResponse::new(
            profile,
            evaluator,
            self.game.position().side_to_move(),
            result,
            lines,
        );
        serde_json::to_string(&response).map_err(|error| error.to_string())
    }

    fn search_engine(
        &self,
        config: SearchConfig,
        evaluator: EvaluatorChoice,
    ) -> Result<SearchEngine, String> {
        match evaluator {
            EvaluatorChoice::Handcrafted => Ok(SearchEngine::new(config)),
            EvaluatorChoice::Model => {
                let model = self
                    .model
                    .as_ref()
                    .ok_or_else(|| "model evaluator selected without a loaded model".to_owned())?;
                Ok(SearchEngine::with_neural(
                    config,
                    Arc::clone(&model.evaluator),
                ))
            }
        }
    }

    fn multi_pv_lines(
        &self,
        profile: BrowserSearchProfile,
        evaluator: EvaluatorChoice,
        config: SearchConfig,
        principal: &SearchResult,
        multi_pv: u8,
    ) -> Result<Vec<SearchLineSummary>, String> {
        let Some(principal_move) = principal.best_move else {
            return Ok(Vec::new());
        };
        let mut lines = vec![SearchLineSummary::principal(principal)];
        if multi_pv == 1 {
            return Ok(lines);
        }

        let legal_moves = self.game.position().legal_moves();
        let divisor = u64::try_from(legal_moves.len()).unwrap_or(u64::MAX).max(1);
        let per_root_nodes = profile.max_nodes().checked_div(divisor).unwrap_or(1).max(1);
        let mut comparison_engine = self.search_engine(config, evaluator)?;
        let mut alternatives = Vec::with_capacity(legal_moves.len().saturating_sub(1));
        for movement in legal_moves {
            if movement == principal_move {
                continue;
            }
            let mut position = self.game.position().clone();
            position
                .make_move(movement)
                .map_err(|error| format!("generated root move became illegal: {error}"))?;
            let result = comparison_engine.search(
                &position,
                SearchLimits {
                    max_depth: profile.max_depth().saturating_sub(1).max(1),
                    max_nodes: Some(per_root_nodes),
                    movetime: None,
                },
                &CancellationToken::new(),
            );
            alternatives.push(SearchLineSummary::alternative(movement, result));
        }
        alternatives.sort_by(|left, right| {
            right
                .score_cp
                .cmp(&left.score_cp)
                .then_with(|| left.best_move.cmp(&right.best_move))
        });
        alternatives.truncate(usize::from(multi_pv.saturating_sub(1)));
        lines.extend(alternatives);
        for (index, line) in lines.iter_mut().enumerate() {
            line.rank = u8::try_from(index + 1).unwrap_or(u8::MAX);
        }
        Ok(lines)
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
    profile: &'static str,
    evaluator: &'static str,
    perspective: &'static str,
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
}

impl SearchResponse {
    fn new(
        profile: BrowserSearchProfile,
        evaluator: EvaluatorChoice,
        perspective: Side,
        result: open_shogi_core::SearchResult,
        lines: Vec<SearchLineSummary>,
    ) -> Self {
        Self {
            schema: SEARCH_SCHEMA,
            profile: profile.name(),
            evaluator: evaluator.name(),
            perspective: side_name(perspective),
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

    fn alternative(root: Move, result: SearchResult) -> Self {
        let mut pv = Vec::with_capacity(result.pv.len() + 1);
        pv.push(to_usi_move(root));
        pv.extend(result.pv.into_iter().map(to_usi_move));
        Self {
            rank: 0,
            best_move: to_usi_move(root),
            score_cp: result.score.saturating_neg(),
            depth: result.depth.saturating_add(1),
            seldepth: result.seldepth.saturating_add(1),
            nodes: result.nodes,
            pv,
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
        SearchTermination::NodeLimit => "node-limit",
        SearchTermination::TimeLimit => "time-limit",
        SearchTermination::Cancelled => "cancelled",
    }
}

fn duration_ns(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
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
}

#[cfg(test)]
mod tests {
    use serde_json::Value;

    use super::{BrowserEngine, MAX_BROWSER_MODEL_BYTES, engine_version};

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
