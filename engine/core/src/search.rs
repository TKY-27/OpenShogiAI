//! Deterministic single-threaded baseline search.

use std::{
    mem::size_of,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};
use web_time::Instant;

use crate::{
    Move, Osaval02SearchAdapter, PieceKind, Position, RuntimeProfile, RuntimeProofCounters, Side,
    TimePlan,
};

#[cfg(feature = "handcrafted")]
use crate::{EvaluationConfig, NeuralEvaluator, evaluate};

/// Base score for forced rule wins/losses, including mate, no legal moves and perpetual check.
/// Distance in plies is subtracted; the root outcome distinguishes the adjudication reason.
pub const MATE_SCORE: i32 = 30_000;
/// Scores beyond this threshold encode a forced rule win/loss rather than a static evaluation.
pub const MATE_THRESHOLD: i32 = MATE_SCORE - 1_000;

const INFINITY: i32 = 32_000;
const MAX_SEARCH_PLY: usize = 256;
const MOVE_BUCKETS: usize = 13_689;

/// Returns whether a score is in the forced rule win/loss namespace.
#[must_use]
pub const fn is_mate_score(score: i32) -> bool {
    score >= MATE_THRESHOLD || score <= -MATE_THRESHOLD
}

/// Cooperative cancellation shared between a search thread and its controller.
#[derive(Clone, Debug, Default)]
pub struct CancellationToken {
    cancelled: Arc<AtomicBool>,
}

/// Monotonic clock used by search deadlines and measured inference time.
///
/// Tests can supply a deterministic implementation; production uses `web_time::Instant` on
/// native and Wasm hosts.
pub trait MonotonicClock: Send + Sync {
    fn now(&self) -> Duration;
}

/// Production monotonic clock with an engine-local zero point.
#[derive(Debug)]
pub struct SystemMonotonicClock {
    origin: Instant,
}

impl Default for SystemMonotonicClock {
    fn default() -> Self {
        Self {
            origin: Instant::now(),
        }
    }
}

impl MonotonicClock for SystemMonotonicClock {
    fn now(&self) -> Duration {
        self.origin.elapsed()
    }
}

impl CancellationToken {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Requests cancellation. The flag remains set for the lifetime of this token.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }
}

/// Hard limits for one deterministic search.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SearchLimits {
    pub max_depth: u8,
    pub max_nodes: Option<u64>,
    pub movetime: Option<Duration>,
}

impl Default for SearchLimits {
    fn default() -> Self {
        Self {
            max_depth: 4,
            max_nodes: None,
            movetime: None,
        }
    }
}

/// Search ladder switches and baseline evaluation configuration.
#[expect(
    clippy::struct_excessive_bools,
    reason = "the roadmap requires each search rung to remain independently switchable"
)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SearchConfig {
    #[cfg(feature = "handcrafted")]
    pub evaluation: EvaluationConfig,
    pub runtime_profile: RuntimeProfile,
    pub transposition_entries: usize,
    pub quiescence_depth: u8,
    pub aspiration_window: i32,
    pub enable_alpha_beta: bool,
    pub enable_iterative_deepening: bool,
    pub enable_transposition_table: bool,
    pub enable_move_ordering: bool,
    pub enable_killers: bool,
    pub enable_history: bool,
    pub enable_quiescence: bool,
    pub enable_pvs: bool,
    pub enable_aspiration: bool,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            #[cfg(feature = "handcrafted")]
            evaluation: EvaluationConfig::default(),
            runtime_profile: if cfg!(feature = "pure-only") {
                RuntimeProfile::PureLearned
            } else {
                RuntimeProfile::Standard
            },
            transposition_entries: 65_536,
            quiescence_depth: 8,
            aspiration_window: 50,
            enable_alpha_beta: true,
            enable_iterative_deepening: true,
            enable_transposition_table: true,
            enable_move_ordering: true,
            enable_killers: true,
            enable_history: true,
            enable_quiescence: true,
            enable_pvs: true,
            enable_aspiration: true,
        }
    }
}

/// Why a search returned.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SearchTermination {
    Completed,
    Stable,
    NodeLimit,
    TimeLimit,
    Cancelled,
    /// The selected model failed its strict runtime inference contract.
    EvaluationError,
}

/// Why a returned score exists, or why evaluation did not start.
/// Terminal rule results never require a model call; interrupted nonterminal searches do not
/// have a score until an evaluator was actually called.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SearchOutcome {
    Evaluated,
    Checkmate,
    NoLegalMoves,
    Repetition,
    PerpetualCheck,
    CancelledBeforeEvaluation,
    NodeLimitBeforeEvaluation,
    TimeLimitBeforeEvaluation,
    EvaluationError,
}

impl SearchOutcome {
    #[must_use]
    pub const fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Checkmate | Self::NoLegalMoves | Self::Repetition | Self::PerpetualCheck
        )
    }

    #[must_use]
    pub const fn has_score(self) -> bool {
        matches!(self, Self::Evaluated) || self.is_terminal()
    }
}

/// Completed evidence for one root move at the latest fully searched depth.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RootMoveStat {
    pub movement: Move,
    pub score: i32,
    pub depth: u8,
    pub nodes: u64,
    pub pv: Vec<Move>,
}

/// Counters useful for deterministic regression tests and external benchmarks.
///
/// `pruned_moves / candidate_moves` is the move-level pruning share. Candidate moves are
/// counted after quiescence filtering; moves at transposition-table returns and unexpanded
/// horizon leaves are excluded.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SearchStats {
    pub tt_probes: u64,
    pub tt_hits: u64,
    pub tt_collisions: u64,
    pub beta_cutoffs: u64,
    /// Moves available at nodes whose post-filter move list was expanded.
    pub candidate_moves: u64,
    /// Candidate moves left unsearched specifically because of beta cutoffs.
    pub pruned_moves: u64,
    pub qnodes: u64,
    /// Number of static evaluations performed by the configured neural model.
    pub neural_inference_calls: u64,
    /// Wall-clock time spent encoding features and running neural inference.
    pub neural_inference_time: Duration,
    /// Number of failed strict OSAVAL02 inferences.
    pub osaval02_inference_errors: u64,
    /// Static evaluations served by a learned model (strict OSAVAL02 or neural leaf).
    pub learned_eval_calls: u64,
    /// Static evaluations served by the handcrafted evaluator, including its residual and
    /// composite blend roles.
    pub handcrafted_eval_calls: u64,
    pub residual_eval_calls: u64,
    pub composite_eval_calls: u64,
    /// Root completions that returned a legal move without any valid evaluation, such as a
    /// failed strict OSAVAL02 inference; deliberate legal timeout move selection and
    /// terminal-rule scores are not evaluation fallbacks.
    pub fallback_count: u64,
}

/// A completed iterative-deepening update.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SearchInfo {
    pub best_move: Option<Move>,
    pub score: i32,
    pub depth: u8,
    pub seldepth: u8,
    pub nodes: u64,
    pub elapsed: Duration,
    pub nps: u64,
    pub pv: Vec<Move>,
    pub root_moves: Vec<RootMoveStat>,
    pub stats: SearchStats,
}

/// Final result of a bounded search.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SearchResult {
    pub best_move: Option<Move>,
    pub score: i32,
    pub depth: u8,
    pub seldepth: u8,
    pub nodes: u64,
    pub elapsed: Duration,
    pub nps: u64,
    pub pv: Vec<Move>,
    pub root_moves: Vec<RootMoveStat>,
    pub stats: SearchStats,
    pub termination: SearchTermination,
    pub outcome: SearchOutcome,
}

/// Result of the bounded checking-move mate search.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MateSearchResult {
    pub found: bool,
    pub pv: Vec<Move>,
    pub nodes: u64,
    pub termination: SearchTermination,
}

/// Stable seeded random legal-move selector used by the first strength rung.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RandomMoveSelector {
    state: u64,
}

impl RandomMoveSelector {
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Returns an indexed legal move using a stable `SplitMix64` stream.
    pub fn select(&mut self, position: &Position) -> Option<Move> {
        let legal_moves = position.legal_moves();
        if legal_moves.is_empty() {
            return None;
        }
        let random = self.next_u64();
        let move_count = u64::try_from(legal_moves.len()).unwrap_or(u64::MAX);
        let index = usize::try_from(random % move_count).unwrap_or_default();
        Some(legal_moves[index])
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Bound {
    Exact,
    Lower,
    Upper,
}

#[derive(Clone, Debug)]
struct TranspositionEntry {
    hash: u64,
    position: Position,
    depth: u8,
    score: i32,
    bound: Bound,
    best_move: Option<Move>,
}

#[derive(Clone, Debug)]
struct NodeValue {
    score: i32,
    pv: Vec<Move>,
}

impl NodeValue {
    fn leaf(score: i32) -> Self {
        Self {
            score,
            pv: Vec::new(),
        }
    }
}

/// Reusable search state with a bounded direct-mapped transposition table.
pub struct SearchEngine {
    config: SearchConfig,
    computation: Option<Arc<crate::ComputationModel>>,
    computation_enabled: bool,
    compute_summary: Option<crate::ComputeControlSummary>,
    computation_order: Vec<Move>,
    phase10v: Option<Arc<crate::Phase10VEvaluator>>,
    v3_state: Option<crate::Phase10VAccumulator>,
    v3_updates: u64,
    v3_refreshes: u64,
    leaf_trace: std::cell::RefCell<Vec<serde_json::Value>>,
    leaf_trace_limit: usize,
    phase10t: Option<Arc<crate::Phase10TEvaluator>>,
    a1_state: Option<crate::Phase10TAccumulator>,
    a1_positions: Vec<Position>,
    a1_game_positions: Vec<Position>,
    a1_game_checks: Vec<bool>,
    pure_history_rejected: bool,
    a1_checks: Vec<bool>,
    #[cfg(feature = "handcrafted")]
    neural: Option<Arc<NeuralEvaluator>>,
    osaval02: Option<Osaval02SearchAdapter>,
    root_osaval02_policy: Vec<(Move, i32)>,
    #[cfg(feature = "handcrafted")]
    neural_mode: NeuralEvaluationMode,
    transposition_table: Vec<Option<TranspositionEntry>>,
    killers: Vec<[Option<Move>; 2]>,
    history: Vec<i32>,
    clock: Arc<dyn MonotonicClock>,
}

/// Static-score semantics for an attached neural artifact.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[cfg(feature = "handcrafted")]
pub enum NeuralEvaluationMode {
    /// The model output is the complete position score.
    #[default]
    PureValue,
    /// The model output is a learned delta added to the configured handcrafted score.
    Residual,
    /// Equal-weight blend of the configured handcrafted and pure-neural scores.
    Composite,
}

#[cfg(feature = "handcrafted")]
impl NeuralEvaluationMode {
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::PureValue => "pure-value",
            Self::Residual => "residual",
            Self::Composite => "composite-50-50",
        }
    }
}

impl SearchEngine {
    /// Attach a computation policy bound to the unchanged OSAVAL03 leaf model.
    /// # Errors
    /// Rejects a missing/different leaf rather than silently running another evaluator.
    pub fn set_computation_model(
        &mut self,
        model: Arc<crate::ComputationModel>,
    ) -> Result<(), String> {
        self.computation = None;
        self.computation_enabled = false;
        if self
            .phase10v
            .as_ref()
            .is_none_or(|leaf| leaf.identity().artifact_sha256 != model.leaf_sha256())
        {
            return Err("computation policy requires its exact OSAVAL03 leaf model".into());
        }
        self.computation = Some(model);
        self.computation_enabled = true;
        Ok(())
    }

    /// Explicit ablation on the same engine and loaded identities.
    /// # Errors
    /// Enabling without a loaded trained model is an error.
    pub fn set_computation_enabled(&mut self, enabled: bool) -> Result<(), String> {
        if enabled && self.computation.is_none() {
            return Err("no learned computation policy loaded".into());
        }
        self.computation_enabled = enabled;
        Ok(())
    }

    #[must_use]
    pub fn compute_control_summary(&self) -> Option<&crate::ComputeControlSummary> {
        self.compute_summary.as_ref()
    }

    fn update_computation(
        &mut self,
        position: &Position,
        info: &SearchInfo,
        previous: Option<&SearchInfo>,
        plan: Option<TimePlan>,
        previous_iteration_ms: f64,
    ) -> bool {
        let Some(model) = self
            .computation
            .as_ref()
            .filter(|_| self.computation_enabled)
        else {
            return false;
        };
        let (risk, priorities) = model.assess(position, info, previous);
        self.computation_order = priorities
            .into_iter()
            .map(|(movement, _)| movement)
            .collect();
        let target = plan
            .filter(|p| {
                matches!(
                    p.mode,
                    crate::TimeControlMode::Clock | crate::TimeControlMode::Casual
                )
            })
            .and_then(|p| model.target_ms(risk, p));
        if let Some(summary) = &mut self.compute_summary {
            summary.decisions += 1;
            summary.predicted_risk = risk;
            summary.target_ms = target.unwrap_or(0.0);
        }
        let elapsed = info.elapsed.as_secs_f64() * 1000.0;
        let iteration_ms = elapsed - previous.map_or(0.0, |p| p.elapsed.as_secs_f64() * 1000.0);
        let growth = if previous_iteration_ms > 0.0 {
            (iteration_ms / previous_iteration_ms).clamp(1.5, 8.0)
        } else {
            2.0
        };
        // Preserve at least two completed depths. Predicted next-iteration cost is a spending
        // estimate, never a relaxation of the recursive hard deadline or a mate certificate.
        target.is_some_and(|target| {
            info.depth >= 2 && elapsed >= target * 0.2 && elapsed + iteration_ms * growth >= target
        })
    }
    #[must_use]
    #[cfg(feature = "handcrafted")]
    pub fn new(config: SearchConfig) -> Self {
        Self::with_clock(config, Arc::new(SystemMonotonicClock::default()))
    }

    /// Builds an engine with an injected monotonic clock.
    #[must_use]
    #[cfg(feature = "handcrafted")]
    pub fn with_clock(config: SearchConfig, clock: Arc<dyn MonotonicClock>) -> Self {
        Self::empty(config, clock)
    }

    fn empty(config: SearchConfig, clock: Arc<dyn MonotonicClock>) -> Self {
        Self {
            config,
            computation: None,
            computation_enabled: false,
            compute_summary: None,
            computation_order: Vec::new(),
            phase10v: None,
            v3_state: None,
            v3_updates: 0,
            v3_refreshes: 0,
            leaf_trace: std::cell::RefCell::new(Vec::new()),
            leaf_trace_limit: 0,
            phase10t: None,
            a1_state: None,
            a1_positions: Vec::new(),
            a1_game_positions: Vec::new(),
            a1_game_checks: Vec::new(),
            pure_history_rejected: false,
            a1_checks: Vec::new(),
            #[cfg(feature = "handcrafted")]
            neural: None,
            osaval02: None,
            root_osaval02_policy: Vec::new(),
            #[cfg(feature = "handcrafted")]
            neural_mode: NeuralEvaluationMode::PureValue,
            transposition_table: vec![None; config.transposition_entries],
            killers: vec![[None; 2]; MAX_SEARCH_PLY],
            history: vec![0; 2 * MOVE_BUCKETS],
            clock,
        }
    }

    /// Builds a search engine that uses one immutable neural evaluator for static scores.
    ///
    /// Rule-terminal scores continue to be assigned by search and never by the model.
    #[must_use]
    #[cfg(feature = "handcrafted")]
    pub fn with_neural(config: SearchConfig, neural: Arc<NeuralEvaluator>) -> Self {
        Self::with_neural_mode(config, neural, NeuralEvaluationMode::PureValue)
    }

    /// Builds an engine with explicit, evidence-visible neural score semantics.
    #[must_use]
    #[cfg(feature = "handcrafted")]
    pub fn with_neural_mode(
        config: SearchConfig,
        neural: Arc<NeuralEvaluator>,
        mode: NeuralEvaluationMode,
    ) -> Self {
        let mut engine = Self::new(config);
        engine.neural = Some(neural);
        engine.neural_mode = mode;
        engine
    }

    /// Builds a neural engine with an injected monotonic clock.
    #[must_use]
    #[cfg(feature = "handcrafted")]
    pub fn with_neural_and_clock(
        config: SearchConfig,
        neural: Arc<NeuralEvaluator>,
        clock: Arc<dyn MonotonicClock>,
    ) -> Self {
        Self::with_neural_mode_and_clock(config, neural, NeuralEvaluationMode::PureValue, clock)
    }

    /// Builds an explicit neural-score mode with an injected monotonic clock.
    #[must_use]
    #[cfg(feature = "handcrafted")]
    pub fn with_neural_mode_and_clock(
        config: SearchConfig,
        neural: Arc<NeuralEvaluator>,
        mode: NeuralEvaluationMode,
        clock: Arc<dyn MonotonicClock>,
    ) -> Self {
        let mut engine = Self::empty(config, clock);
        engine.neural = Some(neural);
        engine.neural_mode = mode;
        engine
    }

    /// Constructs a validated, fail-closed frozen a1 production evaluator.
    /// # Errors
    /// Rejects a missing, malformed, or mismatched expected SHA-256 identity.
    pub fn with_phase10t(
        mut config: SearchConfig,
        evaluator: Arc<crate::Phase10TEvaluator>,
        expected_model_sha256: &str,
    ) -> Result<Self, String> {
        if expected_model_sha256.len() != 64
            || !expected_model_sha256
                .bytes()
                .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
            || evaluator.identity().artifact_sha256 != expected_model_sha256
        {
            return Err(
                "pure_learned a1 model SHA-256 mismatch or invalid expected identity".to_owned(),
            );
        }
        config.runtime_profile = RuntimeProfile::PureLearned;
        #[cfg(feature = "handcrafted")]
        {
            config.evaluation = EvaluationConfig::disabled();
        }
        let mut engine = Self::empty(config, Arc::new(SystemMonotonicClock::default()));
        engine.phase10t = Some(evaluator);
        Ok(engine)
    }

    /// Constructs a validated, fail-closed frozen OSAVAL03 production evaluator.
    /// # Errors
    /// Rejects a missing, malformed, or mismatched expected SHA-256 identity.
    pub fn with_phase10v(
        mut config: SearchConfig,
        evaluator: Arc<crate::Phase10VEvaluator>,
        expected_model_sha256: &str,
    ) -> Result<Self, String> {
        if expected_model_sha256.len() != 64
            || !expected_model_sha256
                .bytes()
                .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
            || evaluator.identity().artifact_sha256 != expected_model_sha256
        {
            return Err(
                "pure_learned OSAVAL03 model SHA-256 mismatch or invalid expected identity"
                    .to_owned(),
            );
        }
        config.runtime_profile = RuntimeProfile::PureLearned;
        #[cfg(feature = "handcrafted")]
        {
            config.evaluation = EvaluationConfig::disabled();
        }
        let mut engine = Self::empty(config, Arc::new(SystemMonotonicClock::default()));
        engine.phase10v = Some(evaluator);
        Ok(engine)
    }

    /// Enable bounded evidence collection at actual static-evaluation calls. Disabled by default.
    pub fn set_leaf_trace_limit(&mut self, limit: usize) {
        self.leaf_trace_limit = limit.min(10_000);
        self.leaf_trace.borrow_mut().clear();
    }

    /// Drain actual evaluated leaves, including root-relative ply and complete model identity.
    pub fn take_leaf_trace(&self) -> Vec<serde_json::Value> {
        std::mem::take(&mut *self.leaf_trace.borrow_mut())
    }

    /// Supplies actual game history by validated replay, before searching its current root.
    /// A standalone SFEN without this call has unknown prior history.
    /// # Errors
    /// Rejects use with another evaluator or an illegal history move.
    pub fn set_phase10t_history(
        &mut self,
        initial: &Position,
        moves: &[Move],
    ) -> Result<(), String> {
        if self.phase10t.is_none() {
            return Err("a1 history requires an a1 evaluator".to_owned());
        }
        self.set_pure_history(initial, moves)
    }

    /// Replay authoritative game history for either pure model format.
    /// # Errors
    /// Rejects non-pure engines and illegal histories.
    pub fn set_pure_history(&mut self, initial: &Position, moves: &[Move]) -> Result<(), String> {
        if self.config.runtime_profile != RuntimeProfile::PureLearned {
            return Err("pure history requires a pure evaluator".into());
        }
        self.pure_history_rejected = true;
        self.a1_game_positions.clear();
        self.a1_game_checks.clear();
        self.clear_transpositions();
        let mut game = crate::Game::new(initial.clone());
        let mut positions = vec![initial.clone()];
        let mut checks = Vec::new();
        for movement in moves {
            game.play(*movement).map_err(|error| error.to_string())?;
            let position = game.position();
            checks.push(position.is_in_check(position.side_to_move()));
            positions.push(position.clone());
        }
        self.a1_game_positions = positions;
        self.a1_game_checks = checks;
        self.pure_history_rejected = false;
        self.clear_transpositions();
        Ok(())
    }

    fn a1_history(&self) -> crate::Osaval02History {
        let Some(current) = self.a1_positions.last() else {
            return crate::Osaval02History::default();
        };
        let occurrences: Vec<usize> = self
            .a1_positions
            .iter()
            .enumerate()
            .filter_map(|(i, p)| current.same_state(p).then_some(i))
            .collect();
        let mut checking = [false; 2];
        if occurrences.len() >= 2 {
            let start = occurrences[0];
            for side in [Side::Black, Side::White] {
                let moves: Vec<usize> = (start..self.a1_checks.len())
                    .filter(|i| self.a1_positions[*i].side_to_move() == side)
                    .collect();
                checking[side.index()] =
                    !moves.is_empty() && moves.iter().all(|i| self.a1_checks[*i]);
            }
        }
        // Match the exact rule adjudicator's deterministic checking-side precedence.
        if checking[0] {
            checking[1] = false;
        }
        crate::Osaval02History {
            available: !self.a1_game_positions.is_empty() || self.a1_positions.len() > 1,
            repetition_count: u8::try_from(occurrences.len().min(4)).expect("bounded repetition"),
            continuous_check_by_us: checking[current.side_to_move().index()],
            continuous_check_by_them: checking[current.side_to_move().opposite().index()],
        }
    }

    fn a1_push(
        &mut self,
        movement: Move,
        position: &Position,
    ) -> (
        Option<crate::Phase10TAccumulator>,
        Option<crate::Phase10VAccumulator>,
    ) {
        if self.phase10t.is_none() && self.config.runtime_profile != RuntimeProfile::PureLearned {
            return (None, None);
        }
        self.a1_checks
            .push(position.is_in_check(position.side_to_move()));
        self.a1_positions.push(position.clone());
        if let Some(evaluator) = &self.phase10v {
            let state = self.v3_state.as_mut().expect("initialized v3 root");
            self.v3_refreshes +=
                u64::from(crate::Phase10VEvaluator::requires_refresh(state, movement));
            self.v3_updates += 1;
            return (
                None,
                Some(evaluator.update_generated_accumulator(state, movement, position)),
            );
        }
        let history = self.a1_history();
        let Some(evaluator) = self.phase10t.as_ref() else {
            return (None, None);
        };
        (
            Some(
                evaluator
                    .update_generated_accumulator(
                        self.a1_state.as_mut().expect("initialized a1 root"),
                        movement,
                        position,
                        history,
                    )
                    .expect("generated move and exact bounded history"),
            ),
            None,
        )
    }

    fn a1_pop(
        &mut self,
        previous: (
            Option<crate::Phase10TAccumulator>,
            Option<crate::Phase10VAccumulator>,
        ),
    ) {
        if let Some(state) = previous.1 {
            self.v3_state = Some(state);
        }
        if let Some(previous) = previous.0 {
            self.a1_state = Some(previous);
        }
        if self.phase10t.is_some() || self.config.runtime_profile == RuntimeProfile::PureLearned {
            self.a1_positions.pop();
            self.a1_checks.pop();
        }
    }

    fn a1_repetition(&self, position: &Position, ply: usize) -> Option<NodeValue> {
        if self.phase10t.is_none() && self.config.runtime_profile != RuntimeProfile::PureLearned {
            return None;
        }
        crate::game::repetition_outcome_from_history(&self.a1_positions, &self.a1_checks).map(
            |outcome| {
                NodeValue::leaf(match outcome {
                    crate::RepetitionOutcome::NoContest => 0,
                    crate::RepetitionOutcome::PerpetualCheckLoss(side) => {
                        if side == position.side_to_move() {
                            -MATE_SCORE + i32::try_from(ply).unwrap_or(i32::MAX)
                        } else {
                            MATE_SCORE - i32::try_from(ply).unwrap_or(i32::MAX)
                        }
                    }
                })
            },
        )
    }

    /// Builds a pure-value search engine backed by the strict OSAVAL02 adapter.
    ///
    /// OSAVAL02 is deliberately a separate constructor from the historical OSAVAL01 path. The
    /// selected artifact is never reinterpreted as OSAVAL01 and never blended with handcrafted
    /// evaluation.
    #[must_use]
    #[cfg(feature = "handcrafted")]
    pub fn with_osaval02(config: SearchConfig, evaluator: Arc<crate::Osaval02Evaluator>) -> Self {
        Self::with_osaval02_history_and_clock(
            config,
            evaluator,
            crate::Osaval02History::default(),
            Arc::new(SystemMonotonicClock::default()),
        )
    }

    /// Build the fail-closed OSAVAL02-only pure learned runtime.
    ///
    /// # Errors
    ///
    /// Rejects an invalid or mismatched expected artifact hash before search starts.
    pub fn with_pure_learned(
        mut config: SearchConfig,
        evaluator: Arc<crate::Osaval02Evaluator>,
        expected_model_sha256: &str,
    ) -> Result<Self, String> {
        if expected_model_sha256.len() != 64
            || !expected_model_sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        {
            return Err("pure_learned requires a lowercase SHA-256 model identity".to_owned());
        }
        if evaluator.identity().artifact_sha256 != expected_model_sha256 {
            return Err("pure_learned model SHA-256 mismatch".to_owned());
        }
        config.runtime_profile = RuntimeProfile::PureLearned;
        #[cfg(feature = "handcrafted")]
        {
            config.evaluation = EvaluationConfig::disabled();
        }
        let mut engine = Self::empty(config, Arc::new(SystemMonotonicClock::default()));
        engine.osaval02 = Some(Osaval02SearchAdapter::new(evaluator));
        Ok(engine)
    }

    /// Build the profile evidence for a completed search.
    ///
    /// The engine owns no opening-book or teacher call path, so `book_hits` and
    /// `teacher_calls` are structurally zero here; protocol layers that do own such paths
    /// must enforce the profile before any search starts.
    #[must_use]
    pub fn runtime_proof(
        &self,
        stats: SearchStats,
        model_sha256: impl Into<String>,
    ) -> RuntimeProofCounters {
        RuntimeProofCounters {
            profile: self.config.runtime_profile.name(),
            profile_schema: if self.phase10v.is_some() {
                crate::PHASE10V_PROFILE_SCHEMA
            } else if self.phase10t.is_some() {
                crate::PHASE10T_PROFILE_SCHEMA
            } else {
                crate::PURE_LEARNED_PROFILE_SCHEMA
            },
            learned_eval_calls: stats.learned_eval_calls,
            accumulator_updates: self.v3_updates,
            accumulator_refreshes: self.v3_refreshes,
            handcrafted_eval_calls: stats.handcrafted_eval_calls,
            residual_eval_calls: stats.residual_eval_calls,
            composite_eval_calls: stats.composite_eval_calls,
            book_hits: 0,
            teacher_calls: 0,
            fallback_count: stats.fallback_count,
            model_sha256: if let Some(evaluator) = &self.phase10v {
                evaluator.identity().artifact_sha256.clone()
            } else if let Some(evaluator) = &self.phase10t {
                evaluator.identity().artifact_sha256.clone()
            } else if let Some(evaluator) = &self.osaval02 {
                evaluator.identity().artifact_sha256.clone()
            } else {
                model_sha256.into()
            },
            evaluator_profile_schema_hash: if self.phase10v.is_some() {
                crate::PHASE10V_PROFILE_SCHEMA_SHA256
            } else if self.phase10t.is_some() {
                crate::PHASE10T_PROFILE_SCHEMA_SHA256
            } else {
                crate::PURE_LEARNED_PROFILE_SCHEMA_SHA256
            },
        }
    }

    /// Builds an OSAVAL02 search engine with explicit history facts.
    #[must_use]
    #[cfg(feature = "handcrafted")]
    pub fn with_osaval02_history(
        config: SearchConfig,
        evaluator: Arc<crate::Osaval02Evaluator>,
        history: crate::Osaval02History,
    ) -> Self {
        Self::with_osaval02_history_and_clock(
            config,
            evaluator,
            history,
            Arc::new(SystemMonotonicClock::default()),
        )
    }

    /// Builds an OSAVAL02 search engine with an injected monotonic clock.
    #[must_use]
    #[cfg(feature = "handcrafted")]
    pub fn with_osaval02_and_clock(
        config: SearchConfig,
        evaluator: Arc<crate::Osaval02Evaluator>,
        clock: Arc<dyn MonotonicClock>,
    ) -> Self {
        Self::with_osaval02_history_and_clock(
            config,
            evaluator,
            crate::Osaval02History::default(),
            clock,
        )
    }

    /// Builds an OSAVAL02 search engine with explicit history and clock dependencies.
    #[must_use]
    #[cfg(feature = "handcrafted")]
    pub fn with_osaval02_history_and_clock(
        config: SearchConfig,
        evaluator: Arc<crate::Osaval02Evaluator>,
        history: crate::Osaval02History,
        clock: Arc<dyn MonotonicClock>,
    ) -> Self {
        let mut engine = Self::empty(config, clock);
        engine.osaval02 = Some(Osaval02SearchAdapter::with_history(evaluator, history));
        engine
    }

    /// Returns the actual storage size of one direct-mapped table slot.
    ///
    /// Entries intentionally retain a full position to make hash collisions harmless.
    #[must_use]
    pub const fn transposition_entry_size_bytes() -> usize {
        size_of::<Option<TranspositionEntry>>()
    }

    /// Converts a byte budget into the largest whole number of table entries that fits.
    #[must_use]
    pub const fn transposition_entries_for_bytes(bytes: usize) -> usize {
        bytes / Self::transposition_entry_size_bytes()
    }

    /// Converts a mebibyte budget into a bounded whole-entry count.
    #[must_use]
    pub const fn transposition_entries_for_megabytes(megabytes: usize) -> usize {
        let bytes = megabytes.saturating_mul(1024 * 1024);
        Self::transposition_entries_for_bytes(bytes)
    }

    #[must_use]
    pub const fn config(&self) -> &SearchConfig {
        &self.config
    }

    /// Invalidates every transposition entry after an evaluator or search-semantics change.
    pub fn clear_transpositions(&mut self) {
        self.transposition_table.fill(None);
    }

    /// Searches without receiving intermediate iteration reports.
    pub fn search(
        &mut self,
        position: &Position,
        limits: SearchLimits,
        cancellation: &CancellationToken,
    ) -> SearchResult {
        self.search_internal(position, limits, cancellation, None, true, |_| {})
    }

    /// Searches and reports each fully completed iterative-deepening iteration.
    pub fn search_with_callback(
        &mut self,
        position: &Position,
        limits: SearchLimits,
        cancellation: &CancellationToken,
        callback: impl FnMut(&SearchInfo),
    ) -> SearchResult {
        self.search_internal(position, limits, cancellation, None, true, callback)
    }

    /// Searches using a time-manager plan and reports completed iterations.
    pub fn search_managed_with_callback(
        &mut self,
        position: &Position,
        plan: TimePlan,
        cancellation: &CancellationToken,
        callback: impl FnMut(&SearchInfo),
    ) -> SearchResult {
        let limits = SearchLimits {
            max_depth: plan.max_depth,
            max_nodes: plan.max_nodes,
            movetime: plan.hard_limit,
        };
        self.search_internal(position, limits, cancellation, Some(plan), true, callback)
    }

    /// Searches using a time-manager plan without iteration reports.
    pub fn search_managed(
        &mut self,
        position: &Position,
        plan: TimePlan,
        cancellation: &CancellationToken,
    ) -> SearchResult {
        self.search_managed_with_callback(position, plan, cancellation, |_| {})
    }

    /// Reuses compatible transposition entries while beginning a fresh iterative-deepening run.
    /// Killer and history heuristics are reset; suspended recursive stacks are never retained.
    pub fn search_reusing_transpositions_with_callback(
        &mut self,
        position: &Position,
        plan: TimePlan,
        cancellation: &CancellationToken,
        callback: impl FnMut(&SearchInfo),
    ) -> SearchResult {
        let limits = SearchLimits {
            max_depth: plan.max_depth,
            max_nodes: plan.max_nodes,
            movetime: plan.hard_limit,
        };
        self.search_internal(position, limits, cancellation, Some(plan), false, callback)
    }

    #[expect(
        clippy::too_many_lines,
        reason = "iterative-deepening completion and hard-deadline fallback remain one state transition"
    )]
    fn search_internal(
        &mut self,
        position: &Position,
        limits: SearchLimits,
        cancellation: &CancellationToken,
        managed: Option<TimePlan>,
        clear_transpositions: bool,
        mut callback: impl FnMut(&SearchInfo),
    ) -> SearchResult {
        self.reset_for_search(clear_transpositions);
        self.computation_order.clear();
        self.compute_summary =
            self.computation
                .as_ref()
                .map(|model| crate::ComputeControlSummary {
                    model_sha256: model.sha256().to_owned(),
                    enabled: self.computation_enabled,
                    ..crate::ComputeControlSummary::default()
                });
        let mut computation_previous: Option<SearchInfo> = None;
        let mut previous_iteration_ms = 0.0;
        self.v3_updates = 0;
        self.v3_refreshes = 0;
        self.leaf_trace.borrow_mut().clear();
        if self.config.runtime_profile == RuntimeProfile::PureLearned
            && (self.pure_history_rejected
                || self
                    .a1_game_positions
                    .last()
                    .is_some_and(|root| root != position))
        {
            return SearchResult {
                best_move: None,
                score: 0,
                depth: 0,
                seldepth: 0,
                nodes: 0,
                elapsed: Duration::ZERO,
                nps: 0,
                pv: Vec::new(),
                root_moves: Vec::new(),
                stats: SearchStats::default(),
                termination: SearchTermination::EvaluationError,
                outcome: SearchOutcome::EvaluationError,
            };
        }
        if self.phase10t.is_some() || self.config.runtime_profile == RuntimeProfile::PureLearned {
            if self
                .a1_game_positions
                .last()
                .is_some_and(|root| root.same_state(position))
            {
                self.a1_positions.clone_from(&self.a1_game_positions);
                self.a1_checks.clone_from(&self.a1_game_checks);
            } else {
                self.a1_game_positions.clear();
                self.a1_game_checks.clear();
                self.a1_positions = vec![position.clone()];
                self.a1_checks.clear();
            }
        }
        let clock = Arc::clone(&self.clock);
        let started = clock.now();
        let root_repetition = self.a1_repetition(position, 0);
        let legal_moves = if root_repetition.is_some() {
            Vec::new()
        } else {
            position.legal_moves()
        };
        let fallback = legal_moves.first().copied();
        let mut context = SearchContext::new(limits, cancellation, started, clock.as_ref());
        let root_terminal = if let Some(repetition) = &root_repetition {
            Some((
                if repetition.score == 0 {
                    SearchOutcome::Repetition
                } else {
                    SearchOutcome::PerpetualCheck
                },
                repetition.score,
            ))
        } else if legal_moves.is_empty() {
            Some((
                if position.is_in_check(position.side_to_move()) {
                    SearchOutcome::Checkmate
                } else {
                    SearchOutcome::NoLegalMoves
                },
                -MATE_SCORE,
            ))
        } else {
            None
        };
        if let Some((outcome, score)) = root_terminal {
            return SearchResult {
                best_move: None,
                score,
                depth: 0,
                seldepth: 0,
                nodes: 0,
                elapsed: context.elapsed(),
                nps: 0,
                pv: Vec::new(),
                root_moves: Vec::new(),
                stats: context.stats,
                termination: SearchTermination::Completed,
                outcome,
            };
        }
        let mut completed = NodeValue {
            score: if context.check_termination().is_err() {
                // A legal fallback still lets callers emit a protocol-compliant move, but a
                // cancelled/zero-budget search must not run an uncounted neural inference.
                0
            } else {
                if let Some(evaluator) = &self.phase10v {
                    self.v3_state =
                        Some(evaluator.accumulator(position).expect("validated position"));
                    self.v3_refreshes = 2;
                }
                if let Some(evaluator) = &self.phase10t {
                    self.a1_state = Some(
                        evaluator
                            .accumulator(position, self.a1_history())
                            .expect("validated search history"),
                    );
                }
                if limits.max_depth > 0 {
                    self.prepare_osaval02_root_policy(position, &mut context);
                }
                let evaluated = self.evaluate_position(position, &mut context);
                if evaluated.is_err() {
                    // A failed strict inference returns the protocol-required legal move
                    // without any valid score; the pure-learned proof must see this fallback.
                    context.stats.fallback_count = context.stats.fallback_count.saturating_add(1);
                }
                evaluated.unwrap_or_default()
            },
            pv: fallback.into_iter().collect(),
        };
        let mut completed_depth = 0;
        let depths: Vec<u8> = if self.config.enable_iterative_deepening {
            (1..=limits.max_depth).collect()
        } else {
            vec![limits.max_depth]
        };

        if limits.max_depth == 0 {
            let _ = context.enter_node(0, false);
        } else {
            for depth in depths {
                context.root_partial = None;
                context.root_moves.clear();
                let aspiration = self.config.enable_aspiration
                    && self.config.enable_alpha_beta
                    && completed_depth > 0
                    && self.config.aspiration_window > 0;
                let (mut alpha, mut beta) = if aspiration {
                    (
                        completed
                            .score
                            .saturating_sub(self.config.aspiration_window),
                        completed
                            .score
                            .saturating_add(self.config.aspiration_window),
                    )
                } else {
                    (-INFINITY, INFINITY)
                };

                let iteration = loop {
                    // An aspiration retry replaces incomplete/bounded root evidence from
                    // the previous attempt; duplicate candidates are not separate moves.
                    context.root_moves.clear();
                    let result =
                        self.negamax(&mut position.clone(), depth, alpha, beta, 0, &mut context);
                    let Ok(value) = result else {
                        break None;
                    };
                    if aspiration && value.score <= alpha {
                        alpha = -INFINITY;
                        continue;
                    }
                    if aspiration && value.score >= beta {
                        beta = INFINITY;
                        continue;
                    }
                    break Some(value);
                };

                let Some(value) = iteration else {
                    if completed_depth == 0
                        && let Some(partial) = context.root_partial.take()
                    {
                        completed = partial;
                    }
                    break;
                };

                completed = value;
                completed_depth = depth;
                context.complete_root_iteration(depth);
                let elapsed = context.elapsed();
                let info = make_info(&completed, completed_depth, &context, elapsed);
                callback(&info);
                if is_mate_score(completed.score) {
                    break;
                }
                if self.update_computation(
                    position,
                    &info,
                    computation_previous.as_ref(),
                    managed,
                    previous_iteration_ms,
                ) {
                    context.termination = Some(SearchTermination::Stable);
                    break;
                }
                if self.computation_enabled {
                    previous_iteration_ms = info.elapsed.as_secs_f64() * 1000.0
                        - computation_previous
                            .as_ref()
                            .map_or(0.0, |p| p.elapsed.as_secs_f64() * 1000.0);
                    computation_previous = Some(info);
                }
                if managed.is_some_and(|plan| {
                    context.should_stop_stable(position, &completed, plan, elapsed)
                }) {
                    context.termination = Some(SearchTermination::Stable);
                    break;
                }
            }
        }

        let elapsed = context.elapsed();
        let termination = context.termination.unwrap_or(SearchTermination::Completed);
        let outcome = if termination == SearchTermination::EvaluationError {
            SearchOutcome::EvaluationError
        } else if context.stats.neural_inference_calls > 0
            || context.stats.handcrafted_eval_calls > 0
        {
            SearchOutcome::Evaluated
        } else {
            match termination {
                SearchTermination::Cancelled => SearchOutcome::CancelledBeforeEvaluation,
                SearchTermination::NodeLimit => SearchOutcome::NodeLimitBeforeEvaluation,
                SearchTermination::TimeLimit => SearchOutcome::TimeLimitBeforeEvaluation,
                _ => SearchOutcome::EvaluationError,
            }
        };
        SearchResult {
            best_move: completed.pv.first().copied().or(fallback),
            score: completed.score,
            depth: completed_depth,
            seldepth: context.seldepth,
            nodes: context.nodes,
            elapsed,
            nps: nodes_per_second(context.nodes, elapsed),
            pv: completed.pv,
            root_moves: context.last_completed_root_moves,
            stats: context.stats,
            termination,
            outcome,
        }
    }

    /// Performs a simple bounded checking-move AND/OR mate search.
    pub fn find_mate(
        &mut self,
        position: &Position,
        max_depth: u8,
        max_nodes: Option<u64>,
        cancellation: &CancellationToken,
    ) -> MateSearchResult {
        let mut context = MateContext {
            attacker: position.side_to_move(),
            max_nodes,
            cancellation,
            nodes: 0,
            termination: None,
        };
        let outcome = mate_dfs(&mut position.clone(), max_depth, &mut context);
        let (found, pv) = match outcome {
            Ok(Some(pv)) => (true, pv),
            Ok(None) | Err(()) => (false, Vec::new()),
        };
        MateSearchResult {
            found,
            pv,
            nodes: context.nodes,
            termination: context.termination.unwrap_or(SearchTermination::Completed),
        }
    }

    fn reset_for_search(&mut self, clear_transpositions: bool) {
        if clear_transpositions {
            self.transposition_table.fill(None);
        }
        self.killers.fill([None; 2]);
        self.history.fill(0);
        self.root_osaval02_policy.clear();
    }

    #[expect(
        clippy::too_many_lines,
        reason = "the main node routine keeps its alpha-beta state transitions together"
    )]
    fn negamax(
        &mut self,
        position: &mut Position,
        depth: u8,
        mut alpha: i32,
        beta: i32,
        ply: usize,
        context: &mut SearchContext<'_>,
    ) -> Result<NodeValue, ()> {
        if depth == 0 && self.config.enable_quiescence {
            return self.quiescence(
                position,
                alpha,
                beta,
                ply,
                self.config.quiescence_depth,
                context,
            );
        }
        context.enter_node(ply, false)?;
        if let Some(repetition) = self.a1_repetition(position, ply) {
            return Ok(repetition);
        }

        // Legal terminal classification must precede every horizon/static evaluation. Besides
        // preserving rule-loss semantics when quiescence is disabled, this guarantees
        // that terminal nodes never invoke the optional neural evaluator.
        if depth == 0 || ply >= MAX_SEARCH_PLY {
            let moves = position.legal_moves();
            if moves.is_empty() {
                return Ok(terminal_node(position, ply));
            }
            return Ok(NodeValue {
                score: self.evaluate_position_kind(position, context, "pv_horizon_leaf")?,
                pv: Vec::new(),
            });
        }

        let alpha_original = alpha;
        let mut tt_move = None;
        if self.config.enable_transposition_table
            && let Some(entry) = self.probe_transposition(position, context)
        {
            tt_move = entry.best_move;
            // The root must still enumerate every move so MultiPV/root evidence remains complete.
            // Retained root entries are used only for move ordering; interior entries may cut.
            if self.phase10t.is_none()
                && self.config.runtime_profile != RuntimeProfile::PureLearned
                && ply > 0
                && entry.depth >= depth
            {
                let score = score_from_transposition(entry.score, ply);
                match entry.bound {
                    Bound::Exact => {
                        return Ok(NodeValue {
                            score,
                            pv: entry.best_move.into_iter().collect(),
                        });
                    }
                    Bound::Lower if score >= beta => {
                        return Ok(NodeValue {
                            score,
                            pv: entry.best_move.into_iter().collect(),
                        });
                    }
                    Bound::Upper if score <= alpha => {
                        return Ok(NodeValue {
                            score,
                            pv: entry.best_move.into_iter().collect(),
                        });
                    }
                    Bound::Lower => alpha = alpha.max(score),
                    Bound::Upper => {}
                }
            }
        }

        let mut moves = position.legal_moves();
        if moves.is_empty() {
            return Ok(terminal_node(position, ply));
        }
        if self.config.enable_move_ordering {
            self.order_moves(position, &mut moves, tt_move, ply);
        }
        if ply == 0 && !self.computation_order.is_empty() {
            let previous = moves.clone();
            moves.sort_by_key(|movement| {
                self.computation_order
                    .iter()
                    .position(|m| m == movement)
                    .unwrap_or(usize::MAX)
            });
            if let Some(summary) = &mut self.compute_summary {
                summary.reordered_moves += u64::try_from(
                    moves
                        .iter()
                        .zip(previous.iter())
                        .filter(|(a, b)| a != b)
                        .count(),
                )
                .unwrap_or(u64::MAX);
            }
        }
        context.stats.candidate_moves = context
            .stats
            .candidate_moves
            .saturating_add(u64::try_from(moves.len()).unwrap_or(u64::MAX));

        let side = position.side_to_move();
        let mut best_score = -INFINITY;
        let mut best_move = None;
        let mut best_pv = Vec::new();
        let mut searched_moves = 0_usize;

        for (index, movement) in moves.iter().copied().enumerate() {
            let nodes_before = context.nodes;
            let quiet = is_quiet(position, movement);
            let undo = position.make_generated_move(movement);
            let a1_previous = self.a1_push(movement, position);
            let child = if self.config.enable_pvs && self.config.enable_alpha_beta && index > 0 {
                let scout = self.negamax(position, depth - 1, -alpha - 1, -alpha, ply + 1, context);
                match scout {
                    Ok(value) if -value.score > alpha && -value.score < beta => {
                        self.negamax(position, depth - 1, -beta, -alpha, ply + 1, context)
                    }
                    other => other,
                }
            } else {
                let (child_alpha, child_beta) = if self.config.enable_alpha_beta {
                    (-beta, -alpha)
                } else {
                    (-INFINITY, INFINITY)
                };
                self.negamax(
                    position,
                    depth - 1,
                    child_alpha,
                    child_beta,
                    ply + 1,
                    context,
                )
            };
            position.unmake_move(undo);
            self.a1_pop(a1_previous);
            let child = child?;
            searched_moves += 1;
            let score = -child.score;

            if ply == 0 {
                let mut pv = Vec::with_capacity(child.pv.len() + 1);
                pv.push(movement);
                pv.extend(child.pv.iter().copied());
                context.root_moves.push(RootMoveStat {
                    movement,
                    score,
                    depth,
                    nodes: context.nodes.saturating_sub(nodes_before),
                    pv,
                });
            }

            if score > best_score {
                best_score = score;
                best_move = Some(movement);
                best_pv.clear();
                best_pv.push(movement);
                best_pv.extend(child.pv);
                if ply == 0 {
                    context.root_partial = Some(NodeValue {
                        score: best_score,
                        pv: best_pv.clone(),
                    });
                }
            }
            alpha = alpha.max(score);
            if self.config.enable_alpha_beta && alpha >= beta {
                context.stats.beta_cutoffs = context.stats.beta_cutoffs.saturating_add(1);
                context.stats.pruned_moves = context
                    .stats
                    .pruned_moves
                    .saturating_add((moves.len() - searched_moves) as u64);
                if quiet {
                    self.record_quiet_cutoff(side, movement, depth, ply);
                }
                break;
            }
        }

        let value = NodeValue {
            score: best_score,
            pv: best_pv,
        };
        if self.config.enable_transposition_table {
            let bound = if best_score <= alpha_original {
                Bound::Upper
            } else if best_score >= beta {
                Bound::Lower
            } else {
                Bound::Exact
            };
            self.store_transposition(
                position,
                depth,
                score_to_transposition(best_score, ply),
                bound,
                best_move,
            );
        }
        Ok(value)
    }

    fn quiescence(
        &mut self,
        position: &mut Position,
        mut alpha: i32,
        beta: i32,
        ply: usize,
        remaining: u8,
        context: &mut SearchContext<'_>,
    ) -> Result<NodeValue, ()> {
        context.enter_node(ply, true)?;
        if let Some(repetition) = self.a1_repetition(position, ply) {
            return Ok(repetition);
        }
        let in_check = position.is_in_check(position.side_to_move());
        let mut moves = position.legal_moves();
        if moves.is_empty() {
            return Ok(terminal_node(position, ply));
        }
        // Checked nodes have no legal stand-pat; skip unused inference and metrics.
        let stand_pat = if in_check {
            -INFINITY
        } else {
            self.evaluate_position_kind(position, context, "quiescence_leaf")?
        };
        if ply >= MAX_SEARCH_PLY {
            return Ok(NodeValue::leaf(if in_check {
                if self.phase10t.is_some() || self.phase10v.is_some() {
                    self.evaluate_position_kind(position, context, "quiescence_leaf")?
                        .max(alpha)
                        .min(beta)
                } else {
                    0_i32.max(alpha).min(beta)
                }
            } else {
                stand_pat.max(alpha)
            }));
        }
        if remaining == 0 && !in_check {
            return Ok(NodeValue::leaf(stand_pat.max(alpha)));
        }

        if !in_check {
            moves.retain(|movement| is_tactical(position, *movement));
        }
        context.stats.candidate_moves = context
            .stats
            .candidate_moves
            .saturating_add(u64::try_from(moves.len()).unwrap_or(u64::MAX));
        if !in_check {
            if stand_pat >= beta {
                context.stats.beta_cutoffs = context.stats.beta_cutoffs.saturating_add(1);
                context.stats.pruned_moves = context
                    .stats
                    .pruned_moves
                    .saturating_add(u64::try_from(moves.len()).unwrap_or(u64::MAX));
                return Ok(NodeValue::leaf(stand_pat));
            }
            alpha = alpha.max(stand_pat);
        }
        if self.config.enable_move_ordering {
            self.order_moves(position, &mut moves, None, ply);
        }

        let mut best = NodeValue {
            score: if in_check { -INFINITY } else { stand_pat },
            pv: Vec::new(),
        };
        let candidate_count = moves.len();
        let mut searched_moves = 0_usize;
        let child_remaining = remaining.saturating_sub(1);
        for movement in moves {
            let undo = position.make_generated_move(movement);
            let a1_previous = self.a1_push(movement, position);
            let child = self.quiescence(position, -beta, -alpha, ply + 1, child_remaining, context);
            position.unmake_move(undo);
            self.a1_pop(a1_previous);
            let child = child?;
            searched_moves += 1;
            let score = -child.score;
            if score > best.score {
                best.score = score;
                best.pv.clear();
                best.pv.push(movement);
                best.pv.extend(child.pv);
            }
            alpha = alpha.max(score);
            if alpha >= beta {
                context.stats.beta_cutoffs = context.stats.beta_cutoffs.saturating_add(1);
                context.stats.pruned_moves = context.stats.pruned_moves.saturating_add(
                    u64::try_from(candidate_count - searched_moves).unwrap_or(u64::MAX),
                );
                break;
            }
        }
        Ok(best)
    }

    fn probe_transposition(
        &self,
        position: &Position,
        context: &mut SearchContext<'_>,
    ) -> Option<TranspositionEntry> {
        if self.transposition_table.is_empty() {
            return None;
        }
        context.stats.tt_probes = context.stats.tt_probes.saturating_add(1);
        let index = transposition_index(position.zobrist_hash(), self.transposition_table.len());
        let entry = self.transposition_table[index].as_ref()?;
        if entry.hash == position.zobrist_hash() && entry.position.same_state(position) {
            context.stats.tt_hits = context.stats.tt_hits.saturating_add(1);
            Some(entry.clone())
        } else {
            context.stats.tt_collisions = context.stats.tt_collisions.saturating_add(1);
            None
        }
    }

    fn store_transposition(
        &mut self,
        position: &Position,
        depth: u8,
        score: i32,
        bound: Bound,
        best_move: Option<Move>,
    ) {
        if self.transposition_table.is_empty() {
            return;
        }
        let index = transposition_index(position.zobrist_hash(), self.transposition_table.len());
        let replace = self.transposition_table[index]
            .as_ref()
            .is_none_or(|entry| entry.hash != position.zobrist_hash() || depth >= entry.depth);
        if replace {
            self.transposition_table[index] = Some(TranspositionEntry {
                hash: position.zobrist_hash(),
                position: position.clone(),
                depth,
                score,
                bound,
                best_move,
            });
        }
    }

    fn order_moves(
        &self,
        position: &Position,
        moves: &mut [Move],
        transposition_move: Option<Move>,
        ply: usize,
    ) {
        moves.sort_by(|left, right| {
            let right_score = self.move_order_score(position, *right, transposition_move, ply);
            let left_score = self.move_order_score(position, *left, transposition_move, ply);
            right_score.cmp(&left_score).then_with(|| left.cmp(right))
        });
    }

    fn move_order_score(
        &self,
        position: &Position,
        movement: Move,
        transposition_move: Option<Move>,
        ply: usize,
    ) -> i32 {
        if Some(movement) == transposition_move {
            return 2_000_000;
        }

        let mut score = 0;
        if let Some(captured) = position.piece_at(movement.destination()) {
            let victim = order_piece_value(captured.kind);
            let attacker = match movement {
                Move::Normal { from, .. } => position
                    .piece_at(from)
                    .map_or(0, |piece| order_piece_value(piece.kind)),
                Move::Drop { piece, .. } => order_piece_value(piece.piece_kind()),
            };
            score += 1_000_000 + 16 * victim - attacker;
        }
        if matches!(movement, Move::Normal { promote: true, .. }) {
            score += 100_000;
        }
        if ply == 0
            && let Some((_, policy_bonus)) = self
                .root_osaval02_policy
                .iter()
                .find(|(candidate, _)| *candidate == movement)
        {
            score += *policy_bonus;
        }
        if is_quiet(position, movement) {
            if self.config.enable_killers
                && let Some(killers) = self.killers.get(ply)
            {
                if killers[0] == Some(movement) {
                    score += 90_000;
                } else if killers[1] == Some(movement) {
                    score += 80_000;
                }
            }
            if self.config.enable_history {
                score += self.history[history_index(position.side_to_move(), movement)];
            }
        }
        score
    }

    fn record_quiet_cutoff(&mut self, side: Side, movement: Move, depth: u8, ply: usize) {
        if self.config.enable_killers
            && let Some(killers) = self.killers.get_mut(ply)
            && killers[0] != Some(movement)
        {
            killers[1] = killers[0];
            killers[0] = Some(movement);
        }
        if self.config.enable_history {
            let index = history_index(side, movement);
            let bonus = i32::from(depth).saturating_mul(i32::from(depth));
            self.history[index] = self.history[index].saturating_add(bonus);
        }
    }

    fn evaluate_position(
        &self,
        position: &Position,
        context: &mut SearchContext<'_>,
    ) -> Result<i32, ()> {
        self.evaluate_position_kind(position, context, "static")
    }

    fn evaluate_position_kind(
        &self,
        position: &Position,
        context: &mut SearchContext<'_>,
        kind: &str,
    ) -> Result<i32, ()> {
        if let Some(evaluator) = &self.phase10v {
            let started = self.clock.now();
            let result = evaluator
                .infer_owned_accumulator(self.v3_state.as_ref().expect("initialized v3 state"));
            context.stats.neural_inference_calls =
                context.stats.neural_inference_calls.saturating_add(1);
            context.stats.learned_eval_calls = context.stats.learned_eval_calls.saturating_add(1);
            context.stats.neural_inference_time = context
                .stats
                .neural_inference_time
                .saturating_add(self.clock.now().saturating_sub(started));
            if self.leaf_trace_limit > 0 {
                let mut trace = self.leaf_trace.borrow_mut();
                if trace.len() < self.leaf_trace_limit {
                    trace.push(serde_json::json!({"sfen": crate::to_sfen(position),
                        "ply": self.a1_positions.len().saturating_sub(self.a1_game_positions.len().max(1)),
                        "kind": kind, "cp": result.cp,
                        "model_sha256": evaluator.identity().artifact_sha256,
                        "evaluation_ordinal": context.stats.learned_eval_calls}));
                }
            }
            return Ok(result.cp);
        }
        if let Some(evaluator) = &self.phase10t {
            let started = self.clock.now();
            let result =
                evaluator.infer_accumulator(self.a1_state.as_ref().expect("initialized a1 state"));
            context.stats.neural_inference_calls =
                context.stats.neural_inference_calls.saturating_add(1);
            context.stats.learned_eval_calls = context.stats.learned_eval_calls.saturating_add(1);
            context.stats.neural_inference_time = context
                .stats
                .neural_inference_time
                .saturating_add(self.clock.now().saturating_sub(started));
            return Ok(result.cp);
        }
        if let Some(osaval02) = &self.osaval02 {
            let started = self.clock.now();
            let result = if self.config.runtime_profile == RuntimeProfile::PureLearned {
                osaval02.evaluate_history(position, self.a1_history())
            } else {
                osaval02.evaluate(position)
            };
            context.stats.neural_inference_calls =
                context.stats.neural_inference_calls.saturating_add(1);
            context.stats.learned_eval_calls = context.stats.learned_eval_calls.saturating_add(1);
            context.stats.neural_inference_time = context
                .stats
                .neural_inference_time
                .saturating_add(self.clock.now().saturating_sub(started));
            return if let Ok(score) = result {
                Ok(score)
            } else {
                context.stats.osaval02_inference_errors =
                    context.stats.osaval02_inference_errors.saturating_add(1);
                context.termination = Some(SearchTermination::EvaluationError);
                Err(())
            };
        }
        #[cfg(feature = "handcrafted")]
        {
            let Some(neural) = &self.neural else {
                context.stats.handcrafted_eval_calls =
                    context.stats.handcrafted_eval_calls.saturating_add(1);
                return Ok(evaluate(position, &self.config.evaluation));
            };
            let started = self.clock.now();
            let learned = neural.evaluate(position);
            context.stats.neural_inference_calls =
                context.stats.neural_inference_calls.saturating_add(1);
            context.stats.learned_eval_calls = context.stats.learned_eval_calls.saturating_add(1);
            context.stats.neural_inference_time = context
                .stats
                .neural_inference_time
                .saturating_add(self.clock.now().saturating_sub(started));
            Ok(match self.neural_mode {
                NeuralEvaluationMode::PureValue => learned,
                NeuralEvaluationMode::Residual => {
                    context.stats.residual_eval_calls =
                        context.stats.residual_eval_calls.saturating_add(1);
                    context.stats.handcrafted_eval_calls =
                        context.stats.handcrafted_eval_calls.saturating_add(1);
                    evaluate(position, &self.config.evaluation).saturating_add(learned)
                }
                NeuralEvaluationMode::Composite => {
                    context.stats.composite_eval_calls =
                        context.stats.composite_eval_calls.saturating_add(1);
                    context.stats.handcrafted_eval_calls =
                        context.stats.handcrafted_eval_calls.saturating_add(1);
                    let handcrafted = evaluate(position, &self.config.evaluation);
                    handcrafted.saturating_add(learned) / 2
                }
            })
        }
        #[cfg(feature = "pure-only")]
        {
            context.termination = Some(SearchTermination::EvaluationError);
            Err(())
        }
    }

    fn prepare_osaval02_root_policy(
        &mut self,
        position: &Position,
        context: &mut SearchContext<'_>,
    ) {
        let Some(osaval02) = &self.osaval02 else {
            return;
        };
        let started = self.clock.now();
        let inference = if self.config.runtime_profile == RuntimeProfile::PureLearned {
            osaval02.infer_history(position, self.a1_history())
        } else {
            osaval02.infer(position)
        };
        context.stats.neural_inference_calls =
            context.stats.neural_inference_calls.saturating_add(1);
        context.stats.learned_eval_calls = context.stats.learned_eval_calls.saturating_add(1);
        context.stats.neural_inference_time = context
            .stats
            .neural_inference_time
            .saturating_add(self.clock.now().saturating_sub(started));
        let Ok(inference) = inference else {
            context.stats.osaval02_inference_errors =
                context.stats.osaval02_inference_errors.saturating_add(1);
            context.termination = Some(SearchTermination::EvaluationError);
            return;
        };
        let legal_moves = position.legal_moves();
        let mut ranked = legal_moves
            .into_iter()
            .filter_map(|movement| inference.policy_logit(movement).map(|_| movement))
            .collect::<Vec<_>>();
        ranked.sort_by(|left, right| {
            inference
                .policy_logit(*right)
                .unwrap_or(f64::NEG_INFINITY)
                .total_cmp(&inference.policy_logit(*left).unwrap_or(f64::NEG_INFINITY))
                .then_with(|| left.cmp(right))
        });
        let count = i32::try_from(ranked.len()).unwrap_or(i32::MAX);
        self.root_osaval02_policy = ranked
            .into_iter()
            .enumerate()
            .map(|(rank, movement)| {
                let rank = i32::try_from(rank).unwrap_or(i32::MAX);
                (movement, count.saturating_sub(rank).saturating_mul(1_000))
            })
            .collect();
    }
}

struct SearchContext<'a> {
    limits: SearchLimits,
    cancellation: &'a CancellationToken,
    started: Duration,
    clock: &'a dyn MonotonicClock,
    nodes: u64,
    seldepth: u8,
    stats: SearchStats,
    termination: Option<SearchTermination>,
    root_partial: Option<NodeValue>,
    root_moves: Vec<RootMoveStat>,
    last_completed_root_moves: Vec<RootMoveStat>,
    previous_best_move: Option<Move>,
    previous_score: Option<i32>,
    stable_iterations: u8,
}

impl<'a> SearchContext<'a> {
    fn new(
        limits: SearchLimits,
        cancellation: &'a CancellationToken,
        started: Duration,
        clock: &'a dyn MonotonicClock,
    ) -> Self {
        Self {
            limits,
            cancellation,
            started,
            clock,
            nodes: 0,
            seldepth: 0,
            stats: SearchStats::default(),
            termination: None,
            root_partial: None,
            root_moves: Vec::new(),
            last_completed_root_moves: Vec::new(),
            previous_best_move: None,
            previous_score: None,
            stable_iterations: 0,
        }
    }

    fn elapsed(&self) -> Duration {
        self.clock.now().saturating_sub(self.started)
    }

    fn complete_root_iteration(&mut self, depth: u8) {
        self.root_moves.sort_by(|left, right| {
            right
                .score
                .cmp(&left.score)
                .then_with(|| left.movement.cmp(&right.movement))
        });
        for stat in &mut self.root_moves {
            stat.depth = depth;
        }
        self.last_completed_root_moves.clone_from(&self.root_moves);
    }

    fn should_stop_stable(
        &mut self,
        position: &Position,
        completed: &NodeValue,
        plan: TimePlan,
        elapsed: Duration,
    ) -> bool {
        if !plan.allow_stable_early_stop || plan.soft_limit.is_none_or(|soft| elapsed < soft) {
            self.observe_stability(completed, plan);
            return false;
        }
        let score_delta = self.observe_stability(completed, plan);
        let legal_moves = position.legal_moves();
        let in_check = position.is_in_check(position.side_to_move());
        let tactical = legal_moves.iter().copied().any(|movement| {
            is_tactical(position, movement) || move_gives_check(position, movement)
        });
        let close_candidates = self.last_completed_root_moves.get(0..2).is_some_and(|top| {
            top[0].score.saturating_sub(top[1].score) <= plan.stability.close_candidate_cp
        });
        let volatile = score_delta.is_some_and(|delta| delta >= plan.stability.volatile_score_cp);
        let difficult = legal_moves.len() >= plan.stability.difficult_branching_moves;
        self.stable_iterations >= plan.stability.required_stable_iterations
            && !in_check
            && !tactical
            && !close_candidates
            && !volatile
            && !difficult
    }

    fn observe_stability(&mut self, completed: &NodeValue, plan: TimePlan) -> Option<i32> {
        let best_move = completed.pv.first().copied();
        let score_delta = self
            .previous_score
            .map(|previous| completed.score.saturating_sub(previous).saturating_abs());
        if self.previous_best_move == best_move
            && score_delta.is_some_and(|delta| delta <= plan.stability.stable_score_cp)
        {
            self.stable_iterations = self.stable_iterations.saturating_add(1);
        } else {
            self.stable_iterations = 0;
        }
        self.previous_best_move = best_move;
        self.previous_score = Some(completed.score);
        score_delta
    }

    fn enter_node(&mut self, ply: usize, quiescence: bool) -> Result<(), ()> {
        self.check_termination()?;
        self.nodes += 1;
        if quiescence {
            self.stats.qnodes = self.stats.qnodes.saturating_add(1);
        }
        self.seldepth = self.seldepth.max(u8::try_from(ply).unwrap_or(u8::MAX));
        Ok(())
    }

    fn check_termination(&mut self) -> Result<(), ()> {
        if self.termination.is_some() {
            return Err(());
        }
        if self.cancellation.is_cancelled() {
            self.termination = Some(SearchTermination::Cancelled);
            return Err(());
        }
        if self
            .limits
            .movetime
            .is_some_and(|limit| self.elapsed() >= limit)
        {
            self.termination = Some(SearchTermination::TimeLimit);
            return Err(());
        }
        if self
            .limits
            .max_nodes
            .is_some_and(|limit| self.nodes >= limit)
        {
            self.termination = Some(SearchTermination::NodeLimit);
            return Err(());
        }
        Ok(())
    }
}

struct MateContext<'a> {
    attacker: Side,
    max_nodes: Option<u64>,
    cancellation: &'a CancellationToken,
    nodes: u64,
    termination: Option<SearchTermination>,
}

impl MateContext<'_> {
    fn enter_node(&mut self) -> Result<(), ()> {
        if self.termination.is_some() {
            return Err(());
        }
        if self.cancellation.is_cancelled() {
            self.termination = Some(SearchTermination::Cancelled);
            return Err(());
        }
        if self.max_nodes.is_some_and(|limit| self.nodes >= limit) {
            self.termination = Some(SearchTermination::NodeLimit);
            return Err(());
        }
        self.nodes += 1;
        Ok(())
    }
}

fn mate_dfs(
    position: &mut Position,
    remaining: u8,
    context: &mut MateContext<'_>,
) -> Result<Option<Vec<Move>>, ()> {
    context.enter_node()?;
    let defender_turn = position.side_to_move() != context.attacker;
    let mut moves = position.legal_moves();
    if moves.is_empty() {
        return Ok((defender_turn && position.is_in_check(position.side_to_move())).then(Vec::new));
    }
    if remaining == 0 {
        return Ok(None);
    }
    moves.sort_unstable();

    if defender_turn {
        if !position.is_in_check(position.side_to_move()) {
            return Ok(None);
        }
        let mut longest_defence = Vec::new();
        for movement in moves {
            let undo = position.make_generated_move(movement);
            let child = mate_dfs(position, remaining - 1, context);
            position.unmake_move(undo);
            let Some(child) = child? else {
                return Ok(None);
            };
            let mut line = vec![movement];
            line.extend(child);
            if line.len() > longest_defence.len() {
                longest_defence = line;
            }
        }
        Ok(Some(longest_defence))
    } else {
        for movement in moves {
            let undo = position.make_generated_move(movement);
            let gives_check = position.is_in_check(position.side_to_move());
            let child = if gives_check {
                mate_dfs(position, remaining - 1, context)
            } else {
                Ok(None)
            };
            position.unmake_move(undo);
            if let Some(child) = child? {
                let mut pv = vec![movement];
                pv.extend(child);
                return Ok(Some(pv));
            }
        }
        Ok(None)
    }
}

fn make_info(
    value: &NodeValue,
    depth: u8,
    context: &SearchContext<'_>,
    elapsed: Duration,
) -> SearchInfo {
    SearchInfo {
        best_move: value.pv.first().copied(),
        score: value.score,
        depth,
        seldepth: context.seldepth,
        nodes: context.nodes,
        elapsed,
        nps: nodes_per_second(context.nodes, elapsed),
        pv: value.pv.clone(),
        root_moves: context.last_completed_root_moves.clone(),
        stats: context.stats,
    }
}

fn nodes_per_second(nodes: u64, elapsed: Duration) -> u64 {
    let nanos = elapsed.as_nanos();
    if nanos == 0 {
        return 0;
    }
    let rate = u128::from(nodes).saturating_mul(1_000_000_000) / nanos;
    u64::try_from(rate).unwrap_or(u64::MAX)
}

fn terminal_node(_position: &Position, ply: usize) -> NodeValue {
    NodeValue::leaf(-MATE_SCORE + i32::try_from(ply).unwrap_or(i32::MAX))
}

fn score_to_transposition(score: i32, ply: usize) -> i32 {
    let ply = i32::try_from(ply).unwrap_or(i32::MAX);
    if score >= MATE_THRESHOLD {
        score.saturating_add(ply)
    } else if score <= -MATE_THRESHOLD {
        score.saturating_sub(ply)
    } else {
        score
    }
}

fn score_from_transposition(score: i32, ply: usize) -> i32 {
    let ply = i32::try_from(ply).unwrap_or(i32::MAX);
    if score >= MATE_THRESHOLD {
        score.saturating_sub(ply)
    } else if score <= -MATE_THRESHOLD {
        score.saturating_add(ply)
    } else {
        score
    }
}

fn order_piece_value(kind: PieceKind) -> i32 {
    match kind {
        PieceKind::Pawn => 100,
        PieceKind::Lance => 300,
        PieceKind::Knight => 320,
        PieceKind::Silver => 400,
        PieceKind::Gold => 500,
        PieceKind::Bishop => 700,
        PieceKind::Rook => 850,
        PieceKind::King => 10_000,
        PieceKind::PromotedPawn
        | PieceKind::PromotedLance
        | PieceKind::PromotedKnight
        | PieceKind::PromotedSilver => 520,
        PieceKind::Horse => 920,
        PieceKind::Dragon => 1_100,
    }
}

fn is_quiet(position: &Position, movement: Move) -> bool {
    position.piece_at(movement.destination()).is_none()
        && !matches!(movement, Move::Normal { promote: true, .. })
}

fn is_tactical(position: &Position, movement: Move) -> bool {
    !is_quiet(position, movement)
}

fn move_gives_check(position: &Position, movement: Move) -> bool {
    let mut child = position.clone();
    if child.make_move(movement).is_err() {
        return false;
    }
    child.is_in_check(child.side_to_move())
}

fn history_index(side: Side, movement: Move) -> usize {
    let bucket = match movement {
        Move::Normal { from, to, promote } => {
            from.index() * 81 + to.index() + usize::from(promote) * 6_561
        }
        Move::Drop { piece, to } => 13_122 + piece.index() * 81 + to.index(),
    };
    side.index() * MOVE_BUCKETS + bucket
}

fn transposition_index(hash: u64, table_len: usize) -> usize {
    let table_len = u64::try_from(table_len).unwrap_or(u64::MAX);
    usize::try_from(hash % table_len).unwrap_or_default()
}

#[cfg(all(test, feature = "handcrafted"))]
mod tests {
    use super::*;
    use crate::neural::NeuralEvaluator;
    use crate::{Hand, Piece, Square, TimeControl, TimeManager};
    use std::sync::atomic::AtomicU64;

    #[derive(Debug)]
    struct SteppingClock {
        milliseconds: AtomicU64,
        step_ms: u64,
    }

    impl SteppingClock {
        const fn new(step_ms: u64) -> Self {
            Self {
                milliseconds: AtomicU64::new(0),
                step_ms,
            }
        }
    }

    impl MonotonicClock for SteppingClock {
        fn now(&self) -> Duration {
            Duration::from_millis(self.milliseconds.fetch_add(self.step_ms, Ordering::Relaxed))
        }
    }

    fn square(file: u8, rank: u8) -> Square {
        Square::new(file, rank).expect("test square")
    }

    fn board_with_kings() -> [Option<Piece>; crate::BOARD_SQUARES] {
        let mut board = [None; crate::BOARD_SQUARES];
        board[square(9, 9).index()] = Some(Piece::new(Side::Black, PieceKind::King));
        board[square(5, 1).index()] = Some(Piece::new(Side::White, PieceKind::King));
        board
    }

    #[test]
    fn random_selector_is_seeded_and_legal() {
        let position = Position::startpos();
        let mut first = RandomMoveSelector::new(42);
        let mut second = RandomMoveSelector::new(42);

        for _ in 0..16 {
            let left = first.select(&position).expect("start position has moves");
            let right = second.select(&position).expect("start position has moves");
            assert_eq!(left, right);
            assert!(position.is_legal_move(left));
        }
    }

    #[test]
    fn fake_clock_enforces_the_twenty_second_casual_hard_limit_without_sleeping() {
        let clock = Arc::new(SteppingClock::new(1));
        let mut engine = SearchEngine::with_clock(SearchConfig::default(), clock);
        let mut plan = TimeManager::default()
            .plan(Side::Black, TimeControl::casual(), 64)
            .expect("casual plan");
        // Exercise the hard deadline independently from the optional convergence rule.
        plan.allow_stable_early_stop = false;
        let position = Position::startpos();
        let result = engine.search_managed(&position, plan, &CancellationToken::new());

        assert_eq!(result.termination, SearchTermination::TimeLimit);
        assert!(result.elapsed <= Duration::from_millis(20_000));
        assert!(
            result
                .best_move
                .is_some_and(|movement| { position.legal_moves().contains(&movement) })
        );
    }

    #[test]
    fn completed_depth_exposes_sorted_legal_root_evidence() {
        let position = Position::startpos();
        let mut engine = SearchEngine::new(SearchConfig {
            aspiration_window: 1,
            ..SearchConfig::default()
        });
        let result = engine.search(
            &position,
            SearchLimits {
                max_depth: 2,
                max_nodes: None,
                movetime: None,
            },
            &CancellationToken::new(),
        );

        assert!(!result.root_moves.is_empty());
        let unique: std::collections::HashSet<_> =
            result.root_moves.iter().map(|stat| stat.movement).collect();
        assert_eq!(
            unique.len(),
            result.root_moves.len(),
            "aspiration retries must replace evidence"
        );
        assert_eq!(unique.len(), position.legal_moves().len());
        assert!(
            result
                .root_moves
                .windows(2)
                .all(|pair| pair[0].score >= pair[1].score)
        );
        assert!(result.root_moves.iter().all(|stat| {
            stat.depth == result.depth
                && stat.pv.first() == Some(&stat.movement)
                && position.legal_moves().contains(&stat.movement)
        }));
    }

    #[test]
    fn fixed_search_is_deterministic_and_returns_a_legal_move() {
        let position = Position::startpos();
        let limits = SearchLimits {
            max_depth: 2,
            max_nodes: None,
            movetime: None,
        };
        let mut engine = SearchEngine::new(SearchConfig::default());
        let first = engine.search(&position, limits, &CancellationToken::new());
        let second = engine.search(&position, limits, &CancellationToken::new());

        assert_eq!(first.best_move, second.best_move);
        assert_eq!(first.score, second.score);
        assert_eq!(first.nodes, second.nodes);
        assert_eq!(first.pv, second.pv);
        assert!(position.is_legal_move(first.best_move.expect("best move")));
    }

    #[test]
    fn optional_neural_evaluator_is_used_and_measured_without_entering_mate_scores() {
        let evaluator = Arc::new(NeuralEvaluator::side_to_move_test_evaluator());
        let mut neural_engine = SearchEngine::with_neural(SearchConfig::default(), evaluator);
        let limits = SearchLimits {
            max_depth: 0,
            max_nodes: None,
            movetime: None,
        };
        let neural = neural_engine.search(&Position::startpos(), limits, &CancellationToken::new());
        let handcrafted = SearchEngine::new(SearchConfig::default()).search(
            &Position::startpos(),
            limits,
            &CancellationToken::new(),
        );

        assert_eq!(neural.score, 70);
        assert!(!is_mate_score(neural.score));
        assert_eq!(neural.stats.neural_inference_calls, 1);
        assert_eq!(handcrafted.stats.neural_inference_calls, 0);
        assert_eq!(handcrafted.stats.neural_inference_time, Duration::ZERO);

        let checkmate =
            crate::parse_sfen("3lkl3/3pRp3/4G4/9/9/9/9/9/K8 w - 1").expect("checkmate position");
        let mut mate_engine = SearchEngine::with_neural(
            SearchConfig::default(),
            Arc::new(NeuralEvaluator::side_to_move_test_evaluator()),
        );
        let mate = mate_engine.search(&checkmate, limits, &CancellationToken::new());
        assert_eq!(mate.score, -MATE_SCORE);
        assert_eq!(mate.stats.neural_inference_calls, 0);
    }

    #[test]
    fn disabled_quiescence_classifies_terminal_horizon_without_neural_inference() {
        let terminal_positions = [
            ("3lkl3/3pRp3/4G4/9/9/9/9/9/K8 w - 1", true),
            ("4k4/3P1P3/4K4/9/9/9/9/9/9 w - 1", false),
        ];

        for (sfen, in_check) in terminal_positions {
            let mut position = crate::parse_sfen(sfen).expect("terminal position");
            assert!(position.legal_moves().is_empty());
            assert_eq!(position.is_in_check(position.side_to_move()), in_check);
            let cancellation = CancellationToken::new();
            let clock = SystemMonotonicClock::default();
            let mut context =
                SearchContext::new(SearchLimits::default(), &cancellation, clock.now(), &clock);
            let mut engine = SearchEngine::with_neural(
                SearchConfig {
                    enable_quiescence: false,
                    ..SearchConfig::default()
                },
                Arc::new(NeuralEvaluator::side_to_move_test_evaluator()),
            );

            let result = engine
                .negamax(&mut position, 0, -INFINITY, INFINITY, 0, &mut context)
                .expect("terminal horizon node");

            assert_eq!(result.score, -MATE_SCORE);
            assert_eq!(context.stats.neural_inference_calls, 0);
        }
    }

    #[test]
    fn node_limit_is_exact_and_never_overshoots() {
        let position = Position::startpos();
        for limit in [0, 1, 2, 17, 100] {
            let mut engine = SearchEngine::new(SearchConfig::default());
            let result = engine.search(
                &position,
                SearchLimits {
                    max_depth: 8,
                    max_nodes: Some(limit),
                    movetime: None,
                },
                &CancellationToken::new(),
            );

            assert_eq!(result.nodes, limit);
            assert_eq!(result.termination, SearchTermination::NodeLimit);
            assert!(position.is_legal_move(result.best_move.expect("fallback move")));
        }
    }

    #[test]
    fn zero_node_budget_does_not_run_neural_fallback_inference() {
        let position = Position::startpos();
        let mut engine = SearchEngine::with_neural(
            SearchConfig::default(),
            Arc::new(NeuralEvaluator::side_to_move_test_evaluator()),
        );
        let result = engine.search(
            &position,
            SearchLimits {
                max_depth: 8,
                max_nodes: Some(0),
                movetime: None,
            },
            &CancellationToken::new(),
        );

        assert_eq!(result.nodes, 0);
        assert_eq!(result.score, 0);
        assert_eq!(result.termination, SearchTermination::NodeLimit);
        assert_eq!(result.stats.neural_inference_calls, 0);
        assert!(position.is_legal_move(result.best_move.expect("legal fallback move")));
    }

    #[test]
    fn pre_cancelled_search_does_not_enter_a_node() {
        let cancellation = CancellationToken::new();
        cancellation.cancel();
        let mut engine = SearchEngine::with_neural(
            SearchConfig::default(),
            Arc::new(NeuralEvaluator::side_to_move_test_evaluator()),
        );
        let result = engine.search(
            &Position::startpos(),
            SearchLimits {
                max_depth: 4,
                max_nodes: None,
                movetime: None,
            },
            &cancellation,
        );

        assert_eq!(result.nodes, 0);
        assert_eq!(result.score, 0);
        assert_eq!(result.termination, SearchTermination::Cancelled);
        assert_eq!(result.stats.neural_inference_calls, 0);
        assert!(Position::startpos().is_legal_move(result.best_move.expect("legal fallback move")));
    }

    #[test]
    fn zero_movetime_stops_before_the_first_node() {
        let mut engine = SearchEngine::with_neural(
            SearchConfig::default(),
            Arc::new(NeuralEvaluator::side_to_move_test_evaluator()),
        );
        let result = engine.search(
            &Position::startpos(),
            SearchLimits {
                max_depth: 4,
                max_nodes: None,
                movetime: Some(Duration::ZERO),
            },
            &CancellationToken::new(),
        );

        assert_eq!(result.nodes, 0);
        assert_eq!(result.termination, SearchTermination::TimeLimit);
        assert_eq!(result.stats.neural_inference_calls, 0);
        assert!(result.best_move.is_some());
    }

    #[test]
    fn each_search_ladder_switch_preserves_a_legal_result() {
        let position = Position::startpos();
        let base = SearchConfig {
            transposition_entries: 64,
            quiescence_depth: 2,
            ..SearchConfig::default()
        };
        let variants = [
            SearchConfig {
                enable_alpha_beta: false,
                ..base
            },
            SearchConfig {
                enable_iterative_deepening: false,
                ..base
            },
            SearchConfig {
                enable_transposition_table: false,
                ..base
            },
            SearchConfig {
                enable_move_ordering: false,
                ..base
            },
            SearchConfig {
                enable_killers: false,
                ..base
            },
            SearchConfig {
                enable_history: false,
                ..base
            },
            SearchConfig {
                enable_quiescence: false,
                ..base
            },
            SearchConfig {
                enable_pvs: false,
                ..base
            },
            SearchConfig {
                enable_aspiration: false,
                ..base
            },
        ];

        for config in variants {
            let mut engine = SearchEngine::new(config);
            let result = engine.search(
                &position,
                SearchLimits {
                    max_depth: 2,
                    max_nodes: None,
                    movetime: None,
                },
                &CancellationToken::new(),
            );
            assert_eq!(result.termination, SearchTermination::Completed);
            assert!(position.is_legal_move(result.best_move.expect("best move")));
        }
    }

    #[test]
    fn alpha_beta_pvs_and_table_preserve_the_full_width_result() {
        let position = Position::startpos();
        let base = SearchConfig {
            transposition_entries: 128,
            quiescence_depth: 0,
            aspiration_window: 0,
            enable_iterative_deepening: false,
            enable_move_ordering: false,
            enable_killers: false,
            enable_history: false,
            enable_quiescence: false,
            enable_aspiration: false,
            ..SearchConfig::default()
        };
        let full_width = SearchConfig {
            enable_alpha_beta: false,
            enable_transposition_table: false,
            enable_pvs: false,
            ..base
        };
        let variants = [
            SearchConfig {
                enable_transposition_table: false,
                enable_pvs: false,
                ..base
            },
            SearchConfig {
                enable_transposition_table: false,
                enable_pvs: true,
                ..base
            },
            SearchConfig {
                enable_transposition_table: true,
                enable_pvs: true,
                ..base
            },
        ];
        let limits = SearchLimits {
            max_depth: 2,
            max_nodes: None,
            movetime: None,
        };
        let expected =
            SearchEngine::new(full_width).search(&position, limits, &CancellationToken::new());

        for config in variants {
            let actual =
                SearchEngine::new(config).search(&position, limits, &CancellationToken::new());
            assert_eq!(actual.score, expected.score);
            assert_eq!(actual.best_move, expected.best_move);
        }
    }

    #[test]
    fn pruning_share_has_a_precise_nonzero_denominator() {
        let position = Position::startpos();
        let limits = SearchLimits {
            max_depth: 3,
            max_nodes: None,
            movetime: None,
        };
        let mut engine = SearchEngine::new(SearchConfig {
            transposition_entries: 256,
            quiescence_depth: 2,
            ..SearchConfig::default()
        });
        let result = engine.search(&position, limits, &CancellationToken::new());

        assert!(result.stats.candidate_moves > 0);
        assert!(result.stats.pruned_moves > 0);
        assert!(result.stats.pruned_moves <= result.stats.candidate_moves);
    }

    #[test]
    fn full_width_search_counts_candidates_without_pruning_them() {
        let position = Position::startpos();
        let mut engine = SearchEngine::new(SearchConfig {
            transposition_entries: 0,
            quiescence_depth: 0,
            enable_alpha_beta: false,
            enable_iterative_deepening: false,
            enable_transposition_table: false,
            enable_move_ordering: false,
            enable_killers: false,
            enable_history: false,
            enable_quiescence: false,
            enable_pvs: false,
            enable_aspiration: false,
            ..SearchConfig::default()
        });
        let result = engine.search(
            &position,
            SearchLimits {
                max_depth: 2,
                max_nodes: None,
                movetime: None,
            },
            &CancellationToken::new(),
        );

        assert!(result.stats.candidate_moves > 0);
        assert_eq!(result.stats.pruned_moves, 0);
    }

    #[test]
    fn callback_only_reports_completed_depths_in_order() {
        let mut engine = SearchEngine::new(SearchConfig {
            quiescence_depth: 2,
            ..SearchConfig::default()
        });
        let mut reports = Vec::new();
        let result = engine.search_with_callback(
            &Position::startpos(),
            SearchLimits {
                max_depth: 3,
                max_nodes: None,
                movetime: None,
            },
            &CancellationToken::new(),
            |info| reports.push((info.depth, info.nodes)),
        );

        assert_eq!(result.depth, 3);
        assert_eq!(reports.len(), 3);
        assert!(
            reports
                .windows(2)
                .all(|pair| { pair[0].0 < pair[1].0 && pair[0].1 < pair[1].1 })
        );
    }

    #[test]
    fn search_handles_the_largest_serialized_move_number_without_mutation() {
        let start = Position::startpos();
        let position = Position::from_parts(
            *start.board(),
            *start.hands(),
            start.side_to_move(),
            u32::MAX,
        )
        .expect("position");
        let mut engine = SearchEngine::new(SearchConfig {
            transposition_entries: 0,
            quiescence_depth: 0,
            ..SearchConfig::default()
        });
        let result = engine.search(
            &position,
            SearchLimits {
                max_depth: 1,
                max_nodes: None,
                movetime: None,
            },
            &CancellationToken::new(),
        );

        assert!(position.is_legal_move(result.best_move.expect("best move")));
        assert_eq!(position.move_number(), u32::MAX);
    }

    #[test]
    fn search_prefers_a_free_rook_capture() {
        let mut board = board_with_kings();
        board[square(5, 5).index()] = Some(Piece::new(Side::Black, PieceKind::Silver));
        board[square(4, 4).index()] = Some(Piece::new(Side::White, PieceKind::Rook));
        let position =
            Position::from_parts(board, [Hand::default(); 2], Side::Black, 1).expect("position");
        let mut engine = SearchEngine::new(SearchConfig::default());
        let result = engine.search(
            &position,
            SearchLimits {
                max_depth: 1,
                max_nodes: None,
                movetime: None,
            },
            &CancellationToken::new(),
        );

        assert_eq!(
            result.best_move,
            Some(Move::Normal {
                from: square(5, 5),
                to: square(4, 4),
                promote: false,
            })
        );
    }

    #[test]
    fn full_position_check_rejects_direct_table_collisions() {
        let mut engine = SearchEngine::new(SearchConfig {
            transposition_entries: 1,
            ..SearchConfig::default()
        });
        let first = Position::startpos();
        let mut second = first.clone();
        let movement = second.legal_moves()[0];
        let _ = second.make_generated_move(movement);
        engine.store_transposition(&first, 2, 123, Bound::Exact, None);
        let cancellation = CancellationToken::new();
        let clock = SystemMonotonicClock::default();
        let mut context =
            SearchContext::new(SearchLimits::default(), &cancellation, clock.now(), &clock);

        assert!(engine.probe_transposition(&second, &mut context).is_none());
        assert_eq!(context.stats.tt_collisions, 1);
        assert_eq!(context.stats.tt_hits, 0);
    }

    #[test]
    fn mate_search_finds_a_one_move_mate() {
        let mut board = board_with_kings();
        board[square(5, 3).index()] = Some(Piece::new(Side::Black, PieceKind::Gold));
        board[square(4, 1).index()] = Some(Piece::new(Side::White, PieceKind::Lance));
        board[square(6, 1).index()] = Some(Piece::new(Side::White, PieceKind::Lance));
        let mut hands = [Hand::default(); 2];
        hands[Side::Black.index()].set(crate::HandPiece::Gold, 1);
        let position = Position::from_parts(board, hands, Side::Black, 1).expect("position");
        let mut engine = SearchEngine::new(SearchConfig::default());
        let result = engine.find_mate(&position, 1, None, &CancellationToken::new());

        assert!(result.found);
        assert_eq!(result.pv.len(), 1);
    }

    #[test]
    fn mate_search_obeys_its_node_limit() {
        let mut engine = SearchEngine::new(SearchConfig::default());
        let result = engine.find_mate(&Position::startpos(), 7, Some(0), &CancellationToken::new());

        assert!(!result.found);
        assert_eq!(result.nodes, 0);
        assert_eq!(result.termination, SearchTermination::NodeLimit);
    }

    #[test]
    fn checked_quiescence_horizon_returns_a_finite_score() {
        let mut board = [None; crate::BOARD_SQUARES];
        board[square(5, 9).index()] = Some(Piece::new(Side::Black, PieceKind::King));
        board[square(1, 1).index()] = Some(Piece::new(Side::White, PieceKind::King));
        board[square(5, 1).index()] = Some(Piece::new(Side::White, PieceKind::Rook));
        let mut position =
            Position::from_parts(board, [Hand::default(); 2], Side::Black, 1).expect("position");
        let cancellation = CancellationToken::new();
        let clock = SystemMonotonicClock::default();
        let mut context =
            SearchContext::new(SearchLimits::default(), &cancellation, clock.now(), &clock);
        let mut engine = SearchEngine::new(SearchConfig::default());
        let result = engine
            .quiescence(&mut position, -INFINITY, INFINITY, 0, 0, &mut context)
            .expect("unbounded quiescence");

        assert!(position.is_in_check(Side::Black));
        assert!(result.score > -MATE_THRESHOLD);
        assert!(result.score < MATE_THRESHOLD);
    }

    #[test]
    fn checked_quiescence_horizon_searches_the_only_legal_evasion() {
        let mut board = [None; crate::BOARD_SQUARES];
        board[square(9, 9).index()] = Some(Piece::new(Side::Black, PieceKind::King));
        board[square(7, 7).index()] = Some(Piece::new(Side::White, PieceKind::King));
        board[square(8, 8).index()] = Some(Piece::new(Side::White, PieceKind::Bishop));
        board[square(9, 1).index()] = Some(Piece::new(Side::White, PieceKind::Rook));
        let mut position =
            Position::from_parts(board, [Hand::default(); 2], Side::Black, 1).expect("position");
        let legal_moves = position.legal_moves();
        assert_eq!(legal_moves.len(), 1);
        assert!(position.is_in_check(Side::Black));

        let cancellation = CancellationToken::new();
        let clock = SystemMonotonicClock::default();
        let mut context =
            SearchContext::new(SearchLimits::default(), &cancellation, clock.now(), &clock);
        let mut engine = SearchEngine::with_neural(
            SearchConfig::default(),
            Arc::new(NeuralEvaluator::side_to_move_test_evaluator()),
        );
        let result = engine
            .quiescence(&mut position, -INFINITY, INFINITY, 0, 0, &mut context)
            .expect("bounded quiescence");

        assert!(!result.pv.is_empty());
        assert!(legal_moves.contains(&result.pv[0]));
        assert_eq!(result.score, -10);
        assert_eq!(context.stats.neural_inference_calls, 1);
        assert_ne!(
            result.score, 70,
            "the illegal stand-pat score must not survive"
        );
    }

    #[test]
    fn checked_quiescence_hard_ply_guard_never_uses_stand_pat() {
        let mut board = [None; crate::BOARD_SQUARES];
        board[square(9, 9).index()] = Some(Piece::new(Side::Black, PieceKind::King));
        board[square(7, 7).index()] = Some(Piece::new(Side::White, PieceKind::King));
        board[square(8, 8).index()] = Some(Piece::new(Side::White, PieceKind::Bishop));
        board[square(9, 1).index()] = Some(Piece::new(Side::White, PieceKind::Rook));
        let mut position =
            Position::from_parts(board, [Hand::default(); 2], Side::Black, 1).expect("position");
        let cancellation = CancellationToken::new();
        let clock = SystemMonotonicClock::default();
        let mut context =
            SearchContext::new(SearchLimits::default(), &cancellation, clock.now(), &clock);
        let mut engine = SearchEngine::with_neural(
            SearchConfig::default(),
            Arc::new(NeuralEvaluator::side_to_move_test_evaluator()),
        );

        let result = engine
            .quiescence(
                &mut position,
                -INFINITY,
                INFINITY,
                MAX_SEARCH_PLY,
                0,
                &mut context,
            )
            .expect("hard-bounded quiescence");

        assert!(position.is_in_check(Side::Black));
        assert_eq!(result.score, 0);
        assert_eq!(context.stats.neural_inference_calls, 0);
        assert_ne!(result.score, 70, "the checked stand-pat score is illegal");
    }

    #[test]
    fn transposition_budget_uses_the_real_full_position_slot_size() {
        let slot_size = SearchEngine::transposition_entry_size_bytes();
        let budget = 32 * 1024 * 1024;
        let entries = SearchEngine::transposition_entries_for_bytes(budget);

        assert!(slot_size >= size_of::<Position>());
        assert!(entries.saturating_mul(slot_size) <= budget);
        assert!(budget - entries * slot_size < slot_size);
        assert_eq!(
            entries,
            SearchEngine::transposition_entries_for_megabytes(32)
        );
    }

    #[test]
    fn mate_score_round_trips_through_transposition_encoding() {
        for ply in [0, 1, 17, 255] {
            for score in [MATE_SCORE - 7, -MATE_SCORE + 7, 345, -921] {
                assert_eq!(
                    score_from_transposition(score_to_transposition(score, ply), ply),
                    score
                );
            }
        }
    }

    #[test]
    fn quiet_check_is_tactical_for_casual_stability() {
        let position = crate::parse_sfen("4k4/9/9/9/9/9/9/4R4/4K4 b - 1").unwrap();
        let movement = crate::parse_usi_move("5h5b").unwrap();

        assert!(position.legal_moves().contains(&movement));
        assert!(!is_tactical(&position, movement));
        assert!(move_gives_check(&position, movement));
    }

    #[test]
    fn neural_score_semantics_are_explicit() {
        let position = Position::startpos();
        let config = SearchConfig::default();
        let handcrafted = evaluate(&position, &config.evaluation);
        let evaluator = Arc::new(NeuralEvaluator::side_to_move_test_evaluator());
        let learned = evaluator.evaluate(&position);
        let score = |mode| {
            let engine = SearchEngine::with_neural_mode(config, Arc::clone(&evaluator), mode);
            let cancellation = CancellationToken::new();
            let clock = SystemMonotonicClock::default();
            let mut context =
                SearchContext::new(SearchLimits::default(), &cancellation, clock.now(), &clock);
            engine
                .evaluate_position(&position, &mut context)
                .expect("test evaluator is valid")
        };

        assert_eq!(score(NeuralEvaluationMode::PureValue), learned);
        assert_eq!(score(NeuralEvaluationMode::Residual), handcrafted + learned);
        assert_eq!(
            score(NeuralEvaluationMode::Composite),
            handcrafted.saturating_add(learned) / 2
        );
    }

    #[test]
    fn profile_counters_measure_the_real_evaluation_call_path() {
        let position = Position::startpos();
        let counted = |engine: &SearchEngine| {
            let cancellation = CancellationToken::new();
            let clock = SystemMonotonicClock::default();
            let mut context =
                SearchContext::new(SearchLimits::default(), &cancellation, clock.now(), &clock);
            engine
                .evaluate_position(&position, &mut context)
                .expect("test evaluator is valid");
            let stats = context.stats;
            (
                stats.learned_eval_calls,
                stats.handcrafted_eval_calls,
                stats.residual_eval_calls,
                stats.composite_eval_calls,
            )
        };

        let neural = Arc::new(NeuralEvaluator::side_to_move_test_evaluator());
        let (learned, handcrafted, residual, composite) = counted(&SearchEngine::with_neural_mode(
            SearchConfig::default(),
            Arc::clone(&neural),
            NeuralEvaluationMode::PureValue,
        ));
        assert_eq!((learned, handcrafted, residual, composite), (1, 0, 0, 0));

        let (learned, handcrafted, residual, composite) = counted(&SearchEngine::with_neural_mode(
            SearchConfig::default(),
            Arc::clone(&neural),
            NeuralEvaluationMode::Residual,
        ));
        assert_eq!((learned, handcrafted, residual, composite), (1, 1, 1, 0));

        let (learned, handcrafted, residual, composite) = counted(&SearchEngine::with_neural_mode(
            SearchConfig::default(),
            Arc::clone(&neural),
            NeuralEvaluationMode::Composite,
        ));
        assert_eq!((learned, handcrafted, residual, composite), (1, 1, 0, 1));

        let (learned, handcrafted, residual, composite) =
            counted(&SearchEngine::new(SearchConfig::default()));
        assert_eq!((learned, handcrafted, residual, composite), (0, 1, 0, 0));
    }

    #[test]
    fn disabled_evaluation_terms_leave_no_handcrafted_contribution() {
        let position = Position::startpos();
        assert_eq!(evaluate(&position, &EvaluationConfig::disabled()), 0);
        let engine = SearchEngine::new(SearchConfig {
            evaluation: EvaluationConfig::disabled(),
            runtime_profile: RuntimeProfile::PureLearned,
            ..SearchConfig::default()
        });
        assert_eq!(engine.config.runtime_profile, RuntimeProfile::PureLearned);
        assert_eq!(engine.config.evaluation, EvaluationConfig::disabled());
    }

    #[test]
    fn standard_runtime_proof_is_never_valid_pure_learned_evidence() {
        let engine = SearchEngine::new(SearchConfig::default());
        let stats = SearchStats::default();
        let proof = engine.runtime_proof(stats, "a".repeat(64));
        assert_eq!(proof.profile, "standard");
        assert!(!proof.valid_pure_learned());
    }
}

#[cfg(test)]
mod pure_history_tests {
    use super::*;
    use std::io::Read;

    fn engine() -> SearchEngine {
        let mut bytes = Vec::new();
        flate2::read::GzDecoder::new(
            &include_bytes!("../../../tests/fixtures/osaval02/pure-history.osaval02.gz")[..],
        )
        .read_to_end(&mut bytes)
        .unwrap();
        let model = Arc::new(crate::Osaval02Evaluator::from_bytes(&bytes).unwrap());
        let hash = model.identity().artifact_sha256.clone();
        SearchEngine::with_pure_learned(
            SearchConfig {
                enable_quiescence: false,
                transposition_entries: 64,
                ..SearchConfig::default()
            },
            model,
            &hash,
        )
        .unwrap()
    }
    fn initial() -> Position {
        crate::parse_sfen("4k4/9/9/9/9/9/9/9/4K4 b - 1").unwrap()
    }
    fn cycle() -> Vec<Move> {
        ["5i6i", "5a6a", "6i5i", "6a5a"]
            .iter()
            .map(|m| crate::parse_usi_move(m).unwrap())
            .collect()
    }
    fn limits() -> SearchLimits {
        SearchLimits {
            max_depth: 0,
            max_nodes: Some(32),
            movetime: None,
        }
    }

    #[test]
    fn pure_search_distinguishes_terminal_evaluated_and_unstarted_results() {
        let mut engine = engine();
        for (sfen, outcome) in [
            (
                "3lkl3/3pRp3/4G4/9/9/9/9/9/K8 w - 1",
                SearchOutcome::Checkmate,
            ),
            (
                "4k4/3P1P3/4K4/9/9/9/9/9/9 w - 1",
                SearchOutcome::NoLegalMoves,
            ),
        ] {
            let position = crate::parse_sfen(sfen).unwrap();
            let result = engine.search(
                &position,
                SearchLimits {
                    max_depth: 64,
                    max_nodes: Some(2000),
                    movetime: None,
                },
                &CancellationToken::new(),
            );
            assert_eq!(result.outcome, outcome);
            assert_eq!(result.score, -MATE_SCORE);
            assert_eq!(result.best_move, None);
            assert_eq!(result.nodes, 0);
            let proof = engine.runtime_proof(result.stats, "");
            assert!(proof.valid_pure_search(&result));
            assert!(!proof.valid_pure_learned());
            let mut forbidden = proof.clone();
            forbidden.book_hits = 1;
            assert!(!forbidden.valid_pure_search(&result));
            let mut misleading = result.clone();
            misleading.outcome = SearchOutcome::Evaluated;
            assert!(!proof.valid_pure_search(&misleading));
        }
        let position = initial();
        let result = engine.search(&position, limits(), &CancellationToken::new());
        assert_eq!(result.outcome, SearchOutcome::Evaluated);
        assert!(result.outcome.has_score());
        let proof = engine.runtime_proof(result.stats, "");
        assert!(proof.valid_pure_search(&result));
        assert!(proof.valid_pure_learned());

        for (nodes, movetime, cancelled, expected) in [
            (
                Some(0),
                None,
                false,
                SearchOutcome::NodeLimitBeforeEvaluation,
            ),
            (
                Some(32),
                Some(Duration::ZERO),
                false,
                SearchOutcome::TimeLimitBeforeEvaluation,
            ),
            (
                Some(32),
                None,
                true,
                SearchOutcome::CancelledBeforeEvaluation,
            ),
        ] {
            let token = CancellationToken::new();
            if cancelled {
                token.cancel();
            }
            let result = engine.search(
                &position,
                SearchLimits {
                    max_depth: 4,
                    max_nodes: nodes,
                    movetime,
                },
                &token,
            );
            assert_eq!(result.outcome, expected);
            assert!(!result.outcome.has_score());
            assert!(position.is_legal_move(result.best_move.unwrap()));
            let proof = engine.runtime_proof(result.stats, "");
            assert!(!proof.valid_pure_learned());
            assert!(proof.valid_pure_search(&result));
            let mut wrong_reason = result.clone();
            wrong_reason.outcome = SearchOutcome::Checkmate;
            assert!(!proof.valid_pure_search(&wrong_reason));
        }
    }

    #[test]
    fn pure_perpetual_check_result_needs_no_inference() {
        let initial = crate::parse_sfen("4k4/5R3/9/9/9/9/9/9/K8 b - 1").unwrap();
        let moves: Vec<_> = ["4b5b", "5a4a", "5b4b", "4a5a"]
            .repeat(3)
            .iter()
            .map(|m| crate::parse_usi_move(m).unwrap())
            .collect();
        let mut position = initial.clone();
        for movement in &moves {
            position.make_move(*movement).unwrap();
        }
        let mut engine = engine();
        engine.set_pure_history(&initial, &moves).unwrap();
        let result = engine.search(&position, limits(), &CancellationToken::new());
        assert_eq!(result.outcome, SearchOutcome::PerpetualCheck);
        assert_eq!(result.score, -MATE_SCORE);
        assert!(
            engine
                .runtime_proof(result.stats, "")
                .valid_pure_search(&result)
        );
    }

    #[test]
    fn osaval02_search_history_changes_leaf_score_and_unmakes_exactly() {
        let mut engine = engine();
        let root = initial();
        engine.set_pure_history(&root, &[]).unwrap();
        let result = engine.search(&root, limits(), &CancellationToken::new());
        assert!(engine.runtime_proof(result.stats, "").valid_pure_learned());
        let initial_score = result.score;
        let initial_history = engine.a1_history();
        assert!(initial_history.available);
        assert_eq!(initial_history.repetition_count, 1);
        let mut position = root.clone();
        let mut previous = Vec::new();
        for movement in cycle() {
            position.make_move(movement).unwrap();
            previous.push(engine.a1_push(movement, &position));
        }
        assert_eq!(engine.a1_history().repetition_count, 2);
        let token = CancellationToken::new();
        let clock = Arc::clone(&engine.clock);
        let mut context = SearchContext::new(limits(), &token, clock.now(), clock.as_ref());
        let score = engine.evaluate_position(&position, &mut context).unwrap();
        assert!(
            score > initial_score + 100,
            "synthetic model scores its repetition-count input"
        );
        assert!(engine.runtime_proof(context.stats, "").valid_pure_learned());
        for state in previous.into_iter().rev() {
            engine.a1_pop(state);
        }
        assert_eq!(engine.a1_history(), initial_history);
        assert_eq!(engine.a1_positions, vec![root]);
    }

    #[test]
    fn osaval02_pure_repetition_and_position_only_tt_are_isolated() {
        let root = initial();
        let mut engine = engine();
        let moves = cycle().repeat(3);
        let mut position = root.clone();
        for movement in &moves {
            position.make_move(*movement).unwrap();
        }
        engine.set_pure_history(&root, &moves).unwrap();
        let terminal = engine.search(&position, limits(), &CancellationToken::new());
        assert_eq!(terminal.best_move, None);
        assert_eq!(terminal.score, 0);
        assert_eq!(terminal.stats.learned_eval_calls, 0);
        assert_eq!(terminal.outcome, SearchOutcome::Repetition);
        assert!(
            engine
                .runtime_proof(terminal.stats, "")
                .valid_pure_search(&terminal)
        );
        engine.set_pure_history(&root, &[]).unwrap();
        engine.search(&root, limits(), &CancellationToken::new());
        engine.store_transposition(&root, 20, 12345, Bound::Exact, None);
        let token = CancellationToken::new();
        let clock = Arc::clone(&engine.clock);
        let mut context = SearchContext::new(
            SearchLimits {
                max_depth: 1,
                max_nodes: Some(32),
                movetime: None,
            },
            &token,
            clock.now(),
            clock.as_ref(),
        );
        let value = engine
            .negamax(&mut root.clone(), 1, -INFINITY, INFINITY, 1, &mut context)
            .unwrap();
        assert!(context.stats.tt_hits > 0);
        assert_ne!(
            value.score, 12345,
            "history-blind TT score must not cut off pure inference"
        );
        assert!(context.stats.learned_eval_calls > 0);
    }

    #[test]
    fn explicit_history_mismatch_or_rejected_history_never_becomes_unknown_history() {
        let root = initial();
        let mut engine = engine();
        let movement = cycle()[0];
        engine.set_pure_history(&root, &[movement]).unwrap();
        let rejected = engine.search(&root, limits(), &CancellationToken::new());
        assert_eq!(rejected.termination, SearchTermination::EvaluationError);
        assert_eq!(rejected.best_move, None);
        assert_eq!(rejected.stats.learned_eval_calls, 0);
        assert_eq!(rejected.outcome, SearchOutcome::EvaluationError);
        assert!(
            !engine
                .runtime_proof(rejected.stats, "")
                .valid_pure_search(&rejected)
        );
        assert!(
            engine
                .set_pure_history(&root, &[crate::parse_usi_move("5i5a").unwrap()])
                .is_err()
        );
        assert_eq!(
            engine
                .search(&root, limits(), &CancellationToken::new())
                .termination,
            SearchTermination::EvaluationError
        );
        engine.set_pure_history(&root, &[]).unwrap();
        let wrong_ply = crate::parse_sfen("4k4/9/9/9/9/9/9/9/4K4 b - 99").unwrap();
        assert_eq!(
            engine
                .search(&wrong_ply, limits(), &CancellationToken::new())
                .termination,
            SearchTermination::EvaluationError
        );
        assert_ne!(
            engine
                .search(&root, limits(), &CancellationToken::new())
                .termination,
            SearchTermination::EvaluationError
        );
        let mut standalone = super::pure_history_tests::engine();
        assert_ne!(
            standalone
                .search(&root, limits(), &CancellationToken::new())
                .termination,
            SearchTermination::EvaluationError
        );
    }
}
