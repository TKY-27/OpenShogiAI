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

use crate::{EvaluationConfig, Move, NeuralEvaluator, PieceKind, Position, Side, evaluate};

/// Base score used for checkmates. Distance in plies is subtracted from this value.
pub const MATE_SCORE: i32 = 30_000;
/// Scores beyond this threshold encode a forced mate rather than a static evaluation.
pub const MATE_THRESHOLD: i32 = MATE_SCORE - 1_000;

const INFINITY: i32 = 32_000;
const MAX_SEARCH_PLY: usize = 256;
const MOVE_BUCKETS: usize = 13_689;

/// Returns whether a score encodes a forced mate.
#[must_use]
pub const fn is_mate_score(score: i32) -> bool {
    score >= MATE_THRESHOLD || score <= -MATE_THRESHOLD
}

/// Cooperative cancellation shared between a search thread and its controller.
#[derive(Clone, Debug, Default)]
pub struct CancellationToken {
    cancelled: Arc<AtomicBool>,
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
    pub evaluation: EvaluationConfig,
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
            evaluation: EvaluationConfig::default(),
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
    NodeLimit,
    TimeLimit,
    Cancelled,
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
    pub stats: SearchStats,
    pub termination: SearchTermination,
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
    neural: Option<Arc<NeuralEvaluator>>,
    transposition_table: Vec<Option<TranspositionEntry>>,
    killers: Vec<[Option<Move>; 2]>,
    history: Vec<i32>,
}

impl SearchEngine {
    #[must_use]
    pub fn new(config: SearchConfig) -> Self {
        Self {
            config,
            neural: None,
            transposition_table: vec![None; config.transposition_entries],
            killers: vec![[None; 2]; MAX_SEARCH_PLY],
            history: vec![0; 2 * MOVE_BUCKETS],
        }
    }

    /// Builds a search engine that uses one immutable neural evaluator for static scores.
    ///
    /// Mate and stalemate scores continue to be assigned by search and never by the model.
    #[must_use]
    pub fn with_neural(config: SearchConfig, neural: Arc<NeuralEvaluator>) -> Self {
        let mut engine = Self::new(config);
        engine.neural = Some(neural);
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

    /// Searches without receiving intermediate iteration reports.
    pub fn search(
        &mut self,
        position: &Position,
        limits: SearchLimits,
        cancellation: &CancellationToken,
    ) -> SearchResult {
        self.search_with_callback(position, limits, cancellation, |_| {})
    }

    /// Searches and reports each fully completed iterative-deepening iteration.
    pub fn search_with_callback(
        &mut self,
        position: &Position,
        limits: SearchLimits,
        cancellation: &CancellationToken,
        mut callback: impl FnMut(&SearchInfo),
    ) -> SearchResult {
        self.reset_for_search();
        let started = Instant::now();
        let legal_moves = position.legal_moves();
        let fallback = legal_moves.first().copied();
        let mut context = SearchContext::new(limits, cancellation, started);
        let mut completed = NodeValue {
            score: if legal_moves.is_empty() {
                if position.is_in_check(position.side_to_move()) {
                    -MATE_SCORE
                } else {
                    0
                }
            } else if context.check_termination().is_err() {
                // A legal fallback still lets callers emit a protocol-compliant move, but a
                // cancelled/zero-budget search must not run an uncounted neural inference.
                0
            } else {
                self.evaluate_position(position, &mut context.stats)
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
                let info = make_info(&completed, completed_depth, &context, started.elapsed());
                callback(&info);
                if is_mate_score(completed.score) {
                    break;
                }
            }
        }

        let elapsed = started.elapsed();
        SearchResult {
            best_move: completed.pv.first().copied().or(fallback),
            score: completed.score,
            depth: completed_depth,
            seldepth: context.seldepth,
            nodes: context.nodes,
            elapsed,
            nps: nodes_per_second(context.nodes, elapsed),
            pv: completed.pv,
            stats: context.stats,
            termination: context.termination.unwrap_or(SearchTermination::Completed),
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

    fn reset_for_search(&mut self) {
        self.transposition_table.fill(None);
        self.killers.fill([None; 2]);
        self.history.fill(0);
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

        // Legal terminal classification must precede every horizon/static evaluation. Besides
        // preserving mate and stalemate semantics when quiescence is disabled, this guarantees
        // that terminal nodes never invoke the optional neural evaluator.
        if depth == 0 || ply >= MAX_SEARCH_PLY {
            let moves = position.legal_moves();
            if moves.is_empty() {
                return Ok(terminal_node(position, ply));
            }
            return Ok(NodeValue {
                score: self.evaluate_position(position, &mut context.stats),
                pv: Vec::new(),
            });
        }

        let alpha_original = alpha;
        let mut tt_move = None;
        if self.config.enable_transposition_table
            && let Some(entry) = self.probe_transposition(position, context)
        {
            tt_move = entry.best_move;
            if entry.depth >= depth {
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
            let quiet = is_quiet(position, movement);
            let undo = position.make_generated_move(movement);
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
            let child = child?;
            searched_moves += 1;
            let score = -child.score;

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
        let in_check = position.is_in_check(position.side_to_move());
        let mut moves = position.legal_moves();
        if moves.is_empty() {
            return Ok(NodeValue::leaf(if in_check {
                -MATE_SCORE + i32::try_from(ply).unwrap_or(i32::MAX)
            } else {
                0
            }));
        }
        // Checked nodes have no legal stand-pat; skip unused inference and metrics.
        let stand_pat = if in_check {
            -INFINITY
        } else {
            self.evaluate_position(position, &mut context.stats)
        };
        if ply >= MAX_SEARCH_PLY {
            return Ok(NodeValue::leaf(if in_check {
                0_i32.max(alpha).min(beta)
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
            let child = self.quiescence(position, -beta, -alpha, ply + 1, child_remaining, context);
            position.unmake_move(undo);
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

    fn evaluate_position(&self, position: &Position, stats: &mut SearchStats) -> i32 {
        let Some(neural) = &self.neural else {
            return evaluate(position, &self.config.evaluation);
        };
        let started = Instant::now();
        let score = neural.evaluate(position);
        stats.neural_inference_calls = stats.neural_inference_calls.saturating_add(1);
        stats.neural_inference_time = stats
            .neural_inference_time
            .saturating_add(started.elapsed());
        score
    }
}

struct SearchContext<'a> {
    limits: SearchLimits,
    cancellation: &'a CancellationToken,
    started: Instant,
    nodes: u64,
    seldepth: u8,
    stats: SearchStats,
    termination: Option<SearchTermination>,
    root_partial: Option<NodeValue>,
}

impl<'a> SearchContext<'a> {
    fn new(limits: SearchLimits, cancellation: &'a CancellationToken, started: Instant) -> Self {
        Self {
            limits,
            cancellation,
            started,
            nodes: 0,
            seldepth: 0,
            stats: SearchStats::default(),
            termination: None,
            root_partial: None,
        }
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
            .is_some_and(|limit| self.started.elapsed() >= limit)
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

fn terminal_node(position: &Position, ply: usize) -> NodeValue {
    NodeValue::leaf(if position.is_in_check(position.side_to_move()) {
        -MATE_SCORE + i32::try_from(ply).unwrap_or(i32::MAX)
    } else {
        0
    })
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neural::NeuralEvaluator;
    use crate::{Hand, Piece, Square};

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
            ("3lkl3/3pRp3/4G4/9/9/9/9/9/K8 w - 1", -MATE_SCORE),
            ("4k4/3P1P3/4K4/9/9/9/9/9/9 w - 1", 0),
        ];

        for (sfen, expected_score) in terminal_positions {
            let mut position = crate::parse_sfen(sfen).expect("terminal position");
            assert!(position.legal_moves().is_empty());
            assert_eq!(
                position.is_in_check(position.side_to_move()),
                expected_score != 0
            );
            let cancellation = CancellationToken::new();
            let mut context =
                SearchContext::new(SearchLimits::default(), &cancellation, Instant::now());
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

            assert_eq!(result.score, expected_score);
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
        let started = Instant::now();
        let cancellation = CancellationToken::new();
        let mut context = SearchContext::new(SearchLimits::default(), &cancellation, started);

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
        let started = Instant::now();
        let mut context = SearchContext::new(SearchLimits::default(), &cancellation, started);
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
        let started = Instant::now();
        let mut context = SearchContext::new(SearchLimits::default(), &cancellation, started);
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
        let mut context =
            SearchContext::new(SearchLimits::default(), &cancellation, Instant::now());
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
}
