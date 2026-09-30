//! Native parallel search: partitioned root iterations.
//!
//! The controller runs the ordinary iterative-deepening loop and owns the published
//! result, so `Threads=1` and USI semantics are unchanged. Interior TT cutoffs cannot be
//! shared for the pure runtime because its scores depend on repetition history a
//! position-only table cannot represent, so the root iteration is the parallel unit:
//! every worker — the controller included — claims root moves of the current depth and
//! searches them against one shared alpha. Published scores (exact or a sound bound)
//! raise that alpha for everyone;
//! a fail-low against an older, lower alpha can never become the best move. The
//! controller merges published results into the same root evidence a serial iteration
//! produces, and joins the helpers (scoped threads) after setting the stop flag.

use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicI32, AtomicU64, AtomicUsize, Ordering},
    },
    time::Duration,
};

use crate::{Move, RootMoveStat, SearchStats};

/// Short worker wait while waiting for the next iteration to be published.
pub(crate) const ITERATION_POLL: Duration = Duration::from_micros(200);

/// Defensive upper bound on total workers (controller plus helpers) for one search.
pub(crate) const MAX_WORKERS: usize = 128;

/// Reservation unit of the shared node budget. A worker tops up its local credit in
/// chunks of this size (bounded by the remaining budget) instead of touching shared
/// state per node. Every reserving context — the controller's iterative search, each
/// helper's per-claim context, and the controller's transient mate-prepass context —
/// pays its work from the same pool. Grants are capped by the ungranted remainder, so
/// outstanding credit never exceeds the declared budget, and every visited node
/// consumes exactly one credit: the visited total cannot exceed the budget at all. It
/// can undershoot by up to one reservation unit per reserving context (credit
/// reserved but unspent when the search ends; every context returns its unspent
/// credit on drop, including the prepass). The exhaustion latch fires on the first
/// refused reservation, so a drained pool ends the search deterministically.
/// `SearchResult.nodes` always reports the exact visited total.
pub(crate) const NODE_RESERVATION_CHUNK: u64 = 64;

/// Coordination state for one bounded parallel search.
pub(crate) struct ParallelSearch {
    stop: AtomicBool,
    /// Helper liveness; the controller waits on missing results only while this is > 0.
    alive_helpers: AtomicUsize,
    /// Set by a helper whose inference failed: the search cannot produce trustworthy
    /// output, so the controller must fail closed instead of publishing a fallback.
    evaluation_failed: AtomicBool,
    cursor: AtomicUsize,
    shared_alpha: AtomicI32,
    /// Set once the iteration's full-window search has published a real alpha; scouts
    /// wait for it so they never run against a not-yet-raised shared alpha.
    primed: AtomicBool,
    /// Swap-to-claim the iteration's single full-window (first-move) search.
    pub(crate) first_full_window: AtomicBool,
    iteration: Mutex<IterationState>,
    /// The engine clock every worker shares, so one absolute deadline means the same
    /// reading in every thread.
    clock: Arc<dyn crate::MonotonicClock>,
    /// Absolute end-of-search instant on the shared clock. Every worker, iteration and
    /// wait of one `go` obeys it; nothing restarts a per-root-move budget.
    deadline: Option<Duration>,
    /// Total node budget shared by all workers; `None` for budgetless searches.
    node_budget: Option<u64>,
    /// Node credit handed out minus returned; the budget guard compares it to
    /// `node_budget` when workers top up their local reservation. Failed or partial
    /// grants subtract their ungrantable remainder immediately, so the counter never
    /// accumulates phantom credit and the effective budget stays the declared one.
    nodes_granted: AtomicU64,
    /// Latched once a worker's reservation found the shared budget empty. The budget
    /// is a search-wide cap: when the pool is drained, every claim after that point is
    /// refused, so abandoned claims can leave no slot the controller would wait on —
    /// the collect wait observes the latch and ends the search with `NodeLimit`.
    budget_exhausted: AtomicBool,
    /// Exact total of nodes visited by helper contexts, flushed when each per-claim
    /// context ends (including cancelled, abandoned and unwound claims).
    helper_nodes: AtomicU64,
    /// Exact total of helper search statistics, flushed with the nodes.
    helper_stats: Mutex<SearchStats>,
    /// Total root-move claims ever made, for diagnostics and test synchronization.
    claims: AtomicU64,
}

struct IterationState {
    epoch: u64,
    depth: u8,
    alpha: i32,
    beta: i32,
    moves: Vec<Move>,
    results: Vec<Option<(RootMoveStat, bool)>>,
}

impl ParallelSearch {
    pub(crate) fn new(
        clock: Arc<dyn crate::MonotonicClock>,
        deadline: Option<Duration>,
        node_budget: Option<u64>,
    ) -> Self {
        Self {
            stop: AtomicBool::new(false),
            alive_helpers: AtomicUsize::new(0),
            evaluation_failed: AtomicBool::new(false),
            cursor: AtomicUsize::new(0),
            shared_alpha: AtomicI32::new(-i32::MAX),
            primed: AtomicBool::new(false),
            first_full_window: AtomicBool::new(false),
            iteration: Mutex::new(IterationState {
                epoch: 0,
                depth: 0,
                alpha: 0,
                beta: 0,
                moves: Vec::new(),
                results: Vec::new(),
            }),
            clock,
            deadline,
            node_budget,
            nodes_granted: AtomicU64::new(0),
            budget_exhausted: AtomicBool::new(false),
            helper_nodes: AtomicU64::new(0),
            helper_stats: Mutex::new(SearchStats::default()),
            claims: AtomicU64::new(0),
        }
    }

    /// The stop flag plus every bounded exit of one search in one call: explicit stop,
    /// a superseding search's cancellation of the shared token and the absolute
    /// deadline all release primer waits through here.
    pub(crate) fn should_abort(&self) -> bool {
        self.is_stopped() || self.deadline_exceeded() || self.budget_exhausted()
    }

    /// Whether a worker's reservation found the shared node budget empty. Latched for
    /// the rest of the search: credit may flow back later, but a drained pool means the
    /// declared cap was reached, and no claim after that point can make progress.
    pub(crate) fn budget_exhausted(&self) -> bool {
        self.budget_exhausted.load(Ordering::Acquire)
    }

    /// Whether the search-wide absolute deadline has passed on the shared clock.
    pub(crate) fn deadline_exceeded(&self) -> bool {
        self.deadline
            .is_some_and(|deadline| self.clock.now() >= deadline)
    }

    /// Reservation unit for `budget`: bounded chunks keep small budgets tight while
    /// large budgets amortize the shared-counter traffic.
    #[must_use]
    pub(crate) fn node_reservation_chunk(budget: u64) -> u64 {
        (budget / 64).clamp(1, NODE_RESERVATION_CHUNK)
    }

    /// The declared budget for diagnostics; `None` means unbounded.
    pub(crate) const fn node_budget(&self) -> Option<u64> {
        self.node_budget
    }

    /// Whether this search carries a shared node budget at all.
    pub(crate) const fn budgeted(&self) -> bool {
        self.node_budget.is_some()
    }

    /// Grants up to `requested` nodes from the shared budget. Returns how many the
    /// caller may actually visit: at most the ungranted remainder of the budget, so
    /// concurrent reservations overshoot only within one reservation unit per worker.
    pub(crate) fn grant_nodes(&self, requested: u64) -> u64 {
        let Some(budget) = self.node_budget else {
            return requested;
        };
        let previous = self.nodes_granted.fetch_add(requested, Ordering::Relaxed);
        let granted = budget.saturating_sub(previous).min(requested);
        if granted < requested {
            // Return the ungrantable remainder at once: concurrent workers may briefly
            // observe the inflated counter (safe direction — they are refused early),
            // and the counter keeps tracking handed-out credit minus returned credit.
            let _ = self
                .nodes_granted
                .fetch_sub(requested - granted, Ordering::Relaxed);
            if granted == 0 {
                // The declared cap was reached: latch the exhaustion so no worker can
                // wait on claims that can never be granted again.
                self.budget_exhausted.store(true, Ordering::Release);
            }
        }
        granted
    }

    /// Returns unused granted credit so other workers can still spend it.
    pub(crate) fn return_nodes(&self, credit: u64) {
        if credit > 0 {
            let _ = self.nodes_granted.fetch_sub(credit, Ordering::Relaxed);
        }
    }

    /// Adds one worker's completed (possibly cancelled or abandoned) claim to the
    /// exact helper totals the controller publishes.
    pub(crate) fn record_helper_work(&self, nodes: u64, stats: &SearchStats) {
        let _ = self.helper_nodes.fetch_add(nodes, Ordering::Relaxed);
        if let Ok(mut totals) = self.helper_stats.lock() {
            totals.accumulate(stats);
        }
    }

    /// Exact helper totals for mid-search reporting; monotonic, never reset.
    pub(crate) fn helper_work_snapshot(&self) -> (u64, SearchStats) {
        (
            self.helper_nodes.load(Ordering::Relaxed),
            self.helper_stats
                .lock()
                .map(|totals| *totals)
                .unwrap_or_default(),
        )
    }

    /// Final helper totals after every helper joined; exact, read once.
    pub(crate) fn drain_helper_work(&self) -> (u64, SearchStats) {
        self.helper_work_snapshot()
    }

    /// Number of root-move claims ever granted, for test synchronization.
    #[cfg(test)]
    pub(crate) fn claim_count(&self) -> u64 {
        self.claims.load(Ordering::Relaxed)
    }

    /// The controller finished (or is cancelling): every helper must exit.
    pub(crate) fn stop(&self) {
        self.stop.store(true, Ordering::Release);
    }

    /// Records that a helper's inference failed; the whole search must fail closed.
    pub(crate) fn fail_evaluation(&self) {
        self.evaluation_failed.store(true, Ordering::Release);
        self.stop();
    }

    #[must_use]
    pub(crate) fn evaluation_failed(&self) -> bool {
        self.evaluation_failed.load(Ordering::Acquire)
    }

    #[must_use]
    pub(crate) fn is_stopped(&self) -> bool {
        self.stop.load(Ordering::Acquire)
    }

    /// Publishes one root iteration for partitioned search. The claim order is the
    /// previous iteration's best-score-first ranking plus any missing legal moves in
    /// canonical order.
    pub(crate) fn begin_iteration(
        &self,
        depth: u8,
        alpha: i32,
        beta: i32,
        ranked: Vec<Move>,
        canonical: &[Move],
    ) -> u64 {
        let Ok(mut iteration) = self.iteration.lock() else {
            return 0;
        };
        let mut moves = ranked;
        for movement in canonical {
            if !moves.contains(movement) {
                moves.push(*movement);
            }
        }
        let count = moves.len();
        self.cursor.store(0, Ordering::Release);
        self.shared_alpha.store(alpha, Ordering::Release);
        self.primed.store(false, Ordering::Release);
        self.first_full_window.store(false, Ordering::Release);
        let epoch = iteration.epoch.wrapping_add(1);
        *iteration = IterationState {
            epoch,
            depth,
            alpha,
            beta,
            moves,
            results: vec![None; count],
        };
        epoch
    }

    /// Claims the next unsearched root move of the current iteration. The cursor lives
    /// under the iteration lock so a claim can never straddle `begin_iteration` and
    /// consume a slot of the epoch it was not issued for.
    pub(crate) fn claim(&self) -> Option<(u64, usize, Move)> {
        let claimed = {
            let iteration = self.iteration.lock().ok()?;
            if self.should_abort() || iteration.epoch == 0 {
                return None;
            }
            let index = self.cursor.fetch_add(1, Ordering::AcqRel);
            let movement = iteration.moves.get(index).copied()?;
            (iteration.epoch, index, movement)
        };
        let _ = self.claims.fetch_add(1, Ordering::Relaxed);
        Some(claimed)
    }

    /// Window parameters of `epoch`, but only while it is still the current iteration.
    /// A claim from an already-replaced epoch is simply dropped: its slot belonged to an
    /// iteration nobody waits on anymore.
    pub(crate) fn window_of(&self, epoch: u64) -> Option<(u8, i32, i32)> {
        let iteration = self.iteration.lock().ok()?;
        if iteration.epoch != epoch {
            return None;
        }
        Some((iteration.depth, iteration.alpha, iteration.beta))
    }

    /// Window parameters of the current iteration.
    pub(crate) fn iteration_window(&self) -> Option<(u64, u8, i32, i32)> {
        let iteration = self.iteration.lock().ok()?;
        if iteration.epoch == 0 {
            return None;
        }
        Some((
            iteration.epoch,
            iteration.depth,
            iteration.alpha,
            iteration.beta,
        ))
    }

    /// Publishes a completed root-move result. The boolean records whether the score
    /// is exact; a scout that was not re-searched only bounded its move at the shared
    /// alpha, and the merge must not let such an entry win a score tie. Late results
    /// for a replaced iteration are dropped.
    pub(crate) fn publish(&self, epoch: u64, index: usize, stat: RootMoveStat, exact: bool) {
        let Ok(mut iteration) = self.iteration.lock() else {
            return;
        };
        if iteration.epoch == epoch
            && let Some(slot) = iteration.results.get_mut(index)
        {
            *slot = Some((stat, exact));
        }
    }

    /// Clone of a published result, for the controller's merge pass.
    pub(crate) fn result(&self, epoch: u64, index: usize) -> Option<(RootMoveStat, bool)> {
        let iteration = self.iteration.lock().ok()?;
        if iteration.epoch != epoch {
            return None;
        }
        iteration.results.get(index)?.clone()
    }

    /// The move at a claim index of the current iteration.
    pub(crate) fn move_at(&self, epoch: u64, index: usize) -> Option<Move> {
        let iteration = self.iteration.lock().ok()?;
        if iteration.epoch != epoch {
            return None;
        }
        iteration.moves.get(index).copied()
    }

    /// Number of moves in the current iteration.
    #[must_use]
    pub(crate) fn move_count(&self, epoch: u64) -> usize {
        self.iteration
            .lock()
            .ok()
            .filter(|iteration| iteration.epoch == epoch)
            .map_or(0, |iteration| iteration.moves.len())
    }

    /// Claims the iteration's single full-window search.
    #[must_use]
    pub(crate) fn claim_full_window(&self) -> bool {
        !self.first_full_window.swap(true, Ordering::AcqRel)
    }

    /// Called by the full-window worker once its result (bound or exact) is published.
    pub(crate) fn mark_primed(&self) {
        self.primed.store(true, Ordering::Release);
    }

    /// Scouts must not run before the first full-window search raised the shared alpha;
    /// otherwise a widened aspiration window would put every worker into a full search.
    /// Returns false if the primer disappeared (panic, stop) or the search-wide deadline
    /// passed while waiting, and priming never happened.
    pub(crate) fn wait_until_primed(&self) -> bool {
        while !self.primed.load(Ordering::Acquire) {
            if self.should_abort() || !self.helpers_alive() {
                return self.primed.load(Ordering::Acquire);
            }
            std::thread::sleep(ITERATION_POLL);
        }
        true
    }

    /// Published scores (exact or a sound bound) raise the shared alpha so later claims
    /// search narrower windows; `fetch_max` keeps a lower bound from lowering it.
    pub(crate) fn raise_alpha(&self, score: i32) {
        let _ = self.shared_alpha.fetch_max(score, Ordering::AcqRel);
    }

    /// Current shared alpha; child windows narrow as it rises.
    #[must_use]
    pub(crate) fn current_alpha(&self) -> i32 {
        self.shared_alpha.load(Ordering::Acquire)
    }

    /// Registers a helper for liveness tracking; the returned guard releases it.
    pub(crate) fn enter_helper(&self) -> HelperGuard<'_> {
        self.alive_helpers.fetch_add(1, Ordering::AcqRel);
        HelperGuard { parallel: self }
    }

    #[must_use]
    pub(crate) fn helpers_alive(&self) -> bool {
        self.alive_helpers.load(Ordering::Acquire) > 0
    }

    /// Waits until a new iteration appears or the search stops or expires.
    pub(crate) fn wait_for_work(&self, seen_epoch: u64) {
        if self.should_abort() {
            return;
        }
        let current = self
            .iteration
            .lock()
            .map_or(seen_epoch, |iteration| iteration.epoch);
        if current == seen_epoch {
            std::thread::sleep(ITERATION_POLL);
        }
    }
}

/// Drops the helper liveness count on exit, including through unwinding.
pub(crate) struct HelperGuard<'a> {
    parallel: &'a ParallelSearch,
}

impl Drop for HelperGuard<'_> {
    fn drop(&mut self) {
        self.parallel.alive_helpers.fetch_sub(1, Ordering::AcqRel);
    }
}

/// Test-only, single-permit injection of a helper inference failure (the escalation
/// path is unreachable with valid models, so the regression test arms one claim).
/// The only caller lives in the handcrafted-gated search tests.
#[cfg(all(test, feature = "handcrafted"))]
static INJECT_HELPER_EVALUATION_FAILURE: AtomicBool = AtomicBool::new(false);

#[cfg(all(test, feature = "handcrafted"))]
pub(crate) fn inject_helper_evaluation_failure() {
    INJECT_HELPER_EVALUATION_FAILURE.store(true, Ordering::SeqCst);
}

/// Consumes the injection permit; true exactly once per arming.
#[cfg(all(test, feature = "handcrafted"))]
pub(crate) fn take_helper_evaluation_failure_injection() -> bool {
    INJECT_HELPER_EVALUATION_FAILURE.swap(false, Ordering::SeqCst)
}

/// Test-only gate that parks a helper inside its primer search while leaving the stop
/// flag observable, modelling a long inference the search cannot interrupt. The
/// escalation paths (deadline, stop, cancellation during a primer wait) are unreachable
/// with a cooperative primer, so the regression tests arm one claim.
#[cfg(test)]
static PRIMER_GATE: AtomicBool = AtomicBool::new(false);

#[cfg(test)]
pub(crate) fn arm_primer_gate() {
    PRIMER_GATE.store(true, Ordering::SeqCst);
}

#[cfg(test)]
pub(crate) fn release_primer_gate() {
    PRIMER_GATE.store(false, Ordering::SeqCst);
}

#[cfg(test)]
pub(crate) fn primer_gate_armed() -> bool {
    PRIMER_GATE.load(Ordering::SeqCst)
}

/// Test-only permit that makes the controller wait until a helper has claimed before
/// its own first claim of a search, so helper-dominated workloads are deterministic.
#[cfg(test)]
static GATE_CONTROLLER_ON_HELPER_CLAIM: AtomicBool = AtomicBool::new(false);

#[cfg(test)]
pub(crate) fn arm_controller_claim_gate() {
    GATE_CONTROLLER_ON_HELPER_CLAIM.store(true, Ordering::SeqCst);
}

/// Test-side, single-shot consumption of the controller claim gate: parks the calling
/// controller until a helper has claimed (bounded, so a broken helper cannot hang the
/// suite), making helper-dominated workloads deterministic.
#[cfg(test)]
pub(crate) fn gate_controller_on_helper_claim(parallel: &ParallelSearch) {
    if !GATE_CONTROLLER_ON_HELPER_CLAIM.swap(false, Ordering::SeqCst) {
        return;
    }
    let limit = web_time::Instant::now() + std::time::Duration::from_secs(10);
    while parallel.claim_count() == 0 && web_time::Instant::now() < limit {
        std::thread::sleep(ITERATION_POLL);
    }
}

/// Test-only permit that makes the next helper `fork_for_worker` panic, exercising the
/// creation-loop unwind path with helpers already running.
#[cfg(test)]
static INJECT_FORK_PANIC: AtomicUsize = AtomicUsize::new(0);

#[cfg(test)]
/// Arms a panic on the `n`-th helper fork (1-based); zero disarms.
pub(crate) fn inject_fork_panic_on(nth: usize) {
    INJECT_FORK_PANIC.store(nth, Ordering::SeqCst);
}

#[cfg(test)]
pub(crate) fn take_fork_panic_injection(fork_index: usize) -> bool {
    fork_index > 0
        && INJECT_FORK_PANIC
            .compare_exchange(fork_index, 0, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
}

/// Test-only permit that fails the `n`-th helper thread spawn, exercising the explicit
/// `spawn_scoped` failure path with (for `n` > 1) helpers already running.
#[cfg(test)]
static INJECT_HELPER_SPAWN_FAILURE: AtomicUsize = AtomicUsize::new(0);

#[cfg(test)]
/// Arms a spawn failure on the `n`-th helper creation (1-based); zero disarms.
pub(crate) fn inject_helper_spawn_failure_on(nth: usize) {
    INJECT_HELPER_SPAWN_FAILURE.store(nth, Ordering::SeqCst);
}

#[cfg(test)]
pub(crate) fn take_helper_spawn_failure_injection(spawn_index: usize) -> bool {
    spawn_index > 0
        && INJECT_HELPER_SPAWN_FAILURE
            .compare_exchange(spawn_index, 0, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
}

/// Test-only permit that panics a helper in the middle of its search once its context
/// has visited exactly the armed node count, exercising the unwind path through a live
/// per-claim context (its flushed work must still reach the shared totals).
#[cfg(test)]
static INJECT_HELPER_PANIC_AT_NODES: AtomicU64 = AtomicU64::new(0);

#[cfg(test)]
pub(crate) fn inject_helper_panic_at_nodes(nodes: u64) {
    INJECT_HELPER_PANIC_AT_NODES.store(nodes, Ordering::SeqCst);
}

#[cfg(test)]
pub(crate) fn helper_panic_node_injection() -> u64 {
    INJECT_HELPER_PANIC_AT_NODES.load(Ordering::SeqCst)
}

#[cfg(test)]
/// Consumes an armed panic permit exactly once (the load-based match keeps the hot
/// path to a plain read; only an actual match pays for this swap).
pub(crate) fn consume_helper_panic_injection() -> bool {
    INJECT_HELPER_PANIC_AT_NODES.swap(0, Ordering::SeqCst) != 0
}

/// Creates one helper thread inside the search scope, reporting creation failure
/// instead of panicking so a failing spawn degrades the worker count explicitly.
#[cfg(not(test))]
pub(crate) fn spawn_helper_thread<'scope, 'env, F>(
    scope: &'scope std::thread::Scope<'scope, 'env>,
    index: usize,
    name: String,
    body: F,
) -> std::io::Result<()>
where
    F: FnOnce() + Send + 'scope,
{
    let _ = index;
    std::thread::Builder::new()
        .name(name)
        .spawn_scoped(scope, body)
        .map(|_| ())
}

/// Test twin of `spawn_helper_thread` with an injected creation-failure permit, so the
/// explicit failure path (helpers already running) is reachable without relying on the
/// OS to fail a thread spawn.
#[cfg(test)]
pub(crate) fn spawn_helper_thread<'scope, 'env, F>(
    scope: &'scope std::thread::Scope<'scope, 'env>,
    index: usize,
    name: String,
    body: F,
) -> std::io::Result<()>
where
    F: FnOnce() + Send + 'scope,
{
    if take_helper_spawn_failure_injection(index) {
        return Err(std::io::Error::other("injected helper spawn failure"));
    }
    std::thread::Builder::new()
        .name(name)
        .spawn_scoped(scope, body)
        .map(|_| ())
}

/// One helper worker: claims root moves of the current iteration until the queue drains
/// or the controller stops the search. Takes ownership because the worker owns its
/// engine and coordination handle outright.
///
/// The helper owns no time or node budget of its own: every claim it runs is bounded by
/// the search-wide absolute deadline and the shared node pool on `parallel`, so neither
/// starting a claim nor starting the helper can revive a spent budget.
#[expect(
    clippy::needless_pass_by_value,
    reason = "the worker closure takes these values by value"
)]
pub(crate) fn run_helper(
    mut engine: crate::SearchEngine,
    root: crate::Position,
    parallel: Arc<ParallelSearch>,
    cancellation: crate::CancellationToken,
) {
    let _guard = parallel.enter_helper();
    engine.set_active_parallel(Arc::clone(&parallel));
    let limits = crate::SearchLimits {
        max_depth: 0,
        max_nodes: None,
        movetime: None,
    };
    let mut prepared_epoch = 0_u64;
    while !parallel.should_abort() && !cancellation.is_cancelled() {
        let Some((claim_epoch, index, movement)) = parallel.claim() else {
            let seen = parallel
                .iteration_window()
                .map_or(0, |(epoch, _, _, _)| epoch);
            parallel.wait_for_work(seen);
            continue;
        };
        // The claim's epoch is authoritative: honoring it (or dropping it when the
        // iteration already moved past) can never consume a live slot without a result.
        let Some((depth, _alpha, beta)) = parallel.window_of(claim_epoch) else {
            continue;
        };
        if claim_epoch != prepared_epoch && engine.prepare_worker_root(&root).is_err() {
            // A helper that cannot even prepare its root (repetition/history/accumulator
            // validation) cannot produce trustworthy output either: fail closed instead
            // of draining into a fallback bestmove.
            parallel.fail_evaluation();
            return;
        }
        prepared_epoch = claim_epoch;
        let first = parallel.claim_full_window();
        if !first && !parallel.wait_until_primed() {
            return;
        }
        let search = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            engine.search_root_move_detached(
                &root,
                movement,
                depth,
                parallel.current_alpha(),
                beta,
                first,
                limits,
                &cancellation,
            )
        }));
        if first {
            // The priming search ended (published, failed, time, cancellation or
            // panic): unblock the scouts so they observe the stop flag.
            parallel.mark_primed();
        }
        match search {
            Ok((Some((stat, exact)), _)) => {
                parallel.raise_alpha(stat.score);
                parallel.publish(claim_epoch, index, stat, exact);
            }
            // A panic leaves this worker's engine state unbalanced. Stop the whole
            // search: surviving helpers never exit on their own, so the controller's
            // drain must not wait on them for a search that cannot complete cleanly.
            Err(_) => {
                parallel.stop();
                return;
            }
            // Inference failure: the search cannot produce trustworthy output, so it
            // must fail closed instead of draining into a fallback bestmove. Deadline
            // and cancellation Nones keep claiming — the controller owns that stop.
            Ok((None, true)) => {
                parallel.fail_evaluation();
                return;
            }
            Ok((None, false)) => {}
        }
    }
}
