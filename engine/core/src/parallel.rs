//! Native parallel search: partitioned root iterations.
//!
//! The controller runs the ordinary iterative-deepening loop and owns the published
//! result, so `Threads=1` and USI semantics are unchanged. Interior TT cutoffs cannot be
//! shared for the pure runtime because its scores depend on repetition history a
//! position-only table cannot represent, so the root iteration is the parallel unit:
//! every worker — the controller included — claims root moves of the current depth and
//! searches them against one shared alpha. Exact scores raise that alpha for everyone;
//! a fail-low against an older, lower alpha can never become the best move. The
//! controller merges published results into the same root evidence a serial iteration
//! produces, and joins the helpers (scoped threads) after setting the stop flag.

use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicI32, AtomicUsize, Ordering},
    },
    time::Duration,
};

use crate::{Move, RootMoveStat};

/// Short worker wait while waiting for the next iteration to be published.
pub(crate) const ITERATION_POLL: Duration = Duration::from_micros(200);

/// Defensive upper bound on total workers (controller plus helpers) for one search.
pub(crate) const MAX_WORKERS: usize = 128;

/// Coordination state for one bounded parallel search.
pub(crate) struct ParallelSearch {
    stop: AtomicBool,
    /// Helper liveness; the controller waits on missing results only while this is > 0.
    alive_helpers: AtomicUsize,
    cursor: AtomicUsize,
    shared_alpha: AtomicI32,
    /// Set once the iteration's full-window search has published a real alpha; scouts
    /// wait for it so they never run against a not-yet-raised shared alpha.
    primed: AtomicBool,
    /// Swap-to-claim the iteration's single full-window (first-move) search.
    pub(crate) first_full_window: AtomicBool,
    iteration: Mutex<IterationState>,
}

struct IterationState {
    epoch: u64,
    depth: u8,
    alpha: i32,
    beta: i32,
    moves: Vec<Move>,
    results: Vec<Option<RootMoveStat>>,
}

impl ParallelSearch {
    pub(crate) fn new() -> Self {
        Self {
            stop: AtomicBool::new(false),
            alive_helpers: AtomicUsize::new(0),
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
        }
    }

    /// The controller finished (or is cancelling): every helper must exit.
    pub(crate) fn stop(&self) {
        self.stop.store(true, Ordering::Release);
    }

    #[must_use]
    pub(crate) fn is_stopped(&self) -> bool {
        self.stop.load(Ordering::Acquire)
    }

    /// Publishes one root iteration for partitioned search. The claim order is the
    /// previous iteration's root ranking (expensive moves first) plus any missing legal
    /// moves in canonical order.
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

    /// Claims the next unsearched root move of the current iteration.
    pub(crate) fn claim(&self) -> Option<(u64, usize, Move)> {
        if self.is_stopped() {
            return None;
        }
        let (epoch, moves) = {
            let iteration = self.iteration.lock().ok()?;
            if iteration.epoch == 0 {
                return None;
            }
            (iteration.epoch, iteration.moves.clone())
        };
        let index = self.cursor.fetch_add(1, Ordering::AcqRel);
        let movement = moves.get(index).copied()?;
        Some((epoch, index, movement))
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

    /// Publishes a completed root-move result. Late results for a replaced iteration are
    /// dropped.
    pub(crate) fn publish(&self, epoch: u64, index: usize, stat: RootMoveStat) {
        let Ok(mut iteration) = self.iteration.lock() else {
            return;
        };
        if iteration.epoch == epoch
            && let Some(slot) = iteration.results.get_mut(index)
        {
            *slot = Some(stat);
        }
    }

    /// Clone of a published result, for the controller's merge pass.
    pub(crate) fn result(&self, epoch: u64, index: usize) -> Option<RootMoveStat> {
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
    pub(crate) fn wait_until_primed(&self) {
        while !self.primed.load(Ordering::Acquire) {
            if self.is_stopped() {
                return;
            }
            std::thread::sleep(ITERATION_POLL);
        }
    }

    /// Exact scores raise the shared alpha so later claims search narrower windows.
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

    /// Waits until a new iteration appears or the search stops.
    pub(crate) fn wait_for_work(&self, seen_epoch: u64) {
        if self.is_stopped() {
            return;
        }
        let current = self
            .iteration
            .lock()
            .map(|iteration| iteration.epoch)
            .unwrap_or(seen_epoch);
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

/// One helper worker: claims root moves of the current iteration until the queue drains
/// or the controller stops the search. Takes ownership because the worker owns its
/// engine and coordination handle outright.
#[expect(
    clippy::needless_pass_by_value,
    reason = "the worker closure takes these values by value"
)]
pub(crate) fn run_helper(
    mut engine: crate::SearchEngine,
    root: crate::Position,
    parallel: Arc<ParallelSearch>,
    cancellation: crate::CancellationToken,
    hard_limit: Option<Duration>,
) {
    let _guard = parallel.enter_helper();
    engine.set_active_parallel(Arc::clone(&parallel));
    let limits = crate::SearchLimits {
        max_depth: 0,
        max_nodes: None,
        movetime: hard_limit,
    };
    let mut prepared_epoch = 0_u64;
    while !parallel.is_stopped() && !cancellation.is_cancelled() {
        let Some((epoch, depth, _alpha, beta)) = parallel.iteration_window() else {
            parallel.wait_for_work(0);
            continue;
        };
        if epoch != prepared_epoch && engine.prepare_worker_root(&root).is_err() {
            parallel.stop();
            return;
        }
        prepared_epoch = epoch;
        let Some((claim_epoch, index, movement)) = parallel.claim() else {
            parallel.wait_for_work(epoch);
            continue;
        };
        if claim_epoch != epoch {
            continue;
        }
        let first = parallel.claim_full_window();
        if !first {
            parallel.wait_until_primed();
            if parallel.is_stopped() || cancellation.is_cancelled() {
                return;
            }
        }
        if let Some(stat) = engine.search_root_move_detached(
            &root,
            movement,
            depth,
            parallel.current_alpha(),
            beta,
            first,
            limits,
            &cancellation,
        ) {
            parallel.raise_alpha(stat.score);
            if first {
                parallel.mark_primed();
            }
            parallel.publish(epoch, index, stat);
        } else if first {
            // The priming search failed (time or cancellation): unblock the scouts so
            // they observe the stop flag instead of waiting forever.
            parallel.mark_primed();
        }
    }
}
