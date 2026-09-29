//! Versioned, adapter-neutral play time management.

use std::time::Duration;

use crate::{Position, SearchInfo, Side};

/// Schema shared by native, USI-mapped, Rust, and Wasm time-control boundaries.
pub const TIME_CONTROL_SCHEMA: &str = "open_shogi_time_control/v1";
/// Casual play may never allocate more than twenty seconds to one move.
pub const CASUAL_HARD_MAX_MS: u64 = 20_000;
/// Defensive upper bound for a configurable deadline safety margin.
pub const MAX_SAFETY_MARGIN_MS: u64 = 1_000;
/// Defensive upper bound for clocks accepted by the shared Rust boundary.
pub const MAX_CLOCK_MS: u64 = 7 * 24 * 60 * 60 * 1_000;
/// Defensive upper bound for one explicit move-time or byoyomi value.
pub const MAX_MOVE_TIME_MS: u64 = 60 * 60 * 1_000;
/// Defensive upper bound for a diagnostic node budget.
pub const MAX_TIME_CONTROL_NODES: u64 = 1_000_000_000;
/// Defensive upper bound for an explicit search depth.
pub const MAX_TIME_CONTROL_DEPTH: u8 = 64;

/// Closed shared request. Millisecond fields are integers at every external boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TimeControl {
    pub black_time_ms: Option<u64>,
    pub white_time_ms: Option<u64>,
    pub byoyomi_ms: Option<u64>,
    pub black_increment_ms: Option<u64>,
    pub white_increment_ms: Option<u64>,
    pub movetime_ms: Option<u64>,
    pub nodes: Option<u64>,
    pub depth: Option<u8>,
    pub infinite: bool,
    pub casual: bool,
    pub safety_margin_ms: u64,
}

impl TimeControl {
    /// Default no-clock play request.
    #[must_use]
    pub const fn casual() -> Self {
        Self {
            black_time_ms: None,
            white_time_ms: None,
            byoyomi_ms: None,
            black_increment_ms: None,
            white_increment_ms: None,
            movetime_ms: None,
            nodes: None,
            depth: None,
            infinite: false,
            casual: true,
            safety_margin_ms: 50,
        }
    }

    /// Validates the closed resource limits without allocating a deadline.
    ///
    /// # Errors
    ///
    /// Returns an error for contradictory modes or values outside defensive bounds.
    pub fn validate(self) -> Result<(), String> {
        for (name, value, maximum) in [
            ("blackTimeMs", self.black_time_ms, MAX_CLOCK_MS),
            ("whiteTimeMs", self.white_time_ms, MAX_CLOCK_MS),
            ("byoyomiMs", self.byoyomi_ms, MAX_MOVE_TIME_MS),
            (
                "blackIncrementMs",
                self.black_increment_ms,
                MAX_MOVE_TIME_MS,
            ),
            (
                "whiteIncrementMs",
                self.white_increment_ms,
                MAX_MOVE_TIME_MS,
            ),
        ] {
            if value.is_some_and(|value| value > maximum) {
                return Err(format!("{name} exceeds {maximum} ms"));
            }
        }
        if self
            .movetime_ms
            .is_some_and(|value| value == 0 || value > MAX_MOVE_TIME_MS)
        {
            return Err(format!("moveTimeMs must be 1..={MAX_MOVE_TIME_MS}"));
        }
        if self
            .nodes
            .is_some_and(|value| value == 0 || value > MAX_TIME_CONTROL_NODES)
        {
            return Err(format!("nodes must be 1..={MAX_TIME_CONTROL_NODES}"));
        }
        if self
            .depth
            .is_some_and(|value| !(1..=MAX_TIME_CONTROL_DEPTH).contains(&value))
        {
            return Err(format!("depth must be 1..={MAX_TIME_CONTROL_DEPTH}"));
        }
        if self.safety_margin_ms > MAX_SAFETY_MARGIN_MS {
            return Err(format!("safetyMarginMs must be 0..={MAX_SAFETY_MARGIN_MS}"));
        }
        let has_clock = self.black_time_ms.is_some()
            || self.white_time_ms.is_some()
            || self.byoyomi_ms.is_some()
            || self.black_increment_ms.is_some()
            || self.white_increment_ms.is_some();
        if self.casual
            && (self.infinite
                || self.movetime_ms.is_some()
                || has_clock
                || self.nodes.is_some()
                || self.depth.is_some())
        {
            return Err("casual mode cannot be combined with another search mode".to_owned());
        }
        if self.infinite
            && (self.movetime_ms.is_some()
                || has_clock
                || self.nodes.is_some()
                || self.depth.is_some())
        {
            return Err("infinite mode cannot be combined with a fixed limit".to_owned());
        }
        Ok(())
    }
}

impl Default for TimeControl {
    fn default() -> Self {
        Self::casual()
    }
}

/// Source of the time budget selected by the manager.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TimeControlMode {
    Casual,
    MoveTime,
    Clock,
    Nodes,
    Depth,
    Infinite,
}

/// Adaptive stability thresholds used only after the soft deadline.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StabilityPolicy {
    pub required_stable_iterations: u8,
    pub stable_score_cp: i32,
    pub close_candidate_cp: i32,
    pub volatile_score_cp: i32,
    pub difficult_branching_moves: usize,
}

impl Default for StabilityPolicy {
    fn default() -> Self {
        Self {
            required_stable_iterations: 2,
            stable_score_cp: 24,
            close_candidate_cp: 80,
            volatile_score_cp: 120,
            difficult_branching_moves: 40,
        }
    }
}

/// Concrete monotonic limits for one search.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TimePlan {
    pub mode: TimeControlMode,
    pub max_depth: u8,
    pub max_nodes: Option<u64>,
    pub soft_limit: Option<Duration>,
    pub hard_limit: Option<Duration>,
    pub allocated_hard_limit: Option<Duration>,
    pub safety_margin: Duration,
    /// Stability and predicted-overrun early stops stay silent before this floor.
    /// Set for per-move budgets (pure byoyomi) whose unused time can never be banked.
    pub min_spend: Option<Duration>,
    pub allow_stable_early_stop: bool,
    pub stability: StabilityPolicy,
}

/// Bounded manager configuration. The casual hard limit cannot exceed the protocol constant.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TimeManagerConfig {
    pub casual_hard_limit_ms: u64,
    pub casual_soft_limit_ms: u64,
    pub stability: StabilityPolicy,
}

impl Default for TimeManagerConfig {
    fn default() -> Self {
        Self {
            casual_hard_limit_ms: CASUAL_HARD_MAX_MS,
            casual_soft_limit_ms: 1_500,
            stability: StabilityPolicy::default(),
        }
    }
}

/// Stateless per-call allocation math plus small cross-move state: a cooldown that
/// temporarily shrinks the difficult-position extension cap after a move that actually
/// spent it, so consecutive extensions cannot drain a finite clock.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TimeManager {
    config: TimeManagerConfig,
    extension_cooldown: u8,
}

/// Live-clock spending plan: a fraction of the remaining clock that decays
/// geometrically and can never reach zero. The fraction itself ramps up as the clock
/// drains: the normal target is `remaining / 64 * 9 / 10` — a 64-move geometric runway
/// shrunk by one tenth, i.e. `27/1920` of the clock — rising linearly to `1/12`
/// (`160/1920`) at `CLOCK_EMERGENCY_FLOOR_MS` after `CLOCK_EMERGENCY_MS`, and holding
/// there below it, so the engine progressively stops hoarding as time becomes dangerous
/// instead of switching regimes at a hard threshold. Extensions are capped both
/// relative to the target and as a fraction of the remaining clock, so no single move
/// can consume a large slice of a sudden-death clock.
const CLOCK_EXTENSION_CAP_MULT: u64 = 5;
const CLOCK_EXTENSION_FRACTION: u64 = 8;
const CLOCK_COOLDOWN_EXTENSION_MULT: u64 = 3;
/// Remaining time where the spend fraction starts ramping up from the normal `27/1920`;
/// the ramp is continuous here by construction.
const CLOCK_EMERGENCY_MS: u64 = 20_000;
/// Remaining time at which the spend fraction reaches `1/12` and stays there.
const CLOCK_EMERGENCY_FLOOR_MS: u64 = 5_000;
/// Ramp span: `CLOCK_EMERGENCY_MS - CLOCK_EMERGENCY_FLOOR_MS`.
const CLOCK_EMERGENCY_SPAN_MS: u64 = CLOCK_EMERGENCY_MS - CLOCK_EMERGENCY_FLOOR_MS;
/// Target-fraction numerators over `CLOCK_FRACTION_SCALE = 1920`.
const CLOCK_FRACTION_SCALE: u64 = 1_920;
const CLOCK_NORMAL_FRACTION: u64 = 27;
const CLOCK_EMERGENCY_FRACTION: u64 = 160;

impl TimeManager {
    /// Constructs a validated manager.
    ///
    /// # Errors
    ///
    /// Returns an error if casual limits exceed the twenty-second hard cap.
    pub fn new(config: TimeManagerConfig) -> Result<Self, String> {
        if config.casual_hard_limit_ms == 0 || config.casual_hard_limit_ms > CASUAL_HARD_MAX_MS {
            return Err(format!(
                "casual hard limit must be 1..={CASUAL_HARD_MAX_MS} ms"
            ));
        }
        if config.casual_soft_limit_ms == 0
            || config.casual_soft_limit_ms > config.casual_hard_limit_ms
        {
            return Err(
                "casual soft limit must be positive and not exceed its hard limit".to_owned(),
            );
        }
        Ok(Self {
            config,
            extension_cooldown: 0,
        })
    }

    /// Reports the wall-clock spend of a completed move. A move that spent well beyond
    /// its soft target engages the extension cooldown for the next few moves.
    pub fn observe_spend(&mut self, plan: &TimePlan, spent: Duration) {
        let extended = plan
            .soft_limit
            .is_some_and(|soft| spent.as_millis() > soft.as_millis() * 6 / 5);
        self.extension_cooldown = if extended {
            3
        } else {
            self.extension_cooldown.saturating_sub(1)
        };
    }

    /// Drops all cross-move state, leaving the validated configuration intact. Match
    /// boundaries (new game, game over, a fresh browser session) must call this so a
    /// previous match's extension cooldown cannot leak into the next one.
    pub const fn reset_spend_history(&mut self) {
        self.extension_cooldown = 0;
    }

    /// Allocates soft and hard monotonic durations for the side to move.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid shared request.
    pub fn plan(
        &self,
        side: Side,
        request: TimeControl,
        default_max_depth: u8,
    ) -> Result<TimePlan, String> {
        request.validate()?;
        if !(1..=MAX_TIME_CONTROL_DEPTH).contains(&default_max_depth) {
            return Err(format!(
                "default max depth must be 1..={MAX_TIME_CONTROL_DEPTH}"
            ));
        }
        let max_depth = request.depth.unwrap_or(default_max_depth);
        if request.infinite {
            return Ok(self.plan_without_deadline(
                TimeControlMode::Infinite,
                max_depth,
                request.nodes,
            ));
        }
        if request.casual || is_implicit_casual(request) {
            let allocated = self.config.casual_hard_limit_ms;
            let hard = deadline_after_margin(allocated, request.safety_margin_ms);
            let soft = self.config.casual_soft_limit_ms.min(hard);
            return Ok(TimePlan {
                mode: TimeControlMode::Casual,
                max_depth,
                max_nodes: request.nodes,
                soft_limit: Some(Duration::from_millis(soft)),
                hard_limit: Some(Duration::from_millis(hard)),
                allocated_hard_limit: Some(Duration::from_millis(allocated)),
                safety_margin: Duration::from_millis(allocated.saturating_sub(hard)),
                min_spend: None,
                allow_stable_early_stop: true,
                stability: self.config.stability,
            });
        }
        // A live game clock is authoritative even when an adapter accidentally includes
        // a preset move time. Diagnostic limits may only further restrict it.
        if has_clock(request) {
            return Ok(self.clock_plan(side, request, max_depth));
        }
        if let Some(movetime) = request.movetime_ms {
            let hard = deadline_after_margin(movetime, request.safety_margin_ms);
            return Ok(TimePlan {
                mode: TimeControlMode::MoveTime,
                max_depth,
                max_nodes: request.nodes,
                soft_limit: Some(Duration::from_millis(hard)),
                hard_limit: Some(Duration::from_millis(hard)),
                allocated_hard_limit: Some(Duration::from_millis(movetime)),
                safety_margin: Duration::from_millis(movetime.saturating_sub(hard)),
                min_spend: None,
                allow_stable_early_stop: false,
                stability: self.config.stability,
            });
        }
        let mode = if request.nodes.is_some() {
            TimeControlMode::Nodes
        } else {
            TimeControlMode::Depth
        };
        Ok(self.plan_without_deadline(mode, max_depth, request.nodes))
    }

    /// Position-aware allocation. Clock spending is a fixed fraction of the remaining
    /// clock, so the position only selects the side whose clock is spent; no position
    /// score or move-count prior restricts legal play.
    ///
    /// # Errors
    /// Returns the same validation errors as `plan`.
    pub fn plan_for_position(
        &self,
        position: &Position,
        request: TimeControl,
        default_max_depth: u8,
    ) -> Result<TimePlan, String> {
        self.plan(position.side_to_move(), request, default_max_depth)
    }

    fn plan_without_deadline(
        &self,
        mode: TimeControlMode,
        max_depth: u8,
        max_nodes: Option<u64>,
    ) -> TimePlan {
        TimePlan {
            mode,
            max_depth,
            max_nodes,
            soft_limit: None,
            hard_limit: None,
            allocated_hard_limit: None,
            safety_margin: Duration::ZERO,
            min_spend: None,
            allow_stable_early_stop: false,
            stability: self.config.stability,
        }
    }

    fn clock_plan(&self, side: Side, request: TimeControl, max_depth: u8) -> TimePlan {
        let remaining = match side {
            Side::Black => request.black_time_ms,
            Side::White => request.white_time_ms,
        }
        .unwrap_or(0);
        let increment = match side {
            Side::Black => request.black_increment_ms,
            Side::White => request.white_increment_ms,
        }
        .unwrap_or(0);
        let byoyomi = request.byoyomi_ms.unwrap_or(0);
        let margin = move_margin_ms(remaining.max(byoyomi), request.safety_margin_ms);

        // A zero base clock with byoyomi is a pure per-move budget: unused time never
        // carries, so the engine should spend most of each period. Stability and
        // predicted-overrun stops stay silent until half the period is gone; proven
        // mates and trivial positions still return early.
        if remaining == 0 && byoyomi > 0 {
            let hard = byoyomi.saturating_sub(margin);
            let soft = (byoyomi * 3 / 4).min(hard);
            return TimePlan {
                mode: TimeControlMode::Clock,
                max_depth,
                max_nodes: request.nodes,
                soft_limit: Some(Duration::from_millis(soft)),
                hard_limit: Some(Duration::from_millis(hard)),
                allocated_hard_limit: Some(Duration::from_millis(byoyomi)),
                safety_margin: Duration::from_millis(byoyomi.saturating_sub(hard)),
                min_spend: Some(Duration::from_millis((byoyomi / 2).min(hard))),
                allow_stable_early_stop: true,
                stability: self.config.stability,
            };
        }

        // Emergency regime: as the clock drains below CLOCK_EMERGENCY_MS, the spend
        // fraction ramps smoothly toward remaining/12, so the engine stops hoarding
        // exactly as fast as its situation becomes dangerous — no discontinuity at the
        // threshold, and the absolute extension cap below never loosens. The fraction
        // stays one scaled division (no quantization), so adjacent clocks differ by at
        // most rounding.
        if remaining > 0 {
            let drained = CLOCK_EMERGENCY_MS
                .saturating_sub(remaining)
                .min(CLOCK_EMERGENCY_SPAN_MS);
            // remaining * fraction(remaining) in one scaled division; the product is
            // far below u64 range for every accepted clock.
            let scaled = remaining.saturating_mul(
                CLOCK_NORMAL_FRACTION * CLOCK_EMERGENCY_SPAN_MS
                    + (CLOCK_EMERGENCY_FRACTION - CLOCK_NORMAL_FRACTION) * drained,
            ) / (CLOCK_FRACTION_SCALE * CLOCK_EMERGENCY_SPAN_MS);
            let target = scaled
                .saturating_add(increment.saturating_mul(3) / 4)
                .saturating_add(byoyomi.saturating_mul(3) / 4)
                .max(1);
            let extension_multiple = if self.extension_cooldown > 0 {
                CLOCK_COOLDOWN_EXTENSION_MULT
            } else {
                CLOCK_EXTENSION_CAP_MULT
            };
            // Two absolute caps: extension headroom relative to the target, and a fraction of
            // the remaining clock no single move may consume even in a critical position.
            let extension_cap = (target.saturating_mul(extension_multiple) / 2)
                .min(remaining.saturating_div(CLOCK_EXTENSION_FRACTION));
            let available = remaining.saturating_add(byoyomi);
            let allocated = extension_cap.max(target).min(available);
            let hard = allocated.saturating_sub(margin.min(allocated.saturating_sub(1)));
            let soft = target.min(hard);
            return TimePlan {
                mode: TimeControlMode::Clock,
                max_depth,
                max_nodes: request.nodes,
                soft_limit: Some(Duration::from_millis(soft)),
                hard_limit: Some(Duration::from_millis(hard)),
                allocated_hard_limit: Some(Duration::from_millis(allocated)),
                safety_margin: Duration::from_millis(allocated.saturating_sub(hard)),
                min_spend: None,
                allow_stable_early_stop: true,
                stability: self.config.stability,
            };
        }
        // A zero base clock cannot manufacture time, with or without increment or
        // byoyomi income: every positive-clock branch returned above.
        TimePlan {
            mode: TimeControlMode::Clock,
            max_depth,
            max_nodes: request.nodes,
            soft_limit: Some(Duration::ZERO),
            hard_limit: Some(Duration::ZERO),
            allocated_hard_limit: Some(Duration::ZERO),
            safety_margin: Duration::ZERO,
            min_spend: None,
            allow_stable_early_stop: true,
            stability: self.config.stability,
        }
    }
}

impl Default for TimeManager {
    fn default() -> Self {
        Self::new(TimeManagerConfig::default()).expect("default time manager is valid")
    }
}

const fn has_clock(request: TimeControl) -> bool {
    request.black_time_ms.is_some()
        || request.white_time_ms.is_some()
        || request.byoyomi_ms.is_some()
        || request.black_increment_ms.is_some()
        || request.white_increment_ms.is_some()
}

const fn is_implicit_casual(request: TimeControl) -> bool {
    !request.infinite
        && request.movetime_ms.is_none()
        && !has_clock(request)
        && request.nodes.is_none()
        && request.depth.is_none()
}

const fn deadline_after_margin(allocated_ms: u64, requested_margin_ms: u64) -> u64 {
    let maximum_margin = allocated_ms.saturating_sub(1);
    allocated_ms.saturating_sub(if requested_margin_ms < maximum_margin {
        requested_margin_ms
    } else {
        maximum_margin
    })
}

/// Deadline margin for live clocks: the requested base plus a small share of the
/// allowance, bounded away from dominating short budgets.
fn move_margin_ms(allowance_ms: u64, requested_margin_ms: u64) -> u64 {
    let scaled = allowance_ms / 100;
    let margin = requested_margin_ms.max(scaled.min(250)).min(250);
    margin.min(allowance_ms.saturating_sub(1))
}

/// Controller-independent estimate of whether another completed depth is worth its cost.
/// Instability is only a spending heuristic, never a calibrated probability of improvement.
#[derive(Default)]
pub(crate) struct AdaptiveTimeBudget {
    previous: Option<SearchInfo>,
    previous_iteration_ms: f64,
    previous_growth: f64,
    stable_iterations: u8,
    target_ms: Option<f64>,
}

impl AdaptiveTimeBudget {
    pub(crate) fn target_ms(&self) -> Option<f64> {
        self.target_ms
    }

    pub(crate) fn observe(&mut self, info: &SearchInfo, plan: TimePlan) -> bool {
        let Some(soft) = plan.soft_limit.filter(|_| plan.allow_stable_early_stop) else {
            return false;
        };
        let elapsed_ms = info.elapsed.as_secs_f64() * 1_000.0;
        let delta = self
            .previous
            .as_ref()
            .map(|previous| info.score.saturating_sub(previous.score).saturating_abs());
        let changed = self
            .previous
            .as_ref()
            .is_some_and(|p| p.best_move != info.best_move);
        if !changed && delta.is_some_and(|delta| delta <= plan.stability.stable_score_cp) {
            self.stable_iterations = self.stable_iterations.saturating_add(1);
        } else {
            self.stable_iterations = 0;
        }
        // Several equally reasonable moves are not evidence that a quiet position must
        // consume its whole allowance. Score swings and changed best moves justify more work.
        let factor = if changed || delta.is_some_and(|d| d >= plan.stability.volatile_score_cp) {
            1.5
        } else if self.stable_iterations >= plan.stability.required_stable_iterations {
            0.6
        } else {
            1.0
        };
        let target = (soft.as_secs_f64() * 1_000.0 * factor).min(
            plan.hard_limit
                .map_or(f64::MAX, |hard| hard.as_secs_f64() * 1_000.0),
        );
        self.target_ms = Some(target);
        let iteration_ms = elapsed_ms
            - self
                .previous
                .as_ref()
                .map_or(0.0, |p| p.elapsed.as_secs_f64() * 1_000.0);
        let growth = if self.previous_iteration_ms > 0.0 {
            (iteration_ms / self.previous_iteration_ms).clamp(1.5, 8.0)
        } else {
            2.0
        };
        // Odd/even depths can alternate cheap and expensive iterations. The last
        // ratio alone underestimates the next expensive depth after a cheap one.
        let predicted_growth = growth.max(self.previous_growth);
        let predicted_stop = info.depth >= 2
            && elapsed_ms >= target * 0.15
            && elapsed_ms + iteration_ms * predicted_growth >= target;
        let stop = (elapsed_ms >= target || predicted_stop)
            && elapsed_ms
                >= plan
                    .min_spend
                    .map_or(0.0, |floor| floor.as_secs_f64() * 1_000.0);
        self.previous_growth = growth;
        self.previous_iteration_ms = iteration_ms;
        self.previous = Some(info.clone());
        stop
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    #[test]
    fn casual_plan_is_bounded_by_twenty_seconds_and_margin() {
        let plan = TimeManager::default()
            .plan(Side::Black, TimeControl::casual(), 64)
            .unwrap();
        assert_eq!(plan.mode, TimeControlMode::Casual);
        assert_eq!(
            plan.allocated_hard_limit,
            Some(Duration::from_millis(CASUAL_HARD_MAX_MS))
        );
        assert_eq!(plan.hard_limit, Some(Duration::from_millis(19_950)));
        assert!(plan.soft_limit < plan.hard_limit);
        assert!(plan.allow_stable_early_stop);
    }

    #[test]
    fn byoyomi_only_never_allocates_more_than_available() {
        let request = TimeControl {
            byoyomi_ms: Some(500),
            casual: false,
            ..TimeControl::casual()
        };
        let plan = TimeManager::default()
            .plan(Side::White, request, 8)
            .unwrap();
        assert_eq!(plan.mode, TimeControlMode::Clock);
        assert_eq!(plan.allocated_hard_limit, Some(Duration::from_millis(500)));
        assert_eq!(plan.hard_limit, Some(Duration::from_millis(450)));
    }

    fn clock_request(milliseconds: u64) -> TimeControl {
        TimeControl {
            black_time_ms: Some(milliseconds),
            white_time_ms: Some(milliseconds),
            casual: false,
            ..TimeControl::casual()
        }
    }

    #[test]
    fn live_clock_is_authoritative_over_preset_movetime_on_both_sides() {
        for remaining in [180_000, 600_000] {
            for side in [Side::Black, Side::White] {
                let request = clock_request(remaining);
                let expected = TimeManager::default().plan(side, request, 64).unwrap();
                let contaminated = TimeManager::default()
                    .plan(
                        side,
                        TimeControl {
                            movetime_ms: Some(60_000),
                            ..request
                        },
                        64,
                    )
                    .unwrap();
                assert_eq!(expected, contaminated);
                assert_eq!(expected.mode, TimeControlMode::Clock);
                // Geometric sudden-death target: remaining / 64 shrunk by one tenth.
                assert_eq!(
                    expected.soft_limit,
                    Some(Duration::from_millis(remaining * 9 / 64 / 10))
                );
                assert!(expected.allow_stable_early_stop);
                // One move can never consume more than an eighth of the clock.
                assert!(expected.hard_limit.unwrap() <= Duration::from_millis(remaining / 8));
            }
        }
    }

    #[test]
    fn pure_byoyomi_spends_most_of_each_period_but_keeps_a_margin() {
        for byoyomi in [1_000_u64, 5_000, 30_000, 60_000] {
            let request = TimeControl {
                byoyomi_ms: Some(byoyomi),
                casual: false,
                ..TimeControl::casual()
            };
            let plan = TimeManager::default()
                .plan(Side::Black, request, 64)
                .unwrap();
            assert_eq!(plan.mode, TimeControlMode::Clock);
            let period = u128::from(byoyomi);
            let hard = plan.hard_limit.unwrap().as_millis();
            let soft = plan.soft_limit.unwrap().as_millis();
            let floor = plan.min_spend.unwrap().as_millis();
            assert!(
                hard >= period * 95 / 100,
                "{byoyomi} keeps most of the period"
            );
            assert!(hard <= period, "never plans past the period");
            assert!(
                soft >= period * 3 / 4,
                "{byoyomi} targets most of the period"
            );
            assert!(floor >= period / 2, "{byoyomi} saving is gated below half");
            assert!(floor <= hard);
        }
    }

    #[test]
    fn short_byoyomi_periods_keep_a_positive_margin() {
        let request = TimeControl {
            byoyomi_ms: Some(1_000),
            casual: false,
            ..TimeControl::casual()
        };
        let plan = TimeManager::default()
            .plan(Side::Black, request, 64)
            .unwrap();
        assert_eq!(plan.hard_limit, Some(Duration::from_millis(950)));
        assert_eq!(plan.soft_limit, Some(Duration::from_millis(750)));
    }

    #[test]
    fn sudden_death_targets_decay_with_the_remaining_clock() {
        let manager = TimeManager::default();
        let early = manager
            .plan(Side::Black, clock_request(120_000), 64)
            .unwrap();
        let late = manager
            .plan(Side::Black, clock_request(60_000), 64)
            .unwrap();
        let half_early = early.soft_limit.unwrap() / 2;
        assert!(
            late.soft_limit.unwrap() <= half_early
                && late.soft_limit.unwrap() + Duration::from_millis(1) >= half_early,
            "targets are a fixed fraction of the remaining clock"
        );
        // The emergency ramp raises the fraction from 27/1920 toward 1/12 below 20s:
        // at 15s the target is 15000 * 1070000 / 28800000, and the hard cap stays
        // within an eighth of the clock.
        let low = manager
            .plan(Side::Black, clock_request(15_000), 64)
            .unwrap();
        assert_eq!(low.soft_limit, Some(Duration::from_millis(557)));
        assert!(low.hard_limit.unwrap() <= Duration::from_millis(15_000 / 8));
        // A zero clock never manufactures time, with or without increment.
        for request in [
            clock_request(0),
            TimeControl {
                black_time_ms: Some(0),
                black_increment_ms: Some(30_000),
                casual: false,
                ..TimeControl::casual()
            },
        ] {
            let plan = manager.plan(Side::Black, request, 64).unwrap();
            assert_eq!(plan.soft_limit, Some(Duration::ZERO));
            assert_eq!(plan.hard_limit, Some(Duration::ZERO));
        }
    }

    #[test]
    fn extension_cooldown_tightens_after_a_long_move() {
        let mut manager = TimeManager::default();
        let request = clock_request(120_000);
        let plan = manager.plan(Side::Black, request, 64).unwrap();
        let free_cap = plan.hard_limit.unwrap();
        manager.observe_spend(&plan, Duration::from_millis(60_000));
        let cooled = manager.plan(Side::Black, request, 64).unwrap();
        let cooled_cap = cooled.hard_limit.unwrap();
        assert!(
            cooled_cap < free_cap,
            "a spent extension cools the next moves"
        );
        for _ in 0..3 {
            manager.observe_spend(&cooled, Duration::from_millis(10));
        }
        let recovered = manager.plan(Side::Black, request, 64).unwrap();
        assert_eq!(recovered.hard_limit.unwrap(), free_cap);
    }

    #[test]
    fn emergency_transition_is_continuous_across_the_whole_clock() {
        let manager = TimeManager::default();
        let mut previous: Option<(u64, u64)> = None;
        for remaining in 1..=120_000_u64 {
            let plan = manager
                .plan(Side::Black, clock_request(remaining), 64)
                .unwrap();
            let soft = plan.soft_limit.unwrap().as_millis() as u64;
            let hard = plan.hard_limit.unwrap().as_millis() as u64;
            assert!(
                soft <= hard,
                "{remaining} ms: soft {soft} above hard {hard}"
            );
            assert!(
                hard * 8 <= remaining.max(8),
                "{remaining} ms: hard cap {hard} exceeds an eighth of the clock"
            );
            if let Some((previous_soft, previous_hard)) = previous {
                assert!(
                    soft.abs_diff(previous_soft) <= 2,
                    "soft target jumps {previous_soft} -> {soft} at {remaining} ms"
                );
                assert!(
                    hard.abs_diff(previous_hard) <= 4,
                    "hard cap jumps {previous_hard} -> {hard} at {remaining} ms"
                );
            }
            previous = Some((soft, hard));
        }
        // The historical cliff: crossing the old threshold must be imperceptible.
        let above = manager
            .plan(Side::Black, clock_request(20_001), 64)
            .unwrap();
        let below = manager
            .plan(Side::Black, clock_request(20_000), 64)
            .unwrap();
        assert_eq!(above.soft_limit, below.soft_limit);
        assert!(
            above
                .hard_limit
                .unwrap()
                .as_millis()
                .abs_diff(below.hard_limit.unwrap().as_millis())
                <= 2
        );
    }

    #[test]
    fn clock_fractions_at_the_documented_sample_points() {
        let manager = TimeManager::default();
        // remaining -> soft target, from the 27/1920 -> 160/1920 ramp over 20s..5s.
        for (remaining, expected_soft) in [
            (60_000, 843),
            (30_000, 421),
            (21_000, 295),
            (20_001, 281),
            (20_000, 281),
            (15_000, 557),
            (10_000, 602),
            (5_000, 416),
            (2_000, 166),
        ] {
            let plan = manager
                .plan(Side::Black, clock_request(remaining), 64)
                .unwrap();
            assert_eq!(
                plan.soft_limit.unwrap(),
                Duration::from_millis(expected_soft),
                "{remaining} ms soft target"
            );
        }
    }

    #[test]
    fn low_clock_cooldown_still_tightens_the_extension_cap() {
        let mut manager = TimeManager::default();
        let plan = manager
            .plan(Side::Black, clock_request(10_000), 64)
            .unwrap();
        let free_cap = plan.hard_limit.unwrap();
        manager.observe_spend(&plan, Duration::from_millis(5_000));
        let cooled = manager
            .plan(Side::Black, clock_request(10_000), 64)
            .unwrap();
        assert!(
            cooled.hard_limit.unwrap() < free_cap,
            "a spent extension cools even a dangerously low clock"
        );
    }

    #[test]
    fn clock_survives_a_full_game_of_mixed_positions() {
        // Deterministic whole-game simulation over the plan arithmetic: quiet moves
        // spend the soft target, volatile and tactically critical moves spend the hard
        // cap. The clock must never flag while the critical moves keep their extensions.
        for (initial, expected_moves) in [(180_000_u64, 100), (600_000, 200)] {
            let mut manager = TimeManager::default();
            let mut remaining = initial;
            let mut moves = 0_u32;
            let mut critical_spends = 0_u32;
            while moves < 1_000 {
                let plan = manager
                    .plan(Side::Black, clock_request(remaining), 64)
                    .unwrap();
                let soft = plan.soft_limit.unwrap().as_millis() as u64;
                let hard = plan.hard_limit.unwrap().as_millis() as u64;
                let critical = moves % 9 == 4;
                let volatile = moves % 7 == 2;
                let spent = if critical || volatile { hard } else { soft };
                assert!(spent <= remaining, "move {moves} spent past the flag");
                remaining -= spent;
                manager.observe_spend(&plan, Duration::from_millis(spent));
                moves += 1;
                if critical {
                    critical_spends += 1;
                    // Even a cooled critical move reserves extension headroom above the
                    // quiet target; the safety margin may consume part of it, but the
                    // allocation itself always exceeds the soft target.
                    assert!(
                        plan.allocated_hard_limit.unwrap() > plan.soft_limit.unwrap(),
                        "move {moves} lost its extension headroom"
                    );
                }
                if remaining < 1_000 {
                    break;
                }
            }
            assert!(
                moves >= expected_moves,
                "{initial} ms clock lasted only {moves} moves"
            );
            assert!(
                critical_spends > 10,
                "simulation never exercised extensions"
            );
        }
    }

    #[test]
    fn fischer_clock_earns_income_and_survives_the_same_way() {
        let mut manager = TimeManager::default();
        let mut remaining = 60_000_u64;
        for moves in 0..300 {
            let request = TimeControl {
                black_time_ms: Some(remaining),
                black_increment_ms: Some(5_000),
                casual: false,
                ..TimeControl::casual()
            };
            let plan = manager.plan(Side::Black, request, 64).unwrap();
            let soft = plan.soft_limit.unwrap().as_millis() as u64;
            let spent = if moves % 11 == 5 {
                plan.hard_limit.unwrap().as_millis() as u64
            } else {
                soft
            };
            assert!(spent <= remaining + 5_000, "move {moves} overshot income");
            remaining = remaining + 5_000 - spent;
            manager.observe_spend(&plan, Duration::from_millis(spent));
            assert!(remaining > 0, "fischer clock flagged on move {moves}");
        }
    }

    #[test]
    fn fischer_increment_adds_income_to_the_target() {
        let request = TimeControl {
            black_time_ms: Some(60_000),
            black_increment_ms: Some(5_000),
            casual: false,
            ..TimeControl::casual()
        };
        let plan = TimeManager::default()
            .plan(Side::Black, request, 64)
            .unwrap();
        let expected = 60_000 / 64 * 9 / 10 + 5_000 * 3 / 4;
        assert_eq!(plan.soft_limit, Some(Duration::from_millis(expected)));
        let plan = TimeManager::default()
            .plan_for_position(&Position::startpos(), request, 64)
            .unwrap();
        assert_eq!(plan.soft_limit, Some(Duration::from_millis(expected)));
    }

    #[test]
    fn exhausted_clock_does_not_borrow_increment_or_manufacture_time() {
        let plan = TimeManager::default()
            .plan(
                Side::White,
                TimeControl {
                    white_time_ms: Some(0),
                    white_increment_ms: Some(30_000),
                    casual: false,
                    ..TimeControl::casual()
                },
                64,
            )
            .unwrap();
        assert_eq!(plan.hard_limit, Some(Duration::ZERO));
        assert_eq!(plan.soft_limit, Some(Duration::ZERO));
    }

    fn iteration(depth: u8, elapsed_ms: u64, score: i32) -> SearchInfo {
        SearchInfo {
            best_move: Position::startpos().legal_moves().first().copied(),
            score,
            depth,
            seldepth: depth,
            nodes: u64::from(depth) * 100,
            elapsed: Duration::from_millis(elapsed_ms),
            nps: 100,
            pv: Vec::new(),
            root_moves: Vec::new(),
            stats: crate::SearchStats::default(),
        }
    }

    #[test]
    fn stable_search_saves_time_without_waiting_for_the_target() {
        let plan = TimeManager::default()
            .plan(Side::Black, clock_request(600_000), 64)
            .unwrap();
        let mut budget = AdaptiveTimeBudget::default();
        assert!(!budget.observe(&iteration(1, 100, 10), plan));
        assert!(!budget.observe(&iteration(2, 400, 15), plan));
        assert!(budget.observe(&iteration(3, 1_600, 18), plan));
        assert_eq!(budget.target_ms(), Some(5_062.2));
        assert!(Duration::from_millis(1_600) < plan.soft_limit.unwrap());
    }

    #[test]
    fn volatile_evidence_can_increase_target_but_never_its_absolute_cap() {
        let mut plan = TimeManager::default()
            .plan(Side::Black, clock_request(600_000), 64)
            .unwrap();
        plan.hard_limit = Some(Duration::from_secs(7));
        let mut budget = AdaptiveTimeBudget::default();
        assert!(!budget.observe(&iteration(1, 100, 10), plan));
        assert!(!budget.observe(&iteration(2, 300, -400), plan));
        assert_eq!(budget.target_ms(), Some(7_000.0));
        assert_eq!(plan.hard_limit, Some(Duration::from_secs(7)));
    }

    #[test]
    fn alternating_depth_cost_does_not_spend_another_expensive_iteration() {
        let plan = TimeManager::default()
            .plan(Side::Black, clock_request(600_000), 64)
            .unwrap();
        let mut budget = AdaptiveTimeBudget::default();
        for (depth, elapsed, score) in [
            (1, 1, 49),
            (2, 2, 61),
            (3, 11, 52),
            (4, 36, 41),
            (5, 298, 45),
        ] {
            assert!(!budget.observe(&iteration(depth, elapsed, score), plan));
        }
        assert!(budget.observe(&iteration(6, 1_098, 48), plan));
        assert_eq!(budget.target_ms(), Some(5_062.2));
    }

    proptest! {
        #[test]
        fn clock_plans_never_exceed_available_time(
            remaining in 0_u64..=MAX_CLOCK_MS,
            byoyomi in 0_u64..=MAX_MOVE_TIME_MS,
            increment in 0_u64..=MAX_MOVE_TIME_MS,
            margin in 0_u64..=MAX_SAFETY_MARGIN_MS,
        ) {
            let request = TimeControl {
                black_time_ms: Some(remaining),
                byoyomi_ms: Some(byoyomi),
                black_increment_ms: Some(increment),
                casual: false,
                safety_margin_ms: margin,
                ..TimeControl::casual()
            };
            let plan = TimeManager::default().plan(Side::Black, request, 64).unwrap();
            let available = remaining.saturating_add(byoyomi);
            prop_assert!(plan.allocated_hard_limit.unwrap() <= Duration::from_millis(available));
            prop_assert!(plan.hard_limit.unwrap() <= plan.allocated_hard_limit.unwrap());
            prop_assert!(plan.soft_limit.unwrap() <= plan.hard_limit.unwrap());
        }
    }
}
