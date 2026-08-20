//! Versioned, adapter-neutral play time management.

use std::time::Duration;

use crate::Side;

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

/// Stateless allocator; callers retain clock state and subtract measured elapsed time.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TimeManager {
    config: TimeManagerConfig,
}

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
        Ok(Self { config })
    }

    /// Allocates soft and hard monotonic durations for the side to move.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid shared request.
    pub fn plan(
        self,
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
                allow_stable_early_stop: true,
                stability: self.config.stability,
            });
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
                allow_stable_early_stop: false,
                stability: self.config.stability,
            });
        }
        if has_clock(request) {
            return Ok(self.clock_plan(side, request, max_depth));
        }
        let mode = if request.nodes.is_some() {
            TimeControlMode::Nodes
        } else {
            TimeControlMode::Depth
        };
        Ok(self.plan_without_deadline(mode, max_depth, request.nodes))
    }

    fn plan_without_deadline(
        self,
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
            allow_stable_early_stop: false,
            stability: self.config.stability,
        }
    }

    fn clock_plan(self, side: Side, request: TimeControl, max_depth: u8) -> TimePlan {
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
        // Increment is earned after the move, so it improves the target share but is not part of
        // the amount that may be consumed before this move is returned.
        let target = (remaining / 30)
            .saturating_add(increment.saturating_mul(3) / 4)
            .saturating_add(byoyomi.saturating_mul(3) / 4)
            .max(1);
        let available = remaining.saturating_add(byoyomi).max(1);
        let allocated = target.saturating_mul(3).max(byoyomi).min(available);
        let hard = deadline_after_margin(allocated, request.safety_margin_ms);
        let soft = target.min(hard);
        TimePlan {
            mode: TimeControlMode::Clock,
            max_depth,
            max_nodes: request.nodes,
            soft_limit: Some(Duration::from_millis(soft)),
            hard_limit: Some(Duration::from_millis(hard)),
            allocated_hard_limit: Some(Duration::from_millis(allocated)),
            safety_margin: Duration::from_millis(allocated.saturating_sub(hard)),
            allow_stable_early_stop: false,
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
            let available = remaining.saturating_add(byoyomi).max(1);
            prop_assert!(plan.allocated_hard_limit.unwrap() <= Duration::from_millis(available));
            prop_assert!(plan.hard_limit.unwrap() <= plan.allocated_hard_limit.unwrap());
            prop_assert!(plan.soft_limit.unwrap() <= plan.hard_limit.unwrap());
        }
    }
}
