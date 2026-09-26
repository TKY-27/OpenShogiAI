//! Authoritative USI-to-internal clock translation shared by both adapters.
//!
//! The maintained USI definition lets the engine spend `remaining + increment` (or
//! `remaining + byoyomi`) on the move, and `ShogiHome`'s native transport subtracts the
//! increment from its already-incremented clock before sending `btime`/`wtime`. The
//! current spendable time is therefore `time + increment`; the increment fields
//! themselves only pace future moves. Normalizing here keeps the shared core and the
//! browser callers — whose inputs already carry current spendable time — unchanged.

use crate::GoParameters;
use open_shogi_core::MAX_CLOCK_MS;

/// Builds the shared request from parsed USI clocks. `casual` is always false: a
/// missing clock is an adapter-level error, never an invitation to spend the casual
/// twenty-second budget on a timed game.
pub(crate) fn time_control(
    parameters: &GoParameters,
    safety_margin_ms: u64,
) -> open_shogi_core::TimeControl {
    open_shogi_core::TimeControl {
        black_time_ms: spendable_time(parameters.black_time_ms, parameters.black_increment_ms),
        white_time_ms: spendable_time(parameters.white_time_ms, parameters.white_increment_ms),
        byoyomi_ms: parameters.byoyomi_ms,
        black_increment_ms: parameters.black_increment_ms,
        white_increment_ms: parameters.white_increment_ms,
        movetime_ms: parameters.movetime_ms,
        nodes: parameters.nodes,
        depth: parameters.depth,
        infinite: parameters.infinite,
        casual: false,
        safety_margin_ms,
    }
}

/// A zero base clock with a positive increment is a valid Fischer turn and must keep
/// its increment as usable time; the defensive `MAX_CLOCK_MS` cap never creates extra
/// time because every accepted USI value is already far below it.
fn spendable_time(remaining_ms: Option<u64>, increment_ms: Option<u64>) -> Option<u64> {
    let remaining = remaining_ms?;
    Some(
        remaining
            .saturating_add(increment_ms.unwrap_or(0))
            .min(MAX_CLOCK_MS),
    )
}

#[cfg(test)]
mod tests {
    use super::time_control;
    use crate::GoParameters;
    use open_shogi_core::TimeControl;

    fn control(parameters: &GoParameters) -> open_shogi_core::TimeControl {
        time_control(parameters, 50)
    }

    #[test]
    fn increment_only_fischer_turns_receive_their_increment() {
        for (base, increment) in [(0, 1_000), (60_000, 5_000)] {
            let parameters = GoParameters {
                black_time_ms: Some(base),
                white_time_ms: Some(base),
                black_increment_ms: Some(increment),
                white_increment_ms: Some(increment),
                ..GoParameters::default()
            };
            let request = control(&parameters);
            assert_eq!(request.black_time_ms, Some(base + increment));
            assert_eq!(request.white_time_ms, Some(base + increment));
            assert_eq!(request.black_increment_ms, Some(increment));
            assert!(!request.casual);
        }
    }

    #[test]
    fn byoyomi_stays_current_time_and_increments_stay_pacing_fields() {
        let parameters = GoParameters {
            black_time_ms: Some(180_000),
            white_time_ms: Some(120_000),
            byoyomi_ms: Some(30_000),
            ..GoParameters::default()
        };
        let request = control(&parameters);
        assert_eq!(request.black_time_ms, Some(180_000));
        assert_eq!(request.white_time_ms, Some(120_000));
        assert_eq!(request.byoyomi_ms, Some(30_000));
        assert_eq!(request.black_increment_ms, None);
    }

    #[test]
    fn fixed_and_unbounded_modes_pass_through_untouched() {
        let movetime = GoParameters {
            movetime_ms: Some(5_000),
            ..GoParameters::default()
        };
        assert_eq!(
            control(&movetime),
            TimeControl {
                movetime_ms: Some(5_000),
                casual: false,
                safety_margin_ms: 50,
                ..TimeControl::casual()
            }
        );
        let infinite = GoParameters {
            infinite: true,
            ..GoParameters::default()
        };
        assert!(control(&infinite).infinite);
    }
}
