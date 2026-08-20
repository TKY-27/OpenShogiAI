//! Deterministic legal-move tree traversal for rule verification.

use std::{error::Error, fmt};

use crate::{IllegalMove, Move, Position};

/// A perft traversal could not produce an exact result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PerftError {
    /// A generated move could not be applied to the same position.
    MoveApplication(IllegalMove),
    /// At least one exact `u64` counter would overflow.
    CounterOverflow,
}

impl fmt::Display for PerftError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MoveApplication(error) => {
                write!(formatter, "generated move could not be applied: {error}")
            }
            Self::CounterOverflow => formatter.write_str("perft counter exceeds `u64`"),
        }
    }
}

impl Error for PerftError {}

/// Aggregate counters collected by a perft traversal.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PerftResult {
    pub nodes: u64,
    pub captures: u64,
    pub promotions: u64,
    pub drops: u64,
    pub checks: u64,
    pub checkmates: u64,
}

impl PerftResult {
    fn add(&mut self, other: Self) -> Result<(), PerftError> {
        checked_add(&mut self.nodes, other.nodes)?;
        checked_add(&mut self.captures, other.captures)?;
        checked_add(&mut self.promotions, other.promotions)?;
        checked_add(&mut self.drops, other.drops)?;
        checked_add(&mut self.checks, other.checks)?;
        checked_add(&mut self.checkmates, other.checkmates)
    }
}

fn checked_add(counter: &mut u64, amount: u64) -> Result<(), PerftError> {
    *counter = counter
        .checked_add(amount)
        .ok_or(PerftError::CounterOverflow)?;
    Ok(())
}

/// Traverses every legal move sequence up to `depth`.
///
/// Depth zero contains the root node. Counts are intended for deterministic rule regression,
/// not as a playing-strength benchmark.
///
/// # Errors
///
/// Returns [`PerftError`] instead of publishing a partial or saturated count.
pub fn perft(position: &Position, depth: u8) -> Result<PerftResult, PerftError> {
    if depth == 0 {
        return Ok(PerftResult {
            nodes: 1,
            ..PerftResult::default()
        });
    }

    let mut working = position.clone();
    perft_inner(&mut working, depth)
}

/// Returns the node count beneath every legal root move.
///
/// # Errors
///
/// Returns [`PerftError`] if a generated move cannot be applied or an exact counter overflows.
pub fn perft_divide(position: &Position, depth: u8) -> Result<Vec<(Move, u64)>, PerftError> {
    if depth == 0 {
        return Ok(Vec::new());
    }

    let mut working = position.clone();
    let moves = working.legal_moves();
    let mut result = Vec::with_capacity(moves.len());
    for mv in moves {
        let undo = working.make_move(mv).map_err(PerftError::MoveApplication)?;
        let subtree = perft_inner(&mut working, depth - 1);
        working.unmake_move(undo);
        result.push((mv, subtree?.nodes));
    }
    Ok(result)
}

fn perft_inner(position: &mut Position, depth: u8) -> Result<PerftResult, PerftError> {
    if depth == 0 {
        return Ok(PerftResult {
            nodes: 1,
            ..PerftResult::default()
        });
    }
    let moves = position.legal_moves();
    let mut result = PerftResult::default();

    for mv in moves {
        let captured = match mv {
            Move::Normal { to, .. } => position.piece_at(to).is_some(),
            Move::Drop { .. } => false,
        };
        let undo = position
            .make_move(mv)
            .map_err(PerftError::MoveApplication)?;
        let gives_check = position.is_in_check(position.side_to_move());
        let checkmate = gives_check && position.is_checkmate();

        if depth == 1 {
            checked_add(&mut result.nodes, 1)?;
            checked_add(&mut result.captures, u64::from(captured))?;
            checked_add(
                &mut result.promotions,
                u64::from(matches!(mv, Move::Normal { promote: true, .. })),
            )?;
            checked_add(
                &mut result.drops,
                u64::from(matches!(mv, Move::Drop { .. })),
            )?;
            checked_add(&mut result.checks, u64::from(gives_check))?;
            checked_add(&mut result.checkmates, u64::from(checkmate))?;
        } else {
            let subtree = perft_inner(position, depth - 1);
            position.unmake_move(undo);
            result.add(subtree?)?;
            continue;
        }
        position.unmake_move(undo);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::{PerftError, PerftResult, perft, perft_divide};
    use crate::Position;

    #[test]
    fn zero_depth_counts_the_root() {
        assert_eq!(
            perft(&Position::startpos(), 0)
                .expect("exact root count")
                .nodes,
            1
        );
    }

    #[test]
    fn first_depth_matches_generated_legal_moves() {
        let position = Position::startpos();
        assert_eq!(
            perft(&position, 1).expect("exact move count").nodes,
            u64::try_from(position.legal_moves().len()).expect("move count fits u64")
        );
    }

    #[test]
    fn first_depth_divide_counts_one_node_per_root_move() {
        let position = Position::startpos();
        let divide = perft_divide(&position, 1).expect("exact divide");
        assert_eq!(divide.len(), position.legal_moves().len());
        assert!(divide.into_iter().all(|(_, nodes)| nodes == 1));
    }

    #[test]
    fn counter_overflow_is_reported() {
        let mut result = PerftResult {
            nodes: u64::MAX,
            ..PerftResult::default()
        };
        assert_eq!(
            result.add(PerftResult {
                nodes: 1,
                ..PerftResult::default()
            }),
            Err(PerftError::CounterOverflow)
        );
    }
}
