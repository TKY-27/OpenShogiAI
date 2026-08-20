//! Deterministic, non-learned evaluation used by the first search engine.

use crate::{HandPiece, PieceKind, Position, Side};

const MATERIAL_VALUES: [i32; 8] = [100, 300, 320, 400, 500, 700, 850, 0];
const PROMOTION_VALUES: [i32; 14] = [0, 0, 0, 0, 0, 0, 0, 0, 420, 220, 200, 120, 220, 250];
const HAND_PREMIUM_PERCENT: i32 = 110;
const KING_DEFENDER_VALUE: i32 = 18;
const KING_ATTACKED_RING_PENALTY: i32 = 14;
const MOBILITY_VALUE: i32 = 2;
const CHECK_VALUE: i32 = 45;
const ENEMY_CAMP_VALUE: i32 = 12;
const TEMPO_VALUE: i32 = 10;

/// Independently switchable terms in the baseline hand-written evaluation.
#[expect(
    clippy::struct_excessive_bools,
    reason = "each named evaluation term must be independently switchable"
)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EvaluationConfig {
    pub material: bool,
    pub hands: bool,
    pub promotions: bool,
    pub king_safety: bool,
    pub mobility: bool,
    pub check: bool,
    pub enemy_camp: bool,
    pub tempo: bool,
}

impl EvaluationConfig {
    /// Material-only profile used for the first reproducible evaluator rung.
    #[must_use]
    pub const fn material_only() -> Self {
        Self {
            material: true,
            hands: false,
            promotions: false,
            king_safety: false,
            mobility: false,
            check: false,
            enemy_camp: false,
            tempo: false,
        }
    }

    /// Stable handcrafted baseline used by evaluator comparisons.
    #[must_use]
    pub const fn handcrafted_baseline() -> Self {
        Self {
            material: true,
            hands: true,
            promotions: true,
            king_safety: false,
            mobility: false,
            check: false,
            enemy_camp: false,
            tempo: true,
        }
    }

    /// Full handcrafted evaluator used by the experimental search profile.
    #[must_use]
    pub const fn handcrafted_experimental() -> Self {
        Self {
            material: true,
            hands: true,
            promotions: true,
            king_safety: true,
            mobility: true,
            check: true,
            enemy_camp: true,
            tempo: true,
        }
    }
}

impl Default for EvaluationConfig {
    fn default() -> Self {
        crate::overall_champion_evaluation()
    }
}

/// Signed term contributions from the side-to-move's perspective.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct EvaluationBreakdown {
    pub material: i32,
    pub hands: i32,
    pub promotions: i32,
    pub king_safety: i32,
    pub mobility: i32,
    pub check: i32,
    pub enemy_camp: i32,
    pub tempo: i32,
}

impl EvaluationBreakdown {
    /// Returns the sum of all enabled term contributions.
    #[must_use]
    pub const fn total(self) -> i32 {
        self.material
            + self.hands
            + self.promotions
            + self.king_safety
            + self.mobility
            + self.check
            + self.enemy_camp
            + self.tempo
    }
}

/// Evaluates a position from the side-to-move's perspective.
#[must_use]
pub fn evaluate(position: &Position, config: &EvaluationConfig) -> i32 {
    evaluate_breakdown(position, config).total()
}

/// Returns the independently attributable baseline evaluation terms.
#[must_use]
pub fn evaluate_breakdown(position: &Position, config: &EvaluationConfig) -> EvaluationBreakdown {
    let perspective = position.side_to_move();
    let opponent = perspective.opposite();
    let relative = |ours: i32, theirs: i32| ours - theirs;

    EvaluationBreakdown {
        material: if config.material {
            relative(
                board_material(position, perspective),
                board_material(position, opponent),
            )
        } else {
            0
        },
        hands: if config.hands {
            relative(
                hand_material(position, perspective),
                hand_material(position, opponent),
            )
        } else {
            0
        },
        promotions: if config.promotions {
            relative(
                promotion_material(position, perspective),
                promotion_material(position, opponent),
            )
        } else {
            0
        },
        king_safety: if config.king_safety {
            relative(
                king_safety(position, perspective),
                king_safety(position, opponent),
            )
        } else {
            0
        },
        mobility: if config.mobility {
            let ours = i32::try_from(position.pseudo_mobility(perspective)).unwrap_or(i32::MAX);
            let theirs = i32::try_from(position.pseudo_mobility(opponent)).unwrap_or(i32::MAX);
            relative(ours, theirs) * MOBILITY_VALUE
        } else {
            0
        },
        check: if config.check {
            i32::from(position.is_in_check(opponent)) * CHECK_VALUE
                - i32::from(position.is_in_check(perspective)) * CHECK_VALUE
        } else {
            0
        },
        enemy_camp: if config.enemy_camp {
            relative(
                enemy_camp_presence(position, perspective),
                enemy_camp_presence(position, opponent),
            )
        } else {
            0
        },
        tempo: i32::from(config.tempo) * TEMPO_VALUE,
    }
}

fn board_material(position: &Position, side: Side) -> i32 {
    position
        .board()
        .iter()
        .flatten()
        .filter(|piece| piece.side == side)
        .map(|piece| MATERIAL_VALUES[piece.kind.unpromoted().index()])
        .sum()
}

fn hand_material(position: &Position, side: Side) -> i32 {
    HandPiece::ALL
        .into_iter()
        .map(|piece| {
            let count = i32::from(position.hand(side).count(piece));
            count * MATERIAL_VALUES[piece.piece_kind().index()] * HAND_PREMIUM_PERCENT / 100
        })
        .sum()
}

fn promotion_material(position: &Position, side: Side) -> i32 {
    position
        .board()
        .iter()
        .flatten()
        .filter(|piece| piece.side == side)
        .map(|piece| PROMOTION_VALUES[piece.kind.index()])
        .sum()
}

fn king_safety(position: &Position, side: Side) -> i32 {
    let Some(king) = position.king_square(side) else {
        return 0;
    };

    let mut score = 0;
    for file_delta in -1..=1 {
        for rank_delta in -1..=1 {
            if file_delta == 0 && rank_delta == 0 {
                continue;
            }
            let Some(square) = king.offset(file_delta, rank_delta) else {
                continue;
            };
            if position
                .piece_at(square)
                .is_some_and(|piece| piece.side == side)
            {
                score += KING_DEFENDER_VALUE;
            }
            if position.square_is_attacked(square, side.opposite()) {
                score -= KING_ATTACKED_RING_PENALTY;
            }
        }
    }
    score
}

fn enemy_camp_presence(position: &Position, side: Side) -> i32 {
    position
        .board()
        .iter()
        .enumerate()
        .filter_map(|(index, piece)| piece.map(|piece| (index, piece)))
        .filter(|(_, piece)| piece.side == side && piece.kind != PieceKind::King)
        .filter(|(index, _)| {
            let rank = index / 9 + 1;
            match side {
                Side::Black => rank <= 3,
                Side::White => rank >= 7,
            }
        })
        .count()
        .try_into()
        .unwrap_or(i32::MAX)
        * ENEMY_CAMP_VALUE
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Move, Piece, Square};

    fn square(file: u8, rank: u8) -> Square {
        Square::new(file, rank).expect("test square")
    }

    fn kings_only() -> [Option<Piece>; crate::BOARD_SQUARES] {
        let mut board = [None; crate::BOARD_SQUARES];
        board[square(5, 9).index()] = Some(Piece::new(Side::Black, PieceKind::King));
        board[square(5, 1).index()] = Some(Piece::new(Side::White, PieceKind::King));
        board
    }

    #[test]
    fn every_term_can_be_disabled_independently() {
        let config = EvaluationConfig {
            material: false,
            hands: false,
            promotions: false,
            king_safety: false,
            mobility: false,
            check: false,
            enemy_camp: false,
            tempo: false,
        };

        assert_eq!(evaluate(&Position::startpos(), &config), 0);
        assert_eq!(
            evaluate_breakdown(&Position::startpos(), &config),
            EvaluationBreakdown::default()
        );
    }

    #[test]
    fn material_and_promotion_are_separate_terms() {
        let mut board = kings_only();
        board[square(5, 5).index()] = Some(Piece::new(Side::Black, PieceKind::PromotedPawn));
        let position =
            Position::from_parts(board, Default::default(), Side::Black, 1).expect("position");
        let breakdown = evaluate_breakdown(&position, &EvaluationConfig::default());

        assert_eq!(breakdown.material, MATERIAL_VALUES[PieceKind::Pawn.index()]);
        assert_eq!(
            breakdown.promotions,
            PROMOTION_VALUES[PieceKind::PromotedPawn.index()]
        );
    }

    #[test]
    fn generated_move_round_trip_preserves_exact_state() {
        let mut position = Position::startpos();
        let original = position.clone();

        for movement in original.legal_moves() {
            let undo = position.make_generated_move(movement);
            assert_eq!(position.zobrist_hash(), position.recompute_zobrist());
            position.unmake_move(undo);
            assert_eq!(position, original);
        }
    }

    #[test]
    fn capture_updates_and_restores_incremental_hash() {
        let mut board = kings_only();
        board[square(5, 5).index()] = Some(Piece::new(Side::Black, PieceKind::Silver));
        board[square(4, 4).index()] = Some(Piece::new(Side::White, PieceKind::Pawn));
        let mut position =
            Position::from_parts(board, Default::default(), Side::Black, 1).expect("position");
        let original = position.clone();
        let movement = Move::Normal {
            from: square(5, 5),
            to: square(4, 4),
            promote: false,
        };

        let undo = position.make_generated_move(movement);
        assert_eq!(position.zobrist_hash(), position.recompute_zobrist());
        position.unmake_move(undo);
        assert_eq!(position, original);
    }

    #[test]
    fn differential_undo_is_smaller_than_a_position_snapshot() {
        assert!(std::mem::size_of::<crate::Undo>() < std::mem::size_of::<Position>());
    }
}
