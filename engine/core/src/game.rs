//! Game history, repetition adjudication, and configurable entering-king rules.

use crate::{HandPiece, Move, PieceKind, Position, Side, Undo};

/// Result of a fourfold position repetition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RepetitionOutcome {
    /// Ordinary repetition. JSA play restarts with colours exchanged rather than recording a
    /// normal draw; adapters may map this to their competition-specific result.
    NoContest,
    /// The named side gave check on every one of its moves in the repeated sequence and loses.
    PerpetualCheckLoss(Side),
}

/// Terminal state recorded by the rules layer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GameEnd {
    Checkmate {
        winner: Side,
    },
    /// Loss without check when the side to move has no legal move (CSA rule 27.1.1).
    NoLegalMoves {
        loser: Side,
    },
    Resignation {
        loser: Side,
    },
    Repetition(RepetitionOutcome),
    Impasse(ImpasseOutcome),
    EnteringKing(EnteringKingDeclaration),
}

/// Result of explicitly adjudicating a mutually agreed entering-king impasse.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ImpasseOutcome {
    NoContest {
        black_points: u8,
        white_points: u8,
    },
    Loss {
        loser: Side,
        black_points: u8,
        white_points: u8,
    },
    /// A composed or incomplete-material position put both sides below the official threshold.
    InvalidMaterial {
        black_points: u8,
        white_points: u8,
    },
}

/// Caller-supplied part of an impasse decision that board geometry cannot prove.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ImpasseCondition {
    /// The players or tournament arbiter have not established an impasse.
    NotEstablished,
    /// At least one king has entered and the players or arbiter established that neither side
    /// has a reasonable prospect of checkmating the other.
    NoMatingProspect,
}

/// Configurable entering-king declaration policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EnteringKingRule {
    /// Declaration is unavailable, useful for rule profiles that adjudicate externally.
    Disabled,
    /// Japanese Shogi Association official 24-point declaration rule, revised in 2025.
    JsaOfficial2025,
    /// A tournament-specific profile. Large pieces are always worth five and other non-kings
    /// one; points count only pieces in the enemy camp plus pieces in hand.
    Custom {
        minimum_camp_pieces: u8,
        win_points: u8,
        no_contest_points: u8,
    },
}

/// Outcome of attempting an entering-king declaration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EnteringKingDeclaration {
    /// The declaration is not enabled for the selected profile.
    Disabled,
    /// The selected side cannot declare now because it is not that side's turn, the game has
    /// ended, or the 500-move boundary has already been reached.
    Unavailable { side: Side },
    /// The declaring side satisfies the win threshold.
    Win { side: Side, points: u8 },
    /// The declaration satisfies the lower threshold and requires a replay/no-contest.
    NoContest { side: Side, points: u8 },
    /// One or more declaration conditions failed; under the official rule the declarer loses.
    InvalidLoss { side: Side },
}

/// A replayable game record independent of CSA or UI presentation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GameRecord {
    pub initial_position: Position,
    pub moves: Vec<Move>,
    pub end: Option<GameEnd>,
}

/// A position plus the history needed for repetition and undo.
#[derive(Clone, Debug)]
pub struct Game {
    position: Position,
    positions: Vec<Position>,
    moves: Vec<Move>,
    gave_check: Vec<bool>,
    undos: Vec<Undo>,
    end: Option<GameEnd>,
}

impl Game {
    /// Starts a game at the standard initial position.
    #[must_use]
    pub fn startpos() -> Self {
        Self::new(Position::startpos())
    }

    /// Starts a game from an arbitrary validated position.
    #[must_use]
    pub fn new(position: Position) -> Self {
        let end = no_legal_move_end(&position);
        Self {
            positions: vec![position.clone()],
            position,
            moves: Vec::new(),
            gave_check: Vec::new(),
            undos: Vec::new(),
            end,
        }
    }

    #[must_use]
    pub const fn position(&self) -> &Position {
        &self.position
    }

    #[must_use]
    pub fn moves(&self) -> &[Move] {
        &self.moves
    }

    #[must_use]
    pub const fn end(&self) -> Option<GameEnd> {
        self.end
    }

    /// Applies a legal move and updates repetition and checkmate adjudication.
    ///
    /// # Errors
    ///
    /// Returns [`crate::IllegalMove`] if the move is not legal in the current position.
    pub fn play(&mut self, mv: Move) -> Result<Option<GameEnd>, crate::IllegalMove> {
        if self.end.is_some() {
            return Err(crate::IllegalMove::GameAlreadyEnded);
        }

        let undo = self.position.make_move(mv)?;
        let checked_side = self.position.side_to_move();
        let check = self.position.is_in_check(checked_side);
        self.moves.push(mv);
        self.gave_check.push(check);
        self.undos.push(undo);
        self.positions.push(self.position.clone());

        self.end = if let Some(repetition) = self.repetition_outcome() {
            Some(GameEnd::Repetition(repetition))
        } else {
            no_legal_move_end(&self.position)
        };
        Ok(self.end)
    }

    /// Records resignation by the side to move.
    ///
    /// # Errors
    ///
    /// Returns [`crate::IllegalMove::GameAlreadyEnded`] without modifying the recorded result
    /// if the game is already terminal.
    pub fn resign(&mut self) -> Result<GameEnd, crate::IllegalMove> {
        if self.end.is_some() {
            return Err(crate::IllegalMove::GameAlreadyEnded);
        }
        let end = GameEnd::Resignation {
            loser: self.position.side_to_move(),
        };
        self.end = Some(end);
        Ok(end)
    }

    /// Undoes the latest played move.
    ///
    /// Returns `false` when the game is already at its initial position.
    #[must_use]
    pub fn undo(&mut self) -> bool {
        let Some(undo) = self.undos.pop() else {
            return false;
        };
        self.position.unmake_move(undo);
        self.moves.pop();
        self.gave_check.pop();
        self.positions.pop();
        self.end = None;
        true
    }

    /// Evaluates the JSA mutual-agreement 24-point impasse rule.
    ///
    /// The rules core cannot infer that neither side has a reasonable mating prospect, so the
    /// caller must supply [`ImpasseCondition::NoMatingProspect`]. It returns `None` unless
    /// that condition is supplied and at least one king has entered the enemy camp.
    #[must_use]
    pub fn adjudicate_impasse(&self, condition: ImpasseCondition) -> Option<ImpasseOutcome> {
        if self.end.is_some()
            || condition != ImpasseCondition::NoMatingProspect
            || !position_has_entering_king(&self.position)
        {
            return None;
        }

        let black_points = total_points(&self.position, Side::Black);
        let white_points = total_points(&self.position, Side::White);
        match (black_points >= 24, white_points >= 24) {
            (true, true) => Some(ImpasseOutcome::NoContest {
                black_points,
                white_points,
            }),
            (false, true) => Some(ImpasseOutcome::Loss {
                loser: Side::Black,
                black_points,
                white_points,
            }),
            (true, false) => Some(ImpasseOutcome::Loss {
                loser: Side::White,
                black_points,
                white_points,
            }),
            (false, false) => Some(ImpasseOutcome::InvalidMaterial {
                black_points,
                white_points,
            }),
        }
    }

    /// Returns whether the official 500-move no-contest point has been reached.
    ///
    /// If move 500 gave check, adjudication is delayed until that checking side first makes a
    /// later non-checking move. The caller must separately confirm no reasonable mating
    /// prospect with [`ImpasseCondition::NoMatingProspect`]. Only moves retained by this
    /// `Game` instance are considered; an isolated SFEN cannot reconstruct this history.
    #[must_use]
    pub fn five_hundred_move_no_contest(&self, condition: ImpasseCondition) -> bool {
        const LIMIT_INDEX: usize = 499;
        if self.end.is_some()
            || condition != ImpasseCondition::NoMatingProspect
            || self.moves.len() <= LIMIT_INDEX
        {
            return false;
        }
        let Some(limit_position) = self.positions.get(LIMIT_INDEX + 1) else {
            return false;
        };
        if !position_has_entering_king(limit_position) {
            return false;
        }
        if !self.gave_check[LIMIT_INDEX] {
            return true;
        }

        let checking_side = self.positions[LIMIT_INDEX].side_to_move();
        (LIMIT_INDEX + 1..self.moves.len()).any(|move_index| {
            self.positions[move_index].side_to_move() == checking_side
                && !self.gave_check[move_index]
        })
    }

    /// Returns the current fourfold repetition result, if any.
    #[must_use]
    pub fn repetition_outcome(&self) -> Option<RepetitionOutcome> {
        repetition_outcome_from_history(&self.positions, &self.gave_check)
    }

    /// Evaluates an entering-king declaration for `side` under a selected profile.
    #[must_use]
    pub fn entering_king_declaration(
        &self,
        side: Side,
        rule: EnteringKingRule,
    ) -> EnteringKingDeclaration {
        if side != self.position.side_to_move()
            || self.end.is_some()
            || self.position.move_number() > 500
        {
            return EnteringKingDeclaration::Unavailable { side };
        }
        let (minimum_camp_pieces, win_points, no_contest_points) = match rule {
            EnteringKingRule::Disabled => return EnteringKingDeclaration::Disabled,
            EnteringKingRule::JsaOfficial2025 => (10, 31, 24),
            EnteringKingRule::Custom {
                minimum_camp_pieces,
                win_points,
                no_contest_points,
            } => (minimum_camp_pieces, win_points, no_contest_points),
        };

        if self.position.is_in_check(side) {
            return EnteringKingDeclaration::InvalidLoss { side };
        }
        let Some(king_square) = self.position.king_square(side) else {
            return EnteringKingDeclaration::InvalidLoss { side };
        };
        if !in_enemy_camp(side, king_square.rank()) {
            return EnteringKingDeclaration::InvalidLoss { side };
        }

        let camp_pieces = self
            .position
            .board()
            .iter()
            .enumerate()
            .filter(|(index, piece)| {
                let Some(piece) = piece else {
                    return false;
                };
                piece.side == side
                    && piece.kind != PieceKind::King
                    && crate::Square::from_index(*index)
                        .is_some_and(|square| in_enemy_camp(side, square.rank()))
            })
            .count();
        if camp_pieces < usize::from(minimum_camp_pieces) {
            return EnteringKingDeclaration::InvalidLoss { side };
        }

        let points = declaration_points(&self.position, side);
        if points >= win_points {
            EnteringKingDeclaration::Win { side, points }
        } else if points >= no_contest_points {
            EnteringKingDeclaration::NoContest { side, points }
        } else {
            EnteringKingDeclaration::InvalidLoss { side }
        }
    }

    /// Exports a presentation-independent record.
    #[must_use]
    pub fn record(&self) -> GameRecord {
        GameRecord {
            initial_position: self.positions[0].clone(),
            moves: self.moves.clone(),
            end: self.end,
        }
    }
}

/// Replays a complete legal move sequence and classifies repetition at its final position.
///
/// Unlike [`Game::play`], this record-oriented helper does not stop when an earlier repetition
/// occurs. This is necessary for formats such as CSA that may preserve play after an agreed
/// restart and later terminate on a different repetition.
///
/// # Errors
///
/// Returns [`crate::IllegalMove`] if any move is illegal in sequence.
pub fn repetition_outcome_from_moves(
    initial_position: &Position,
    moves: &[Move],
) -> Result<Option<RepetitionOutcome>, crate::IllegalMove> {
    let mut position = initial_position.clone();
    let mut positions = Vec::with_capacity(moves.len().saturating_add(1));
    let mut gave_check = Vec::with_capacity(moves.len());
    positions.push(position.clone());
    for &movement in moves {
        position.make_move(movement)?;
        gave_check.push(position.is_in_check(position.side_to_move()));
        positions.push(position.clone());
    }
    Ok(repetition_outcome_from_history(&positions, &gave_check))
}

pub(crate) fn repetition_outcome_from_history(
    positions: &[Position],
    gave_check: &[bool],
) -> Option<RepetitionOutcome> {
    if positions.len() != gave_check.len().checked_add(1)? {
        return None;
    }
    let current = positions.last()?;
    let occurrences: Vec<usize> = positions
        .iter()
        .enumerate()
        .filter_map(|(index, position)| current.same_state(position).then_some(index))
        .collect();
    if occurrences.len() < 4 {
        return None;
    }

    let start = occurrences[occurrences.len() - 4];
    let finish = occurrences[occurrences.len() - 1];
    for side in [Side::Black, Side::White] {
        let mut has_move = false;
        let all_checks = (start..finish)
            .filter(|move_index| positions[*move_index].side_to_move() == side)
            .all(|move_index| {
                has_move = true;
                gave_check[move_index]
            });
        if has_move && all_checks {
            return Some(RepetitionOutcome::PerpetualCheckLoss(side));
        }
    }
    Some(RepetitionOutcome::NoContest)
}

fn no_legal_move_end(position: &Position) -> Option<GameEnd> {
    if !position.legal_moves().is_empty() {
        return None;
    }
    let loser = position.side_to_move();
    Some(if position.is_in_check(loser) {
        GameEnd::Checkmate {
            winner: loser.opposite(),
        }
    } else {
        // Shogi has no pass or chess-style stalemate draw. Keep this separate from checkmate.
        // https://www.computer-shogi.org/wcsc36/rule.pdf, article 27, paragraph 1, item 1.
        GameEnd::NoLegalMoves { loser }
    })
}

fn in_enemy_camp(side: Side, rank: u8) -> bool {
    match side {
        Side::Black => rank <= 3,
        Side::White => rank >= 7,
    }
}

fn position_has_entering_king(position: &Position) -> bool {
    [Side::Black, Side::White].into_iter().any(|side| {
        position
            .king_square(side)
            .is_some_and(|king| in_enemy_camp(side, king.rank()))
    })
}

fn declaration_points(position: &Position, side: Side) -> u8 {
    let board_points = position
        .board()
        .iter()
        .enumerate()
        .filter_map(|(index, piece)| {
            let piece = (*piece)?;
            let square = crate::Square::from_index(index)?;
            (piece.side == side
                && piece.kind != PieceKind::King
                && in_enemy_camp(side, square.rank()))
            .then_some(u16::from(piece_points(piece.kind)))
        })
        .sum::<u16>();
    let hand_points = HandPiece::ALL
        .iter()
        .map(|piece| {
            u16::from(position.hand(side).count(*piece))
                * u16::from(piece_points(piece.piece_kind()))
        })
        .sum::<u16>();
    u8::try_from(board_points + hand_points).unwrap_or(u8::MAX)
}

fn total_points(position: &Position, side: Side) -> u8 {
    let board_points = position
        .board()
        .iter()
        .flatten()
        .filter(|piece| piece.side == side)
        .map(|piece| u16::from(piece_points(piece.kind)))
        .sum::<u16>();
    let hand_points = HandPiece::ALL
        .iter()
        .map(|piece| {
            u16::from(position.hand(side).count(*piece))
                * u16::from(piece_points(piece.piece_kind()))
        })
        .sum::<u16>();
    u8::try_from(board_points + hand_points).unwrap_or(u8::MAX)
}

const fn piece_points(kind: PieceKind) -> u8 {
    match kind.unpromoted() {
        PieceKind::Bishop | PieceKind::Rook => 5,
        PieceKind::King => 0,
        _ => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::{Game, ImpasseCondition, RepetitionOutcome, repetition_outcome_from_moves};
    use crate::{Move, Side, Square, parse_sfen, parse_usi_move};

    #[test]
    fn no_legal_moves_without_check_is_a_loss_and_undo_restores_play() {
        let position = parse_sfen("4k4/3P1P3/4K4/9/9/9/9/9/9 w - 1").unwrap();
        assert!(!position.is_in_check(Side::White));
        assert!(position.legal_moves().is_empty());
        let expected = Some(super::GameEnd::NoLegalMoves { loser: Side::White });
        assert_eq!(Game::new(position).end(), expected);

        let initial = parse_sfen("4k4/3P1P3/5K3/9/9/9/9/9/9 b - 1").unwrap();
        let mut game = Game::new(initial.clone());
        assert_eq!(
            game.play(parse_usi_move("4c5c").unwrap()).unwrap(),
            expected
        );
        assert!(game.play(parse_usi_move("5a4a").unwrap()).is_err());
        assert!(game.undo());
        assert_eq!(game.end(), None);
        assert_eq!(game.position(), &initial);
    }

    #[test]
    fn ordinary_fourfold_repetition_is_detected() {
        let mut game = Game::startpos();
        let cycle = ["5i6h", "5a6b", "6h5i", "6b5a"];
        for _ in 0..3 {
            for notation in cycle {
                let mv = parse_usi_move(notation).expect("valid move notation");
                game.play(mv).expect("legal king shuffle");
            }
        }

        assert_eq!(
            game.repetition_outcome(),
            Some(RepetitionOutcome::NoContest)
        );
    }

    #[test]
    fn record_replay_continues_past_an_earlier_ordinary_repetition() {
        let position = parse_sfen("4k4/5R3/9/9/9/9/9/9/K8 b - 1").expect("repetition fixture");
        let mut moves = Vec::new();
        for _ in 0..3 {
            moves.extend(
                ["9i8h", "5a6a", "8h9i", "6a5a"]
                    .map(|notation| parse_usi_move(notation).expect("ordinary repetition move")),
            );
        }
        for _ in 0..3 {
            moves.extend(
                ["4b5b", "5a4a", "5b4b", "4a5a"]
                    .map(|notation| parse_usi_move(notation).expect("perpetual-check move")),
            );
        }

        assert_eq!(
            repetition_outcome_from_moves(&position, &moves).expect("legal complete record"),
            Some(RepetitionOutcome::PerpetualCheckLoss(Side::Black))
        );
    }

    #[test]
    fn resignation_records_side_to_move() {
        let mut game = Game::startpos();
        assert_eq!(
            game.resign().expect("active game can be resigned"),
            super::GameEnd::Resignation { loser: Side::Black },
        );
        assert_eq!(game.resign(), Err(crate::IllegalMove::GameAlreadyEnded));
    }

    #[test]
    fn five_hundred_move_rule_is_immediate_without_check() {
        let mut game =
            Game::new(parse_sfen("4K4/9/9/9/9/9/9/9/4k4 b - 1").expect("impasse fixture"));
        let placeholder = Move::Normal {
            from: Square::new(7, 7).expect("square"),
            to: Square::new(7, 6).expect("square"),
            promote: false,
        };
        game.moves = vec![placeholder; 500];
        game.gave_check = vec![false; 500];
        game.positions = vec![game.position.clone(); 501];

        assert!(game.five_hundred_move_no_contest(ImpasseCondition::NoMatingProspect));
    }

    #[test]
    fn five_hundred_move_check_waits_for_the_checker_to_break() {
        let mut game =
            Game::new(parse_sfen("4K4/9/9/9/9/9/9/9/4k4 b - 1").expect("impasse fixture"));
        let placeholder = Move::Normal {
            from: Square::new(7, 7).expect("square"),
            to: Square::new(7, 6).expect("square"),
            promote: false,
        };
        game.moves = vec![placeholder; 500];
        game.gave_check = vec![false; 500];
        game.gave_check[499] = true;
        game.positions = vec![game.position.clone(); 501];
        assert!(!game.five_hundred_move_no_contest(ImpasseCondition::NoMatingProspect));

        game.moves.push(placeholder);
        game.gave_check.push(false);
        assert!(game.five_hundred_move_no_contest(ImpasseCondition::NoMatingProspect));
    }

    #[test]
    fn five_hundred_move_rule_uses_the_limit_position_for_entering_king() {
        let entering = parse_sfen("4K4/9/9/9/9/9/9/9/4k4 b - 1").expect("entering-king fixture");
        let outside = crate::Position::startpos();
        let placeholder = Move::Normal {
            from: Square::new(7, 7).expect("square"),
            to: Square::new(7, 6).expect("square"),
            promote: false,
        };

        let mut entered_at_limit = Game::new(outside.clone());
        entered_at_limit.moves = vec![placeholder; 500];
        entered_at_limit.gave_check = vec![false; 500];
        entered_at_limit.positions = vec![outside.clone(); 501];
        entered_at_limit.positions[500] = entering.clone();
        assert!(entered_at_limit.five_hundred_move_no_contest(ImpasseCondition::NoMatingProspect));

        let mut entered_after_limit = Game::new(entering);
        entered_after_limit.moves = vec![placeholder; 500];
        entered_after_limit.gave_check = vec![false; 500];
        entered_after_limit.positions = vec![outside; 501];
        assert!(
            !entered_after_limit.five_hundred_move_no_contest(ImpasseCondition::NoMatingProspect)
        );
    }
}
