//! Board state, move generation, and legality checking.

use std::{error::Error, fmt};

use crate::{BOARD_SQUARES, Hand, HandPiece, Move, Piece, PieceKind, Side, Square};

const MATERIAL_LIMITS: [(PieceKind, u16); 8] = [
    (PieceKind::Pawn, 18),
    (PieceKind::Lance, 4),
    (PieceKind::Knight, 4),
    (PieceKind::Silver, 4),
    (PieceKind::Gold, 4),
    (PieceKind::Bishop, 2),
    (PieceKind::Rook, 2),
    (PieceKind::King, 2),
];

const STEP_KING: [(i8, i8); 8] = [
    (-1, -1),
    (0, -1),
    (1, -1),
    (-1, 0),
    (1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
];
const STEP_ORTHOGONAL: [(i8, i8); 4] = [(0, -1), (-1, 0), (1, 0), (0, 1)];
const STEP_DIAGONAL: [(i8, i8); 4] = [(-1, -1), (1, -1), (-1, 1), (1, 1)];

/// A structurally invalid position.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PositionError {
    /// Move numbers are one-based.
    InvalidMoveNumber,
    /// A side has no king on the board.
    MissingKing(Side),
    /// A side has more than one king on the board.
    MultipleKings(Side),
    /// More pieces of a base kind exist than a standard shogi set contains.
    TooManyPieces {
        /// The base piece kind whose inventory is invalid.
        kind: PieceKind,
        /// The observed count on the board and in both hands.
        count: u16,
        /// The maximum count in a standard shogi set.
        maximum: u16,
    },
    /// An unpromoted piece occupies a rank from which it can never move.
    PieceOnDeadSquare {
        /// Location of the immobile piece.
        square: Square,
        /// The offending piece.
        piece: Piece,
    },
    /// A side has two unpromoted pawns on one file.
    DuplicatePawn {
        /// Side owning the pawns.
        side: Side,
        /// Shogi file number in `1..=9`.
        file: u8,
    },
}

impl fmt::Display for PositionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::InvalidMoveNumber => formatter.write_str("move number must be at least one"),
            Self::MissingKing(side) => write!(formatter, "{side:?} king is missing"),
            Self::MultipleKings(side) => write!(formatter, "{side:?} has multiple kings"),
            Self::TooManyPieces {
                kind,
                count,
                maximum,
            } => write!(
                formatter,
                "too many {kind:?} pieces: found {count}, maximum is {maximum}"
            ),
            Self::PieceOnDeadSquare { square, piece } => {
                write!(formatter, "{piece:?} cannot move from {square}")
            }
            Self::DuplicatePawn { side, file } => {
                write!(
                    formatter,
                    "{side:?} has multiple unpromoted pawns on file {file}"
                )
            }
        }
    }
}

impl Error for PositionError {}

/// A rejected attempt to advance a game or position.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IllegalMove {
    /// The move is not legal in the current position.
    MoveNotLegal(Move),
    /// The enclosing game has already reached a terminal result.
    GameAlreadyEnded,
    /// Advancing the serialized one-based move counter would exceed `u32`.
    MoveNumberOverflow,
}

impl IllegalMove {
    /// Returns the rejected move.
    #[must_use]
    pub const fn attempted(self) -> Option<Move> {
        match self {
            Self::MoveNotLegal(movement) => Some(movement),
            Self::GameAlreadyEnded | Self::MoveNumberOverflow => None,
        }
    }
}

impl fmt::Display for IllegalMove {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MoveNotLegal(movement) => {
                write!(formatter, "move {movement:?} is not legal in this position")
            }
            Self::GameAlreadyEnded => formatter.write_str("the game has already ended"),
            Self::MoveNumberOverflow => {
                formatter.write_str("the position move number cannot be incremented")
            }
        }
    }
}

impl Error for IllegalMove {}

/// Complete state needed to reverse one move.
///
/// Values are created by [`Position::make_move`] and consumed by
/// [`Position::unmake_move`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Undo {
    movement: Move,
    moved_piece: Piece,
    captured_piece: Option<Piece>,
    side_to_move: Side,
    move_number: u32,
    zobrist: u64,
}

/// A validated shogi position.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Position {
    board: [Option<Piece>; BOARD_SQUARES],
    hands: [Hand; 2],
    side_to_move: Side,
    move_number: u32,
    zobrist: u64,
}

impl Position {
    /// Returns the standard initial shogi position.
    #[must_use]
    pub fn startpos() -> Self {
        let mut board = [None; BOARD_SQUARES];
        let back_rank = [
            PieceKind::Lance,
            PieceKind::Knight,
            PieceKind::Silver,
            PieceKind::Gold,
            PieceKind::King,
            PieceKind::Gold,
            PieceKind::Silver,
            PieceKind::Knight,
            PieceKind::Lance,
        ];

        for (column, kind) in back_rank.into_iter().enumerate() {
            board[column] = Some(Piece::new(Side::White, kind));
            board[72 + column] = Some(Piece::new(Side::Black, kind));
        }
        board[10] = Some(Piece::new(Side::White, PieceKind::Rook));
        board[16] = Some(Piece::new(Side::White, PieceKind::Bishop));
        board[64] = Some(Piece::new(Side::Black, PieceKind::Bishop));
        board[70] = Some(Piece::new(Side::Black, PieceKind::Rook));
        for column in 0..9 {
            board[18 + column] = Some(Piece::new(Side::White, PieceKind::Pawn));
            board[54 + column] = Some(Piece::new(Side::Black, PieceKind::Pawn));
        }

        let mut position = Self {
            board,
            hands: [Hand::default(); 2],
            side_to_move: Side::Black,
            move_number: 1,
            zobrist: 0,
        };
        position.zobrist = position.recompute_zobrist();
        position
    }

    /// Builds and validates a position from its constituent state.
    ///
    /// The supplied board must contain exactly one king for each side. Piece inventory,
    /// immobile unpromoted pieces, double pawns, and the one-based move number are validated.
    ///
    /// # Errors
    ///
    /// Returns [`PositionError`] when the supplied state violates a structural shogi rule.
    pub fn from_parts(
        board: [Option<Piece>; BOARD_SQUARES],
        hands: [Hand; 2],
        side_to_move: Side,
        move_number: u32,
    ) -> Result<Self, PositionError> {
        let mut position = Self {
            board,
            hands,
            side_to_move,
            move_number,
            zobrist: 0,
        };
        position.validate()?;
        position.zobrist = position.recompute_zobrist();
        Ok(position)
    }

    /// Returns the board in SFEN scan order.
    #[must_use]
    pub const fn board(&self) -> &[Option<Piece>; BOARD_SQUARES] {
        &self.board
    }

    /// Returns both hands, indexed by [`Side::index`].
    #[must_use]
    pub const fn hands(&self) -> &[Hand; 2] {
        &self.hands
    }

    /// Returns one side's hand.
    #[must_use]
    pub const fn hand(&self, side: Side) -> &Hand {
        &self.hands[side.index()]
    }

    /// Returns the side whose turn it is.
    #[must_use]
    pub const fn side_to_move(&self) -> Side {
        self.side_to_move
    }

    /// Returns the one-based move number.
    #[must_use]
    pub const fn move_number(&self) -> u32 {
        self.move_number
    }

    /// Returns the deterministic hash of board, hands, and side to move.
    #[must_use]
    pub const fn zobrist_hash(&self) -> u64 {
        self.zobrist
    }

    /// Returns the piece on a square, if any.
    #[must_use]
    pub const fn piece_at(&self, square: Square) -> Option<Piece> {
        self.board[square.index()]
    }

    /// Returns whether this position has the same board, hands, and side to move.
    ///
    /// Move numbers are deliberately ignored, matching the state represented by the
    /// Zobrist hash.
    #[must_use]
    pub fn same_state(&self, other: &Self) -> bool {
        self.board == other.board
            && self.hands == other.hands
            && self.side_to_move == other.side_to_move
    }

    /// Generates every legal move in deterministic order.
    #[must_use]
    pub fn legal_moves(&self) -> Vec<Move> {
        self.pseudo_legal_moves()
            .into_iter()
            .filter(|movement| self.is_legal_candidate(*movement, true))
            .collect()
    }

    /// Returns whether a move is legal in this position.
    #[must_use]
    pub fn is_legal_move(&self, movement: Move) -> bool {
        self.pseudo_legal_moves().contains(&movement) && self.is_legal_candidate(movement, true)
    }

    /// Applies a legal move and returns the state needed to undo it.
    ///
    /// # Errors
    ///
    /// Returns [`IllegalMove`] if `movement` is not legal in the current position.
    pub fn make_move(&mut self, movement: Move) -> Result<Undo, IllegalMove> {
        if self.move_number == u32::MAX {
            return Err(IllegalMove::MoveNumberOverflow);
        }
        if !self.is_legal_move(movement) {
            return Err(IllegalMove::MoveNotLegal(movement));
        }

        let undo = self.apply_unchecked(movement, true);
        debug_assert_eq!(self.zobrist, self.recompute_zobrist());
        Ok(undo)
    }

    /// Applies a move returned by [`Self::legal_moves`] without regenerating the legal list.
    ///
    /// Search and traversal code must only pass a move generated from this exact position.
    /// Keeping this crate-private preserves the checked public API boundary.
    pub(crate) fn make_generated_move(&mut self, movement: Move) -> Undo {
        // Search does not serialize its temporary nodes, so their move number is deliberately
        // stable. This also lets a structurally valid maximum-number SFEN be analyzed safely.
        let undo = self.apply_unchecked(movement, false);
        debug_assert_eq!(self.zobrist, self.recompute_zobrist());
        undo
    }

    /// Restores the exact state captured before a successful move.
    ///
    /// # Panics
    ///
    /// May panic if the opaque token is applied to a different position than the one that
    /// produced it. Misordered tokens can also restore a logically unrelated state. Callers
    /// must therefore treat undo tokens as ordered, single-use values.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "an undo token is single-use state and is intentionally consumed"
    )]
    pub fn unmake_move(&mut self, undo: Undo) {
        let moving_side = undo.side_to_move;
        match undo.movement {
            Move::Normal { from, to, .. } => {
                self.board[from.index()] = Some(undo.moved_piece);
                self.board[to.index()] = undo.captured_piece;
                if let Some(captured) = undo.captured_piece {
                    let hand_piece = HandPiece::from_piece_kind(captured.kind)
                        .expect("legal moves never capture a king");
                    self.hands[moving_side.index()]
                        .remove(hand_piece)
                        .expect("undo removes the hand piece added by the move");
                }
            }
            Move::Drop { piece, to } => {
                self.board[to.index()] = None;
                self.hands[moving_side.index()]
                    .add(piece)
                    .expect("undo restores the dropped hand piece");
            }
        }
        self.side_to_move = undo.side_to_move;
        self.move_number = undo.move_number;
        self.zobrist = undo.zobrist;
        debug_assert_eq!(self.zobrist, self.recompute_zobrist());
    }

    /// Returns whether `side`'s king is attacked.
    #[must_use]
    pub fn is_in_check(&self, side: Side) -> bool {
        self.king_square(side)
            .is_some_and(|king| self.is_square_attacked(king, side.opposite()))
    }

    /// Returns whether the side to move is checkmated.
    #[must_use]
    pub fn is_checkmate(&self) -> bool {
        self.is_in_check(self.side_to_move) && self.legal_moves().is_empty()
    }

    /// Validates all structural invariants enforced by [`Self::from_parts`].
    ///
    /// # Errors
    ///
    /// Returns [`PositionError`] for the first invalid invariant in stable validation order.
    pub fn validate(&self) -> Result<(), PositionError> {
        if self.move_number == 0 {
            return Err(PositionError::InvalidMoveNumber);
        }

        for side in [Side::Black, Side::White] {
            let kings = self
                .board
                .iter()
                .flatten()
                .filter(|piece| piece.side == side && piece.kind == PieceKind::King)
                .count();
            match kings {
                0 => return Err(PositionError::MissingKing(side)),
                1 => {}
                _ => return Err(PositionError::MultipleKings(side)),
            }
        }

        for (kind, maximum) in MATERIAL_LIMITS {
            let board_count = self
                .board
                .iter()
                .flatten()
                .filter(|piece| piece.kind.unpromoted() == kind)
                .count();
            let hand_count = HandPiece::from_piece_kind(kind).map_or(0, |hand_piece| {
                usize::from(self.hands[0].count(hand_piece))
                    + usize::from(self.hands[1].count(hand_piece))
            });
            let count = u16::try_from(board_count + hand_count).unwrap_or(u16::MAX);
            if count > maximum {
                return Err(PositionError::TooManyPieces {
                    kind,
                    count,
                    maximum,
                });
            }
        }

        for square in Square::all() {
            if let Some(piece) = self.piece_at(square)
                && is_dead_end(piece.kind, piece.side, square.rank())
            {
                return Err(PositionError::PieceOnDeadSquare { square, piece });
            }
        }

        for side in [Side::Black, Side::White] {
            for file in 1..=9 {
                let pawn_count = (1..=9)
                    .filter_map(|rank| Square::new(file, rank))
                    .filter_map(|square| self.piece_at(square))
                    .filter(|piece| piece.side == side && piece.kind == PieceKind::Pawn)
                    .count();
                if pawn_count > 1 {
                    return Err(PositionError::DuplicatePawn { side, file });
                }
            }
        }

        Ok(())
    }

    /// Recomputes the deterministic Zobrist hash from state.
    #[must_use]
    pub fn recompute_zobrist(&self) -> u64 {
        let mut hash = zobrist_key(0x10, self.side_to_move.index() as u64);

        for square in Square::all() {
            if let Some(piece) = self.piece_at(square) {
                let identity = ((piece.side.index() * PieceKind::ALL.len() + piece.kind.index())
                    * BOARD_SQUARES
                    + square.index()) as u64;
                hash ^= zobrist_key(0x20, identity);
            }
        }

        for side in [Side::Black, Side::White] {
            for hand_piece in HandPiece::ALL {
                for copy in 0..self.hands[side.index()].count(hand_piece) {
                    let identity = ((side.index() * HandPiece::ALL.len() + hand_piece.index())
                        * 256
                        + usize::from(copy)) as u64;
                    hash ^= zobrist_key(0x30, identity);
                }
            }
        }
        hash
    }

    fn pseudo_legal_moves(&self) -> Vec<Move> {
        let mut moves = Vec::new();
        for from in Square::all() {
            let Some(piece) = self.piece_at(from) else {
                continue;
            };
            if piece.side != self.side_to_move {
                continue;
            }
            self.generate_piece_moves(from, piece, &mut moves);
        }
        self.generate_drops(&mut moves);
        moves
    }

    fn generate_piece_moves(&self, from: Square, piece: Piece, moves: &mut Vec<Move>) {
        let forward = piece.side.forward();
        match piece.kind {
            PieceKind::Pawn => self.add_steps(from, piece, &[(0, forward)], moves),
            PieceKind::Lance => self.add_slides(from, piece, &[(0, forward)], moves),
            PieceKind::Knight => {
                self.add_steps(from, piece, &[(-1, 2 * forward), (1, 2 * forward)], moves);
            }
            PieceKind::Silver => self.add_steps(
                from,
                piece,
                &[
                    (-1, forward),
                    (0, forward),
                    (1, forward),
                    (-1, -forward),
                    (1, -forward),
                ],
                moves,
            ),
            PieceKind::Gold
            | PieceKind::PromotedPawn
            | PieceKind::PromotedLance
            | PieceKind::PromotedKnight
            | PieceKind::PromotedSilver => self.add_steps(
                from,
                piece,
                &[
                    (-1, forward),
                    (0, forward),
                    (1, forward),
                    (-1, 0),
                    (1, 0),
                    (0, -forward),
                ],
                moves,
            ),
            PieceKind::Bishop => self.add_slides(from, piece, &STEP_DIAGONAL, moves),
            PieceKind::Rook => self.add_slides(from, piece, &STEP_ORTHOGONAL, moves),
            PieceKind::King => self.add_steps(from, piece, &STEP_KING, moves),
            PieceKind::Horse => {
                self.add_slides(from, piece, &STEP_DIAGONAL, moves);
                self.add_steps(from, piece, &STEP_ORTHOGONAL, moves);
            }
            PieceKind::Dragon => {
                self.add_slides(from, piece, &STEP_ORTHOGONAL, moves);
                self.add_steps(from, piece, &STEP_DIAGONAL, moves);
            }
        }
    }

    fn add_steps(
        &self,
        from: Square,
        piece: Piece,
        directions: &[(i8, i8)],
        moves: &mut Vec<Move>,
    ) {
        for &(file_delta, rank_delta) in directions {
            let Some(to) = from.offset(file_delta, rank_delta) else {
                continue;
            };
            if self.destination_is_available(to, piece.side) {
                Self::add_move_with_promotion(from, to, piece, moves);
            }
        }
    }

    fn add_slides(
        &self,
        from: Square,
        piece: Piece,
        directions: &[(i8, i8)],
        moves: &mut Vec<Move>,
    ) {
        for &(file_delta, rank_delta) in directions {
            let mut cursor = from;
            while let Some(to) = cursor.offset(file_delta, rank_delta) {
                cursor = to;
                match self.piece_at(to) {
                    None => Self::add_move_with_promotion(from, to, piece, moves),
                    Some(target) if target.side != piece.side && target.kind != PieceKind::King => {
                        Self::add_move_with_promotion(from, to, piece, moves);
                        break;
                    }
                    Some(_) => break,
                }
            }
        }
    }

    fn destination_is_available(&self, square: Square, side: Side) -> bool {
        self.piece_at(square)
            .is_none_or(|target| target.side != side && target.kind != PieceKind::King)
    }

    fn add_move_with_promotion(from: Square, to: Square, piece: Piece, moves: &mut Vec<Move>) {
        let can_promote = piece.kind.is_promotable()
            && (in_promotion_zone(piece.side, from.rank())
                || in_promotion_zone(piece.side, to.rank()));
        let must_promote = is_dead_end(piece.kind, piece.side, to.rank());

        if !must_promote {
            moves.push(Move::Normal {
                from,
                to,
                promote: false,
            });
        }
        if can_promote {
            moves.push(Move::Normal {
                from,
                to,
                promote: true,
            });
        }
    }

    fn generate_drops(&self, moves: &mut Vec<Move>) {
        let side = self.side_to_move;
        for hand_piece in HandPiece::ALL {
            if self.hands[side.index()].count(hand_piece) == 0 {
                continue;
            }
            for to in Square::all() {
                if self.piece_at(to).is_some()
                    || is_dead_end(hand_piece.piece_kind(), side, to.rank())
                    || (hand_piece == HandPiece::Pawn && self.has_unpromoted_pawn(side, to.file()))
                {
                    continue;
                }
                moves.push(Move::Drop {
                    piece: hand_piece,
                    to,
                });
            }
        }
    }

    fn has_unpromoted_pawn(&self, side: Side, file: u8) -> bool {
        (1..=9)
            .filter_map(|rank| Square::new(file, rank))
            .filter_map(|square| self.piece_at(square))
            .any(|piece| piece.side == side && piece.kind == PieceKind::Pawn)
    }

    fn is_legal_candidate(&self, movement: Move, enforce_pawn_drop_mate: bool) -> bool {
        let moving_side = self.side_to_move;
        let mut next = self.clone();
        let _ = next.apply_unchecked(movement, false);
        if next.is_in_check(moving_side) {
            return false;
        }

        if enforce_pawn_drop_mate
            && matches!(
                movement,
                Move::Drop {
                    piece: HandPiece::Pawn,
                    ..
                }
            )
            && next.is_in_check(next.side_to_move)
            && next.has_no_legal_normal_reply()
        {
            return false;
        }
        true
    }

    fn has_no_legal_normal_reply(&self) -> bool {
        !self.pseudo_legal_moves().into_iter().any(|movement| {
            matches!(movement, Move::Normal { .. }) && self.is_legal_candidate(movement, false)
        })
    }

    fn apply_unchecked(&mut self, movement: Move, advance_move_number: bool) -> Undo {
        let moving_side = self.side_to_move;
        let previous_move_number = self.move_number;
        let previous_zobrist = self.zobrist;
        let moved_piece;
        let captured_piece;

        match movement {
            Move::Normal { from, to, promote } => {
                let mut piece = self.board[from.index()]
                    .take()
                    .expect("pseudo-legal normal move has an origin piece");
                moved_piece = piece;
                captured_piece = self.board[to.index()].take();

                self.zobrist ^= board_zobrist_key(from, piece);
                if let Some(captured) = captured_piece {
                    self.zobrist ^= board_zobrist_key(to, captured);
                    let hand_piece = HandPiece::from_piece_kind(captured.kind)
                        .expect("legal move generation never captures a king");
                    let old_count = self.hands[moving_side.index()].count(hand_piece);
                    self.hands[moving_side.index()]
                        .add(hand_piece)
                        .expect("validated material inventory cannot overflow a hand");
                    self.zobrist ^= hand_zobrist_key(moving_side, hand_piece, old_count);
                }
                if promote {
                    piece.kind = piece
                        .kind
                        .promoted()
                        .expect("pseudo-legal promotion uses a promotable piece");
                }
                self.board[to.index()] = Some(piece);
                self.zobrist ^= board_zobrist_key(to, piece);
            }
            Move::Drop { piece, to } => {
                moved_piece = Piece::new(moving_side, piece.piece_kind());
                captured_piece = None;
                let old_count = self.hands[moving_side.index()].count(piece);
                self.hands[moving_side.index()]
                    .remove(piece)
                    .expect("pseudo-legal drop has a piece in hand");
                self.zobrist ^= hand_zobrist_key(moving_side, piece, old_count - 1);
                self.board[to.index()] = Some(moved_piece);
                self.zobrist ^= board_zobrist_key(to, moved_piece);
            }
        }
        self.zobrist ^= side_zobrist_key(moving_side);
        self.side_to_move = moving_side.opposite();
        self.zobrist ^= side_zobrist_key(self.side_to_move);
        if advance_move_number {
            self.move_number = self
                .move_number
                .checked_add(1)
                .expect("public move application checks move-number capacity");
        }

        Undo {
            movement,
            moved_piece,
            captured_piece,
            side_to_move: moving_side,
            move_number: previous_move_number,
            zobrist: previous_zobrist,
        }
    }

    pub(crate) fn king_square(&self, side: Side) -> Option<Square> {
        Square::all().find(|&square| {
            self.piece_at(square)
                .is_some_and(|piece| piece.side == side && piece.kind == PieceKind::King)
        })
    }

    #[cfg(feature = "handcrafted")]
    pub(crate) fn pseudo_mobility(&self, side: Side) -> usize {
        let mut position = self.clone();
        position.side_to_move = side;
        position.pseudo_legal_moves().len()
    }

    pub(crate) fn square_is_attacked(&self, square: Square, attacker: Side) -> bool {
        self.is_square_attacked(square, attacker)
    }

    fn is_square_attacked(&self, target: Square, attacker: Side) -> bool {
        Square::all().any(|from| {
            self.piece_at(from).is_some_and(|piece| {
                piece.side == attacker && self.piece_attacks(from, piece, target)
            })
        })
    }

    pub(crate) fn piece_attacks(&self, from: Square, piece: Piece, target: Square) -> bool {
        if from == target {
            return false;
        }
        let file_delta = i16::from(target.file()) - i16::from(from.file());
        let rank_delta = i16::from(target.rank()) - i16::from(from.rank());
        let forward = i16::from(piece.side.forward());

        match piece.kind {
            PieceKind::Pawn => file_delta == 0 && rank_delta == forward,
            PieceKind::Lance => {
                file_delta == 0
                    && rank_delta.signum() == forward
                    && self.path_is_clear(from, target)
            }
            PieceKind::Knight => file_delta.abs() == 1 && rank_delta == 2 * forward,
            PieceKind::Silver => {
                (rank_delta == forward && file_delta.abs() <= 1)
                    || (rank_delta == -forward && file_delta.abs() == 1)
            }
            PieceKind::Gold
            | PieceKind::PromotedPawn
            | PieceKind::PromotedLance
            | PieceKind::PromotedKnight
            | PieceKind::PromotedSilver => {
                (rank_delta == forward && file_delta.abs() <= 1)
                    || (rank_delta == 0 && file_delta.abs() == 1)
                    || (rank_delta == -forward && file_delta == 0)
            }
            PieceKind::Bishop => {
                file_delta.abs() == rank_delta.abs() && self.path_is_clear(from, target)
            }
            PieceKind::Rook => {
                (file_delta == 0 || rank_delta == 0) && self.path_is_clear(from, target)
            }
            PieceKind::King => file_delta.abs() <= 1 && rank_delta.abs() <= 1,
            PieceKind::Horse => {
                (file_delta.abs() == rank_delta.abs() && self.path_is_clear(from, target))
                    || ((file_delta == 0 && rank_delta.abs() == 1)
                        || (rank_delta == 0 && file_delta.abs() == 1))
            }
            PieceKind::Dragon => {
                ((file_delta == 0 || rank_delta == 0) && self.path_is_clear(from, target))
                    || (file_delta.abs() == 1 && rank_delta.abs() == 1)
            }
        }
    }

    fn path_is_clear(&self, from: Square, target: Square) -> bool {
        let file_delta = (i16::from(target.file()) - i16::from(from.file())).signum();
        let rank_delta = (i16::from(target.rank()) - i16::from(from.rank())).signum();
        let file_step = i8::try_from(file_delta).expect("signum is in i8 range");
        let rank_step = i8::try_from(rank_delta).expect("signum is in i8 range");
        let mut cursor = from;
        while let Some(square) = cursor.offset(file_step, rank_step) {
            if square == target {
                return true;
            }
            if self.piece_at(square).is_some() {
                return false;
            }
            cursor = square;
        }
        false
    }
}

fn in_promotion_zone(side: Side, rank: u8) -> bool {
    match side {
        Side::Black => rank <= 3,
        Side::White => rank >= 7,
    }
}

fn is_dead_end(kind: PieceKind, side: Side, rank: u8) -> bool {
    let distance_to_last = match side {
        Side::Black => rank - 1,
        Side::White => 9 - rank,
    };
    match kind {
        PieceKind::Pawn | PieceKind::Lance => distance_to_last == 0,
        PieceKind::Knight => distance_to_last <= 1,
        _ => false,
    }
}

fn zobrist_key(domain: u64, identity: u64) -> u64 {
    let mut value = 0x9e37_79b9_7f4a_7c15_u64
        ^ domain.wrapping_mul(0xd6e8_feb8_6659_fd93)
        ^ identity.wrapping_mul(0xa076_1d64_78bd_642f);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn side_zobrist_key(side: Side) -> u64 {
    zobrist_key(0x10, side.index() as u64)
}

fn board_zobrist_key(square: Square, piece: Piece) -> u64 {
    let identity = ((piece.side.index() * PieceKind::ALL.len() + piece.kind.index())
        * BOARD_SQUARES
        + square.index()) as u64;
    zobrist_key(0x20, identity)
}

fn hand_zobrist_key(side: Side, piece: HandPiece, copy: u8) -> u64 {
    let identity =
        ((side.index() * HandPiece::ALL.len() + piece.index()) * 256 + usize::from(copy)) as u64;
    zobrist_key(0x30, identity)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square(file: u8, rank: u8) -> Square {
        Square::new(file, rank).expect("test square")
    }

    fn board_with_kings(black: Square, white: Square) -> [Option<Piece>; BOARD_SQUARES] {
        let mut board = [None; BOARD_SQUARES];
        board[black.index()] = Some(Piece::new(Side::Black, PieceKind::King));
        board[white.index()] = Some(Piece::new(Side::White, PieceKind::King));
        board
    }

    #[test]
    fn start_position_has_expected_legal_moves_and_hash() {
        let position = Position::startpos();

        assert_eq!(position.legal_moves().len(), 30);
        assert_eq!(position.zobrist_hash(), position.recompute_zobrist());
        assert!(position.validate().is_ok());
    }

    #[test]
    fn no_piece_attacks_its_own_origin_square() {
        let position = Position::startpos();
        let origin = square(5, 5);
        for kind in PieceKind::ALL {
            assert!(!position.piece_attacks(origin, Piece::new(Side::Black, kind), origin));
        }
    }

    #[test]
    fn make_and_unmake_are_exactly_symmetric() {
        let mut position = Position::startpos();
        let original = position.clone();
        let movement = position.legal_moves()[0];

        let undo = position
            .make_move(movement)
            .expect("generated move is legal");
        assert_eq!(position.zobrist_hash(), position.recompute_zobrist());
        position.unmake_move(undo);

        assert_eq!(position, original);
    }

    #[test]
    fn illegal_move_does_not_modify_position() {
        let mut position = Position::startpos();
        let original = position.clone();
        let from = square(5, 5);
        let to = square(5, 4);

        assert!(
            position
                .make_move(Move::Normal {
                    from,
                    to,
                    promote: false,
                })
                .is_err()
        );
        assert_eq!(position, original);
    }

    #[test]
    fn pinned_piece_cannot_expose_king() {
        let mut board = board_with_kings(square(5, 9), square(1, 1));
        board[square(5, 1).index()] = Some(Piece::new(Side::White, PieceKind::Rook));
        board[square(5, 8).index()] = Some(Piece::new(Side::Black, PieceKind::Gold));
        let position =
            Position::from_parts(board, [Hand::default(); 2], Side::Black, 1).expect("position");

        assert!(!position.is_legal_move(Move::Normal {
            from: square(5, 8),
            to: square(4, 7),
            promote: false,
        }));
        assert!(position.is_legal_move(Move::Normal {
            from: square(5, 8),
            to: square(5, 7),
            promote: false,
        }));
    }

    #[test]
    fn pawn_drop_mate_is_not_legal() {
        let mut board = board_with_kings(square(9, 9), square(5, 1));
        board[square(6, 1).index()] = Some(Piece::new(Side::White, PieceKind::Rook));
        board[square(4, 1).index()] = Some(Piece::new(Side::White, PieceKind::Rook));
        board[square(6, 2).index()] = Some(Piece::new(Side::White, PieceKind::Pawn));
        board[square(4, 2).index()] = Some(Piece::new(Side::White, PieceKind::Pawn));
        board[square(5, 3).index()] = Some(Piece::new(Side::Black, PieceKind::Gold));
        let mut hands = [Hand::default(); 2];
        hands[Side::Black.index()].set(HandPiece::Pawn, 1);
        let position = Position::from_parts(board, hands, Side::Black, 1).expect("position");
        let pawn_drop = Move::Drop {
            piece: HandPiece::Pawn,
            to: square(5, 2),
        };

        assert!(!position.is_legal_move(pawn_drop));
        assert!(!position.legal_moves().contains(&pawn_drop));
    }

    #[test]
    fn validation_rejects_nifu_and_dead_unpromoted_piece() {
        let mut nifu_board = board_with_kings(square(5, 9), square(5, 1));
        nifu_board[square(4, 4).index()] = Some(Piece::new(Side::Black, PieceKind::Pawn));
        nifu_board[square(4, 5).index()] = Some(Piece::new(Side::Black, PieceKind::Pawn));
        assert!(matches!(
            Position::from_parts(nifu_board, [Hand::default(); 2], Side::Black, 1),
            Err(PositionError::DuplicatePawn {
                side: Side::Black,
                file: 4
            })
        ));

        let mut dead_board = board_with_kings(square(5, 9), square(5, 1));
        dead_board[square(1, 1).index()] = Some(Piece::new(Side::Black, PieceKind::Knight));
        assert!(matches!(
            Position::from_parts(dead_board, [Hand::default(); 2], Side::Black, 1),
            Err(PositionError::PieceOnDeadSquare { .. })
        ));
    }
}
