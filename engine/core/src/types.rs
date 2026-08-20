//! Compact value types used throughout the rules engine.

use std::fmt;

/// Number of squares on a standard shogi board.
pub const BOARD_SQUARES: usize = 81;

/// Number of piece kinds that may be held in hand.
pub const HAND_KIND_COUNT: usize = 7;

/// A player and piece colour.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Side {
    /// Sente, represented by uppercase pieces in SFEN and `+` in CSA.
    Black,
    /// Gote, represented by lowercase pieces in SFEN and `-` in CSA.
    White,
}

impl Side {
    /// Returns the other side.
    #[must_use]
    pub const fn opposite(self) -> Self {
        match self {
            Self::Black => Self::White,
            Self::White => Self::Black,
        }
    }

    /// Rank direction toward the opponent from this side's perspective.
    #[must_use]
    pub const fn forward(self) -> i8 {
        match self {
            Self::Black => -1,
            Self::White => 1,
        }
    }

    /// Index suitable for fixed two-side arrays.
    #[must_use]
    pub const fn index(self) -> usize {
        match self {
            Self::Black => 0,
            Self::White => 1,
        }
    }
}

/// One square, stored in SFEN scan order: rank `a` to `i`, file `9` to `1`.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Square(u8);

impl Square {
    /// Creates a square from shogi file and rank numbers in `1..=9`.
    #[must_use]
    pub fn new(file: u8, rank: u8) -> Option<Self> {
        if !(1..=9).contains(&file) || !(1..=9).contains(&rank) {
            return None;
        }
        let row = rank - 1;
        let column = 9 - file;
        Some(Self(row * 9 + column))
    }

    /// Creates a square from its compact index.
    #[must_use]
    pub fn from_index(index: usize) -> Option<Self> {
        if index < BOARD_SQUARES {
            u8::try_from(index).ok().map(Self)
        } else {
            None
        }
    }

    /// Returns the compact board index.
    #[must_use]
    pub const fn index(self) -> usize {
        self.0 as usize
    }

    /// Returns the shogi file number, `1..=9`.
    #[must_use]
    pub const fn file(self) -> u8 {
        9 - self.0 % 9
    }

    /// Returns the rank number where `1` is USI rank `a`.
    #[must_use]
    pub const fn rank(self) -> u8 {
        self.0 / 9 + 1
    }

    /// Returns a square offset by file and rank, or `None` outside the board.
    #[must_use]
    pub fn offset(self, file_delta: i8, rank_delta: i8) -> Option<Self> {
        let file = i16::from(self.file()) + i16::from(file_delta);
        let rank = i16::from(self.rank()) + i16::from(rank_delta);
        if !(1..=9).contains(&file) || !(1..=9).contains(&rank) {
            return None;
        }
        let file = u8::try_from(file).ok()?;
        let rank = u8::try_from(rank).ok()?;
        Self::new(file, rank)
    }

    /// Returns the USI rank character.
    #[must_use]
    pub fn usi_rank(self) -> char {
        char::from(b'a' + self.rank() - 1)
    }

    /// Iterates over all board squares in SFEN scan order.
    pub fn all() -> impl Iterator<Item = Self> {
        (0_u8..81).map(Self)
    }
}

impl fmt::Display for Square {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}{}", self.file(), self.usi_rank())
    }
}

/// All board piece states, including promoted forms.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum PieceKind {
    Pawn,
    Lance,
    Knight,
    Silver,
    Gold,
    Bishop,
    Rook,
    King,
    PromotedPawn,
    PromotedLance,
    PromotedKnight,
    PromotedSilver,
    Horse,
    Dragon,
}

impl PieceKind {
    /// All board piece states in a stable hashing order.
    pub const ALL: [Self; 14] = [
        Self::Pawn,
        Self::Lance,
        Self::Knight,
        Self::Silver,
        Self::Gold,
        Self::Bishop,
        Self::Rook,
        Self::King,
        Self::PromotedPawn,
        Self::PromotedLance,
        Self::PromotedKnight,
        Self::PromotedSilver,
        Self::Horse,
        Self::Dragon,
    ];

    /// Stable index for fixed tables.
    #[must_use]
    pub const fn index(self) -> usize {
        match self {
            Self::Pawn => 0,
            Self::Lance => 1,
            Self::Knight => 2,
            Self::Silver => 3,
            Self::Gold => 4,
            Self::Bishop => 5,
            Self::Rook => 6,
            Self::King => 7,
            Self::PromotedPawn => 8,
            Self::PromotedLance => 9,
            Self::PromotedKnight => 10,
            Self::PromotedSilver => 11,
            Self::Horse => 12,
            Self::Dragon => 13,
        }
    }

    /// Returns the promoted form, if this kind can promote.
    #[must_use]
    pub const fn promoted(self) -> Option<Self> {
        match self {
            Self::Pawn => Some(Self::PromotedPawn),
            Self::Lance => Some(Self::PromotedLance),
            Self::Knight => Some(Self::PromotedKnight),
            Self::Silver => Some(Self::PromotedSilver),
            Self::Bishop => Some(Self::Horse),
            Self::Rook => Some(Self::Dragon),
            _ => None,
        }
    }

    /// Returns the unpromoted form.
    #[must_use]
    pub const fn unpromoted(self) -> Self {
        match self {
            Self::PromotedPawn => Self::Pawn,
            Self::PromotedLance => Self::Lance,
            Self::PromotedKnight => Self::Knight,
            Self::PromotedSilver => Self::Silver,
            Self::Horse => Self::Bishop,
            Self::Dragon => Self::Rook,
            other => other,
        }
    }

    /// Whether this value is a promoted piece state.
    #[must_use]
    pub const fn is_promoted(self) -> bool {
        self.unpromoted().index() != self.index()
    }

    /// Whether an unpromoted piece of this kind may promote.
    #[must_use]
    pub const fn is_promotable(self) -> bool {
        self.promoted().is_some()
    }

    /// SFEN base letter, before side casing and optional promotion prefix.
    #[must_use]
    pub const fn sfen_letter(self) -> char {
        match self.unpromoted() {
            Self::Pawn => 'P',
            Self::Lance => 'L',
            Self::Knight => 'N',
            Self::Silver => 'S',
            Self::Gold => 'G',
            Self::Bishop => 'B',
            Self::Rook => 'R',
            Self::King => 'K',
            _ => unreachable!(),
        }
    }

    /// CSA two-letter code for this exact piece state.
    #[must_use]
    pub const fn csa_code(self) -> &'static str {
        match self {
            Self::Pawn => "FU",
            Self::Lance => "KY",
            Self::Knight => "KE",
            Self::Silver => "GI",
            Self::Gold => "KI",
            Self::Bishop => "KA",
            Self::Rook => "HI",
            Self::King => "OU",
            Self::PromotedPawn => "TO",
            Self::PromotedLance => "NY",
            Self::PromotedKnight => "NK",
            Self::PromotedSilver => "NG",
            Self::Horse => "UM",
            Self::Dragon => "RY",
        }
    }
}

/// A piece on the board.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Piece {
    pub side: Side,
    pub kind: PieceKind,
}

impl Piece {
    #[must_use]
    pub const fn new(side: Side, kind: PieceKind) -> Self {
        Self { side, kind }
    }
}

/// Piece kinds that may exist in hand.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum HandPiece {
    Pawn,
    Lance,
    Knight,
    Silver,
    Gold,
    Bishop,
    Rook,
}

impl HandPiece {
    pub const ALL: [Self; HAND_KIND_COUNT] = [
        Self::Pawn,
        Self::Lance,
        Self::Knight,
        Self::Silver,
        Self::Gold,
        Self::Bishop,
        Self::Rook,
    ];

    /// CSA/SFEN conventional display order.
    pub const DISPLAY_ORDER: [Self; HAND_KIND_COUNT] = [
        Self::Rook,
        Self::Bishop,
        Self::Gold,
        Self::Silver,
        Self::Knight,
        Self::Lance,
        Self::Pawn,
    ];

    #[must_use]
    pub const fn index(self) -> usize {
        match self {
            Self::Pawn => 0,
            Self::Lance => 1,
            Self::Knight => 2,
            Self::Silver => 3,
            Self::Gold => 4,
            Self::Bishop => 5,
            Self::Rook => 6,
        }
    }

    #[must_use]
    pub const fn piece_kind(self) -> PieceKind {
        match self {
            Self::Pawn => PieceKind::Pawn,
            Self::Lance => PieceKind::Lance,
            Self::Knight => PieceKind::Knight,
            Self::Silver => PieceKind::Silver,
            Self::Gold => PieceKind::Gold,
            Self::Bishop => PieceKind::Bishop,
            Self::Rook => PieceKind::Rook,
        }
    }

    #[must_use]
    pub const fn from_piece_kind(kind: PieceKind) -> Option<Self> {
        match kind.unpromoted() {
            PieceKind::Pawn => Some(Self::Pawn),
            PieceKind::Lance => Some(Self::Lance),
            PieceKind::Knight => Some(Self::Knight),
            PieceKind::Silver => Some(Self::Silver),
            PieceKind::Gold => Some(Self::Gold),
            PieceKind::Bishop => Some(Self::Bishop),
            PieceKind::Rook => Some(Self::Rook),
            PieceKind::King
            | PieceKind::PromotedPawn
            | PieceKind::PromotedLance
            | PieceKind::PromotedKnight
            | PieceKind::PromotedSilver
            | PieceKind::Horse
            | PieceKind::Dragon => None,
        }
    }
}

/// Counts of unpromoted captured pieces held by one side.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct Hand {
    counts: [u8; HAND_KIND_COUNT],
}

impl Hand {
    #[must_use]
    pub const fn count(self, piece: HandPiece) -> u8 {
        self.counts[piece.index()]
    }

    pub(crate) fn set(&mut self, piece: HandPiece, count: u8) {
        self.counts[piece.index()] = count;
    }

    pub(crate) fn add(&mut self, piece: HandPiece) -> Result<(), ()> {
        let count = &mut self.counts[piece.index()];
        *count = count.checked_add(1).ok_or(())?;
        Ok(())
    }

    pub(crate) fn remove(&mut self, piece: HandPiece) -> Result<(), ()> {
        let count = &mut self.counts[piece.index()];
        *count = count.checked_sub(1).ok_or(())?;
        Ok(())
    }

    #[must_use]
    pub fn total(self) -> u16 {
        self.counts.iter().map(|count| u16::from(*count)).sum()
    }
}

/// A normal move or a drop from hand.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Move {
    Normal {
        from: Square,
        to: Square,
        promote: bool,
    },
    Drop {
        piece: HandPiece,
        to: Square,
    },
}

impl Move {
    #[must_use]
    pub const fn destination(self) -> Square {
        match self {
            Self::Normal { to, .. } | Self::Drop { to, .. } => to,
        }
    }
}
