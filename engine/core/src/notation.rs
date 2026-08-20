//! Strict parsers and canonical writers for shogi interchange notation.

use std::{error::Error, fmt, fmt::Write as _};

use crate::{
    Hand, HandPiece, Move, Piece, PieceKind, Position, RepetitionOutcome, Side, Square,
    game::repetition_outcome_from_history,
};

const MAX_SFEN_BYTES: usize = 1_024;
const MAX_USI_MOVE_BYTES: usize = 5;
const MAX_CSA_BYTES: usize = 1_048_576;
const MAX_CSA_LINE_BYTES: usize = 4_096;
const MAX_CSA_LINES: usize = 200_100;
const MAX_CSA_MOVES: usize = 100_000;

/// An error returned when SFEN, USI move notation, or CSA output is invalid.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum NotationError {
    /// The input exceeded the parser's documented defensive limit.
    InputTooLong {
        /// Maximum accepted byte length.
        maximum: usize,
    },
    /// The input did not have the required top-level structure.
    InvalidStructure(String),
    /// A board field was malformed.
    InvalidBoard(String),
    /// A hand field was malformed.
    InvalidHands(String),
    /// A side-to-move field was malformed.
    InvalidSide(String),
    /// A move number was malformed.
    InvalidMoveNumber(String),
    /// A USI or CSA move was malformed.
    InvalidMove(String),
    /// The parsed position violated a position invariant.
    InvalidPosition(String),
    /// A move could not be applied to the supplied position.
    IllegalMove(String),
}

impl fmt::Display for NotationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InputTooLong { maximum } => {
                write!(formatter, "notation input exceeds the {maximum}-byte limit")
            }
            Self::InvalidStructure(reason) => write!(formatter, "invalid notation: {reason}"),
            Self::InvalidBoard(reason) => write!(formatter, "invalid board field: {reason}"),
            Self::InvalidHands(reason) => write!(formatter, "invalid hands field: {reason}"),
            Self::InvalidSide(reason) => write!(formatter, "invalid side field: {reason}"),
            Self::InvalidMoveNumber(reason) => {
                write!(formatter, "invalid move number: {reason}")
            }
            Self::InvalidMove(reason) => write!(formatter, "invalid move: {reason}"),
            Self::InvalidPosition(reason) => write!(formatter, "invalid position: {reason}"),
            Self::IllegalMove(reason) => write!(formatter, "illegal move: {reason}"),
        }
    }
}

impl Error for NotationError {}

/// A recognized CSA terminal or protocol special-move line.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum CsaSpecialMove {
    /// `%TORYO`: resignation.
    Resign,
    /// `%CHUDAN`: game interruption.
    Interrupted,
    /// `%SENNICHITE`: repetition draw.
    Repetition,
    /// `%OUTE_SENNICHITE`: loss by perpetual check.
    PerpetualCheck,
    /// `%ILLEGAL_MOVE`: an illegal move ended the game.
    IllegalMove,
    /// `%+ILLEGAL_ACTION`: Black committed an illegal action.
    BlackIllegalAction,
    /// `%-ILLEGAL_ACTION`: White committed an illegal action.
    WhiteIllegalAction,
    /// `%TIME_UP`: loss on time.
    TimeUp,
    /// `%JISHOGI`: entering-king adjudication.
    EnteringKing,
    /// `%KACHI`: entering-king declaration win.
    Win,
    /// `%HIKIWAKE`: draw.
    Draw,
    /// `%MAX_MOVES`: the configured move limit was reached.
    MaxMoves,
    /// `%MATTA`: takeback or illegal interruption.
    Takeback,
    /// `%TSUMI`: checkmate.
    Checkmate,
    /// `%FUZUMI`: no checkmate.
    NoMate,
    /// `%ERROR`: protocol or record error.
    Error,
    /// An unrecognized code, preserved without its leading `%`.
    Other(String),
}

impl CsaSpecialMove {
    fn parse(code: &str) -> Result<Self, NotationError> {
        if code.is_empty() {
            return Err(NotationError::InvalidMove(
                "CSA special-move code is empty".to_owned(),
            ));
        }
        let known = match code {
            "TORYO" => Self::Resign,
            "CHUDAN" => Self::Interrupted,
            "SENNICHITE" => Self::Repetition,
            "OUTE_SENNICHITE" => Self::PerpetualCheck,
            "ILLEGAL_MOVE" => Self::IllegalMove,
            "+ILLEGAL_ACTION" => Self::BlackIllegalAction,
            "-ILLEGAL_ACTION" => Self::WhiteIllegalAction,
            "TIME_UP" => Self::TimeUp,
            "JISHOGI" => Self::EnteringKing,
            "KACHI" => Self::Win,
            "HIKIWAKE" => Self::Draw,
            "MAX_MOVES" => Self::MaxMoves,
            "MATTA" => Self::Takeback,
            "TSUMI" => Self::Checkmate,
            "FUZUMI" => Self::NoMate,
            "ERROR" => Self::Error,
            other => {
                if !other.bytes().all(|byte| {
                    byte.is_ascii_uppercase()
                        || byte.is_ascii_digit()
                        || byte == b'_'
                        || byte == b'+'
                        || byte == b'-'
                }) {
                    return Err(NotationError::InvalidMove(
                        "CSA special-move code contains invalid characters".to_owned(),
                    ));
                }
                Self::Other(other.to_owned())
            }
        };
        Ok(known)
    }

    fn code(&self) -> Result<&str, NotationError> {
        match self {
            Self::Resign => Ok("TORYO"),
            Self::Interrupted => Ok("CHUDAN"),
            Self::Repetition => Ok("SENNICHITE"),
            Self::PerpetualCheck => Ok("OUTE_SENNICHITE"),
            Self::IllegalMove => Ok("ILLEGAL_MOVE"),
            Self::BlackIllegalAction => Ok("+ILLEGAL_ACTION"),
            Self::WhiteIllegalAction => Ok("-ILLEGAL_ACTION"),
            Self::TimeUp => Ok("TIME_UP"),
            Self::EnteringKing => Ok("JISHOGI"),
            Self::Win => Ok("KACHI"),
            Self::Draw => Ok("HIKIWAKE"),
            Self::MaxMoves => Ok("MAX_MOVES"),
            Self::Takeback => Ok("MATTA"),
            Self::Checkmate => Ok("TSUMI"),
            Self::NoMate => Ok("FUZUMI"),
            Self::Error => Ok("ERROR"),
            Self::Other(code) => {
                if code.is_empty()
                    || !code.bytes().all(|byte| {
                        byte.is_ascii_uppercase()
                            || byte.is_ascii_digit()
                            || byte == b'_'
                            || byte == b'+'
                            || byte == b'-'
                    })
                {
                    Err(NotationError::InvalidMove(
                        "CSA special-move code contains invalid characters".to_owned(),
                    ))
                } else {
                    Ok(code)
                }
            }
        }
    }
}

/// A parsed CSA game and the position from which its moves begin.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CsaGame {
    /// CSA version line. Phase 1 accepts `V3.0`.
    pub version: String,
    /// Optional Black player name from `N+`.
    pub black_name: Option<String>,
    /// Optional White player name from `N-`.
    pub white_name: Option<String>,
    /// `$KEY:VALUE` metadata in input order, without the leading `$`.
    pub metadata: Vec<(String, String)>,
    /// Initial position before the first recorded move.
    pub initial_position: Position,
    /// Legal moves, replayed from `initial_position` while parsing.
    pub moves: Vec<Move>,
    /// Optional terminal CSA `%` line.
    pub special_move: Option<CsaSpecialMove>,
    /// Whether the terminal line was verified from replayable rules state.
    pub result_validation: CsaResultValidation,
}

/// Confidence attached to a parsed CSA terminal line.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CsaResultValidation {
    /// The record has no terminal `%` line.
    Missing,
    /// The terminal condition is deterministic and was verified by replay.
    Verified,
    /// The terminal condition depends on clocks, declarations, agreements, or another
    /// tournament condition that the record alone cannot establish.
    ExternalCondition,
}

/// A CSA parse failure with its one-based source line, when applicable.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum CsaParseError {
    /// The complete input exceeded the defensive byte limit.
    InputTooLong {
        /// Maximum accepted byte length.
        maximum: usize,
    },
    /// The input contained too many lines.
    TooManyLines {
        /// Maximum accepted number of lines.
        maximum: usize,
    },
    /// A single line exceeded the defensive byte limit.
    LineTooLong {
        /// One-based source line.
        line: usize,
        /// Maximum accepted byte length.
        maximum: usize,
    },
    /// A line or required section was structurally invalid.
    Invalid {
        /// One-based source line, or `None` for an absent required section.
        line: Option<usize>,
        /// Human-readable diagnostic.
        reason: String,
    },
    /// A notation-level error occurred on a line.
    Notation {
        /// One-based source line.
        line: usize,
        /// Underlying notation error.
        source: NotationError,
    },
    /// A CSA move was syntactically valid but illegal in the replayed position.
    IllegalMove {
        /// One-based source line.
        line: usize,
        /// Human-readable legality diagnostic.
        reason: String,
    },
}

impl CsaParseError {
    /// Returns the one-based line associated with this error, if one exists.
    #[must_use]
    pub const fn line(&self) -> Option<usize> {
        match self {
            Self::LineTooLong { line, .. }
            | Self::Notation { line, .. }
            | Self::IllegalMove { line, .. } => Some(*line),
            Self::Invalid { line, .. } => *line,
            Self::InputTooLong { .. } | Self::TooManyLines { .. } => None,
        }
    }
}

impl fmt::Display for CsaParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InputTooLong { maximum } => {
                write!(formatter, "CSA input exceeds the {maximum}-byte limit")
            }
            Self::TooManyLines { maximum } => {
                write!(formatter, "CSA input exceeds the {maximum}-line limit")
            }
            Self::LineTooLong { line, maximum } => {
                write!(
                    formatter,
                    "CSA line {line} exceeds the {maximum}-byte limit"
                )
            }
            Self::Invalid {
                line: Some(line),
                reason,
            } => write!(formatter, "invalid CSA at line {line}: {reason}"),
            Self::Invalid { line: None, reason } => write!(formatter, "invalid CSA: {reason}"),
            Self::Notation { line, source } => {
                write!(formatter, "invalid CSA notation at line {line}: {source}")
            }
            Self::IllegalMove { line, reason } => {
                write!(formatter, "illegal CSA move at line {line}: {reason}")
            }
        }
    }
}

impl Error for CsaParseError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Notation { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// Parses a strict four-field SFEN position.
///
/// # Errors
///
/// Returns [`NotationError`] for oversized input, non-canonical field syntax, malformed
/// pieces or hands, missing/duplicate kings, or a position rejected by [`Position`].
pub fn parse_sfen(input: &str) -> Result<Position, NotationError> {
    if input.len() > MAX_SFEN_BYTES {
        return Err(NotationError::InputTooLong {
            maximum: MAX_SFEN_BYTES,
        });
    }
    if input.is_empty()
        || input.starts_with(' ')
        || input.ends_with(' ')
        || input
            .bytes()
            .any(|byte| matches!(byte, b'\t' | b'\r' | b'\n'))
        || input.contains("  ")
    {
        return Err(NotationError::InvalidStructure(
            "SFEN must use exactly one ASCII space between fields".to_owned(),
        ));
    }
    let fields = input.split(' ').collect::<Vec<_>>();
    let [board_field, side_field, hand_field, move_number_field] = fields.as_slice() else {
        return Err(NotationError::InvalidStructure(
            "SFEN must contain exactly four fields".to_owned(),
        ));
    };

    let board = parse_sfen_board(board_field)?;
    require_exactly_one_king_per_side(&board)?;
    let hands = parse_sfen_hands(hand_field)?;
    let side = match *side_field {
        "b" => Side::Black,
        "w" => Side::White,
        other => {
            return Err(NotationError::InvalidSide(format!(
                "expected `b` or `w`, found `{other}`"
            )));
        }
    };
    let move_number = parse_move_number(move_number_field)?;

    Position::from_parts(board, hands, side, move_number)
        .map_err(|error| NotationError::InvalidPosition(error.to_string()))
}

/// Serializes a position as canonical four-field SFEN.
#[must_use]
pub fn to_sfen(position: &Position) -> String {
    let mut output = String::with_capacity(128);
    for rank_index in 0..9 {
        if rank_index > 0 {
            output.push('/');
        }
        let mut empty = 0_u8;
        for column in 0..9 {
            match position.board()[rank_index * 9 + column] {
                None => empty += 1,
                Some(piece) => {
                    if empty > 0 {
                        output.push(char::from(b'0' + empty));
                        empty = 0;
                    }
                    push_sfen_piece(&mut output, piece);
                }
            }
        }
        if empty > 0 {
            output.push(char::from(b'0' + empty));
        }
    }
    output.push(' ');
    output.push(match position.side_to_move() {
        Side::Black => 'b',
        Side::White => 'w',
    });
    output.push(' ');
    push_sfen_hands(&mut output, position.hands());
    write!(output, " {}", position.move_number()).expect("writing to String cannot fail");
    output
}

/// Parses one USI normal move or drop.
///
/// # Errors
///
/// Returns [`NotationError`] if `input` is not exactly a valid four- or five-byte USI move.
/// Position-dependent legality is intentionally checked when applying the returned move.
pub fn parse_usi_move(input: &str) -> Result<Move, NotationError> {
    if input.len() > MAX_USI_MOVE_BYTES {
        return Err(NotationError::InputTooLong {
            maximum: MAX_USI_MOVE_BYTES,
        });
    }
    let bytes = input.as_bytes();
    if bytes.len() == 4 && bytes[1] == b'*' {
        let piece = hand_piece_from_sfen_letter(bytes[0] as char).ok_or_else(|| {
            NotationError::InvalidMove("USI drop piece must be one of `PLNSGBR`".to_owned())
        })?;
        if !bytes[0].is_ascii_uppercase() {
            return Err(NotationError::InvalidMove(
                "USI drop piece must be uppercase".to_owned(),
            ));
        }
        let to = parse_usi_square(&bytes[2..4])?;
        return Ok(Move::Drop { piece, to });
    }
    if bytes.len() != 4 && bytes.len() != 5 {
        return Err(NotationError::InvalidMove(
            "USI normal move must contain four bytes plus optional `+`".to_owned(),
        ));
    }
    if bytes.len() == 5 && bytes[4] != b'+' {
        return Err(NotationError::InvalidMove(
            "the fifth USI move byte must be `+`".to_owned(),
        ));
    }
    let from = parse_usi_square(&bytes[0..2])?;
    let to = parse_usi_square(&bytes[2..4])?;
    Ok(Move::Normal {
        from,
        to,
        promote: bytes.len() == 5,
    })
}

/// Serializes one move as canonical USI move notation.
#[must_use]
pub fn to_usi_move(mv: Move) -> String {
    match mv {
        Move::Normal { from, to, promote } => {
            let mut output = format!("{from}{to}");
            if promote {
                output.push('+');
            }
            output
        }
        Move::Drop { piece, to } => {
            format!("{}*{to}", piece.piece_kind().sfen_letter())
        }
    }
}

/// Serializes a legal move in the supplied position as one CSA move line.
///
/// The position is required because a CSA normal move records the resulting piece kind, not
/// an explicit promotion marker.
///
/// # Errors
///
/// Returns [`NotationError`] if a normal move's source piece is absent, belongs to the other
/// side, or cannot produce the requested promoted state. Full move legality is checked when a
/// caller applies the move, including by [`to_csa_game`].
pub fn to_csa_move(position: &Position, mv: Move) -> Result<String, NotationError> {
    if !position.is_legal_move(mv) {
        return Err(NotationError::IllegalMove(format!(
            "{} is not legal in the supplied position",
            to_usi_move(mv)
        )));
    }
    let side_marker = csa_side_marker(position.side_to_move());
    match mv {
        Move::Normal { from, to, promote } => {
            let source = position.piece_at(from).ok_or_else(|| {
                NotationError::InvalidPosition(format!("move source {from} is empty"))
            })?;
            if source.side != position.side_to_move() {
                return Err(NotationError::InvalidPosition(format!(
                    "piece on {from} belongs to the other side"
                )));
            }
            let result_kind = if promote {
                source.kind.promoted().ok_or_else(|| {
                    NotationError::InvalidMove(format!(
                        "{:?} cannot promote on move {}",
                        source.kind,
                        to_usi_move(mv)
                    ))
                })?
            } else {
                source.kind
            };
            Ok(format!(
                "{side_marker}{}{}{}{}{}",
                from.file(),
                from.rank(),
                to.file(),
                to.rank(),
                result_kind.csa_code()
            ))
        }
        Move::Drop { piece, to } => Ok(format!(
            "{side_marker}00{}{}{}",
            to.file(),
            to.rank(),
            piece.piece_kind().csa_code()
        )),
    }
}

/// Parses one CSA V3.0 game, including its initial position and legal move sequence.
///
/// Moves are replayed through [`Position::make_move`]. Consequently, syntactically valid but
/// illegal records are rejected at the exact source line instead of being repaired.
///
/// # Errors
///
/// Returns [`CsaParseError`] for oversized input, malformed or duplicate sections, invalid
/// initial state, malformed moves, or a move that is illegal in the replayed position.
#[expect(
    clippy::too_many_lines,
    reason = "the line-oriented CSA grammar is kept in one explicit state machine"
)]
pub fn parse_csa_game(input: &str) -> Result<CsaGame, CsaParseError> {
    if input.len() > MAX_CSA_BYTES {
        return Err(CsaParseError::InputTooLong {
            maximum: MAX_CSA_BYTES,
        });
    }
    if input.is_empty() {
        return Err(CsaParseError::Invalid {
            line: None,
            reason: "CSA input is empty".to_owned(),
        });
    }
    let statements = collect_csa_statements(input)?;

    let mut version = None;
    let mut black_name = None;
    let mut white_name = None;
    let mut metadata = Vec::new();
    let mut initial = CsaInitialPosition::default();
    let mut initial_position = None;
    let mut replay = None;
    let mut replay_positions = Vec::new();
    let mut replay_gave_check = Vec::new();
    let mut moves = Vec::new();
    let mut special_move = None;
    let mut result_validation = CsaResultValidation::Missing;
    let mut turn_line = None;
    let mut last_was_move = false;

    for (line_number, line) in statements {
        if line.starts_with('\'') {
            last_was_move = false;
            continue;
        }
        if !line.starts_with('V') && version.is_none() {
            return Err(invalid_csa_line(
                line_number,
                "the CSA version must be the first non-comment statement",
            ));
        }
        if special_move.is_some() {
            if let Some(time) = line.strip_prefix('T') {
                if !last_was_move || !is_valid_csa_time(time) {
                    return Err(invalid_csa_line(
                        line_number,
                        "time line must be `T` followed by seconds, optionally with milliseconds",
                    ));
                }
                last_was_move = false;
                continue;
            }
            return Err(invalid_csa_line(
                line_number,
                "only one time line is allowed after a special-move line",
            ));
        }

        if let Some(version_text) = line.strip_prefix('V') {
            if replay.is_some() || version.is_some() {
                return Err(invalid_csa_line(
                    line_number,
                    "version line is duplicated or out of order",
                ));
            }
            validate_csa_version(version_text)
                .map_err(|reason| invalid_csa_line(line_number, reason))?;
            version = Some(line.to_owned());
            last_was_move = false;
            continue;
        }
        if let Some(name) = line.strip_prefix("N+") {
            require_before_turn(replay.is_some(), line_number, "player name")?;
            set_csa_name(&mut black_name, name, line_number, "Black")?;
            last_was_move = false;
            continue;
        }
        if let Some(name) = line.strip_prefix("N-") {
            require_before_turn(replay.is_some(), line_number, "player name")?;
            set_csa_name(&mut white_name, name, line_number, "White")?;
            last_was_move = false;
            continue;
        }
        if let Some(metadata_line) = line.strip_prefix('$') {
            require_before_turn(replay.is_some(), line_number, "metadata")?;
            let (key, value) = parse_csa_metadata(metadata_line)
                .map_err(|reason| invalid_csa_line(line_number, reason))?;
            metadata.push((key.to_owned(), value.to_owned()));
            last_was_move = false;
            continue;
        }
        if let Some(removals) = line.strip_prefix("PI") {
            require_before_turn(replay.is_some(), line_number, "initial position")?;
            initial
                .set_startpos(removals)
                .map_err(|reason| invalid_csa_line(line_number, reason))?;
            last_was_move = false;
            continue;
        }
        if line.starts_with('P') {
            require_before_turn(replay.is_some(), line_number, "initial position")?;
            initial
                .parse_line(line)
                .map_err(|error| CsaParseError::Notation {
                    line: line_number,
                    source: error,
                })?;
            last_was_move = false;
            continue;
        }
        if line == "+" || line == "-" {
            if replay.is_some() {
                return Err(invalid_csa_line(
                    line_number,
                    "side-to-move line is duplicated",
                ));
            }
            if version.is_none() {
                return Err(invalid_csa_line(
                    line_number,
                    "a V3.0 version line is required before the position",
                ));
            }
            let side = if line == "+" {
                Side::Black
            } else {
                Side::White
            };
            let position =
                initial
                    .clone()
                    .finish(side)
                    .map_err(|error| CsaParseError::Notation {
                        line: line_number,
                        source: error,
                    })?;
            initial_position = Some(position.clone());
            replay_positions.push(position.clone());
            replay = Some(position);
            turn_line = Some(line_number);
            last_was_move = false;
            continue;
        }
        if let Some(code) = line.strip_prefix('%') {
            if replay.is_none() {
                return Err(invalid_csa_line(
                    line_number,
                    "special move appears before the side-to-move line",
                ));
            }
            let parsed = CsaSpecialMove::parse(code).map_err(|source| CsaParseError::Notation {
                line: line_number,
                source,
            })?;
            let position = replay.as_ref().ok_or_else(|| {
                invalid_csa_line(
                    line_number,
                    "special move appears before the side-to-move line",
                )
            })?;
            result_validation = classify_csa_result(
                position,
                repetition_outcome_from_history(&replay_positions, &replay_gave_check),
                &parsed,
            )
            .map_err(|reason| invalid_csa_line(line_number, reason))?;
            special_move = Some(parsed);
            last_was_move = true;
            continue;
        }
        if let Some(time) = line.strip_prefix('T') {
            if !last_was_move || !is_valid_csa_time(time) {
                return Err(invalid_csa_line(
                    line_number,
                    "time line must be `T` followed by seconds, optionally with milliseconds",
                ));
            }
            last_was_move = false;
            continue;
        }
        if line.starts_with('+') || line.starts_with('-') {
            if moves.len() >= MAX_CSA_MOVES {
                return Err(invalid_csa_line(
                    line_number,
                    "CSA move count exceeds the defensive limit",
                ));
            }
            let position = replay.as_mut().ok_or_else(|| {
                invalid_csa_line(line_number, "move appears before the side-to-move line")
            })?;
            let mv = parse_csa_move(position, line).map_err(|source| CsaParseError::Notation {
                line: line_number,
                source,
            })?;
            position
                .make_move(mv)
                .map_err(|error| CsaParseError::IllegalMove {
                    line: line_number,
                    reason: error.to_string(),
                })?;
            replay_gave_check.push(position.is_in_check(position.side_to_move()));
            replay_positions.push(position.clone());
            moves.push(mv);
            last_was_move = true;
            continue;
        }
        return Err(invalid_csa_line(
            line_number,
            "unrecognized CSA record line",
        ));
    }

    let version = version.ok_or_else(|| CsaParseError::Invalid {
        line: None,
        reason: "CSA V3.0 version line is required".to_owned(),
    })?;
    let initial_position = initial_position.ok_or_else(|| CsaParseError::Invalid {
        line: turn_line,
        reason: "initial position and side-to-move line are required".to_owned(),
    })?;
    Ok(CsaGame {
        version,
        black_name,
        white_name,
        metadata,
        initial_position,
        moves,
        special_move,
        result_validation,
    })
}

fn classify_csa_result(
    position: &Position,
    repetition: Option<RepetitionOutcome>,
    special: &CsaSpecialMove,
) -> Result<CsaResultValidation, String> {
    match special {
        CsaSpecialMove::Checkmate if position.is_checkmate() => Ok(CsaResultValidation::Verified),
        CsaSpecialMove::Checkmate => {
            Err("terminal result `%TSUMI` is inconsistent with the replayed position".to_owned())
        }
        CsaSpecialMove::Repetition if matches!(repetition, Some(RepetitionOutcome::NoContest)) => {
            Ok(CsaResultValidation::Verified)
        }
        CsaSpecialMove::Repetition => {
            // JSA rules permit the players to agree to a repetition restart before the
            // fourth occurrence. A record cannot prove that agreement.
            Ok(CsaResultValidation::ExternalCondition)
        }
        CsaSpecialMove::PerpetualCheck
            if matches!(repetition, Some(RepetitionOutcome::PerpetualCheckLoss(_))) =>
        {
            Ok(CsaResultValidation::Verified)
        }
        CsaSpecialMove::PerpetualCheck => Err(
            "terminal result `%OUTE_SENNICHITE` is inconsistent with the replayed history"
                .to_owned(),
        ),
        _ => Ok(CsaResultValidation::ExternalCondition),
    }
}

/// Writes a CSA game in canonical full-board form.
///
/// The output is normalized to UTF-8 CSA V3.0 and always contains nine board rows, one hand
/// line per side, a turn line, canonical move lines, and the optional special move. Moves are
/// checked by replaying them from the stored initial position.
///
/// # Errors
///
/// Returns [`NotationError`] if metadata cannot be represented safely, a special code is
/// invalid, or any recorded move is illegal in sequence.
pub fn to_csa_game(game: &CsaGame) -> Result<String, NotationError> {
    validate_csa_version(game.version.strip_prefix('V').ok_or_else(|| {
        NotationError::InvalidStructure("CSA version must start with uppercase `V`".to_owned())
    })?)
    .map_err(NotationError::InvalidStructure)?;
    if game.moves.len() > MAX_CSA_MOVES {
        return Err(NotationError::InputTooLong {
            maximum: MAX_CSA_MOVES,
        });
    }
    if game.initial_position.move_number() != 1 {
        return Err(NotationError::InvalidStructure(
            "CSA has no initial move-number field; canonical output requires move number one"
                .to_owned(),
        ));
    }

    let mut output = String::with_capacity(512 + game.moves.len().saturating_mul(8));
    push_checked_csa_line(&mut output, "'CSA encoding=UTF-8")?;
    push_checked_csa_line(&mut output, "V3.0")?;
    if let Some(name) = &game.black_name {
        validate_csa_text(name, "Black player name")?;
        push_checked_csa_line(&mut output, &format!("N+{name}"))?;
    }
    if let Some(name) = &game.white_name {
        validate_csa_text(name, "White player name")?;
        push_checked_csa_line(&mut output, &format!("N-{name}"))?;
    }
    for (key, value) in &game.metadata {
        validate_csa_metadata(key, value)?;
        push_checked_csa_line(&mut output, &format!("${key}:{value}"))?;
    }
    push_csa_board(&mut output, &game.initial_position)?;
    push_csa_hand(&mut output, Side::Black, &game.initial_position)?;
    push_csa_hand(&mut output, Side::White, &game.initial_position)?;
    push_checked_csa_line(
        &mut output,
        match game.initial_position.side_to_move() {
            Side::Black => "+",
            Side::White => "-",
        },
    )?;

    let mut replay = game.initial_position.clone();
    let mut replay_positions = vec![replay.clone()];
    let mut replay_gave_check = Vec::with_capacity(game.moves.len());
    for &mv in &game.moves {
        let line = to_csa_move(&replay, mv)?;
        push_checked_csa_line(&mut output, &line)?;
        replay
            .make_move(mv)
            .map_err(|error| NotationError::IllegalMove(error.to_string()))?;
        replay_gave_check.push(replay.is_in_check(replay.side_to_move()));
        replay_positions.push(replay.clone());
    }
    if let Some(special) = &game.special_move {
        let actual_validation = classify_csa_result(
            &replay,
            repetition_outcome_from_history(&replay_positions, &replay_gave_check),
            special,
        )
        .map_err(NotationError::InvalidStructure)?;
        if actual_validation != game.result_validation {
            return Err(NotationError::InvalidStructure(format!(
                "CSA result validation is {:?}, but replay classifies it as {actual_validation:?}",
                game.result_validation
            )));
        }
        push_checked_csa_line(&mut output, &format!("%{}", special.code()?))?;
    } else if game.result_validation != CsaResultValidation::Missing {
        return Err(NotationError::InvalidStructure(
            "CSA result validation must be `Missing` when no terminal line is present".to_owned(),
        ));
    }
    if output.len() > MAX_CSA_BYTES {
        return Err(NotationError::InputTooLong {
            maximum: MAX_CSA_BYTES,
        });
    }
    Ok(output)
}

#[derive(Clone, Debug)]
struct CsaInitialPosition {
    board: [Option<Piece>; 81],
    hands: [Hand; 2],
    row_seen: [bool; 9],
    saw_placement: bool,
    startpos: bool,
    all_remaining: Option<Side>,
}

impl Default for CsaInitialPosition {
    fn default() -> Self {
        Self {
            board: [None; 81],
            hands: [Hand::default(); 2],
            row_seen: [false; 9],
            saw_placement: false,
            startpos: false,
            all_remaining: None,
        }
    }
}

impl CsaInitialPosition {
    fn set_startpos(&mut self, removals: &str) -> Result<(), String> {
        if self.startpos
            || self.saw_placement
            || self.row_seen.iter().any(|seen| *seen)
            || self.hands.iter().any(|hand| hand.total() > 0)
            || self.all_remaining.is_some()
        {
            return Err(
                "`PI` cannot be combined with another initial-position declaration".to_owned(),
            );
        }
        if !removals.is_ascii() || !removals.len().is_multiple_of(4) {
            return Err("`PI` removals must be four-byte square and piece entries".to_owned());
        }
        let position = Position::startpos();
        self.board = *position.board();
        self.hands = *position.hands();
        for offset in (0..removals.len()).step_by(4) {
            let entry = &removals[offset..offset + 4];
            let bytes = entry.as_bytes();
            let square = parse_csa_square(bytes[0], bytes[1]).map_err(|error| error.to_string())?;
            let expected = piece_kind_from_csa_code(&entry[2..]).ok_or_else(|| {
                format!("unknown CSA piece code `{}` in `PI` removal", &entry[2..])
            })?;
            let actual = self.board[square.index()].ok_or_else(|| {
                format!(
                    "`PI` removal square {}{} is empty",
                    square.file(),
                    square.rank()
                )
            })?;
            if actual.kind != expected {
                return Err(format!(
                    "`PI` removal expected {expected:?} on {}{} but found {:?}",
                    square.file(),
                    square.rank(),
                    actual.kind
                ));
            }
            self.board[square.index()] = None;
        }
        self.startpos = true;
        Ok(())
    }

    fn parse_line(&mut self, line: &str) -> Result<(), NotationError> {
        if !line.is_ascii() {
            return Err(NotationError::InvalidBoard(
                "CSA position lines must contain only ASCII".to_owned(),
            ));
        }
        if self.startpos {
            return Err(NotationError::InvalidBoard(
                "`PI` cannot be combined with `P` position lines".to_owned(),
            ));
        }
        if self.all_remaining.is_some() {
            return Err(NotationError::InvalidBoard(
                "`00AL` must be the final initial-position entry".to_owned(),
            ));
        }
        let bytes = line.as_bytes();
        if bytes.len() >= 2 && (b'1'..=b'9').contains(&bytes[1]) {
            self.parse_board_row(line)
        } else if line.starts_with("P+") || line.starts_with("P-") {
            self.parse_piece_line(line)
        } else {
            Err(NotationError::InvalidBoard(
                "position line must be `P1`-`P9`, `P+`, or `P-`".to_owned(),
            ))
        }
    }

    fn parse_board_row(&mut self, line: &str) -> Result<(), NotationError> {
        if self.saw_placement {
            return Err(NotationError::InvalidBoard(
                "CSA board rows cannot be combined with coordinate placements".to_owned(),
            ));
        }
        if line.len() != 29 {
            return Err(NotationError::InvalidBoard(
                "CSA board row must contain exactly nine three-byte cells".to_owned(),
            ));
        }
        let rank = usize::from(line.as_bytes()[1] - b'1');
        if self.row_seen[rank] {
            return Err(NotationError::InvalidBoard(format!(
                "CSA board row {} is duplicated",
                rank + 1
            )));
        }
        for column in 0..9 {
            let start = 2 + column * 3;
            let cell = &line[start..start + 3];
            self.board[rank * 9 + column] = if cell == " * " {
                None
            } else {
                let side = match cell.as_bytes()[0] {
                    b'+' => Side::Black,
                    b'-' => Side::White,
                    _ => {
                        return Err(NotationError::InvalidBoard(format!(
                            "invalid CSA board cell `{cell}`"
                        )));
                    }
                };
                let kind = piece_kind_from_csa_code(&cell[1..]).ok_or_else(|| {
                    NotationError::InvalidBoard(format!("invalid CSA piece code `{}`", &cell[1..]))
                })?;
                Some(Piece::new(side, kind))
            };
        }
        self.row_seen[rank] = true;
        Ok(())
    }

    fn parse_piece_line(&mut self, line: &str) -> Result<(), NotationError> {
        let side = if line.starts_with("P+") {
            Side::Black
        } else {
            Side::White
        };
        let entries = &line[2..];
        if !entries.len().is_multiple_of(4) {
            return Err(NotationError::InvalidBoard(
                "CSA piece line must contain four-byte entries".to_owned(),
            ));
        }
        for entry_start in (0..entries.len()).step_by(4) {
            let entry = &entries[entry_start..entry_start + 4];
            if entry == "00AL" {
                if entry_start + 4 != entries.len() {
                    return Err(NotationError::InvalidHands(
                        "`00AL` must be the final entry on its position line".to_owned(),
                    ));
                }
                if self.all_remaining.replace(side).is_some() {
                    return Err(NotationError::InvalidHands(
                        "`00AL` may appear only once".to_owned(),
                    ));
                }
                continue;
            }
            let kind = piece_kind_from_csa_code(&entry[2..]).ok_or_else(|| {
                NotationError::InvalidBoard(format!("invalid CSA piece code `{}`", &entry[2..]))
            })?;
            let file = entry.as_bytes()[0];
            let rank = entry.as_bytes()[1];
            if file == b'0' && rank == b'0' {
                let hand_piece = HandPiece::from_piece_kind(kind).ok_or_else(|| {
                    NotationError::InvalidHands(format!(
                        "promoted piece or king `{}` cannot be in hand",
                        &entry[2..]
                    ))
                })?;
                self.hands[side.index()].add(hand_piece).map_err(|()| {
                    NotationError::InvalidHands("CSA hand count exceeds 255".to_owned())
                })?;
                continue;
            }
            let square = parse_csa_square(file, rank)?;
            if self.row_seen.iter().any(|seen| *seen) {
                return Err(NotationError::InvalidBoard(
                    "CSA coordinate placements cannot be combined with board rows".to_owned(),
                ));
            }
            if self.board[square.index()].is_some() {
                return Err(NotationError::InvalidBoard(format!(
                    "CSA square {file}{rank} is assigned more than once"
                )));
            }
            self.board[square.index()] = Some(Piece::new(side, kind));
            self.saw_placement = true;
        }
        Ok(())
    }

    fn finish(mut self, side: Side) -> Result<Position, NotationError> {
        let rows = self.row_seen.iter().filter(|seen| **seen).count();
        if !self.startpos && rows != 0 && rows != 9 {
            return Err(NotationError::InvalidBoard(format!(
                "CSA full-board position has {rows} rows instead of nine"
            )));
        }
        if !self.startpos && rows == 0 && !self.saw_placement {
            return Err(NotationError::InvalidBoard(
                "CSA initial position is missing".to_owned(),
            ));
        }
        if let Some(owner) = self.all_remaining {
            self.add_all_remaining(owner)?;
        }
        require_exactly_one_king_per_side(&self.board)?;
        Position::from_parts(self.board, self.hands, side, 1)
            .map_err(|error| NotationError::InvalidPosition(error.to_string()))
    }

    fn add_all_remaining(&mut self, owner: Side) -> Result<(), NotationError> {
        const STOCK: [(HandPiece, u16); 7] = [
            (HandPiece::Pawn, 18),
            (HandPiece::Lance, 4),
            (HandPiece::Knight, 4),
            (HandPiece::Silver, 4),
            (HandPiece::Gold, 4),
            (HandPiece::Bishop, 2),
            (HandPiece::Rook, 2),
        ];
        for (hand_piece, maximum) in STOCK {
            let board_count = u16::try_from(
                self.board
                    .iter()
                    .flatten()
                    .filter(|piece| piece.kind.unpromoted() == hand_piece.piece_kind())
                    .count(),
            )
            .map_err(|_| {
                NotationError::InvalidHands("CSA board material count overflow".to_owned())
            })?;
            let hand_count = u16::from(self.hands[0].count(hand_piece))
                + u16::from(self.hands[1].count(hand_piece));
            let used = board_count.checked_add(hand_count).ok_or_else(|| {
                NotationError::InvalidHands("CSA material count overflow".to_owned())
            })?;
            let remaining = maximum.checked_sub(used).ok_or_else(|| {
                NotationError::InvalidHands(format!(
                    "CSA contains more {hand_piece:?} pieces than a standard set"
                ))
            })?;
            let existing = u16::from(self.hands[owner.index()].count(hand_piece));
            let count = existing
                .checked_add(remaining)
                .ok_or_else(|| NotationError::InvalidHands("CSA hand count overflow".to_owned()))?;
            self.hands[owner.index()].set(
                hand_piece,
                u8::try_from(count).map_err(|_| {
                    NotationError::InvalidHands("CSA hand count exceeds 255".to_owned())
                })?,
            );
        }
        Ok(())
    }
}

/// Parses one position-aware CSA move.
///
/// This validates the side marker, source piece, destination piece code, and promotion shape.
/// Complete shogi legality is checked by [`Position::make_move`].
///
/// # Errors
///
/// Returns [`NotationError`] when the seven-byte move is malformed or inconsistent with
/// `position`.
pub fn parse_csa_move(position: &Position, line: &str) -> Result<Move, NotationError> {
    if line.len() != 7 || !line.is_ascii() {
        return Err(NotationError::InvalidMove(
            "CSA move must contain exactly seven ASCII bytes".to_owned(),
        ));
    }
    let bytes = line.as_bytes();
    let side = match bytes[0] {
        b'+' => Side::Black,
        b'-' => Side::White,
        _ => {
            return Err(NotationError::InvalidMove(
                "CSA move must start with `+` or `-`".to_owned(),
            ));
        }
    };
    if side != position.side_to_move() {
        return Err(NotationError::InvalidMove(
            "CSA move side does not match the side to move".to_owned(),
        ));
    }
    let to = parse_csa_square(bytes[3], bytes[4])?;
    let result_kind = piece_kind_from_csa_code(&line[5..7]).ok_or_else(|| {
        NotationError::InvalidMove(format!("unknown CSA piece code `{}`", &line[5..7]))
    })?;
    if bytes[1] == b'0' && bytes[2] == b'0' {
        let piece = HandPiece::from_piece_kind(result_kind).ok_or_else(|| {
            NotationError::InvalidMove("CSA drops must use an unpromoted non-king piece".to_owned())
        })?;
        return Ok(Move::Drop { piece, to });
    }
    let from = parse_csa_square(bytes[1], bytes[2])?;
    let source = position
        .piece_at(from)
        .ok_or_else(|| NotationError::InvalidMove(format!("CSA move source {from} is empty")))?;
    if source.side != side {
        return Err(NotationError::InvalidMove(
            "CSA move source belongs to the other side".to_owned(),
        ));
    }
    let promote = if result_kind == source.kind {
        false
    } else if source.kind.promoted() == Some(result_kind) {
        true
    } else {
        return Err(NotationError::InvalidMove(format!(
            "CSA result piece {} is incompatible with source {:?}",
            result_kind.csa_code(),
            source.kind
        )));
    };
    Ok(Move::Normal { from, to, promote })
}

fn parse_csa_square(file: u8, rank: u8) -> Result<Square, NotationError> {
    if !(b'1'..=b'9').contains(&file) || !(b'1'..=b'9').contains(&rank) {
        return Err(NotationError::InvalidMove(
            "CSA square must contain two digits from `1` through `9`".to_owned(),
        ));
    }
    Square::new(file - b'0', rank - b'0')
        .ok_or_else(|| NotationError::InvalidMove("CSA square lies outside the board".to_owned()))
}

fn piece_kind_from_csa_code(code: &str) -> Option<PieceKind> {
    PieceKind::ALL
        .into_iter()
        .find(|kind| kind.csa_code() == code)
}

fn csa_side_marker(side: Side) -> char {
    match side {
        Side::Black => '+',
        Side::White => '-',
    }
}

fn push_csa_board(output: &mut String, position: &Position) -> Result<(), NotationError> {
    for rank in 1..=9 {
        let mut line = format!("P{rank}");
        for file in (1..=9).rev() {
            let square = Square::new(file, rank).expect("board coordinate");
            match position.piece_at(square) {
                Some(piece) => {
                    line.push(csa_side_marker(piece.side));
                    line.push_str(piece.kind.csa_code());
                }
                None => line.push_str(" * "),
            }
        }
        push_checked_csa_line(output, &line)?;
    }
    Ok(())
}

fn push_csa_hand(
    output: &mut String,
    side: Side,
    position: &Position,
) -> Result<(), NotationError> {
    let mut line = format!("P{}", csa_side_marker(side));
    for hand_piece in HandPiece::DISPLAY_ORDER {
        for _ in 0..position.hand(side).count(hand_piece) {
            line.push_str("00");
            line.push_str(hand_piece.piece_kind().csa_code());
        }
    }
    push_checked_csa_line(output, &line)
}

fn push_checked_csa_line(output: &mut String, line: &str) -> Result<(), NotationError> {
    validate_csa_text(line, "CSA line")?;
    if line.len() > MAX_CSA_LINE_BYTES {
        return Err(NotationError::InputTooLong {
            maximum: MAX_CSA_LINE_BYTES,
        });
    }
    output.push_str(line);
    output.push('\n');
    Ok(())
}

fn validate_csa_version(version_without_prefix: &str) -> Result<(), String> {
    if version_without_prefix != "3.0" {
        return Err("only the audited CSA V3.0 record format is supported".to_owned());
    }
    Ok(())
}

fn set_csa_name(
    destination: &mut Option<String>,
    name: &str,
    line: usize,
    side_name: &str,
) -> Result<(), CsaParseError> {
    if destination.is_some() {
        return Err(invalid_csa_line(
            line,
            format!("{side_name} player name is duplicated"),
        ));
    }
    validate_csa_text(name, "player name")
        .map_err(|error| invalid_csa_line(line, error.to_string()))?;
    *destination = Some(name.to_owned());
    Ok(())
}

fn parse_csa_metadata(line: &str) -> Result<(&str, &str), &'static str> {
    let (key, value) = line
        .split_once(':')
        .ok_or("CSA metadata must contain `:`")?;
    if key.is_empty()
        || !key
            .bytes()
            .all(|byte| byte.is_ascii_uppercase() || byte.is_ascii_digit() || byte == b'_')
    {
        return Err("CSA metadata key must contain only `A`-`Z`, digits, and `_`");
    }
    if value.contains(['\r', '\n']) {
        return Err("CSA metadata value contains a line break");
    }
    Ok((key, value))
}

fn validate_csa_metadata(key: &str, value: &str) -> Result<(), NotationError> {
    if key.is_empty()
        || !key
            .bytes()
            .all(|byte| byte.is_ascii_uppercase() || byte.is_ascii_digit() || byte == b'_')
    {
        return Err(NotationError::InvalidStructure(
            "CSA metadata key must contain only `A`-`Z`, digits, and `_`".to_owned(),
        ));
    }
    validate_csa_text(value, "CSA metadata value")
}

fn validate_csa_text(text: &str, description: &str) -> Result<(), NotationError> {
    if text.contains(['\r', '\n', ',']) {
        return Err(NotationError::InvalidStructure(format!(
            "{description} contains a line break or CSA statement delimiter"
        )));
    }
    Ok(())
}

fn collect_csa_statements(input: &str) -> Result<Vec<(usize, &str)>, CsaParseError> {
    let raw_lines = input.split('\n').collect::<Vec<_>>();
    if raw_lines.len() > MAX_CSA_LINES {
        return Err(CsaParseError::TooManyLines {
            maximum: MAX_CSA_LINES,
        });
    }

    let mut statements = Vec::with_capacity(raw_lines.len());
    for (line_index, raw_line) in raw_lines.iter().enumerate() {
        let line_number = line_index + 1;
        let has_line_feed = line_index + 1 < raw_lines.len();
        let line = if has_line_feed {
            raw_line.strip_suffix('\r').unwrap_or(raw_line)
        } else {
            raw_line
        };
        if line.as_bytes().contains(&b'\r') {
            return Err(invalid_csa_line(
                line_number,
                "bare carriage return is not allowed",
            ));
        }
        if line.len() > MAX_CSA_LINE_BYTES {
            return Err(CsaParseError::LineTooLong {
                line: line_number,
                maximum: MAX_CSA_LINE_BYTES,
            });
        }
        if line.is_empty() {
            if line_index + 1 == raw_lines.len() {
                continue;
            }
            return Err(invalid_csa_line(line_number, "blank lines are not allowed"));
        }

        if !line.starts_with('\'') && line.contains(',') {
            for statement in line.split(',') {
                if statement.is_empty() {
                    return Err(invalid_csa_line(
                        line_number,
                        "CSA multi-statement line contains an empty statement",
                    ));
                }
                statements.push((line_number, statement));
            }
        } else {
            statements.push((line_number, line));
        }
        if statements.len() > MAX_CSA_LINES {
            return Err(CsaParseError::TooManyLines {
                maximum: MAX_CSA_LINES,
            });
        }
    }
    Ok(statements)
}

fn require_before_turn(
    replay_started: bool,
    line: usize,
    section: &str,
) -> Result<(), CsaParseError> {
    if replay_started {
        Err(invalid_csa_line(
            line,
            format!("{section} appears after the side-to-move line"),
        ))
    } else {
        Ok(())
    }
}

fn is_valid_csa_time(time: &str) -> bool {
    let Some((seconds, milliseconds)) = time.split_once('.') else {
        return !time.is_empty() && time.bytes().all(|byte| byte.is_ascii_digit());
    };
    !seconds.is_empty()
        && seconds.bytes().all(|byte| byte.is_ascii_digit())
        && (1..=3).contains(&milliseconds.len())
        && milliseconds.bytes().all(|byte| byte.is_ascii_digit())
        && !milliseconds.contains('.')
}

fn invalid_csa_line(line: usize, reason: impl Into<String>) -> CsaParseError {
    CsaParseError::Invalid {
        line: Some(line),
        reason: reason.into(),
    }
}

fn parse_sfen_board(field: &str) -> Result<[Option<Piece>; 81], NotationError> {
    let ranks = field.split('/').collect::<Vec<_>>();
    if ranks.len() != 9 {
        return Err(NotationError::InvalidBoard(
            "board must contain exactly nine slash-separated ranks".to_owned(),
        ));
    }
    let mut board = [None; 81];
    for (rank_index, rank) in ranks.iter().enumerate() {
        let mut file_offset = 0_usize;
        let mut promoted = false;
        let mut previous_was_empty_count = false;
        for character in rank.chars() {
            if character == '+' {
                if promoted {
                    return Err(NotationError::InvalidBoard(
                        "consecutive promotion markers are not allowed".to_owned(),
                    ));
                }
                promoted = true;
                continue;
            }
            if let Some(empty) = character.to_digit(10) {
                if promoted || previous_was_empty_count || !(1..=9).contains(&empty) {
                    return Err(NotationError::InvalidBoard(
                        "rank empty counts must be single digits `1` through `9`".to_owned(),
                    ));
                }
                file_offset = file_offset
                    .checked_add(empty as usize)
                    .ok_or_else(|| NotationError::InvalidBoard("rank width overflow".to_owned()))?;
                previous_was_empty_count = true;
            } else {
                if file_offset >= 9 {
                    return Err(NotationError::InvalidBoard(format!(
                        "rank {} contains more than nine squares",
                        rank_index + 1
                    )));
                }
                let side = if character.is_ascii_uppercase() {
                    Side::Black
                } else if character.is_ascii_lowercase() {
                    Side::White
                } else {
                    return Err(NotationError::InvalidBoard(format!(
                        "invalid piece character `{character}`"
                    )));
                };
                let mut kind = piece_kind_from_sfen_letter(character).ok_or_else(|| {
                    NotationError::InvalidBoard(format!("invalid piece character `{character}`"))
                })?;
                if promoted {
                    kind = kind.promoted().ok_or_else(|| {
                        NotationError::InvalidBoard(format!(
                            "`{character}` cannot have a promotion marker"
                        ))
                    })?;
                }
                board[rank_index * 9 + file_offset] = Some(Piece::new(side, kind));
                file_offset += 1;
                promoted = false;
                previous_was_empty_count = false;
            }
            if file_offset > 9 {
                return Err(NotationError::InvalidBoard(format!(
                    "rank {} contains more than nine squares",
                    rank_index + 1
                )));
            }
        }
        if promoted {
            return Err(NotationError::InvalidBoard(
                "promotion marker must be followed by a promotable piece".to_owned(),
            ));
        }
        if file_offset != 9 {
            return Err(NotationError::InvalidBoard(format!(
                "rank {} contains {file_offset} squares instead of nine",
                rank_index + 1
            )));
        }
    }
    Ok(board)
}

fn parse_sfen_hands(field: &str) -> Result<[Hand; 2], NotationError> {
    if field == "-" {
        return Ok([Hand::default(); 2]);
    }
    if field.is_empty() {
        return Err(NotationError::InvalidHands(
            "hands must be `-` or a piece sequence".to_owned(),
        ));
    }
    let mut hands = [Hand::default(); 2];
    let mut previous_order = None;
    let mut digits = String::new();
    for character in field.chars() {
        if character.is_ascii_digit() {
            digits.push(character);
            if digits.len() > 3 {
                return Err(NotationError::InvalidHands(
                    "hand count contains too many digits".to_owned(),
                ));
            }
            continue;
        }
        let side = if character.is_ascii_uppercase() {
            Side::Black
        } else if character.is_ascii_lowercase() {
            Side::White
        } else {
            return Err(NotationError::InvalidHands(format!(
                "invalid hand piece `{character}`"
            )));
        };
        let hand_piece = hand_piece_from_sfen_letter(character).ok_or_else(|| {
            NotationError::InvalidHands(format!("invalid hand piece `{character}`"))
        })?;
        let order = hand_display_order(side, hand_piece);
        if previous_order.is_some_and(|previous| order <= previous) {
            return Err(NotationError::InvalidHands(
                "hand pieces must be unique and in canonical `RBGSNLP` then `rbgsnlp` order"
                    .to_owned(),
            ));
        }
        previous_order = Some(order);
        let count = if digits.is_empty() {
            1
        } else {
            let parsed = digits.parse::<u16>().map_err(|_| {
                NotationError::InvalidHands("hand count is not an integer".to_owned())
            })?;
            if !(2..=u16::from(u8::MAX)).contains(&parsed) {
                return Err(NotationError::InvalidHands(
                    "explicit hand count must be between 2 and 255".to_owned(),
                ));
            }
            u8::try_from(parsed).expect("checked hand count")
        };
        hands[side.index()].set(hand_piece, count);
        digits.clear();
    }
    if !digits.is_empty() {
        return Err(NotationError::InvalidHands(
            "hand count must be followed by a piece".to_owned(),
        ));
    }
    Ok(hands)
}

fn parse_move_number(field: &str) -> Result<u32, NotationError> {
    if field.is_empty()
        || !field.bytes().all(|byte| byte.is_ascii_digit())
        || (field.len() > 1 && field.starts_with('0'))
    {
        return Err(NotationError::InvalidMoveNumber(
            "move number must be a canonical positive decimal integer".to_owned(),
        ));
    }
    let move_number = field
        .parse::<u32>()
        .map_err(|_| NotationError::InvalidMoveNumber("move number exceeds `u32`".to_owned()))?;
    if move_number == 0 {
        return Err(NotationError::InvalidMoveNumber(
            "move number must be at least one".to_owned(),
        ));
    }
    Ok(move_number)
}

fn require_exactly_one_king_per_side(board: &[Option<Piece>; 81]) -> Result<(), NotationError> {
    let mut kings = [0_u8; 2];
    for piece in board.iter().flatten() {
        if piece.kind == PieceKind::King {
            kings[piece.side.index()] += 1;
        }
    }
    if kings != [1, 1] {
        return Err(NotationError::InvalidPosition(format!(
            "expected exactly one king for each side, found Black={} White={}",
            kings[0], kings[1]
        )));
    }
    Ok(())
}

fn push_sfen_piece(output: &mut String, piece: Piece) {
    if piece.kind.is_promoted() {
        output.push('+');
    }
    let base = piece.kind.sfen_letter();
    output.push(match piece.side {
        Side::Black => base,
        Side::White => base.to_ascii_lowercase(),
    });
}

fn push_sfen_hands(output: &mut String, hands: &[Hand; 2]) {
    let mut wrote_piece = false;
    for side in [Side::Black, Side::White] {
        for hand_piece in HandPiece::DISPLAY_ORDER {
            let count = hands[side.index()].count(hand_piece);
            if count == 0 {
                continue;
            }
            wrote_piece = true;
            if count > 1 {
                write!(output, "{count}").expect("writing to String cannot fail");
            }
            let letter = hand_piece.piece_kind().sfen_letter();
            output.push(match side {
                Side::Black => letter,
                Side::White => letter.to_ascii_lowercase(),
            });
        }
    }
    if !wrote_piece {
        output.push('-');
    }
}

fn piece_kind_from_sfen_letter(character: char) -> Option<PieceKind> {
    match character.to_ascii_uppercase() {
        'P' => Some(PieceKind::Pawn),
        'L' => Some(PieceKind::Lance),
        'N' => Some(PieceKind::Knight),
        'S' => Some(PieceKind::Silver),
        'G' => Some(PieceKind::Gold),
        'B' => Some(PieceKind::Bishop),
        'R' => Some(PieceKind::Rook),
        'K' => Some(PieceKind::King),
        _ => None,
    }
}

fn hand_piece_from_sfen_letter(character: char) -> Option<HandPiece> {
    match character.to_ascii_uppercase() {
        'P' => Some(HandPiece::Pawn),
        'L' => Some(HandPiece::Lance),
        'N' => Some(HandPiece::Knight),
        'S' => Some(HandPiece::Silver),
        'G' => Some(HandPiece::Gold),
        'B' => Some(HandPiece::Bishop),
        'R' => Some(HandPiece::Rook),
        _ => None,
    }
}

fn hand_display_order(side: Side, piece: HandPiece) -> usize {
    let within_side = HandPiece::DISPLAY_ORDER
        .iter()
        .position(|candidate| *candidate == piece)
        .expect("all hand pieces are in display order");
    side.index() * HandPiece::DISPLAY_ORDER.len() + within_side
}

fn parse_usi_square(bytes: &[u8]) -> Result<Square, NotationError> {
    if bytes.len() != 2 || !(b'1'..=b'9').contains(&bytes[0]) || !(b'a'..=b'i').contains(&bytes[1])
    {
        return Err(NotationError::InvalidMove(
            "USI square must be a file `1`-`9` followed by rank `a`-`i`".to_owned(),
        ));
    }
    Square::new(bytes[0] - b'0', bytes[1] - b'a' + 1)
        .ok_or_else(|| NotationError::InvalidMove("USI square lies outside the board".to_owned()))
}
