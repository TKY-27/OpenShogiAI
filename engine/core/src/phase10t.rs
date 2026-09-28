// Native, random-lineage Phase 10T a1 inference.
//
// The wire format is intentionally separate from OSAVAL01/02.  It contains only the frozen
// 8433-feature accumulator, a 32-unit hidden layer, and direct cp/WDL heads.  No handcrafted
// evaluator, book, teacher, residual, or composite path is reachable from this evaluator.

use std::{
    error::Error,
    fmt,
    io::{self, Read},
    path::Path,
};

use sha2::{Digest, Sha256};

use crate::{HandPiece, PieceKind, Position, Side};

pub const PHASE10T_MODEL_MAGIC: &[u8; 8] = b"OSAT10A1";
pub const PHASE10T_FEATURE_COUNT: usize = 8_433;
pub const PHASE10T_ACCUMULATOR_WIDTH: usize = 128;
pub const PHASE10T_HIDDEN_WIDTH: usize = 32;
pub const PHASE10T_HEAD_COUNT: usize = 4;
const FORMAT_VERSION: u32 = 1;
const FEATURE_SCHEMA_VERSION: u32 = 1;
const HEADER_BYTES: usize = 44;
const CHECKSUM_BYTES: usize = 32;
const MAX_MODEL_BYTES: usize = 64 * 1024 * 1024;
// A position activates fewer than 512 features. With every parameter <= 1e6,
// accumulators, 256-input hidden sums and 32-input output sums remain below 5e24,
// including biases: safely inside f32 on both Python and native/Wasm hosts.
// Reject otherwise-finite artifacts that could overflow before inference.
const MAX_PARAMETER_MAGNITUDE: f32 = 1_000_000.0;
const BOARD_FEATURES: usize = 17 * 17 * 28;
const KING_OFFSET: usize = BOARD_FEATURES;
const HAND_OFFSET: usize = KING_OFFSET + 81;
const HISTORY_OFFSET: usize = HAND_OFFSET + 2 * 7 * 18;
const EXPECTED_PAYLOAD_BYTES: usize = (PHASE10T_FEATURE_COUNT * PHASE10T_ACCUMULATOR_WIDTH
    + PHASE10T_ACCUMULATOR_WIDTH
    + 2 * PHASE10T_ACCUMULATOR_WIDTH * PHASE10T_HIDDEN_WIDTH
    + PHASE10T_HIDDEN_WIDTH
    + PHASE10T_HIDDEN_WIDTH * PHASE10T_HEAD_COUNT
    + PHASE10T_HEAD_COUNT)
    * size_of::<f32>();

/// The verified identity of one a1 artifact.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Phase10TIdentity {
    pub version: u32,
    pub feature_schema_version: u32,
    pub feature_count: usize,
    pub accumulator_width: usize,
    pub hidden_width: usize,
    pub head_count: usize,
    pub seed: u64,
    pub artifact_sha256: String,
}

/// Direct score and WDL logits from the a1 head.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Phase10TInference {
    pub cp: i32,
    pub wdl_logits: [f32; 3],
}

/// A rejected or unreadable a1 artifact.
#[derive(Debug)]
pub enum Phase10TError {
    Io(io::Error),
    TooLarge(usize),
    Truncated,
    InvalidMagic,
    UnsupportedVersion(u32),
    UnsupportedFeatureSchema(u32),
    ArchitectureMismatch,
    ScaleMismatch,
    ChecksumMismatch,
    PayloadLengthMismatch,
    NonFinite(&'static str),
    UnsupportedMagnitude(&'static str),
    TrailingData,
    InvalidHistory,
}

impl fmt::Display for Phase10TError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "a1 model I/O failed: {error}"),
            Self::TooLarge(bytes) => write!(formatter, "a1 model is too large: {bytes} bytes"),
            Self::Truncated => formatter.write_str("a1 model is truncated"),
            Self::InvalidMagic => formatter.write_str("a1 model magic is invalid"),
            Self::UnsupportedVersion(version) => {
                write!(formatter, "unsupported a1 version {version}")
            }
            Self::UnsupportedFeatureSchema(version) => {
                write!(formatter, "unsupported a1 feature schema {version}")
            }
            Self::ArchitectureMismatch => {
                formatter.write_str("a1 architecture dimensions are not frozen")
            }
            Self::ScaleMismatch => formatter.write_str("a1 score scale is not one"),
            Self::ChecksumMismatch => formatter.write_str("a1 payload checksum mismatch"),
            Self::PayloadLengthMismatch => formatter.write_str("a1 payload length mismatch"),
            Self::UnsupportedMagnitude(field) => write!(
                formatter,
                "a1 {field} exceeds the supported finite arithmetic bound"
            ),
            Self::NonFinite(field) => write!(formatter, "a1 {field} is non-finite"),
            Self::InvalidHistory => formatter.write_str("a1 history or move transition is invalid"),
            Self::TrailingData => formatter.write_str("a1 model contains trailing data"),
        }
    }
}

impl Error for Phase10TError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            _ => None,
        }
    }
}

impl From<io::Error> for Phase10TError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

/// Immutable native evaluator for the Phase 10T a1 artifact.
#[derive(Clone, Debug)]
pub struct Phase10TEvaluator {
    identity: Phase10TIdentity,
    table: Vec<f32>,
    bias: [f32; PHASE10T_ACCUMULATOR_WIDTH],
    hidden_weight: Vec<f32>,
    hidden_bias: [f32; PHASE10T_HIDDEN_WIDTH],
    head_weight: Vec<f32>,
    head_bias: [f32; PHASE10T_HEAD_COUNT],
}

impl Phase10TEvaluator {
    /// Parses and verifies a complete a1 artifact.
    ///
    /// # Errors
    ///
    /// Returns an error when the artifact is truncated, oversized, has the wrong schema,
    /// contains non-finite values, or fails its payload checksum.
    pub fn from_bytes(data: &[u8]) -> Result<Self, Phase10TError> {
        if data.len() > MAX_MODEL_BYTES {
            return Err(Phase10TError::TooLarge(data.len()));
        }
        if data.len() < HEADER_BYTES + CHECKSUM_BYTES {
            return Err(Phase10TError::Truncated);
        }
        if &data[..8] != PHASE10T_MODEL_MAGIC {
            return Err(Phase10TError::InvalidMagic);
        }
        let version = read_u32(data, 8)?;
        if version != FORMAT_VERSION {
            return Err(Phase10TError::UnsupportedVersion(version));
        }
        let feature_schema_version = read_u32(data, 12)?;
        if feature_schema_version != FEATURE_SCHEMA_VERSION {
            return Err(Phase10TError::UnsupportedFeatureSchema(
                feature_schema_version,
            ));
        }
        if read_u32(data, 16)? as usize != PHASE10T_FEATURE_COUNT
            || read_u32(data, 20)? as usize != PHASE10T_ACCUMULATOR_WIDTH
            || read_u32(data, 24)? as usize != PHASE10T_HIDDEN_WIDTH
            || read_u32(data, 28)? as usize != PHASE10T_HEAD_COUNT
        {
            return Err(Phase10TError::ArchitectureMismatch);
        }
        let seed = read_u64(data, 32)?;
        let scale = read_f32(data, 40)?;
        if !scale.is_finite() {
            return Err(Phase10TError::NonFinite("score scale"));
        }
        if scale.to_bits() != 1.0_f32.to_bits() {
            return Err(Phase10TError::ScaleMismatch);
        }
        let payload_end = data.len() - CHECKSUM_BYTES;
        let payload = &data[HEADER_BYTES..payload_end];
        if payload.len() != EXPECTED_PAYLOAD_BYTES {
            return Err(Phase10TError::PayloadLengthMismatch);
        }
        let digest = Sha256::digest(payload);
        if digest.as_slice() != &data[payload_end..] {
            return Err(Phase10TError::ChecksumMismatch);
        }
        let mut cursor = 0;
        let table = read_values(
            payload,
            &mut cursor,
            PHASE10T_FEATURE_COUNT * PHASE10T_ACCUMULATOR_WIDTH,
            "table",
        )?;
        let bias_values = read_values(payload, &mut cursor, PHASE10T_ACCUMULATOR_WIDTH, "bias")?;
        let hidden_weight = read_values(
            payload,
            &mut cursor,
            2 * PHASE10T_ACCUMULATOR_WIDTH * PHASE10T_HIDDEN_WIDTH,
            "hidden weight",
        )?;
        let hidden_bias_values =
            read_values(payload, &mut cursor, PHASE10T_HIDDEN_WIDTH, "hidden bias")?;
        let head_weight = read_values(
            payload,
            &mut cursor,
            PHASE10T_HIDDEN_WIDTH * PHASE10T_HEAD_COUNT,
            "head weight",
        )?;
        let head_bias_values = read_values(payload, &mut cursor, PHASE10T_HEAD_COUNT, "head bias")?;
        if cursor != payload.len() {
            return Err(Phase10TError::TrailingData);
        }
        let bias = array_from_slice(bias_values);
        let hidden_bias = array_from_slice(hidden_bias_values);
        let head_bias = array_from_slice(head_bias_values);
        let artifact_sha256 = hex_digest(&Sha256::digest(data));
        Ok(Self {
            identity: Phase10TIdentity {
                version,
                feature_schema_version,
                feature_count: PHASE10T_FEATURE_COUNT,
                accumulator_width: PHASE10T_ACCUMULATOR_WIDTH,
                hidden_width: PHASE10T_HIDDEN_WIDTH,
                head_count: PHASE10T_HEAD_COUNT,
                seed,
                artifact_sha256,
            },
            table,
            bias,
            hidden_weight,
            hidden_bias,
            head_weight,
            head_bias,
        })
    }

    /// Reads and verifies one a1 artifact.
    ///
    /// # Errors
    ///
    /// Returns an I/O or artifact-validation error when the path cannot be read or the file is
    /// not an exact Phase 10T a1 artifact.
    pub fn load_file(path: impl AsRef<Path>) -> Result<Self, Phase10TError> {
        let mut file = crate::AnchoredFile::open_existing(path.as_ref())?;
        let identity = file.stable_identity()?;
        let maximum = u64::try_from(MAX_MODEL_BYTES).unwrap_or(u64::MAX);
        if identity.length() > maximum {
            return Err(Phase10TError::TooLarge(
                usize::try_from(identity.length()).unwrap_or(usize::MAX),
            ));
        }
        let mut bytes = Vec::new();
        file.reader()
            .take(maximum.saturating_add(1))
            .read_to_end(&mut bytes)?;
        file.verify_stable_read(&identity, u64::try_from(bytes.len()).unwrap_or(u64::MAX))?;
        Self::from_bytes(&bytes)
    }

    #[must_use]
    pub const fn identity(&self) -> &Phase10TIdentity {
        &self.identity
    }

    /// Evaluates score and WDL logits from the current side-to-move perspective.
    /// # Panics
    /// Panics only if the internal default history contract is violated.
    #[must_use]
    pub fn infer(&self, position: &Position) -> Phase10TInference {
        self.infer_accumulator(
            &self
                .accumulator(position, crate::Osaval02History::default())
                .expect("default history is valid"),
        )
    }

    /// Builds both perspectives once at a search root.
    /// # Errors
    /// Rejects inconsistent bounded history facts.
    pub fn accumulator(
        &self,
        position: &Position,
        history: crate::Osaval02History,
    ) -> Result<Phase10TAccumulator, Phase10TError> {
        validate_history(history)?;
        let mut values = [self.bias; 2];
        for perspective in [Side::Black, Side::White] {
            for feature in feature_ids(position, perspective, history) {
                self.add_feature(&mut values[perspective.index()], feature, 1.0);
            }
        }
        Ok(Phase10TAccumulator {
            position: position.clone(),
            kings: [
                position
                    .king_square(Side::Black)
                    .ok_or(Phase10TError::InvalidHistory)?,
                position
                    .king_square(Side::White)
                    .ok_or(Phase10TError::InvalidHistory)?,
            ],
            values,
            history,
        })
    }

    fn add_feature(
        &self,
        values: &mut [f32; PHASE10T_ACCUMULATOR_WIDTH],
        feature: usize,
        sign: f32,
    ) {
        for (target, value) in values.iter_mut().zip(
            &self.table
                [feature * PHASE10T_ACCUMULATOR_WIDTH..(feature + 1) * PHASE10T_ACCUMULATOR_WIDTH],
        ) {
            *target += sign * value;
        }
    }

    /// Updates a saved accumulator after a checked legal move.
    /// # Errors
    /// Rejects an illegal move, inconsistent result, or invalid history.
    pub fn update_accumulator(
        &self,
        state: &mut Phase10TAccumulator,
        movement: crate::Move,
        after: &Position,
        history: crate::Osaval02History,
    ) -> Result<Phase10TAccumulator, Phase10TError> {
        let mut checked = state.position.clone();
        checked
            .make_move(movement)
            .map_err(|_| Phase10TError::InvalidHistory)?;
        if checked != *after {
            return Err(Phase10TError::InvalidHistory);
        }
        self.update_generated_accumulator(state, movement, after, history)
    }

    /// Applies only the authoritative move's board/hand deltas; a moved king rebuilds its perspective.
    /// The returned snapshot restores the parent exactly, without inverse floating-point operations.
    /// # Errors
    /// Rejects invalid history facts or a position that is not the supplied move's result.
    pub(crate) fn update_generated_accumulator(
        &self,
        state: &mut Phase10TAccumulator,
        movement: crate::Move,
        after: &Position,
        history: crate::Osaval02History,
    ) -> Result<Phase10TAccumulator, Phase10TError> {
        validate_history(history)?;
        let before = state.clone();
        let side = before.position.side_to_move();
        if let crate::Move::Normal { from, to, .. } = movement
            && before
                .position
                .piece_at(from)
                .is_some_and(|piece| piece.kind == PieceKind::King)
        {
            state.kings[side.index()] = to;
        }
        for perspective in [Side::Black, Side::White] {
            let king = before.kings[perspective.index()];
            if state.kings[perspective.index()] != king {
                state.values[perspective.index()] = self.bias;
                for feature in feature_ids(after, perspective, history) {
                    self.add_feature(&mut state.values[perspective.index()], feature, 1.0);
                }
                continue;
            }
            let values = &mut state.values[perspective.index()];
            let mut squares = vec![movement.destination()];
            if let crate::Move::Normal { from, .. } = movement {
                squares.push(from);
            }
            for square in squares {
                if let Some(piece) = before.position.piece_at(square) {
                    self.add_feature(
                        values,
                        board_feature(square.index(), piece, king.index(), perspective),
                        -1.0,
                    );
                }
                if let Some(piece) = after.piece_at(square) {
                    self.add_feature(
                        values,
                        board_feature(square.index(), piece, king.index(), perspective),
                        1.0,
                    );
                }
            }
            for piece in HandPiece::ALL {
                let old = before.position.hand(side).count(piece);
                let new = after.hand(side).count(piece);
                for ordinal in old.min(new)..old.max(new) {
                    let feature = HAND_OFFSET
                        + ((side.index() ^ perspective.index()) * 7 + piece.index()) * 18
                        + usize::from(ordinal);
                    self.add_feature(values, feature, if old > new { -1.0 } else { 1.0 });
                }
            }
            let old = history_features(&before.position, perspective, before.history);
            let new = history_features(after, perspective, history);
            for feature in &old {
                if !new.contains(feature) {
                    self.add_feature(values, *feature, -1.0);
                }
            }
            for feature in &new {
                if !old.contains(feature) {
                    self.add_feature(values, *feature, 1.0);
                }
            }
        }
        state.position = after.clone();
        state.history = history;
        Ok(before)
    }

    /// Evaluates a saved accumulator without feature scanning or legal move generation.
    #[must_use]
    pub fn infer_accumulator(&self, state: &Phase10TAccumulator) -> Phase10TInference {
        let position = &state.position;
        let accumulators = &state.values;
        let stm = position.side_to_move().index();
        let other = 1 - stm;
        let mut hidden = [0.0_f32; PHASE10T_HIDDEN_WIDTH];
        for (hidden_index, output) in hidden.iter_mut().enumerate() {
            let row_start = hidden_index;
            let mut sum = f64::from(self.hidden_bias[hidden_index]);
            for (feature, (&stm_value, &other_value)) in accumulators[stm]
                .iter()
                .zip(accumulators[other].iter())
                .enumerate()
            {
                sum += f64::from(stm_value.max(0.0))
                    * f64::from(self.hidden_weight[feature * PHASE10T_HIDDEN_WIDTH + row_start]);
                sum += f64::from(other_value.max(0.0))
                    * f64::from(
                        self.hidden_weight[(PHASE10T_ACCUMULATOR_WIDTH + feature)
                            * PHASE10T_HIDDEN_WIDTH
                            + row_start],
                    );
            }
            *output = finite_relu(sum);
        }
        let mut output = [0.0_f32; PHASE10T_HEAD_COUNT];
        for (head, target) in output.iter_mut().enumerate() {
            let mut sum = f64::from(self.head_bias[head]);
            for (hidden_index, value) in hidden.iter().enumerate() {
                sum += f64::from(*value)
                    * f64::from(self.head_weight[hidden_index * PHASE10T_HEAD_COUNT + head]);
            }
            *target = finite_f64_to_f32(sum);
        }
        Phase10TInference {
            cp: round_and_clamp_cp(f64::from(output[0])),
            wdl_logits: [output[1], output[2], output[3]],
        }
    }

    #[must_use]
    pub fn evaluate(&self, position: &Position) -> i32 {
        self.infer(position).cp
    }
}

fn feature_ids(
    position: &Position,
    perspective: Side,
    history: crate::Osaval02History,
) -> Vec<usize> {
    let king = position
        .board()
        .iter()
        .enumerate()
        .find_map(|(index, piece)| {
            piece
                .filter(|piece| piece.side == perspective && piece.kind == PieceKind::King)
                .map(|_| index)
        })
        .expect("validated Position has one king per side");
    let oriented_king = if perspective == Side::White {
        80 - king
    } else {
        king
    };
    let king_rank = oriented_king / 9;
    let king_column = oriented_king % 9;
    let mut features = Vec::with_capacity(64);
    features.push(KING_OFFSET + oriented_king);
    for (square_index, piece) in position.board().iter().enumerate() {
        let Some(piece) = piece else {
            continue;
        };
        let oriented = if perspective == Side::White {
            80 - square_index
        } else {
            square_index
        };
        let rank = oriented / 9;
        let column = oriented % 9;
        let relative = (isize::try_from(rank).expect("board rank fits")
            - isize::try_from(king_rank).expect("king rank fits")
            + 8)
            * 17
            + isize::try_from(column).expect("board column fits")
            - isize::try_from(king_column).expect("king column fits")
            + 8;
        debug_assert!((0..289).contains(&relative));
        let owner = piece.side.index() ^ perspective.index();
        features.push(
            usize::try_from(relative).expect("relative feature coordinate is non-negative") * 28
                + owner * 14
                + piece.kind.index(),
        );
    }
    for side in [Side::Black, Side::White] {
        for hand_piece in HandPiece::ALL {
            let count = position.hand(side).count(hand_piece);
            for ordinal in 0..count {
                features.push(
                    HAND_OFFSET
                        + ((side.index() ^ perspective.index()) * 7 + hand_piece.index()) * 18
                        + usize::from(ordinal),
                );
            }
        }
    }
    features.extend(history_features(position, perspective, history));
    features.sort_unstable();
    features.dedup();
    features
}

fn read_u32(data: &[u8], offset: usize) -> Result<u32, Phase10TError> {
    let bytes = data
        .get(offset..offset + 4)
        .ok_or(Phase10TError::Truncated)?;
    Ok(u32::from_le_bytes(
        bytes.try_into().map_err(|_| Phase10TError::Truncated)?,
    ))
}

fn read_u64(data: &[u8], offset: usize) -> Result<u64, Phase10TError> {
    let bytes = data
        .get(offset..offset + 8)
        .ok_or(Phase10TError::Truncated)?;
    Ok(u64::from_le_bytes(
        bytes.try_into().map_err(|_| Phase10TError::Truncated)?,
    ))
}

fn read_f32(data: &[u8], offset: usize) -> Result<f32, Phase10TError> {
    Ok(f32::from_bits(read_u32(data, offset)?))
}

fn read_values(
    payload: &[u8],
    cursor: &mut usize,
    count: usize,
    field: &'static str,
) -> Result<Vec<f32>, Phase10TError> {
    let bytes = count
        .checked_mul(4)
        .ok_or(Phase10TError::PayloadLengthMismatch)?;
    let end = cursor
        .checked_add(bytes)
        .ok_or(Phase10TError::PayloadLengthMismatch)?;
    let range = payload.get(*cursor..end).ok_or(Phase10TError::Truncated)?;
    let mut values = Vec::with_capacity(count);
    for chunk in range.as_chunks::<4>().0 {
        let value = f32::from_le_bytes(*chunk);
        if !value.is_finite() {
            return Err(Phase10TError::NonFinite(field));
        }
        if value.abs() > MAX_PARAMETER_MAGNITUDE {
            return Err(Phase10TError::UnsupportedMagnitude(field));
        }
        values.push(value);
    }
    *cursor = end;
    Ok(values)
}

fn array_from_slice<const N: usize>(values: Vec<f32>) -> [f32; N] {
    values.try_into().expect("validated tensor dimension")
}

fn finite_relu(value: f64) -> f32 {
    let maximum = f64::from(f32::MAX);
    #[expect(
        clippy::cast_possible_truncation,
        reason = "value is clamped to f32 range"
    )]
    {
        value.max(0.0).min(maximum) as f32
    }
}

fn finite_f64_to_f32(value: f64) -> f32 {
    let maximum = f64::from(f32::MAX);
    #[expect(
        clippy::cast_possible_truncation,
        reason = "value is clamped to f32 range"
    )]
    {
        value.clamp(-maximum, maximum) as f32
    }
}

fn round_and_clamp_cp(value: f64) -> i32 {
    let maximum = f64::from(crate::MATE_THRESHOLD - 1);
    #[expect(
        clippy::cast_possible_truncation,
        reason = "value is bounded to the cp namespace"
    )]
    {
        value.clamp(-maximum, maximum).round() as i32
    }
}

fn hex_digest(digest: &[u8]) -> String {
    let mut value = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use fmt::Write as _;
        write!(&mut value, "{byte:02x}").expect("String formatting cannot fail");
    }
    value
}

/// Exact saved accumulator state; unmake replaces the state with its parent snapshot.
#[derive(Clone, Debug, PartialEq)]
pub struct Phase10TAccumulator {
    position: Position,
    kings: [crate::Square; 2],
    values: [[f32; PHASE10T_ACCUMULATOR_WIDTH]; 2],
    history: crate::Osaval02History,
}

fn validate_history(history: crate::Osaval02History) -> Result<(), Phase10TError> {
    if !(1..=4).contains(&history.repetition_count)
        || (!history.available && history != crate::Osaval02History::default())
        || (history.continuous_check_by_us && history.continuous_check_by_them)
    {
        return Err(Phase10TError::InvalidHistory);
    }
    Ok(())
}

fn history_features(
    position: &Position,
    perspective: Side,
    history: crate::Osaval02History,
) -> Vec<usize> {
    let mut result = vec![
        HISTORY_OFFSET + usize::from(history.available),
        HISTORY_OFFSET + 2 + usize::from(history.repetition_count) - 1,
    ];
    let (own, other) = if perspective == position.side_to_move() {
        (
            history.continuous_check_by_us,
            history.continuous_check_by_them,
        )
    } else {
        (
            history.continuous_check_by_them,
            history.continuous_check_by_us,
        )
    };
    if own {
        result.push(HISTORY_OFFSET + 6);
    }
    if other {
        result.push(HISTORY_OFFSET + 7);
    }
    result
}

fn board_feature(square: usize, piece: crate::Piece, king: usize, perspective: Side) -> usize {
    let orient = |index| {
        if perspective == Side::White {
            80 - index
        } else {
            index
        }
    };
    let square = orient(square);
    let king = orient(king);
    let relative = (square / 9 + 8 - king / 9) * 17 + square % 9 + 8 - king % 9;
    relative * 28 + (piece.side.index() ^ perspective.index()) * 14 + piece.kind.index()
}

#[cfg(test)]
mod tests {
    use sha2::{Digest, Sha256};

    use super::{
        CHECKSUM_BYTES, EXPECTED_PAYLOAD_BYTES, FEATURE_SCHEMA_VERSION, FORMAT_VERSION,
        HEADER_BYTES, PHASE10T_ACCUMULATOR_WIDTH, PHASE10T_FEATURE_COUNT, PHASE10T_HEAD_COUNT,
        PHASE10T_HIDDEN_WIDTH, PHASE10T_MODEL_MAGIC, Phase10TEvaluator,
    };

    #[test]
    fn parser_rejects_wrong_magic() {
        let mut bytes = Phase10TModelFixture::bytes();
        bytes[0] ^= 1;
        assert!(matches!(
            Phase10TEvaluator::from_bytes(&bytes),
            Err(super::Phase10TError::InvalidMagic)
        ));
    }

    #[test]
    fn parser_accepts_random_fixture_and_evaluates() {
        let evaluator = Phase10TEvaluator::from_bytes(&Phase10TModelFixture::bytes()).unwrap();
        let position = crate::Position::startpos();
        let result = evaluator.infer(&position);
        assert!(result.cp.abs() < crate::MATE_THRESHOLD);
        assert_eq!(&Phase10TModelFixture::bytes()[..8], PHASE10T_MODEL_MAGIC);
    }

    #[test]
    fn a1_move_local_full_and_snapshot_unmake_parity() {
        let mut evaluator = Phase10TEvaluator::from_bytes(&Phase10TModelFixture::bytes()).unwrap();
        for (index, value) in evaluator.table.iter_mut().enumerate() {
            *value = f32::from(u16::try_from(index % 97).unwrap()) / 9700.0 - 0.005;
        }
        for value in &mut evaluator.hidden_weight {
            *value = 0.01;
        }
        for value in &mut evaluator.head_weight {
            *value = 0.1;
        }
        let cases = [
            ("4k4/9/9/9/4P4/9/9/9/4K4 b - 1", "5e5d"),
            ("4k4/9/9/4p4/4P4/9/9/9/4K4 b - 1", "5e5d"),
            ("4k4/9/9/4P4/9/9/9/9/4K4 b - 1", "5d5c+"),
            ("4k4/9/9/9/9/9/9/9/4K4 b P 1", "P*5e"),
            ("4k4/9/9/9/4P4/9/9/9/4K4 b - 1", "5i5h"),
        ];
        for (sfen, usi) in cases {
            let mut position = crate::parse_sfen(sfen).unwrap();
            let mut state = evaluator
                .accumulator(&position, crate::Osaval02History::default())
                .unwrap();
            let original = state.clone();
            let movement = crate::parse_usi_move(usi).unwrap();
            let undo = position.make_move(movement).unwrap();
            let history = crate::Osaval02History {
                available: true,
                repetition_count: 2,
                continuous_check_by_us: false,
                continuous_check_by_them: true,
            };
            let previous = evaluator
                .update_accumulator(&mut state, movement, &position, history)
                .unwrap();
            let full = evaluator.accumulator(&position, history).unwrap();
            for (incremental, rebuilt) in state
                .values
                .iter()
                .flatten()
                .zip(full.values.iter().flatten())
            {
                assert!(
                    (incremental - rebuilt).abs() < 1e-6,
                    "{usi}: {incremental} vs {rebuilt}"
                );
            }
            assert_eq!(
                evaluator.infer_accumulator(&state).cp,
                evaluator.infer_accumulator(&full).cp
            );
            position.unmake_move(undo);
            state = previous;
            assert_eq!(state, original);
            assert_eq!(state.position, position);
        }
    }

    #[test]
    fn a1_input_relu_precedes_hidden_projection() {
        let mut evaluator = Phase10TEvaluator::from_bytes(&Phase10TModelFixture::bytes()).unwrap();
        evaluator.bias.fill(-1.0);
        evaluator.hidden_weight.fill(-1.0);
        evaluator.head_weight.fill(1.0);
        assert_eq!(evaluator.evaluate(&crate::Position::startpos()), 0);
    }

    #[test]
    fn a1_legal_search_has_only_learned_calls_and_validated_hash() {
        let evaluator = std::sync::Arc::new(
            Phase10TEvaluator::from_bytes(&Phase10TModelFixture::bytes()).unwrap(),
        );
        assert!(
            crate::SearchEngine::with_phase10t(
                crate::SearchConfig::default(),
                evaluator.clone(),
                &"0".repeat(64)
            )
            .is_err()
        );
        let hash = evaluator.identity().artifact_sha256.clone();
        let mut engine =
            crate::SearchEngine::with_phase10t(crate::SearchConfig::default(), evaluator, &hash)
                .unwrap();
        let position = crate::Position::startpos();
        let result = engine.search(
            &position,
            crate::SearchLimits {
                max_depth: 1,
                max_nodes: Some(64),
                movetime: None,
            },
            &crate::CancellationToken::new(),
        );
        assert!(position.legal_moves().contains(&result.best_move.unwrap()));
        assert!(
            engine
                .runtime_proof(result.stats, hash)
                .valid_pure_learned()
        );
    }

    #[test]
    fn a1_rejects_schema_nonfinite_checksum_and_history() {
        let bytes = Phase10TModelFixture::bytes();
        let mut schema = bytes.clone();
        schema[12] = 2;
        assert!(matches!(
            Phase10TEvaluator::from_bytes(&schema),
            Err(super::Phase10TError::UnsupportedFeatureSchema(2))
        ));
        let mut corrupt = bytes.clone();
        corrupt[HEADER_BYTES] ^= 1;
        assert!(matches!(
            Phase10TEvaluator::from_bytes(&corrupt),
            Err(super::Phase10TError::ChecksumMismatch)
        ));
        let mut nonfinite = bytes.clone();
        nonfinite[HEADER_BYTES..HEADER_BYTES + 4].copy_from_slice(&f32::NAN.to_le_bytes());
        let end = nonfinite.len() - CHECKSUM_BYTES;
        let checksum = Sha256::digest(&nonfinite[HEADER_BYTES..end]);
        nonfinite[end..].copy_from_slice(&checksum);
        assert!(matches!(
            Phase10TEvaluator::from_bytes(&nonfinite),
            Err(super::Phase10TError::NonFinite(_))
        ));
        let mut overflowing = bytes.clone();
        overflowing[HEADER_BYTES..HEADER_BYTES + 4].copy_from_slice(&f32::MAX.to_le_bytes());
        let end = overflowing.len() - CHECKSUM_BYTES;
        let checksum = Sha256::digest(&overflowing[HEADER_BYTES..end]);
        overflowing[end..].copy_from_slice(&checksum);
        assert!(matches!(
            Phase10TEvaluator::from_bytes(&overflowing),
            Err(super::Phase10TError::UnsupportedMagnitude(_))
        ));
        let evaluator = Phase10TEvaluator::from_bytes(&bytes).unwrap();
        assert!(
            evaluator
                .accumulator(
                    &crate::Position::startpos(),
                    crate::Osaval02History {
                        repetition_count: 2,
                        ..Default::default()
                    }
                )
                .is_err()
        );
    }

    #[test]
    fn a1_actual_and_prospective_fourfold_history_is_exact() {
        let evaluator = std::sync::Arc::new(
            Phase10TEvaluator::from_bytes(&Phase10TModelFixture::bytes()).unwrap(),
        );
        let hash = evaluator.identity().artifact_sha256.clone();
        let initial = crate::parse_sfen("4k4/9/9/9/9/9/9/9/4K4 b - 1").unwrap();
        let cycle: Vec<_> = ["5i6i", "5a6a", "6i5i", "6a5a"]
            .iter()
            .map(|value| crate::parse_usi_move(value).unwrap())
            .collect();
        let moves = cycle.repeat(3);
        let mut position = initial.clone();
        for movement in &moves {
            position.make_move(*movement).unwrap();
        }
        let mut engine =
            crate::SearchEngine::with_phase10t(crate::SearchConfig::default(), evaluator, &hash)
                .unwrap();
        engine.set_phase10t_history(&initial, &moves).unwrap();
        let result = engine.search(
            &position,
            crate::SearchLimits {
                max_depth: 1,
                max_nodes: Some(32),
                movetime: None,
            },
            &crate::CancellationToken::new(),
        );
        assert_eq!(result.best_move, None);
        assert_eq!(result.score, 0);
        assert_eq!(result.stats.learned_eval_calls, 0);
        // The same final transition remains exact when supplied as a search-tree edge.
        let evaluator = Phase10TEvaluator::from_bytes(&Phase10TModelFixture::bytes()).unwrap();
        let mut state = evaluator
            .accumulator(&initial, crate::Osaval02History::default())
            .unwrap();
        let mut position = initial.clone();
        let mut snapshots = Vec::new();
        for (index, movement) in moves.iter().enumerate() {
            position.make_move(*movement).unwrap();
            snapshots.push(
                evaluator
                    .update_accumulator(
                        &mut state,
                        *movement,
                        &position,
                        crate::Osaval02History {
                            available: true,
                            repetition_count: u8::try_from(index / 4 + 1).unwrap(),
                            ..Default::default()
                        },
                    )
                    .unwrap(),
            );
        }
        for before in snapshots.into_iter().rev() {
            state = before;
        }
        assert_eq!(state.position, initial);
    }

    #[test]
    fn a1_loader_rejects_missing_and_non_regular_paths() {
        assert!(Phase10TEvaluator::load_file("/nonexistent/open-shogi-a1-model.osat10a1").is_err());
        assert!(Phase10TEvaluator::load_file(std::env::temp_dir()).is_err());
    }

    #[test]
    fn a1_explicit_history_root_mismatch_fails_without_fallback() {
        let model = std::sync::Arc::new(
            Phase10TEvaluator::from_bytes(&Phase10TModelFixture::bytes()).unwrap(),
        );
        let hash = model.identity().artifact_sha256.clone();
        let mut engine =
            crate::SearchEngine::with_phase10t(crate::SearchConfig::default(), model, &hash)
                .unwrap();
        let root = crate::Position::startpos();
        engine
            .set_pure_history(&root, &[crate::parse_usi_move("7g7f").unwrap()])
            .unwrap();
        let result = engine.search(
            &root,
            crate::SearchLimits {
                max_depth: 1,
                max_nodes: Some(8),
                movetime: None,
            },
            &crate::CancellationToken::new(),
        );
        assert_eq!(
            result.termination,
            crate::SearchTermination::EvaluationError
        );
        assert_eq!(result.best_move, None);
        assert_eq!(result.stats.learned_eval_calls, 0);
        assert_eq!(result.stats.fallback_count, 0);
    }

    struct Phase10TModelFixture;

    impl Phase10TModelFixture {
        fn bytes() -> Vec<u8> {
            let payload = vec![0_u8; EXPECTED_PAYLOAD_BYTES];
            let mut data = Vec::with_capacity(HEADER_BYTES + payload.len() + CHECKSUM_BYTES);
            data.extend_from_slice(PHASE10T_MODEL_MAGIC);
            data.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
            data.extend_from_slice(&FEATURE_SCHEMA_VERSION.to_le_bytes());
            data.extend_from_slice(
                &u32::try_from(PHASE10T_FEATURE_COUNT)
                    .expect("frozen feature count fits u32")
                    .to_le_bytes(),
            );
            data.extend_from_slice(
                &u32::try_from(PHASE10T_ACCUMULATOR_WIDTH)
                    .expect("frozen accumulator width fits u32")
                    .to_le_bytes(),
            );
            data.extend_from_slice(
                &u32::try_from(PHASE10T_HIDDEN_WIDTH)
                    .expect("frozen hidden width fits u32")
                    .to_le_bytes(),
            );
            data.extend_from_slice(
                &u32::try_from(PHASE10T_HEAD_COUNT)
                    .expect("frozen head count fits u32")
                    .to_le_bytes(),
            );
            data.extend_from_slice(&0_u64.to_le_bytes());
            data.extend_from_slice(&1.0_f32.to_le_bytes());
            data.extend_from_slice(&payload);
            data.extend_from_slice(&Sha256::digest(&payload));
            data
        }
    }
}
