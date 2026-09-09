// Native, random-lineage Phase 10V OSAVAL03 inference.
//
// The wire format is intentionally separate from OSAVAL01/02.  It contains only the frozen
// 8427-feature accumulator, a 16-unit clipped hidden layer, and direct cp/WDL heads.  No handcrafted
// evaluator, book, teacher, residual, or composite path is reachable from this evaluator.

use std::{
    error::Error,
    fmt,
    io::{self, Read},
    path::Path,
};

use sha2::{Digest, Sha256};

use crate::{HandPiece, PieceKind, Position, Side};

pub const PHASE10V_MODEL_MAGIC: &[u8; 8] = b"OSAVAL03";
pub const PHASE10V_FEATURE_COUNT: usize = 8_427;
pub const PHASE10V_ACCUMULATOR_WIDTH: usize = 256;
pub const PHASE10V_HIDDEN_WIDTH: usize = 16;
pub const PHASE10V_HEAD_COUNT: usize = 4;
const FORMAT_VERSION: u32 = 3;
const FEATURE_SCHEMA_VERSION: u32 = 1;
const HEADER_BYTES: usize = 44;
const CHECKSUM_BYTES: usize = 32;
const MAX_MODEL_BYTES: usize = 64 * 1024 * 1024;
// Bounded parameters and clipped activations keep all supported dimensions finite.
const MAX_PARAMETER_MAGNITUDE: f32 = 1_000_000.0;
// Table and bias are exact Q20 values. At most 81 active terms plus a bias, even
// transient move deltas, stay far below 2^53 when scaled by 2^20. Therefore f64
// additions/subtractions are exact and full/incremental accumulation is identical.
const ACCUMULATOR_SCALE: f64 = 1_048_576.0;
const BOARD_FEATURES: usize = 17 * 17 * 28;
const KING_OFFSET: usize = BOARD_FEATURES;
const HAND_OFFSET: usize = KING_OFFSET + 81;
const STM_OFFSET: usize = HAND_OFFSET + 2 * 7 * 18;
fn expected_payload_bytes(width: usize) -> usize {
    (PHASE10V_FEATURE_COUNT * width
        + width
        + 3 * width * PHASE10V_HIDDEN_WIDTH
        + PHASE10V_HIDDEN_WIDTH
        + PHASE10V_HIDDEN_WIDTH * PHASE10V_HEAD_COUNT
        + PHASE10V_HEAD_COUNT)
        * 4
}

/// The verified identity of one OSAVAL03 artifact.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Phase10VIdentity {
    pub version: u32,
    pub feature_schema_version: u32,
    pub feature_count: usize,
    pub accumulator_width: usize,
    pub hidden_width: usize,
    pub head_count: usize,
    pub seed: u64,
    pub artifact_sha256: String,
}

/// Direct score and WDL logits from the OSAVAL03 head.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Phase10VInference {
    pub cp: i32,
    pub wdl_logits: [f32; 3],
}

/// A rejected or unreadable OSAVAL03 artifact.
#[derive(Debug)]
pub enum Phase10VError {
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
    UnsupportedAccumulatorGrid(&'static str),
    IncompatibleAccumulator,
    TrailingData,
    InvalidTransition,
}

impl fmt::Display for Phase10VError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "OSAVAL03 model I/O failed: {error}"),
            Self::TooLarge(bytes) => {
                write!(formatter, "OSAVAL03 model is too large: {bytes} bytes")
            }
            Self::Truncated => formatter.write_str("OSAVAL03 model is truncated"),
            Self::InvalidMagic => formatter.write_str("OSAVAL03 model magic is invalid"),
            Self::UnsupportedVersion(version) => {
                write!(formatter, "unsupported OSAVAL03 version {version}")
            }
            Self::UnsupportedFeatureSchema(version) => {
                write!(formatter, "unsupported OSAVAL03 feature schema {version}")
            }
            Self::ArchitectureMismatch => {
                formatter.write_str("OSAVAL03 architecture dimensions are not frozen")
            }
            Self::ScaleMismatch => formatter.write_str("OSAVAL03 score scale is not one"),
            Self::ChecksumMismatch => formatter.write_str("OSAVAL03 payload checksum mismatch"),
            Self::PayloadLengthMismatch => formatter.write_str("OSAVAL03 payload length mismatch"),
            Self::UnsupportedMagnitude(field) => write!(
                formatter,
                "OSAVAL03 {field} exceeds the supported finite arithmetic bound"
            ),
            Self::UnsupportedAccumulatorGrid(field) => {
                write!(formatter, "OSAVAL03 {field} must use exact Q20 values")
            }
            Self::IncompatibleAccumulator => {
                formatter.write_str("OSAVAL03 accumulator belongs to an incompatible model")
            }
            Self::NonFinite(field) => write!(formatter, "OSAVAL03 {field} is non-finite"),
            Self::InvalidTransition => {
                formatter.write_str("OSAVAL03 position or move transition is invalid")
            }
            Self::TrailingData => formatter.write_str("OSAVAL03 model contains trailing data"),
        }
    }
}

impl Error for Phase10VError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            _ => None,
        }
    }
}

impl From<io::Error> for Phase10VError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

/// Immutable native evaluator for the Phase 10V OSAVAL03 artifact.
#[derive(Clone, Debug)]
pub struct Phase10VEvaluator {
    identity: Phase10VIdentity,
    table: Vec<f32>,
    bias: Vec<f64>,
    artifact_digest: [u8; 32],
    hidden_weight: Vec<f32>,
    hidden_bias: [f32; PHASE10V_HIDDEN_WIDTH],
    head_weight: Vec<f32>,
    head_bias: [f32; PHASE10V_HEAD_COUNT],
}

impl Phase10VEvaluator {
    /// Parses and verifies a complete OSAVAL03 artifact.
    ///
    /// # Errors
    ///
    /// Returns an error when the artifact is truncated, oversized, has the wrong schema,
    /// contains non-finite values, or fails its payload checksum.
    pub fn from_bytes(data: &[u8]) -> Result<Self, Phase10VError> {
        if data.len() > MAX_MODEL_BYTES {
            return Err(Phase10VError::TooLarge(data.len()));
        }
        if data.len() < HEADER_BYTES + CHECKSUM_BYTES {
            return Err(Phase10VError::Truncated);
        }
        if &data[..8] != PHASE10V_MODEL_MAGIC {
            return Err(Phase10VError::InvalidMagic);
        }
        let version = read_u32(data, 8)?;
        if version != FORMAT_VERSION {
            return Err(Phase10VError::UnsupportedVersion(version));
        }
        let feature_schema_version = read_u32(data, 12)?;
        if feature_schema_version != FEATURE_SCHEMA_VERSION {
            return Err(Phase10VError::UnsupportedFeatureSchema(
                feature_schema_version,
            ));
        }
        if read_u32(data, 16)? as usize != PHASE10V_FEATURE_COUNT
            || ![256, 512].contains(&(read_u32(data, 20)? as usize))
            || read_u32(data, 24)? as usize != PHASE10V_HIDDEN_WIDTH
            || read_u32(data, 28)? as usize != PHASE10V_HEAD_COUNT
        {
            return Err(Phase10VError::ArchitectureMismatch);
        }
        let width = read_u32(data, 20)? as usize;
        let seed = read_u64(data, 32)?;
        let scale = read_f32(data, 40)?;
        if !scale.is_finite() {
            return Err(Phase10VError::NonFinite("score scale"));
        }
        if scale.to_bits() != 1.0_f32.to_bits() {
            return Err(Phase10VError::ScaleMismatch);
        }
        let payload_end = data.len() - CHECKSUM_BYTES;
        let payload = &data[HEADER_BYTES..payload_end];
        if payload.len() != expected_payload_bytes(width) {
            return Err(Phase10VError::PayloadLengthMismatch);
        }
        let digest = Sha256::digest(&data[..payload_end]);
        if digest.as_slice() != &data[payload_end..] {
            return Err(Phase10VError::ChecksumMismatch);
        }
        let mut cursor = 0;
        let table = read_values(
            payload,
            &mut cursor,
            PHASE10V_FEATURE_COUNT * width,
            "table",
        )?;
        let bias_values = read_values(payload, &mut cursor, width, "bias")?;
        let hidden_weight = read_values(
            payload,
            &mut cursor,
            3 * width * PHASE10V_HIDDEN_WIDTH,
            "hidden weight",
        )?;
        let hidden_bias_values =
            read_values(payload, &mut cursor, PHASE10V_HIDDEN_WIDTH, "hidden bias")?;
        let head_weight = read_values(
            payload,
            &mut cursor,
            PHASE10V_HIDDEN_WIDTH * PHASE10V_HEAD_COUNT,
            "head weight",
        )?;
        let head_bias_values = read_values(payload, &mut cursor, PHASE10V_HEAD_COUNT, "head bias")?;
        if cursor != payload.len() {
            return Err(Phase10VError::TrailingData);
        }
        validate_accumulator_grid(&table, "table")?;
        validate_accumulator_grid(&bias_values, "bias")?;
        let bias = bias_values.into_iter().map(f64::from).collect();
        let hidden_bias = array_from_slice(hidden_bias_values);
        let head_bias = array_from_slice(head_bias_values);
        let artifact_digest: [u8; 32] = Sha256::digest(data).into();
        let artifact_sha256 = hex_digest(&artifact_digest);
        Ok(Self {
            identity: Phase10VIdentity {
                version,
                feature_schema_version,
                feature_count: PHASE10V_FEATURE_COUNT,
                accumulator_width: width,
                hidden_width: PHASE10V_HIDDEN_WIDTH,
                head_count: PHASE10V_HEAD_COUNT,
                seed,
                artifact_sha256,
            },
            table,
            bias,
            artifact_digest,
            hidden_weight,
            hidden_bias,
            head_weight,
            head_bias,
        })
    }

    /// Reads and verifies one OSAVAL03 artifact.
    ///
    /// # Errors
    ///
    /// Returns an I/O or artifact-validation error when the path cannot be read or the file is
    /// not an exact Phase 10V OSAVAL03 artifact.
    pub fn load_file(path: impl AsRef<Path>) -> Result<Self, Phase10VError> {
        let mut file = crate::AnchoredFile::open_existing(path.as_ref())?;
        let identity = file.stable_identity()?;
        let maximum = u64::try_from(MAX_MODEL_BYTES).unwrap_or(u64::MAX);
        if identity.length() > maximum {
            return Err(Phase10VError::TooLarge(
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
    pub const fn identity(&self) -> &Phase10VIdentity {
        &self.identity
    }

    /// Evaluates score and WDL logits from the current side-to-move perspective.
    /// # Panics
    /// Panics only if the internal position contract is violated.
    #[must_use]
    pub fn infer(&self, position: &Position) -> Phase10VInference {
        self.infer_owned_accumulator(&self.accumulator(position).expect("position is valid"))
    }

    /// Builds both perspectives once at a search root.
    /// # Errors
    /// Rejects positions without both kings.
    pub fn accumulator(&self, position: &Position) -> Result<Phase10VAccumulator, Phase10VError> {
        let kings = [
            position
                .king_square(Side::Black)
                .ok_or(Phase10VError::InvalidTransition)?,
            position
                .king_square(Side::White)
                .ok_or(Phase10VError::InvalidTransition)?,
        ];
        let mut values = [self.bias.clone(), self.bias.clone()];
        for perspective in [Side::Black, Side::White] {
            for feature in feature_ids(position, perspective) {
                self.add_feature(&mut values[perspective.index()], feature, 1.0);
            }
        }
        Ok(Phase10VAccumulator {
            position: position.clone(),
            artifact_digest: self.artifact_digest,
            kings,
            values,
        })
    }

    pub(crate) fn requires_refresh(state: &Phase10VAccumulator, movement: crate::Move) -> bool {
        match movement {
            crate::Move::Normal { from, .. } => state
                .position
                .piece_at(from)
                .is_some_and(|p| p.kind == PieceKind::King),
            crate::Move::Drop { .. } => false,
        }
    }

    fn add_feature(&self, values: &mut [f64], feature: usize, sign: f64) {
        for (target, value) in values.iter_mut().zip(
            &self.table[feature * self.identity.accumulator_width
                ..(feature + 1) * self.identity.accumulator_width],
        ) {
            *target += sign * f64::from(*value);
        }
    }

    /// Updates a saved accumulator after a checked legal move.
    /// # Errors
    /// Rejects an illegal move, inconsistent result, or invalid transition.
    pub fn update_accumulator(
        &self,
        state: &mut Phase10VAccumulator,
        movement: crate::Move,
        after: &Position,
    ) -> Result<Phase10VAccumulator, Phase10VError> {
        self.validate_accumulator(state)?;
        let mut checked = state.position.clone();
        checked
            .make_move(movement)
            .map_err(|_| Phase10VError::InvalidTransition)?;
        if checked != *after {
            return Err(Phase10VError::InvalidTransition);
        }
        Ok(self.update_generated_accumulator(state, movement, after))
    }

    /// Applies only the authoritative move's board/hand deltas; a moved king rebuilds its perspective.
    /// The returned snapshot restores the parent exactly, without inverse floating-point operations.
    /// Only the search may call this with its already validated generated move.
    pub(crate) fn update_generated_accumulator(
        &self,
        state: &mut Phase10VAccumulator,
        movement: crate::Move,
        after: &Position,
    ) -> Phase10VAccumulator {
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
                state.values[perspective.index()].clone_from(&self.bias);
                for feature in feature_ids(after, perspective) {
                    self.add_feature(&mut state.values[perspective.index()], feature, 1.0);
                }
                continue;
            }
            let values = &mut state.values[perspective.index()];
            let origin = match movement {
                crate::Move::Normal { from, .. } => Some(from),
                crate::Move::Drop { .. } => None,
            };
            for square in [Some(movement.destination()), origin].into_iter().flatten() {
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
            let old = STM_OFFSET + usize::from(before.position.side_to_move() != perspective);
            let new = STM_OFFSET + usize::from(after.side_to_move() != perspective);
            self.add_feature(values, old, -1.0);
            self.add_feature(values, new, 1.0);
        }
        state.position = after.clone();
        before
    }

    /// Evaluates a saved accumulator without feature scanning or legal move generation.
    /// # Errors
    /// Rejects an accumulator produced by different model bytes or dimensions.
    pub fn infer_accumulator(
        &self,
        state: &Phase10VAccumulator,
    ) -> Result<Phase10VInference, Phase10VError> {
        self.validate_accumulator(state)?;
        Ok(self.infer_owned_accumulator(state))
    }

    fn validate_accumulator(&self, state: &Phase10VAccumulator) -> Result<(), Phase10VError> {
        if state.artifact_digest != self.artifact_digest
            || state
                .values
                .iter()
                .any(|values| values.len() != self.identity.accumulator_width)
        {
            return Err(Phase10VError::IncompatibleAccumulator);
        }
        Ok(())
    }

    /// Search owns the model/state pairing and avoids repeated identity checks per node.
    pub(crate) fn infer_owned_accumulator(&self, state: &Phase10VAccumulator) -> Phase10VInference {
        let position = &state.position;
        let accumulators = &state.values;
        let stm = position.side_to_move().index();
        let other = 1 - stm;
        let mut hidden = [0.0_f32; PHASE10V_HIDDEN_WIDTH];
        for (hidden_index, output) in hidden.iter_mut().enumerate() {
            let row_start = hidden_index;
            let mut sum = f64::from(self.hidden_bias[hidden_index]);
            for (feature, (&stm_value, &other_value)) in accumulators[stm]
                .iter()
                .zip(accumulators[other].iter())
                .enumerate()
            {
                // The learned layers consume f32 clipped inputs in every runtime.
                let stm_value = finite_relu(stm_value);
                let other_value = finite_relu(other_value);
                sum += f64::from(stm_value.clamp(0.0, 1.0))
                    * f64::from(self.hidden_weight[feature * PHASE10V_HIDDEN_WIDTH + row_start]);
                sum += f64::from(other_value.clamp(0.0, 1.0))
                    * f64::from(
                        self.hidden_weight[(self.identity.accumulator_width + feature)
                            * PHASE10V_HIDDEN_WIDTH
                            + row_start],
                    );
                sum += f64::from(stm_value.clamp(0.0, 1.0) * other_value.clamp(0.0, 1.0))
                    * f64::from(
                        self.hidden_weight[(2 * self.identity.accumulator_width + feature)
                            * PHASE10V_HIDDEN_WIDTH
                            + row_start],
                    );
            }
            *output = finite_relu(sum);
        }
        let mut output = [0.0_f32; PHASE10V_HEAD_COUNT];
        for (head, target) in output.iter_mut().enumerate() {
            let mut sum = f64::from(self.head_bias[head]);
            for (hidden_index, value) in hidden.iter().enumerate() {
                sum += f64::from(*value)
                    * f64::from(self.head_weight[hidden_index * PHASE10V_HEAD_COUNT + head]);
            }
            *target = finite_f64_to_f32(sum);
        }
        Phase10VInference {
            cp: round_and_clamp_cp(f64::from(output[0])),
            wdl_logits: [output[1], output[2], output[3]],
        }
    }

    #[must_use]
    pub fn evaluate(&self, position: &Position) -> i32 {
        self.infer(position).cp
    }
}

fn feature_ids(position: &Position, perspective: Side) -> Vec<usize> {
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
    features.push(STM_OFFSET + usize::from(position.side_to_move() != perspective));
    features.sort_unstable();
    features.dedup();
    features
}

fn read_u32(data: &[u8], offset: usize) -> Result<u32, Phase10VError> {
    let bytes = data
        .get(offset..offset + 4)
        .ok_or(Phase10VError::Truncated)?;
    Ok(u32::from_le_bytes(
        bytes.try_into().map_err(|_| Phase10VError::Truncated)?,
    ))
}

fn read_u64(data: &[u8], offset: usize) -> Result<u64, Phase10VError> {
    let bytes = data
        .get(offset..offset + 8)
        .ok_or(Phase10VError::Truncated)?;
    Ok(u64::from_le_bytes(
        bytes.try_into().map_err(|_| Phase10VError::Truncated)?,
    ))
}

fn read_f32(data: &[u8], offset: usize) -> Result<f32, Phase10VError> {
    Ok(f32::from_bits(read_u32(data, offset)?))
}

fn read_values(
    payload: &[u8],
    cursor: &mut usize,
    count: usize,
    field: &'static str,
) -> Result<Vec<f32>, Phase10VError> {
    let bytes = count
        .checked_mul(4)
        .ok_or(Phase10VError::PayloadLengthMismatch)?;
    let end = cursor
        .checked_add(bytes)
        .ok_or(Phase10VError::PayloadLengthMismatch)?;
    let range = payload.get(*cursor..end).ok_or(Phase10VError::Truncated)?;
    let mut values = Vec::with_capacity(count);
    for chunk in range.chunks_exact(4) {
        let value = f32::from_le_bytes(chunk.try_into().map_err(|_| Phase10VError::Truncated)?);
        if !value.is_finite() {
            return Err(Phase10VError::NonFinite(field));
        }
        if value.abs() > MAX_PARAMETER_MAGNITUDE {
            return Err(Phase10VError::UnsupportedMagnitude(field));
        }
        values.push(value);
    }
    *cursor = end;
    Ok(values)
}

fn validate_accumulator_grid(values: &[f32], field: &'static str) -> Result<(), Phase10VError> {
    if values
        .iter()
        .any(|&value| (f64::from(value) * ACCUMULATOR_SCALE).fract() != 0.0)
    {
        return Err(Phase10VError::UnsupportedAccumulatorGrid(field));
    }
    Ok(())
}

fn array_from_slice<const N: usize>(values: Vec<f32>) -> [f32; N] {
    values.try_into().expect("validated tensor dimension")
}

fn finite_relu(value: f64) -> f32 {
    let maximum = 1.0;
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
    let maximum = f64::from(20_000.min(crate::MATE_THRESHOLD - 1));
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
pub struct Phase10VAccumulator {
    position: Position,
    kings: [crate::Square; 2],
    values: [Vec<f64>; 2],
    artifact_digest: [u8; 32],
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
    use super::*;

    fn fixture(width: usize) -> Vec<u8> {
        let mut bytes = PHASE10V_MODEL_MAGIC.to_vec();
        for value in [3, 1, 8427, u32::try_from(width).unwrap(), 16, 4] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes.extend_from_slice(&71_u64.to_le_bytes());
        bytes.extend_from_slice(&1_f32.to_le_bytes());
        bytes.resize(HEADER_BYTES + expected_payload_bytes(width), 0);
        let checksum = Sha256::digest(&bytes);
        bytes.extend_from_slice(&checksum);
        bytes
    }

    fn resign(bytes: &mut [u8]) {
        let end = bytes.len() - CHECKSUM_BYTES;
        let checksum = Sha256::digest(&bytes[..end]);
        bytes[end..].copy_from_slice(&checksum);
    }

    #[test]
    fn v3_strict_format_and_header_integrity() {
        for width in [256, 512] {
            let bytes = fixture(width);
            assert_eq!(
                Phase10VEvaluator::from_bytes(&bytes)
                    .unwrap()
                    .identity()
                    .accumulator_width,
                width
            );
            let mut changed = bytes.clone();
            changed[32] ^= 1;
            assert!(matches!(
                Phase10VEvaluator::from_bytes(&changed),
                Err(Phase10VError::ChecksumMismatch)
            ));
            changed = bytes.clone();
            changed[20..24].copy_from_slice(&128_u32.to_le_bytes());
            assert!(matches!(
                Phase10VEvaluator::from_bytes(&changed),
                Err(Phase10VError::ArchitectureMismatch)
            ));
            changed = bytes.clone();
            changed[0] ^= 1;
            assert!(matches!(
                Phase10VEvaluator::from_bytes(&changed),
                Err(Phase10VError::InvalidMagic)
            ));
            changed = bytes.clone();
            changed.push(0);
            assert!(Phase10VEvaluator::from_bytes(&changed).is_err());
            for invalid in [f32::NAN, f32::INFINITY, 1_000_001.0] {
                changed = bytes.clone();
                changed[HEADER_BYTES..HEADER_BYTES + 4].copy_from_slice(&invalid.to_le_bytes());
                resign(&mut changed);
                assert!(Phase10VEvaluator::from_bytes(&changed).is_err());
            }
        }
        assert!(Phase10VEvaluator::load_file("/nonexistent/open-shogi-v3-model").is_err());
        assert!(Phase10VEvaluator::load_file(std::env::temp_dir()).is_err());
    }

    #[test]
    fn v3_off_grid_parameters_are_rejected() {
        for width in [256, 512] {
            for offset in [
                HEADER_BYTES,
                HEADER_BYTES + PHASE10V_FEATURE_COUNT * width * 4,
            ] {
                let mut bytes = fixture(width);
                bytes[offset..offset + 4].copy_from_slice(&0.01_f32.to_le_bytes());
                resign(&mut bytes);
                assert!(matches!(
                    Phase10VEvaluator::from_bytes(&bytes),
                    Err(Phase10VError::UnsupportedAccumulatorGrid(_))
                ));
            }
        }
    }

    #[test]
    fn v3_large_feature_cancellation_preserves_exact_cp() {
        for width in [256, 512] {
            let mut bytes = fixture(width);
            let before = crate::parse_sfen("4k4/9/9/9/4P4/9/9/9/4K4 b - 1").unwrap();
            let movement = crate::parse_usi_move("5e5d").unwrap();
            let mut after = before.clone();
            after.make_move(movement).unwrap();
            for side in [Side::Black, Side::White] {
                let next = feature_ids(&after, side);
                let source = feature_ids(&before, side)
                    .into_iter()
                    .find(|feature| *feature < BOARD_FEATURES && !next.contains(feature))
                    .unwrap();
                let offset = HEADER_BYTES + source * width * 4;
                bytes[offset..offset + 4].copy_from_slice(&1_000_000_f32.to_le_bytes());
            }
            let small = finite_f64_to_f32((0.01 * ACCUMULATOR_SCALE).round() / ACCUMULATOR_SCALE);
            // The previous f32 update lost this term, causing a 100 cp error.
            assert_eq!(
                ((small + 1_000_000_f32) - 1_000_000_f32).to_bits(),
                0.0_f32.to_bits()
            );
            let bias_offset = HEADER_BYTES + PHASE10V_FEATURE_COUNT * width * 4;
            bytes[bias_offset..bias_offset + 4].copy_from_slice(&small.to_le_bytes());
            let hidden_offset = bias_offset + width * 4;
            bytes[hidden_offset..hidden_offset + 4].copy_from_slice(&1_f32.to_le_bytes());
            let head_offset =
                hidden_offset + (3 * width * PHASE10V_HIDDEN_WIDTH + PHASE10V_HIDDEN_WIDTH) * 4;
            bytes[head_offset..head_offset + 4].copy_from_slice(&10_000_f32.to_le_bytes());
            resign(&mut bytes);
            let model = Phase10VEvaluator::from_bytes(&bytes).unwrap();
            let mut state = model.accumulator(&before).unwrap();
            model
                .update_accumulator(&mut state, movement, &after)
                .unwrap();
            let full = model.accumulator(&after).unwrap();
            assert_eq!(state.values, full.values);
            assert_eq!(model.infer_accumulator(&state).unwrap().cp, 100);
            assert_eq!(model.infer_accumulator(&full).unwrap().cp, 100);
        }
    }

    #[test]
    fn v3_accumulators_cannot_cross_model_identity_or_width() {
        let model = Phase10VEvaluator::from_bytes(&fixture(256)).unwrap();
        let position = Position::startpos();
        let state = model.accumulator(&position).unwrap();
        for width in [256, 512] {
            let mut bytes = fixture(width);
            bytes[32] ^= 1; // Same architecture can still belong to another artifact.
            resign(&mut bytes);
            let other = Phase10VEvaluator::from_bytes(&bytes).unwrap();
            assert!(matches!(
                other.infer_accumulator(&state),
                Err(Phase10VError::IncompatibleAccumulator)
            ));
            let mut state = state.clone();
            let saved = state.clone();
            let movement = crate::parse_usi_move("7g7f").unwrap();
            let mut after = position.clone();
            after.make_move(movement).unwrap();
            assert!(matches!(
                other.update_accumulator(&mut state, movement, &after),
                Err(Phase10VError::IncompatibleAccumulator)
            ));
            assert_eq!(state, saved);
        }
    }

    #[test]
    fn v3_move_local_parity_and_exact_unmake_both_widths() {
        for width in [256, 512] {
            let mut model = Phase10VEvaluator::from_bytes(&fixture(width)).unwrap();
            for (i, value) in model.table.iter_mut().enumerate() {
                let raw = f64::from(u16::try_from(i % 97).unwrap()) / 9700.0 - 0.002;
                *value = finite_f64_to_f32((raw * ACCUMULATOR_SCALE).round() / ACCUMULATOR_SCALE);
            }
            model.hidden_weight.fill(0.01);
            model.head_weight.fill(0.1);
            for (sfen, usi) in [
                ("4k4/9/9/9/4P4/9/9/9/4K4 b - 1", "5e5d"),
                ("4k4/9/9/4p4/4P4/9/9/9/4K4 b - 1", "5e5d"),
                ("4k4/9/9/4P4/9/9/9/9/4K4 b - 1", "5d5c+"),
                ("4k4/9/9/9/9/9/9/9/4K4 b P 1", "P*5e"),
                ("4k4/9/9/9/4P4/9/9/9/4K4 b - 1", "5i5h"),
                ("4k4/9/9/9/4p4/9/9/9/4K4 w - 1", "5e5f"),
                ("4k4/9/9/9/9/4p4/9/9/4K4 w - 1", "5f5g+"),
            ] {
                let mut position = crate::parse_sfen(sfen).unwrap();
                let mut state = model.accumulator(&position).unwrap();
                let original = state.clone();
                let movement = crate::parse_usi_move(usi).unwrap();
                let undo = position.make_move(movement).unwrap();
                let previous = model
                    .update_accumulator(&mut state, movement, &position)
                    .unwrap();
                let full = model.accumulator(&position).unwrap();
                for (a, b) in state
                    .values
                    .iter()
                    .flatten()
                    .zip(full.values.iter().flatten())
                {
                    assert_eq!(a.to_bits(), b.to_bits(), "{usi}");
                }
                assert_eq!(
                    model.infer_accumulator(&state).unwrap().cp,
                    model.infer_accumulator(&full).unwrap().cp
                );
                position.unmake_move(undo);
                state = previous;
                assert_eq!(state, original);
                assert_eq!(state.position, position);
            }
            let mut position = Position::startpos();
            let mut state = model.accumulator(&position).unwrap();
            let initial = state.clone();
            let mut snapshots = Vec::new();
            for ply in 0..80 {
                let moves = position.legal_moves();
                if moves.is_empty() {
                    break;
                }
                let movement = moves[(ply * 17 + 5) % moves.len()];
                position.make_move(movement).unwrap();
                snapshots.push(model.update_generated_accumulator(&mut state, movement, &position));
                let full = model.accumulator(&position).unwrap();
                assert!(
                    state
                        .values
                        .iter()
                        .flatten()
                        .zip(full.values.iter().flatten())
                        .all(|(a, b)| a.to_bits() == b.to_bits())
                );
                assert_eq!(
                    model.infer_accumulator(&state).unwrap().cp,
                    model.infer_accumulator(&full).unwrap().cp
                );
            }
            for snapshot in snapshots.into_iter().rev() {
                state = snapshot;
            }
            assert_eq!(state, initial);
        }
    }

    #[test]
    fn v3_clipping_pair_interaction_score_perspective_and_mate_namespace() {
        let mut model = Phase10VEvaluator::from_bytes(&fixture(256)).unwrap();
        // One own pawn in hand activates a known learned input, independent of board pieces.
        model.table[HAND_OFFSET * 256] = 1.0;
        model.hidden_weight[0] = 1.0;
        model.hidden_weight[256 * 16 + 1] = 1.0;
        model.head_weight[0] = 500.0;
        model.head_weight[4] = -500.0;
        let black = crate::parse_sfen("4k4/9/9/9/9/9/9/9/4K4 b P 1").unwrap();
        let white = crate::parse_sfen("4k4/9/9/9/9/9/9/9/4K4 w P 1").unwrap();
        assert_eq!(model.evaluate(&black), 500);
        assert_eq!(model.evaluate(&white), -500);
        model.head_bias[0] = 1_000_000.0;
        assert_eq!(model.evaluate(&black), 20_000);
        assert!(!crate::is_mate_score(model.evaluate(&black)));
        model.head_bias[0] = -1_000_000.0;
        assert_eq!(model.evaluate(&white), -20_000);
        model.head_bias.fill(0.0);
        model.table.fill(0.0);
        model.bias.fill(2.0);
        model.hidden_weight.fill(0.0);
        model.head_weight.fill(0.0);
        model.hidden_weight[2 * 256 * 16] = 2.0;
        model.head_weight[0] = 123.0;
        assert_eq!(model.evaluate(&black), 123); // Both accumulator and hidden clipping.
        model.bias.fill(-1.0);
        assert_eq!(model.evaluate(&black), 0);
    }

    #[test]
    fn v3_legal_search_incremental_proof_trace_and_history_adjudication() {
        let model = std::sync::Arc::new(Phase10VEvaluator::from_bytes(&fixture(256)).unwrap());
        let hash = model.identity().artifact_sha256.clone();
        assert!(
            crate::SearchEngine::with_phase10v(
                crate::SearchConfig::default(),
                model.clone(),
                &"0".repeat(64)
            )
            .is_err()
        );
        let mut engine = crate::SearchEngine::with_phase10v(
            crate::SearchConfig::default(),
            model.clone(),
            &hash,
        )
        .unwrap();
        engine.set_leaf_trace_limit(8);
        let position = Position::startpos();
        let limits = crate::SearchLimits {
            max_depth: 2,
            max_nodes: Some(128),
            movetime: None,
        };
        let result = engine.search(&position, limits, &crate::CancellationToken::new());
        assert!(position.legal_moves().contains(&result.best_move.unwrap()));
        let proof = engine.runtime_proof(result.stats, &hash);
        assert!(proof.valid_pure_learned());
        assert!(proof.accumulator_updates > 0);
        assert!(proof.accumulator_refreshes >= 2);
        let trace = engine.take_leaf_trace();
        assert_eq!(trace.len(), 8);
        assert!(trace.iter().all(|row| row["model_sha256"] == hash
            && crate::parse_sfen(row["sfen"].as_str().unwrap()).is_ok()));
        assert!(engine.take_leaf_trace().is_empty());
        engine
            .set_pure_history(&position, &[crate::parse_usi_move("7g7f").unwrap()])
            .unwrap();
        let failed = engine.search(&position, limits, &crate::CancellationToken::new());
        assert_eq!(
            failed.termination,
            crate::SearchTermination::EvaluationError
        );
        let failed_proof = engine.runtime_proof(failed.stats, &hash);
        assert_eq!(failed_proof.accumulator_updates, 0);
        assert_eq!(failed_proof.accumulator_refreshes, 0);
        assert!(engine.take_leaf_trace().is_empty());
        assert!(crate::parse_sfen("9/9/9/9/9/9/9/9/9 b - 1").is_err());
        let initial = crate::parse_sfen("4k4/9/9/9/9/9/9/9/4K4 b - 1").unwrap();
        let moves: Vec<_> = ["5i6i", "5a6a", "6i5i", "6a5a"]
            .repeat(3)
            .iter()
            .map(|s| crate::parse_usi_move(s).unwrap())
            .collect();
        let mut position = initial.clone();
        for movement in &moves {
            position.make_move(*movement).unwrap();
        }
        engine.set_pure_history(&initial, &moves).unwrap();
        let result = engine.search(&position, limits, &crate::CancellationToken::new());
        assert_eq!(result.score, 0);
        assert_eq!(result.best_move, None);
        assert_eq!(result.stats.learned_eval_calls, 0);
    }
}
