//! Strict, deterministic inference for the versioned `OpenShogiAI` value-network format.

use std::{
    error::Error,
    fmt::{self, Write as _},
    io::{self, Read},
    path::Path,
};

use sha2::{Digest, Sha256};

use crate::{AnchoredFile, BOARD_SQUARES, HandPiece, PieceKind, Position, Side, Square};

const MODEL_MAGIC: &[u8; 8] = b"OSAVAL01";
const FORMAT_VERSION: u32 = 1;
const ARCHITECTURE_VERSION: u32 = 1;
const FEATURE_SCHEMA_VERSION: u32 = 1;
const CHECKSUM_BYTES: usize = 32;
const HEADER_BYTES: usize = MODEL_MAGIC.len() + 10 * size_of::<u32>() + size_of::<f32>();
const MIN_MODEL_BYTES: usize = HEADER_BYTES + CHECKSUM_BYTES;
const MAX_HIDDEN_LAYERS: usize = 16;
const MAX_HIDDEN_DIMENSION: usize = 8_192;
const MAX_PARAMETERS: usize = 16_000_000;

const BOARD_FEATURES: usize = 2 * 14 * BOARD_SQUARES;
const HAND_FEATURES: usize = 2 * 7;
const SIDE_FEATURES: usize = 1;
const KING_FEATURES: usize = 4;
const ATTACK_FEATURES: usize = 2 * BOARD_SQUARES;

/// Enables absolute Black/White piece-kind planes in SFEN square order.
pub const FEATURE_BOARD_PIECES: u32 = 1 << 0;
/// Enables normalized Black/White hand counts.
pub const FEATURE_HAND_COUNTS: u32 = 1 << 1;
/// Enables the side-to-move scalar (`1` for Black, `-1` for White).
pub const FEATURE_SIDE_TO_MOVE: u32 = 1 << 2;
/// Enables normalized Black then White king file/rank coordinates.
pub const FEATURE_KING_COORDINATES: u32 = 1 << 3;
/// Enables Black then White pseudo-attack maps in SFEN square order.
pub const FEATURE_ATTACK_MAPS: u32 = 1 << 4;

const SUPPORTED_FEATURES: u32 = FEATURE_BOARD_PIECES
    | FEATURE_HAND_COUNTS
    | FEATURE_SIDE_TO_MOVE
    | FEATURE_KING_COORDINATES
    | FEATURE_ATTACK_MAPS;

/// Maximum accepted model-file size, including its trailing checksum.
pub const MAX_NEURAL_MODEL_BYTES: usize = 64 * 1024 * 1024;

/// Largest neural centipawn score, kept strictly below the mate-score namespace.
pub const MAX_NEURAL_SCORE_CP: i32 = crate::search::MATE_THRESHOLD - 1;

/// Hidden-layer activation encoded in the model header.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NeuralActivation {
    Relu,
}

impl NeuralActivation {
    fn parse(value: u32) -> Result<Self, NeuralModelError> {
        match value {
            0 => Ok(Self::Relu),
            other => Err(NeuralModelError::UnsupportedActivation(other)),
        }
    }

    fn apply(self, value: f64) -> f32 {
        match self {
            Self::Relu => finite_f64_to_f32(value.max(0.0)),
        }
    }
}

/// Weight representation encoded for every layer in one model.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NeuralQuantization {
    Float32,
    Int8,
}

impl NeuralQuantization {
    fn parse(value: u32) -> Result<Self, NeuralModelError> {
        match value {
            0 => Ok(Self::Float32),
            1 => Ok(Self::Int8),
            other => Err(NeuralModelError::UnsupportedQuantization(other)),
        }
    }
}

/// Immutable identity and schema metadata verified from a model file.
#[derive(Clone, Debug, PartialEq)]
pub struct NeuralModelIdentity {
    pub format_version: u32,
    pub architecture_version: u32,
    pub feature_schema_version: u32,
    pub feature_flags: u32,
    pub input_dimension: usize,
    pub hidden_layers: usize,
    pub hidden_dimension: usize,
    pub activation: NeuralActivation,
    pub quantization: NeuralQuantization,
    pub layer_count: usize,
    pub output_scale_cp: f32,
    pub sha256: [u8; CHECKSUM_BYTES],
}

impl NeuralModelIdentity {
    /// Returns the canonical lowercase hexadecimal SHA-256 identity.
    #[must_use]
    pub fn sha256_hex(&self) -> String {
        let mut encoded = String::with_capacity(CHECKSUM_BYTES * 2);
        for byte in self.sha256 {
            write!(&mut encoded, "{byte:02x}").expect("writing to a String cannot fail");
        }
        encoded
    }
}

/// A rejected model file or feature schema.
#[derive(Debug)]
pub enum NeuralModelError {
    Io(io::Error),
    NotRegularFile,
    ModelTooLarge {
        bytes: usize,
        maximum: usize,
    },
    ChangedWhileReading,
    Truncated,
    InvalidMagic,
    ChecksumMismatch,
    UnsupportedFormatVersion(u32),
    UnsupportedArchitectureVersion(u32),
    UnsupportedFeatureSchemaVersion(u32),
    UnsupportedFeatureFlags(u32),
    UnsupportedActivation(u32),
    UnsupportedQuantization(u32),
    InvalidArchitecture(&'static str),
    InputDimensionMismatch {
        declared: usize,
        expected: usize,
    },
    LayerDimensionMismatch {
        layer: usize,
        declared_input: usize,
        declared_output: usize,
        expected_input: usize,
        expected_output: usize,
    },
    ParameterLimitExceeded {
        parameters: usize,
        maximum: usize,
    },
    NonFiniteValue(&'static str),
    NonPositiveScale(&'static str),
    TrailingData {
        bytes: usize,
    },
}

impl fmt::Display for NeuralModelError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "model I/O failed: {error}"),
            Self::NotRegularFile => formatter.write_str("model path is not a regular file"),
            Self::ModelTooLarge { bytes, maximum } => {
                write!(formatter, "model is {bytes} bytes; maximum is {maximum}")
            }
            Self::ChangedWhileReading => formatter.write_str("model changed while it was read"),
            Self::Truncated => formatter.write_str("model is truncated"),
            Self::InvalidMagic => formatter.write_str("model magic is not OSAVAL01"),
            Self::ChecksumMismatch => formatter.write_str("model SHA-256 checksum does not match"),
            Self::UnsupportedFormatVersion(version) => {
                write!(formatter, "unsupported model format version {version}")
            }
            Self::UnsupportedArchitectureVersion(version) => {
                write!(formatter, "unsupported architecture version {version}")
            }
            Self::UnsupportedFeatureSchemaVersion(version) => {
                write!(formatter, "unsupported feature schema version {version}")
            }
            Self::UnsupportedFeatureFlags(flags) => {
                write!(formatter, "unsupported feature flags 0x{flags:08x}")
            }
            Self::UnsupportedActivation(activation) => {
                write!(formatter, "unsupported activation {activation}")
            }
            Self::UnsupportedQuantization(quantization) => {
                write!(formatter, "unsupported quantization {quantization}")
            }
            Self::InvalidArchitecture(message) => formatter.write_str(message),
            Self::InputDimensionMismatch { declared, expected } => write!(
                formatter,
                "input dimension {declared} does not match feature schema dimension {expected}"
            ),
            Self::LayerDimensionMismatch {
                layer,
                declared_input,
                declared_output,
                expected_input,
                expected_output,
            } => write!(
                formatter,
                "layer {layer} is {declared_input}x{declared_output}; expected {expected_input}x{expected_output}"
            ),
            Self::ParameterLimitExceeded {
                parameters,
                maximum,
            } => write!(
                formatter,
                "model has {parameters} parameters; maximum is {maximum}"
            ),
            Self::NonFiniteValue(field) => write!(formatter, "{field} is not finite"),
            Self::NonPositiveScale(field) => {
                write!(formatter, "{field} must be finite and positive")
            }
            Self::TrailingData { bytes } => {
                write!(formatter, "model has {bytes} unexpected payload bytes")
            }
        }
    }
}

impl Error for NeuralModelError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            _ => None,
        }
    }
}

impl From<io::Error> for NeuralModelError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

#[derive(Debug)]
enum LayerWeights {
    Float32(Vec<f32>),
    Int8 { scale: f32, values: Vec<i8> },
}

#[derive(Debug)]
struct Layer {
    input_dimension: usize,
    output_dimension: usize,
    weights: LayerWeights,
    biases: Vec<f32>,
}

/// Thread-safe immutable value-network evaluator.
#[derive(Debug)]
pub struct NeuralEvaluator {
    identity: NeuralModelIdentity,
    layers: Vec<Layer>,
}

impl NeuralEvaluator {
    /// Parses and verifies a complete in-memory model artifact.
    ///
    /// # Errors
    ///
    /// Returns an error for oversized, corrupt, unsupported, non-finite, or structurally
    /// inconsistent data.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, NeuralModelError> {
        if bytes.len() > MAX_NEURAL_MODEL_BYTES {
            return Err(NeuralModelError::ModelTooLarge {
                bytes: bytes.len(),
                maximum: MAX_NEURAL_MODEL_BYTES,
            });
        }
        if bytes.len() < MIN_MODEL_BYTES {
            return Err(NeuralModelError::Truncated);
        }

        let payload_length = bytes.len() - CHECKSUM_BYTES;
        let (payload, encoded_checksum) = bytes.split_at(payload_length);
        let actual_checksum: [u8; CHECKSUM_BYTES] = Sha256::digest(payload).into();
        if encoded_checksum != actual_checksum {
            return Err(NeuralModelError::ChecksumMismatch);
        }

        Self::parse_payload(payload, actual_checksum)
    }

    /// Reads at most 64 MiB plus one sentinel byte, then parses and verifies a model artifact.
    ///
    /// # Errors
    ///
    /// Returns an I/O error or any error documented by [`Self::from_bytes`].
    pub fn load_file(path: impl AsRef<Path>) -> Result<Self, NeuralModelError> {
        Self::load_file_with(path.as_ref(), || Ok(()))
    }

    fn load_file_with(
        path: &Path,
        before_read: impl FnOnce() -> Result<(), NeuralModelError>,
    ) -> Result<Self, NeuralModelError> {
        let mut file = AnchoredFile::open_existing(path).map_err(|error| match error.kind() {
            io::ErrorKind::InvalidInput => NeuralModelError::NotRegularFile,
            _ => NeuralModelError::Io(error),
        })?;
        let identity = file.stable_identity()?;
        let maximum = u64::try_from(MAX_NEURAL_MODEL_BYTES).unwrap_or(u64::MAX);
        let file_length = identity.length();
        if file_length > maximum {
            return Err(NeuralModelError::ModelTooLarge {
                bytes: usize::try_from(file_length).unwrap_or(usize::MAX),
                maximum: MAX_NEURAL_MODEL_BYTES,
            });
        }
        before_read()?;
        let mut bytes = Vec::new();
        file.reader()
            .take(maximum.saturating_add(1))
            .read_to_end(&mut bytes)?;
        let bytes_read = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
        file.verify_stable_read(&identity, bytes_read)
            .map_err(|_| NeuralModelError::ChangedWhileReading)?;
        Self::from_bytes(&bytes)
    }

    /// Returns verified immutable model metadata.
    #[must_use]
    pub const fn identity(&self) -> &NeuralModelIdentity {
        &self.identity
    }

    /// Returns the model's weight representation.
    #[must_use]
    pub const fn quantization(&self) -> NeuralQuantization {
        self.identity.quantization
    }

    /// Returns the verified trailing SHA-256 identity.
    #[must_use]
    pub const fn sha256(&self) -> &[u8; CHECKSUM_BYTES] {
        &self.identity.sha256
    }

    /// Evaluates a position in centipawns from the current side-to-move perspective.
    ///
    /// The network output already uses that perspective; it is not colour-flipped here.
    /// Static scores are rounded and clamped strictly below the mate-score namespace.
    ///
    /// # Panics
    ///
    /// Panics only if internal model validation invariants have been violated after parsing.
    #[must_use]
    pub fn evaluate(&self, position: &Position) -> i32 {
        let mut values = encode_neural_features(position, self.identity.feature_flags)
            .expect("a parsed model always has supported feature flags");
        for (index, layer) in self.layers.iter().enumerate() {
            let final_layer = index + 1 == self.layers.len();
            values = layer.forward(&values, (!final_layer).then_some(self.identity.activation));
        }
        let scaled = f64::from(values[0]) * f64::from(self.identity.output_scale_cp);
        round_and_clamp_centipawns(scaled)
    }

    #[cfg(test)]
    pub(crate) fn side_to_move_test_evaluator() -> Self {
        Self {
            identity: NeuralModelIdentity {
                format_version: FORMAT_VERSION,
                architecture_version: ARCHITECTURE_VERSION,
                feature_schema_version: FEATURE_SCHEMA_VERSION,
                feature_flags: FEATURE_SIDE_TO_MOVE,
                input_dimension: 1,
                hidden_layers: 1,
                hidden_dimension: 1,
                activation: NeuralActivation::Relu,
                quantization: NeuralQuantization::Float32,
                layer_count: 2,
                output_scale_cp: 10.0,
                sha256: [0; CHECKSUM_BYTES],
            },
            layers: vec![
                Layer {
                    input_dimension: 1,
                    output_dimension: 1,
                    weights: LayerWeights::Float32(vec![2.0]),
                    biases: vec![0.0],
                },
                Layer {
                    input_dimension: 1,
                    output_dimension: 1,
                    weights: LayerWeights::Float32(vec![3.0]),
                    biases: vec![1.0],
                },
            ],
        }
    }

    #[expect(
        clippy::too_many_lines,
        reason = "the strict header and layer validation is intentionally kept in wire order"
    )]
    fn parse_payload(
        payload: &[u8],
        sha256: [u8; CHECKSUM_BYTES],
    ) -> Result<Self, NeuralModelError> {
        let mut cursor = Cursor::new(payload);
        if cursor.take(MODEL_MAGIC.len())? != MODEL_MAGIC {
            return Err(NeuralModelError::InvalidMagic);
        }

        let format_version = cursor.read_u32()?;
        if format_version != FORMAT_VERSION {
            return Err(NeuralModelError::UnsupportedFormatVersion(format_version));
        }
        let architecture_version = cursor.read_u32()?;
        if architecture_version != ARCHITECTURE_VERSION {
            return Err(NeuralModelError::UnsupportedArchitectureVersion(
                architecture_version,
            ));
        }
        let feature_schema_version = cursor.read_u32()?;
        if feature_schema_version != FEATURE_SCHEMA_VERSION {
            return Err(NeuralModelError::UnsupportedFeatureSchemaVersion(
                feature_schema_version,
            ));
        }

        let feature_flags = cursor.read_u32()?;
        let expected_input_dimension = neural_feature_dimension(feature_flags)
            .ok_or(NeuralModelError::UnsupportedFeatureFlags(feature_flags))?;
        let input_dimension = usize_from_u32(cursor.read_u32()?);
        if input_dimension != expected_input_dimension {
            return Err(NeuralModelError::InputDimensionMismatch {
                declared: input_dimension,
                expected: expected_input_dimension,
            });
        }

        let hidden_layers = usize_from_u32(cursor.read_u32()?);
        if hidden_layers == 0 || hidden_layers > MAX_HIDDEN_LAYERS {
            return Err(NeuralModelError::InvalidArchitecture(
                "hidden layer count must be 1..=16",
            ));
        }
        let hidden_dimension = usize_from_u32(cursor.read_u32()?);
        if hidden_dimension == 0 || hidden_dimension > MAX_HIDDEN_DIMENSION {
            return Err(NeuralModelError::InvalidArchitecture(
                "hidden dimension must be 1..=8192",
            ));
        }
        let activation = NeuralActivation::parse(cursor.read_u32()?)?;
        let quantization = NeuralQuantization::parse(cursor.read_u32()?)?;
        let layer_count = usize_from_u32(cursor.read_u32()?);
        let expected_layer_count = hidden_layers + 1;
        if layer_count != expected_layer_count {
            return Err(NeuralModelError::InvalidArchitecture(
                "layer count must equal hidden layer count plus one output layer",
            ));
        }
        let output_scale_cp = cursor.read_finite_f32("output scale")?;
        if output_scale_cp <= 0.0 {
            return Err(NeuralModelError::NonPositiveScale("output scale"));
        }

        let mut layers = Vec::with_capacity(layer_count);
        let mut parameter_count = 0_usize;
        for layer_index in 0..layer_count {
            let declared_input = usize_from_u32(cursor.read_u32()?);
            let declared_output = usize_from_u32(cursor.read_u32()?);
            let expected_input = if layer_index == 0 {
                input_dimension
            } else {
                hidden_dimension
            };
            let expected_output = if layer_index < hidden_layers {
                hidden_dimension
            } else {
                1
            };
            if declared_input != expected_input || declared_output != expected_output {
                return Err(NeuralModelError::LayerDimensionMismatch {
                    layer: layer_index,
                    declared_input,
                    declared_output,
                    expected_input,
                    expected_output,
                });
            }

            let weight_count = declared_input.checked_mul(declared_output).ok_or(
                NeuralModelError::ParameterLimitExceeded {
                    parameters: usize::MAX,
                    maximum: MAX_PARAMETERS,
                },
            )?;
            parameter_count = parameter_count
                .checked_add(weight_count)
                .and_then(|count| count.checked_add(declared_output))
                .ok_or(NeuralModelError::ParameterLimitExceeded {
                    parameters: usize::MAX,
                    maximum: MAX_PARAMETERS,
                })?;
            if parameter_count > MAX_PARAMETERS {
                return Err(NeuralModelError::ParameterLimitExceeded {
                    parameters: parameter_count,
                    maximum: MAX_PARAMETERS,
                });
            }

            let weights = match quantization {
                NeuralQuantization::Float32 => {
                    LayerWeights::Float32(cursor.read_f32_values(weight_count, "weight")?)
                }
                NeuralQuantization::Int8 => {
                    let scale = cursor.read_finite_f32("quantization scale")?;
                    if scale <= 0.0 {
                        return Err(NeuralModelError::NonPositiveScale("quantization scale"));
                    }
                    let values = cursor
                        .take(weight_count)?
                        .iter()
                        .map(|byte| i8::from_le_bytes([*byte]))
                        .collect();
                    LayerWeights::Int8 { scale, values }
                }
            };
            let biases = cursor.read_f32_values(declared_output, "bias")?;
            layers.push(Layer {
                input_dimension: declared_input,
                output_dimension: declared_output,
                weights,
                biases,
            });
        }
        if cursor.remaining() != 0 {
            return Err(NeuralModelError::TrailingData {
                bytes: cursor.remaining(),
            });
        }

        Ok(Self {
            identity: NeuralModelIdentity {
                format_version,
                architecture_version,
                feature_schema_version,
                feature_flags,
                input_dimension,
                hidden_layers,
                hidden_dimension,
                activation,
                quantization,
                layer_count,
                output_scale_cp,
                sha256,
            },
            layers,
        })
    }
}

impl Layer {
    fn forward(&self, input: &[f32], activation: Option<NeuralActivation>) -> Vec<f32> {
        debug_assert_eq!(input.len(), self.input_dimension);
        let mut output = Vec::with_capacity(self.output_dimension);
        match &self.weights {
            LayerWeights::Float32(weights) => {
                for (row, bias) in weights.chunks_exact(self.input_dimension).zip(&self.biases) {
                    let mut dot_product = 0.0_f64;
                    for (value, weight) in input.iter().zip(row) {
                        dot_product += f64::from(*value) * f64::from(*weight);
                    }
                    let sum = f64::from(*bias) + dot_product;
                    output.push(
                        activation
                            .map_or_else(|| finite_f64_to_f32(sum), |function| function.apply(sum)),
                    );
                }
            }
            LayerWeights::Int8 { scale, values } => {
                let scale = f64::from(*scale);
                for (row, bias) in values.chunks_exact(self.input_dimension).zip(&self.biases) {
                    let mut dot_product = 0.0_f64;
                    for (value, weight) in input.iter().zip(row) {
                        dot_product += f64::from(*value) * f64::from(*weight);
                    }
                    let sum = f64::from(*bias) + scale * dot_product;
                    output.push(
                        activation
                            .map_or_else(|| finite_f64_to_f32(sum), |function| function.apply(sum)),
                    );
                }
            }
        }
        output
    }
}

/// Recomputes the exact schema dimension for a supported nonempty feature mask.
#[must_use]
pub const fn neural_feature_dimension(feature_flags: u32) -> Option<usize> {
    if feature_flags == 0 || feature_flags & !SUPPORTED_FEATURES != 0 {
        return None;
    }
    let mut dimension = 0;
    if feature_flags & FEATURE_BOARD_PIECES != 0 {
        dimension += BOARD_FEATURES;
    }
    if feature_flags & FEATURE_HAND_COUNTS != 0 {
        dimension += HAND_FEATURES;
    }
    if feature_flags & FEATURE_SIDE_TO_MOVE != 0 {
        dimension += SIDE_FEATURES;
    }
    if feature_flags & FEATURE_KING_COORDINATES != 0 {
        dimension += KING_FEATURES;
    }
    if feature_flags & FEATURE_ATTACK_MAPS != 0 {
        dimension += ATTACK_FEATURES;
    }
    Some(dimension)
}

/// Encodes enabled feature groups in ascending feature-bit order.
///
/// Within a group, colours are Black then White, piece kinds follow [`PieceKind::ALL`], and
/// squares use SFEN scan order. The feature mask must be a supported nonempty subset.
///
/// # Errors
///
/// Returns [`NeuralModelError::UnsupportedFeatureFlags`] for zero or unknown feature bits.
///
/// # Panics
///
/// Panics if a [`Position`] violates its invariant of containing exactly one king per side.
pub fn encode_neural_features(
    position: &Position,
    feature_flags: u32,
) -> Result<Vec<f32>, NeuralModelError> {
    let dimension = neural_feature_dimension(feature_flags)
        .ok_or(NeuralModelError::UnsupportedFeatureFlags(feature_flags))?;
    let mut features = Vec::with_capacity(dimension);

    if feature_flags & FEATURE_BOARD_PIECES != 0 {
        let offset = features.len();
        features.resize(offset + BOARD_FEATURES, 0.0);
        for (square_index, piece) in position.board().iter().enumerate() {
            if let Some(piece) = piece {
                let plane = piece.side.index() * PieceKind::ALL.len() + piece.kind.index();
                features[offset + plane * BOARD_SQUARES + square_index] = 1.0;
            }
        }
    }

    if feature_flags & FEATURE_HAND_COUNTS != 0 {
        const LEGAL_MAXIMA: [f32; 7] = [18.0, 4.0, 4.0, 4.0, 4.0, 2.0, 2.0];
        for side in [Side::Black, Side::White] {
            for hand_piece in HandPiece::ALL {
                features.push(
                    f32::from(position.hand(side).count(hand_piece))
                        / LEGAL_MAXIMA[hand_piece.index()],
                );
            }
        }
    }

    if feature_flags & FEATURE_SIDE_TO_MOVE != 0 {
        features.push(match position.side_to_move() {
            Side::Black => 1.0,
            Side::White => -1.0,
        });
    }

    if feature_flags & FEATURE_KING_COORDINATES != 0 {
        for side in [Side::Black, Side::White] {
            let king = position
                .king_square(side)
                .expect("validated positions retain exactly one king per side");
            features.push(normalize_coordinate(king.file()));
            features.push(normalize_coordinate(king.rank()));
        }
    }

    if feature_flags & FEATURE_ATTACK_MAPS != 0 {
        for side in [Side::Black, Side::White] {
            for square in Square::all() {
                features.push(f32::from(u8::from(
                    position.square_is_attacked(square, side),
                )));
            }
        }
    }

    debug_assert_eq!(features.len(), dimension);
    Ok(features)
}

fn normalize_coordinate(coordinate: u8) -> f32 {
    (f32::from(coordinate) - 5.0) / 4.0
}

fn finite_f64_to_f32(value: f64) -> f32 {
    let maximum = f64::from(f32::MAX);
    let clamped = value.clamp(-maximum, maximum);
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the value is explicitly clamped to the finite f32 range"
    )]
    {
        clamped as f32
    }
}

fn round_and_clamp_centipawns(value: f64) -> i32 {
    let maximum = f64::from(MAX_NEURAL_SCORE_CP);
    let clamped = value.clamp(-maximum, maximum).round();
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the rounded value is explicitly clamped to a small i32 range"
    )]
    {
        clamped as i32
    }
}

fn usize_from_u32(value: u32) -> usize {
    usize::try_from(value).unwrap_or(usize::MAX)
}

struct Cursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Cursor<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn take(&mut self, length: usize) -> Result<&'a [u8], NeuralModelError> {
        let end = self
            .offset
            .checked_add(length)
            .ok_or(NeuralModelError::Truncated)?;
        let value = self
            .bytes
            .get(self.offset..end)
            .ok_or(NeuralModelError::Truncated)?;
        self.offset = end;
        Ok(value)
    }

    fn read_u32(&mut self) -> Result<u32, NeuralModelError> {
        let bytes: [u8; size_of::<u32>()] = self
            .take(size_of::<u32>())?
            .try_into()
            .map_err(|_| NeuralModelError::Truncated)?;
        Ok(u32::from_le_bytes(bytes))
    }

    fn read_finite_f32(&mut self, field: &'static str) -> Result<f32, NeuralModelError> {
        let bytes: [u8; size_of::<f32>()] = self
            .take(size_of::<f32>())?
            .try_into()
            .map_err(|_| NeuralModelError::Truncated)?;
        let value = f32::from_le_bytes(bytes);
        if !value.is_finite() {
            return Err(NeuralModelError::NonFiniteValue(field));
        }
        Ok(value)
    }

    fn read_f32_values(
        &mut self,
        count: usize,
        field: &'static str,
    ) -> Result<Vec<f32>, NeuralModelError> {
        let byte_count = count
            .checked_mul(size_of::<f32>())
            .ok_or(NeuralModelError::Truncated)?;
        let bytes = self.take(byte_count)?;
        bytes
            .chunks_exact(size_of::<f32>())
            .map(|chunk| {
                let value = f32::from_le_bytes(
                    chunk
                        .try_into()
                        .expect("chunks_exact always yields four-byte slices"),
                );
                if value.is_finite() {
                    Ok(value)
                } else {
                    Err(NeuralModelError::NonFiniteValue(field))
                }
            })
            .collect()
    }

    const fn remaining(&self) -> usize {
        self.bytes.len() - self.offset
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;

    use super::*;
    use crate::{Hand, Move, Piece};

    fn append_u32(bytes: &mut Vec<u8>, value: u32) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn append_f32(bytes: &mut Vec<u8>, value: f32) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn signed_model(quantization: NeuralQuantization) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MODEL_MAGIC);
        for value in [
            FORMAT_VERSION,
            ARCHITECTURE_VERSION,
            FEATURE_SCHEMA_VERSION,
            FEATURE_SIDE_TO_MOVE,
            1,
            1,
            1,
            0,
            match quantization {
                NeuralQuantization::Float32 => 0,
                NeuralQuantization::Int8 => 1,
            },
            2,
        ] {
            append_u32(&mut bytes, value);
        }
        append_f32(&mut bytes, 10.0);

        append_u32(&mut bytes, 1);
        append_u32(&mut bytes, 1);
        match quantization {
            NeuralQuantization::Float32 => append_f32(&mut bytes, 2.0),
            NeuralQuantization::Int8 => {
                append_f32(&mut bytes, 0.5);
                bytes.push(4_u8);
            }
        }
        append_f32(&mut bytes, 0.0);

        append_u32(&mut bytes, 1);
        append_u32(&mut bytes, 1);
        match quantization {
            NeuralQuantization::Float32 => append_f32(&mut bytes, 3.0),
            NeuralQuantization::Int8 => {
                append_f32(&mut bytes, 0.5);
                bytes.push(6_u8);
            }
        }
        append_f32(&mut bytes, 1.0);
        append_checksum(&mut bytes);
        bytes
    }

    fn append_checksum(bytes: &mut Vec<u8>) {
        let checksum = Sha256::digest(bytes.as_slice());
        bytes.extend_from_slice(&checksum);
    }

    fn resign(bytes: &mut [u8]) {
        let payload_length = bytes.len() - CHECKSUM_BYTES;
        let checksum: [u8; CHECKSUM_BYTES] = Sha256::digest(&bytes[..payload_length]).into();
        bytes[payload_length..].copy_from_slice(&checksum);
    }

    #[test]
    fn float_and_int8_inference_match_and_do_not_flip_white_output() {
        let float = NeuralEvaluator::from_bytes(&signed_model(NeuralQuantization::Float32))
            .expect("float model");
        let quantized = NeuralEvaluator::from_bytes(&signed_model(NeuralQuantization::Int8))
            .expect("quantized model");
        let black = Position::startpos();
        let mut white = Position::startpos();
        let movement: Move = white.legal_moves()[0];
        white.make_move(movement).expect("legal move");

        assert_eq!(float.evaluate(&black), 70);
        assert_eq!(quantized.evaluate(&black), 70);
        assert_eq!(float.evaluate(&white), 10);
        assert_eq!(quantized.evaluate(&white), 10);
        assert_eq!(float.quantization(), NeuralQuantization::Float32);
        assert_eq!(quantized.quantization(), NeuralQuantization::Int8);
    }

    #[test]
    fn version_one_rejects_non_relu_activations() {
        let mut bytes = signed_model(NeuralQuantization::Float32);
        bytes[36..40].copy_from_slice(&1_u32.to_le_bytes());
        resign(&mut bytes);
        assert!(matches!(
            NeuralEvaluator::from_bytes(&bytes),
            Err(NeuralModelError::UnsupportedActivation(1))
        ));
    }

    #[test]
    #[expect(
        clippy::float_cmp,
        reason = "these schema features are encoded as exact binary zero/one values"
    )]
    fn feature_encoder_matches_group_order_and_position_attack_semantics() {
        let flags = SUPPORTED_FEATURES;
        let position = Position::startpos();
        let features = encode_neural_features(&position, flags).expect("supported features");
        assert_eq!(features.len(), 2_449);
        assert_eq!(
            features[..BOARD_FEATURES]
                .iter()
                .filter(|value| **value == 1.0)
                .count(),
            40
        );

        let black_pawn = Square::new(9, 7).expect("square").index();
        assert_eq!(features[black_pawn], 1.0);
        let white_pawn_plane = PieceKind::ALL.len();
        let white_pawn = Square::new(9, 3).expect("square").index();
        assert_eq!(features[white_pawn_plane * BOARD_SQUARES + white_pawn], 1.0);

        let hand_offset = BOARD_FEATURES;
        assert!(
            features[hand_offset..hand_offset + HAND_FEATURES]
                .iter()
                .all(|value| *value == 0.0)
        );
        let side_offset = hand_offset + HAND_FEATURES;
        assert_eq!(features[side_offset], 1.0);
        let king_offset = side_offset + SIDE_FEATURES;
        assert_eq!(
            &features[king_offset..king_offset + KING_FEATURES],
            &[0.0, 1.0, 0.0, -1.0]
        );

        let attack_offset = king_offset + KING_FEATURES;
        for side in [Side::Black, Side::White] {
            for square in Square::all() {
                let expected = f32::from(u8::from(position.square_is_attacked(square, side)));
                let index = attack_offset + side.index() * BOARD_SQUARES + square.index();
                assert_eq!(features[index], expected);
            }
        }
    }

    #[test]
    #[expect(
        clippy::float_cmp,
        reason = "normalization endpoints and exact hand ratios are binary zero/one values"
    )]
    fn king_file_rank_orientation_and_hand_maxima_match_the_wire_contract() {
        let mut board = [None; BOARD_SQUARES];
        board[Square::new(9, 9).expect("square").index()] =
            Some(Piece::new(Side::Black, PieceKind::King));
        board[Square::new(1, 1).expect("square").index()] =
            Some(Piece::new(Side::White, PieceKind::King));
        let mut hands = [Hand::default(); 2];
        for (piece, maximum) in HandPiece::ALL.into_iter().zip([18, 4, 4, 4, 4, 2, 2]) {
            hands[Side::Black.index()].set(piece, maximum);
        }
        let position = Position::from_parts(board, hands, Side::White, 1).expect("position");

        let features = encode_neural_features(
            &position,
            FEATURE_HAND_COUNTS | FEATURE_SIDE_TO_MOVE | FEATURE_KING_COORDINATES,
        )
        .expect("features");
        assert!(features[..7].iter().all(|value| *value == 1.0));
        assert!(features[7..14].iter().all(|value| *value == 0.0));
        assert_eq!(features[14], -1.0);
        assert_eq!(&features[15..], &[1.0, 1.0, -1.0, -1.0]);
    }

    #[test]
    fn parser_rejects_corruption_unknown_flags_bad_dimensions_and_nonfinite_values() {
        let mut corrupt = signed_model(NeuralQuantization::Float32);
        corrupt[60] ^= 1;
        assert!(matches!(
            NeuralEvaluator::from_bytes(&corrupt),
            Err(NeuralModelError::ChecksumMismatch)
        ));

        let mut flags = signed_model(NeuralQuantization::Float32);
        flags[20..24].copy_from_slice(&(1_u32 << 31).to_le_bytes());
        resign(&mut flags);
        assert!(matches!(
            NeuralEvaluator::from_bytes(&flags),
            Err(NeuralModelError::UnsupportedFeatureFlags(_))
        ));

        let mut dimensions = signed_model(NeuralQuantization::Float32);
        dimensions[52..56].copy_from_slice(&2_u32.to_le_bytes());
        resign(&mut dimensions);
        assert!(matches!(
            NeuralEvaluator::from_bytes(&dimensions),
            Err(NeuralModelError::LayerDimensionMismatch { .. })
        ));

        let mut nonfinite = signed_model(NeuralQuantization::Float32);
        nonfinite[60..64].copy_from_slice(&f32::NAN.to_le_bytes());
        resign(&mut nonfinite);
        assert!(matches!(
            NeuralEvaluator::from_bytes(&nonfinite),
            Err(NeuralModelError::NonFiniteValue("weight"))
        ));
    }

    #[test]
    fn parser_rejects_truncation_and_trailing_payload() {
        assert!(matches!(
            NeuralEvaluator::from_bytes(b"OSAVAL01"),
            Err(NeuralModelError::Truncated)
        ));

        let mut trailing = signed_model(NeuralQuantization::Float32);
        let checksum = trailing.split_off(trailing.len() - CHECKSUM_BYTES);
        trailing.push(0);
        trailing.extend_from_slice(&checksum);
        resign(&mut trailing);
        assert!(matches!(
            NeuralEvaluator::from_bytes(&trailing),
            Err(NeuralModelError::TrailingData { bytes: 1 })
        ));
    }

    #[test]
    fn file_loader_rejects_oversized_sparse_file_before_reading_it() {
        let temporary = std::env::temp_dir()
            .canonicalize()
            .expect("canonical temporary directory");
        assert!(matches!(
            NeuralEvaluator::load_file(&temporary),
            Err(NeuralModelError::NotRegularFile)
        ));

        let path = temporary.join(format!(
            "open-shogi-core-oversized-model-{}.osaval",
            std::process::id()
        ));
        let file = File::create(&path).expect("create sparse model");
        file.set_len(u64::try_from(MAX_NEURAL_MODEL_BYTES).unwrap_or(u64::MAX) + 1)
            .expect("size sparse model");
        drop(file);

        let result = NeuralEvaluator::load_file(&path);
        std::fs::remove_file(&path).expect("remove sparse model");
        assert!(matches!(
            result,
            Err(NeuralModelError::ModelTooLarge { .. })
        ));
    }

    #[test]
    fn file_loader_rejects_same_descriptor_mutation() {
        let path = temporary_model_path("changed-while-reading");
        let first = signed_model(NeuralQuantization::Float32);
        let mut replacement = first.clone();
        replacement[60..64].copy_from_slice(&4.0_f32.to_le_bytes());
        resign(&mut replacement);
        assert_eq!(first.len(), replacement.len());
        std::fs::write(&path, first).expect("write initial model");

        let result = NeuralEvaluator::load_file_with(&path, || {
            std::fs::write(&path, replacement.as_slice()).map_err(NeuralModelError::Io)
        });
        std::fs::remove_file(&path).expect("remove changed model");

        assert!(matches!(result, Err(NeuralModelError::ChangedWhileReading)));
    }

    #[cfg(unix)]
    #[test]
    fn file_loader_rejects_ancestor_symlink_redirection() {
        use std::os::unix::fs::symlink;

        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock after epoch")
            .as_nanos();
        let directory = std::env::temp_dir().canonicalize().unwrap().join(format!(
            "open-shogi-core-model-ancestor-{}-{nonce}",
            std::process::id()
        ));
        let external = directory.with_extension("external");
        std::fs::create_dir_all(&directory).expect("create model parent fixture");
        std::fs::create_dir_all(&external).expect("create external fixture");
        std::fs::write(
            external.join("model.osaval"),
            signed_model(NeuralQuantization::Float32),
        )
        .expect("write external model fixture");
        symlink(&external, directory.join("models")).expect("create ancestor symlink");

        assert!(NeuralEvaluator::load_file(directory.join("models/model.osaval")).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn file_loader_rejects_a_model_symlink() {
        use std::os::unix::fs::symlink;

        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock after epoch")
            .as_nanos();
        let directory = std::env::temp_dir()
            .canonicalize()
            .expect("canonical temporary directory")
            .join(format!(
                "open-shogi-core-model-link-{}-{nonce}",
                std::process::id()
            ));
        std::fs::create_dir_all(&directory).expect("create model link fixture");
        let target = directory.join("target.osaval");
        let link = directory.join("link.osaval");
        std::fs::write(&target, signed_model(NeuralQuantization::Float32))
            .expect("write model fixture");
        symlink(&target, &link).expect("create model symlink");

        assert!(matches!(
            NeuralEvaluator::load_file(&link),
            Err(NeuralModelError::NotRegularFile)
        ));

        std::fs::remove_file(link).expect("remove model symlink");
        std::fs::remove_file(target).expect("remove model fixture");
        std::fs::remove_dir(directory).expect("remove model fixture directory");
    }

    #[test]
    fn model_identity_is_verified_and_evaluator_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<NeuralEvaluator>();

        let bytes = signed_model(NeuralQuantization::Float32);
        let evaluator = NeuralEvaluator::from_bytes(&bytes).expect("model");
        let expected: [u8; CHECKSUM_BYTES] =
            Sha256::digest(&bytes[..bytes.len() - CHECKSUM_BYTES]).into();
        assert_eq!(evaluator.sha256(), &expected);
        assert_eq!(evaluator.identity().sha256_hex().len(), 64);
    }

    fn temporary_model_path(label: &str) -> std::path::PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock after epoch")
            .as_nanos();
        std::env::temp_dir().canonicalize().unwrap().join(format!(
            "open-shogi-core-{label}-{}-{nonce}.osaval",
            std::process::id()
        ))
    }

    #[test]
    fn neural_scores_are_rounded_and_kept_out_of_mate_namespace() {
        assert_eq!(round_and_clamp_centipawns(12.5), 13);
        assert_eq!(
            round_and_clamp_centipawns(f64::INFINITY),
            MAX_NEURAL_SCORE_CP
        );
        assert_eq!(
            round_and_clamp_centipawns(f64::NEG_INFINITY),
            -MAX_NEURAL_SCORE_CP
        );
    }
}
