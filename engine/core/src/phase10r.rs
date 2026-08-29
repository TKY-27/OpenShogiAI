//! Validated OSAVAL02 sparse inference shared by native and WebAssembly callers.

use std::{
    collections::{BTreeMap, HashSet},
    error::Error,
    fmt::{self, Write as _},
    io::{self, Read},
    path::Path,
    sync::Arc,
};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{AnchoredFile, HandPiece, Move, Piece, PieceKind, Position, Side, Square, to_sfen};

const MAGIC: &[u8; 8] = b"OSAVAL02";
const FORMAT_VERSION: u32 = 2;
const ENDIAN_MARKER: u32 = 0x0102_0304;
const HEADER_BYTES: usize = 4_064;
const CHECKSUM_BYTES: usize = 32;
const CONTAINER_OVERHEAD_BYTES: usize = HEADER_BYTES + CHECKSUM_BYTES;
const DESCRIPTOR_OFFSET: usize = 512;
const DESCRIPTOR_BYTES: usize = 112;
const MAX_TENSORS: usize = (HEADER_BYTES - DESCRIPTOR_OFFSET) / DESCRIPTOR_BYTES;
const POLICY_CLASSES: usize = 13_689;
const SCALAR_INPUTS: usize = 64;
const TRUNK_HIDDEN: usize = 128;
const MOVE_DIMENSION: usize = 16;
const HASH_SEED: u64 = 20_260_729;
const MAX_TRIPLES: usize = 256;
const MAX_NON_MATE_CP: i32 = 28_999;
const EXPORTER_VERSION: &str = "OpenShogiAI-osaval02-py/v1";
const FEATURE_HASH_DOMAIN: &[u8] = b"OpenShogiAI/phase10r/features/v1\0";

const FEATURE_SCHEMA_SHA256: &str =
    "fb5d69c96ae45ed308ee18ab7fd16d4fbefe0f5778e0b0ee2144879bcc7881df";
const ARCHITECTURE_CONFIG_SHA256: &str =
    "50a6873b521f389c766010a0ed83fa5d2399d18a05a860fec78ef35f523dfb6b";
const TARGET_SEMANTICS_SHA256: &str =
    "bafe7ba97319fa12d0c7a8ef3e3634fe033926fa77fcb781b67cd8ab12dda3bb";
const INPUT_NORMALIZATION_SHA256: &str =
    "b19dea246a17e059010418f2ac04b5a75fb1e21a30ba050159e539a9f5f69185";
const MOVE_INDEX_SHA256: &str = "096b227cadae6e585977b688495f261b163ebd7d3b4d2cc7276555a1b297de2a";
const FLOAT_CONFIG_SHA256: &str =
    "4ab7ec4932dd595a80b3ccddcf46f0fe42fe318c62558d79ca91da3218580327";
const INT8_CONFIG_SHA256: &str = "315c92526d38149c67d5fb8b97b88a177b7ee87932910ebdd20e42479d3c7041";

const DTYPE_FLOAT32: u32 = 1;
const DTYPE_INT8: u32 = 2;
const HEAD_WDL: u32 = 1 << 0;
const HEAD_SCORE: u32 = 1 << 1;
const HEAD_MATE: u32 = 1 << 2;
const HEAD_UNCERTAINTY: u32 = 1 << 3;
const HEAD_POLICY: u32 = 1 << 4;
const HEADS_PAIR: u32 = HEAD_WDL | HEAD_MATE | HEAD_UNCERTAINTY | HEAD_POLICY;
const HEADS_PRIMARY: u32 = HEADS_PAIR | HEAD_SCORE;

/// Exact browser-side artifact bound for the frozen model matrix.
pub const MAX_OSAVAL02_MODEL_BYTES: usize = 16 * 1024 * 1024;

/// One of the two trainable Phase 10R architectures.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Osaval02Variant {
    SparsePairPolicyWdl,
    FactorizedPairTriplePolicyScore,
}

impl Osaval02Variant {
    fn parse(code: u32) -> Result<Self, Osaval02Error> {
        match code {
            1 => Ok(Self::SparsePairPolicyWdl),
            2 => Ok(Self::FactorizedPairTriplePolicyScore),
            _ => Err(Osaval02Error::Invalid(
                "unsupported OSAVAL02 architecture identifier",
            )),
        }
    }

    /// Stable architecture identifier used by Python and Wasm.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::SparsePairPolicyWdl => "sparse-pair-policy-wdl",
            Self::FactorizedPairTriplePolicyScore => "factorized-pair-triple-policy-score",
        }
    }

    const fn trunk_input(self) -> usize {
        match self {
            Self::SparsePairPolicyWdl => 48,
            Self::FactorizedPairTriplePolicyScore => 56,
        }
    }

    const fn value_outputs(self) -> usize {
        match self {
            Self::SparsePairPolicyWdl => 8,
            Self::FactorizedPairTriplePolicyScore => 9,
        }
    }

    const fn head_flags(self) -> u32 {
        match self {
            Self::SparsePairPolicyWdl => HEADS_PAIR,
            Self::FactorizedPairTriplePolicyScore => HEADS_PRIMARY,
        }
    }

    const fn triple_cap(self) -> usize {
        match self {
            Self::SparsePairPolicyWdl => 0,
            Self::FactorizedPairTriplePolicyScore => MAX_TRIPLES,
        }
    }
}

/// Weight representation of every tensor in one artifact.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Osaval02Quantization {
    Float32,
    Int8,
}

impl Osaval02Quantization {
    fn parse(code: u32) -> Result<Self, Osaval02Error> {
        match code {
            0 => Ok(Self::Float32),
            1 => Ok(Self::Int8),
            _ => Err(Osaval02Error::Invalid("unsupported OSAVAL02 quantization")),
        }
    }

    /// Stable quantization name used at machine boundaries.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Float32 => "float32",
            Self::Int8 => "int8",
        }
    }

    const fn dtype(self) -> u32 {
        match self {
            Self::Float32 => DTYPE_FLOAT32,
            Self::Int8 => DTYPE_INT8,
        }
    }

    const fn item_bytes(self) -> usize {
        match self {
            Self::Float32 => 4,
            Self::Int8 => 1,
        }
    }
}

/// Immutable hash-bound OSAVAL02 identity.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Osaval02Identity {
    pub format_version: u32,
    pub variant_id: &'static str,
    pub quantization: &'static str,
    pub parameter_count: usize,
    pub artifact_bytes: usize,
    pub artifact_sha256: String,
    pub weight_payload_sha256: String,
    pub feature_schema_sha256: &'static str,
    pub architecture_config_sha256: &'static str,
    pub target_semantics_sha256: &'static str,
    pub input_normalization_sha256: &'static str,
    pub dataset_manifest_sha256: String,
    pub move_index_sha256: &'static str,
    pub exporter_version: String,
    pub git_commit: String,
    pub training_run_reference: String,
}

/// Bounded history facts available at inference time.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields, rename_all = "camelCase")]
pub struct Osaval02History {
    pub available: bool,
    pub repetition_count: u8,
    pub continuous_check_by_us: bool,
    pub continuous_check_by_them: bool,
}

impl Default for Osaval02History {
    fn default() -> Self {
        Self {
            available: false,
            repetition_count: 1,
            continuous_check_by_us: false,
            continuous_check_by_them: false,
        }
    }
}

impl Osaval02History {
    fn validate(self) -> Result<(), Osaval02Error> {
        if !(1..=4).contains(&self.repetition_count) {
            return Err(Osaval02Error::Invalid(
                "history repetition count must be in 1..=4",
            ));
        }
        if !self.available
            && (self.repetition_count != 1
                || self.continuous_check_by_us
                || self.continuous_check_by_them)
        {
            return Err(Osaval02Error::Invalid(
                "unavailable history must use standalone default facts",
            ));
        }
        if self.continuous_check_by_us && self.continuous_check_by_them {
            return Err(Osaval02Error::Invalid(
                "both sides cannot own one continuous-check history",
            ));
        }
        Ok(())
    }
}

/// A malformed, corrupt, incompatible, or unsafe OSAVAL02 artifact/input.
#[derive(Debug)]
pub enum Osaval02Error {
    Io(io::Error),
    NotRegularFile,
    ChangedWhileReading,
    TooLarge { bytes: usize, maximum: usize },
    Invalid(&'static str),
    InvalidOwned(String),
}

impl fmt::Display for Osaval02Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "OSAVAL02 I/O failed: {error}"),
            Self::NotRegularFile => formatter.write_str("OSAVAL02 path is not a regular file"),
            Self::ChangedWhileReading => formatter.write_str("OSAVAL02 changed while it was read"),
            Self::TooLarge { bytes, maximum } => {
                write!(formatter, "OSAVAL02 is {bytes} bytes; maximum is {maximum}")
            }
            Self::Invalid(message) => formatter.write_str(message),
            Self::InvalidOwned(message) => formatter.write_str(message),
        }
    }
}

impl Error for Osaval02Error {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            _ => None,
        }
    }
}

impl From<io::Error> for Osaval02Error {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

#[derive(Clone, Debug)]
enum TensorValues {
    Float32(Vec<f32>),
    Int8(Vec<i8>),
}

#[derive(Clone, Debug)]
struct Tensor {
    dimensions: Vec<usize>,
    scale: f32,
    values: TensorValues,
}

impl Tensor {
    fn value(&self, index: usize) -> f64 {
        match &self.values {
            TensorValues::Float32(values) => f64::from(values[index]),
            TensorValues::Int8(values) => f64::from(values[index]) * f64::from(self.scale),
        }
    }
}

#[derive(Clone, Debug)]
struct TensorSpec {
    name: &'static str,
    dimensions: Vec<usize>,
}

impl TensorSpec {
    fn elements(&self) -> usize {
        self.dimensions.iter().product()
    }
}

/// Immutable validated sparse evaluator.
#[derive(Debug)]
pub struct Osaval02Evaluator {
    identity: Osaval02Identity,
    variant: Osaval02Variant,
    quantization: Osaval02Quantization,
    calibration_scale: f32,
    calibration_bias: f32,
    wdl_epsilon: f32,
    tensors: BTreeMap<String, Tensor>,
}

impl Osaval02Evaluator {
    /// Parse and verify one complete in-memory OSAVAL02 artifact.
    ///
    /// # Errors
    ///
    /// Returns an explicit error for every malformed, corrupt, oversized, or incompatible
    /// identity, section, tensor, or parameter.
    ///
    /// # Panics
    ///
    /// Panics only if an internal fixed-width chunk invariant is violated after its exact byte
    /// length has already been validated.
    #[expect(
        clippy::too_many_lines,
        clippy::float_cmp,
        reason = "strict wire validation remains in byte order and exact float metadata is canonical"
    )]
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Osaval02Error> {
        if bytes.len() > MAX_OSAVAL02_MODEL_BYTES {
            return Err(Osaval02Error::TooLarge {
                bytes: bytes.len(),
                maximum: MAX_OSAVAL02_MODEL_BYTES,
            });
        }
        if bytes.len() < CONTAINER_OVERHEAD_BYTES {
            return Err(Osaval02Error::Invalid("OSAVAL02 is truncated"));
        }
        let unsigned_length = bytes.len() - CHECKSUM_BYTES;
        let (unsigned, checksum) = bytes.split_at(unsigned_length);
        if checksum != Sha256::digest(unsigned).as_slice() {
            return Err(Osaval02Error::Invalid(
                "OSAVAL02 trailing SHA-256 does not match",
            ));
        }
        let header = &unsigned[..HEADER_BYTES];
        let payload = &unsigned[HEADER_BYTES..];
        if &header[..8] != MAGIC {
            return Err(Osaval02Error::Invalid("model magic is not OSAVAL02"));
        }
        if read_u32(header, 8)? != FORMAT_VERSION {
            return Err(Osaval02Error::Invalid(
                "unsupported OSAVAL02 format version",
            ));
        }
        if read_u32(header, 12)? != ENDIAN_MARKER {
            return Err(Osaval02Error::Invalid(
                "OSAVAL02 byte-order marker is invalid",
            ));
        }
        if read_usize_u32(header, 16)? != HEADER_BYTES
            || read_usize_u64(header, 496)? != HEADER_BYTES
        {
            return Err(Osaval02Error::Invalid(
                "OSAVAL02 header or payload offset is invalid",
            ));
        }
        if read_usize_u64(header, 20)? != bytes.len() {
            return Err(Osaval02Error::Invalid(
                "OSAVAL02 declared file length is invalid",
            ));
        }
        if read_u64(header, 504)? != 0 {
            return Err(Osaval02Error::Invalid(
                "OSAVAL02 reserved header field is nonzero",
            ));
        }
        let variant = Osaval02Variant::parse(read_u32(header, 28)?)?;
        let quantization = Osaval02Quantization::parse(read_u32(header, 32)?)?;
        let specs = tensor_specs(variant);
        let tensor_count = read_usize_u32(header, 36)?;
        if tensor_count != specs.len() || tensor_count > MAX_TENSORS {
            return Err(Osaval02Error::Invalid(
                "OSAVAL02 tensor count is incompatible",
            ));
        }
        if read_u32(header, 40)? != variant.head_flags() {
            return Err(Osaval02Error::Invalid(
                "OSAVAL02 output heads are incompatible",
            ));
        }
        let parameter_count = read_usize_u32(header, 44)?;
        let expected_parameters: usize = specs.iter().map(TensorSpec::elements).sum();
        if parameter_count != expected_parameters {
            return Err(Osaval02Error::Invalid(
                "OSAVAL02 parameter count is incompatible",
            ));
        }
        let declared_dimensions = [
            read_usize_u32(header, 48)?,
            read_usize_u32(header, 52)?,
            read_usize_u32(header, 56)?,
            read_usize_u32(header, 60)?,
        ];
        if declared_dimensions
            != [
                POLICY_CLASSES,
                SCALAR_INPUTS,
                variant.trunk_input(),
                TRUNK_HIDDEN,
            ]
        {
            return Err(Osaval02Error::Invalid(
                "OSAVAL02 frozen dimensions are incompatible",
            ));
        }
        if read_u64(header, 64)? != HASH_SEED {
            return Err(Osaval02Error::Invalid(
                "OSAVAL02 feature hash seed is incompatible",
            ));
        }
        if read_usize_u32(header, 72)? != variant.triple_cap() {
            return Err(Osaval02Error::Invalid(
                "OSAVAL02 triple cap is incompatible",
            ));
        }
        if read_u32(header, 76)? != 1 || read_u32(header, 92)? != MAX_NON_MATE_CP as u32 {
            return Err(Osaval02Error::Invalid(
                "OSAVAL02 runtime flags are incompatible",
            ));
        }
        let calibration_scale = read_f32(header, 80)?;
        let calibration_bias = read_f32(header, 84)?;
        let wdl_epsilon = read_f32(header, 88)?;
        if !calibration_scale.is_finite() || calibration_scale <= 0.0 {
            return Err(Osaval02Error::Invalid(
                "OSAVAL02 calibration scale is invalid",
            ));
        }
        if !calibration_bias.is_finite() {
            return Err(Osaval02Error::Invalid(
                "OSAVAL02 calibration bias is invalid",
            ));
        }
        if !wdl_epsilon.is_finite() || !(0.0..=0.01).contains(&wdl_epsilon) || wdl_epsilon == 0.0 {
            return Err(Osaval02Error::Invalid("OSAVAL02 WDL epsilon is invalid"));
        }

        let hashes = (0..8)
            .map(|index| hex(&header[96 + index * 32..128 + index * 32]))
            .collect::<Vec<_>>();
        for (observed, expected, label) in [
            (&hashes[0], FEATURE_SCHEMA_SHA256, "feature schema"),
            (&hashes[1], ARCHITECTURE_CONFIG_SHA256, "architecture"),
            (&hashes[2], TARGET_SEMANTICS_SHA256, "target semantics"),
            (
                &hashes[3],
                INPUT_NORMALIZATION_SHA256,
                "input normalization",
            ),
        ] {
            if observed != expected {
                return Err(Osaval02Error::InvalidOwned(format!(
                    "OSAVAL02 {label} hash is incompatible"
                )));
            }
        }
        if hashes[4].bytes().all(|byte| byte == b'0') {
            return Err(Osaval02Error::Invalid(
                "OSAVAL02 dataset/training identity is missing",
            ));
        }
        let expected_quantization_hash = match quantization {
            Osaval02Quantization::Float32 => FLOAT_CONFIG_SHA256,
            Osaval02Quantization::Int8 => INT8_CONFIG_SHA256,
        };
        if hashes[6] != expected_quantization_hash {
            return Err(Osaval02Error::Invalid(
                "OSAVAL02 quantization-config hash is incompatible",
            ));
        }
        if hashes[7] != MOVE_INDEX_SHA256 {
            return Err(Osaval02Error::Invalid(
                "OSAVAL02 move-index table hash is incompatible",
            ));
        }
        let exporter_version = read_fixed_text(header, 352, 32, "exporter version", false)?;
        if exporter_version != EXPORTER_VERSION {
            return Err(Osaval02Error::Invalid(
                "OSAVAL02 exporter version is incompatible",
            ));
        }
        let git_commit = std::str::from_utf8(&header[384..424])
            .map_err(|_| Osaval02Error::Invalid("OSAVAL02 Git commit identity is not ASCII"))?
            .to_owned();
        if git_commit.len() != 40
            || !git_commit
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(Osaval02Error::Invalid(
                "OSAVAL02 Git commit identity is invalid",
            ));
        }
        let training_run_reference =
            read_fixed_text(header, 424, 64, "training run reference", true)?;
        if read_usize_u32(header, 488)? != DESCRIPTOR_OFFSET
            || read_usize_u32(header, 492)? != DESCRIPTOR_BYTES
        {
            return Err(Osaval02Error::Invalid(
                "OSAVAL02 tensor-table layout is incompatible",
            ));
        }
        if hex(&Sha256::digest(payload)) != hashes[5] {
            return Err(Osaval02Error::Invalid(
                "OSAVAL02 weight payload SHA-256 does not match",
            ));
        }

        let mut tensors = BTreeMap::new();
        let mut next_offset = 0_usize;
        for (index, spec) in specs.iter().enumerate() {
            let descriptor = DESCRIPTOR_OFFSET + index * DESCRIPTOR_BYTES;
            let name = read_tensor_name(header, descriptor)?;
            let dtype = read_u32(header, descriptor + 48)?;
            let rank = read_usize_u32(header, descriptor + 52)?;
            let dimensions = (0..4)
                .map(|dimension| read_usize_u32(header, descriptor + 56 + dimension * 4))
                .collect::<Result<Vec<_>, _>>()?;
            let offset = read_usize_u64(header, descriptor + 72)?;
            let length = read_usize_u64(header, descriptor + 80)?;
            let elements = read_usize_u64(header, descriptor + 88)?;
            let scale = read_f32(header, descriptor + 96)?;
            let zero_point = read_i32(header, descriptor + 100)?;
            let flags = read_u32(header, descriptor + 104)?;
            if header[descriptor + 108..descriptor + DESCRIPTOR_BYTES]
                .iter()
                .any(|byte| *byte != 0)
            {
                return Err(Osaval02Error::Invalid(
                    "OSAVAL02 tensor descriptor padding is nonzero",
                ));
            }
            if name != spec.name || rank != spec.dimensions.len() {
                return Err(Osaval02Error::Invalid(
                    "OSAVAL02 tensor name or rank is incompatible",
                ));
            }
            if dimensions[..rank] != spec.dimensions
                || dimensions[rank..].iter().any(|dimension| *dimension != 1)
            {
                return Err(Osaval02Error::InvalidOwned(format!(
                    "OSAVAL02 tensor {name} shape is incompatible"
                )));
            }
            if dtype != quantization.dtype() {
                return Err(Osaval02Error::InvalidOwned(format!(
                    "OSAVAL02 tensor {name} dtype is incompatible"
                )));
            }
            let expected_elements = spec.elements();
            let expected_length = expected_elements
                .checked_mul(quantization.item_bytes())
                .ok_or(Osaval02Error::Invalid(
                    "OSAVAL02 tensor requests an oversized allocation",
                ))?;
            if offset != next_offset || elements != expected_elements || length != expected_length {
                return Err(Osaval02Error::InvalidOwned(format!(
                    "OSAVAL02 tensor {name} size or offset is incompatible"
                )));
            }
            if flags != 1 || zero_point != 0 || !scale.is_finite() || scale <= 0.0 {
                return Err(Osaval02Error::InvalidOwned(format!(
                    "OSAVAL02 tensor {name} quantization metadata is invalid"
                )));
            }
            if quantization == Osaval02Quantization::Float32 && scale != 1.0 {
                return Err(Osaval02Error::InvalidOwned(format!(
                    "OSAVAL02 float tensor {name} scale must be one"
                )));
            }
            let end = offset.checked_add(length).ok_or(Osaval02Error::Invalid(
                "OSAVAL02 tensor requests an oversized allocation",
            ))?;
            let encoded = payload.get(offset..end).ok_or(Osaval02Error::Invalid(
                "OSAVAL02 tensor requests an oversized allocation",
            ))?;
            let values = match quantization {
                Osaval02Quantization::Float32 => {
                    let mut values = Vec::with_capacity(expected_elements);
                    for chunk in encoded.chunks_exact(4) {
                        let value = f32::from_le_bytes(
                            chunk
                                .try_into()
                                .expect("four-byte chunks always convert to f32 bytes"),
                        );
                        if !value.is_finite() {
                            return Err(Osaval02Error::InvalidOwned(format!(
                                "OSAVAL02 tensor {name} contains a non-finite parameter"
                            )));
                        }
                        values.push(value);
                    }
                    TensorValues::Float32(values)
                }
                Osaval02Quantization::Int8 => TensorValues::Int8(
                    encoded
                        .iter()
                        .map(|byte| i8::from_le_bytes([*byte]))
                        .collect(),
                ),
            };
            tensors.insert(
                name,
                Tensor {
                    dimensions: spec.dimensions.clone(),
                    scale,
                    values,
                },
            );
            next_offset = end;
        }
        if next_offset != payload.len() {
            return Err(Osaval02Error::Invalid(
                "OSAVAL02 has a trailing invalid section",
            ));
        }
        let descriptor_end = DESCRIPTOR_OFFSET + tensor_count * DESCRIPTOR_BYTES;
        if header[descriptor_end..].iter().any(|byte| *byte != 0) {
            return Err(Osaval02Error::Invalid(
                "OSAVAL02 unused header bytes must be zero",
            ));
        }
        let expected_payload_bytes = parameter_count
            .checked_mul(quantization.item_bytes())
            .ok_or(Osaval02Error::Invalid(
                "OSAVAL02 tensor requests an oversized allocation",
            ))?;
        if payload.len() != expected_payload_bytes
            || bytes.len() != CONTAINER_OVERHEAD_BYTES + expected_payload_bytes
        {
            return Err(Osaval02Error::Invalid(
                "OSAVAL02 artifact size differs from the frozen model matrix",
            ));
        }
        let identity = Osaval02Identity {
            format_version: FORMAT_VERSION,
            variant_id: variant.name(),
            quantization: quantization.name(),
            parameter_count,
            artifact_bytes: bytes.len(),
            artifact_sha256: hex(&Sha256::digest(bytes)),
            weight_payload_sha256: hashes[5].clone(),
            feature_schema_sha256: FEATURE_SCHEMA_SHA256,
            architecture_config_sha256: ARCHITECTURE_CONFIG_SHA256,
            target_semantics_sha256: TARGET_SEMANTICS_SHA256,
            input_normalization_sha256: INPUT_NORMALIZATION_SHA256,
            dataset_manifest_sha256: hashes[4].clone(),
            move_index_sha256: MOVE_INDEX_SHA256,
            exporter_version,
            git_commit,
            training_run_reference,
        };
        Ok(Self {
            identity,
            variant,
            quantization,
            calibration_scale,
            calibration_bias,
            wdl_epsilon,
            tensors,
        })
    }

    /// Read one stable regular file and validate it without following symlinks.
    ///
    /// # Errors
    ///
    /// Returns an I/O, stable-read, size, or format/compatibility error.
    pub fn load_file(path: impl AsRef<Path>) -> Result<Self, Osaval02Error> {
        let path = path.as_ref();
        let mut file = AnchoredFile::open_existing(path).map_err(|error| match error.kind() {
            io::ErrorKind::InvalidInput => Osaval02Error::NotRegularFile,
            _ => Osaval02Error::Io(error),
        })?;
        let identity = file.stable_identity()?;
        let maximum = u64::try_from(MAX_OSAVAL02_MODEL_BYTES).unwrap_or(u64::MAX);
        if identity.length() > maximum {
            return Err(Osaval02Error::TooLarge {
                bytes: usize::try_from(identity.length()).unwrap_or(usize::MAX),
                maximum: MAX_OSAVAL02_MODEL_BYTES,
            });
        }
        let mut bytes = Vec::new();
        file.reader()
            .take(maximum.saturating_add(1))
            .read_to_end(&mut bytes)?;
        file.verify_stable_read(&identity, u64::try_from(bytes.len()).unwrap_or(u64::MAX))
            .map_err(|_| Osaval02Error::ChangedWhileReading)?;
        Self::from_bytes(&bytes)
    }

    /// Return the validated versioned identity.
    #[must_use]
    pub const fn identity(&self) -> &Osaval02Identity {
        &self.identity
    }

    /// Return the frozen architecture variant.
    #[must_use]
    pub const fn variant(&self) -> Osaval02Variant {
        self.variant
    }

    /// Return the validated weight representation.
    #[must_use]
    pub const fn quantization(&self) -> Osaval02Quantization {
        self.quantization
    }

    /// Infer every required head and policy logit for the exact legal root.
    ///
    /// # Errors
    ///
    /// Returns an error for inconsistent history, missing internal tensors, or non-finite output.
    ///
    /// # Panics
    ///
    /// Panics only if validated internal tensor dimensions or fixed head counts are violated.
    pub fn evaluate_score(
        &self,
        position: &Position,
        history: Osaval02History,
    ) -> Result<i32, Osaval02Error> {
        let (_, _, _, outputs) = self.forward_values(position, history)?;
        let (_, raw_score, _, _) = self.score_components(&outputs)?;
        let calibrated = raw_score.mul_add(
            f64::from(self.calibration_scale),
            f64::from(self.calibration_bias),
        );
        if !calibrated.is_finite() {
            return Err(Osaval02Error::Invalid(
                "OSAVAL02 calibrated score is not finite",
            ));
        }
        Ok(round_and_clamp(calibrated))
    }

    #[expect(
        clippy::too_many_lines,
        reason = "all output heads share one auditable deterministic inference order"
    )]
    pub fn infer(
        &self,
        position: &Position,
        history: Osaval02History,
    ) -> Result<Osaval02Inference, Osaval02Error> {
        let (legal_moves, features, hidden, outputs) = self.forward_values(position, history)?;
        let wdl_values = softmax(&outputs[..3]);
        let wdl = WdlOutput {
            loss: wdl_values[0],
            draw: wdl_values[1],
            win: wdl_values[2],
        };
        let (transformed, raw_score, source_head, mate_offset) = self.score_components(&outputs)?;
        let calibrated = raw_score.mul_add(
            f64::from(self.calibration_scale),
            f64::from(self.calibration_bias),
        );
        let calibrated_cp = round_and_clamp(calibrated);
        let mate_values = softmax(&outputs[mate_offset..mate_offset + 3]);
        let mate_index = (0..3)
            .max_by(|left, right| {
                mate_values[*left]
                    .total_cmp(&mate_values[*right])
                    .then_with(|| right.cmp(left))
            })
            .expect("three mate values always have a maximum");
        let distance_raw = outputs[mate_offset + 3];
        let mate_distance = distance_raw
            .abs()
            .min(512.0_f64.ln_1p())
            .exp_m1()
            .copysign(distance_raw);
        let log_variance = outputs[mate_offset + 4].clamp(-20.0, 20.0);

        let context = linear(
            self.tensor("policy.context_weight")?,
            self.tensor("policy.context_bias")?,
            &hidden,
            MOVE_DIMENSION,
            false,
        );
        let move_offset = self.tensor("policy.move_offset")?;
        let move_embeddings = self.tensor("policy.move_embeddings")?;
        let temperature = self
            .tensor("policy.log_temperature")?
            .value(0)
            .clamp(-4.0, 4.0)
            .exp();
        let mut policies = legal_moves
            .iter()
            .map(|movement| {
                let index = encode_move(*movement);
                let base = index * MOVE_DIMENSION;
                let logit = temperature
                    * (0..MOVE_DIMENSION)
                        .map(|dimension| {
                            context[dimension]
                                * (move_embeddings.value(base + dimension)
                                    + move_offset.value(dimension))
                        })
                        .sum::<f64>();
                PolicyOutput {
                    movement: crate::to_usi_move(*movement),
                    index,
                    logit,
                }
            })
            .collect::<Vec<_>>();
        policies.sort_by(|left, right| {
            right
                .logit
                .total_cmp(&left.logit)
                .then_with(|| left.index.cmp(&right.index))
        });
        let terminal = if legal_moves.is_empty() {
            if position.is_in_check(position.side_to_move()) {
                TerminalOutput {
                    kind: Some("checkmate"),
                    search_score_cp: Some(-30_000),
                }
            } else {
                TerminalOutput {
                    kind: Some("no_legal_move"),
                    search_score_cp: Some(0),
                }
            }
        } else {
            TerminalOutput {
                kind: None,
                search_score_cp: None,
            }
        };
        let canonical = canonical_state(position)?;
        let inference = Osaval02Inference {
            schema: "open_shogiai_osaval02_inference/v1",
            identity: InferenceIdentity {
                format_version: FORMAT_VERSION,
                variant_id: self.variant.name(),
                quantization: self.quantization.name(),
                artifact_sha256: self.identity.artifact_sha256.clone(),
                weight_payload_sha256: self.identity.weight_payload_sha256.clone(),
            },
            position_sha256: hex(&Sha256::digest(canonical.as_bytes())),
            feature_sha256: features.checksum,
            history,
            legal_moves: policies,
            wdl,
            score: ScoreOutput {
                source_head,
                transformed,
                calibrated_cp,
                perspective: "current_side_to_move",
            },
            mate: MateOutput {
                class: ["mated", "no_mate_label", "mating"][mate_index],
                probabilities: MateProbabilities {
                    mated: mate_values[0],
                    no_mate_label: mate_values[1],
                    mating: mate_values[2],
                },
                distance_plies: mate_distance,
                score_conversion: "forbidden",
            },
            uncertainty: UncertaintyOutput {
                log_variance,
                variance: log_variance.exp(),
                mixed_into_score: false,
            },
            terminal,
        };
        inference.validate_finite()?;
        Ok(inference)
    }

    fn forward_values(
        &self,
        position: &Position,
        history: Osaval02History,
    ) -> Result<(Vec<Move>, FeatureSet, Vec<f64>, Vec<f64>), Osaval02Error> {
        history.validate()?;
        let legal_moves = position.legal_moves();
        let features = encode_features(position, &legal_moves, history, self.variant)?;
        let king_piece = embedding_sum(
            self.tensor("king_piece_embeddings")?,
            &features.king_piece,
            4,
        );
        let king_hand = embedding_sum(self.tensor("king_hand_embeddings")?, &features.king_hand, 4);
        let pair = embedding_sum(self.tensor("pair_hash_embeddings")?, &features.pair, 8);
        let projected = linear(
            self.tensor("scalar_projection.weight")?,
            self.tensor("scalar_projection.bias")?,
            &features.scalars,
            32,
            false,
        );
        let mut trunk_input = Vec::with_capacity(self.variant.trunk_input());
        trunk_input.extend(king_piece);
        trunk_input.extend(king_hand);
        trunk_input.extend(pair);
        if self.variant == Osaval02Variant::FactorizedPairTriplePolicyScore {
            trunk_input.extend(embedding_sum(
                self.tensor("triple_hash_embeddings")?,
                &features.triple,
                8,
            ));
        }
        trunk_input.extend(projected);
        let hidden = linear(
            self.tensor("trunk.0.weight")?,
            self.tensor("trunk.0.bias")?,
            &trunk_input,
            TRUNK_HIDDEN,
            true,
        );
        let hidden = linear(
            self.tensor("trunk.1.weight")?,
            self.tensor("trunk.1.bias")?,
            &hidden,
            TRUNK_HIDDEN,
            true,
        );
        let outputs = linear(
            self.tensor("value_heads.weight")?,
            self.tensor("value_heads.bias")?,
            &hidden,
            self.variant.value_outputs(),
            false,
        );
        Ok((legal_moves, features, hidden, outputs))
    }

    fn score_components(
        &self,
        outputs: &[f64],
    ) -> Result<(f64, f64, &'static str, usize), Osaval02Error> {
        let (transformed, raw_score, source_head, mate_offset) = match self.variant {
            Osaval02Variant::SparsePairPolicyWdl => {
                let wdl_values = softmax(&outputs[..3]);
                let epsilon = f64::from(self.wdl_epsilon);
                let transformed = ((wdl_values[2] + epsilon) / (wdl_values[0] + epsilon)).ln();
                (transformed, transformed, "wdl_log_odds", 3)
            }
            Osaval02Variant::FactorizedPairTriplePolicyScore => {
                let transformed = outputs[3];
                let raw = transformed
                    .abs()
                    .min(1.0)
                    .mul_add(3000.0_f64.ln_1p(), 0.0)
                    .exp_m1()
                    .copysign(transformed);
                (transformed, raw, "direct_transformed_score", 4)
            }
        };
        if !transformed.is_finite() || !raw_score.is_finite() {
            return Err(Osaval02Error::Invalid(
                "OSAVAL02 score output is not finite",
            ));
        }
        Ok((transformed, raw_score, source_head, mate_offset))
    }

    fn tensor(&self, name: &str) -> Result<&Tensor, Osaval02Error> {
        self.tensors.get(name).ok_or(Osaval02Error::Invalid(
            "OSAVAL02 required tensor is missing",
        ))
    }
}

/// Search-facing adapter for a validated OSAVAL02 evaluator.
///
/// The adapter keeps the model's history contract beside the evaluator and exposes only the
/// two values search needs: the calibrated current-side score and legal-root policy logits.
/// Search still owns terminal, mate, and quiescence semantics; no OSAVAL01 or handcrafted score
/// is substituted when this adapter is selected.
#[derive(Clone, Debug)]
pub struct Osaval02SearchAdapter {
    evaluator: Arc<Osaval02Evaluator>,
    history: Osaval02History,
}

impl Osaval02SearchAdapter {
    /// Build an adapter with the standalone history facts required for a USI/browser root.
    #[must_use]
    pub fn new(evaluator: Arc<Osaval02Evaluator>) -> Self {
        Self {
            evaluator,
            history: Osaval02History::default(),
        }
    }

    /// Build an adapter with an explicit validated-history contract.
    #[must_use]
    pub const fn with_history(evaluator: Arc<Osaval02Evaluator>, history: Osaval02History) -> Self {
        Self { evaluator, history }
    }

    /// Return the immutable evaluator identity used by runtime evidence.
    #[must_use]
    pub fn identity(&self) -> &Osaval02Identity {
        self.evaluator.identity()
    }

    /// Return the configured bounded history facts.
    #[must_use]
    pub const fn history(&self) -> Osaval02History {
        self.history
    }

    /// Run the strict OSAVAL02 inference and return its calibrated current-side search score.
    pub fn evaluate(&self, position: &Position) -> Result<i32, Osaval02Error> {
        self.evaluator.evaluate_score(position, self.history)
    }

    /// Run the strict OSAVAL02 inference and return the legal policy logit for one move.
    pub fn policy_logit(
        &self,
        position: &Position,
        movement: Move,
    ) -> Result<Option<f64>, Osaval02Error> {
        Ok(self
            .evaluator
            .infer(position, self.history)?
            .policy_logit(movement))
    }

    /// Run one root inference for policy-guided move ordering without changing search score
    /// semantics.
    pub fn infer(&self, position: &Position) -> Result<Osaval02Inference, Osaval02Error> {
        self.evaluator.infer(position, self.history)
    }
}

/// Complete deterministic inference document returned by native and Wasm callers.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Osaval02Inference {
    schema: &'static str,
    identity: InferenceIdentity,
    position_sha256: String,
    feature_sha256: String,
    history: Osaval02History,
    legal_moves: Vec<PolicyOutput>,
    wdl: WdlOutput,
    score: ScoreOutput,
    mate: MateOutput,
    uncertainty: UncertaintyOutput,
    terminal: TerminalOutput,
}

impl Osaval02Inference {
    /// Return the calibrated current-side score. Mate scores are never derived from this value.
    #[must_use]
    pub const fn calibrated_score_cp(&self) -> i32 {
        self.score.calibrated_cp
    }

    /// Return the legal-root policy logit for a move, if that move is legal at this root.
    #[must_use]
    pub fn policy_logit(&self, movement: Move) -> Option<f64> {
        let index = encode_move(movement);
        self.legal_moves
            .iter()
            .find(|policy| policy.index == index)
            .map(|policy| policy.logit)
    }

    fn validate_finite(&self) -> Result<(), Osaval02Error> {
        let values = [
            self.wdl.loss,
            self.wdl.draw,
            self.wdl.win,
            self.score.transformed,
            self.mate.probabilities.mated,
            self.mate.probabilities.no_mate_label,
            self.mate.probabilities.mating,
            self.mate.distance_plies,
            self.uncertainty.log_variance,
            self.uncertainty.variance,
        ];
        if values.iter().any(|value| !value.is_finite())
            || self
                .legal_moves
                .iter()
                .any(|policy| !policy.logit.is_finite())
        {
            return Err(Osaval02Error::Invalid(
                "OSAVAL02 inference produced NaN or infinity",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct InferenceIdentity {
    format_version: u32,
    variant_id: &'static str,
    quantization: &'static str,
    artifact_sha256: String,
    weight_payload_sha256: String,
}

#[derive(Clone, Debug, Serialize)]
struct WdlOutput {
    loss: f64,
    draw: f64,
    win: f64,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ScoreOutput {
    source_head: &'static str,
    transformed: f64,
    calibrated_cp: i32,
    perspective: &'static str,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct MateOutput {
    class: &'static str,
    probabilities: MateProbabilities,
    distance_plies: f64,
    score_conversion: &'static str,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct MateProbabilities {
    mated: f64,
    no_mate_label: f64,
    mating: f64,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct UncertaintyOutput {
    log_variance: f64,
    variance: f64,
    mixed_into_score: bool,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct PolicyOutput {
    #[serde(rename = "move")]
    movement: String,
    index: usize,
    logit: f64,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct TerminalOutput {
    kind: Option<&'static str>,
    search_score_cp: Option<i32>,
}

#[derive(Debug)]
struct FeatureSet {
    king_piece: Vec<(usize, i8)>,
    king_hand: Vec<(usize, i8)>,
    pair: Vec<(usize, i8)>,
    triple: Vec<(usize, i8)>,
    scalars: Vec<f64>,
    checksum: String,
}

fn tensor_specs(variant: Osaval02Variant) -> Vec<TensorSpec> {
    let mut specs = vec![
        tensor_spec("king_piece_embeddings", &[367_416, 4]),
        tensor_spec("king_hand_embeddings", &[43_092, 4]),
        tensor_spec("pair_hash_embeddings", &[65_536, 8]),
    ];
    if variant == Osaval02Variant::FactorizedPairTriplePolicyScore {
        specs.push(tensor_spec("triple_hash_embeddings", &[32_768, 8]));
    }
    specs.extend([
        tensor_spec("scalar_projection.weight", &[32, 64]),
        tensor_spec("scalar_projection.bias", &[32]),
        tensor_spec("trunk.0.weight", &[128, variant.trunk_input()]),
        tensor_spec("trunk.0.bias", &[128]),
        tensor_spec("trunk.1.weight", &[128, 128]),
        tensor_spec("trunk.1.bias", &[128]),
        tensor_spec("value_heads.weight", &[variant.value_outputs(), 128]),
        tensor_spec("value_heads.bias", &[variant.value_outputs()]),
        tensor_spec("policy.move_embeddings", &[POLICY_CLASSES, MOVE_DIMENSION]),
        tensor_spec("policy.context_weight", &[MOVE_DIMENSION, TRUNK_HIDDEN]),
        tensor_spec("policy.context_bias", &[MOVE_DIMENSION]),
        tensor_spec("policy.move_offset", &[MOVE_DIMENSION]),
        tensor_spec("policy.log_temperature", &[1]),
    ]);
    specs
}

fn tensor_spec(name: &'static str, dimensions: &[usize]) -> TensorSpec {
    TensorSpec {
        name,
        dimensions: dimensions.to_vec(),
    }
}

#[expect(
    clippy::too_many_lines,
    reason = "feature emission stays together so Python/Rust wire order is auditable"
)]
fn encode_features(
    position: &Position,
    legal_moves: &[Move],
    history: Osaval02History,
    variant: Osaval02Variant,
) -> Result<FeatureSet, Osaval02Error> {
    let pieces = Square::all()
        .filter_map(|square| position.piece_at(square).map(|piece| (piece, square)))
        .collect::<Vec<_>>();
    let us = position.side_to_move();
    let kings = [
        position
            .king_square(us)
            .ok_or(Osaval02Error::Invalid("current-side king is missing"))?,
        position
            .king_square(us.opposite())
            .ok_or(Osaval02Error::Invalid("opponent king is missing"))?,
    ];
    let king_pieces = [
        position
            .piece_at(kings[0])
            .ok_or(Osaval02Error::Invalid("current-side king piece is missing"))?,
        position.piece_at(kings[1]).ok_or(Osaval02Error::Invalid(
            "opponent-side king piece is missing",
        ))?,
    ];
    let attack_masks = pieces
        .iter()
        .map(|(piece, square)| attack_mask(position, *piece, *square))
        .collect::<Vec<_>>();
    let mut king_piece = Vec::with_capacity(pieces.len() * 2);
    for (piece, square) in &pieces {
        let owner = usize::from(piece.side != us);
        for (king_role, king) in kings.iter().enumerate() {
            let index = ((((owner * 14 + piece.kind.index()) * 2 + king_role) * 81 + king.index())
                * 81)
                + square.index();
            king_piece.push((index, 1));
        }
    }
    let mut king_hand = Vec::new();
    for side in [us, us.opposite()] {
        let owner = usize::from(side != us);
        for hand_piece in HandPiece::ALL {
            let count = usize::from(position.hand(side).count(hand_piece));
            if count == 0 {
                continue;
            }
            for (king_role, king) in kings.iter().enumerate() {
                let index = ((((owner * 7 + hand_piece.index()) * 19 + count) * 2 + king_role)
                    * 81)
                    + king.index();
                king_hand.push((index, 1));
            }
        }
    }
    let mut pinned = [false; 81];
    for (piece, square) in &pieces {
        pinned[square.index()] = is_pinned(position, *piece, *square);
    }
    let mut pair = Vec::new();
    for left_index in 0..pieces.len() {
        let (left_piece, left_square) = pieces[left_index];
        for (right_index, (right_piece, right_square)) in
            pieces.iter().copied().enumerate().skip(left_index + 1)
        {
            let flags = pair_flags(
                left_index,
                right_index,
                (left_piece, left_square),
                (right_piece, right_square),
                kings,
                us,
                &pinned,
                &attack_masks,
            );
            let mut key = vec![
                1,
                relative_owner(left_piece, us),
                u8_from_usize(left_piece.kind.index()),
                u8_from_usize(left_square.index()),
                relative_owner(right_piece, us),
                u8_from_usize(right_piece.kind.index()),
                u8_from_usize(right_square.index()),
            ];
            for square in [left_square, right_square] {
                for king in kings {
                    key.extend(signed_offsets(square, king));
                }
            }
            key.extend(flags.to_le_bytes());
            pair.push(signed_bucket(&key, 65_536));
        }
    }
    let mut drop_count = 0_usize;
    for movement in legal_moves {
        let Move::Drop { piece, to } = movement else {
            continue;
        };
        drop_count += 1;
        let count = position.hand(us).count(*piece);
        let mut key = vec![
            2,
            0,
            u8_from_usize(piece.index()),
            count,
            u8_from_usize(to.index()),
        ];
        for king in kings {
            key.extend(signed_offsets(*to, king));
        }
        let mut flags = u8::from(position.square_is_attacked(*to, us));
        flags |= u8::from(position.square_is_attacked(*to, us.opposite())) << 1;
        flags |= u8::from(kings.iter().any(|king| chebyshev(*to, *king) <= 2)) << 2;
        key.push(flags);
        pair.push(signed_bucket(&key, 65_536));
    }
    pair.sort_unstable();

    let mut triple = Vec::new();
    if variant == Osaval02Variant::FactorizedPairTriplePolicyScore {
        let mut candidates = Vec::new();
        for first in 0..pieces.len() {
            for second in first + 1..pieces.len() {
                for third in second + 1..pieces.len() {
                    let group = [pieces[first], pieces[second], pieces[third]];
                    let Some(category) = triple_category(
                        [first, second, third],
                        &pieces,
                        kings,
                        king_pieces,
                        &pinned,
                        &attack_masks,
                    ) else {
                        continue;
                    };
                    let squares = [group[0].1.index(), group[1].1.index(), group[2].1.index()];
                    candidates.push((category, squares, [first, second, third]));
                }
            }
        }
        candidates.sort_by(|left, right| (left.0, left.1).cmp(&(right.0, right.1)));
        let mut seen = HashSet::new();
        for (category, _, piece_indices) in candidates {
            let mut key = vec![3, category];
            for index in piece_indices {
                let (piece, square) = pieces[index];
                key.extend([
                    relative_owner(piece, us),
                    u8_from_usize(piece.kind.index()),
                    u8_from_usize(square.index()),
                ]);
                for king in kings {
                    key.extend(signed_offsets(square, king));
                }
            }
            if !seen.insert(key.clone()) {
                continue;
            }
            triple.push(signed_bucket(&key, 32_768));
            if triple.len() == MAX_TRIPLES {
                break;
            }
        }
    }
    let scalars = scalar_features(
        position,
        legal_moves,
        history,
        &pieces,
        kings,
        &pinned,
        &attack_masks,
        pair.len(),
        triple.len(),
        drop_count,
    );
    let mut digest = Sha256::new();
    for (tag, rows) in [
        (1_u8, &king_piece),
        (2, &king_hand),
        (3, &pair),
        (4, &triple),
    ] {
        for (index, sign) in rows {
            digest.update([tag]);
            digest.update(u32::try_from(*index).unwrap_or(u32::MAX).to_le_bytes());
            digest.update(sign.to_le_bytes());
        }
    }
    for scalar in &scalars {
        digest.update(f64_to_f32(*scalar).to_le_bytes());
    }
    Ok(FeatureSet {
        king_piece,
        king_hand,
        pair,
        triple,
        scalars,
        checksum: hex(&digest.finalize()),
    })
}

#[expect(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "the frozen 64-scalar surface is clearer with explicit source values"
)]
fn scalar_features(
    position: &Position,
    legal_moves: &[Move],
    history: Osaval02History,
    pieces: &[(Piece, Square)],
    kings: [Square; 2],
    pinned: &[bool; 81],
    attack_masks: &[u128],
    pair_count: usize,
    triple_count: usize,
    drop_count: usize,
) -> Vec<f64> {
    const HAND_MAXIMA: [f64; 7] = [18.0, 4.0, 4.0, 4.0, 4.0, 2.0, 2.0];
    let us = position.side_to_move();
    let mut values = Vec::with_capacity(SCALAR_INPUTS);
    for side in [us, us.opposite()] {
        let mut counts = [0_u8; 14];
        for (piece, _) in pieces {
            if piece.side == side {
                counts[piece.kind.index()] = counts[piece.kind.index()].saturating_add(1);
            }
        }
        values.extend(counts.map(|count| f64::from(count) / 18.0));
    }
    for side in [us, us.opposite()] {
        values.extend(
            HandPiece::ALL.map(|piece| {
                f64::from(position.hand(side).count(piece)) / HAND_MAXIMA[piece.index()]
            }),
        );
    }
    let attacks_us = side_attack_mask(pieces, attack_masks, us);
    let attacks_them = side_attack_mask(pieces, attack_masks, us.opposite());
    let attacked_us = attacks_us.count_ones();
    let attacked_them = attacks_them.count_ones();
    values.extend([
        f64::from(attacked_us) / 81.0,
        f64::from(attacked_them) / 81.0,
    ]);
    values.push(
        usize_f64(
            king_zone(kings[0])
                .into_iter()
                .filter(|square| attack_mask_contains(attacks_them, *square))
                .count(),
        ) / 25.0,
    );
    values.push(
        usize_f64(
            king_zone(kings[1])
                .into_iter()
                .filter(|square| attack_mask_contains(attacks_us, *square))
                .count(),
        ) / 25.0,
    );
    values.push(usize_f64(legal_moves.len()) / 600.0);
    values.push(f64::from(u8::from(attack_mask_contains(
        attacks_them,
        kings[0],
    ))));
    values.push(f64::from(u8::from(attack_mask_contains(
        attacks_us, kings[1],
    ))));
    values.push(
        usize_f64(
            pieces
                .iter()
                .filter(|(piece, square)| piece.side == us && pinned[square.index()])
                .count(),
        ) / 20.0,
    );
    values.push(
        usize_f64(
            pieces
                .iter()
                .filter(|(piece, square)| piece.side != us && pinned[square.index()])
                .count(),
        ) / 20.0,
    );
    values.push(
        usize_f64(
            pieces
                .iter()
                .filter(|(piece, _)| piece.side == us && piece.kind.index() >= 8)
                .count(),
        ) / 10.0,
    );
    values.push(
        usize_f64(
            pieces
                .iter()
                .filter(|(piece, _)| piece.side != us && piece.kind.index() >= 8)
                .count(),
        ) / 10.0,
    );
    values.push(usize_f64(pieces.len()) / 40.0);
    let hands = [Side::Black, Side::White]
        .into_iter()
        .map(|side| usize::from(position.hand(side).total()))
        .sum::<usize>();
    values.push(usize_f64(hands) / 38.0);
    values.push(f64::from(position.move_number().saturating_sub(1).min(512)) / 512.0);
    values.push(f64::from(u8::from(history.available)));
    values.push(f64::from(history.repetition_count) / 4.0);
    values.push(f64::from(u8::from(history.continuous_check_by_us)));
    values.push(f64::from(u8::from(history.continuous_check_by_them)));
    values.push(usize_f64(pair_count.min(780)) / 780.0);
    values.push(usize_f64(triple_count) / 256.0);
    values.push(usize_f64(drop_count) / 567.0);
    values.push(1.0);
    assert_eq!(values.len(), SCALAR_INPUTS);
    values
        .into_iter()
        .map(|value| f64::from(f64_to_f32(value)))
        .collect()
}

fn triple_category(
    piece_indices: [usize; 3],
    all_pieces: &[(Piece, Square)],
    kings: [Square; 2],
    king_pieces: [Piece; 2],
    pinned: &[bool; 81],
    attack_masks: &[u128],
) -> Option<u8> {
    let pieces = piece_indices.map(|index| all_pieces[index]);
    for king in kings {
        if pieces.iter().any(|(_, square)| *square == king)
            && pieces
                .iter()
                .all(|(_, square)| *square == king || chebyshev(*square, king) <= 2)
        {
            return Some(0);
        }
    }
    for (king_index, king) in kings.into_iter().enumerate() {
        let king_piece = king_pieces[king_index];
        if !pieces.iter().any(|(_, square)| *square == king) {
            continue;
        }
        for (pinned_piece, pinned_square) in pieces {
            if !pinned[pinned_square.index()] || pinned_piece.side != king_piece.side {
                continue;
            }
            if piece_indices.iter().any(|index| {
                let (other, _) = all_pieces[*index];
                other.side != king_piece.side
                    && attack_mask_contains(attack_masks[*index], pinned_square)
            }) {
                return Some(1);
            }
        }
        if piece_indices.iter().any(|index| {
            let (piece, _) = all_pieces[*index];
            piece.side != king_piece.side && attack_mask_contains(attack_masks[*index], king)
        }) {
            return Some(2);
        }
    }
    for (target_index, (_, target_square)) in piece_indices
        .iter()
        .map(|index| (*index, all_pieces[*index]))
    {
        let attackers = piece_indices
            .iter()
            .filter(|index| {
                **index != target_index
                    && attack_mask_contains(attack_masks[**index], target_square)
            })
            .count();
        if attackers == 2 {
            return Some(3);
        }
    }
    let edges = [(0, 1), (0, 2), (1, 2)]
        .into_iter()
        .filter(|(left, right)| manhattan(pieces[*left].1, pieces[*right].1) <= 4)
        .count();
    (edges >= 2).then_some(4)
}

fn pair_flags(
    left_index: usize,
    right_index: usize,
    left: (Piece, Square),
    right: (Piece, Square),
    kings: [Square; 2],
    us: Side,
    pinned: &[bool; 81],
    attack_masks: &[u128],
) -> u16 {
    let left_attacks = attack_mask_contains(attack_masks[left_index], right.1);
    let right_attacks = attack_mask_contains(attack_masks[right_index], left.1);
    let mut flags = u16::from(left_attacks);
    flags |= u16::from(right_attacks) << 1;
    flags |= u16::from(left.0.side == right.0.side && (left_attacks || right_attacks)) << 2;
    flags |= u16::from(pinned[left.1.index()]) << 3;
    flags |= u16::from(pinned[right.1.index()]) << 4;
    let left_king = kings[usize::from(left.0.side == us)];
    let right_king = kings[usize::from(right.0.side == us)];
    flags |= u16::from(attack_mask_contains(attack_masks[left_index], left_king)) << 5;
    flags |= u16::from(attack_mask_contains(attack_masks[right_index], right_king)) << 6;
    let common = attack_masks[left_index] & attack_masks[right_index] != 0;
    flags |= u16::from(common) << 7;
    flags |= u16::from(kings.iter().any(|king| chebyshev(left.1, *king) <= 2)) << 8;
    flags |= u16::from(kings.iter().any(|king| chebyshev(right.1, *king) <= 2)) << 9;
    flags
}

fn attack_mask(position: &Position, piece: Piece, square: Square) -> u128 {
    Square::all().fold(0_u128, |mask, target| {
        if position.piece_attacks(square, piece, target) {
            mask | (1_u128 << target.index())
        } else {
            mask
        }
    })
}

fn attack_mask_contains(mask: u128, square: Square) -> bool {
    mask & (1_u128 << square.index()) != 0
}

fn side_attack_mask(pieces: &[(Piece, Square)], attack_masks: &[u128], side: Side) -> u128 {
    pieces
        .iter()
        .zip(attack_masks)
        .filter(|((piece, _), _)| piece.side == side)
        .fold(0_u128, |mask, (_, attacks)| mask | attacks)
}

fn is_pinned(position: &Position, piece: Piece, square: Square) -> bool {
    if piece.kind == PieceKind::King {
        return false;
    }
    let Some(king) = position.king_square(piece.side) else {
        return false;
    };
    let file_delta = i16::from(square.file()) - i16::from(king.file());
    let rank_delta = i16::from(square.rank()) - i16::from(king.rank());
    if !(file_delta == 0 || rank_delta == 0 || file_delta.abs() == rank_delta.abs()) {
        return false;
    }
    if !path_clear(position, king, square) {
        return false;
    }
    let file_step = sign_i16(file_delta);
    let rank_step = sign_i16(rank_delta);
    let mut file = i16::from(square.file()) + file_step;
    let mut rank = i16::from(square.rank()) + rank_step;
    while (1..=9).contains(&file) && (1..=9).contains(&rank) {
        let Some(candidate_square) = Square::new(
            u8::try_from(file).expect("bounded file"),
            u8::try_from(rank).expect("bounded rank"),
        ) else {
            return false;
        };
        if let Some(candidate) = position.piece_at(candidate_square) {
            if candidate.side == piece.side {
                return false;
            }
            let orthogonal = file_step == 0 || rank_step == 0;
            if orthogonal && matches!(candidate.kind, PieceKind::Rook | PieceKind::Dragon) {
                return true;
            }
            if !orthogonal && matches!(candidate.kind, PieceKind::Bishop | PieceKind::Horse) {
                return true;
            }
            if file_step == 0 && candidate.kind == PieceKind::Lance {
                let lance_forward = i16::from(candidate.side.forward());
                return (i16::from(king.rank()) - rank).signum() == lance_forward;
            }
            return false;
        }
        file += file_step;
        rank += rank_step;
    }
    false
}

fn path_clear(position: &Position, origin: Square, target: Square) -> bool {
    let file_step = sign_i16(i16::from(target.file()) - i16::from(origin.file()));
    let rank_step = sign_i16(i16::from(target.rank()) - i16::from(origin.rank()));
    let mut file = i16::from(origin.file()) + file_step;
    let mut rank = i16::from(origin.rank()) + rank_step;
    while (file, rank) != (i16::from(target.file()), i16::from(target.rank())) {
        let square = Square::new(
            u8::try_from(file).expect("path file remains on board"),
            u8::try_from(rank).expect("path rank remains on board"),
        )
        .expect("aligned path remains on board");
        if position.piece_at(square).is_some() {
            return false;
        }
        file += file_step;
        rank += rank_step;
    }
    true
}

fn signed_bucket(key: &[u8], buckets: usize) -> (usize, i8) {
    let mut digest = Sha256::new();
    digest.update(FEATURE_HASH_DOMAIN);
    digest.update(HASH_SEED.to_le_bytes());
    digest.update(key);
    let digest = digest.finalize();
    let bucket = usize::try_from(u32::from_le_bytes(
        digest[..4]
            .try_into()
            .expect("SHA-256 prefix always has four bytes"),
    ))
    .unwrap_or(usize::MAX)
        % buckets;
    (bucket, if digest[4] & 1 == 0 { 1 } else { -1 })
}

fn embedding_sum(tensor: &Tensor, rows: &[(usize, i8)], dimension: usize) -> Vec<f64> {
    let mut result = vec![0.0; dimension];
    for (row, sign) in rows {
        let base = row * dimension;
        for (index, value) in result.iter_mut().enumerate() {
            *value += f64::from(*sign) * tensor.value(base + index);
        }
    }
    if !rows.is_empty() {
        let denominator = usize_f64(rows.len());
        for value in &mut result {
            *value /= denominator;
        }
    }
    result
}

fn linear(
    weights: &Tensor,
    biases: &Tensor,
    inputs: &[f64],
    outputs: usize,
    relu: bool,
) -> Vec<f64> {
    debug_assert_eq!(weights.dimensions, [outputs, inputs.len()]);
    debug_assert_eq!(biases.dimensions, [outputs]);
    (0..outputs)
        .map(|row| {
            let base = row * inputs.len();
            let mut total = biases.value(row);
            for (column, value) in inputs.iter().enumerate() {
                total += *value * weights.value(base + column);
            }
            if relu { total.max(0.0) } else { total }
        })
        .collect()
}

fn softmax(values: &[f64]) -> Vec<f64> {
    let maximum = values
        .iter()
        .copied()
        .max_by(f64::total_cmp)
        .expect("softmax input is nonempty");
    let exponentials = values
        .iter()
        .map(|value| (*value - maximum).exp())
        .collect::<Vec<_>>();
    let denominator: f64 = exponentials.iter().sum();
    exponentials
        .into_iter()
        .map(|value| value / denominator)
        .collect()
}

/// Encode a legal move in the frozen 13,689-class bijection.
#[must_use]
pub const fn encode_osaval02_move(movement: Move) -> usize {
    encode_move(movement)
}

const fn encode_move(movement: Move) -> usize {
    match movement {
        Move::Normal { from, to, promote } => {
            (from.index() * 81 + to.index()) * 2 + promote as usize
        }
        Move::Drop { piece, to } => 13_122 + drop_index(piece) * 81 + to.index(),
    }
}

const fn drop_index(piece: HandPiece) -> usize {
    match piece {
        HandPiece::Rook => 0,
        HandPiece::Bishop => 1,
        HandPiece::Gold => 2,
        HandPiece::Silver => 3,
        HandPiece::Knight => 4,
        HandPiece::Lance => 5,
        HandPiece::Pawn => 6,
    }
}

fn relative_owner(piece: Piece, us: Side) -> u8 {
    u8::from(piece.side != us)
}

fn signed_offsets(square: Square, king: Square) -> [u8; 2] {
    let file = i16::from(square.file()) - i16::from(king.file());
    let rank = i16::from(square.rank()) - i16::from(king.rank());
    [
        i8::try_from(file)
            .expect("board file delta fits i8")
            .cast_unsigned(),
        i8::try_from(rank)
            .expect("board rank delta fits i8")
            .cast_unsigned(),
    ]
}

fn king_zone(king: Square) -> Vec<Square> {
    let mut result = Vec::with_capacity(25);
    for rank in king.rank().saturating_sub(2).max(1)..=king.rank().saturating_add(2).min(9) {
        for file in king.file().saturating_sub(2).max(1)..=king.file().saturating_add(2).min(9) {
            result.push(Square::new(file, rank).expect("bounded king zone"));
        }
    }
    result
}

fn chebyshev(left: Square, right: Square) -> usize {
    usize::from(
        left.file()
            .abs_diff(right.file())
            .max(left.rank().abs_diff(right.rank())),
    )
}

fn manhattan(left: Square, right: Square) -> usize {
    usize::from(left.file().abs_diff(right.file()) + left.rank().abs_diff(right.rank()))
}

fn canonical_state(position: &Position) -> Result<String, Osaval02Error> {
    let sfen = to_sfen(position);
    sfen.rsplit_once(' ')
        .map(|(state, _)| state.to_owned())
        .ok_or(Osaval02Error::Invalid(
            "canonical SFEN lacks its move number",
        ))
}

fn round_and_clamp(value: f64) -> i32 {
    let rounded = value
        .clamp(f64::from(-MAX_NON_MATE_CP), f64::from(MAX_NON_MATE_CP))
        .round();
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the value is finite and clamped to a small i32 range"
    )]
    {
        rounded as i32
    }
}

fn read_tensor_name(bytes: &[u8], offset: usize) -> Result<String, Osaval02Error> {
    read_fixed_text(bytes, offset, 48, "tensor name", false)
}

fn read_fixed_text(
    bytes: &[u8],
    offset: usize,
    width: usize,
    label: &'static str,
    allow_full_width: bool,
) -> Result<String, Osaval02Error> {
    let field = bytes
        .get(offset..offset + width)
        .ok_or(Osaval02Error::Invalid("OSAVAL02 is truncated"))?;
    let value_bytes = match field.iter().position(|byte| *byte == 0) {
        Some(nul) => {
            if nul == 0 || field[nul..].iter().any(|byte| *byte != 0) {
                return Err(Osaval02Error::InvalidOwned(format!(
                    "OSAVAL02 {label} padding is invalid"
                )));
            }
            &field[..nul]
        }
        None if allow_full_width && field.iter().all(|byte| (0x21..=0x7e).contains(byte)) => field,
        None => {
            return Err(Osaval02Error::InvalidOwned(format!(
                "OSAVAL02 {label} lacks NUL padding"
            )));
        }
    };
    let value = std::str::from_utf8(value_bytes)
        .map_err(|_| Osaval02Error::InvalidOwned(format!("OSAVAL02 {label} is not ASCII")))?;
    if !value.bytes().all(|byte| (0x21..=0x7e).contains(&byte)) {
        return Err(Osaval02Error::InvalidOwned(format!(
            "OSAVAL02 {label} is not printable ASCII"
        )));
    }
    Ok(value.to_owned())
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, Osaval02Error> {
    let encoded = bytes
        .get(offset..offset + 4)
        .ok_or(Osaval02Error::Invalid("OSAVAL02 is truncated"))?;
    Ok(u32::from_le_bytes(
        encoded.try_into().expect("four-byte range converts to u32"),
    ))
}

fn read_i32(bytes: &[u8], offset: usize) -> Result<i32, Osaval02Error> {
    let encoded = bytes
        .get(offset..offset + 4)
        .ok_or(Osaval02Error::Invalid("OSAVAL02 is truncated"))?;
    Ok(i32::from_le_bytes(
        encoded.try_into().expect("four-byte range converts to i32"),
    ))
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64, Osaval02Error> {
    let encoded = bytes
        .get(offset..offset + 8)
        .ok_or(Osaval02Error::Invalid("OSAVAL02 is truncated"))?;
    Ok(u64::from_le_bytes(
        encoded
            .try_into()
            .expect("eight-byte range converts to u64"),
    ))
}

fn read_f32(bytes: &[u8], offset: usize) -> Result<f32, Osaval02Error> {
    Ok(f32::from_bits(read_u32(bytes, offset)?))
}

fn read_usize_u32(bytes: &[u8], offset: usize) -> Result<usize, Osaval02Error> {
    usize::try_from(read_u32(bytes, offset)?)
        .map_err(|_| Osaval02Error::Invalid("OSAVAL02 integer requests an oversized allocation"))
}

fn read_usize_u64(bytes: &[u8], offset: usize) -> Result<usize, Osaval02Error> {
    usize::try_from(read_u64(bytes, offset)?)
        .map_err(|_| Osaval02Error::Invalid("OSAVAL02 integer requests an oversized allocation"))
}

fn f64_to_f32(value: f64) -> f32 {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "frozen scalar features are small finite normalized values"
    )]
    {
        value as f32
    }
}

fn usize_f64(value: usize) -> f64 {
    f64::from(u32::try_from(value).unwrap_or(u32::MAX))
}

fn u8_from_usize(value: usize) -> u8 {
    u8::try_from(value).expect("frozen board/table field fits u8")
}

const fn sign_i16(value: i16) -> i16 {
    if value < 0 {
        -1
    } else if value > 0 {
        1
    } else {
        0
    }
}

fn hex(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}
