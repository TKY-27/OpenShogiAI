//! Provenance-bound `OpenShogiAI` opening-book v2 runtime.

use std::{
    collections::BTreeMap,
    io::{BufRead, BufReader, Cursor, Read},
    path::Path,
};

use flate2::bufread::MultiGzDecoder;
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::{AnchoredFile, Move, Position, parse_sfen, parse_usi_move, to_sfen, to_usi_move};

pub const OPENING_BOOK_SCHEMA: &str = "open_shogi_opening_book/v2";
const RULE_PROFILE: &str = "standard-shogi/v1";
const MAX_COMPRESSED_BYTES: usize = 64 * 1024 * 1024;
const MAX_DECOMPRESSED_BYTES: u64 = 256 * 1024 * 1024;
const MAX_LINE_BYTES: usize = 1024 * 1024;
const MAX_RECORDS: usize = 1_000_000;

/// Opening style is enforced only by book selection, never legal move generation.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum OpeningProfile {
    Unrestricted,
    IbishaPreferred,
    #[default]
    IbishaStrict,
}

impl OpeningProfile {
    /// Parses one closed opening-profile name.
    ///
    /// # Errors
    ///
    /// Returns an error for names outside the three versioned profiles.
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "unrestricted" => Ok(Self::Unrestricted),
            "ibisha_preferred" | "ibisha-preferred" => Ok(Self::IbishaPreferred),
            "ibisha_strict" | "ibisha-strict" => Ok(Self::IbishaStrict),
            _ => Err(
                "opening profile must be unrestricted, ibisha_preferred, or ibisha_strict"
                    .to_owned(),
            ),
        }
    }

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Unrestricted => "unrestricted",
            Self::IbishaPreferred => "ibisha_preferred",
            Self::IbishaStrict => "ibisha_strict",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OpeningPolicy {
    pub profile: OpeningProfile,
    pub minimum_sample_count: u64,
    pub maximum_teacher_loss_cp: i32,
}

impl Default for OpeningPolicy {
    fn default() -> Self {
        Self {
            profile: OpeningProfile::IbishaStrict,
            minimum_sample_count: 2,
            maximum_teacher_loss_cp: 80,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OpeningBookChoice {
    pub movement: Move,
    pub sample_count: u64,
    pub teacher_score_cp: i32,
    pub teacher_depth: u8,
    pub teacher_nodes: u64,
    pub opening_classification: String,
    pub provenance_references: Vec<String>,
}

#[derive(Debug)]
pub struct OpeningBookV2 {
    entries: BTreeMap<String, Vec<OpeningBookChoice>>,
    positions: usize,
    candidates: usize,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct Record {
    schema: String,
    state_key: String,
    state_sfen: String,
    rule_profile: String,
    build_version: String,
    provenance_references: Vec<String>,
    candidates: Vec<Candidate>,
    #[serde(rename = "recordChecksum")]
    checksum: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct Candidate {
    move_usi: String,
    sample_count: u64,
    source_distribution: BTreeMap<String, u64>,
    black_results: Results,
    white_results: Results,
    teacher_score_cp: Option<i32>,
    score_uncertainty_cp: Option<u32>,
    teacher_depth: Option<u8>,
    teacher_nodes: Option<u64>,
    opening_classification: String,
    provenance_references: Vec<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Results {
    wins: u64,
    losses: u64,
    draws: u64,
    unknown: u64,
}

impl OpeningBookV2 {
    /// Opens a regular non-symlink file through the retained secure descriptor boundary.
    ///
    /// # Errors
    ///
    /// Returns an error for an unsafe path, unstable read, oversized file, or invalid book.
    pub fn load_file(path: impl AsRef<Path>) -> Result<Self, String> {
        let mut file =
            AnchoredFile::open_existing(path.as_ref()).map_err(|error| error.to_string())?;
        let identity = file.stable_identity().map_err(|error| error.to_string())?;
        if identity.length() > u64::try_from(MAX_COMPRESSED_BYTES).unwrap_or(u64::MAX) {
            return Err("opening book exceeds its compressed byte limit".to_owned());
        }
        let mut bytes = Vec::new();
        file.reader()
            .take(u64::try_from(MAX_COMPRESSED_BYTES).unwrap_or(u64::MAX) + 1)
            .read_to_end(&mut bytes)
            .map_err(|error| error.to_string())?;
        file.verify_stable_read(&identity, u64::try_from(bytes.len()).unwrap_or(u64::MAX))
            .map_err(|_| "opening book changed while reading".to_owned())?;
        Self::from_compressed_bytes(&bytes)
    }

    /// Parses and fully validates a compressed book snapshot.
    ///
    /// # Errors
    ///
    /// Returns an error for corrupt compression, incompatible records, invalid provenance,
    /// illegal moves, count mismatches, or exceeded resource bounds.
    pub fn from_compressed_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.is_empty() || bytes.len() > MAX_COMPRESSED_BYTES {
            return Err(format!(
                "opening book compressed size must be 1..={MAX_COMPRESSED_BYTES} bytes"
            ));
        }
        let decoder = MultiGzDecoder::new(BufReader::new(Cursor::new(bytes)));
        let mut reader = BufReader::new(decoder.take(MAX_DECOMPRESSED_BYTES + 1));
        let mut entries = BTreeMap::<String, Vec<OpeningBookChoice>>::new();
        let mut line = Vec::new();
        let mut decoded_bytes = 0_u64;
        let mut positions = 0_usize;
        loop {
            line.clear();
            let read = read_record(&mut reader, &mut line, &mut decoded_bytes)?;
            if read == 0 {
                break;
            }
            while matches!(line.last(), Some(b'\n' | b'\r')) {
                line.pop();
            }
            positions = positions.saturating_add(1);
            if positions > MAX_RECORDS || line.is_empty() {
                return Err("opening book record count/shape is invalid".to_owned());
            }
            let value: Value = serde_json::from_slice(&line)
                .map_err(|error| format!("invalid opening record {positions}: {error}"))?;
            validate_checksum(&value, positions)?;
            let record: Record = serde_json::from_value(value)
                .map_err(|error| format!("invalid opening record {positions}: {error}"))?;
            let (state, choices) = validate_record(record, positions)?;
            if entries.insert(state, choices).is_some() {
                return Err("opening book contains a duplicate canonical position".to_owned());
            }
        }
        if entries.is_empty() {
            return Err("opening book contains no positions".to_owned());
        }
        let candidates = entries.values().map(Vec::len).sum();
        Ok(Self {
            entries,
            positions,
            candidates,
        })
    }

    /// Chooses the strongest teacher-safe candidate under the selected style profile.
    #[must_use]
    pub fn select(&self, position: &Position, policy: OpeningPolicy) -> Option<OpeningBookChoice> {
        if policy.minimum_sample_count == 0 || policy.maximum_teacher_loss_cp < 0 {
            return None;
        }
        let choices = self.entries.get(&state_sfen(position))?;
        let best = choices
            .iter()
            .filter(|choice| choice.sample_count >= policy.minimum_sample_count)
            .map(|choice| choice.teacher_score_cp)
            .max()?;
        choices
            .iter()
            .filter(|choice| choice.sample_count >= policy.minimum_sample_count)
            .filter(|choice| {
                best.saturating_sub(choice.teacher_score_cp) <= policy.maximum_teacher_loss_cp
            })
            .find(|choice| style_accepts(policy.profile, &choice.opening_classification))
            .cloned()
    }

    #[must_use]
    pub const fn positions(&self) -> usize {
        self.positions
    }

    #[must_use]
    pub const fn candidates(&self) -> usize {
        self.candidates
    }
}

#[expect(
    clippy::too_many_lines,
    reason = "record validation keeps the checksum, legality, counts, teacher, and provenance gate together"
)]
fn validate_record(
    record: Record,
    number: usize,
) -> Result<(String, Vec<OpeningBookChoice>), String> {
    if record.schema != OPENING_BOOK_SCHEMA
        || record.rule_profile != RULE_PROFILE
        || record.build_version.is_empty()
        || record.build_version.len() > 128
        || !record.build_version.is_ascii()
        || record.candidates.is_empty()
        || record.candidates.len() > 256
    {
        return Err(format!(
            "opening compatibility/bounds mismatch in record {number}"
        ));
    }
    validate_sha256(&record.state_key, number)?;
    validate_sha256(&record.checksum, number)?;
    validate_references(&record.provenance_references, number)?;
    if record.state_key != sha256_hex(record.state_sfen.as_bytes()) {
        return Err(format!("opening state key mismatch in record {number}"));
    }
    let position = parse_sfen(&format!("{} 1", record.state_sfen))
        .map_err(|error| format!("invalid opening position {number}: {error}"))?;
    if state_sfen(&position) != record.state_sfen {
        return Err(format!("noncanonical opening position in record {number}"));
    }
    let mut choices = Vec::with_capacity(record.candidates.len());
    for candidate in record.candidates {
        let movement = parse_usi_move(&candidate.move_usi)
            .map_err(|error| format!("invalid opening move in record {number}: {error}"))?;
        if !position.legal_moves().contains(&movement) {
            return Err(format!("illegal opening move in record {number}"));
        }
        let source_total = candidate
            .source_distribution
            .values()
            .try_fold(0_u64, |sum, value| sum.checked_add(*value));
        let result_total = total(&candidate.black_results).and_then(|black| {
            total(&candidate.white_results).and_then(|white| black.checked_add(white))
        });
        if candidate.sample_count == 0
            || source_total != Some(candidate.sample_count)
            || result_total != Some(candidate.sample_count)
            || candidate.source_distribution.is_empty()
            || candidate
                .source_distribution
                .values()
                .any(|value| *value == 0)
            || candidate
                .score_uncertainty_cp
                .is_some_and(|value| value > 10_000)
            || !matches!(
                candidate.opening_classification.as_str(),
                "ibisha" | "ibisha-vs-furibisha" | "furibisha" | "unclassified"
            )
        {
            return Err(format!(
                "opening candidate counts are invalid in record {number}"
            ));
        }
        let (Some(score), Some(depth), Some(nodes)) = (
            candidate.teacher_score_cp,
            candidate.teacher_depth,
            candidate.teacher_nodes,
        ) else {
            return Err(format!(
                "opening candidate lacks teacher evidence in record {number}"
            ));
        };
        if !(-32_000..=32_000).contains(&score) || !(1..=64).contains(&depth) || nodes == 0 {
            return Err(format!(
                "opening teacher evidence is invalid in record {number}"
            ));
        }
        validate_references(&candidate.provenance_references, number)?;
        if candidate
            .provenance_references
            .iter()
            .any(|reference| !record.provenance_references.contains(reference))
        {
            return Err(format!(
                "opening candidate provenance mismatch in record {number}"
            ));
        }
        choices.push(OpeningBookChoice {
            movement,
            sample_count: candidate.sample_count,
            teacher_score_cp: score,
            teacher_depth: depth,
            teacher_nodes: nodes,
            opening_classification: candidate.opening_classification,
            provenance_references: candidate.provenance_references,
        });
    }
    choices.sort_by(|left, right| {
        right
            .teacher_score_cp
            .cmp(&left.teacher_score_cp)
            .then_with(|| right.sample_count.cmp(&left.sample_count))
            .then_with(|| to_usi_move(left.movement).cmp(&to_usi_move(right.movement)))
    });
    if choices
        .windows(2)
        .any(|pair| pair[0].movement == pair[1].movement)
    {
        return Err(format!("duplicate opening candidate in record {number}"));
    }
    Ok((record.state_sfen, choices))
}

fn total(results: &Results) -> Option<u64> {
    results
        .wins
        .checked_add(results.losses)
        .and_then(|value| value.checked_add(results.draws))
        .and_then(|value| value.checked_add(results.unknown))
}

fn validate_checksum(value: &Value, number: usize) -> Result<(), String> {
    let expected = value
        .get("recordChecksum")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("opening record {number} lacks checksum"))?;
    validate_sha256(expected, number)?;
    let mut canonical = value.clone();
    canonical
        .as_object_mut()
        .ok_or_else(|| format!("opening record {number} is not an object"))?
        .remove("recordChecksum");
    let bytes = serde_json::to_vec(&canonical).map_err(|error| error.to_string())?;
    if sha256_hex(&bytes) != expected {
        return Err(format!("opening checksum mismatch in record {number}"));
    }
    Ok(())
}

fn validate_references(references: &[String], number: usize) -> Result<(), String> {
    if references.is_empty() || references.len() > 64 {
        return Err(format!("opening provenance is invalid in record {number}"));
    }
    for reference in references {
        validate_sha256(reference, number)?;
    }
    if references.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(format!(
            "opening provenance is not sorted in record {number}"
        ));
    }
    Ok(())
}

fn validate_sha256(value: &str, number: usize) -> Result<(), String> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(format!("invalid SHA-256 in opening record {number}"));
    }
    Ok(())
}

fn style_accepts(profile: OpeningProfile, classification: &str) -> bool {
    match profile {
        OpeningProfile::Unrestricted => true,
        OpeningProfile::IbishaPreferred | OpeningProfile::IbishaStrict => {
            matches!(classification, "ibisha" | "ibisha-vs-furibisha")
        }
    }
}

fn state_sfen(position: &Position) -> String {
    to_sfen(position)
        .rsplit_once(' ')
        .map_or_else(|| to_sfen(position), |(state, _)| state.to_owned())
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn read_record(
    reader: &mut impl BufRead,
    line: &mut Vec<u8>,
    decoded: &mut u64,
) -> Result<usize, String> {
    let mut total = 0_usize;
    loop {
        let buffer = reader.fill_buf().map_err(|error| error.to_string())?;
        if buffer.is_empty() {
            return Ok(total);
        }
        let newline = buffer.iter().position(|byte| *byte == b'\n');
        let length = newline.map_or(buffer.len(), |index| index + 1);
        if line.len().saturating_add(length) > MAX_LINE_BYTES {
            return Err("opening record exceeds its byte limit".to_owned());
        }
        *decoded = decoded
            .checked_add(u64::try_from(length).unwrap_or(u64::MAX))
            .ok_or_else(|| "opening size overflow".to_owned())?;
        if *decoded > MAX_DECOMPRESSED_BYTES {
            return Err("opening book exceeds its decompressed byte limit".to_owned());
        }
        line.extend_from_slice(&buffer[..length]);
        reader.consume(length);
        total = total.saturating_add(length);
        if newline.is_some() {
            return Ok(total);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::{Compression, write::GzEncoder};
    use std::io::Write;

    fn fixture() -> Vec<u8> {
        let provenance = ["1".repeat(64), "2".repeat(64)];
        let mut record = serde_json::json!({
            "schema": OPENING_BOOK_SCHEMA,
            "stateKey": "eb5bc2ef917ec96fe2172f96d7060ec4f39322caf929fc1177d2c9fc8b937ebc",
            "stateSfen": "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b -",
            "ruleProfile": RULE_PROFILE,
            "buildVersion": "fixture",
            "provenanceReferences": provenance,
            "candidates": [{
                "moveUsi": "2g2f", "sampleCount": 2,
                "sourceDistribution": {"aobazero-no-noise": 2},
                "blackResults": {"wins": 1, "losses": 1, "draws": 0, "unknown": 0},
                "whiteResults": {"wins": 0, "losses": 0, "draws": 0, "unknown": 0},
                "teacherScoreCp": 20, "scoreUncertaintyCp": null,
                "teacherDepth": 8, "teacherNodes": 25000,
                "openingClassification": "ibisha-vs-furibisha",
                "provenanceReferences": provenance
            }]
        });
        let checksum = sha256_hex(&serde_json::to_vec(&record).unwrap());
        record
            .as_object_mut()
            .unwrap()
            .insert("recordChecksum".to_owned(), Value::String(checksum));
        let mut encoder = GzEncoder::new(Vec::new(), Compression::fast());
        writeln!(encoder, "{}", serde_json::to_string(&record).unwrap()).unwrap();
        encoder.finish().unwrap()
    }

    #[test]
    fn strict_profile_selects_a_teacher_safe_legal_ibisha_move() {
        let book = OpeningBookV2::from_compressed_bytes(&fixture()).unwrap();
        let choice = book
            .select(&Position::startpos(), OpeningPolicy::default())
            .unwrap();
        assert_eq!(to_usi_move(choice.movement), "2g2f");
        assert_eq!(book.positions(), 1);
        assert_eq!(book.candidates(), 1);
    }

    #[test]
    fn corruption_fails_closed() {
        let mut bytes = fixture();
        bytes.push(1);
        assert!(OpeningBookV2::from_compressed_bytes(&bytes).is_err());
    }
}
