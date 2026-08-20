use std::{
    collections::BTreeMap,
    fmt,
    io::{BufRead, BufReader, Cursor, Read},
};

use flate2::bufread::MultiGzDecoder;
use open_shogi_core::{Move, Position, parse_sfen, parse_usi_move, to_sfen, to_usi_move};
use serde::{
    Deserialize, Deserializer,
    de::{self, MapAccess, Visitor},
};
use serde_json::Value;

use crate::checksum::{sha256_bytes, sha256_text};

const LEGACY_SCHEMA: &str = "phase3_opening_export/v1";
pub const OPENING_BOOK_SCHEMA: &str = "open_shogi_opening_book/v2";
const STANDARD_RULE_PROFILE: &str = "standard-shogi/v1";
const MAX_COMPRESSED_BYTES: u64 = 64 * 1024 * 1024;
const MAX_DECOMPRESSED_BYTES: u64 = 256 * 1024 * 1024;
const MAX_LINE_BYTES: usize = 1024 * 1024;
const MAX_RECORDS: usize = 1_000_000;
const MAX_TEXT_BYTES: usize = 16 * 1024;

pub fn run(arguments: &[String]) -> Result<(), String> {
    let [command, flag, path] = arguments else {
        return Err("usage: opening-book verify --book FILE".to_owned());
    };
    if command != "verify" || flag != "--book" || path.is_empty() {
        return Err("usage: opening-book verify --book FILE".to_owned());
    }
    let artifact = crate::checksum::read_file_artifact(
        std::path::Path::new(path),
        MAX_COMPRESSED_BYTES,
    )?;
    let book = open_shogi_core::OpeningBookV2::from_compressed_bytes(&artifact.bytes)?;
    println!(
        "{}",
        serde_json::json!({
            "schema": OPENING_BOOK_SCHEMA,
            "path": path,
            "sha256": artifact.sha256,
            "size": artifact.size,
            "positions": book.positions(),
            "candidates": book.candidates(),
            "status": "valid",
        })
    );
    Ok(())
}

#[derive(Clone, Debug)]
pub struct OpeningChoice {
    pub movement: Move,
    pub count: u64,
    pub score_rate: Option<f64>,
    pub teacher_score_cp: Option<i32>,
    pub teacher_depth: Option<u8>,
    pub teacher_nodes: Option<u64>,
    pub opening_classification: String,
    pub provenance_references: Vec<String>,
}

/// Opening style is enforced only by validated book/policy selection, never move generation.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum OpeningProfile {
    Unrestricted,
    IbishaPreferred,
    #[default]
    IbishaStrict,
}

impl OpeningProfile {
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

#[derive(Debug)]
pub struct OpeningBook {
    entries: BTreeMap<String, Vec<OpeningChoice>>,
    records: usize,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct OpeningRecord {
    schema: String,
    state_key: String,
    state_sfen: String,
    move_usi: String,
    count: u64,
    wins: u64,
    losses: u64,
    draws: u64,
    unknown: u64,
    score_rate: Option<f64>,
    decisive_n: u64,
    decisive_win_rate: Option<f64>,
    decisive_win_rate_wilson95_low: Option<f64>,
    decisive_win_rate_wilson95_high: Option<f64>,
    black_wins: u64,
    white_wins: u64,
    side_specific_decisive_n: u64,
    black_decisive_win_rate: Option<f64>,
    white_decisive_win_rate: Option<f64>,
    average_full_plies: f64,
    average_remaining_plies: f64,
    #[serde(deserialize_with = "deserialize_unique_source_counts")]
    source_counts: BTreeMap<String, u64>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct OpeningRecordV2 {
    schema: String,
    state_key: String,
    state_sfen: String,
    rule_profile: String,
    build_version: String,
    provenance_references: Vec<String>,
    candidates: Vec<OpeningCandidateV2>,
    record_checksum: String,
}

#[derive(Deserialize)]
struct SchemaProbe {
    schema: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct OpeningCandidateV2 {
    move_usi: String,
    sample_count: u64,
    source_distribution: BTreeMap<String, u64>,
    black_results: OpeningResultsV2,
    white_results: OpeningResultsV2,
    teacher_score_cp: Option<i32>,
    score_uncertainty_cp: Option<u32>,
    teacher_depth: Option<u8>,
    teacher_nodes: Option<u64>,
    opening_classification: String,
    provenance_references: Vec<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct OpeningResultsV2 {
    wins: u64,
    losses: u64,
    draws: u64,
    unknown: u64,
}

fn deserialize_unique_source_counts<'de, D>(
    deserializer: D,
) -> Result<BTreeMap<String, u64>, D::Error>
where
    D: Deserializer<'de>,
{
    struct UniqueSourceCountsVisitor;

    impl<'de> Visitor<'de> for UniqueSourceCountsVisitor {
        type Value = BTreeMap<String, u64>;

        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("an object with unique source names")
        }

        fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
        where
            A: MapAccess<'de>,
        {
            let mut values = BTreeMap::new();
            while let Some((source, count)) = map.next_entry::<String, u64>()? {
                if values.insert(source.clone(), count).is_some() {
                    return Err(de::Error::custom(format!(
                        "duplicate sourceCounts key: {source}"
                    )));
                }
            }
            Ok(values)
        }
    }

    deserializer.deserialize_map(UniqueSourceCountsVisitor)
}

impl OpeningBook {
    #[cfg(test)]
    pub fn load(path: &std::path::Path) -> Result<Self, String> {
        let artifact = crate::checksum::read_file_artifact(path, MAX_COMPRESSED_BYTES)?;
        Self::from_compressed_bytes(&artifact.bytes)
    }

    #[expect(
        clippy::too_many_lines,
        reason = "legacy and v2 streaming validation share one bounded decompression pass"
    )]
    pub fn from_compressed_bytes(bytes: &[u8]) -> Result<Self, String> {
        let compressed_bytes = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
        if compressed_bytes == 0 || compressed_bytes > MAX_COMPRESSED_BYTES {
            return Err(format!(
                "opening database compressed size must be 1..={MAX_COMPRESSED_BYTES} bytes"
            ));
        }

        // A single-member decoder reports EOF at the end of the first member and can silently
        // leave appended data unread. Multi-member decoding forces validation through the end of
        // the supplied artifact; an appended valid member becomes visible input, while trailing
        // garbage is rejected by the gzip decoder.
        let decoder = MultiGzDecoder::new(BufReader::new(Cursor::new(bytes)));
        let bounded = decoder.take(MAX_DECOMPRESSED_BYTES + 1);
        let mut reader = BufReader::new(bounded);
        let mut entries = BTreeMap::<String, Vec<OpeningChoice>>::new();
        let mut line = Vec::new();
        let mut records = 0_usize;
        let mut decoded_bytes = 0_u64;
        loop {
            line.clear();
            let bytes = read_bounded_record(&mut reader, &mut line, &mut decoded_bytes)?;
            if bytes == 0 {
                break;
            }
            while matches!(line.last(), Some(b'\n' | b'\r')) {
                line.pop();
            }
            if line.is_empty() {
                return Err("opening database contains an empty record".to_owned());
            }
            records += 1;
            if records > MAX_RECORDS {
                return Err(format!("opening database exceeds {MAX_RECORDS} records"));
            }
            let probe: SchemaProbe = serde_json::from_slice(&line)
                .map_err(|error| format!("invalid opening JSON record {records}: {error}"))?;
            match probe.schema.as_str() {
                LEGACY_SCHEMA => {
                    let record: OpeningRecord = serde_json::from_slice(&line).map_err(|error| {
                        format!("invalid legacy opening record {records}: {error}")
                    })?;
                    validate_record(&record, records)?;
                    let position = validated_position(&record.state_sfen, records)?;
                    let movement = validated_move(&position, &record.move_usi, records)?;
                    entries
                        .entry(record.state_sfen)
                        .or_default()
                        .push(OpeningChoice {
                            movement,
                            count: record.count,
                            score_rate: record.score_rate,
                            teacher_score_cp: None,
                            teacher_depth: None,
                            teacher_nodes: None,
                            opening_classification: "unclassified".to_owned(),
                            provenance_references: Vec::new(),
                        });
                }
                OPENING_BOOK_SCHEMA => {
                    let value: Value = serde_json::from_slice(&line).map_err(|error| {
                        format!("invalid opening v2 record {records}: {error}")
                    })?;
                    validate_record_checksum(&value, records)?;
                    let record: OpeningRecordV2 = serde_json::from_value(value).map_err(|error| {
                        format!("invalid opening v2 record {records}: {error}")
                    })?;
                    let position = validate_record_v2(&record, records)?;
                    let choices = entries.entry(record.state_sfen.clone()).or_default();
                    for candidate in record.candidates {
                        let movement =
                            validated_move(&position, &candidate.move_usi, records)?;
                        let scored = candidate
                            .black_results
                            .wins
                            .checked_add(candidate.black_results.losses)
                            .and_then(|value| value.checked_add(candidate.black_results.draws))
                            .and_then(|value| value.checked_add(candidate.white_results.wins))
                            .and_then(|value| value.checked_add(candidate.white_results.losses))
                            .and_then(|value| value.checked_add(candidate.white_results.draws))
                            .ok_or_else(|| format!("opening result overflow in record {records}"))?;
                        let wins = candidate
                            .black_results
                            .wins
                            .checked_add(candidate.white_results.wins)
                            .ok_or_else(|| format!("opening result overflow in record {records}"))?;
                        let draws = candidate
                            .black_results
                            .draws
                            .checked_add(candidate.white_results.draws)
                            .ok_or_else(|| format!("opening result overflow in record {records}"))?;
                        let score_rate = if scored > 0 {
                            Some(
                                (exact_f64(wins, records)?
                                    + 0.5 * exact_f64(draws, records)?)
                                    / exact_f64(scored, records)?,
                            )
                        } else {
                            None
                        };
                        choices.push(OpeningChoice {
                            movement,
                            count: candidate.sample_count,
                            score_rate,
                            teacher_score_cp: candidate.teacher_score_cp,
                            teacher_depth: candidate.teacher_depth,
                            teacher_nodes: candidate.teacher_nodes,
                            opening_classification: candidate.opening_classification,
                            provenance_references: candidate.provenance_references,
                        });
                    }
                }
                _ => return Err(format!("opening schema mismatch in record {records}")),
            }
        }
        if records == 0 {
            return Err("opening database contains no records".to_owned());
        }
        for choices in entries.values_mut() {
            sort_choices(choices);
            for pair in choices.windows(2) {
                if pair[0].movement == pair[1].movement {
                    return Err("opening database contains a duplicate state/move pair".to_owned());
                }
            }
        }
        Ok(Self { entries, records })
    }

    #[cfg(test)]
    pub fn select(&self, position: &Position) -> Option<OpeningChoice> {
        self.entries
            .get(&state_sfen(position))
            .and_then(|choices| choices.first())
            .cloned()
    }

    /// Returns the strongest safe validated move for one opening profile.
    pub fn select_with_policy(
        &self,
        position: &Position,
        policy: OpeningPolicy,
    ) -> Option<OpeningChoice> {
        if policy.minimum_sample_count == 0 || policy.maximum_teacher_loss_cp < 0 {
            return None;
        }
        let choices = self.entries.get(&state_sfen(position))?;
        let best_teacher_score = choices
            .iter()
            .filter(|choice| choice.count >= policy.minimum_sample_count)
            .filter_map(|choice| choice.teacher_score_cp)
            .max();
        let safe = || {
            choices
                .iter()
                .filter(|choice| choice.count >= policy.minimum_sample_count)
                .filter(|choice| {
                    choice
                        .teacher_score_cp
                        .zip(best_teacher_score)
                        .is_some_and(|(score, best)| {
                            best.saturating_sub(score) <= policy.maximum_teacher_loss_cp
                        })
                })
        };
        match policy.profile {
            OpeningProfile::Unrestricted => safe().next().cloned(),
            OpeningProfile::IbishaPreferred => safe()
                .find(|choice| is_ibisha(&choice.opening_classification))
                .or_else(|| safe().next())
                .cloned(),
            OpeningProfile::IbishaStrict => safe()
                .find(|choice| is_ibisha(&choice.opening_classification))
                .cloned(),
        }
    }

    pub const fn records(&self) -> usize {
        self.records
    }
}

fn sort_choices(choices: &mut [OpeningChoice]) {
    choices.sort_by(|left, right| {
        right
            .teacher_score_cp
            .cmp(&left.teacher_score_cp)
            .then_with(|| right.count.cmp(&left.count))
            .then_with(|| {
                right
                    .score_rate
                    .unwrap_or(f64::NEG_INFINITY)
                    .total_cmp(&left.score_rate.unwrap_or(f64::NEG_INFINITY))
            })
            .then_with(|| to_usi_move(left.movement).cmp(&to_usi_move(right.movement)))
    });
}

fn is_ibisha(classification: &str) -> bool {
    matches!(classification, "ibisha" | "ibisha-vs-furibisha")
}

fn validated_position(state: &str, number: usize) -> Result<Position, String> {
    let position = parse_sfen(&format!("{state} 1"))
        .map_err(|error| format!("invalid stateSfen in record {number}: {error}"))?;
    if state_sfen(&position) != state {
        return Err(format!("non-canonical stateSfen in record {number}"));
    }
    Ok(position)
}

fn validated_move(position: &Position, notation: &str, number: usize) -> Result<Move, String> {
    let movement = parse_usi_move(notation)
        .map_err(|error| format!("invalid moveUsi in record {number}: {error}"))?;
    if !position.legal_moves().contains(&movement) {
        return Err(format!("illegal moveUsi in opening record {number}"));
    }
    Ok(movement)
}

fn validate_record_checksum(value: &Value, number: usize) -> Result<(), String> {
    let expected = value
        .get("recordChecksum")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("opening record {number} lacks recordChecksum"))?;
    validate_sha256(expected, "recordChecksum", number)?;
    let mut canonical = value.clone();
    canonical
        .as_object_mut()
        .ok_or_else(|| format!("opening record {number} is not an object"))?
        .remove("recordChecksum");
    let bytes = serde_json::to_vec(&canonical)
        .map_err(|error| format!("cannot canonicalize opening record {number}: {error}"))?;
    if sha256_bytes(&bytes) != expected {
        return Err(format!("opening record checksum mismatch in record {number}"));
    }
    Ok(())
}

fn validate_record_v2(record: &OpeningRecordV2, number: usize) -> Result<Position, String> {
    if record.schema != OPENING_BOOK_SCHEMA || record.rule_profile != STANDARD_RULE_PROFILE {
        return Err(format!("opening v2 compatibility mismatch in record {number}"));
    }
    if record.build_version.is_empty()
        || record.build_version.len() > 128
        || !record.build_version.is_ascii()
        || record.candidates.is_empty()
        || record.candidates.len() > 256
    {
        return Err(format!("opening v2 bounds are invalid in record {number}"));
    }
    validate_sha256(&record.state_key, "stateKey", number)?;
    validate_sha256(&record.record_checksum, "recordChecksum", number)?;
    if record.state_key != sha256_text(&record.state_sfen) {
        return Err(format!("opening stateKey is invalid in record {number}"));
    }
    validate_provenance_references(&record.provenance_references, number)?;
    let position = validated_position(&record.state_sfen, number)?;
    for candidate in &record.candidates {
        if candidate.move_usi.is_empty()
            || candidate.move_usi.len() > 16
            || candidate.sample_count == 0
            || candidate.source_distribution.is_empty()
            || candidate.source_distribution.values().any(|count| *count == 0)
            || candidate
                .source_distribution
                .values()
                .try_fold(0_u64, |sum, count| sum.checked_add(*count))
                != Some(candidate.sample_count)
            || !matches!(
                candidate.opening_classification.as_str(),
                "ibisha" | "ibisha-vs-furibisha" | "furibisha" | "unclassified"
            )
        {
            return Err(format!("opening candidate is invalid in record {number}"));
        }
        let black_total = result_total(&candidate.black_results)
            .ok_or_else(|| format!("opening result overflow in record {number}"))?;
        let white_total = result_total(&candidate.white_results)
            .ok_or_else(|| format!("opening result overflow in record {number}"))?;
        if black_total.checked_add(white_total) != Some(candidate.sample_count)
            || candidate.sample_count > u64::from(u32::MAX)
        {
            return Err(format!("opening result count mismatch in record {number}"));
        }
        match (
            candidate.teacher_score_cp,
            candidate.teacher_depth,
            candidate.teacher_nodes,
        ) {
            (Some(score), Some(depth), Some(nodes))
                if (-32_000..=32_000).contains(&score)
                    && (1..=64).contains(&depth)
                    && nodes > 0 => {}
            (None, None, None) => {}
            _ => return Err(format!("opening teacher evidence is invalid in record {number}")),
        }
        if candidate
            .score_uncertainty_cp
            .is_some_and(|uncertainty| uncertainty > 10_000)
        {
            return Err(format!("opening uncertainty is invalid in record {number}"));
        }
        validate_provenance_references(&candidate.provenance_references, number)?;
        if candidate
            .provenance_references
            .iter()
            .any(|reference| !record.provenance_references.contains(reference))
        {
            return Err(format!(
                "candidate provenance is absent from its record in record {number}"
            ));
        }
    }
    Ok(position)
}

fn result_total(results: &OpeningResultsV2) -> Option<u64> {
    results
        .wins
        .checked_add(results.losses)
        .and_then(|value| value.checked_add(results.draws))
        .and_then(|value| value.checked_add(results.unknown))
}

fn validate_provenance_references(references: &[String], number: usize) -> Result<(), String> {
    if references.is_empty() || references.len() > 64 {
        return Err(format!("opening provenance is invalid in record {number}"));
    }
    let mut previous = None;
    for reference in references {
        validate_sha256(reference, "provenance reference", number)?;
        if previous.is_some_and(|previous: &String| previous >= reference) {
            return Err(format!(
                "opening provenance must be sorted and unique in record {number}"
            ));
        }
        previous = Some(reference);
    }
    Ok(())
}

fn validate_sha256(value: &str, name: &str, number: usize) -> Result<(), String> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(format!("opening {name} is invalid in record {number}"));
    }
    Ok(())
}

fn read_bounded_record(
    reader: &mut impl BufRead,
    line: &mut Vec<u8>,
    decoded_bytes: &mut u64,
) -> Result<usize, String> {
    let mut consumed_total = 0_usize;
    loop {
        let buffer = reader
            .fill_buf()
            .map_err(|error| format!("cannot decompress opening database: {error}"))?;
        if buffer.is_empty() {
            return Ok(consumed_total);
        }
        let newline = buffer.iter().position(|byte| *byte == b'\n');
        let content_length = newline.map_or(buffer.len(), |index| index + 1);
        let next_length = line
            .len()
            .checked_add(content_length)
            .ok_or_else(|| "opening record size overflow".to_owned())?;
        if next_length > MAX_LINE_BYTES {
            return Err(format!(
                "opening database record exceeds {MAX_LINE_BYTES} bytes"
            ));
        }
        let next_decoded = decoded_bytes
            .checked_add(
                u64::try_from(content_length)
                    .map_err(|_| "opening record size overflow".to_owned())?,
            )
            .ok_or_else(|| "opening database decompressed size overflow".to_owned())?;
        if next_decoded > MAX_DECOMPRESSED_BYTES {
            return Err(format!(
                "opening database exceeds {MAX_DECOMPRESSED_BYTES} decompressed bytes"
            ));
        }
        line.extend_from_slice(&buffer[..content_length]);
        reader.consume(content_length);
        *decoded_bytes = next_decoded;
        consumed_total = consumed_total
            .checked_add(content_length)
            .ok_or_else(|| "opening record size overflow".to_owned())?;
        if newline.is_some() {
            return Ok(consumed_total);
        }
    }
}

fn validate_record(record: &OpeningRecord, number: usize) -> Result<(), String> {
    if record.schema != LEGACY_SCHEMA {
        return Err(format!("opening schema mismatch in record {number}"));
    }
    if record.state_key.len() != 64
        || !record
            .state_key
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        || record.state_key != sha256_text(&record.state_sfen)
    {
        return Err(format!("opening stateKey is invalid in record {number}"));
    }
    if record.state_sfen.is_empty()
        || record.state_sfen.len() > MAX_TEXT_BYTES
        || record.move_usi.is_empty()
        || record.move_usi.len() > 16
    {
        return Err(format!(
            "opening text field is out of bounds in record {number}"
        ));
    }
    if record.count == 0
        || checked_sum(&[record.wins, record.losses, record.draws, record.unknown])
            != Some(record.count)
        || checked_sum(&[record.wins, record.losses]) != Some(record.decisive_n)
        || checked_sum(&[record.black_wins, record.white_wins])
            != Some(record.side_specific_decisive_n)
        || record.side_specific_decisive_n != record.decisive_n
        || record.source_counts.is_empty()
        || record.source_counts.values().any(|count| *count == 0)
        || record
            .source_counts
            .keys()
            .any(|source| source.is_empty() || source.len() > MAX_TEXT_BYTES)
        || record
            .source_counts
            .values()
            .try_fold(0_u64, |sum, value| sum.checked_add(*value))
            != Some(record.count)
    {
        return Err(format!("opening count must be positive in record {number}"));
    }
    let scored = checked_sum(&[record.wins, record.losses, record.draws])
        .ok_or_else(|| format!("opening count overflow in record {number}"))?;
    let wins = exact_f64(record.wins, number)?;
    let draws = exact_f64(record.draws, number)?;
    let scored_f64 = exact_f64(scored, number)?;
    let decisive_n = exact_f64(record.decisive_n, number)?;
    let side_decisive_n = exact_f64(record.side_specific_decisive_n, number)?;
    let black_wins = exact_f64(record.black_wins, number)?;
    let white_wins = exact_f64(record.white_wins, number)?;
    let expected_score_rate = (scored != 0).then(|| (wins + 0.5 * draws) / scored_f64);
    let expected_decisive_rate = (record.decisive_n != 0).then(|| wins / decisive_n);
    let expected_black_rate =
        (record.side_specific_decisive_n != 0).then(|| black_wins / side_decisive_n);
    let expected_white_rate =
        (record.side_specific_decisive_n != 0).then(|| white_wins / side_decisive_n);
    let (expected_wilson_low, expected_wilson_high) =
        wilson_interval_95(record.wins, record.decisive_n, number)?;
    if !optional_rate_matches(record.score_rate, expected_score_rate)
        || !optional_rate_matches(record.decisive_win_rate, expected_decisive_rate)
        || !optional_rate_matches(record.decisive_win_rate_wilson95_low, expected_wilson_low)
        || !optional_rate_matches(record.decisive_win_rate_wilson95_high, expected_wilson_high)
        || !optional_rate_matches(record.black_decisive_win_rate, expected_black_rate)
        || !optional_rate_matches(record.white_decisive_win_rate, expected_white_rate)
    {
        return Err(format!("opening scoreRate is invalid in record {number}"));
    }
    if !record.average_full_plies.is_finite()
        || !record.average_remaining_plies.is_finite()
        || record.average_full_plies <= 0.0
        || record.average_remaining_plies < 0.0
        || record.average_remaining_plies > record.average_full_plies
    {
        return Err(format!(
            "opening ply averages are invalid in record {number}"
        ));
    }
    Ok(())
}

fn checked_sum(values: &[u64]) -> Option<u64> {
    values
        .iter()
        .try_fold(0_u64, |sum, value| sum.checked_add(*value))
}

fn optional_rate_matches(observed: Option<f64>, expected: Option<f64>) -> bool {
    match (observed, expected) {
        (Some(observed), Some(expected)) => {
            observed.is_finite()
                && (0.0..=1.0).contains(&observed)
                && (observed - expected).abs() <= 1.0e-12
        }
        (None, None) => true,
        _ => false,
    }
}

fn exact_f64(value: u64, number: usize) -> Result<f64, String> {
    let exact = u32::try_from(value)
        .map_err(|_| format!("opening count exceeds the exact numeric limit in record {number}"))?;
    Ok(f64::from(exact))
}

fn wilson_interval_95(
    wins: u64,
    trials: u64,
    number: usize,
) -> Result<(Option<f64>, Option<f64>), String> {
    if trials == 0 {
        return Ok((None, None));
    }
    let z = 1.959_963_984_540_054_f64;
    let trials = exact_f64(trials, number)?;
    let proportion = exact_f64(wins, number)? / trials;
    let z_squared = z * z;
    let denominator = 1.0 + z_squared / trials;
    let centre = proportion + z_squared / (2.0 * trials);
    let margin =
        z * ((proportion * (1.0 - proportion) + z_squared / (4.0 * trials)) / trials).sqrt();
    Ok((
        Some(((centre - margin) / denominator).max(0.0)),
        Some(((centre + margin) / denominator).min(1.0)),
    ))
}

fn state_sfen(position: &Position) -> String {
    to_sfen(position)
        .rsplit_once(' ')
        .map_or_else(|| to_sfen(position), |(state, _)| state.to_owned())
}

#[cfg(test)]
mod tests {
    use std::{fs::File, io::Write, path::PathBuf};

    use flate2::{Compression, write::GzEncoder};
    use open_shogi_core::{Position, to_usi_move};

    use super::{
        MAX_LINE_BYTES, OPENING_BOOK_SCHEMA, OpeningBook, OpeningPolicy, OpeningProfile,
    };

    #[test]
    fn loads_and_selects_a_legal_deterministic_move() {
        let path = temporary_path("valid");
        let file = File::create(&path).unwrap();
        let mut gzip = GzEncoder::new(file, Compression::fast());
        writeln!(gzip, "{}", valid_record("7g7f", 2)).unwrap();
        gzip.finish().unwrap();
        let book = OpeningBook::load(&path).unwrap();
        let choice = book.select(&Position::startpos()).unwrap();
        assert_eq!(to_usi_move(choice.movement), "7g7f");
        assert_eq!(choice.count, 2);
        assert_eq!(choice.score_rate, Some(0.5));
        assert_eq!(book.records(), 1);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_an_illegal_opening_move() {
        let path = temporary_path("illegal");
        let file = File::create(&path).unwrap();
        let mut gzip = GzEncoder::new(file, Compression::fast());
        writeln!(gzip, "{}", valid_record("5a5b", 1)).unwrap();
        gzip.finish().unwrap();
        assert!(OpeningBook::load(&path).is_err());
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_unknown_fields_and_trailing_non_gzip_data() {
        let mut unknown_record = valid_record("7g7f", 1);
        unknown_record.pop();
        unknown_record.push_str(",\"extra\":true}");
        let unknown = compressed_record(&unknown_record);
        assert!(OpeningBook::from_compressed_bytes(&unknown).is_err());

        let mut trailing = compressed_record(&valid_record("7g7f", 1));
        trailing.extend_from_slice(b"untrusted trailing bytes");
        assert!(OpeningBook::from_compressed_bytes(&trailing).is_err());
    }

    #[test]
    fn rejects_duplicate_source_count_keys_without_last_value_wins() {
        let canonical = valid_record("7g7f", 1);
        assert!(OpeningBook::from_compressed_bytes(&compressed_record(&canonical)).is_ok());
        let duplicate = canonical.replace(
            "\"sourceCounts\":{\"fixture\":1}",
            "\"sourceCounts\":{\"fixture\":1,\"fixture\":1}",
        );

        let error = OpeningBook::from_compressed_bytes(&compressed_record(&duplicate)).unwrap_err();
        assert!(error.contains("duplicate sourceCounts key"));
    }

    #[test]
    fn oversized_record_is_rejected_without_growing_to_the_decompressed_cap() {
        let oversized = vec![b'x'; MAX_LINE_BYTES + 1];
        let mut gzip = GzEncoder::new(Vec::new(), Compression::fast());
        gzip.write_all(&oversized).unwrap();
        let compressed = gzip.finish().unwrap();

        let error = OpeningBook::from_compressed_bytes(&compressed).unwrap_err();
        assert!(error.contains("record exceeds"));
    }

    #[test]
    fn accepts_the_canonical_phase3_export_and_unknown_only_rows() {
        let canonical = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../data/processed/phase3/opening/aobazero-no-noise-pd-sample100.jsonl.gz");
        if canonical.exists() {
            let book = OpeningBook::load(&canonical).unwrap();
            assert_eq!(book.records(), 8_855);
        }

        let unknown = valid_record("7g7f", 1)
            .replace("\"wins\":1", "\"wins\":0")
            .replace("\"unknown\":0", "\"unknown\":1")
            .replace("\"scoreRate\":1", "\"scoreRate\":null")
            .replace("\"decisiveN\":1", "\"decisiveN\":0")
            .replace("\"decisiveWinRate\":1", "\"decisiveWinRate\":null")
            .replace(
                "\"decisiveWinRateWilson95Low\":0.20654931437723745",
                "\"decisiveWinRateWilson95Low\":null",
            )
            .replace(
                "\"decisiveWinRateWilson95High\":1",
                "\"decisiveWinRateWilson95High\":null",
            )
            .replace("\"blackWins\":1", "\"blackWins\":0")
            .replace("\"sideSpecificDecisiveN\":1", "\"sideSpecificDecisiveN\":0")
            .replace(
                "\"blackDecisiveWinRate\":1",
                "\"blackDecisiveWinRate\":null",
            )
            .replace(
                "\"whiteDecisiveWinRate\":0",
                "\"whiteDecisiveWinRate\":null",
            );
        let book = OpeningBook::from_compressed_bytes(&compressed_record(&unknown)).unwrap();
        assert_eq!(book.select(&Position::startpos()).unwrap().score_rate, None);
    }

    #[test]
    fn v2_book_enforces_teacher_safety_and_ibisha_profiles() {
        let record = valid_v2_record();
        let book = OpeningBook::from_compressed_bytes(&compressed_record(&record)).unwrap();
        let position = Position::startpos();
        let unrestricted = book
            .select_with_policy(
                &position,
                OpeningPolicy {
                    profile: OpeningProfile::Unrestricted,
                    minimum_sample_count: 2,
                    maximum_teacher_loss_cp: 80,
                },
            )
            .unwrap();
        assert_eq!(to_usi_move(unrestricted.movement), "7g7f");

        let strict = book
            .select_with_policy(&position, OpeningPolicy::default())
            .unwrap();
        assert_eq!(to_usi_move(strict.movement), "2g2f");
        assert_eq!(strict.opening_classification, "ibisha-vs-furibisha");
        assert_eq!(strict.teacher_score_cp, Some(50));

        assert!(book
            .select_with_policy(
                &position,
                OpeningPolicy {
                    maximum_teacher_loss_cp: 5,
                    ..OpeningPolicy::default()
                },
            )
            .is_none());

        let preferred_fallback = book
            .select_with_policy(
                &position,
                OpeningPolicy {
                    profile: OpeningProfile::IbishaPreferred,
                    maximum_teacher_loss_cp: 5,
                    ..OpeningPolicy::default()
                },
            )
            .unwrap();
        assert_eq!(to_usi_move(preferred_fallback.movement), "7g7f");
    }

    #[test]
    fn v2_book_rejects_checksum_corruption() {
        let record = valid_v2_record().replace("\"teacherScoreCp\":60", "\"teacherScoreCp\":61");
        let error = OpeningBook::from_compressed_bytes(&compressed_record(&record)).unwrap_err();
        assert!(error.contains("checksum mismatch"));
    }

    fn compressed_record(record: &str) -> Vec<u8> {
        let mut gzip = GzEncoder::new(Vec::new(), Compression::fast());
        gzip.write_all(record.as_bytes()).unwrap();
        gzip.write_all(b"\n").unwrap();
        gzip.finish().unwrap()
    }

    fn valid_record(movement: &str, count: u64) -> String {
        let (wins, losses, rate, low, high, white_rate) = if count == 2 {
            (
                1,
                1,
                0.5,
                0.094_531_205_734_230_74,
                0.905_468_794_265_769_3,
                0.5,
            )
        } else {
            (1, 0, 1.0, 0.206_549_314_377_237_45, 1.0, 0.0)
        };
        format!(
            "{{\"schema\":\"phase3_opening_export/v1\",\"stateKey\":\"eb5bc2ef917ec96fe2172f96d7060ec4f39322caf929fc1177d2c9fc8b937ebc\",\"stateSfen\":\"lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b -\",\"moveUsi\":\"{movement}\",\"count\":{count},\"wins\":{wins},\"losses\":{losses},\"draws\":0,\"unknown\":0,\"scoreRate\":{rate},\"decisiveN\":{count},\"decisiveWinRate\":{rate},\"decisiveWinRateWilson95Low\":{low},\"decisiveWinRateWilson95High\":{high},\"blackWins\":{wins},\"whiteWins\":{losses},\"sideSpecificDecisiveN\":{count},\"blackDecisiveWinRate\":{rate},\"whiteDecisiveWinRate\":{white_rate},\"averageFullPlies\":100.0,\"averageRemainingPlies\":100.0,\"sourceCounts\":{{\"fixture\":{count}}}}}"
        )
    }

    fn valid_v2_record() -> String {
        let provenance = ["1".repeat(64), "2".repeat(64)];
        let results = |wins, losses| {
            serde_json::json!({"wins": wins, "losses": losses, "draws": 0, "unknown": 0})
        };
        let mut value = serde_json::json!({
            "schema": OPENING_BOOK_SCHEMA,
            "stateKey": "eb5bc2ef917ec96fe2172f96d7060ec4f39322caf929fc1177d2c9fc8b937ebc",
            "stateSfen": "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b -",
            "ruleProfile": "standard-shogi/v1",
            "buildVersion": "fixture-v2",
            "provenanceReferences": provenance,
            "candidates": [
                {
                    "moveUsi": "7g7f",
                    "sampleCount": 10,
                    "sourceDistribution": {"aobazero-no-noise": 10},
                    "blackResults": results(6, 4),
                    "whiteResults": results(0, 0),
                    "teacherScoreCp": 60,
                    "scoreUncertaintyCp": 20,
                    "teacherDepth": 8,
                    "teacherNodes": 25000,
                    "openingClassification": "unclassified",
                    "provenanceReferences": provenance,
                },
                {
                    "moveUsi": "2g2f",
                    "sampleCount": 3,
                    "sourceDistribution": {"aobazero-no-noise": 3},
                    "blackResults": results(2, 1),
                    "whiteResults": results(0, 0),
                    "teacherScoreCp": 50,
                    "scoreUncertaintyCp": null,
                    "teacherDepth": 8,
                    "teacherNodes": 25000,
                    "openingClassification": "ibisha-vs-furibisha",
                    "provenanceReferences": provenance,
                }
            ]
        });
        let checksum = crate::checksum::sha256_bytes(&serde_json::to_vec(&value).unwrap());
        value
            .as_object_mut()
            .unwrap()
            .insert("recordChecksum".to_owned(), serde_json::json!(checksum));
        serde_json::to_string(&value).unwrap()
    }

    fn temporary_path(label: &str) -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().canonicalize().unwrap().join(format!(
            "open-shogi-opening-{label}-{}-{nonce}.jsonl.gz",
            std::process::id()
        ))
    }
}
