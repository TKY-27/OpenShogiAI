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

use crate::checksum::sha256_text;

const SCHEMA: &str = "phase3_opening_export/v1";
const MAX_COMPRESSED_BYTES: u64 = 64 * 1024 * 1024;
const MAX_DECOMPRESSED_BYTES: u64 = 256 * 1024 * 1024;
const MAX_LINE_BYTES: usize = 1024 * 1024;
const MAX_RECORDS: usize = 1_000_000;
const MAX_TEXT_BYTES: usize = 16 * 1024;

#[derive(Clone, Debug)]
pub struct OpeningChoice {
    pub movement: Move,
    pub count: u64,
    pub score_rate: Option<f64>,
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
            let record: OpeningRecord = serde_json::from_slice(&line)
                .map_err(|error| format!("invalid opening JSON record {records}: {error}"))?;
            validate_record(&record, records)?;
            let position = parse_sfen(&format!("{} 1", record.state_sfen))
                .map_err(|error| format!("invalid stateSfen in record {records}: {error}"))?;
            if state_sfen(&position) != record.state_sfen {
                return Err(format!("non-canonical stateSfen in record {records}"));
            }
            let movement = parse_usi_move(&record.move_usi)
                .map_err(|error| format!("invalid moveUsi in record {records}: {error}"))?;
            if !position.legal_moves().contains(&movement) {
                return Err(format!("illegal moveUsi in opening record {records}"));
            }
            entries
                .entry(record.state_sfen)
                .or_default()
                .push(OpeningChoice {
                    movement,
                    count: record.count,
                    score_rate: record.score_rate,
                });
        }
        if records == 0 {
            return Err("opening database contains no records".to_owned());
        }
        for choices in entries.values_mut() {
            choices.sort_by(|left, right| {
                right
                    .count
                    .cmp(&left.count)
                    .then_with(|| {
                        right
                            .score_rate
                            .unwrap_or(f64::NEG_INFINITY)
                            .total_cmp(&left.score_rate.unwrap_or(f64::NEG_INFINITY))
                    })
                    .then_with(|| to_usi_move(left.movement).cmp(&to_usi_move(right.movement)))
            });
            for pair in choices.windows(2) {
                if pair[0].movement == pair[1].movement {
                    return Err("opening database contains a duplicate state/move pair".to_owned());
                }
            }
        }
        Ok(Self { entries, records })
    }

    pub fn select(&self, position: &Position) -> Option<OpeningChoice> {
        self.entries
            .get(&state_sfen(position))
            .and_then(|choices| choices.first())
            .cloned()
    }

    pub const fn records(&self) -> usize {
        self.records
    }
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
    if record.schema != SCHEMA {
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

    use super::{MAX_LINE_BYTES, OpeningBook};

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
