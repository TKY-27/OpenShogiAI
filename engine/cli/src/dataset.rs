use std::{
    collections::hash_map::RandomState,
    ffi::{OsStr, OsString},
    fs,
    hash::BuildHasher,
    io::{self, Read, Write},
    path::{Component, Path, PathBuf},
    sync::Arc,
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

use open_shogi_core::{
    AnchoredDir, AnchoredFile, CsaGame, CsaResultValidation, CsaSpecialMove, EntryKind,
    RepetitionOutcome, Side, StableFileIdentity, parse_csa_game, repetition_outcome_from_moves,
    to_csa_game, to_sfen, to_usi_move,
};

use crate::args::{parse_next, path_next};

const EXPORT_SCHEMA: &str = "phase3_csa_export/v1";
const MAX_GAMES: usize = 10_000;
const MAX_DIRECTORY_ENTRIES: usize = 100_000;
const MAX_CSA_BYTES: usize = 1_048_576;
// `phase3_csa_export/v1` caps each physical UTF-8 JSONL record, including its trailing LF.
const MAX_JSONL_RECORD_BYTES: usize = 32 * 1_048_576;
const MAX_OUTPUT_BYTES: u64 = 1_073_741_824;
const TEMPORARY_CREATE_ATTEMPTS: u64 = 100;

static TEMPORARY_FILE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy)]
struct ExportLimits {
    directory_entries: usize,
    csa_bytes: usize,
    jsonl_record_bytes: usize,
    output_bytes: u64,
}

impl ExportLimits {
    const PRODUCTION: Self = Self {
        directory_entries: MAX_DIRECTORY_ENTRIES,
        csa_bytes: MAX_CSA_BYTES,
        jsonl_record_bytes: MAX_JSONL_RECORD_BYTES,
        output_bytes: MAX_OUTPUT_BYTES,
    };
}

struct ExportConfig {
    input_dir: PathBuf,
    output: PathBuf,
    max_games: usize,
}

struct InputCandidate {
    directory: Arc<AnchoredDir>,
    file_name: OsString,
    enumerated_kind: EntryKind,
    enumerated_identity: Result<StableFileIdentity, String>,
}

#[derive(Clone, Copy)]
enum ExportOutcome {
    Winner(Side),
    Draw,
    Unknown,
}

pub fn run(arguments: &[String]) -> Result<(), String> {
    let config = parse_arguments(arguments)?;
    export_directory(&config, ExportLimits::PRODUCTION)
}

fn parse_arguments(arguments: &[String]) -> Result<ExportConfig, String> {
    let mut input_dir = None;
    let mut output = None;
    let mut max_games = None;
    let mut index = 0;
    while index < arguments.len() {
        match arguments[index].as_str() {
            "--input-dir" if input_dir.is_none() => {
                input_dir = Some(path_next(arguments, &mut index, "--input-dir")?);
            }
            "--output" if output.is_none() => {
                output = Some(path_next(arguments, &mut index, "--output")?);
            }
            "--max-games" if max_games.is_none() => {
                max_games = Some(parse_next(arguments, &mut index, "--max-games")?);
            }
            "--input-dir" | "--output" | "--max-games" => {
                return Err(format!("{} may be specified only once", arguments[index]));
            }
            argument => return Err(format!("unknown export-csa-jsonl argument: {argument}")),
        }
        index += 1;
    }

    let input_dir =
        input_dir.ok_or_else(|| "export-csa-jsonl requires --input-dir DIR".to_owned())?;
    let output = output.ok_or_else(|| "export-csa-jsonl requires --output FILE".to_owned())?;
    let max_games =
        max_games.ok_or_else(|| "export-csa-jsonl requires --max-games N".to_owned())?;
    if !(1..=MAX_GAMES).contains(&max_games) {
        return Err(format!("--max-games must be between 1 and {MAX_GAMES}"));
    }
    Ok(ExportConfig {
        input_dir,
        output,
        max_games,
    })
}

fn export_directory(config: &ExportConfig, limits: ExportLimits) -> Result<(), String> {
    validate_output_path(&config.output)?;
    let mut candidates = collect_candidates(&config.input_dir, limits.directory_entries)?;
    candidates.sort_by(|left, right| {
        left.file_name
            .as_encoded_bytes()
            .cmp(right.file_name.as_encoded_bytes())
    });
    candidates.truncate(config.max_games);

    let mut output = AtomicOutput::create(&config.output, limits.output_bytes)?;
    for candidate in &candidates {
        let line = export_candidate(candidate, limits);
        output.write_line(&line)?;
    }
    output.commit()
}

fn collect_candidates(
    input_dir: &Path,
    maximum_entries: usize,
) -> Result<Vec<InputCandidate>, String> {
    let directory = Arc::new(
        AnchoredDir::open_existing(input_dir)
            .map_err(|error| format!("cannot anchor input directory without symlinks: {error}"))?,
    );
    let entries = directory
        .entries(maximum_entries)
        .map_err(|error| format!("cannot read bounded input directory: {error}"))?;
    let mut candidates = Vec::new();
    for entry in entries {
        let file_name = entry.name;
        if Path::new(&file_name).extension() != Some(OsStr::new("csa")) {
            continue;
        }
        let enumerated_identity = if entry.kind == EntryKind::File {
            directory
                .open_regular(&file_name)
                .and_then(|file| file.stable_identity())
                .map_err(|error| format!("inspect_failed: {}", io_error_name(error.kind())))
        } else {
            Err(match entry.kind {
                EntryKind::Symlink => "symlink: symbolic links are not accepted".to_owned(),
                EntryKind::Directory | EntryKind::Other => {
                    "non_regular_file: input is not a regular file".to_owned()
                }
                EntryKind::File => unreachable!("file entries are opened above"),
            })
        };
        candidates.push(InputCandidate {
            directory: Arc::clone(&directory),
            file_name,
            enumerated_kind: entry.kind,
            enumerated_identity,
        });
    }
    Ok(candidates)
}

fn export_candidate(candidate: &InputCandidate, limits: ExportLimits) -> String {
    let safe_name = candidate.file_name.to_string_lossy();
    let Some(file_name) = candidate.file_name.to_str() else {
        return rejected_row(
            &safe_name,
            "invalid_filename: filename is not valid UTF-8",
            limits.jsonl_record_bytes,
        );
    };
    if file_name.chars().any(char::is_control) {
        return rejected_row(
            file_name,
            "invalid_filename: filename contains a control character",
            limits.jsonl_record_bytes,
        );
    }
    if !is_single_path_component(&candidate.file_name) {
        return rejected_row(
            file_name,
            "invalid_filename: filename is not one normal path component",
            limits.jsonl_record_bytes,
        );
    }

    match load_and_normalize(candidate, limits.csa_bytes) {
        Ok((game, normalized_csa)) => {
            match accepted_row(file_name, &game, &normalized_csa, limits.jsonl_record_bytes) {
                Ok(line) => line,
                Err(reason) => rejected_row(file_name, &reason, limits.jsonl_record_bytes),
            }
        }
        Err(reason) => rejected_row(file_name, &reason, limits.jsonl_record_bytes),
    }
}

fn load_and_normalize(
    candidate: &InputCandidate,
    maximum_bytes: usize,
) -> Result<(CsaGame, String), String> {
    if candidate.enumerated_kind != EntryKind::File {
        return Err(candidate
            .enumerated_identity
            .as_ref()
            .expect_err("non-file enumeration stores a rejection")
            .clone());
    }
    let enumerated_identity = candidate.enumerated_identity.as_ref().map_err(Clone::clone)?;
    let mut file = candidate
        .directory
        .open_regular(&candidate.file_name)
        .map_err(|error| format!("open_failed: {}", io_error_name(error.kind())))?;
    let opened_identity = file
        .stable_identity()
        .map_err(|error| format!("inspect_failed: {}", io_error_name(error.kind())))?;
    if &opened_identity != enumerated_identity {
        return Err("path_changed: input changed after directory enumeration".to_owned());
    }
    if opened_identity.length()
        > u64::try_from(maximum_bytes).expect("CSA byte limit fits in u64")
    {
        return Err(format!(
            "input_too_large: CSA file exceeds the {maximum_bytes}-byte defensive limit"
        ));
    }

    let mut bytes = Vec::with_capacity(
        usize::try_from(opened_identity.length())
            .unwrap_or(maximum_bytes)
            .min(maximum_bytes),
    );
    {
        let mut limited = file
            .reader()
            .take(u64::try_from(maximum_bytes + 1).expect("CSA byte limit fits in u64"));
        limited
            .read_to_end(&mut bytes)
            .map_err(|error| format!("read_failed: {}", io_error_name(error.kind())))?;
    }
    if bytes.len() > maximum_bytes {
        return Err(format!(
            "input_too_large: CSA file exceeds the {maximum_bytes}-byte defensive limit"
        ));
    }
    file.verify_stable_read(
        &opened_identity,
        u64::try_from(bytes.len()).expect("CSA length fits in u64"),
    )
    .map_err(|_| "path_changed: input changed while it was being read".to_owned())?;
    let contents = decode_csa_bytes(&bytes)?;
    let parsed = parse_csa_game(contents).map_err(|error| format!("invalid_csa: {error}"))?;
    let normalized_csa =
        to_csa_game(&parsed).map_err(|error| format!("normalization_failed: {error}"))?;
    let normalized_game = parse_csa_game(&normalized_csa)
        .map_err(|error| format!("normalization_failed: normalized CSA is invalid: {error}"))?;
    Ok((normalized_game, normalized_csa))
}

fn decode_csa_bytes(bytes: &[u8]) -> Result<&str, String> {
    let raw_first_line = bytes
        .split(|byte| *byte == b'\n')
        .next()
        .unwrap_or_default();
    let first_line = raw_first_line.strip_suffix(b"\r").unwrap_or(raw_first_line);
    let declared_utf8 = first_line == b"'CSA encoding=UTF-8";
    if first_line.starts_with(b"'CSA encoding=") && !declared_utf8 {
        return Err(
            "unsupported_encoding: convert the CSA record to UTF-8 before export".to_owned(),
        );
    }
    let text = std::str::from_utf8(bytes)
        .map_err(|_| "invalid_utf8: convert the CSA record to UTF-8 before export".to_owned())?;
    if !declared_utf8 && !text.is_ascii() {
        return Err(
            "missing_utf8_declaration: non-ASCII CSA requires `'CSA encoding=UTF-8`".to_owned(),
        );
    }
    Ok(text)
}

fn accepted_row(
    file_name: &str,
    game: &CsaGame,
    normalized_csa: &str,
    maximum_bytes: usize,
) -> Result<String, String> {
    let mut row = JsonLine::new(maximum_bytes);
    row.raw("{\"schema\":")?;
    row.string(EXPORT_SCHEMA)?;
    row.raw(",\"status\":\"ok\",\"inputFile\":")?;
    row.string(file_name)?;
    row.raw(",\"normalizedCsa\":")?;
    row.string(normalized_csa)?;
    row.raw(",\"initialSfen\":")?;
    row.string(&to_sfen(&game.initial_position))?;
    row.raw(",\"positionSfens\":[")?;

    let mut position = game.initial_position.clone();
    row.string(&to_sfen(&position))?;
    for &movement in &game.moves {
        position
            .make_move(movement)
            .map_err(|error| format!("replay_failed: {error}"))?;
        row.raw(",")?;
        row.string(&to_sfen(&position))?;
    }

    row.raw("],\"usiMoves\":[")?;
    for (index, &movement) in game.moves.iter().enumerate() {
        if index > 0 {
            row.raw(",")?;
        }
        row.string(&to_usi_move(movement))?;
    }
    row.raw("],\"blackName\":")?;
    row.optional_string(nonempty_name(game.black_name.as_deref()))?;
    row.raw(",\"whiteName\":")?;
    row.optional_string(nonempty_name(game.white_name.as_deref()))?;
    row.raw(",\"terminalReason\":")?;
    row.optional_string(game.special_move.as_ref().map(csa_special_code))?;
    row.raw(",\"outcome\":")?;
    row.string(outcome_name(classify_outcome(game, &position)))?;
    row.raw(",\"resultValidation\":")?;
    row.string(result_validation_name(game.result_validation))?;
    row.raw("}")?;
    Ok(row.finish())
}

fn rejected_row(file_name: &str, reason: &str, maximum_bytes: usize) -> String {
    let mut row = JsonLine::new(maximum_bytes);
    let result = (|| {
        row.raw("{\"schema\":")?;
        row.string(EXPORT_SCHEMA)?;
        row.raw(",\"status\":\"rejected\",\"inputFile\":")?;
        row.string(file_name)?;
        row.raw(",\"reason\":")?;
        row.string(reason)?;
        row.raw("}")
    })();
    if result.is_err() {
        return concat!(
            "{\"schema\":\"phase3_csa_export/v1\",\"status\":\"rejected\",",
            "\"inputFile\":\"<unrepresentable>\",",
            "\"reason\":\"record_too_large: rejection row exceeded its defensive limit\"}"
        )
        .to_owned();
    }
    row.finish()
}

#[expect(
    clippy::match_same_arms,
    reason = "known unknown-result codes stay explicit while the non-exhaustive core enum needs a fallback"
)]
fn classify_outcome(game: &CsaGame, final_position: &open_shogi_core::Position) -> ExportOutcome {
    let Some(special) = &game.special_move else {
        return ExportOutcome::Unknown;
    };
    match special {
        CsaSpecialMove::Resign
        | CsaSpecialMove::IllegalMove
        | CsaSpecialMove::TimeUp
        | CsaSpecialMove::Checkmate => {
            ExportOutcome::Winner(final_position.side_to_move().opposite())
        }
        CsaSpecialMove::BlackIllegalAction => ExportOutcome::Winner(Side::White),
        CsaSpecialMove::WhiteIllegalAction => ExportOutcome::Winner(Side::Black),
        CsaSpecialMove::Win => ExportOutcome::Winner(final_position.side_to_move()),
        CsaSpecialMove::Draw => ExportOutcome::Draw,
        CsaSpecialMove::Repetition if game.result_validation == CsaResultValidation::Verified => {
            ExportOutcome::Draw
        }
        CsaSpecialMove::PerpetualCheck => perpetual_check_outcome(game),
        CsaSpecialMove::Interrupted
        | CsaSpecialMove::EnteringKing
        | CsaSpecialMove::MaxMoves
        | CsaSpecialMove::Takeback
        | CsaSpecialMove::NoMate
        | CsaSpecialMove::Error
        | CsaSpecialMove::Other(_) => ExportOutcome::Unknown,
        _ => ExportOutcome::Unknown,
    }
}

fn perpetual_check_outcome(game: &CsaGame) -> ExportOutcome {
    match repetition_outcome_from_moves(&game.initial_position, &game.moves) {
        Ok(Some(RepetitionOutcome::PerpetualCheckLoss(loser))) => {
            ExportOutcome::Winner(loser.opposite())
        }
        _ => ExportOutcome::Unknown,
    }
}

fn nonempty_name(name: Option<&str>) -> Option<&str> {
    name.filter(|value| !value.is_empty())
}

fn outcome_name(outcome: ExportOutcome) -> &'static str {
    match outcome {
        ExportOutcome::Winner(Side::Black) => "black_win",
        ExportOutcome::Winner(Side::White) => "white_win",
        ExportOutcome::Draw => "draw",
        ExportOutcome::Unknown => "unknown",
    }
}

fn result_validation_name(validation: CsaResultValidation) -> &'static str {
    match validation {
        CsaResultValidation::Verified => "verified",
        CsaResultValidation::ExternalCondition => "external_condition",
        CsaResultValidation::Missing => "missing",
    }
}

fn csa_special_code(special: &CsaSpecialMove) -> &str {
    match special {
        CsaSpecialMove::Resign => "TORYO",
        CsaSpecialMove::Interrupted => "CHUDAN",
        CsaSpecialMove::Repetition => "SENNICHITE",
        CsaSpecialMove::PerpetualCheck => "OUTE_SENNICHITE",
        CsaSpecialMove::IllegalMove => "ILLEGAL_MOVE",
        CsaSpecialMove::BlackIllegalAction => "+ILLEGAL_ACTION",
        CsaSpecialMove::WhiteIllegalAction => "-ILLEGAL_ACTION",
        CsaSpecialMove::TimeUp => "TIME_UP",
        CsaSpecialMove::EnteringKing => "JISHOGI",
        CsaSpecialMove::Win => "KACHI",
        CsaSpecialMove::Draw => "HIKIWAKE",
        CsaSpecialMove::MaxMoves => "MAX_MOVES",
        CsaSpecialMove::Takeback => "MATTA",
        CsaSpecialMove::Checkmate => "TSUMI",
        CsaSpecialMove::NoMate => "FUZUMI",
        CsaSpecialMove::Error => "ERROR",
        CsaSpecialMove::Other(code) => code,
        _ => "UNKNOWN",
    }
}

fn validate_output_path(path: &Path) -> Result<(), String> {
    let file_name = path
        .file_name()
        .and_then(OsStr::to_str)
        .ok_or_else(|| "output path must have a valid UTF-8 file name".to_owned())?;
    if file_name.is_empty() || file_name.chars().any(char::is_control) {
        return Err("output file name contains an invalid control character".to_owned());
    }
    let parent = usable_parent(path);
    let parent_metadata = fs::symlink_metadata(parent)
        .map_err(|error| format!("cannot inspect output directory: {error}"))?;
    if parent_metadata.file_type().is_symlink() {
        return Err("output directory must not be a symbolic link".to_owned());
    }
    if !parent_metadata.is_dir() {
        return Err("output parent path is not a directory".to_owned());
    }
    match fs::symlink_metadata(path) {
        Ok(_) => Err("output file already exists; refusing to overwrite it".to_owned()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(format!("cannot inspect output path: {error}")),
    }
}

struct AtomicOutput {
    destination: PathBuf,
    parent: AnchoredDir,
    destination_name: OsString,
    file: Option<AnchoredFile>,
    bytes_written: u64,
    maximum_bytes: u64,
    destination_created: bool,
    committed: bool,
}

impl AtomicOutput {
    fn create(destination: &Path, maximum_bytes: u64) -> Result<Self, String> {
        let parent = usable_parent(destination).to_path_buf();
        let parent = AnchoredDir::open_existing(&parent)
            .map_err(|error| format!("cannot anchor output directory without symlinks: {error}"))?;
        let file_name = destination
            .file_name()
            .and_then(OsStr::to_str)
            .ok_or_else(|| "output path must have a valid UTF-8 file name".to_owned())?;
        let destination_name = OsString::from(file_name);
        if parent
            .entry_kind(&destination_name)
            .map_err(|error| format!("cannot inspect output path: {error}"))?
            .is_some()
        {
            return Err("output file already exists; refusing to overwrite it".to_owned());
        }

        for _ in 0..TEMPORARY_CREATE_ATTEMPTS {
            let sequence = TEMPORARY_FILE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
            let nonce = temporary_nonce(sequence);
            let temporary_name = OsString::from(format!(
                ".{file_name}.tmp-{}-{nonce:016x}",
                std::process::id(),
            ));
            match parent.create_new_regular(&temporary_name) {
                Ok(file) => {
                    return Ok(Self {
                        destination: destination.to_path_buf(),
                        parent,
                        destination_name,
                        file: Some(file),
                        bytes_written: 0,
                        maximum_bytes,
                        destination_created: false,
                        committed: false,
                    });
                }
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
                Err(error) => return Err(format!("cannot create temporary output: {error}")),
            }
        }
        Err("cannot create a unique temporary output file".to_owned())
    }

    fn write_line(&mut self, line: &str) -> Result<(), String> {
        let line_bytes = u64::try_from(line.len()).expect("JSON line length fits in u64");
        let new_size = self
            .bytes_written
            .checked_add(line_bytes)
            .and_then(|size| size.checked_add(1))
            .ok_or_else(|| "JSONL output size overflow".to_owned())?;
        if new_size > self.maximum_bytes {
            return Err(format!(
                "JSONL output exceeds the {}-byte defensive limit",
                self.maximum_bytes
            ));
        }
        let file = self
            .file
            .as_mut()
            .ok_or_else(|| "output file is no longer writable".to_owned())?;
        file.writer()
            .write_all(line.as_bytes())
            .and_then(|()| file.writer().write_all(b"\n"))
            .map_err(|error| format!("cannot write temporary output: {error}"))?;
        self.bytes_written = new_size;
        Ok(())
    }

    fn commit(self) -> Result<(), String> {
        self.commit_with_post_link(|_| Ok(()))
    }

    fn commit_with_post_link<F>(self, post_link: F) -> Result<(), String>
    where
        F: FnOnce(&Path) -> Result<(), String>,
    {
        self.commit_with_hooks(post_link, |_| Ok(()))
    }

    fn commit_with_hooks<F, G>(mut self, post_link: F, post_cleanup: G) -> Result<(), String>
    where
        F: FnOnce(&Path) -> Result<(), String>,
        G: FnOnce(&Path) -> Result<(), String>,
    {
        let file = self
            .file
            .take()
            .ok_or_else(|| "output file is no longer writable".to_owned())?;
        let result = (|| {
            file.sync_all()
                .map_err(|error| format!("cannot sync temporary output: {error}"))?;
            let (expected_sha256, expected_size) = file
                .retained_sha256(self.maximum_bytes)
                .map_err(|error| format!("cannot bind temporary output content: {error}"))?;
            if expected_size != self.bytes_written {
                return Err("temporary output byte count changed before publication".to_owned());
            }
            file.publish_as_new(&self.parent, &self.destination_name)
                .map_err(|error| {
                    if error.kind() == io::ErrorKind::AlreadyExists {
                        "output file appeared during export; refusing to overwrite it".to_owned()
                    } else {
                        format!("cannot atomically finalize output: {error}")
                    }
                })?;
            self.destination_created = true;
            post_link(&self.destination)?;
            self.parent
                .verify_link_to(&self.destination_name, &file)
                .map_err(|error| format!("finalized output changed after publication: {error}"))?;
            self.parent
                .remove_anchored_file(&file)
                .map_err(|error| format!("cannot remove temporary output: {error}"))?;
            post_cleanup(&self.destination)?;
            self.parent
                .sync()
                .map_err(|error| format!("cannot sync finalized output directory: {error}"))?;
            self.parent
                .verify_retained_link_content(
                    &self.destination_name,
                    &file,
                    &expected_sha256,
                    expected_size,
                )
                .map_err(|error| {
                    format!("finalized output changed after temporary cleanup: {error}")
                })?;
            Ok(())
        })();
        if let Err(error) = result {
            self.file = Some(file);
            return Err(error);
        }

        self.destination_created = false;
        self.committed = true;
        drop(file);
        Ok(())
    }
}

impl Drop for AtomicOutput {
    fn drop(&mut self) {
        if !self.committed
            && let Some(file) = self.file.take()
        {
            if self.destination_created {
                let _ = self.parent.remove_link_to(&self.destination_name, &file);
            }
            let _ = self.parent.remove_anchored_file(&file);
        }
    }
}

struct JsonLine {
    value: String,
    maximum_line_bytes: usize,
    maximum_record_bytes: usize,
}

impl JsonLine {
    fn new(maximum_record_bytes: usize) -> Self {
        Self {
            value: String::new(),
            maximum_line_bytes: maximum_record_bytes.saturating_sub(1),
            maximum_record_bytes,
        }
    }

    fn raw(&mut self, value: &str) -> Result<(), String> {
        if self
            .value
            .len()
            .checked_add(value.len())
            .is_none_or(|length| length > self.maximum_line_bytes)
        {
            return Err(format!(
                "record_too_large: JSONL record including LF exceeds the {}-byte defensive limit",
                self.maximum_record_bytes
            ));
        }
        self.value.push_str(value);
        Ok(())
    }

    fn string(&mut self, value: &str) -> Result<(), String> {
        self.raw("\"")?;
        for character in value.chars() {
            match character {
                '"' => self.raw("\\\"")?,
                '\\' => self.raw("\\\\")?,
                '\u{08}' => self.raw("\\b")?,
                '\u{0C}' => self.raw("\\f")?,
                '\n' => self.raw("\\n")?,
                '\r' => self.raw("\\r")?,
                '\t' => self.raw("\\t")?,
                character if character <= '\u{1F}' => {
                    self.raw(&format!("\\u{:04x}", u32::from(character)))?;
                }
                character => {
                    let mut encoded = [0; 4];
                    self.raw(character.encode_utf8(&mut encoded))?;
                }
            }
        }
        self.raw("\"")
    }

    fn optional_string(&mut self, value: Option<&str>) -> Result<(), String> {
        match value {
            Some(value) => self.string(value),
            None => self.raw("null"),
        }
    }

    fn finish(self) -> String {
        self.value
    }
}

fn is_single_path_component(file_name: &OsStr) -> bool {
    let mut components = Path::new(file_name).components();
    matches!(components.next(), Some(Component::Normal(_))) && components.next().is_none()
}

fn usable_parent(path: &Path) -> &Path {
    path.parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
}

fn temporary_nonce(sequence: u64) -> u64 {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    RandomState::new().hash_one((std::process::id(), sequence, timestamp))
}

fn io_error_name(kind: io::ErrorKind) -> &'static str {
    match kind {
        io::ErrorKind::NotFound => "not_found",
        io::ErrorKind::PermissionDenied => "permission_denied",
        io::ErrorKind::AlreadyExists => "already_exists",
        io::ErrorKind::InvalidData => "invalid_data",
        io::ErrorKind::InvalidInput => "invalid_input",
        io::ErrorKind::UnexpectedEof => "unexpected_eof",
        _ => "io_error",
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        path::{Path, PathBuf},
        time::{SystemTime, UNIX_EPOCH},
    };

    use open_shogi_core::{
        CsaGame, CsaResultValidation, CsaSpecialMove, parse_sfen, parse_usi_move, to_csa_game,
    };

    use super::{
        AtomicOutput, ExportConfig, ExportLimits, JsonLine, MAX_CSA_BYTES, MAX_GAMES,
        collect_candidates, export_candidate, export_directory, parse_arguments,
    };

    const LEGAL_CSA: &str = "\
V3.0
N+Black
N-White
PI
+
+7776FU
%TORYO
";

    #[test]
    fn arguments_require_bounded_max_games() {
        assert!(parse_arguments(&arguments("input", "output", "0")).is_err());
        assert!(
            parse_arguments(&arguments("input", "output", &(MAX_GAMES + 1).to_string())).is_err()
        );
        assert!(parse_arguments(&arguments("input", "output", "1")).is_ok());
    }

    #[test]
    fn export_is_sorted_and_has_legal_shape_and_lengths() {
        let directory = TestDirectory::new("shape");
        fs::write(directory.path().join("z.csa"), LEGAL_CSA).unwrap();
        fs::write(directory.path().join("a.csa"), "V3.0\nPI\n+\n%HIKIWAKE\n").unwrap();
        fs::write(directory.path().join("ignored.txt"), "not CSA").unwrap();
        let output = directory.path().join("games.jsonl");

        export(&directory, &output, 10).unwrap();

        let rendered = fs::read_to_string(output).unwrap();
        let lines = rendered.lines().collect::<Vec<_>>();
        assert_eq!(lines.len(), 2);
        assert!(lines[0].contains("\"inputFile\":\"a.csa\""));
        assert!(lines[1].contains("\"inputFile\":\"z.csa\""));
        assert!(lines[1].contains("\"schema\":\"phase3_csa_export/v1\""));
        assert!(lines[1].contains("\"status\":\"ok\""));
        assert!(lines[1].contains("\"normalizedCsa\":\"'CSA encoding=UTF-8\\nV3.0\\n"));
        assert!(lines[1].contains("\"usiMoves\":[\"7g7f\"]"));
        assert!(lines[1].contains("\"blackName\":\"Black\",\"whiteName\":\"White\""));
        assert!(lines[1].contains("\"terminalReason\":\"TORYO\""));
        assert!(lines[1].contains("\"outcome\":\"black_win\""));
        assert!(lines[1].contains("\"resultValidation\":\"external_condition\""));
        let position_array = lines[1]
            .split_once("\"positionSfens\":[")
            .unwrap()
            .1
            .split_once("],\"usiMoves\"")
            .unwrap()
            .0;
        assert_eq!(position_array.matches("\",\"").count(), 1);
        assert!(rendered.ends_with('\n'));
        assert!(!rendered.contains('\r'));
    }

    #[test]
    fn max_games_selects_the_sorted_prefix() {
        let directory = TestDirectory::new("max-games");
        for name in ["c.csa", "a.csa", "b.csa"] {
            fs::write(directory.path().join(name), LEGAL_CSA).unwrap();
        }
        let output = directory.path().join("games.jsonl");

        export(&directory, &output, 2).unwrap();

        let rendered = fs::read_to_string(output).unwrap();
        let lines = rendered.lines().collect::<Vec<_>>();
        assert_eq!(lines.len(), 2);
        assert!(lines[0].contains("\"inputFile\":\"a.csa\""));
        assert!(lines[1].contains("\"inputFile\":\"b.csa\""));
        assert!(!rendered.contains("\"inputFile\":\"c.csa\""));
    }

    #[test]
    fn candidate_replacement_after_enumeration_is_rejected() {
        let directory = TestDirectory::new("candidate-replacement");
        let input = directory.path().join("game.csa");
        fs::write(&input, LEGAL_CSA).unwrap();
        let candidates =
            collect_candidates(directory.path(), ExportLimits::PRODUCTION.directory_entries)
                .unwrap();
        assert_eq!(candidates.len(), 1);

        fs::remove_file(&input).unwrap();
        fs::write(&input, "V3.0\nPI\n+\n%CHUDAN\n").unwrap();

        let row = export_candidate(&candidates[0], ExportLimits::PRODUCTION);
        assert!(row.contains("\"status\":\"rejected\""));
        assert!(
            row.contains("\"reason\":\"path_changed: input changed after directory enumeration\"")
        );
    }

    #[test]
    fn same_inode_same_length_rewrite_with_restored_mtime_is_rejected() {
        let directory = TestDirectory::new("candidate-in-place-rewrite");
        let input = directory.path().join("game.csa");
        fs::write(&input, LEGAL_CSA).unwrap();
        let candidates =
            collect_candidates(directory.path(), ExportLimits::PRODUCTION.directory_entries)
                .unwrap();
        let original_modified = fs::metadata(&input).unwrap().modified().unwrap();
        let replacement = LEGAL_CSA.replace("+7776FU", "+2726FU");
        assert_eq!(replacement.len(), LEGAL_CSA.len());
        fs::write(&input, replacement).unwrap();
        std::fs::File::options()
            .write(true)
            .open(&input)
            .unwrap()
            .set_times(std::fs::FileTimes::new().set_modified(original_modified))
            .unwrap();

        let row = export_candidate(&candidates[0], ExportLimits::PRODUCTION);
        assert!(row.contains("\"status\":\"rejected\""));
        assert!(row.contains("path_changed: input changed after directory enumeration"));
    }

    #[cfg(unix)]
    #[test]
    fn input_parent_rename_and_symlink_substitution_cannot_redirect_a_candidate() {
        use std::os::unix::fs::symlink;

        let fixture = TestDirectory::new("input-parent-swap");
        let input = fixture.path().join("input");
        let moved = fixture.path().join("input-moved");
        let external = fixture.path().join("external");
        fs::create_dir(&input).unwrap();
        fs::create_dir(&external).unwrap();
        fs::write(input.join("game.csa"), LEGAL_CSA).unwrap();
        fs::write(external.join("game.csa"), "attacker").unwrap();
        let candidates =
            collect_candidates(&input, ExportLimits::PRODUCTION.directory_entries).unwrap();

        fs::rename(&input, &moved).unwrap();
        symlink(&external, &input).unwrap();

        let row = export_candidate(&candidates[0], ExportLimits::PRODUCTION);
        assert!(row.contains("\"status\":\"ok\""));
        assert!(row.contains("\"normalizedCsa\""));
        assert!(!row.contains("attacker"));
    }

    #[test]
    fn invalid_csa_produces_a_rejection_without_aborting_the_batch() {
        let directory = TestDirectory::new("invalid");
        fs::write(directory.path().join("bad.csa"), "V3.0\nPI\n+\n+7775FU\n").unwrap();
        fs::write(directory.path().join("good.csa"), LEGAL_CSA).unwrap();
        let output = directory.path().join("games.jsonl");

        export(&directory, &output, 10).unwrap();

        let rendered = fs::read_to_string(output).unwrap();
        assert!(rendered.contains(
            "\"status\":\"rejected\",\"inputFile\":\"bad.csa\",\"reason\":\"invalid_csa:"
        ));
        assert!(rendered.contains("\"status\":\"ok\",\"inputFile\":\"good.csa\""));
    }

    #[cfg(unix)]
    #[test]
    fn symlink_and_nonregular_csa_inputs_are_rejected() {
        use std::os::unix::fs::symlink;

        let directory = TestDirectory::new("symlink");
        fs::write(directory.path().join("target"), LEGAL_CSA).unwrap();
        symlink("target", directory.path().join("link.csa")).unwrap();
        fs::create_dir(directory.path().join("nested.csa")).unwrap();
        let output = directory.path().join("games.jsonl");

        export(&directory, &output, 10).unwrap();

        let rendered = fs::read_to_string(output).unwrap();
        assert!(rendered.contains(
            "\"inputFile\":\"link.csa\",\"reason\":\"symlink: symbolic links are not accepted\""
        ));
        assert!(rendered.contains(
            "\"inputFile\":\"nested.csa\",\"reason\":\"non_regular_file: input is not a regular file\""
        ));
    }

    #[cfg(unix)]
    #[test]
    fn control_characters_in_input_names_are_rejected_and_escaped() {
        let directory = TestDirectory::new("control-name");
        fs::write(directory.path().join("line\nbreak.csa"), LEGAL_CSA).unwrap();
        let output = directory.path().join("games.jsonl");

        export(&directory, &output, 10).unwrap();

        let rendered = fs::read_to_string(output).unwrap();
        assert!(rendered.contains(
            "\"inputFile\":\"line\\nbreak.csa\",\"reason\":\"invalid_filename: filename contains a control character\""
        ));
        assert_eq!(rendered.lines().count(), 1);
    }

    #[test]
    fn defensive_caps_are_enforced_without_final_output() {
        let directory = TestDirectory::new("caps");
        fs::write(directory.path().join("one.csa"), LEGAL_CSA).unwrap();
        fs::write(directory.path().join("two.txt"), "ignored").unwrap();
        let directory_limit_output = directory.path().join("directory-limit.jsonl");
        let limits = ExportLimits {
            directory_entries: 1,
            ..ExportLimits::PRODUCTION
        };
        let config = ExportConfig {
            input_dir: directory.path().to_path_buf(),
            output: directory_limit_output.clone(),
            max_games: 10,
        };
        assert!(export_directory(&config, limits).is_err());
        assert!(!directory_limit_output.exists());

        let oversized_output = directory.path().join("oversized.jsonl");
        fs::write(
            directory.path().join("large.csa"),
            vec![b'x'; MAX_CSA_BYTES + 1],
        )
        .unwrap();
        export(&directory, &oversized_output, 10).unwrap();
        assert!(
            fs::read_to_string(oversized_output)
                .unwrap()
                .contains("\"reason\":\"input_too_large:")
        );
    }

    #[test]
    fn existing_output_is_never_overwritten() {
        let directory = TestDirectory::new("non-overwrite");
        fs::write(directory.path().join("game.csa"), LEGAL_CSA).unwrap();
        let output = directory.path().join("games.jsonl");
        fs::write(&output, "sentinel").unwrap();

        assert!(export(&directory, &output, 10).is_err());
        assert_eq!(fs::read_to_string(output).unwrap(), "sentinel");
    }

    #[test]
    fn output_appearing_during_export_is_preserved() {
        let directory = TestDirectory::new("late-output");
        let destination = directory.path().join("games.jsonl");
        let mut output = AtomicOutput::create(&destination, 1_024).unwrap();
        output.write_line("{}").unwrap();
        fs::write(&destination, "sentinel").unwrap();

        assert!(output.commit().is_err());
        assert_eq!(fs::read_to_string(destination).unwrap(), "sentinel");
        assert!(fs::read_dir(directory.path()).unwrap().all(|entry| {
            !entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .contains(".tmp-")
        }));
    }

    #[test]
    fn replaced_temporary_path_is_not_published_or_deleted() {
        let directory = TestDirectory::new("temp-replacement");
        let destination = directory.path().join("games.jsonl");
        let mut output = AtomicOutput::create(&destination, 1_024).unwrap();
        output.write_line("{\"safe\":true}").unwrap();
        let temporary = output.file.as_ref().unwrap().display().to_path_buf();
        fs::remove_file(&temporary).unwrap();
        fs::write(&temporary, "{\"attacker\":true}\n").unwrap();

        assert!(output.commit().is_err());
        assert!(!destination.exists());
        assert_eq!(
            fs::read_to_string(temporary).unwrap(),
            "{\"attacker\":true}\n"
        );
    }

    #[cfg(unix)]
    #[test]
    fn substituted_temporary_symlink_is_not_followed_or_deleted() {
        use std::os::unix::fs::symlink;

        let directory = TestDirectory::new("temp-symlink");
        let destination = directory.path().join("games.jsonl");
        let foreign = directory.path().join("foreign");
        fs::write(&foreign, "foreign").unwrap();
        let mut output = AtomicOutput::create(&destination, 1_024).unwrap();
        output.write_line("{\"safe\":true}").unwrap();
        let temporary = output.file.as_ref().unwrap().display().to_path_buf();
        fs::remove_file(&temporary).unwrap();
        symlink(&foreign, &temporary).unwrap();

        assert!(output.commit().is_err());
        assert!(!destination.exists());
        assert!(
            fs::symlink_metadata(&temporary)
                .unwrap()
                .file_type()
                .is_symlink()
        );
        assert_eq!(fs::read_to_string(foreign).unwrap(), "foreign");
    }

    #[test]
    fn substituted_destination_after_link_is_preserved_and_rejected() {
        let directory = TestDirectory::new("destination-substitution");
        let destination = directory.path().join("games.jsonl");
        let replacement = directory.path().join("replacement");
        fs::write(&replacement, "{\"attacker\":true}\n").unwrap();
        let mut output = AtomicOutput::create(&destination, 1_024).unwrap();
        output.write_line("{\"safe\":true}").unwrap();

        let result = output.commit_with_post_link(|published| {
            fs::remove_file(published).map_err(|error| error.to_string())?;
            fs::rename(&replacement, published).map_err(|error| error.to_string())
        });

        assert!(result.is_err());
        assert_eq!(
            fs::read_to_string(destination).unwrap(),
            "{\"attacker\":true}\n"
        );
        assert!(!replacement.exists());
    }

    #[test]
    fn substituted_destination_after_temporary_cleanup_is_preserved_and_rejected() {
        let directory = TestDirectory::new("destination-final-substitution");
        let destination = directory.path().join("games.jsonl");
        let replacement = directory.path().join("replacement");
        fs::write(&replacement, "{\"attacker\":true}\n").unwrap();
        let mut output = AtomicOutput::create(&destination, 1_024).unwrap();
        output.write_line("{\"safe\":true}").unwrap();

        let result = output.commit_with_hooks(
            |_| Ok(()),
            |published| {
                fs::remove_file(published).map_err(|error| error.to_string())?;
                fs::rename(&replacement, published).map_err(|error| error.to_string())
            },
        );

        assert!(result.is_err());
        assert_eq!(
            fs::read_to_string(destination).unwrap(),
            "{\"attacker\":true}\n"
        );
        assert!(!replacement.exists());
    }

    #[cfg(unix)]
    #[test]
    fn output_parent_swap_cannot_redirect_streaming_writes() {
        use std::os::unix::fs::symlink;

        let fixture = TestDirectory::new("output-parent-swap");
        let parent = fixture.path().join("output");
        let moved = fixture.path().join("output-moved");
        let external = fixture.path().join("external");
        fs::create_dir(&parent).unwrap();
        fs::create_dir(&external).unwrap();
        let destination = parent.join("games.jsonl");
        let mut output = AtomicOutput::create(&destination, 1_024).unwrap();
        fs::rename(&parent, &moved).unwrap();
        symlink(&external, &parent).unwrap();

        output.write_line("{\"safe\":true}").unwrap();
        output.commit().unwrap();

        assert_eq!(
            fs::read_to_string(moved.join("games.jsonl")).unwrap(),
            "{\"safe\":true}\n"
        );
        assert!(!external.join("games.jsonl").exists());
    }

    #[test]
    fn json_strings_escape_quotes_backslashes_and_controls() {
        let mut json = JsonLine::new(1_024);
        json.string("quote\" slash\\ \u{08}\u{0c}\n\r\t\u{01} 日本")
            .unwrap();
        assert_eq!(
            json.finish(),
            "\"quote\\\" slash\\\\ \\b\\f\\n\\r\\t\\u0001 日本\""
        );
        let controls = (0..=0x1f).filter_map(char::from_u32).collect::<String>();
        let mut escaped_controls = JsonLine::new(1_024);
        escaped_controls.string(&controls).unwrap();
        assert!(escaped_controls.finish().bytes().all(|byte| byte >= b' '));

        let directory = TestDirectory::new("escaping");
        let source = "V3.0\nN+A\"B\\\\\t\u{01}\nPI\n+\n%CHUDAN\n";
        fs::write(directory.path().join("escape.csa"), source).unwrap();
        let output = directory.path().join("games.jsonl");
        export(&directory, &output, 10).unwrap();
        let rendered = fs::read_to_string(output).unwrap();
        assert!(rendered.contains("\"blackName\":\"A\\\"B\\\\\\\\\\t\\u0001\""));
    }

    #[test]
    fn jsonl_record_limit_includes_the_trailing_line_feed() {
        let mut exact = JsonLine::new(4);
        exact.raw("abc").unwrap();
        let line = exact.finish();
        assert_eq!(line.len() + 1, 4);

        let mut oversized = JsonLine::new(4);
        assert!(oversized.raw("abcd").is_err());
    }

    #[test]
    fn empty_player_names_normalize_to_null() {
        let directory = TestDirectory::new("empty-names");
        fs::write(
            directory.path().join("game.csa"),
            "V3.0\nN+\nN-\nPI\n+\n%CHUDAN\n",
        )
        .unwrap();
        let output = directory.path().join("games.jsonl");

        export(&directory, &output, 1).unwrap();

        let rendered = fs::read_to_string(output).unwrap();
        assert!(rendered.contains("\"blackName\":null,\"whiteName\":null"));
    }

    #[test]
    fn only_explicit_or_replay_verified_repetition_is_labeled_as_draw() {
        let directory = TestDirectory::new("outcomes");
        fs::write(
            directory.path().join("draw.csa"),
            "V3.0\nPI\n+\n%HIKIWAKE\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("max.csa"),
            "V3.0\nPI\n+\n%MAX_MOVES\n",
        )
        .unwrap();
        fs::write(
            directory.path().join("repetition.csa"),
            "V3.0\nPI\n+\n%SENNICHITE\n",
        )
        .unwrap();
        let initial = parse_sfen("4k4/9/9/9/9/9/9/9/4K4 b - 1").unwrap();
        let mut moves = Vec::new();
        for _ in 0..3 {
            moves.extend(
                ["5i5h", "5a5b", "5h5i", "5b5a"].map(|notation| parse_usi_move(notation).unwrap()),
            );
        }
        let verified_repetition = CsaGame {
            version: "V3.0".to_owned(),
            black_name: None,
            white_name: None,
            metadata: Vec::new(),
            initial_position: initial,
            moves,
            special_move: Some(CsaSpecialMove::Repetition),
            result_validation: CsaResultValidation::Verified,
        };
        fs::write(
            directory.path().join("verified-repetition.csa"),
            to_csa_game(&verified_repetition).unwrap(),
        )
        .unwrap();
        let output = directory.path().join("games.jsonl");

        export(&directory, &output, 4).unwrap();

        let lines = fs::read_to_string(output).unwrap();
        let lines = lines.lines().collect::<Vec<_>>();
        assert!(lines[0].contains("\"inputFile\":\"draw.csa\""));
        assert!(lines[0].contains("\"outcome\":\"draw\""));
        assert!(lines[1].contains("\"inputFile\":\"max.csa\""));
        assert!(lines[1].contains("\"outcome\":\"unknown\""));
        assert!(lines[2].contains("\"inputFile\":\"repetition.csa\""));
        assert!(lines[2].contains("\"outcome\":\"unknown\""));
        assert!(lines[3].contains("\"inputFile\":\"verified-repetition.csa\""));
        assert!(lines[3].contains("\"outcome\":\"draw\""));
        assert!(lines[3].contains("\"resultValidation\":\"verified\""));
    }

    #[test]
    fn perpetual_check_after_an_earlier_restart_keeps_its_winner() {
        let directory = TestDirectory::new("continued-perpetual");
        let initial = parse_sfen("4k4/5R3/9/9/9/9/9/9/K8 b - 1").expect("repetition fixture");
        let mut moves = Vec::new();
        for _ in 0..3 {
            moves.extend(
                ["9i8h", "5a6a", "8h9i", "6a5a"]
                    .map(|notation| parse_usi_move(notation).expect("ordinary repetition move")),
            );
        }
        for _ in 0..3 {
            moves.extend(
                ["4b5b", "5a4a", "5b4b", "4a5a"]
                    .map(|notation| parse_usi_move(notation).expect("perpetual-check move")),
            );
        }
        let game = CsaGame {
            version: "V3.0".to_owned(),
            black_name: None,
            white_name: None,
            metadata: Vec::new(),
            initial_position: initial,
            moves,
            special_move: Some(CsaSpecialMove::PerpetualCheck),
            result_validation: CsaResultValidation::Verified,
        };
        fs::write(
            directory.path().join("game.csa"),
            to_csa_game(&game).expect("verified continued-history record"),
        )
        .unwrap();
        let output = directory.path().join("games.jsonl");

        export(&directory, &output, 1).unwrap();

        let rendered = fs::read_to_string(output).unwrap();
        assert!(rendered.contains("\"terminalReason\":\"OUTE_SENNICHITE\""));
        assert!(rendered.contains("\"outcome\":\"white_win\""));
        assert!(rendered.contains("\"resultValidation\":\"verified\""));
    }

    #[test]
    fn fatal_input_and_output_failures_leave_no_partial_destination() {
        let directory = TestDirectory::new("fatal");
        let missing_input = directory.path().join("missing");
        let output = directory.path().join("missing.jsonl");
        let config = ExportConfig {
            input_dir: missing_input,
            output: output.clone(),
            max_games: 1,
        };
        assert!(export_directory(&config, ExportLimits::PRODUCTION).is_err());
        assert!(!output.exists());

        fs::write(directory.path().join("game.csa"), LEGAL_CSA).unwrap();
        let capped_output = directory.path().join("capped.jsonl");
        let limits = ExportLimits {
            output_bytes: 1,
            ..ExportLimits::PRODUCTION
        };
        let config = ExportConfig {
            input_dir: directory.path().to_path_buf(),
            output: capped_output.clone(),
            max_games: 1,
        };
        assert!(export_directory(&config, limits).is_err());
        assert!(!capped_output.exists());
        assert!(fs::read_dir(directory.path()).unwrap().all(|entry| {
            !entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .contains(".tmp-")
        }));
    }

    fn arguments(input: &str, output: &str, max_games: &str) -> Vec<String> {
        [
            "--input-dir",
            input,
            "--output",
            output,
            "--max-games",
            max_games,
        ]
        .map(str::to_owned)
        .to_vec()
    }

    fn export(directory: &TestDirectory, output: &Path, max_games: usize) -> Result<(), String> {
        export_directory(
            &ExportConfig {
                input_dir: directory.path().to_path_buf(),
                output: output.to_path_buf(),
                max_games,
            },
            ExportLimits::PRODUCTION,
        )
    }

    struct TestDirectory {
        path: PathBuf,
    }

    impl TestDirectory {
        fn new(label: &str) -> Self {
            let nonce = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("clock is after epoch")
                .as_nanos();
            let path = std::env::temp_dir().canonicalize().unwrap().join(format!(
                "open-shogi-dataset-{label}-{}-{nonce}",
                std::process::id()
            ));
            fs::create_dir(&path).unwrap();
            Self { path }
        }

        fn path(&self) -> &Path {
            &self.path
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }
}
