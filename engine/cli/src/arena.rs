use std::{
    ffi::OsStr,
    fmt::Write as _,
    io,
    path::PathBuf,
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

#[cfg(test)]
use std::{fs::File, io::Write as _, path::Path};

use open_shogi_core::{
    AnchoredDir, AnchoredFile, CancellationToken, CsaGame, CsaResultValidation, CsaSpecialMove,
    EngineIdentity, EntryKind, EvaluationConfig, Game, GameEnd, MAX_NEURAL_MODEL_BYTES, Move,
    MAX_SECURE_CLEANUP_QUARANTINE_ENTRIES, NeuralEvaluationMode, NeuralEvaluator,
    NeuralQuantization, Osaval02Evaluator, Osaval02Quantization, Position, RandomMoveSelector,
    RepetitionOutcome, SearchConfig, SearchEngine, SearchLimits, SearchStats, SearchTermination,
    Side,
    parse_csa_game, parse_sfen, to_csa_game, to_sfen,
};

use crate::{
    args::{next_value, parse_next, path_next},
    checksum::{
        read_file_artifact, read_open_file_artifact, read_retained_file_artifact, sha256_bytes,
    },
    opening::{OpeningBook, OpeningPolicy, OpeningProfile},
    tools::splitmix64,
};

const SCHEMA: &str = "phase2_arena_report/v2";
const STATE_SCHEMA: &str = "phase2_arena_state/v4";
const MAX_GAMES: u32 = 10_000;
const DEFAULT_GAMES: u32 = 20;
const DEFAULT_MAX_PLIES: u32 = 512;
const DEFAULT_SEED: u64 = 0x0015_4149_5F41_5232;
const MAX_JSON_SAFE_INTEGER: u64 = 9_007_199_254_740_991;
const DEFAULT_NODES: u64 = 5_000;
const MAX_NODES_PER_MOVE: u64 = 1_000_000_000;
const MAX_MOVETIME_MS: u64 = 3_600_000;
const MAX_PLIES: u32 = 10_000;
const MAX_STATE_BYTES: usize = 5_242_880;
const MAX_REPORT_BYTES: usize = 5_242_880;
const MAX_REPORT_FIXED_BYTES: usize = 65_536;
const MAX_REPORT_DYNAMIC_TEXT_BYTES: usize = 1_024;
const LOCK_FILE_NAME: &str = ".arena.lock";
const SECURE_CLEANUP_DIRECTORY_NAME: &str = ".open-shogi-cleanup";
const JOURNAL_FILE_NAME: &str = ".arena-game-journal";
const JOURNAL_SCHEMA: &str = "phase2_arena_game_journal/v1";
const MAX_CSA_BYTES: u64 = 16 * 1024 * 1024;
const MAX_JOURNAL_BYTES: u64 = MAX_CSA_BYTES + 64 * 1024;
const MAX_PRESERVED_UNPROVEN_TEMPORARIES: usize = MAX_SECURE_CLEANUP_QUARANTINE_ENTRIES;
// A complete arena may contain every game plus the full tolerated crash-remnant budget.
// Keep a small fixed allowance for the state, report, journal, and reserved entries.
const MAX_ARENA_DIRECTORY_ENTRIES: usize =
    MAX_GAMES as usize + MAX_PRESERVED_UNPROVEN_TEMPORARIES + 32;
#[cfg(test)]
static TEMPORARY_FILE_SEQUENCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PlayerKind {
    Random,
    Material,
    HandcraftedBaseline,
    HandcraftedExperimental,
    Neural,
    Residual,
    Composite,
}

impl PlayerKind {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "random" => Ok(Self::Random),
            "material" => Ok(Self::Material),
            "handcrafted" | "handcrafted-baseline" => Ok(Self::HandcraftedBaseline),
            "search" | "experimental" | "handcrafted-experimental" => {
                Ok(Self::HandcraftedExperimental)
            }
            "neural" => Ok(Self::Neural),
            "residual" => Ok(Self::Residual),
            "composite" => Ok(Self::Composite),
            _ => Err(
                "player type must be random, material, handcrafted-baseline, handcrafted-experimental, neural, residual, or composite"
                    .to_owned(),
            ),
        }
    }

    const fn is_search(self) -> bool {
        !matches!(self, Self::Random)
    }

    const fn uses_model(self) -> bool {
        matches!(self, Self::Neural | Self::Residual | Self::Composite)
    }

    const fn neural_mode(self) -> Option<NeuralEvaluationMode> {
        match self {
            Self::Neural => Some(NeuralEvaluationMode::PureValue),
            Self::Residual => Some(NeuralEvaluationMode::Residual),
            Self::Composite => Some(NeuralEvaluationMode::Composite),
            _ => None,
        }
    }

    const fn report_name(self) -> &'static str {
        match self {
            Self::Random => "random",
            Self::Material => "material",
            Self::HandcraftedBaseline => "handcrafted-baseline",
            Self::HandcraftedExperimental => "handcrafted-experimental",
            Self::Neural => "neural",
            Self::Residual => "residual",
            Self::Composite => "composite",
        }
    }
}

#[derive(Clone, Debug)]
struct PlayerSpec {
    kind: PlayerKind,
    depth: u8,
    hash_megabytes: usize,
    transposition: bool,
    model_path: Option<PathBuf>,
    model_sha256: Option<String>,
    model_size: Option<u64>,
    model: Option<LoadedModel>,
    opening: bool,
    opening_profile: OpeningProfile,
}

#[derive(Clone, Debug)]
enum LoadedModel {
    Osaval01(Arc<NeuralEvaluator>),
    Osaval02(Arc<Osaval02Evaluator>),
}

impl PlayerSpec {
    fn label(&self) -> String {
        match self.kind {
            PlayerKind::Random => "random".to_owned(),
            PlayerKind::Material => self.search_label("material"),
            PlayerKind::HandcraftedBaseline => self.search_label("handcrafted-baseline"),
            PlayerKind::HandcraftedExperimental => self.search_label("handcrafted-experimental"),
            PlayerKind::Neural => self.search_label("neural"),
            PlayerKind::Residual => self.search_label("residual"),
            PlayerKind::Composite => self.search_label("composite-50-50"),
        }
    }

    fn search_label(&self, evaluator: &str) -> String {
        let model = self
            .model_sha256
            .as_deref()
            .map_or_else(String::new, |hash| format!(":m-{}", &hash[..12]));
        let opening = if self.opening {
            format!("on:{}", self.opening_profile.name())
        } else {
            "off".to_owned()
        };
        format!(
            "search:{evaluator}:d{}:h{}:tt-{}:book-{opening}{model}",
            self.depth,
            self.hash_megabytes,
            if self.transposition { "on" } else { "off" }
        )
    }

    fn model_payload_sha256(&self) -> Option<String> {
        self.model.as_ref().map(|model| match model {
            LoadedModel::Osaval01(model) => model.identity().sha256_hex(),
            LoadedModel::Osaval02(model) => model.identity().weight_payload_sha256.clone(),
        })
    }

    fn model_architecture_version(&self) -> Option<u32> {
        self.model.as_ref().map(|model| match model {
            LoadedModel::Osaval01(model) => model.identity().architecture_version,
            LoadedModel::Osaval02(model) => model.identity().format_version,
        })
    }

    fn model_quantization(&self) -> Option<&'static str> {
        self.model.as_ref().map(|model| match model {
            LoadedModel::Osaval01(model) => match model.quantization() {
                NeuralQuantization::Float32 => "float32",
                NeuralQuantization::Int8 => "int8",
            },
            LoadedModel::Osaval02(model) => match model.quantization() {
                Osaval02Quantization::Float32 => "float32",
                Osaval02Quantization::Int8 => "int8",
            },
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum Budget {
    Nodes(u64),
    MoveTime(u64),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ArenaConfigSignaturePlayer {
    pub(crate) label: String,
    pub(crate) evaluator_kind: String,
    pub(crate) search_depth: Option<u64>,
    pub(crate) hash_megabytes: Option<u64>,
    pub(crate) transposition: Option<bool>,
    pub(crate) model_artifact_sha256: Option<String>,
    pub(crate) model_artifact_size: Option<u64>,
    pub(crate) model_payload_sha256: Option<String>,
    pub(crate) architecture_version: Option<u64>,
    pub(crate) quantization: Option<String>,
    pub(crate) opening_enabled: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ArenaConfigSignatureOpening {
    pub(crate) enabled: bool,
    pub(crate) artifact_sha256: Option<String>,
    pub(crate) artifact_size: Option<u64>,
    pub(crate) max_plies: Option<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ArenaConfigSignature {
    pub(crate) games: u64,
    pub(crate) seed: u64,
    pub(crate) initial_sfen: String,
    pub(crate) max_plies: u64,
    pub(crate) git_commit: Option<String>,
    pub(crate) budget_kind: String,
    pub(crate) budget_value: u64,
    pub(crate) player_a: ArenaConfigSignaturePlayer,
    pub(crate) player_b: ArenaConfigSignaturePlayer,
    pub(crate) opening: ArenaConfigSignatureOpening,
}

#[derive(Clone, Debug)]
struct ArenaConfig {
    games: u32,
    player_a: PlayerSpec,
    player_b: PlayerSpec,
    budget: Budget,
    initial_position: Position,
    initial_sfen: String,
    max_plies: u32,
    seed: u64,
    git_commit: Option<String>,
    output_dir: PathBuf,
    resume: bool,
    opening_book_path: Option<PathBuf>,
    opening_book_sha256: Option<String>,
    opening_book_size: Option<u64>,
    opening_max_plies: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ArenaResult {
    BlackWin,
    WhiteWin,
    Draw,
    MaxPlies,
}

impl ArenaResult {
    const fn json(self) -> &'static str {
        match self {
            Self::BlackWin => "black_win",
            Self::WhiteWin => "white_win",
            Self::Draw => "draw",
            Self::MaxPlies => "max_plies",
        }
    }

    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "black_win" => Ok(Self::BlackWin),
            "white_win" => Ok(Self::WhiteWin),
            "draw" => Ok(Self::Draw),
            "max_plies" => Ok(Self::MaxPlies),
            _ => Err("arena state has an invalid result".to_owned()),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct GameSummary {
    id: u32,
    black: String,
    white: String,
    result: ArenaResult,
    moves: u32,
    csa_path: String,
    csa_sha256: String,
    csa_size: u64,
    special_move: String,
    search_nodes: u64,
    search_elapsed_ns: u64,
    depth_sum: u64,
    searches: u64,
    tt_probes: u64,
    tt_hits: u64,
    cutoffs: u64,
    candidate_moves: u64,
    pruned_moves: u64,
    neural_inference_calls: u64,
    neural_inference_time_ns: u64,
    player_a_search_nodes: u64,
    player_a_search_elapsed_ns: u64,
    player_a_depth_sum: u64,
    player_a_searches: u64,
    player_a_neural_inference_calls: u64,
    player_a_neural_inference_time_ns: u64,
    player_b_search_nodes: u64,
    player_b_search_elapsed_ns: u64,
    player_b_depth_sum: u64,
    player_b_searches: u64,
    player_b_neural_inference_calls: u64,
    player_b_neural_inference_time_ns: u64,
    illegal_moves: u64,
    search_winner: bool,
    player_a_winner: bool,
    player_b_winner: bool,
}

#[derive(Clone, Copy, Debug, Default)]
struct SearchTotals {
    nodes: u64,
    elapsed_ns: u64,
    depth_sum: u64,
    searches: u64,
    tt_probes: u64,
    tt_hits: u64,
    cutoffs: u64,
    candidate_moves: u64,
    pruned_moves: u64,
    neural_inference_calls: u64,
    neural_inference_time_ns: u64,
}

struct GameOutcome {
    result: ArenaResult,
    special: CsaSpecialMove,
    validation: CsaResultValidation,
}

struct PendingGame {
    summary: GameSummary,
    csa: String,
}

#[derive(Debug)]
struct ArenaStorage {
    root: AnchoredDir,
    games: AnchoredDir,
}

pub fn run(arguments: &[String]) -> Result<(), String> {
    let mut config = parse_arguments(arguments)?;
    preflight_report_size(&config)?;
    let opening_book = load_opening_book(&mut config)?;
    if let Some(book) = &opening_book {
        println!("arena loaded {} opening records", book.records());
    }
    let storage = prepare_output_directory(&config)?;
    let signature = config_signature(&config);
    let state_exists = storage
        .root
        .entry_kind(OsStr::new("arena.state"))
        .map_err(|error| format!("cannot inspect arena state: {error}"))?
        .is_some();
    if config.resume && !state_exists {
        return Err("arena --resume requires an existing arena.state".to_owned());
    }
    recover_owned_temporary_files(&storage)?;
    let (started_at, mut summaries) = if config.resume && state_exists {
        read_state(&storage.root, &signature)?
    } else {
        (timestamp(), Vec::new())
    };
    if summaries.len() > usize::try_from(config.games).expect("game count fits usize") {
        return Err("arena resume state contains more games than requested".to_owned());
    }
    if !config.resume || !state_exists {
        write_state(&storage.root, &signature, &started_at, &summaries)?;
    }
    recover_game_journal(
        &config,
        opening_book.as_ref(),
        &storage,
        &signature,
        &started_at,
        &mut summaries,
    )?;
    validate_resume_artifacts(&config, opening_book.as_ref(), &storage, &summaries)?;

    let resume_report = if config.resume {
        validate_resume_report(&storage.root, &config, &started_at, &summaries)?
    } else {
        ResumeReportState::Recoverable
    };
    if config.resume
        && summaries.len() == usize::try_from(config.games).expect("game count fits usize")
    {
        if resume_report == ResumeReportState::Completed {
            println!(
                "arena report {}",
                config.output_dir.join("arena-report.json").display()
            );
            return Ok(());
        }
        let completed_at = timestamp();
        write_report(
            &storage.root,
            &config,
            &started_at,
            Some(&completed_at),
            &summaries,
        )?;
        println!(
            "arena report {}",
            config.output_dir.join("arena-report.json").display()
        );
        return Ok(());
    }

    for id in u32::try_from(summaries.len()).expect("summary count fits u32")..config.games {
        let pending = play_game(&config, opening_book.as_ref(), id)?;
        commit_pending_game(
            &config,
            opening_book.as_ref(),
            &storage,
            &signature,
            &started_at,
            &mut summaries,
            &pending,
        )?;
        write_report(&storage.root, &config, &started_at, None, &summaries)?;
        println!("arena completed game {}/{}", id + 1, config.games);
    }
    let completed_at = timestamp();
    write_report(
        &storage.root,
        &config,
        &started_at,
        Some(&completed_at),
        &summaries,
    )?;
    println!(
        "arena report {}",
        config.output_dir.join("arena-report.json").display()
    );
    Ok(())
}

#[expect(
    clippy::too_many_lines,
    reason = "the bounded arena option parser keeps paired player settings explicit"
)]
fn parse_arguments(arguments: &[String]) -> Result<ArenaConfig, String> {
    let mut games = DEFAULT_GAMES;
    let mut player_a = PlayerSpec {
        kind: PlayerKind::HandcraftedExperimental,
        depth: 4,
        hash_megabytes: 16,
        transposition: true,
        model_path: None,
        model_sha256: None,
        model_size: None,
        model: None,
        opening: false,
        opening_profile: OpeningProfile::IbishaStrict,
    };
    let mut player_b = PlayerSpec {
        kind: PlayerKind::Random,
        depth: 4,
        hash_megabytes: 16,
        transposition: true,
        model_path: None,
        model_sha256: None,
        model_size: None,
        model: None,
        opening: false,
        opening_profile: OpeningProfile::IbishaStrict,
    };
    let mut nodes = None;
    let mut movetime_ms = None;
    let mut sfen = None;
    let mut max_plies = DEFAULT_MAX_PLIES;
    let mut seed = DEFAULT_SEED;
    let mut git_commit = None;
    let mut output_dir = PathBuf::from("artifacts/arena");
    let mut resume = false;
    let mut opening_book_path = None;
    let mut opening_max_plies = 24_u32;
    let mut seen_options = std::collections::BTreeSet::new();
    let mut index = 0;
    while index < arguments.len() {
        let option = arguments[index].as_str();
        if !seen_options.insert(option) {
            return Err(format!("duplicate arena argument: {option}"));
        }
        match option {
            "--games" => games = parse_next(arguments, &mut index, "--games")?,
            "--player-a" => {
                player_a.kind =
                    PlayerKind::parse(next_value(arguments, &mut index, "--player-a")?)?;
            }
            "--player-b" => {
                player_b.kind =
                    PlayerKind::parse(next_value(arguments, &mut index, "--player-b")?)?;
            }
            "--a-depth" => player_a.depth = parse_next(arguments, &mut index, "--a-depth")?,
            "--b-depth" => player_b.depth = parse_next(arguments, &mut index, "--b-depth")?,
            "--a-hash-mb" => {
                player_a.hash_megabytes = parse_next(arguments, &mut index, "--a-hash-mb")?;
            }
            "--b-hash-mb" => {
                player_b.hash_megabytes = parse_next(arguments, &mut index, "--b-hash-mb")?;
            }
            "--a-no-tt" => player_a.transposition = false,
            "--b-no-tt" => player_b.transposition = false,
            "--a-model" => {
                player_a.model_path = Some(path_next(arguments, &mut index, "--a-model")?);
            }
            "--b-model" => {
                player_b.model_path = Some(path_next(arguments, &mut index, "--b-model")?);
            }
            "--a-opening" => player_a.opening = true,
            "--b-opening" => player_b.opening = true,
            "--a-opening-profile" => {
                player_a.opening_profile =
                    OpeningProfile::parse(next_value(arguments, &mut index, option)?)?;
            }
            "--b-opening-profile" => {
                player_b.opening_profile =
                    OpeningProfile::parse(next_value(arguments, &mut index, option)?)?;
            }
            "--opening-book" => {
                opening_book_path = Some(path_next(arguments, &mut index, "--opening-book")?);
            }
            "--opening-max-plies" => {
                opening_max_plies = parse_next(arguments, &mut index, "--opening-max-plies")?;
            }
            "--nodes" => nodes = Some(parse_next(arguments, &mut index, "--nodes")?),
            "--movetime-ms" => {
                movetime_ms = Some(parse_next(arguments, &mut index, "--movetime-ms")?);
            }
            "--sfen" => sfen = Some(next_value(arguments, &mut index, "--sfen")?.to_owned()),
            "--max-plies" => {
                max_plies = parse_next(arguments, &mut index, "--max-plies")?;
            }
            "--seed" => seed = parse_next(arguments, &mut index, "--seed")?,
            "--git-commit" => {
                git_commit = Some(next_value(arguments, &mut index, "--git-commit")?.to_owned());
            }
            "--output-dir" => {
                output_dir = path_next(arguments, &mut index, "--output-dir")?;
            }
            "--resume" => resume = true,
            argument => return Err(format!("unknown arena argument: {argument}")),
        }
        index += 1;
    }
    if games == 0 || games > MAX_GAMES {
        return Err(format!("--games must be 1..={MAX_GAMES}"));
    }
    if max_plies == 0 || max_plies > MAX_PLIES {
        return Err(format!("--max-plies must be 1..={MAX_PLIES}"));
    }
    if seed > MAX_JSON_SAFE_INTEGER {
        return Err(format!(
            "--seed must not exceed the JSON-safe integer limit {MAX_JSON_SAFE_INTEGER}"
        ));
    }
    validate_player("a", &player_a)?;
    validate_player("b", &player_b)?;
    resolve_models(&mut player_a, &mut player_b)?;
    if (player_a.opening || player_b.opening) && opening_book_path.is_none() {
        return Err("--a-opening/--b-opening requires --opening-book".to_owned());
    }
    if opening_book_path.is_some() && !player_a.opening && !player_b.opening {
        return Err("--opening-book requires --a-opening or --b-opening".to_owned());
    }
    if opening_max_plies == 0 || opening_max_plies > MAX_PLIES {
        return Err(format!("--opening-max-plies must be 1..={MAX_PLIES}"));
    }
    let opening_book_sha256 = None;
    let budget = parse_budget(nodes, movetime_ms)?;
    let requested_sfen = sfen.unwrap_or_else(|| {
        "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1".to_owned()
    });
    let initial_position = parse_initial_position(&requested_sfen)?;
    let initial_sfen = to_sfen(&initial_position);
    validate_git_commit(git_commit.as_deref())?;
    Ok(ArenaConfig {
        games,
        player_a,
        player_b,
        budget,
        initial_position,
        initial_sfen,
        max_plies,
        seed,
        git_commit,
        output_dir,
        resume,
        opening_book_path,
        opening_book_sha256,
        opening_book_size: None,
        opening_max_plies,
    })
}

fn validate_player(name: &str, player: &PlayerSpec) -> Result<(), String> {
    if player.depth == 0 || player.depth > 64 {
        return Err(format!("--{name}-depth must be 1..=64"));
    }
    if player.hash_megabytes == 0 || player.hash_megabytes > 1_024 {
        return Err(format!("--{name}-hash-mb must be 1..=1024"));
    }
    match (player.kind, player.model_path.is_some()) {
        (kind, false) if kind.uses_model() => {
            return Err(format!("model-backed player {name} requires --{name}-model"));
        }
        (kind, true) if kind.uses_model() => {}
        (_, false) => {}
        (_, true) => return Err(format!("--{name}-model requires a model-backed player {name}")),
    }
    Ok(())
}

fn resolve_models(player_a: &mut PlayerSpec, player_b: &mut PlayerSpec) -> Result<(), String> {
    resolve_model_identity("a", player_a)?;
    if player_a.model_path.is_some() && player_a.model_path == player_b.model_path {
        player_b.model_sha256.clone_from(&player_a.model_sha256);
        player_b.model_size = player_a.model_size;
        player_b.model.clone_from(&player_a.model);
        return Ok(());
    }
    resolve_model_identity("b", player_b)
}

fn resolve_model_identity(name: &str, player: &mut PlayerSpec) -> Result<(), String> {
    let Some(path) = player.model_path.as_deref() else {
        return Ok(());
    };
    let artifact = read_file_artifact(
        path,
        u64::try_from(MAX_NEURAL_MODEL_BYTES).unwrap_or(u64::MAX),
    )?;
    let model = if artifact.bytes.starts_with(b"OSAVAL02") {
        Osaval02Evaluator::from_bytes(&artifact.bytes)
            .map(|model| LoadedModel::Osaval02(Arc::new(model)))
            .map_err(|error| format!("cannot load --{name}-model {}: {error}", path.display()))?
    } else {
        NeuralEvaluator::from_bytes(&artifact.bytes)
            .map(|model| LoadedModel::Osaval01(Arc::new(model)))
            .map_err(|error| format!("cannot load --{name}-model {}: {error}", path.display()))?
    };
    player.model_sha256 = Some(artifact.sha256);
    player.model_size = Some(artifact.size);
    player.model = Some(model);
    Ok(())
}

fn load_opening_book(config: &mut ArenaConfig) -> Result<Option<OpeningBook>, String> {
    let Some(path) = config.opening_book_path.as_deref() else {
        return Ok(None);
    };
    let artifact = read_file_artifact(path, 64 * 1024 * 1024)?;
    let book = OpeningBook::from_compressed_bytes(&artifact.bytes)?;
    config.opening_book_sha256 = Some(artifact.sha256);
    config.opening_book_size = Some(artifact.size);
    Ok(Some(book))
}

fn parse_budget(nodes: Option<u64>, movetime_ms: Option<u64>) -> Result<Budget, String> {
    match (nodes, movetime_ms) {
        (Some(_), Some(_)) => Err("--nodes and --movetime-ms are mutually exclusive".to_owned()),
        (Some(0), _) | (_, Some(0)) => Err("search budget must be positive".to_owned()),
        (Some(value), None) if value > MAX_NODES_PER_MOVE => Err(format!(
            "--nodes must not exceed the defensive limit {MAX_NODES_PER_MOVE}"
        )),
        (None, Some(value)) if value > MAX_MOVETIME_MS => Err(format!(
            "--movetime-ms must not exceed the defensive limit {MAX_MOVETIME_MS}"
        )),
        (Some(value), None) => Ok(Budget::Nodes(value)),
        (None, Some(value)) => Ok(Budget::MoveTime(value)),
        (None, None) => Ok(Budget::Nodes(DEFAULT_NODES)),
    }
}

fn parse_initial_position(sfen: &str) -> Result<Position, String> {
    let position = parse_sfen(sfen).map_err(|error| format!("invalid --sfen: {error}"))?;
    if position.move_number() != 1 {
        return Err("--sfen move number must be 1 for canonical CSA output".to_owned());
    }
    Ok(position)
}

fn validate_git_commit(commit: Option<&str>) -> Result<(), String> {
    if let Some(commit) = commit
        && (!(7..=64).contains(&commit.len())
            || !commit
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase()))
    {
        return Err(
            "--git-commit must be a 7..=64 character lowercase hexadecimal object ID".to_owned(),
        );
    }
    Ok(())
}

fn prepare_output_directory(config: &ArenaConfig) -> Result<ArenaStorage, String> {
    let root = if config.resume {
        AnchoredDir::open_existing(&config.output_dir)
    } else {
        AnchoredDir::open_or_create_all(&config.output_dir)
    }
    .map_err(|error| format!("cannot open arena output directory without symlinks: {error}"))?;
    match root.try_lock_exclusive() {
        Ok(()) => {}
        Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
            return Err("another arena run is already using this output directory".to_owned());
        }
        Err(error) => return Err(format!("cannot lock arena output directory: {error}")),
    }
    if !config.resume {
        for entry in root
            .entries(MAX_ARENA_DIRECTORY_ENTRIES)
            .map_err(|error| format!("cannot inspect output directory: {error}"))?
        {
            if entry.name != OsStr::new(LOCK_FILE_NAME) {
                return Err(
                    "output directory is not empty; use --resume or another path".to_owned(),
                );
            }
        }
    }
    let games = match root
        .entry_kind(OsStr::new("games"))
        .map_err(|error| format!("cannot inspect arena games directory: {error}"))?
    {
        Some(EntryKind::Directory) => root.open_dir(OsStr::new("games")),
        None if config.resume => {
            return Err("arena games directory is missing during resume".to_owned());
        }
        None => root.create_dir(OsStr::new("games")),
        Some(_) => return Err("arena games entry is not a directory".to_owned()),
    }
    .map_err(|error| format!("cannot open arena games directory: {error}"))?;
    Ok(ArenaStorage { root, games })
}

#[expect(
    clippy::similar_names,
    clippy::too_many_lines,
    reason = "A/B evidence is intentionally symmetric and one game transaction stays together"
)]
fn play_game(
    config: &ArenaConfig,
    opening_book: Option<&OpeningBook>,
    id: u32,
) -> Result<PendingGame, String> {
    let (black_spec, white_spec) = if id.is_multiple_of(2) {
        (&config.player_a, &config.player_b)
    } else {
        (&config.player_b, &config.player_a)
    };
    let black = black_spec.label();
    let white = white_spec.label();
    let game_seed = mix_game_seed(config.seed, id);
    let mut black_player = Player::new(black_spec, game_seed ^ 0xB1AC_0000_0000_0001)?;
    let mut white_player = Player::new(white_spec, game_seed ^ 0xA117_E000_0000_0002)?;
    let mut game = Game::new(config.initial_position.clone());
    let mut black_totals = SearchTotals::default();
    let mut white_totals = SearchTotals::default();
    let mut outcome = None;
    let mut illegal_moves = 0_u64;
    for _ in 0..config.max_plies {
        if let Some(end) = game.end() {
            outcome = Some(outcome_from_end(end));
            break;
        }
        let side = game.position().side_to_move();
        let selected = match side {
            Side::Black => black_player.select(
                game.position(),
                &config.budget,
                opening_for_player(opening_book, black_spec, game.position(), config),
                OpeningPolicy {
                    profile: black_spec.opening_profile,
                    ..OpeningPolicy::default()
                },
                &mut black_totals,
            ),
            Side::White => white_player.select(
                game.position(),
                &config.budget,
                opening_for_player(opening_book, white_spec, game.position(), config),
                OpeningPolicy {
                    profile: white_spec.opening_profile,
                    ..OpeningPolicy::default()
                },
                &mut white_totals,
            ),
        };
        let Some(movement) = selected else {
            outcome = Some(GameOutcome {
                result: winner_result(side.opposite()),
                special: CsaSpecialMove::Resign,
                validation: CsaResultValidation::ExternalCondition,
            });
            break;
        };
        if game.play(movement).is_err() {
            illegal_moves += 1;
            outcome = Some(GameOutcome {
                result: winner_result(side.opposite()),
                special: match side {
                    Side::Black => CsaSpecialMove::BlackIllegalAction,
                    Side::White => CsaSpecialMove::WhiteIllegalAction,
                },
                validation: CsaResultValidation::ExternalCondition,
            });
            break;
        }
    }
    let outcome = final_outcome(outcome, game.end());
    let relative_csa = expected_csa_path(id);
    let (saved_csa, csa) = encode_csa(
        &config.initial_position,
        game.moves(),
        &black,
        &white,
        &outcome,
    )?;
    let winner_classification =
        classify_winner(id, black_spec.kind, white_spec.kind, outcome.result);
    let (player_a_totals, player_b_totals) = if id.is_multiple_of(2) {
        (black_totals, white_totals)
    } else {
        (white_totals, black_totals)
    };
    let totals = combine_search_totals(player_a_totals, player_b_totals);
    let summary = GameSummary {
        id,
        black,
        white,
        result: outcome.result,
        moves: u32::try_from(game.moves().len()).expect("bounded game move count fits u32"),
        csa_path: relative_csa,
        csa_sha256: saved_csa.sha256,
        csa_size: saved_csa.size,
        special_move: csa_special_code(&outcome.special)?.to_owned(),
        search_nodes: totals.nodes,
        search_elapsed_ns: totals.elapsed_ns,
        depth_sum: totals.depth_sum,
        searches: totals.searches,
        tt_probes: totals.tt_probes,
        tt_hits: totals.tt_hits,
        cutoffs: totals.cutoffs,
        candidate_moves: totals.candidate_moves,
        pruned_moves: totals.pruned_moves,
        neural_inference_calls: totals.neural_inference_calls,
        neural_inference_time_ns: totals.neural_inference_time_ns,
        player_a_search_nodes: player_a_totals.nodes,
        player_a_search_elapsed_ns: player_a_totals.elapsed_ns,
        player_a_depth_sum: player_a_totals.depth_sum,
        player_a_searches: player_a_totals.searches,
        player_a_neural_inference_calls: player_a_totals.neural_inference_calls,
        player_a_neural_inference_time_ns: player_a_totals.neural_inference_time_ns,
        player_b_search_nodes: player_b_totals.nodes,
        player_b_search_elapsed_ns: player_b_totals.elapsed_ns,
        player_b_depth_sum: player_b_totals.depth_sum,
        player_b_searches: player_b_totals.searches,
        player_b_neural_inference_calls: player_b_totals.neural_inference_calls,
        player_b_neural_inference_time_ns: player_b_totals.neural_inference_time_ns,
        illegal_moves,
        search_winner: winner_classification.0,
        player_a_winner: winner_classification.1,
        player_b_winner: winner_classification.2,
    };
    Ok(PendingGame { summary, csa })
}

fn combine_search_totals(left: SearchTotals, right: SearchTotals) -> SearchTotals {
    SearchTotals {
        nodes: left.nodes.saturating_add(right.nodes),
        elapsed_ns: left.elapsed_ns.saturating_add(right.elapsed_ns),
        depth_sum: left.depth_sum.saturating_add(right.depth_sum),
        searches: left.searches.saturating_add(right.searches),
        tt_probes: left.tt_probes.saturating_add(right.tt_probes),
        tt_hits: left.tt_hits.saturating_add(right.tt_hits),
        cutoffs: left.cutoffs.saturating_add(right.cutoffs),
        candidate_moves: left.candidate_moves.saturating_add(right.candidate_moves),
        pruned_moves: left.pruned_moves.saturating_add(right.pruned_moves),
        neural_inference_calls: left
            .neural_inference_calls
            .saturating_add(right.neural_inference_calls),
        neural_inference_time_ns: left
            .neural_inference_time_ns
            .saturating_add(right.neural_inference_time_ns),
    }
}

fn classify_winner(
    id: u32,
    black_kind: PlayerKind,
    white_kind: PlayerKind,
    result: ArenaResult,
) -> (bool, bool, bool) {
    let decisive = matches!(result, ArenaResult::BlackWin | ArenaResult::WhiteWin);
    let search_winner = match result {
        ArenaResult::BlackWin => black_kind.is_search(),
        ArenaResult::WhiteWin => white_kind.is_search(),
        ArenaResult::Draw | ArenaResult::MaxPlies => false,
    };
    let player_a_winner = match result {
        ArenaResult::BlackWin => id.is_multiple_of(2),
        ArenaResult::WhiteWin => !id.is_multiple_of(2),
        ArenaResult::Draw | ArenaResult::MaxPlies => false,
    };
    (search_winner, player_a_winner, decisive && !player_a_winner)
}

fn final_outcome(outcome: Option<GameOutcome>, game_end: Option<GameEnd>) -> GameOutcome {
    outcome
        .or_else(|| game_end.map(outcome_from_end))
        .unwrap_or(GameOutcome {
            result: ArenaResult::MaxPlies,
            special: CsaSpecialMove::MaxMoves,
            validation: CsaResultValidation::ExternalCondition,
        })
}

enum Player {
    Random(RandomMoveSelector),
    Search { engine: SearchEngine, depth: u8 },
}

impl Player {
    fn new(spec: &PlayerSpec, seed: u64) -> Result<Self, String> {
        match spec.kind {
            PlayerKind::Random => Ok(Self::Random(RandomMoveSelector::new(seed))),
            evaluator => {
                let mut config = SearchConfig {
                    transposition_entries: hash_entries(spec.hash_megabytes),
                    enable_transposition_table: spec.transposition,
                    ..SearchConfig::default()
                };
                config.evaluation = evaluation_config(evaluator);
                let engine = if let Some(mode) = evaluator.neural_mode() {
                    let model = spec
                        .model
                        .as_ref()
                        .ok_or_else(|| "model-backed player lacks a loaded model".to_owned())?;
                    match model {
                        LoadedModel::Osaval01(model) => {
                            SearchEngine::with_neural_mode(config, Arc::clone(model), mode)
                        }
                        LoadedModel::Osaval02(model) => {
                            if mode != NeuralEvaluationMode::PureValue {
                                return Err(
                                    "OSAVAL02 only supports pure-value semantics; score blending is forbidden"
                                        .to_owned(),
                                );
                            }
                            SearchEngine::with_osaval02(config, Arc::clone(model))
                        }
                    }
                } else {
                    SearchEngine::new(config)
                };
                Ok(Self::Search {
                    engine,
                    depth: spec.depth,
                })
            }
        }
    }

    fn select(
        &mut self,
        position: &Position,
        budget: &Budget,
        opening_book: Option<&OpeningBook>,
        opening_policy: OpeningPolicy,
        totals: &mut SearchTotals,
    ) -> Option<Move> {
        if let Some(choice) =
            opening_book.and_then(|book| book.select_with_policy(position, opening_policy))
        {
            return Some(choice.movement);
        }
        match self {
            Self::Random(random) => random.select(position),
            Self::Search { engine, depth } => {
                let limits = match budget {
                    Budget::Nodes(nodes) => SearchLimits {
                        max_depth: *depth,
                        max_nodes: Some(*nodes),
                        movetime: None,
                    },
                    Budget::MoveTime(milliseconds) => SearchLimits {
                        max_depth: *depth,
                        max_nodes: None,
                        movetime: Some(Duration::from_millis(*milliseconds)),
                    },
                };
                let result = engine.search(position, limits, &CancellationToken::new());
                totals.nodes = totals.nodes.saturating_add(result.nodes);
                totals.elapsed_ns = totals
                    .elapsed_ns
                    .saturating_add(u64::try_from(result.elapsed.as_nanos()).unwrap_or(u64::MAX));
                totals.depth_sum = totals.depth_sum.saturating_add(u64::from(result.depth));
                totals.searches = totals.searches.saturating_add(1);
                add_stats(totals, result.stats);
                if result.termination == SearchTermination::EvaluationError {
                    None
                } else {
                    result.best_move
                }
            }
        }
    }
}

fn opening_for_player<'a>(
    opening_book: Option<&'a OpeningBook>,
    spec: &PlayerSpec,
    position: &Position,
    config: &ArenaConfig,
) -> Option<&'a OpeningBook> {
    if spec.opening && position.move_number() <= config.opening_max_plies {
        opening_book
    } else {
        None
    }
}

const fn evaluation_config(kind: PlayerKind) -> EvaluationConfig {
    match kind {
        PlayerKind::Material => EvaluationConfig::material_only(),
        PlayerKind::HandcraftedBaseline => EvaluationConfig::handcrafted_baseline(),
        PlayerKind::HandcraftedExperimental
        | PlayerKind::Neural
        | PlayerKind::Residual
        | PlayerKind::Composite
        | PlayerKind::Random => {
            EvaluationConfig::handcrafted_experimental()
        }
    }
}

fn add_stats(totals: &mut SearchTotals, stats: SearchStats) {
    totals.tt_probes = totals.tt_probes.saturating_add(stats.tt_probes);
    totals.tt_hits = totals.tt_hits.saturating_add(stats.tt_hits);
    totals.cutoffs = totals.cutoffs.saturating_add(stats.beta_cutoffs);
    totals.candidate_moves = totals.candidate_moves.saturating_add(stats.candidate_moves);
    totals.pruned_moves = totals.pruned_moves.saturating_add(stats.pruned_moves);
    totals.neural_inference_calls = totals
        .neural_inference_calls
        .saturating_add(stats.neural_inference_calls);
    totals.neural_inference_time_ns = totals
        .neural_inference_time_ns
        .saturating_add(u64::try_from(stats.neural_inference_time.as_nanos()).unwrap_or(u64::MAX));
}

fn outcome_from_end(end: GameEnd) -> GameOutcome {
    match end {
        GameEnd::Checkmate { winner } => GameOutcome {
            result: winner_result(winner),
            special: CsaSpecialMove::Checkmate,
            validation: CsaResultValidation::Verified,
        },
        GameEnd::Repetition(RepetitionOutcome::NoContest) => GameOutcome {
            result: ArenaResult::Draw,
            special: CsaSpecialMove::Repetition,
            validation: CsaResultValidation::Verified,
        },
        GameEnd::Repetition(RepetitionOutcome::PerpetualCheckLoss(loser)) => GameOutcome {
            result: winner_result(loser.opposite()),
            special: CsaSpecialMove::PerpetualCheck,
            validation: CsaResultValidation::Verified,
        },
        GameEnd::Resignation { loser } => GameOutcome {
            result: winner_result(loser.opposite()),
            special: CsaSpecialMove::Resign,
            validation: CsaResultValidation::ExternalCondition,
        },
        GameEnd::Impasse(_) => GameOutcome {
            result: ArenaResult::Draw,
            special: CsaSpecialMove::EnteringKing,
            validation: CsaResultValidation::ExternalCondition,
        },
        GameEnd::EnteringKing(declaration) => {
            let winner = match declaration {
                open_shogi_core::EnteringKingDeclaration::Win { side, .. } => Some(side),
                open_shogi_core::EnteringKingDeclaration::InvalidLoss { side } => {
                    Some(side.opposite())
                }
                _ => None,
            };
            GameOutcome {
                result: winner.map_or(ArenaResult::Draw, winner_result),
                special: CsaSpecialMove::Win,
                validation: CsaResultValidation::ExternalCondition,
            }
        }
    }
}

const fn winner_result(side: Side) -> ArenaResult {
    match side {
        Side::Black => ArenaResult::BlackWin,
        Side::White => ArenaResult::WhiteWin,
    }
}

struct SavedArtifact {
    sha256: String,
    size: u64,
}

fn encode_csa(
    initial_position: &Position,
    moves: &[Move],
    black: &str,
    white: &str,
    outcome: &GameOutcome,
) -> Result<(SavedArtifact, String), String> {
    let record = CsaGame {
        version: "V3.0".to_owned(),
        black_name: Some(black.to_owned()),
        white_name: Some(white.to_owned()),
        metadata: Vec::new(),
        initial_position: initial_position.clone(),
        moves: moves.to_vec(),
        special_move: Some(outcome.special.clone()),
        result_validation: outcome.validation,
    };
    let contents = to_csa_game(&record).map_err(|error| format!("CSA write failed: {error}"))?;
    let sha256 = sha256_bytes(contents.as_bytes());
    let size = u64::try_from(contents.len()).unwrap_or(u64::MAX);
    Ok((SavedArtifact { sha256, size }, contents))
}

fn commit_pending_game(
    config: &ArenaConfig,
    opening_book: Option<&OpeningBook>,
    storage: &ArenaStorage,
    signature: &str,
    started_at: &str,
    summaries: &mut Vec<GameSummary>,
    pending: &PendingGame,
) -> Result<(), String> {
    commit_pending_game_with_hooks(
        config,
        opening_book,
        storage,
        signature,
        started_at,
        summaries,
        pending,
        |_| Ok(()),
        |_| Ok(()),
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "the game transaction keeps deterministic state, retained CSA, and failpoint hooks adjacent"
)]
fn commit_pending_game_with_hooks(
    config: &ArenaConfig,
    opening_book: Option<&OpeningBook>,
    storage: &ArenaStorage,
    signature: &str,
    started_at: &str,
    summaries: &mut Vec<GameSummary>,
    pending: &PendingGame,
    after_state: impl FnOnce(&ArenaStorage) -> Result<(), String>,
    after_cleanup: impl FnOnce(&ArenaStorage) -> Result<(), String>,
) -> Result<(), String> {
    if pending.summary.id != u32::try_from(summaries.len()).expect("game count fits u32") {
        return Err("arena pending game is not the next deterministic game".to_owned());
    }
    let journal = encode_game_journal(signature, &pending.summary, &pending.csa);
    storage
        .root
        .publish_new_atomic(OsStr::new(JOURNAL_FILE_NAME), journal.as_bytes())
        .map_err(|error| format!("cannot publish arena game journal: {error}"))?;
    let mut journal_file = storage
        .root
        .open_regular(OsStr::new(JOURNAL_FILE_NAME))
        .map_err(|error| format!("cannot retain arena game journal: {error}"))?;
    let journal_artifact = read_retained_file_artifact(
        &mut journal_file,
        &storage.root.display().join(JOURNAL_FILE_NAME),
        MAX_JOURNAL_BYTES,
    )?;
    if journal_artifact.bytes != journal.as_bytes() {
        return Err("arena game journal changed immediately after publication".to_owned());
    }
    let csa_name = csa_file_name(pending.summary.id);
    storage
        .games
        .publish_new_atomic(OsStr::new(&csa_name), pending.csa.as_bytes())
        .map_err(|error| format!("cannot publish arena CSA: {error}"))?;
    let (csa_file, csa_sha256, csa_size) = retain_and_validate_game_csa(
        config,
        opening_book,
        storage,
        &pending.summary,
        pending.csa.as_bytes(),
    )?;
    summaries.push(pending.summary.clone());
    write_state(&storage.root, signature, started_at, summaries)?;
    after_state(storage)?;
    remove_journal(&storage.root, &journal_file)?;
    after_cleanup(storage)?;
    validate_final_game_commit(
        storage,
        signature,
        started_at,
        summaries,
        &csa_name,
        &csa_file,
        &csa_sha256,
        csa_size,
    )
}

fn encode_game_journal(signature: &str, summary: &GameSummary, csa: &str) -> String {
    format!(
        "{JOURNAL_SCHEMA}\nsignature={}\nrow={}\ncsa-size={}\n\n{csa}",
        hex_encode(signature.as_bytes()),
        hex_encode(game_state_row(summary).as_bytes()),
        csa.len()
    )
}

fn recover_game_journal(
    config: &ArenaConfig,
    opening_book: Option<&OpeningBook>,
    storage: &ArenaStorage,
    signature: &str,
    started_at: &str,
    summaries: &mut Vec<GameSummary>,
) -> Result<(), String> {
    recover_game_journal_with_hooks(
        config,
        opening_book,
        storage,
        signature,
        started_at,
        summaries,
        |_| Ok(()),
        |_| Ok(()),
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "journal recovery keeps deterministic state, retained CSA, and failpoint hooks adjacent"
)]
fn recover_game_journal_with_hooks(
    config: &ArenaConfig,
    opening_book: Option<&OpeningBook>,
    storage: &ArenaStorage,
    signature: &str,
    started_at: &str,
    summaries: &mut Vec<GameSummary>,
    after_state: impl FnOnce(&ArenaStorage) -> Result<(), String>,
    after_cleanup: impl FnOnce(&ArenaStorage) -> Result<(), String>,
) -> Result<(), String> {
    let Some(kind) = storage
        .root
        .entry_kind(OsStr::new(JOURNAL_FILE_NAME))
        .map_err(|error| format!("cannot inspect arena game journal: {error}"))?
    else {
        return Ok(());
    };
    if kind != EntryKind::File {
        return Err("arena game journal is not a regular file".to_owned());
    }
    let mut journal_file = storage
        .root
        .open_regular(OsStr::new(JOURNAL_FILE_NAME))
        .map_err(|error| format!("cannot open arena game journal: {error}"))?;
    let artifact = read_retained_file_artifact(
        &mut journal_file,
        &config.output_dir.join(JOURNAL_FILE_NAME),
        MAX_JOURNAL_BYTES,
    )
    .map_err(|error| format!("cannot read arena game journal: {error}"))?;
    let journal = String::from_utf8(artifact.bytes)
        .map_err(|_| "arena game journal is not valid UTF-8".to_owned())?;
    let (summary, csa) = parse_game_journal(&journal, signature)?;
    if summary.csa_path != expected_csa_path(summary.id) {
        return Err("arena game journal CSA path is not deterministic".to_owned());
    }
    let expected_id = u32::try_from(summaries.len()).expect("game count fits u32");
    let state_already_committed = if summary.id == expected_id {
        false
    } else if summary.id.checked_add(1) == Some(expected_id) && summaries.last() == Some(&summary) {
        true
    } else {
        return Err("arena game journal does not match the next state transaction".to_owned());
    };

    let csa_name = csa_file_name(summary.id);
    if let Some(kind) = storage
        .games
        .entry_kind(OsStr::new(&csa_name))
        .map_err(|error| format!("cannot inspect arena journal CSA: {error}"))?
    {
        if kind != EntryKind::File {
            return Err("arena journal CSA is not a regular file".to_owned());
        }
    } else {
        storage
            .games
            .publish_new_atomic(OsStr::new(&csa_name), csa.as_bytes())
            .map_err(|error| format!("cannot recover arena journal CSA: {error}"))?;
    }
    let (csa_file, csa_sha256, csa_size) = retain_and_validate_game_csa(
        config,
        opening_book,
        storage,
        &summary,
        csa.as_bytes(),
    )?;
    if !state_already_committed {
        summaries.push(summary);
        write_state(&storage.root, signature, started_at, summaries)?;
    }
    after_state(storage)?;
    remove_journal(&storage.root, &journal_file)?;
    after_cleanup(storage)?;
    validate_final_game_commit(
        storage,
        signature,
        started_at,
        summaries,
        &csa_name,
        &csa_file,
        &csa_sha256,
        csa_size,
    )
}

fn retain_and_validate_game_csa(
    config: &ArenaConfig,
    opening_book: Option<&OpeningBook>,
    storage: &ArenaStorage,
    summary: &GameSummary,
    expected_bytes: &[u8],
) -> Result<(AnchoredFile, [u8; 32], u64), String> {
    let csa_name = csa_file_name(summary.id);
    let mut file = storage
        .games
        .open_regular(OsStr::new(&csa_name))
        .map_err(|error| format!("cannot retain arena CSA: {error}"))?;
    let artifact = read_retained_file_artifact(
        &mut file,
        &config.output_dir.join(&summary.csa_path),
        MAX_CSA_BYTES,
    )?;
    if artifact.bytes != expected_bytes
        || artifact.sha256 != summary.csa_sha256
        || artifact.size != summary.csa_size
    {
        return Err("arena journal CSA conflicts with the deterministic game record".to_owned());
    }
    validate_resumed_game(config, opening_book, storage, summary)?;
    let (sha256, size) = file
        .retained_sha256(MAX_CSA_BYTES)
        .map_err(|error| format!("cannot bind retained arena CSA: {error}"))?;
    if hex_encode(&sha256) != summary.csa_sha256 || size != summary.csa_size {
        return Err("retained arena CSA identity differs from its state row".to_owned());
    }
    Ok((file, sha256, size))
}

#[expect(
    clippy::too_many_arguments,
    reason = "the final transaction check binds retained CSA identity, state identity, and run identity"
)]
fn validate_final_game_commit(
    storage: &ArenaStorage,
    signature: &str,
    started_at: &str,
    summaries: &[GameSummary],
    csa_name: &str,
    csa_file: &AnchoredFile,
    csa_sha256: &[u8; 32],
    csa_size: u64,
) -> Result<(), String> {
    storage
        .games
        .sync()
        .map_err(|error| format!("cannot sync committed arena CSA directory: {error}"))?;
    storage
        .games
        .verify_retained_link_content(OsStr::new(csa_name), csa_file, csa_sha256, csa_size)
        .map_err(|error| format!("committed arena CSA changed during finalization: {error}"))?;
    let (observed_started_at, observed_summaries) = read_state(&storage.root, signature)?;
    if observed_started_at != started_at || observed_summaries != summaries {
        return Err("committed arena state changed during game finalization".to_owned());
    }
    Ok(())
}

fn parse_game_journal<'a>(
    journal: &'a str,
    signature: &str,
) -> Result<(GameSummary, &'a str), String> {
    let (header, csa) = journal
        .split_once("\n\n")
        .ok_or_else(|| "arena game journal lacks its payload delimiter".to_owned())?;
    let mut lines = header.lines();
    if lines.next() != Some(JOURNAL_SCHEMA) {
        return Err("arena game journal schema mismatch".to_owned());
    }
    let encoded_signature = lines
        .next()
        .and_then(|line| line.strip_prefix("signature="))
        .ok_or_else(|| "arena game journal lacks its signature".to_owned())?;
    if hex_decode(encoded_signature)? != signature.as_bytes() {
        return Err("arena game journal configuration mismatch".to_owned());
    }
    let row = lines
        .next()
        .and_then(|line| line.strip_prefix("row="))
        .ok_or_else(|| "arena game journal lacks its state row".to_owned())?;
    let row = String::from_utf8(hex_decode(row)?)
        .map_err(|_| "arena game journal state row is not UTF-8".to_owned())?;
    let encoded_size = lines
        .next()
        .and_then(|line| line.strip_prefix("csa-size="))
        .ok_or_else(|| "arena game journal lacks its CSA size".to_owned())?;
    if lines.next().is_some() {
        return Err("arena game journal has unexpected header fields".to_owned());
    }
    let size: u64 = parse_state(encoded_size, "journal CSA size")?;
    if size > MAX_CSA_BYTES || size != u64::try_from(csa.len()).unwrap_or(u64::MAX) {
        return Err("arena game journal CSA size mismatch".to_owned());
    }
    let id = row
        .split('\t')
        .nth(1)
        .ok_or_else(|| "arena game journal has a malformed state row".to_owned())?;
    let id = parse_state(id, "journal game id")?;
    let summary = parse_game_state_row(&row, id)?;
    if summary.csa_size != size || summary.csa_sha256 != sha256_bytes(csa.as_bytes()) {
        return Err("arena game journal CSA identity mismatch".to_owned());
    }
    Ok((summary, csa))
}

fn remove_journal(root: &AnchoredDir, journal: &AnchoredFile) -> Result<(), String> {
    root.remove_anchored_file(journal)
        .map_err(|error| format!("cannot remove committed arena game journal: {error}"))?;
    Ok(())
}

fn recover_owned_temporary_files(storage: &ArenaStorage) -> Result<(), String> {
    let mut remaining = MAX_PRESERVED_UNPROVEN_TEMPORARIES;
    reserve_cleanup_quarantine_budget(&storage.root, &mut remaining)?;
    reserve_cleanup_quarantine_budget(&storage.games, &mut remaining)?;
    remove_owned_temporary_files(
        &storage.root,
        &["arena.state", "arena-report.json", JOURNAL_FILE_NAME],
        &mut remaining,
    )?;
    remove_owned_game_temporary_files(&storage.games, &mut remaining)
}

fn reserve_cleanup_quarantine_budget(
    directory: &AnchoredDir,
    remaining: &mut usize,
) -> Result<(), String> {
    let preserved = directory
        .cleanup_quarantine_entry_count(*remaining)
        .map_err(|error| {
            format!(
                "cannot validate bounded cleanup quarantine in {}: {error}; preserve every entry and remove remnants manually after inspection",
                directory.display().display()
            )
        })?;
    *remaining = remaining.saturating_sub(preserved);
    Ok(())
}

fn consume_unproven_temporary_budget(
    directory: &AnchoredDir,
    remaining: &mut usize,
    kind: &str,
) -> Result<(), String> {
    let Some(next) = remaining.checked_sub(1) else {
        return Err(format!(
            "arena has more than {MAX_PRESERVED_UNPROVEN_TEMPORARIES} total unproven {kind} and private cleanup remnants in {}; preserve them and remove them manually after inspection",
            directory.display().display()
        ));
    };
    *remaining = next;
    Ok(())
}

fn remove_owned_temporary_files(
    directory: &AnchoredDir,
    targets: &[&str],
    remaining: &mut usize,
) -> Result<(), String> {
    let mut preserved = 0_usize;
    for entry in directory
        .entries(MAX_ARENA_DIRECTORY_ENTRIES)
        .map_err(|error| format!("cannot inspect arena temporary files: {error}"))?
    {
        let Some(name) = entry.name.to_str().map(str::to_owned) else {
            continue;
        };
        if let Some(target) = owned_temporary_target(&name, targets)
            && (entry.kind != EntryKind::File
                || !recover_proven_temporary(directory, &entry.name, OsStr::new(target))?)
        {
            preserved += 1;
            consume_unproven_temporary_budget(directory, remaining, "crash temporaries")?;
        }
    }
    if preserved != 0 {
        eprintln!(
            "arena warning: preserved {preserved} unproven crash temporary entr{} in {}; they are never consumed or deleted",
            if preserved == 1 { "y" } else { "ies" },
            directory.display().display()
        );
    }
    directory
        .sync()
        .map_err(|error| format!("cannot sync arena temporary directory: {error}"))
}

fn remove_owned_game_temporary_files(
    directory: &AnchoredDir,
    remaining: &mut usize,
) -> Result<(), String> {
    let mut preserved = 0_usize;
    for entry in directory
        .entries(MAX_ARENA_DIRECTORY_ENTRIES)
        .map_err(|error| format!("cannot inspect arena game temporary files: {error}"))?
    {
        let Some(name) = entry.name.to_str().map(str::to_owned) else {
            continue;
        };
        if let Some(target) = owned_game_temporary_target(&name)
            && (entry.kind != EntryKind::File
                || !recover_proven_temporary(directory, &entry.name, OsStr::new(target))?)
        {
            preserved += 1;
            consume_unproven_temporary_budget(directory, remaining, "game crash temporaries")?;
        }
    }
    if preserved != 0 {
        eprintln!(
            "arena warning: preserved {preserved} unproven game crash temporary entr{} in {}; they are never consumed or deleted",
            if preserved == 1 { "y" } else { "ies" },
            directory.display().display()
        );
    }
    directory
        .sync()
        .map_err(|error| format!("cannot sync arena game directory: {error}"))
}

fn recover_proven_temporary(
    directory: &AnchoredDir,
    temporary_name: &OsStr,
    target_name: &OsStr,
) -> Result<bool, String> {
    let temporary = directory
        .open_regular(temporary_name)
        .map_err(|error| format!("cannot retain arena temporary file: {error}"))?;
    let target = match directory.open_regular(target_name) {
        Ok(target) => target,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(format!("cannot inspect arena temporary target: {error}")),
    };
    if !temporary
        .is_same_file_as(&target)
        .map_err(|error| format!("cannot compare arena temporary identity: {error}"))?
    {
        return Ok(false);
    }
    directory
        .remove_anchored_file(&temporary)
        .map_err(|error| format!("cannot remove proven arena temporary file: {error}"))?;
    Ok(true)
}

fn is_owned_temporary_name(name: &str, target: &str) -> bool {
    let Some(suffix) = name.strip_prefix(&format!(".{target}.tmp-")) else {
        return false;
    };
    let Some((process, sequence)) = suffix.split_once('-') else {
        return false;
    };
    !process.is_empty()
        && !sequence.is_empty()
        && process.bytes().all(|byte| byte.is_ascii_digit())
        && sequence.bytes().all(|byte| byte.is_ascii_digit())
}

fn owned_temporary_target<'a>(name: &str, targets: &'a [&str]) -> Option<&'a str> {
    targets
        .iter()
        .copied()
        .find(|target| is_owned_temporary_name(name, target))
}

fn owned_game_temporary_target(name: &str) -> Option<&str> {
    let target = name.strip_prefix('.')?.split_once(".tmp-")?.0;
    let stem = target.strip_suffix(".csa")?;
    (stem.starts_with("game-")
        && !stem[5..].is_empty()
        && stem[5..].bytes().all(|byte| byte.is_ascii_digit())
        && is_owned_temporary_name(name, target))
    .then_some(target)
}

fn config_signature(config: &ArenaConfig) -> String {
    String::from_utf8(arena_config_signature_bytes(&arena_signature(config)))
        .expect("canonical arena signature contains valid UTF-8")
}

fn arena_signature(config: &ArenaConfig) -> ArenaConfigSignature {
    let (budget_kind, budget_value) = match config.budget {
        Budget::Nodes(value) => ("nodes", value),
        Budget::MoveTime(value) => ("movetime_ms", value),
    };
    ArenaConfigSignature {
        games: u64::from(config.games),
        seed: config.seed,
        initial_sfen: config.initial_sfen.clone(),
        max_plies: u64::from(config.max_plies),
        git_commit: config.git_commit.clone(),
        budget_kind: budget_kind.to_owned(),
        budget_value,
        player_a: arena_signature_player(&config.player_a),
        player_b: arena_signature_player(&config.player_b),
        opening: ArenaConfigSignatureOpening {
            enabled: config.opening_book_sha256.is_some(),
            artifact_sha256: config.opening_book_sha256.clone(),
            artifact_size: config.opening_book_size,
            max_plies: config
                .opening_book_sha256
                .as_ref()
                .map(|_| u64::from(config.opening_max_plies)),
        },
    }
}

fn arena_signature_player(player: &PlayerSpec) -> ArenaConfigSignaturePlayer {
    ArenaConfigSignaturePlayer {
        label: player.label(),
        evaluator_kind: player.kind.report_name().to_owned(),
        search_depth: player.kind.is_search().then_some(u64::from(player.depth)),
        hash_megabytes: player
            .kind
            .is_search()
            .then(|| u64::try_from(player.hash_megabytes).unwrap_or(u64::MAX)),
        transposition: player.kind.is_search().then_some(player.transposition),
        model_artifact_sha256: player.model_sha256.clone(),
        model_artifact_size: player.model_size,
        model_payload_sha256: player.model_payload_sha256(),
        architecture_version: player.model_architecture_version().map(u64::from),
        quantization: player
            .model_quantization()
            .map(str::to_owned),
        opening_enabled: player.opening,
    }
}

/// Builds the exact language-neutral byte string hashed into run.configSha256.
///
/// The wire form is compact UTF-8 JSON with the fixed field order below, JSON strings,
/// unsigned base-10 integers, lowercase Booleans, and literal `null` optionals.
pub(crate) fn arena_config_signature_bytes(signature: &ArenaConfigSignature) -> Vec<u8> {
    let mut output = String::with_capacity(1_024);
    write!(
        output,
        "{{\"schema\":\"phase2_arena_config_signature/v1\",\"games\":{},\"seed\":{},\"initialSfen\":{},\"maxPlies\":{},\"gitCommit\":",
        signature.games,
        signature.seed,
        json_string(&signature.initial_sfen),
        signature.max_plies,
    )
    .expect("writing to String cannot fail");
    write_optional_json_string(&mut output, signature.git_commit.as_deref());
    write!(
        output,
        ",\"budget\":{{\"kind\":{},\"value\":{}}},\"playerA\":",
        json_string(&signature.budget_kind),
        signature.budget_value,
    )
    .expect("writing to String cannot fail");
    write_signature_player(&mut output, &signature.player_a);
    output.push_str(",\"playerB\":");
    write_signature_player(&mut output, &signature.player_b);
    output.push_str(",\"opening\":");
    write_signature_opening(&mut output, &signature.opening);
    output.push('}');
    output.into_bytes()
}

fn write_signature_player(output: &mut String, player: &ArenaConfigSignaturePlayer) {
    write!(
        output,
        "{{\"label\":{},\"evaluatorKind\":{},\"searchDepth\":",
        json_string(&player.label),
        json_string(&player.evaluator_kind),
    )
    .expect("writing to String cannot fail");
    write_optional_u64(output, player.search_depth);
    output.push_str(",\"hashMegabytes\":");
    write_optional_u64(output, player.hash_megabytes);
    output.push_str(",\"transposition\":");
    match player.transposition {
        Some(value) => output.push_str(if value { "true" } else { "false" }),
        None => output.push_str("null"),
    }
    output.push_str(",\"modelArtifactSha256\":");
    write_optional_json_string(output, player.model_artifact_sha256.as_deref());
    output.push_str(",\"modelArtifactSize\":");
    write_optional_u64(output, player.model_artifact_size);
    output.push_str(",\"modelPayloadSha256\":");
    write_optional_json_string(output, player.model_payload_sha256.as_deref());
    output.push_str(",\"architectureVersion\":");
    write_optional_u64(output, player.architecture_version);
    output.push_str(",\"quantization\":");
    write_optional_json_string(output, player.quantization.as_deref());
    write!(
        output,
        ",\"openingEnabled\":{}}}",
        if player.opening_enabled {
            "true"
        } else {
            "false"
        }
    )
    .expect("writing to String cannot fail");
}

fn write_signature_opening(output: &mut String, opening: &ArenaConfigSignatureOpening) {
    write!(
        output,
        "{{\"enabled\":{},\"artifactSha256\":",
        if opening.enabled { "true" } else { "false" }
    )
    .expect("writing to String cannot fail");
    write_optional_json_string(output, opening.artifact_sha256.as_deref());
    output.push_str(",\"artifactSize\":");
    write_optional_u64(output, opening.artifact_size);
    output.push_str(",\"maxPlies\":");
    write_optional_u64(output, opening.max_plies);
    output.push('}');
}

fn write_state(
    root: &AnchoredDir,
    signature: &str,
    started_at: &str,
    games: &[GameSummary],
) -> Result<(), String> {
    let mut output = format!(
        "{STATE_SCHEMA}\nsignature={}\nstarted={}\n",
        hex_encode(signature.as_bytes()),
        hex_encode(started_at.as_bytes())
    );
    for game in games {
        writeln!(output, "{}", game_state_row(game)).expect("writing to String cannot fail");
    }
    if output.len() > MAX_STATE_BYTES {
        return Err("arena state exceeds the 5 MiB resume-state limit".to_owned());
    }
    root.replace_atomic(OsStr::new("arena.state"), output.as_bytes())
        .map_err(|error| format!("cannot write arena state: {error}"))
}

fn game_state_row(game: &GameSummary) -> String {
    [
        "game".to_owned(),
        game.id.to_string(),
        game.black.clone(),
        game.white.clone(),
        game.result.json().to_owned(),
        game.moves.to_string(),
        game.csa_path.clone(),
        game.csa_sha256.clone(),
        game.csa_size.to_string(),
        game.special_move.clone(),
        game.search_nodes.to_string(),
        game.search_elapsed_ns.to_string(),
        game.depth_sum.to_string(),
        game.searches.to_string(),
        game.tt_probes.to_string(),
        game.tt_hits.to_string(),
        game.cutoffs.to_string(),
        game.candidate_moves.to_string(),
        game.pruned_moves.to_string(),
        game.neural_inference_calls.to_string(),
        game.neural_inference_time_ns.to_string(),
        game.illegal_moves.to_string(),
        u8::from(game.search_winner).to_string(),
        u8::from(game.player_a_winner).to_string(),
        u8::from(game.player_b_winner).to_string(),
        game.player_a_search_nodes.to_string(),
        game.player_a_search_elapsed_ns.to_string(),
        game.player_a_depth_sum.to_string(),
        game.player_a_searches.to_string(),
        game.player_a_neural_inference_calls.to_string(),
        game.player_a_neural_inference_time_ns.to_string(),
        game.player_b_search_nodes.to_string(),
        game.player_b_search_elapsed_ns.to_string(),
        game.player_b_depth_sum.to_string(),
        game.player_b_searches.to_string(),
        game.player_b_neural_inference_calls.to_string(),
        game.player_b_neural_inference_time_ns.to_string(),
    ]
    .join("\t")
}

fn read_state(root: &AnchoredDir, signature: &str) -> Result<(String, Vec<GameSummary>), String> {
    let file = root
        .open_regular(OsStr::new("arena.state"))
        .map_err(|error| format!("cannot open arena resume state: {error}"))?;
    let artifact = read_open_file_artifact(
        file,
        &root.display().join("arena.state"),
        u64::try_from(MAX_STATE_BYTES).expect("state byte limit fits u64"),
    )
    .map_err(|error| {
        let size_error = format!("exceeds the {MAX_STATE_BYTES}-byte read limit");
        if error.ends_with(&size_error) {
            "arena state exceeds 5 MiB limit".to_owned()
        } else {
            format!("cannot read resume state: {error}")
        }
    })?;
    let contents = String::from_utf8(artifact.bytes)
        .map_err(|_| "arena state is not valid UTF-8".to_owned())?;
    if !contents.ends_with('\n') || contents.contains('\r') {
        return Err("arena state must use canonical LF-terminated lines".to_owned());
    }
    let mut lines = contents.lines();
    if lines.next() != Some(STATE_SCHEMA) {
        return Err("arena state schema mismatch".to_owned());
    }
    let encoded_signature = lines
        .next()
        .and_then(|line| line.strip_prefix("signature="))
        .ok_or_else(|| "arena state lacks signature".to_owned())?;
    if hex_decode(encoded_signature)? != signature.as_bytes() {
        return Err("arena resume configuration mismatch".to_owned());
    }
    let started = lines
        .next()
        .and_then(|line| line.strip_prefix("started="))
        .ok_or_else(|| "arena state lacks start time".to_owned())?;
    let started =
        String::from_utf8(hex_decode(started)?).map_err(|_| "invalid state time".to_owned())?;
    if !is_utc_timestamp(&started) {
        return Err("arena state start time is not ISO-8601 UTC".to_owned());
    }
    let mut games = Vec::new();
    for line in lines {
        let expected_id = u32::try_from(games.len()).expect("state length fits u32");
        games.push(parse_game_state_row(line, expected_id)?);
        if games.len() > usize::try_from(MAX_GAMES).expect("game cap fits usize") {
            return Err("arena state exceeds game limit".to_owned());
        }
    }
    Ok((started, games))
}

fn parse_game_state_row(line: &str, expected_id: u32) -> Result<GameSummary, String> {
    let fields = line.split('\t').collect::<Vec<_>>();
    if fields.len() != 37 || fields[0] != "game" {
        return Err("malformed arena resume game row".to_owned());
    }
    let id = parse_state(fields[1], "game id")?;
    if id != expected_id {
        return Err("arena state game IDs are not contiguous".to_owned());
    }
    Ok(GameSummary {
        id,
        black: validate_state_text(fields[2])?,
        white: validate_state_text(fields[3])?,
        result: ArenaResult::parse(fields[4])?,
        moves: parse_state(fields[5], "move count")?,
        csa_path: validate_state_text(fields[6])?,
        csa_sha256: validate_sha256(fields[7], "CSA SHA-256")?,
        csa_size: parse_state(fields[8], "CSA size")?,
        special_move: validate_state_text(fields[9])?,
        search_nodes: parse_state(fields[10], "nodes")?,
        search_elapsed_ns: parse_state(fields[11], "elapsed ns")?,
        depth_sum: parse_state(fields[12], "depth")?,
        searches: parse_state(fields[13], "searches")?,
        tt_probes: parse_state(fields[14], "probes")?,
        tt_hits: parse_state(fields[15], "hits")?,
        cutoffs: parse_state(fields[16], "cutoffs")?,
        candidate_moves: parse_state(fields[17], "candidate moves")?,
        pruned_moves: parse_state(fields[18], "pruned moves")?,
        neural_inference_calls: parse_state(fields[19], "neural inference calls")?,
        neural_inference_time_ns: parse_state(fields[20], "neural inference time")?,
        illegal_moves: parse_state(fields[21], "illegal moves")?,
        search_winner: parse_state_bool(fields[22])?,
        player_a_winner: parse_state_bool(fields[23])?,
        player_b_winner: parse_state_bool(fields[24])?,
        player_a_search_nodes: parse_state(fields[25], "player A nodes")?,
        player_a_search_elapsed_ns: parse_state(fields[26], "player A elapsed ns")?,
        player_a_depth_sum: parse_state(fields[27], "player A depth")?,
        player_a_searches: parse_state(fields[28], "player A searches")?,
        player_a_neural_inference_calls: parse_state(
            fields[29],
            "player A neural inference calls",
        )?,
        player_a_neural_inference_time_ns: parse_state(
            fields[30],
            "player A neural inference time",
        )?,
        player_b_search_nodes: parse_state(fields[31], "player B nodes")?,
        player_b_search_elapsed_ns: parse_state(fields[32], "player B elapsed ns")?,
        player_b_depth_sum: parse_state(fields[33], "player B depth")?,
        player_b_searches: parse_state(fields[34], "player B searches")?,
        player_b_neural_inference_calls: parse_state(
            fields[35],
            "player B neural inference calls",
        )?,
        player_b_neural_inference_time_ns: parse_state(
            fields[36],
            "player B neural inference time",
        )?,
    })
}

fn validate_resume_artifacts(
    config: &ArenaConfig,
    opening_book: Option<&OpeningBook>,
    storage: &ArenaStorage,
    games: &[GameSummary],
) -> Result<(), String> {
    validate_resume_layout(config, storage, games)?;
    for summary in games {
        validate_resumed_game(config, opening_book, storage, summary)?;
    }
    Ok(())
}

fn validate_resume_layout(
    config: &ArenaConfig,
    storage: &ArenaStorage,
    games: &[GameSummary],
) -> Result<(), String> {
    if !config.resume {
        return Ok(());
    }
    for entry in storage
        .root
        .entries(MAX_ARENA_DIRECTORY_ENTRIES)
        .map_err(|error| format!("cannot inspect arena output directory: {error}"))?
    {
        let name = entry.name;
        if name.to_str().is_some_and(|name| {
            owned_temporary_target(
                name,
                &["arena.state", "arena-report.json", JOURNAL_FILE_NAME],
            )
            .is_some()
        }) {
            // Recovery has already bounded these unproven remnants. They are preserved but
            // never opened, consumed, or treated as arena state.
            continue;
        }
        if name == OsStr::new(SECURE_CLEANUP_DIRECTORY_NAME) && entry.kind == EntryKind::Directory {
            continue;
        }
        if !matches!(
            name.to_str(),
            Some(LOCK_FILE_NAME | "arena.state" | "arena-report.json" | "games")
        ) {
            return Err(format!(
                "arena resume output contains an unexpected entry: {}",
                storage.root.display().join(&name).display()
            ));
        }
        if entry.kind == EntryKind::Symlink {
            return Err("arena resume output contains a symlink".to_owned());
        }
        match name.to_str() {
            Some("games") if entry.kind != EntryKind::Directory => {
                return Err("arena resume games entry is not a directory".to_owned());
            }
            Some(LOCK_FILE_NAME | "arena.state" | "arena-report.json")
                if entry.kind != EntryKind::File =>
            {
                return Err("arena resume mutable entry is not a regular file".to_owned());
            }
            _ => {}
        }
    }
    let expected = games
        .iter()
        .map(|game| game.csa_path.as_str())
        .collect::<std::collections::BTreeSet<_>>();
    for entry in storage
        .games
        .entries(MAX_ARENA_DIRECTORY_ENTRIES)
        .map_err(|error| format!("cannot inspect arena games directory: {error}"))?
    {
        let relative = format!("games/{}", entry.name.to_string_lossy());
        if !expected.contains(relative.as_str()) {
            if entry
                .name
                .to_str()
                .and_then(owned_game_temporary_target)
                .is_some()
            {
                // As above, recovery bounded this preserved name without consuming its entry.
                continue;
            }
            if entry.name == OsStr::new(SECURE_CLEANUP_DIRECTORY_NAME)
                && entry.kind == EntryKind::Directory
            {
                continue;
            }
            return Err(format!(
                "arena resume found an uncommitted or conflicting game record: {relative}"
            ));
        }
        if entry.kind != EntryKind::File {
            return Err("arena resume game record is not a regular non-symlink file".to_owned());
        }
    }
    Ok(())
}

#[expect(
    clippy::too_many_lines,
    reason = "resume validation keeps the complete game evidence contract together"
)]
fn validate_resumed_game(
    config: &ArenaConfig,
    opening_book: Option<&OpeningBook>,
    storage: &ArenaStorage,
    summary: &GameSummary,
) -> Result<(), String> {
    let expected_path = expected_csa_path(summary.id);
    if summary.csa_path != expected_path {
        return Err("arena resume CSA path does not match its deterministic game ID".to_owned());
    }
    let (expected_black, expected_white) = assigned_player_labels(config, summary.id);
    if summary.black != expected_black || summary.white != expected_white {
        return Err(
            "arena resume player assignment does not match the deterministic schedule".to_owned(),
        );
    }
    let json_bounded_counters = [
        summary.csa_size,
        summary.search_nodes,
        summary.depth_sum,
        summary.searches,
        summary.tt_probes,
        summary.tt_hits,
        summary.cutoffs,
        summary.candidate_moves,
        summary.pruned_moves,
        summary.neural_inference_calls,
        summary.neural_inference_time_ns,
        summary.player_a_search_nodes,
        summary.player_a_depth_sum,
        summary.player_a_searches,
        summary.player_a_neural_inference_calls,
        summary.player_a_neural_inference_time_ns,
        summary.player_b_search_nodes,
        summary.player_b_depth_sum,
        summary.player_b_searches,
        summary.player_b_neural_inference_calls,
        summary.player_b_neural_inference_time_ns,
        summary.illegal_moves,
    ];
    let random_metrics_invalid = [
        (
            config.player_a.kind,
            [
                summary.player_a_search_nodes,
                summary.player_a_search_elapsed_ns,
                summary.player_a_depth_sum,
                summary.player_a_searches,
                summary.player_a_neural_inference_calls,
                summary.player_a_neural_inference_time_ns,
            ],
        ),
        (
            config.player_b.kind,
            [
                summary.player_b_search_nodes,
                summary.player_b_search_elapsed_ns,
                summary.player_b_depth_sum,
                summary.player_b_searches,
                summary.player_b_neural_inference_calls,
                summary.player_b_neural_inference_time_ns,
            ],
        ),
    ]
    .into_iter()
    .any(|(kind, metrics)| {
        kind == PlayerKind::Random && metrics.into_iter().any(|value| value != 0)
    });
    let fixed_node_metrics_invalid = match config.budget {
        Budget::Nodes(limit) => {
            summary.player_a_search_nodes > limit.saturating_mul(summary.player_a_searches)
                || summary.player_b_search_nodes > limit.saturating_mul(summary.player_b_searches)
        }
        Budget::MoveTime(_) => false,
    };
    if json_bounded_counters
        .iter()
        .any(|value| *value > MAX_JSON_SAFE_INTEGER)
        || summary.moves > config.max_plies
        || (summary.result == ArenaResult::MaxPlies && summary.moves != config.max_plies)
        || summary.csa_size == 0
        || summary.tt_hits > summary.tt_probes
        || summary.tt_probes > summary.search_nodes
        || summary.cutoffs > summary.search_nodes
        || summary.pruned_moves > summary.candidate_moves
        || summary.searches > u64::from(summary.moves).saturating_add(1)
        || summary.player_a_searches > u64::from(summary.moves).saturating_add(1)
        || summary.player_b_searches > u64::from(summary.moves).saturating_add(1)
        || summary.player_a_depth_sum
            > u64::from(config.player_a.depth).saturating_mul(summary.player_a_searches)
        || summary.player_b_depth_sum
            > u64::from(config.player_b.depth).saturating_mul(summary.player_b_searches)
        || fixed_node_metrics_invalid
        || (config.player_a.kind.is_search()
            && summary.player_a_search_nodes < summary.player_a_searches)
        || (config.player_b.kind.is_search()
            && summary.player_b_search_nodes < summary.player_b_searches)
        || random_metrics_invalid
        || summary.illegal_moves > 1
        || (summary.neural_inference_calls == 0 && summary.neural_inference_time_ns != 0)
        || (summary.player_a_neural_inference_calls == 0
            && summary.player_a_neural_inference_time_ns != 0)
        || (summary.player_b_neural_inference_calls == 0
            && summary.player_b_neural_inference_time_ns != 0)
        || summary.player_a_neural_inference_calls
            > summary
                .player_a_search_nodes
                .saturating_add(summary.player_a_searches)
        || summary.player_b_neural_inference_calls
            > summary
                .player_b_search_nodes
                .saturating_add(summary.player_b_searches)
        || (summary.player_a_searches == 0 && summary.player_a_neural_inference_calls != 0)
        || (summary.player_b_searches == 0 && summary.player_b_neural_inference_calls != 0)
        || summary.search_nodes
            != summary
                .player_a_search_nodes
                .saturating_add(summary.player_b_search_nodes)
        || summary
            .player_a_search_elapsed_ns
            .checked_add(summary.player_b_search_elapsed_ns)
            != Some(summary.search_elapsed_ns)
        || summary.depth_sum
            != summary
                .player_a_depth_sum
                .saturating_add(summary.player_b_depth_sum)
        || summary.searches
            != summary
                .player_a_searches
                .saturating_add(summary.player_b_searches)
        || summary.neural_inference_calls
            != summary
                .player_a_neural_inference_calls
                .saturating_add(summary.player_b_neural_inference_calls)
        || summary.neural_inference_time_ns
            != summary
                .player_a_neural_inference_time_ns
                .saturating_add(summary.player_b_neural_inference_time_ns)
    {
        return Err("arena resume state metrics violate game invariants".to_owned());
    }
    let path = config.output_dir.join(&summary.csa_path);
    let file = storage
        .games
        .open_regular(OsStr::new(&csa_file_name(summary.id)))
        .map_err(|error| format!("cannot open arena resume CSA: {error}"))?;
    let artifact = read_open_file_artifact(file, &path, MAX_CSA_BYTES)?;
    if artifact.sha256 != summary.csa_sha256 || artifact.size != summary.csa_size {
        return Err("arena resume CSA identity mismatch".to_owned());
    }
    let text = std::str::from_utf8(&artifact.bytes)
        .map_err(|_| "arena resume CSA is not valid UTF-8".to_owned())?;
    let parsed =
        parse_csa_game(text).map_err(|error| format!("arena resume CSA replay failed: {error}"))?;
    let canonical = to_csa_game(&parsed)
        .map_err(|error| format!("arena resume CSA canonicalization failed: {error}"))?;
    if canonical.as_bytes() != artifact.bytes {
        return Err("arena resume CSA is not canonical".to_owned());
    }
    if parsed.version != "V3.0"
        || !parsed.metadata.is_empty()
        || parsed.initial_position != config.initial_position
        || parsed.black_name.as_deref() != Some(summary.black.as_str())
        || parsed.white_name.as_deref() != Some(summary.white.as_str())
        || parsed.moves.len() != usize::try_from(summary.moves).unwrap_or(usize::MAX)
    {
        return Err("arena resume CSA game identity mismatch".to_owned());
    }
    let special = parsed
        .special_move
        .as_ref()
        .ok_or_else(|| "arena resume CSA lacks a terminal result".to_owned())?;
    if csa_special_code(special)? != summary.special_move {
        return Err("arena resume CSA terminal identity mismatch".to_owned());
    }
    let expected_validation = match special {
        CsaSpecialMove::Checkmate | CsaSpecialMove::Repetition | CsaSpecialMove::PerpetualCheck => {
            CsaResultValidation::Verified
        }
        CsaSpecialMove::Resign
        | CsaSpecialMove::BlackIllegalAction
        | CsaSpecialMove::WhiteIllegalAction
        | CsaSpecialMove::MaxMoves => CsaResultValidation::ExternalCondition,
        _ => return Err("arena resume CSA contains an unsupported terminal result".to_owned()),
    };
    if parsed.result_validation != expected_validation
        || matches!(
            special,
            CsaSpecialMove::BlackIllegalAction | CsaSpecialMove::WhiteIllegalAction
        ) != (summary.illegal_moves == 1)
    {
        return Err("arena resume CSA terminal validation mismatch".to_owned());
    }
    let replayed_result = replayed_csa_result(&parsed)?;
    if replayed_result != summary.result {
        return Err("arena resume CSA result does not match state".to_owned());
    }
    let (black_spec, white_spec) = assigned_player_specs(config, summary.id);
    let (black_searches, white_searches) = expected_resume_search_counts(
        config,
        opening_book,
        &parsed,
        black_spec,
        white_spec,
        special,
    )?;
    let (first_player_searches, second_player_searches) = if summary.id.is_multiple_of(2) {
        (black_searches, white_searches)
    } else {
        (white_searches, black_searches)
    };
    if summary.player_a_searches != first_player_searches
        || summary.player_b_searches != second_player_searches
    {
        return Err("arena resume search counts disagree with CSA turn selections".to_owned());
    }
    if (!config.player_a.kind.uses_model()
        && (summary.player_a_neural_inference_calls != 0
            || summary.player_a_neural_inference_time_ns != 0))
        || (!config.player_b.kind.uses_model()
            && (summary.player_b_neural_inference_calls != 0
                || summary.player_b_neural_inference_time_ns != 0))
    {
        return Err("arena resume neural metrics disagree with evaluator kinds".to_owned());
    }
    let expected_winners =
        classify_winner(summary.id, black_spec.kind, white_spec.kind, summary.result);
    if summary.search_winner != expected_winners.0
        || summary.player_a_winner != expected_winners.1
        || summary.player_b_winner != expected_winners.2
    {
        return Err("arena resume winner classification mismatch".to_owned());
    }
    Ok(())
}

fn expected_resume_search_counts(
    config: &ArenaConfig,
    opening_book: Option<&OpeningBook>,
    game: &CsaGame,
    black_spec: &PlayerSpec,
    white_spec: &PlayerSpec,
    special: &CsaSpecialMove,
) -> Result<(u64, u64), String> {
    let mut position = game.initial_position.clone();
    let mut black_searches = 0_u64;
    let mut white_searches = 0_u64;
    for &movement in &game.moves {
        let (spec, searches) = match position.side_to_move() {
            Side::Black => (black_spec, &mut black_searches),
            Side::White => (white_spec, &mut white_searches),
        };
        let book_choice = opening_for_player(opening_book, spec, &position, config).and_then(
            |book| {
                book.select_with_policy(
                    &position,
                    OpeningPolicy {
                        profile: spec.opening_profile,
                        ..OpeningPolicy::default()
                    },
                )
            },
        );
        if let Some(choice) = book_choice {
            if choice.movement != movement {
                return Err(
                    "arena resume CSA move differs from its deterministic opening choice"
                        .to_owned(),
                );
            }
        } else if spec.kind.is_search() {
            *searches = searches.saturating_add(1);
        }
        position
            .make_move(movement)
            .map_err(|error| format!("arena resume search replay failed: {error}"))?;
    }

    if matches!(
        special,
        CsaSpecialMove::Resign
            | CsaSpecialMove::BlackIllegalAction
            | CsaSpecialMove::WhiteIllegalAction
    ) {
        let (spec, searches) = match position.side_to_move() {
            Side::Black => (black_spec, &mut black_searches),
            Side::White => (white_spec, &mut white_searches),
        };
        if opening_for_player(opening_book, spec, &position, config)
            .and_then(|book| {
                book.select_with_policy(
                    &position,
                    OpeningPolicy {
                        profile: spec.opening_profile,
                        ..OpeningPolicy::default()
                    },
                )
            })
            .is_some()
        {
            return Err(
                "arena resume terminal selection contradicts a deterministic opening hit"
                    .to_owned(),
            );
        }
        if spec.kind.is_search() {
            *searches = searches.saturating_add(1);
        }
    }
    Ok((black_searches, white_searches))
}

fn replayed_csa_result(game: &CsaGame) -> Result<ArenaResult, String> {
    let mut replay = Game::new(game.initial_position.clone());
    for &movement in &game.moves {
        replay
            .play(movement)
            .map_err(|error| format!("arena resume CSA move replay failed: {error}"))?;
    }
    let special = game
        .special_move
        .as_ref()
        .ok_or_else(|| "arena resume CSA lacks a terminal result".to_owned())?;
    let end = replay.end();
    match (special, end) {
        (CsaSpecialMove::Checkmate, Some(GameEnd::Checkmate { winner })) => {
            Ok(winner_result(winner))
        }
        (CsaSpecialMove::Repetition, Some(GameEnd::Repetition(RepetitionOutcome::NoContest))) => {
            Ok(ArenaResult::Draw)
        }
        (
            CsaSpecialMove::PerpetualCheck,
            Some(GameEnd::Repetition(RepetitionOutcome::PerpetualCheckLoss(loser))),
        ) => Ok(winner_result(loser.opposite())),
        (CsaSpecialMove::Resign, None) => {
            Ok(winner_result(replay.position().side_to_move().opposite()))
        }
        (CsaSpecialMove::BlackIllegalAction, None)
            if replay.position().side_to_move() == Side::Black =>
        {
            Ok(ArenaResult::WhiteWin)
        }
        (CsaSpecialMove::WhiteIllegalAction, None)
            if replay.position().side_to_move() == Side::White =>
        {
            Ok(ArenaResult::BlackWin)
        }
        (CsaSpecialMove::MaxMoves, None) => Ok(ArenaResult::MaxPlies),
        _ => Err(
            "arena resume CSA contains an unsupported or inconsistent terminal result".to_owned(),
        ),
    }
}

fn assigned_player_specs(config: &ArenaConfig, id: u32) -> (&PlayerSpec, &PlayerSpec) {
    if id.is_multiple_of(2) {
        (&config.player_a, &config.player_b)
    } else {
        (&config.player_b, &config.player_a)
    }
}

fn assigned_player_labels(config: &ArenaConfig, id: u32) -> (String, String) {
    let (black, white) = assigned_player_specs(config, id);
    (black.label(), white.label())
}

fn expected_csa_path(id: u32) -> String {
    format!("games/{}", csa_file_name(id))
}

fn csa_file_name(id: u32) -> String {
    format!("game-{:06}.csa", id + 1)
}

#[expect(
    clippy::similar_names,
    reason = "the symmetric player labels are compared in one bounded report calculation"
)]
fn preflight_report_size(config: &ArenaConfig) -> Result<(), String> {
    let identity = EngineIdentity::current();
    let player_a_label = config.player_a.label();
    let player_b_label = config.player_b.label();
    for (name, value) in [
        ("initial SFEN", config.initial_sfen.as_str()),
        ("player A label", player_a_label.as_str()),
        ("player B label", player_b_label.as_str()),
        ("engine name", identity.name),
        ("engine version", identity.version),
    ] {
        if json_string(value).len() > MAX_REPORT_DYNAMIC_TEXT_BYTES {
            return Err(format!(
                "arena {name} exceeds the bounded report text limit"
            ));
        }
    }

    let last_id = config.games.saturating_sub(1);
    let row_bytes = [last_id, last_id.saturating_sub(1)]
        .into_iter()
        .map(|id| maximum_report_game_bytes(config, id))
        .max()
        .unwrap_or(0);
    let games_bytes = row_bytes
        .checked_add(1)
        .and_then(|row| row.checked_mul(usize::try_from(config.games).ok()?))
        .ok_or_else(|| "arena report size preflight overflowed".to_owned())?;
    let worst_case = MAX_REPORT_FIXED_BYTES
        .checked_add(games_bytes)
        .ok_or_else(|| "arena report size preflight overflowed".to_owned())?;
    if worst_case > MAX_REPORT_BYTES {
        return Err(format!(
            "--games={} cannot be guaranteed to fit the Evaluation Lab 5 MiB report limit (worst-case {worst_case} bytes)",
            config.games
        ));
    }
    let state_row_bytes = [last_id, last_id.saturating_sub(1)]
        .into_iter()
        .map(|id| game_state_row(&maximum_game_summary(config, id)).len() + 1)
        .max()
        .unwrap_or(0);
    let state_worst_case = MAX_REPORT_FIXED_BYTES
        .checked_add(
            state_row_bytes
                .checked_mul(usize::try_from(config.games).unwrap_or(usize::MAX))
                .ok_or_else(|| "arena state size preflight overflowed".to_owned())?,
        )
        .ok_or_else(|| "arena state size preflight overflowed".to_owned())?;
    if state_worst_case > MAX_STATE_BYTES {
        return Err(format!(
            "--games={} cannot be guaranteed to fit the 5 MiB resume-state limit (worst-case {state_worst_case} bytes)",
            config.games
        ));
    }
    Ok(())
}

fn maximum_report_game_bytes(config: &ArenaConfig, id: u32) -> usize {
    let game = maximum_game_summary(config, id);
    let mut output = String::new();
    write_report_game(
        &mut output,
        &game,
        MAX_JSON_SAFE_INTEGER,
        MAX_JSON_SAFE_INTEGER,
    );
    output.len()
}

fn maximum_game_summary(config: &ArenaConfig, id: u32) -> GameSummary {
    let (black, white) = assigned_player_labels(config, id);
    let maximum = MAX_JSON_SAFE_INTEGER;
    GameSummary {
        id,
        black,
        white,
        result: ArenaResult::BlackWin,
        moves: u32::MAX,
        csa_path: expected_csa_path(id),
        csa_sha256: "f".repeat(64),
        csa_size: maximum,
        special_move: "OUTE_SENNICHITE".to_owned(),
        search_nodes: maximum,
        search_elapsed_ns: u64::MAX,
        depth_sum: maximum,
        searches: maximum,
        tt_probes: maximum,
        tt_hits: maximum,
        cutoffs: maximum,
        candidate_moves: maximum,
        pruned_moves: maximum,
        neural_inference_calls: maximum,
        neural_inference_time_ns: maximum,
        player_a_search_nodes: maximum,
        player_a_search_elapsed_ns: u64::MAX,
        player_a_depth_sum: maximum,
        player_a_searches: maximum,
        player_a_neural_inference_calls: maximum,
        player_a_neural_inference_time_ns: maximum,
        player_b_search_nodes: maximum,
        player_b_search_elapsed_ns: u64::MAX,
        player_b_depth_sum: maximum,
        player_b_searches: maximum,
        player_b_neural_inference_calls: maximum,
        player_b_neural_inference_time_ns: maximum,
        illegal_moves: maximum,
        search_winner: false,
        player_a_winner: false,
        player_b_winner: false,
    }
}

fn csa_special_code(special: &CsaSpecialMove) -> Result<&str, String> {
    match special {
        CsaSpecialMove::Resign => Ok("TORYO"),
        CsaSpecialMove::Interrupted => Ok("CHUDAN"),
        CsaSpecialMove::Repetition => Ok("SENNICHITE"),
        CsaSpecialMove::PerpetualCheck => Ok("OUTE_SENNICHITE"),
        CsaSpecialMove::IllegalMove => Ok("ILLEGAL_MOVE"),
        CsaSpecialMove::BlackIllegalAction => Ok("+ILLEGAL_ACTION"),
        CsaSpecialMove::WhiteIllegalAction => Ok("-ILLEGAL_ACTION"),
        CsaSpecialMove::TimeUp => Ok("TIME_UP"),
        CsaSpecialMove::EnteringKing => Ok("JISHOGI"),
        CsaSpecialMove::Win => Ok("KACHI"),
        CsaSpecialMove::Draw => Ok("HIKIWAKE"),
        CsaSpecialMove::MaxMoves => Ok("MAX_MOVES"),
        CsaSpecialMove::Takeback => Ok("MATTA"),
        CsaSpecialMove::Checkmate => Ok("TSUMI"),
        CsaSpecialMove::NoMate => Ok("FUZUMI"),
        CsaSpecialMove::Error => Ok("ERROR"),
        CsaSpecialMove::Other(code) => {
            if code.is_empty() {
                Err("empty CSA special-move code".to_owned())
            } else {
                Ok(code)
            }
        }
        _ => Err("unsupported CSA special-move code".to_owned()),
    }
}

fn write_report(
    root: &AnchoredDir,
    config: &ArenaConfig,
    started_at: &str,
    completed_at: Option<&str>,
    games: &[GameSummary],
) -> Result<(), String> {
    let output = render_report(config, started_at, completed_at, games)?;
    root.replace_atomic(OsStr::new("arena-report.json"), &output)
        .map_err(|error| format!("cannot write arena report: {error}"))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ResumeReportState {
    Recoverable,
    Completed,
}

fn validate_resume_report(
    root: &AnchoredDir,
    config: &ArenaConfig,
    started_at: &str,
    games: &[GameSummary],
) -> Result<ResumeReportState, String> {
    let kind = root
        .entry_kind(OsStr::new("arena-report.json"))
        .map_err(|error| format!("cannot inspect arena resume report: {error}"))?;
    let Some(kind) = kind else {
        // State and every retained CSA have already been independently replayed. Repeated
        // crashes may advance that durable state multiple games without ever publishing a
        // report, so absence is recoverable at every validated state prefix.
        return Ok(ResumeReportState::Recoverable);
    };
    if kind != EntryKind::File {
        return Err("arena resume report is not a regular file".to_owned());
    }
    let path = config.output_dir.join("arena-report.json");
    let file = root
        .open_regular(OsStr::new("arena-report.json"))
        .map_err(|error| format!("cannot open completed arena report: {error}"))?;
    let artifact = read_open_file_artifact(file, &path, MAX_REPORT_BYTES as u64)?;
    let value: serde_json::Value = serde_json::from_slice(&artifact.bytes)
        .map_err(|error| format!("completed arena report is invalid JSON: {error}"))?;
    let run = value
        .as_object()
        .and_then(|root| root.get("run"))
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| "arena resume report lacks its run object".to_owned())?;
    let completed_at = run
        .get("completedAt")
        .ok_or_else(|| "arena resume report lacks completedAt".to_owned())?;
    if let Some(completed_at) = completed_at.as_str() {
        if games.len() != usize::try_from(config.games).expect("game count fits usize") {
            return Err("incomplete arena state cannot have a completed report".to_owned());
        }
        if !is_utc_timestamp(completed_at) {
            return Err("completed arena report has an invalid completedAt".to_owned());
        }
        if completed_at < started_at {
            return Err("completed arena report predates its state".to_owned());
        }
        let expected = render_report(config, started_at, Some(completed_at), games)?;
        if artifact.bytes != expected {
            return Err(
                "completed arena report conflicts with validated state and CSA evidence".to_owned(),
            );
        }
        return Ok(ResumeReportState::Completed);
    }
    if !completed_at.is_null() {
        return Err("arena resume report has an invalid completedAt".to_owned());
    }
    let report_games = value
        .as_object()
        .and_then(|root| root.get("games"))
        .and_then(serde_json::Value::as_array)
        .map(Vec::len)
        .ok_or_else(|| "arena resume report lacks its game array".to_owned())?;
    if report_games > games.len() {
        return Err("arena resume report is ahead of its state".to_owned());
    }
    let expected = render_report(config, started_at, None, &games[..report_games])?;
    if artifact.bytes != expected {
        return Err("arena resume report conflicts with validated state and CSA evidence".to_owned());
    }
    Ok(ResumeReportState::Recoverable)
}

#[expect(
    clippy::similar_names,
    clippy::too_many_lines,
    reason = "the closed v2 report and deliberately symmetric A/B evidence stay together"
)]
fn render_report(
    config: &ArenaConfig,
    started_at: &str,
    completed_at: Option<&str>,
    games: &[GameSummary],
) -> Result<Vec<u8>, String> {
    let nodes = saturating_sum(games.iter().map(|game| game.search_nodes));
    let player_a_elapsed_ms =
        allocate_elapsed_milliseconds(games.iter().map(|game| game.player_a_search_elapsed_ns));
    let player_b_elapsed_ms =
        allocate_elapsed_milliseconds(games.iter().map(|game| game.player_b_search_elapsed_ns));
    let player_a_search_elapsed_ms = saturating_sum(player_a_elapsed_ms.iter().copied());
    let player_b_search_elapsed_ms = saturating_sum(player_b_elapsed_ms.iter().copied());
    let elapsed_ms = player_a_search_elapsed_ms.saturating_add(player_b_search_elapsed_ms);
    let depth_sum = saturating_sum(games.iter().map(|game| game.depth_sum));
    let searches = saturating_sum(games.iter().map(|game| game.searches));
    let probes = saturating_sum(games.iter().map(|game| game.tt_probes));
    let hits = saturating_sum(games.iter().map(|game| game.tt_hits));
    let cutoffs = saturating_sum(games.iter().map(|game| game.cutoffs));
    let candidate_moves = saturating_sum(games.iter().map(|game| game.candidate_moves));
    let pruned_moves = saturating_sum(games.iter().map(|game| game.pruned_moves));
    let neural_inference_calls =
        saturating_sum(games.iter().map(|game| game.neural_inference_calls));
    let neural_inference_time_ns =
        saturating_sum(games.iter().map(|game| game.neural_inference_time_ns));
    let player_a_search_nodes = saturating_sum(games.iter().map(|game| game.player_a_search_nodes));
    let player_a_depth_sum = saturating_sum(games.iter().map(|game| game.player_a_depth_sum));
    let player_a_searches = saturating_sum(games.iter().map(|game| game.player_a_searches));
    let player_a_neural_inference_calls = saturating_sum(
        games
            .iter()
            .map(|game| game.player_a_neural_inference_calls),
    );
    let player_a_neural_inference_time_ns = saturating_sum(
        games
            .iter()
            .map(|game| game.player_a_neural_inference_time_ns),
    );
    let player_b_search_nodes = saturating_sum(games.iter().map(|game| game.player_b_search_nodes));
    let player_b_depth_sum = saturating_sum(games.iter().map(|game| game.player_b_depth_sum));
    let player_b_searches = saturating_sum(games.iter().map(|game| game.player_b_searches));
    let player_b_neural_inference_calls = saturating_sum(
        games
            .iter()
            .map(|game| game.player_b_neural_inference_calls),
    );
    let player_b_neural_inference_time_ns = saturating_sum(
        games
            .iter()
            .map(|game| game.player_b_neural_inference_time_ns),
    );
    let illegal = saturating_sum(games.iter().map(|game| game.illegal_moves));
    let draws = games
        .iter()
        .filter(|game| game.result == ArenaResult::Draw)
        .count();
    let finished = games
        .iter()
        .filter(|game| game.result != ArenaResult::MaxPlies)
        .count();
    let search_wins = games.iter().filter(|game| game.search_winner).count();
    let player_wins = (
        games.iter().filter(|game| game.player_a_winner).count(),
        games.iter().filter(|game| game.player_b_winner).count(),
    );
    let nps = decimal_ratio_scaled(nodes, 1_000, elapsed_ms);
    let average_depth = decimal_ratio(depth_sum, searches);
    let tt_hit_rate = decimal_ratio(hits, probes);
    let cutoff_rate = decimal_ratio(cutoffs, nodes);
    let pruning_rate = decimal_ratio(pruned_moves, candidate_moves);
    let milliseconds_per_move = decimal_ratio(elapsed_ms, searches);
    validate_report_counters(
        games,
        &player_a_elapsed_ms,
        &player_b_elapsed_ms,
        &[
            nodes,
            elapsed_ms,
            depth_sum,
            searches,
            probes,
            hits,
            cutoffs,
            candidate_moves,
            pruned_moves,
            neural_inference_calls,
            neural_inference_time_ns,
            player_a_search_nodes,
            player_a_search_elapsed_ms,
            player_a_depth_sum,
            player_a_searches,
            player_a_neural_inference_calls,
            player_a_neural_inference_time_ns,
            player_b_search_nodes,
            player_b_search_elapsed_ms,
            player_b_depth_sum,
            player_b_searches,
            player_b_neural_inference_calls,
            player_b_neural_inference_time_ns,
            illegal,
        ],
    )?;
    validate_report_config_identity(config)?;
    let identity = EngineIdentity::current();
    let engine = format!(
        "{} {} a={} b={} budget={:?}",
        identity.name,
        identity.version,
        config.player_a.label(),
        config.player_b.label(),
        config.budget
    );

    let mut output = String::with_capacity(1_024 + games.len().saturating_mul(180));
    write!(
        output,
        "{{\"schema\":\"{SCHEMA}\",\"run\":{{\"seed\":{},\"gameLimit\":{},\"engine\":{},\"gitCommit\":",
        config.seed,
        config.games,
        json_string(&engine)
    )
    .expect("writing to String cannot fail");
    match &config.git_commit {
        Some(value) => output.push_str(&json_string(value)),
        None => output.push_str("null"),
    }
    write!(
        output,
        ",\"startedAt\":{},\"completedAt\":",
        json_string(started_at)
    )
    .expect("writing to String cannot fail");
    match completed_at {
        Some(value) => output.push_str(&json_string(value)),
        None => output.push_str("null"),
    }
    write!(
        output,
        ",\"initialSfen\":{},\"maxPlies\":{},\"configSha256\":{},\"budget\":",
        json_string(&config.initial_sfen),
        config.max_plies,
        json_string(&sha256_bytes(config_signature(config).as_bytes())),
    )
    .expect("writing to String cannot fail");
    write_budget_identity(&mut output, &config.budget);
    output.push_str(",\"playerA\":");
    write_player_identity(&mut output, &config.player_a);
    output.push_str(",\"playerB\":");
    write_player_identity(&mut output, &config.player_b);
    output.push_str(",\"opening\":");
    write_opening_identity(&mut output, config);
    write!(
        output,
        "}},\"metrics\":{{\"games\":{},\"finishedGames\":{},\"playerAWins\":{},\"playerBWins\":{},\"searchWins\":{},\"draws\":{},\"nodesPerSecond\":{nps},\"averageDepth\":{average_depth},\"ttHitRate\":{tt_hit_rate},\"cutoffRate\":{cutoff_rate},\"pruningRate\":{pruning_rate},\"millisecondsPerMove\":{milliseconds_per_move},\"neuralInferenceCalls\":{neural_inference_calls},\"neuralInferenceTimeNs\":{neural_inference_time_ns},\"playerASearchNodes\":{player_a_search_nodes},\"playerASearchElapsedMs\":{player_a_search_elapsed_ms},\"playerADepthSum\":{player_a_depth_sum},\"playerASearches\":{player_a_searches},\"playerANeuralInferenceCalls\":{player_a_neural_inference_calls},\"playerANeuralInferenceTimeNs\":{player_a_neural_inference_time_ns},\"playerBSearchNodes\":{player_b_search_nodes},\"playerBSearchElapsedMs\":{player_b_search_elapsed_ms},\"playerBDepthSum\":{player_b_depth_sum},\"playerBSearches\":{player_b_searches},\"playerBNeuralInferenceCalls\":{player_b_neural_inference_calls},\"playerBNeuralInferenceTimeNs\":{player_b_neural_inference_time_ns},\"peakMemoryBytes\":null,\"illegalMoves\":{illegal}}},\"games\":[",
        games.len(),
        finished,
        player_wins.0,
        player_wins.1,
        search_wins,
        draws
    )
    .expect("writing to String cannot fail");
    for (index, game) in games.iter().enumerate() {
        if index > 0 {
            output.push(',');
        }
        write_report_game(
            &mut output,
            game,
            player_a_elapsed_ms[index],
            player_b_elapsed_ms[index],
        );
    }
    output.push_str("]}");
    if output.len() > MAX_REPORT_BYTES {
        return Err("arena report exceeds the Evaluation Lab 5 MiB limit".to_owned());
    }
    Ok(output.into_bytes())
}

#[expect(
    clippy::similar_names,
    reason = "the public v2 row keeps symmetric A/B elapsed evidence adjacent"
)]
fn write_report_game(
    output: &mut String,
    game: &GameSummary,
    player_a_elapsed_ms: u64,
    player_b_elapsed_ms: u64,
) {
    write!(
        output,
        "{{\"id\":{},\"black\":{},\"white\":{},\"result\":\"{}\",\"moves\":{},\"csaPath\":{},\"csaSha256\":{},\"csaSize\":{},\"neuralInferenceCalls\":{},\"neuralInferenceTimeNs\":{},\"playerASearchNodes\":{},\"playerASearchElapsedMs\":{},\"playerADepthSum\":{},\"playerASearches\":{},\"playerANeuralInferenceCalls\":{},\"playerANeuralInferenceTimeNs\":{},\"playerBSearchNodes\":{},\"playerBSearchElapsedMs\":{},\"playerBDepthSum\":{},\"playerBSearches\":{},\"playerBNeuralInferenceCalls\":{},\"playerBNeuralInferenceTimeNs\":{}}}",
        game.id,
        json_string(&game.black),
        json_string(&game.white),
        game.result.json(),
        game.moves,
        json_string(&game.csa_path),
        json_string(&game.csa_sha256),
        game.csa_size,
        game.neural_inference_calls,
        game.neural_inference_time_ns,
        game.player_a_search_nodes,
        player_a_elapsed_ms,
        game.player_a_depth_sum,
        game.player_a_searches,
        game.player_a_neural_inference_calls,
        game.player_a_neural_inference_time_ns,
        game.player_b_search_nodes,
        player_b_elapsed_ms,
        game.player_b_depth_sum,
        game.player_b_searches,
        game.player_b_neural_inference_calls,
        game.player_b_neural_inference_time_ns,
    )
    .expect("writing to String cannot fail");
}

fn validate_report_config_identity(config: &ArenaConfig) -> Result<(), String> {
    for (name, player) in [("A", &config.player_a), ("B", &config.player_b)] {
        let model_identity_parts = [
            player.model_sha256.is_some(),
            player.model_size.is_some(),
            player.model_payload_sha256().is_some(),
            player.model_architecture_version().is_some(),
            player.model_quantization().is_some(),
        ];
        let valid_presence = if player.kind.uses_model() {
            model_identity_parts.into_iter().all(std::convert::identity)
        } else {
            model_identity_parts
                .into_iter()
                .all(|is_present| !is_present)
        };
        if !valid_presence {
            return Err(format!(
                "arena player {name} lacks a complete immutable model identity"
            ));
        }
        if let Some(hash) = player.model_sha256.as_deref()
            && (!is_canonical_sha256(hash)
                || player.model_size == Some(0)
                || player
                    .model_size
                    .is_some_and(|size| size > MAX_JSON_SAFE_INTEGER)
                || !is_canonical_sha256(
                    player
                        .model_payload_sha256()
                        .as_deref()
                        .expect("complete neural identity has a payload hash"),
                ))
        {
            return Err(format!("arena player {name} has an invalid model identity"));
        }
    }
    let has_opening = config.opening_book_path.is_some();
    if has_opening != config.opening_book_sha256.is_some()
        || has_opening != config.opening_book_size.is_some()
        || has_opening != (config.player_a.opening || config.player_b.opening)
    {
        return Err("arena opening configuration lacks a complete immutable identity".to_owned());
    }
    if let Some(hash) = config.opening_book_sha256.as_deref()
        && (!is_canonical_sha256(hash)
            || config.opening_book_size == Some(0)
            || config
                .opening_book_size
                .is_some_and(|size| size > MAX_JSON_SAFE_INTEGER))
    {
        return Err("arena opening configuration has an invalid artifact identity".to_owned());
    }
    Ok(())
}

fn write_budget_identity(output: &mut String, budget: &Budget) {
    let (kind, value) = match budget {
        Budget::Nodes(value) => ("nodes", *value),
        Budget::MoveTime(value) => ("movetime_ms", *value),
    };
    write!(
        output,
        "{{\"kind\":{},\"value\":{value}}}",
        json_string(kind)
    )
    .expect("writing to String cannot fail");
}

fn write_player_identity(output: &mut String, player: &PlayerSpec) {
    write!(
        output,
        "{{\"label\":{},\"evaluatorKind\":{},\"searchDepth\":",
        json_string(&player.label()),
        json_string(player.kind.report_name()),
    )
    .expect("writing to String cannot fail");
    if player.kind.is_search() {
        write!(output, "{}", player.depth).expect("writing to String cannot fail");
    } else {
        output.push_str("null");
    }
    output.push_str(",\"hashMegabytes\":");
    if player.kind.is_search() {
        write!(output, "{}", player.hash_megabytes).expect("writing to String cannot fail");
    } else {
        output.push_str("null");
    }
    output.push_str(",\"transposition\":");
    if player.kind.is_search() {
        output.push_str(if player.transposition {
            "true"
        } else {
            "false"
        });
    } else {
        output.push_str("null");
    }
    output.push_str(",\"modelArtifactSha256\":");
    write_optional_json_string(output, player.model_sha256.as_deref());
    output.push_str(",\"modelArtifactSize\":");
    write_optional_u64(output, player.model_size);
    output.push_str(",\"modelPayloadSha256\":");
    write_optional_json_string(output, player.model_payload_sha256().as_deref());
    output.push_str(",\"architectureVersion\":");
    write_optional_u64(output, player.model_architecture_version().map(u64::from));
    output.push_str(",\"quantization\":");
    write_optional_json_string(output, player.model_quantization());
    write!(
        output,
        ",\"openingEnabled\":{}}}",
        if player.opening { "true" } else { "false" }
    )
    .expect("writing to String cannot fail");
}

fn write_opening_identity(output: &mut String, config: &ArenaConfig) {
    let enabled = config.opening_book_sha256.is_some();
    output.push_str(if enabled {
        "{\"enabled\":true,\"artifactSha256\":"
    } else {
        "{\"enabled\":false,\"artifactSha256\":"
    });
    write_optional_json_string(output, config.opening_book_sha256.as_deref());
    output.push_str(",\"artifactSize\":");
    write_optional_u64(output, config.opening_book_size);
    output.push_str(",\"maxPlies\":");
    if enabled {
        write!(output, "{}", config.opening_max_plies).expect("writing to String cannot fail");
    } else {
        output.push_str("null");
    }
    output.push('}');
}

fn write_optional_json_string(output: &mut String, value: Option<&str>) {
    match value {
        Some(value) => output.push_str(&json_string(value)),
        None => output.push_str("null"),
    }
}

fn write_optional_u64(output: &mut String, value: Option<u64>) {
    match value {
        Some(value) => write!(output, "{value}").expect("writing to String cannot fail"),
        None => output.push_str("null"),
    }
}

fn saturating_sum(values: impl Iterator<Item = u64>) -> u64 {
    values.fold(0_u64, u64::saturating_add)
}

fn allocate_elapsed_milliseconds(values: impl Iterator<Item = u64>) -> Vec<u64> {
    let mut submillisecond_remainder = 0_u128;
    values
        .map(|nanoseconds| {
            let accumulated = submillisecond_remainder + u128::from(nanoseconds);
            submillisecond_remainder = accumulated % 1_000_000;
            u64::try_from(accumulated / 1_000_000)
                .expect("one u64 nanosecond counter always fits after millisecond conversion")
        })
        .collect()
}

#[expect(
    clippy::similar_names,
    reason = "the public v2 counter validator keeps symmetric A/B evidence adjacent"
)]
fn validate_report_counters(
    games: &[GameSummary],
    player_a_elapsed_ms: &[u64],
    player_b_elapsed_ms: &[u64],
    totals: &[u64],
) -> Result<(), String> {
    if totals.iter().any(|value| *value > MAX_JSON_SAFE_INTEGER) {
        return Err("arena aggregate metric exceeds the JSON-safe integer limit".to_owned());
    }
    for (index, game) in games.iter().enumerate() {
        let counters = [
            u64::from(game.id),
            u64::from(game.moves),
            game.csa_size,
            game.neural_inference_calls,
            game.neural_inference_time_ns,
            game.player_a_search_nodes,
            player_a_elapsed_ms[index],
            game.player_a_depth_sum,
            game.player_a_searches,
            game.player_a_neural_inference_calls,
            game.player_a_neural_inference_time_ns,
            game.player_b_search_nodes,
            player_b_elapsed_ms[index],
            game.player_b_depth_sum,
            game.player_b_searches,
            game.player_b_neural_inference_calls,
            game.player_b_neural_inference_time_ns,
        ];
        if counters.iter().any(|value| *value > MAX_JSON_SAFE_INTEGER) {
            return Err(format!(
                "arena game {} metric exceeds the JSON-safe integer limit",
                game.id
            ));
        }
    }
    Ok(())
}

fn decimal_ratio(numerator: u64, denominator: u64) -> String {
    decimal_ratio_scaled(numerator, 1, denominator)
}

fn decimal_ratio_scaled(numerator: u64, multiplier: u64, denominator: u64) -> String {
    if denominator == 0 {
        return "0.000000".to_owned();
    }
    let numerator = u128::from(numerator).saturating_mul(u128::from(multiplier));
    let denominator = u128::from(denominator);
    let scaled = numerator.saturating_mul(1_000_000) / denominator;
    format!("{}.{:06}", scaled / 1_000_000, scaled % 1_000_000)
}

fn timestamp() -> String {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    timestamp_from_unix(seconds)
}

fn timestamp_from_unix(seconds: u64) -> String {
    let days = i64::try_from(seconds / 86_400).unwrap_or(i64::MAX);
    let seconds_of_day = seconds % 86_400;
    let hour = seconds_of_day / 3_600;
    let minute = seconds_of_day % 3_600 / 60;
    let second = seconds_of_day % 60;
    let (year, month, day) = civil_from_days(days);
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z")
}

fn civil_from_days(days_since_epoch: i64) -> (i64, i64, i64) {
    let adjusted = days_since_epoch.saturating_add(719_468);
    let era = if adjusted >= 0 {
        adjusted
    } else {
        adjusted - 146_096
    } / 146_097;
    let day_of_era = adjusted - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let mut year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_prime = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_prime + 2) / 5 + 1;
    let month = month_prime + if month_prime < 10 { 3 } else { -9 };
    year += i64::from(month <= 2);
    (year, month, day)
}

fn is_utc_timestamp(value: &str) -> bool {
    let bytes = value.as_bytes();
    if bytes.len() != 20
        || bytes[4] != b'-'
        || bytes[7] != b'-'
        || bytes[10] != b'T'
        || bytes[13] != b':'
        || bytes[16] != b':'
        || bytes[19] != b'Z'
    {
        return false;
    }
    for index in [0, 1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18] {
        if !bytes[index].is_ascii_digit() {
            return false;
        }
    }
    let number =
        |start: usize| u16::from(bytes[start] - b'0') * 10 + u16::from(bytes[start + 1] - b'0');
    let year = u32::from(bytes[0] - b'0') * 1_000
        + u32::from(bytes[1] - b'0') * 100
        + u32::from(bytes[2] - b'0') * 10
        + u32::from(bytes[3] - b'0');
    let month = number(5);
    let day = number(8);
    let leap = year.is_multiple_of(4) && (!year.is_multiple_of(100) || year.is_multiple_of(400));
    let maximum_day = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if leap => 29,
        2 => 28,
        _ => return false,
    };
    year != 0
        && (1..=maximum_day).contains(&day)
        && number(11) <= 23
        && number(14) <= 59
        && number(17) <= 59
}

fn json_string(value: &str) -> String {
    let mut output = String::with_capacity(value.len() + 2);
    output.push('"');
    for character in value.chars() {
        match character {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            '\u{08}' => output.push_str("\\b"),
            '\u{0C}' => output.push_str("\\f"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            character if character <= '\u{1F}' => {
                write!(output, "\\u{:04x}", u32::from(character))
                    .expect("writing to String cannot fail");
            }
            character => output.push(character),
        }
    }
    output.push('"');
    output
}

#[cfg(test)]
fn atomic_write(path: &Path, contents: &str) -> Result<(), String> {
    if path.exists() || path.is_symlink() {
        reject_symlink_or_non_file(path, "arena mutable output")?;
    }
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| "output path has no valid file name".to_owned())?;
    let sequence = TEMPORARY_FILE_SEQUENCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let temporary = path.with_file_name(format!(
        ".{file_name}.tmp-{}-{sequence}",
        std::process::id()
    ));
    let mut temporary_file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .map_err(|error| format!("cannot create {}: {error}", temporary.display()))?;

    let write_result = temporary_file
        .write_all(contents.as_bytes())
        .and_then(|()| temporary_file.sync_all());
    drop(temporary_file);
    if let Err(error) = write_result {
        let _ = std::fs::remove_file(&temporary);
        return Err(format!("cannot write {}: {error}", temporary.display()));
    }

    if let Err(error) = std::fs::rename(&temporary, path) {
        let _ = std::fs::remove_file(&temporary);
        return Err(format!("cannot finalize output: {error}"));
    }
    sync_directory(path.parent().unwrap_or_else(|| Path::new(".")))
}

#[cfg(test)]
fn atomic_write_new(path: &Path, contents: &str) -> Result<(), String> {
    if path.exists() || path.is_symlink() {
        return Err(format!(
            "refusing to overwrite existing arena artifact {}",
            path.display()
        ));
    }
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| "output path has no valid file name".to_owned())?;
    let sequence = TEMPORARY_FILE_SEQUENCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let temporary = path.with_file_name(format!(
        ".{file_name}.tmp-{}-{sequence}",
        std::process::id()
    ));
    let mut temporary_file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .map_err(|error| format!("cannot create {}: {error}", temporary.display()))?;
    let write_result = temporary_file
        .write_all(contents.as_bytes())
        .and_then(|()| temporary_file.sync_all());
    drop(temporary_file);
    if let Err(error) = write_result {
        let _ = std::fs::remove_file(&temporary);
        return Err(format!("cannot write {}: {error}", temporary.display()));
    }
    if let Err(error) = std::fs::hard_link(&temporary, path) {
        let _ = std::fs::remove_file(&temporary);
        return Err(format!(
            "cannot publish new arena artifact {}: {error}",
            path.display()
        ));
    }
    std::fs::remove_file(&temporary)
        .map_err(|error| format!("cannot remove {}: {error}", temporary.display()))?;
    sync_directory(path.parent().unwrap_or_else(|| Path::new(".")))?;
    Ok(())
}

#[cfg(test)]
fn reject_symlink_or_non_file(path: &Path, name: &str) -> Result<(), String> {
    let metadata = std::fs::symlink_metadata(path)
        .map_err(|error| format!("cannot inspect {name} {}: {error}", path.display()))?;
    if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
        return Err(format!("{name} must be a regular non-symlink file"));
    }
    Ok(())
}

#[cfg(test)]
fn sync_directory(path: &Path) -> Result<(), String> {
    File::open(path)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| format!("cannot sync directory {}: {error}", path.display()))
}

fn hash_entries(megabytes: usize) -> usize {
    SearchEngine::transposition_entries_for_megabytes(megabytes).max(1)
}

fn mix_game_seed(seed: u64, id: u32) -> u64 {
    let mut state = seed ^ u64::from(id);
    splitmix64(&mut state)
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len().saturating_mul(2));
    for byte in bytes {
        output.push(char::from(HEX[usize::from(byte >> 4)]));
        output.push(char::from(HEX[usize::from(byte & 0x0F)]));
    }
    output
}

fn hex_decode(value: &str) -> Result<Vec<u8>, String> {
    if !value.len().is_multiple_of(2) {
        return Err("invalid state hex length".to_owned());
    }
    value
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            let high = hex_digit(pair[0])?;
            let low = hex_digit(pair[1])?;
            Ok(high << 4 | low)
        })
        .collect()
}

fn hex_digit(byte: u8) -> Result<u8, String> {
    match byte {
        b'0'..=b'9' => Ok(byte - b'0'),
        b'a'..=b'f' => Ok(byte - b'a' + 10),
        _ => Err("invalid state hex digit".to_owned()),
    }
}

fn parse_state<T>(value: &str, name: &str) -> Result<T, String>
where
    T: std::str::FromStr + ToString,
{
    if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(format!("invalid arena state {name}"));
    }
    let parsed = value
        .parse::<T>()
        .map_err(|_| format!("invalid arena state {name}"))?;
    if parsed.to_string() != value {
        return Err(format!("invalid arena state {name}"));
    }
    Ok(parsed)
}

fn parse_state_bool(value: &str) -> Result<bool, String> {
    match value {
        "0" => Ok(false),
        "1" => Ok(true),
        _ => Err("invalid state Boolean".to_owned()),
    }
}

fn validate_sha256(value: &str, name: &str) -> Result<String, String> {
    if !is_canonical_sha256(value) {
        return Err(format!("invalid arena state {name}"));
    }
    Ok(value.to_owned())
}

fn is_canonical_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn validate_state_text(value: &str) -> Result<String, String> {
    if value.is_empty()
        || value.len() > 256
        || value
            .bytes()
            .any(|byte| byte.is_ascii_control() || byte == b'\\')
    {
        return Err("invalid arena state text".to_owned());
    }
    Ok(value.to_owned())
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeSet, io::Write as _, path::PathBuf};

    use flate2::{Compression, write::GzEncoder};
    use sha2::{Digest, Sha256};

    use open_shogi_core::AnchoredDir;

    use super::{
        ArenaConfig, ArenaResult, Budget, GameOutcome, GameSummary, LOCK_FILE_NAME, MAX_GAMES,
        MAX_STATE_BYTES, OpeningProfile, PendingGame, Player, PlayerKind, PlayerSpec, atomic_write,
        atomic_write_new, classify_winner, commit_pending_game_with_hooks, config_signature,
        encode_csa, encode_game_journal, expected_resume_search_counts, final_outcome, hex_decode,
        hex_encode, is_utc_timestamp, json_string, load_opening_book, parse_arguments, parse_state,
        preflight_report_size, prepare_output_directory, read_state, recover_game_journal,
        recover_game_journal_with_hooks,
        recover_owned_temporary_files, render_report, replayed_csa_result, run,
        timestamp_from_unix, validate_resume_artifacts, validate_resume_report, write_report,
        write_state, ResumeReportState,
    };

    #[test]
    fn arguments_are_bounded_and_budget_is_exclusive() {
        assert!(parse_arguments(&["--games".into(), "0".into()]).is_err());
        assert!(
            parse_arguments(&[
                "--nodes".into(),
                "1".into(),
                "--movetime-ms".into(),
                "1".into()
            ])
            .is_err()
        );
        let config = parse_arguments(&[
            "--player-a".into(),
            "search".into(),
            "--player-b".into(),
            "search".into(),
        ])
        .unwrap();
        assert_eq!(config.player_b.kind, PlayerKind::HandcraftedExperimental);
        assert!(parse_arguments(&["--git-commit".into(), "not-a-hash".into()]).is_err());
        assert!(parse_arguments(&["--git-commit".into(), "abcdef012345".into()]).is_ok());
        assert!(parse_arguments(&["--git-commit".into(), "ABCDEF0".into()]).is_err());
        assert!(parse_arguments(&["--seed".into(), "9007199254740992".into()]).is_err());
        assert!(parse_arguments(&["--seed".into(), "9007199254740991".into()]).is_ok());
        assert!(
            parse_arguments(&["--games".into(), "1".into(), "--games".into(), "2".into(),])
                .is_err()
        );

        let canonical = parse_arguments(&[
            "--sfen".into(),
            "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1".into(),
        ])
        .unwrap();
        assert_eq!(
            canonical.initial_sfen,
            "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
        );
    }

    #[test]
    fn report_preflight_accepts_its_maximum_and_rejects_every_larger_count() {
        let mut config = parse_arguments(&[]).unwrap();
        let largest = (1..=MAX_GAMES)
            .take_while(|games| {
                config.games = *games;
                preflight_report_size(&config).is_ok()
            })
            .last()
            .expect("one game fits the report");
        config.games = largest;
        assert!(preflight_report_size(&config).is_ok());
        assert!(largest < MAX_GAMES);
        for games in largest + 1..=MAX_GAMES {
            config.games = games;
            assert!(preflight_report_size(&config).is_err());
        }
    }

    #[test]
    fn directory_bound_covers_a_complete_arena_and_the_preserved_temp_budget() {
        assert!(
            super::MAX_ARENA_DIRECTORY_ENTRIES
                > MAX_GAMES as usize + super::MAX_PRESERVED_UNPROVEN_TEMPORARIES
        );
    }

    #[test]
    fn elapsed_nanoseconds_are_aggregated_losslessly_before_report_conversion() {
        let nanoseconds = [600_000_u64, 600_000, 2_900_000, 100_000];
        let rows = super::allocate_elapsed_milliseconds(nanoseconds.into_iter());
        assert_eq!(rows, [0, 1, 3, 0]);
        assert_eq!(rows.into_iter().sum::<u64>(), 4);
    }

    #[test]
    fn json_escaping_covers_controls_and_quotes() {
        assert_eq!(json_string("a\"\\\n\u{01}"), "\"a\\\"\\\\\\n\\u0001\"");
    }

    #[test]
    fn state_round_trip_and_mismatch_refusal() {
        let directory = std::env::temp_dir()
            .canonicalize()
            .unwrap()
            .join(format!("open-shogi-arena-test-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("arena.state");
        let root = AnchoredDir::open_existing(&directory).unwrap();
        let config = parse_arguments(&[]).unwrap();
        let signature = config_signature(&config);
        let games = vec![GameSummary {
            id: 0,
            black: "search:d4:h16:tt-on".into(),
            white: "random".into(),
            result: ArenaResult::BlackWin,
            moves: 12,
            csa_path: "games/game-000001.csa".into(),
            csa_sha256: "a".repeat(64),
            csa_size: 100,
            special_move: "TORYO".into(),
            search_nodes: 10,
            search_elapsed_ns: 2_000_000,
            depth_sum: 3,
            searches: 4,
            tt_probes: 5,
            tt_hits: 2,
            cutoffs: 1,
            candidate_moves: 8,
            pruned_moves: 3,
            neural_inference_calls: 0,
            neural_inference_time_ns: 0,
            player_a_search_nodes: 10,
            player_a_search_elapsed_ns: 2_000_000,
            player_a_depth_sum: 3,
            player_a_searches: 4,
            player_a_neural_inference_calls: 0,
            player_a_neural_inference_time_ns: 0,
            player_b_search_nodes: 0,
            player_b_search_elapsed_ns: 0,
            player_b_depth_sum: 0,
            player_b_searches: 0,
            player_b_neural_inference_calls: 0,
            player_b_neural_inference_time_ns: 0,
            illegal_moves: 0,
            search_winner: true,
            player_a_winner: true,
            player_b_winner: false,
        }];
        write_state(&root, &signature, "2026-07-29T00:00:00Z", &games).unwrap();
        let (started, loaded) = read_state(&root, &signature).unwrap();
        assert_eq!(started, "2026-07-29T00:00:00Z");
        assert_eq!(loaded.len(), 1);
        assert!(read_state(&root, "different").is_err());

        std::fs::write(&path, "phase2_arena_state/v1\n").unwrap();
        assert_eq!(
            read_state(&root, &signature).unwrap_err(),
            "arena state schema mismatch"
        );
    }

    #[test]
    fn oversized_state_is_rejected_after_a_bounded_read() {
        let directory = temporary_arena_directory("oversized-state");
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("arena.state");
        std::fs::write(&path, vec![b'x'; MAX_STATE_BYTES + 1]).unwrap();
        let root = AnchoredDir::open_existing(&directory).unwrap();
        assert_eq!(
            read_state(&root, "unused").unwrap_err(),
            "arena state exceeds 5 MiB limit"
        );
    }

    #[test]
    fn output_directory_lock_refuses_a_concurrent_arena() {
        let mut config = parse_arguments(&[]).unwrap();
        config.output_dir = temporary_arena_directory("lock");
        let _first_lock = prepare_output_directory(&config).unwrap();
        let error = prepare_output_directory(&config).unwrap_err();
        assert_eq!(
            error,
            "another arena run is already using this output directory"
        );
    }

    #[test]
    fn replacing_a_legacy_lock_file_cannot_bypass_the_directory_inode_lock() {
        let mut config = parse_arguments(&[]).unwrap();
        config.output_dir = temporary_arena_directory("lock-recreate");
        let _first_run = prepare_output_directory(&config).unwrap();
        let legacy_lock = config.output_dir.join(LOCK_FILE_NAME);
        std::fs::write(&legacy_lock, b"first").unwrap();
        std::fs::remove_file(&legacy_lock).unwrap();
        std::fs::write(&legacy_lock, b"replacement").unwrap();

        let mut resume = config;
        resume.resume = true;
        let error = prepare_output_directory(&resume).unwrap_err();

        assert_eq!(
            error,
            "another arena run is already using this output directory"
        );
        assert_eq!(std::fs::read(legacy_lock).unwrap(), b"replacement");
    }

    #[cfg(unix)]
    #[test]
    fn output_directory_rejects_an_intermediate_symlink() {
        use std::os::unix::fs::symlink;

        let root = temporary_arena_directory("output-symlink");
        std::fs::create_dir_all(root.join("real")).unwrap();
        symlink(root.join("real"), root.join("alias")).unwrap();
        let mut config = parse_arguments(&[]).unwrap();
        config.output_dir = root.join("alias/arena");

        assert!(prepare_output_directory(&config).is_err());
        assert!(!root.join("real/arena").exists());
    }

    #[test]
    fn resume_refuses_to_recreate_a_missing_games_directory() {
        let mut config = parse_arguments(&[]).unwrap();
        config.output_dir = temporary_arena_directory("missing-games");
        config.resume = true;
        std::fs::create_dir_all(&config.output_dir).unwrap();

        assert!(prepare_output_directory(&config).is_err());
        assert!(!config.output_dir.join("games").exists());
    }

    #[test]
    fn resume_without_state_fails_before_modifying_any_arena_bytes() {
        let root = temporary_arena_directory("resume-missing-state");
        std::fs::create_dir_all(root.join("games")).unwrap();
        std::fs::write(root.join("games/foreign-evidence"), b"preserve").unwrap();
        let before = std::fs::read(root.join("games/foreign-evidence")).unwrap();

        let error = run(&[
            "--resume".to_owned(),
            "--output-dir".to_owned(),
            root.display().to_string(),
        ])
        .unwrap_err();

        assert!(error.contains("requires an existing arena.state"), "{error}");
        assert_eq!(
            std::fs::read(root.join("games/foreign-evidence")).unwrap(),
            before
        );
        let names = std::fs::read_dir(&root)
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect::<Vec<_>>();
        assert_eq!(names, [std::ffi::OsString::from("games")]);
    }

    #[test]
    fn resolved_neural_player_does_not_reopen_model_path() {
        let directory = temporary_arena_directory("model-snapshot");
        std::fs::create_dir_all(&directory).unwrap();
        let model_path = directory.join("model.osaval");
        std::fs::write(&model_path, model_bytes()).unwrap();
        let config = parse_arguments(&[
            "--player-a".into(),
            "neural".into(),
            "--a-model".into(),
            model_path.to_string_lossy().into_owned(),
        ])
        .unwrap();
        std::fs::remove_file(&model_path).unwrap();

        assert!(Player::new(&config.player_a, 7).is_ok());
    }

    #[test]
    fn loaded_opening_book_does_not_reopen_source_path() {
        let directory = temporary_arena_directory("opening-snapshot");
        std::fs::create_dir_all(&directory).unwrap();
        let book_path = directory.join("opening.jsonl.gz");
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder
            .write_all(
                br#"{"schema":"phase3_opening_export/v1","stateKey":"eb5bc2ef917ec96fe2172f96d7060ec4f39322caf929fc1177d2c9fc8b937ebc","stateSfen":"lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b -","moveUsi":"7g7f","count":1,"wins":1,"losses":0,"draws":0,"unknown":0,"scoreRate":1.0,"decisiveN":1,"decisiveWinRate":1.0,"decisiveWinRateWilson95Low":0.20654931437723745,"decisiveWinRateWilson95High":1.0,"blackWins":1,"whiteWins":0,"sideSpecificDecisiveN":1,"blackDecisiveWinRate":1.0,"whiteDecisiveWinRate":0.0,"averageFullPlies":100.0,"averageRemainingPlies":100.0,"sourceCounts":{"fixture":1}}
"#,
            )
            .unwrap();
        std::fs::write(&book_path, encoder.finish().unwrap()).unwrap();
        let mut config = parse_arguments(&[
            "--a-opening".into(),
            "--opening-book".into(),
            book_path.to_string_lossy().into_owned(),
        ])
        .unwrap();
        let book = load_opening_book(&mut config).unwrap().unwrap();
        std::fs::remove_file(&book_path).unwrap();

        assert_eq!(book.records(), 1);
        assert!(config.opening_book_sha256.is_some());
    }

    #[test]
    fn arena_opening_profile_is_explicit_and_strict_by_default() {
        let default = parse_arguments(&[]).unwrap();
        assert_eq!(
            default.player_a.opening_profile,
            OpeningProfile::IbishaStrict
        );
        let unrestricted = parse_arguments(&[
            "--a-opening".into(),
            "--a-opening-profile".into(),
            "unrestricted".into(),
            "--opening-book".into(),
            "ignored.jsonl.gz".into(),
        ])
        .unwrap();
        assert_eq!(
            unrestricted.player_a.opening_profile,
            OpeningProfile::Unrestricted
        );
        assert!(unrestricted.player_a.label().contains("book-on:unrestricted"));
    }

    #[test]
    fn atomic_write_does_not_reuse_the_predictable_legacy_temporary_path() {
        let directory = temporary_arena_directory("atomic-write");
        std::fs::create_dir_all(&directory).unwrap();
        let target = directory.join("arena.state");
        let planted_path = directory.join(".arena.state.tmp");
        std::fs::write(&planted_path, "sentinel").unwrap();

        atomic_write(&target, "updated").unwrap();

        assert_eq!(std::fs::read_to_string(target).unwrap(), "updated");
        assert_eq!(std::fs::read_to_string(planted_path).unwrap(), "sentinel");
    }

    #[test]
    fn immutable_arena_artifact_refuses_an_existing_destination() {
        let directory = temporary_arena_directory("immutable-write");
        std::fs::create_dir_all(&directory).unwrap();
        let target = directory.join("game.csa");
        std::fs::write(&target, "sentinel").unwrap();

        assert!(atomic_write_new(&target, "replacement").is_err());
        assert_eq!(std::fs::read_to_string(target).unwrap(), "sentinel");
    }

    #[test]
    fn win_classification_distinguishes_players_and_preserves_search_wins() {
        assert_eq!(
            classify_winner(
                0,
                PlayerKind::HandcraftedExperimental,
                PlayerKind::Random,
                ArenaResult::BlackWin,
            ),
            (true, true, false)
        );
        assert_eq!(
            classify_winner(
                1,
                PlayerKind::Neural,
                PlayerKind::HandcraftedBaseline,
                ArenaResult::WhiteWin,
            ),
            (true, true, false)
        );
        assert_eq!(
            classify_winner(
                0,
                PlayerKind::HandcraftedBaseline,
                PlayerKind::Neural,
                ArenaResult::Draw,
            ),
            (false, false, false)
        );
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "one resume test mutates every CSA/state binding independently"
    )]
    fn resume_verifies_csa_hash_replay_assignment_and_crash_window() {
        let mut config = parse_arguments(&[]).unwrap();
        config.games = 2;
        config.max_plies = 1;
        config.output_dir = temporary_arena_directory("resume-integrity");
        let storage = prepare_output_directory(&config).unwrap();
        let black = config.player_a.label();
        let white = config.player_b.label();
        let outcome = GameOutcome {
            result: ArenaResult::MaxPlies,
            special: open_shogi_core::CsaSpecialMove::MaxMoves,
            validation: open_shogi_core::CsaResultValidation::ExternalCondition,
        };
        let relative = "games/game-000001.csa".to_owned();
        let (saved, csa) = encode_csa(
            &config.initial_position,
            &[open_shogi_core::parse_usi_move("7g7f").unwrap()],
            &black,
            &white,
            &outcome,
        )
        .unwrap();
        storage
            .games
            .publish_new_atomic(std::ffi::OsStr::new("game-000001.csa"), csa.as_bytes())
            .unwrap();
        let summary = GameSummary {
            id: 0,
            black,
            white,
            result: ArenaResult::MaxPlies,
            moves: 1,
            csa_path: relative,
            csa_sha256: saved.sha256,
            csa_size: saved.size,
            special_move: "MAX_MOVES".into(),
            search_nodes: 1,
            search_elapsed_ns: 0,
            depth_sum: 0,
            searches: 1,
            tt_probes: 0,
            tt_hits: 0,
            cutoffs: 0,
            candidate_moves: 0,
            pruned_moves: 0,
            neural_inference_calls: 0,
            neural_inference_time_ns: 0,
            player_a_search_nodes: 1,
            player_a_search_elapsed_ns: 0,
            player_a_depth_sum: 0,
            player_a_searches: 1,
            player_a_neural_inference_calls: 0,
            player_a_neural_inference_time_ns: 0,
            player_b_search_nodes: 0,
            player_b_search_elapsed_ns: 0,
            player_b_depth_sum: 0,
            player_b_searches: 0,
            player_b_neural_inference_calls: 0,
            player_b_neural_inference_time_ns: 0,
            illegal_moves: 0,
            search_winner: false,
            player_a_winner: false,
            player_b_winner: false,
        };
        drop(storage);
        config.resume = true;
        let storage = prepare_output_directory(&config).unwrap();
        assert!(
            validate_resume_artifacts(&config, None, &storage, std::slice::from_ref(&summary))
                .is_ok()
        );

        assert_lossless_elapsed_resume(&config, &storage, &summary);

        let mut impossible_random_metrics = summary.clone();
        impossible_random_metrics.player_b_searches = 1;
        impossible_random_metrics.searches = 1;
        assert!(
            validate_resume_artifacts(&config, None, &storage, &[impossible_random_metrics])
                .is_err(),
            "a random player cannot contribute search metrics"
        );

        std::fs::create_dir(config.output_dir.join("arena-report.json")).unwrap();
        assert!(
            validate_resume_artifacts(&config, None, &storage, std::slice::from_ref(&summary))
                .is_err()
        );
        std::fs::remove_dir(config.output_dir.join("arena-report.json")).unwrap();

        std::fs::write(
            config.output_dir.join("games/game-000002.csa"),
            "crash-window",
        )
        .unwrap();
        assert!(
            validate_resume_artifacts(&config, None, &storage, std::slice::from_ref(&summary))
                .is_err()
        );
        std::fs::remove_file(config.output_dir.join("games/game-000002.csa")).unwrap();

        std::fs::write(
            config.output_dir.join(&summary.csa_path),
            "tampered canonical record",
        )
        .unwrap();
        assert!(validate_resume_artifacts(&config, None, &storage, &[summary]).is_err());
    }

    #[test]
    fn resume_report_accepts_any_exact_durable_prefix_after_repeated_crashes() {
        let mut config = parse_arguments(&[]).unwrap();
        config.games = 3;
        config.max_plies = 1;
        config.output_dir = temporary_arena_directory("resume-report-windows");
        let storage = prepare_output_directory(&config).unwrap();
        let started = "2026-08-13T00:00:00Z";
        let mut games = Vec::new();
        for id in 0..3_u32 {
            let mut summary = pending_game_fixture(&config).summary;
            summary.id = id;
            summary.csa_path = super::expected_csa_path(id);
            games.push(summary);
        }

        assert_eq!(
            validate_resume_report(&storage.root, &config, started, &[]).unwrap(),
            ResumeReportState::Recoverable
        );
        assert_eq!(
            validate_resume_report(&storage.root, &config, started, &games[..1]).unwrap(),
            ResumeReportState::Recoverable
        );
        assert_eq!(
            validate_resume_report(&storage.root, &config, started, &games[..2]).unwrap(),
            ResumeReportState::Recoverable,
            "consecutive state-first crashes may leave no report after game two"
        );
        assert_eq!(
            validate_resume_report(&storage.root, &config, started, &games).unwrap(),
            ResumeReportState::Recoverable,
            "a final state-first crash may leave no completed report"
        );

        write_report(&storage.root, &config, started, None, &games[..2]).unwrap();
        assert_eq!(
            validate_resume_report(&storage.root, &config, started, &games).unwrap(),
            ResumeReportState::Recoverable
        );

        write_report(&storage.root, &config, started, None, &games[..1]).unwrap();
        assert_eq!(
            validate_resume_report(&storage.root, &config, started, &games).unwrap(),
            ResumeReportState::Recoverable,
            "repeated crashes may leave an exact report two games behind durable state"
        );

        write_report(&storage.root, &config, started, None, &[]).unwrap();
        assert_eq!(
            validate_resume_report(&storage.root, &config, started, &games).unwrap(),
            ResumeReportState::Recoverable,
            "an exact empty prefix remains recoverable after repeated crashes"
        );

        write_report(&storage.root, &config, started, None, &games).unwrap();
        assert_eq!(
            validate_resume_report(&storage.root, &config, started, &games).unwrap(),
            ResumeReportState::Recoverable
        );

        let completed = "2026-08-13T00:00:01Z";
        write_report(
            &storage.root,
            &config,
            started,
            Some(completed),
            &games,
        )
        .unwrap();
        let completed_bytes =
            std::fs::read(config.output_dir.join("arena-report.json")).unwrap();
        assert_eq!(
            validate_resume_report(&storage.root, &config, started, &games).unwrap(),
            ResumeReportState::Completed
        );
        assert_eq!(
            validate_resume_report(&storage.root, &config, started, &games).unwrap(),
            ResumeReportState::Completed
        );
        assert_eq!(
            std::fs::read(config.output_dir.join("arena-report.json")).unwrap(),
            completed_bytes,
            "completed resume validation must be byte-idempotent"
        );

        assert!(
            validate_resume_report(&storage.root, &config, started, &games[..2]).is_err(),
            "an incomplete state cannot accept a completed report"
        );

        let mut ahead_config = config.clone();
        ahead_config.games = 4;
        let mut ahead_games = games.clone();
        let mut fourth = pending_game_fixture(&ahead_config).summary;
        fourth.id = 3;
        fourth.csa_path = super::expected_csa_path(3);
        ahead_games.push(fourth);
        let ahead = render_report(&ahead_config, started, None, &ahead_games).unwrap();
        storage
            .root
            .replace_atomic(std::ffi::OsStr::new("arena-report.json"), &ahead)
            .unwrap();
        assert!(validate_resume_report(&storage.root, &config, started, &games).is_err());

        storage
            .root
            .replace_atomic(std::ffi::OsStr::new("arena-report.json"), b"{}")
            .unwrap();
        assert!(validate_resume_report(&storage.root, &config, started, &games).is_err());
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "opening resume covers hit, miss, boundary, and terminal selection accounting"
    )]
    fn opening_resume_replays_hits_misses_boundaries_and_terminal_selections_exactly() {
        let mut config = parse_arguments(&[]).unwrap();
        config.player_a.opening = true;
        config.player_a.opening_profile = OpeningProfile::Unrestricted;
        config.player_b.kind = PlayerKind::HandcraftedExperimental;
        config.player_b.opening = true;
        config.player_b.opening_profile = OpeningProfile::Unrestricted;
        config.opening_max_plies = 2;
        let first = open_shogi_core::parse_usi_move("7g7f").unwrap();
        let second = open_shogi_core::parse_usi_move("3c3d").unwrap();
        let mut after_first = config.initial_position.clone();
        after_first.make_move(first).unwrap();
        let book = opening_book_fixture(&[
            (&config.initial_position, "7g7f"),
            (&after_first, "3c3d"),
        ]);
        let outcome = GameOutcome {
            result: ArenaResult::MaxPlies,
            special: open_shogi_core::CsaSpecialMove::MaxMoves,
            validation: open_shogi_core::CsaResultValidation::ExternalCondition,
        };
        let (_, csa) = encode_csa(
            &config.initial_position,
            &[first, second],
            &config.player_a.label(),
            &config.player_b.label(),
            &outcome,
        )
        .unwrap();
        let parsed = open_shogi_core::parse_csa_game(&csa).unwrap();
        assert_eq!(
            expected_resume_search_counts(
                &config,
                Some(&book),
                &parsed,
                &config.player_a,
                &config.player_b,
                &open_shogi_core::CsaSpecialMove::MaxMoves,
            )
            .unwrap(),
            (0, 0)
        );

        config.opening_max_plies = 1;
        assert_eq!(
            expected_resume_search_counts(
                &config,
                Some(&book),
                &parsed,
                &config.player_a,
                &config.player_b,
                &open_shogi_core::CsaSpecialMove::MaxMoves,
            )
            .unwrap(),
            (0, 1),
            "the opening maximum is checked at each pre-move position"
        );

        let wrong_second = open_shogi_core::parse_usi_move("8c8d").unwrap();
        let (_, wrong_csa) = encode_csa(
            &config.initial_position,
            &[first, wrong_second],
            &config.player_a.label(),
            &config.player_b.label(),
            &outcome,
        )
        .unwrap();
        let wrong = open_shogi_core::parse_csa_game(&wrong_csa).unwrap();
        config.opening_max_plies = 2;
        assert!(
            expected_resume_search_counts(
                &config,
                Some(&book),
                &wrong,
                &config.player_a,
                &config.player_b,
                &open_shogi_core::CsaSpecialMove::MaxMoves,
            )
            .is_err()
        );

        let (_, resign_csa) = encode_csa(
            &config.initial_position,
            &[],
            &config.player_a.label(),
            &config.player_b.label(),
            &GameOutcome {
                result: ArenaResult::WhiteWin,
                special: open_shogi_core::CsaSpecialMove::Resign,
                validation: open_shogi_core::CsaResultValidation::ExternalCondition,
            },
        )
        .unwrap();
        let resign = open_shogi_core::parse_csa_game(&resign_csa).unwrap();
        assert!(
            expected_resume_search_counts(
                &config,
                Some(&book),
                &resign,
                &config.player_a,
                &config.player_b,
                &open_shogi_core::CsaSpecialMove::Resign,
            )
            .is_err(),
            "a terminal selection cannot skip a deterministic legal book hit"
        );
        assert_eq!(
            expected_resume_search_counts(
                &config,
                None,
                &resign,
                &config.player_a,
                &config.player_b,
                &open_shogi_core::CsaSpecialMove::Resign,
            )
            .unwrap(),
            (1, 0)
        );
    }

    fn assert_lossless_elapsed_resume(
        config: &ArenaConfig,
        storage: &super::ArenaStorage,
        summary: &GameSummary,
    ) {
        let mut lossless_elapsed = summary.clone();
        lossless_elapsed.search_elapsed_ns = super::MAX_JSON_SAFE_INTEGER + 1;
        lossless_elapsed.player_a_search_elapsed_ns = super::MAX_JSON_SAFE_INTEGER + 1;
        assert!(
            validate_resume_artifacts(config, None, storage, &[lossless_elapsed]).is_ok(),
            "private v4 nanosecond counters are not constrained by JSON integer precision"
        );
    }

    #[test]
    fn journal_recovers_every_durable_game_commit_failpoint() {
        for stage in 0..=2 {
            let mut config = parse_arguments(&[]).unwrap();
            config.games = 1;
            config.max_plies = 1;
            config.output_dir = temporary_arena_directory(&format!("journal-stage-{stage}"));
            let storage = prepare_output_directory(&config).unwrap();
            let signature = config_signature(&config);
            let started = "2026-08-13T00:00:00Z";
            let pending = pending_game_fixture(&config);
            write_state(&storage.root, &signature, started, &[]).unwrap();
            storage
                .root
                .publish_new_atomic(
                    std::ffi::OsStr::new(super::JOURNAL_FILE_NAME),
                    encode_game_journal(&signature, &pending.summary, &pending.csa).as_bytes(),
                )
                .unwrap();
            if stage >= 1 {
                storage
                    .games
                    .publish_new_atomic(
                        std::ffi::OsStr::new("game-000001.csa"),
                        pending.csa.as_bytes(),
                    )
                    .unwrap();
            }
            let mut summaries = Vec::new();
            if stage >= 2 {
                summaries.push(pending.summary.clone());
                write_state(&storage.root, &signature, started, &summaries).unwrap();
            }

            recover_game_journal(
                &config,
                None,
                &storage,
                &signature,
                started,
                &mut summaries,
            )
            .unwrap();
            assert_eq!(summaries, [pending.summary]);
            assert_eq!(
                std::fs::read(config.output_dir.join("games/game-000001.csa")).unwrap(),
                pending.csa.as_bytes()
            );
            assert!(!config.output_dir.join(super::JOURNAL_FILE_NAME).exists());
            let (_, state) = read_state(&storage.root, &signature).unwrap();
            assert_eq!(state, summaries);
        }
    }

    #[test]
    fn game_commit_rejects_csa_substitution_after_state_or_journal_cleanup() {
        for after_cleanup in [false, true] {
            let label = if after_cleanup {
                "commit-csa-after-cleanup"
            } else {
                "commit-csa-after-state"
            };
            let mut config = parse_arguments(&[]).unwrap();
            config.games = 1;
            config.max_plies = 1;
            config.output_dir = temporary_arena_directory(label);
            let storage = prepare_output_directory(&config).unwrap();
            let signature = config_signature(&config);
            let started = "2026-08-13T00:00:00Z";
            let pending = pending_game_fixture(&config);
            write_state(&storage.root, &signature, started, &[]).unwrap();
            let target = config.output_dir.join(&pending.summary.csa_path);
            let replacement = config.output_dir.join(format!("{label}.foreign"));
            std::fs::write(&replacement, b"foreign arena CSA").unwrap();
            let mut summaries = Vec::new();

            let result = commit_pending_game_with_hooks(
                &config,
                None,
                &storage,
                &signature,
                started,
                &mut summaries,
                &pending,
                |_| {
                    if !after_cleanup {
                        substitute_arena_csa(&target, &replacement)?;
                    }
                    Ok(())
                },
                |_| {
                    if after_cleanup {
                        substitute_arena_csa(&target, &replacement)?;
                    }
                    Ok(())
                },
            );

            assert!(result.is_err(), "{label}");
            assert_eq!(std::fs::read(&target).unwrap(), b"foreign arena CSA");
            assert!(!replacement.exists());
            assert!(!config.output_dir.join(super::JOURNAL_FILE_NAME).exists());
            assert!(!config.output_dir.join("arena-report.json").exists());
            assert_eq!(summaries, [pending.summary]);
            assert!(validate_resume_artifacts(&config, None, &storage, &summaries).is_err());
        }
    }

    #[test]
    fn journal_recovery_rejects_csa_substitution_after_state_or_cleanup() {
        for after_cleanup in [false, true] {
            let label = if after_cleanup {
                "recover-csa-after-cleanup"
            } else {
                "recover-csa-after-state"
            };
            let mut config = parse_arguments(&[]).unwrap();
            config.games = 1;
            config.max_plies = 1;
            config.output_dir = temporary_arena_directory(label);
            let storage = prepare_output_directory(&config).unwrap();
            let signature = config_signature(&config);
            let started = "2026-08-13T00:00:00Z";
            let pending = pending_game_fixture(&config);
            write_state(&storage.root, &signature, started, &[]).unwrap();
            storage
                .root
                .publish_new_atomic(
                    std::ffi::OsStr::new(super::JOURNAL_FILE_NAME),
                    encode_game_journal(&signature, &pending.summary, &pending.csa).as_bytes(),
                )
                .unwrap();
            let target = config.output_dir.join(&pending.summary.csa_path);
            let replacement = config.output_dir.join(format!("{label}.foreign"));
            std::fs::write(&replacement, b"foreign recovered CSA").unwrap();
            let mut summaries = Vec::new();

            let result = recover_game_journal_with_hooks(
                &config,
                None,
                &storage,
                &signature,
                started,
                &mut summaries,
                |_| {
                    if !after_cleanup {
                        substitute_arena_csa(&target, &replacement)?;
                    }
                    Ok(())
                },
                |_| {
                    if after_cleanup {
                        substitute_arena_csa(&target, &replacement)?;
                    }
                    Ok(())
                },
            );

            assert!(result.is_err(), "{label}");
            assert_eq!(std::fs::read(&target).unwrap(), b"foreign recovered CSA");
            assert!(!replacement.exists());
            assert!(!config.output_dir.join(super::JOURNAL_FILE_NAME).exists());
            assert!(!config.output_dir.join("arena-report.json").exists());
            assert_eq!(summaries, [pending.summary]);
            assert!(validate_resume_artifacts(&config, None, &storage, &summaries).is_err());
        }
    }

    #[cfg(unix)]
    #[test]
    fn arena_temp_recovery_preserves_unproven_entries_without_redirecting_or_blocking() {
        use std::os::unix::fs::symlink;

        let mut config = parse_arguments(&[]).unwrap();
        config.output_dir = temporary_arena_directory("temp-recovery");
        let storage = prepare_output_directory(&config).unwrap();
        let external = config.output_dir.with_extension("outside");
        std::fs::write(&external, b"outside").unwrap();
        let malicious = config.output_dir.join(".arena.state.tmp-123-7");
        symlink(&external, &malicious).unwrap();
        recover_owned_temporary_files(&storage).unwrap();
        assert_eq!(std::fs::read(&external).unwrap(), b"outside");
        assert!(
            std::fs::symlink_metadata(&malicious)
                .unwrap()
                .file_type()
                .is_symlink()
        );
        std::fs::remove_file(&malicious).unwrap();

        std::fs::write(&malicious, b"unproven temporary").unwrap();
        let unrelated = config.output_dir.join(".unrelated.tmp-123-7");
        std::fs::write(&unrelated, b"keep").unwrap();
        recover_owned_temporary_files(&storage).unwrap();
        assert_eq!(std::fs::read(&malicious).unwrap(), b"unproven temporary");

        let game_orphan = config.output_dir.join("games/.game-000001.csa.tmp-123-7");
        std::fs::write(&game_orphan, b"unproven game temporary").unwrap();
        recover_owned_temporary_files(&storage).unwrap();
        config.resume = true;
        assert_eq!(std::fs::read(&unrelated).unwrap(), b"keep");
        std::fs::remove_file(&unrelated).unwrap();
        assert!(validate_resume_artifacts(&config, None, &storage, &[]).is_ok());
        assert_eq!(
            std::fs::read(&game_orphan).unwrap(),
            b"unproven game temporary"
        );

        std::fs::remove_file(&malicious).unwrap();
        std::fs::remove_file(&game_orphan).unwrap();

        let target = config.output_dir.join("arena.state");
        std::fs::write(&target, b"published").unwrap();
        let proven = config.output_dir.join(".arena.state.tmp-123-8");
        std::fs::hard_link(&target, &proven).unwrap();
        recover_owned_temporary_files(&storage).unwrap();
        assert!(!proven.exists());
        assert_eq!(std::fs::read(target).unwrap(), b"published");
    }

    #[test]
    fn repeated_pid_style_prelink_orphans_are_preserved_and_bounded() {
        let mut config = parse_arguments(&[]).unwrap();
        config.output_dir = temporary_arena_directory("temp-collisions");
        let storage = prepare_output_directory(&config).unwrap();
        let mut paths = Vec::new();
        for sequence in 0..64 {
            let path = config.output_dir.join(format!(
                ".arena.state.tmp-{}-{sequence}",
                std::process::id()
            ));
            std::fs::write(&path, format!("foreign-{sequence}")).unwrap();
            paths.push(path);
        }

        recover_owned_temporary_files(&storage).unwrap();
        storage
            .root
            .replace_atomic(std::ffi::OsStr::new("arena.state"), b"resumed")
            .unwrap();

        for (sequence, path) in paths.iter().enumerate() {
            assert_eq!(
                std::fs::read_to_string(path).unwrap(),
                format!("foreign-{sequence}")
            );
        }
        assert_eq!(
            std::fs::read(config.output_dir.join("arena.state")).unwrap(),
            b"resumed"
        );
    }

    #[test]
    fn excessive_unproven_temporaries_fail_closed_without_deleting_foreign_entries() {
        let mut config = parse_arguments(&[]).unwrap();
        config.output_dir = temporary_arena_directory("temp-remediation-cap");
        let storage = prepare_output_directory(&config).unwrap();
        let count = super::MAX_PRESERVED_UNPROVEN_TEMPORARIES + 1;
        let mut paths = Vec::with_capacity(count);
        for sequence in 0..count {
            let path = config
                .output_dir
                .join(format!(".arena.state.tmp-4242-{sequence}"));
            std::fs::write(&path, format!("foreign-{sequence}")).unwrap();
            paths.push(path);
        }

        let error = recover_owned_temporary_files(&storage).unwrap_err();

        assert!(error.contains("remove them manually after inspection"));
        for (sequence, path) in paths.iter().enumerate() {
            assert_eq!(
                std::fs::read_to_string(path).unwrap(),
                format!("foreign-{sequence}")
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn private_quarantine_and_public_crash_remnants_share_one_arena_cap() {
        use std::os::unix::fs::PermissionsExt as _;

        let mut config = parse_arguments(&[]).unwrap();
        config.output_dir = temporary_arena_directory("combined-remediation-cap");
        let storage = prepare_output_directory(&config).unwrap();
        let cleanup = config
            .output_dir
            .join(super::SECURE_CLEANUP_DIRECTORY_NAME);
        std::fs::create_dir(&cleanup).unwrap();
        std::fs::set_permissions(&cleanup, std::fs::Permissions::from_mode(0o700)).unwrap();
        let private = cleanup.join(".entry-4242-0");
        std::fs::write(&private, b"private-foreign").unwrap();

        let mut public = Vec::new();
        for sequence in 0..super::MAX_PRESERVED_UNPROVEN_TEMPORARIES {
            let path = config
                .output_dir
                .join(format!(".arena.state.tmp-4242-{sequence}"));
            std::fs::write(&path, format!("public-foreign-{sequence}")).unwrap();
            public.push(path);
        }

        let error = recover_owned_temporary_files(&storage).unwrap_err();

        assert!(error.contains("total unproven"), "{error}");
        assert_eq!(std::fs::read(&private).unwrap(), b"private-foreign");
        for (sequence, path) in public.iter().enumerate() {
            assert_eq!(
                std::fs::read_to_string(path).unwrap(),
                format!("public-foreign-{sequence}")
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn arena_parent_swap_cannot_redirect_state_outside_the_anchored_directory() {
        use std::os::unix::fs::symlink;

        let mut config = parse_arguments(&[]).unwrap();
        config.output_dir = temporary_arena_directory("parent-swap");
        let storage = prepare_output_directory(&config).unwrap();
        let moved = config.output_dir.with_extension("anchored");
        let external = config.output_dir.with_extension("external");
        std::fs::create_dir_all(&external).unwrap();
        std::fs::rename(&config.output_dir, &moved).unwrap();
        symlink(&external, &config.output_dir).unwrap();

        write_state(
            &storage.root,
            &config_signature(&config),
            "2026-08-13T00:00:00Z",
            &[],
        )
        .unwrap();
        assert!(moved.join("arena.state").exists());
        assert!(!external.join("arena.state").exists());
    }

    fn pending_game_fixture(config: &ArenaConfig) -> PendingGame {
        let black = config.player_a.label();
        let white = config.player_b.label();
        let movement = open_shogi_core::parse_usi_move("7g7f").unwrap();
        let outcome = GameOutcome {
            result: ArenaResult::MaxPlies,
            special: open_shogi_core::CsaSpecialMove::MaxMoves,
            validation: open_shogi_core::CsaResultValidation::ExternalCondition,
        };
        let (saved, csa) = encode_csa(
            &config.initial_position,
            &[movement],
            &black,
            &white,
            &outcome,
        )
        .unwrap();
        PendingGame {
            summary: GameSummary {
                id: 0,
                black,
                white,
                result: ArenaResult::MaxPlies,
                moves: 1,
                csa_path: "games/game-000001.csa".to_owned(),
                csa_sha256: saved.sha256,
                csa_size: saved.size,
                special_move: "MAX_MOVES".to_owned(),
                search_nodes: 1,
                search_elapsed_ns: 0,
                depth_sum: 0,
                searches: 1,
                tt_probes: 0,
                tt_hits: 0,
                cutoffs: 0,
                candidate_moves: 0,
                pruned_moves: 0,
                neural_inference_calls: 0,
                neural_inference_time_ns: 0,
                player_a_search_nodes: 1,
                player_a_search_elapsed_ns: 0,
                player_a_depth_sum: 0,
                player_a_searches: 1,
                player_a_neural_inference_calls: 0,
                player_a_neural_inference_time_ns: 0,
                player_b_search_nodes: 0,
                player_b_search_elapsed_ns: 0,
                player_b_depth_sum: 0,
                player_b_searches: 0,
                player_b_neural_inference_calls: 0,
                player_b_neural_inference_time_ns: 0,
                illegal_moves: 0,
                search_winner: false,
                player_a_winner: false,
                player_b_winner: false,
            },
            csa,
        }
    }

    fn substitute_arena_csa(target: &std::path::Path, replacement: &std::path::Path) -> Result<(), String> {
        std::fs::remove_file(target).map_err(|error| error.to_string())?;
        std::fs::rename(replacement, target).map_err(|error| error.to_string())
    }

    #[test]
    fn state_hex_is_strict_and_reversible() {
        assert_eq!(hex_decode(&hex_encode(b"a b")).unwrap(), b"a b");
        assert!(hex_decode("0x").is_err());
    }

    #[test]
    fn state_unsigned_decimal_fields_have_one_canonical_spelling() {
        assert_eq!(parse_state::<u64>("0", "fixture").unwrap(), 0);
        assert_eq!(parse_state::<u64>("42", "fixture").unwrap(), 42);
        for value in ["+0", "00", "01", "+42", "-0"] {
            assert!(parse_state::<u64>(value, "fixture").is_err(), "{value}");
        }
    }

    #[test]
    fn signature_includes_behavioral_configuration() {
        let position = open_shogi_core::Position::startpos();
        let base = ArenaConfig {
            games: 2,
            player_a: PlayerSpec {
                kind: PlayerKind::HandcraftedExperimental,
                depth: 2,
                hash_megabytes: 1,
                transposition: true,
                model_path: None,
                model_sha256: None,
                model_size: None,
                model: None,
                opening: false,
                opening_profile: OpeningProfile::IbishaStrict,
            },
            player_b: PlayerSpec {
                kind: PlayerKind::Random,
                depth: 2,
                hash_megabytes: 1,
                transposition: true,
                model_path: None,
                model_sha256: None,
                model_size: None,
                model: None,
                opening: false,
                opening_profile: OpeningProfile::IbishaStrict,
            },
            budget: Budget::Nodes(1),
            initial_position: position,
            initial_sfen: "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1".into(),
            max_plies: 2,
            seed: 1,
            git_commit: None,
            output_dir: PathBuf::from("ignored"),
            resume: false,
            opening_book_path: None,
            opening_book_sha256: None,
            opening_book_size: None,
            opening_max_plies: 24,
        };
        let mut changed = base.clone();
        changed.max_plies = 3;
        assert_ne!(config_signature(&base), config_signature(&changed));
    }

    #[test]
    fn config_signature_has_a_language_neutral_known_byte_and_hash_vector() {
        let signature = super::ArenaConfigSignature {
            games: 2,
            seed: 7,
            initial_sfen: "fixture sfen".to_owned(),
            max_plies: 3,
            git_commit: Some("abc".to_owned()),
            budget_kind: "nodes".to_owned(),
            budget_value: 500,
            player_a: super::ArenaConfigSignaturePlayer {
                label: "A".to_owned(),
                evaluator_kind: "neural".to_owned(),
                search_depth: Some(4),
                hash_megabytes: Some(16),
                transposition: Some(true),
                model_artifact_sha256: Some("00".repeat(32)),
                model_artifact_size: Some(123),
                model_payload_sha256: Some("11".repeat(32)),
                architecture_version: Some(1),
                quantization: Some("int8".to_owned()),
                opening_enabled: true,
            },
            player_b: super::ArenaConfigSignaturePlayer {
                label: "B".to_owned(),
                evaluator_kind: "random".to_owned(),
                search_depth: None,
                hash_megabytes: None,
                transposition: None,
                model_artifact_sha256: None,
                model_artifact_size: None,
                model_payload_sha256: None,
                architecture_version: None,
                quantization: None,
                opening_enabled: false,
            },
            opening: super::ArenaConfigSignatureOpening {
                enabled: true,
                artifact_sha256: Some("22".repeat(32)),
                artifact_size: Some(456),
                max_plies: Some(24),
            },
        };
        let expected = concat!(
            "{\"schema\":\"phase2_arena_config_signature/v1\",\"games\":2,\"seed\":7,",
            "\"initialSfen\":\"fixture sfen\",\"maxPlies\":3,\"gitCommit\":\"abc\",",
            "\"budget\":{\"kind\":\"nodes\",\"value\":500},\"playerA\":{",
            "\"label\":\"A\",\"evaluatorKind\":\"neural\",\"searchDepth\":4,",
            "\"hashMegabytes\":16,\"transposition\":true,\"modelArtifactSha256\":\"",
            "0000000000000000000000000000000000000000000000000000000000000000\",",
            "\"modelArtifactSize\":123,\"modelPayloadSha256\":\"",
            "1111111111111111111111111111111111111111111111111111111111111111\",",
            "\"architectureVersion\":1,\"quantization\":\"int8\",\"openingEnabled\":true},",
            "\"playerB\":{\"label\":\"B\",\"evaluatorKind\":\"random\",",
            "\"searchDepth\":null,\"hashMegabytes\":null,\"transposition\":null,",
            "\"modelArtifactSha256\":null,\"modelArtifactSize\":null,",
            "\"modelPayloadSha256\":null,\"architectureVersion\":null,",
            "\"quantization\":null,\"openingEnabled\":false},\"opening\":{",
            "\"enabled\":true,\"artifactSha256\":\"",
            "2222222222222222222222222222222222222222222222222222222222222222\",",
            "\"artifactSize\":456,\"maxPlies\":24}}"
        );
        let bytes = super::arena_config_signature_bytes(&signature);
        assert_eq!(bytes, expected.as_bytes());
        assert_eq!(
            crate::checksum::sha256_bytes(&bytes),
            "da23e594633deefb8b329a1202045e4f4c559eeb1c048ccec83f6aa7913d9848"
        );
    }

    #[test]
    fn utc_timestamp_conversion_handles_epoch_and_leap_day() {
        assert_eq!(timestamp_from_unix(0), "1970-01-01T00:00:00Z");
        assert_eq!(timestamp_from_unix(951_782_400), "2000-02-29T00:00:00Z");
        assert!(is_utc_timestamp("2000-02-29T23:59:59Z"));
        assert!(!is_utc_timestamp("1900-02-29T00:00:00Z"));
        assert!(!is_utc_timestamp("2026-04-31T00:00:00Z"));
        assert!(!is_utc_timestamp("0000-01-01T00:00:00Z"));
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "this schema test enumerates every closed v2 key in one reviewable assertion"
    )]
    fn report_uses_lab_schema_and_does_not_count_max_plies_as_draw() {
        let directory = temporary_arena_directory("report-schema");
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("arena-report.json");
        let root = AnchoredDir::open_existing(&directory).unwrap();
        let model_path = directory.join("model.osaval");
        let model_artifact = model_bytes();
        let expected_artifact_sha256 = format!("{:x}", Sha256::digest(&model_artifact));
        std::fs::write(&model_path, model_artifact).unwrap();
        let mut config = parse_arguments(&[
            "--player-a".into(),
            "neural".into(),
            "--a-model".into(),
            model_path.to_string_lossy().into_owned(),
        ])
        .unwrap();
        std::fs::remove_file(model_path).unwrap();
        config.git_commit = Some("abcdef012345".into());
        let game = GameSummary {
            id: 0,
            black: config.player_a.label(),
            white: config.player_b.label(),
            result: ArenaResult::MaxPlies,
            moves: 2,
            csa_path: "games/game-000001.csa".into(),
            csa_sha256: "a".repeat(64),
            csa_size: 100,
            special_move: "MAX_MOVES".into(),
            search_nodes: 10,
            search_elapsed_ns: 2_000_000,
            depth_sum: 3,
            searches: 1,
            tt_probes: 5,
            tt_hits: 2,
            cutoffs: 1,
            candidate_moves: 8,
            pruned_moves: 3,
            neural_inference_calls: 2,
            neural_inference_time_ns: 300,
            player_a_search_nodes: 10,
            player_a_search_elapsed_ns: 2_000_000,
            player_a_depth_sum: 3,
            player_a_searches: 1,
            player_a_neural_inference_calls: 2,
            player_a_neural_inference_time_ns: 300,
            player_b_search_nodes: 0,
            player_b_search_elapsed_ns: 0,
            player_b_depth_sum: 0,
            player_b_searches: 0,
            player_b_neural_inference_calls: 0,
            player_b_neural_inference_time_ns: 0,
            illegal_moves: 0,
            search_winner: false,
            player_a_winner: false,
            player_b_winner: false,
        };
        write_report(
            &root,
            &config,
            "2026-07-29T00:00:00Z",
            Some("2026-07-29T00:01:00Z"),
            &[game],
        )
        .unwrap();
        let report = std::fs::read_to_string(path).unwrap();
        assert!(report.starts_with("{\"schema\":\"phase2_arena_report/v2\""));
        assert!(report.contains("\"gitCommit\":\"abcdef012345\""));
        assert!(report.contains("\"draws\":0"));
        assert!(report.contains("\"playerAWins\":0"));
        assert!(report.contains("\"playerBWins\":0"));
        assert!(report.contains("\"pruningRate\":0.375000"));
        assert!(report.contains("\"neuralInferenceCalls\":2"));
        assert!(report.contains("\"neuralInferenceTimeNs\":300"));
        assert!(report.contains("\"playerASearchNodes\":10"));
        assert!(report.contains("\"playerBSearchNodes\":0"));
        assert!(report.contains("\"result\":\"max_plies\""));
        assert!(report.contains("\"peakMemoryBytes\":null"));

        let value: serde_json::Value = serde_json::from_str(&report).unwrap();
        assert_eq!(
            object_keys(&value),
            key_set(&["games", "metrics", "run", "schema"])
        );
        let run = &value["run"];
        assert_eq!(
            object_keys(run),
            key_set(&[
                "budget",
                "completedAt",
                "configSha256",
                "engine",
                "gameLimit",
                "gitCommit",
                "initialSfen",
                "maxPlies",
                "opening",
                "playerA",
                "playerB",
                "seed",
                "startedAt",
            ])
        );
        let player_keys = key_set(&[
            "architectureVersion",
            "evaluatorKind",
            "hashMegabytes",
            "label",
            "modelArtifactSha256",
            "modelArtifactSize",
            "modelPayloadSha256",
            "openingEnabled",
            "quantization",
            "searchDepth",
            "transposition",
        ]);
        assert_eq!(object_keys(&run["playerA"]), player_keys);
        assert_eq!(object_keys(&run["playerB"]), player_keys);
        assert_eq!(run["playerA"]["evaluatorKind"], "neural");
        assert_eq!(
            run["playerA"]["modelArtifactSha256"],
            expected_artifact_sha256
        );
        assert_eq!(run["playerA"]["architectureVersion"], 1);
        assert_eq!(run["playerA"]["quantization"], "float32");
        assert_eq!(
            run["playerB"]["modelArtifactSha256"],
            serde_json::Value::Null
        );
        assert_eq!(
            object_keys(&run["opening"]),
            key_set(&["artifactSha256", "artifactSize", "enabled", "maxPlies"])
        );
        assert_eq!(run["opening"]["enabled"], false);
        assert_eq!(run["opening"]["artifactSha256"], serde_json::Value::Null);
        assert_eq!(run["configSha256"].as_str().unwrap().len(), 64);
        assert_eq!(
            object_keys(&value["metrics"]),
            key_set(&[
                "averageDepth",
                "cutoffRate",
                "draws",
                "finishedGames",
                "games",
                "illegalMoves",
                "millisecondsPerMove",
                "neuralInferenceCalls",
                "neuralInferenceTimeNs",
                "nodesPerSecond",
                "peakMemoryBytes",
                "playerADepthSum",
                "playerANeuralInferenceCalls",
                "playerANeuralInferenceTimeNs",
                "playerASearchElapsedMs",
                "playerASearchNodes",
                "playerASearches",
                "playerAWins",
                "playerBDepthSum",
                "playerBNeuralInferenceCalls",
                "playerBNeuralInferenceTimeNs",
                "playerBSearchElapsedMs",
                "playerBSearchNodes",
                "playerBSearches",
                "playerBWins",
                "pruningRate",
                "searchWins",
                "ttHitRate",
            ])
        );
        assert_eq!(
            object_keys(&value["games"][0]),
            key_set(&[
                "black",
                "csaPath",
                "csaSha256",
                "csaSize",
                "id",
                "moves",
                "neuralInferenceCalls",
                "neuralInferenceTimeNs",
                "playerADepthSum",
                "playerANeuralInferenceCalls",
                "playerANeuralInferenceTimeNs",
                "playerASearchElapsedMs",
                "playerASearchNodes",
                "playerASearches",
                "playerBDepthSum",
                "playerBNeuralInferenceCalls",
                "playerBNeuralInferenceTimeNs",
                "playerBSearchElapsedMs",
                "playerBSearchNodes",
                "playerBSearches",
                "result",
                "white",
            ])
        );
    }

    #[test]
    fn terminal_state_on_last_allowed_ply_takes_precedence_over_cap() {
        let outcome = final_outcome(
            None,
            Some(open_shogi_core::GameEnd::Checkmate {
                winner: open_shogi_core::Side::White,
            }),
        );
        assert_eq!(outcome.result, ArenaResult::WhiteWin);
        assert_eq!(outcome.special, open_shogi_core::CsaSpecialMove::Checkmate);
    }

    #[test]
    fn resume_result_replay_rejects_terminal_claims_the_arena_never_emits() {
        let unsupported_draw = open_shogi_core::CsaGame {
            version: "V3.0".to_owned(),
            black_name: Some("a".to_owned()),
            white_name: Some("b".to_owned()),
            metadata: Vec::new(),
            initial_position: open_shogi_core::Position::startpos(),
            moves: Vec::new(),
            special_move: Some(open_shogi_core::CsaSpecialMove::Draw),
            result_validation: open_shogi_core::CsaResultValidation::ExternalCondition,
        };
        assert!(replayed_csa_result(&unsupported_draw).is_err());

        let checkmate = open_shogi_core::parse_sfen("3lkl3/3pRp3/4G4/9/9/9/9/9/K8 w - 1").unwrap();
        let false_max_moves = open_shogi_core::CsaGame {
            initial_position: checkmate,
            special_move: Some(open_shogi_core::CsaSpecialMove::MaxMoves),
            ..unsupported_draw
        };
        assert!(replayed_csa_result(&false_max_moves).is_err());
    }

    fn temporary_arena_directory(label: &str) -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().canonicalize().unwrap().join(format!(
            "open-shogi-arena-{label}-{}-{nonce}",
            std::process::id()
        ))
    }

    fn opening_book_fixture(
        records: &[(&open_shogi_core::Position, &str)],
    ) -> crate::opening::OpeningBook {
        let mut gzip = GzEncoder::new(Vec::new(), Compression::fast());
        for (position, movement) in records {
            let complete = open_shogi_core::to_sfen(position);
            let state = complete
                .rsplit_once(' ')
                .expect("canonical SFEN has a move number")
                .0;
            let provenance = ["1".repeat(64)];
            let mut record = serde_json::json!({
                "schema": "open_shogi_opening_book/v2",
                "stateKey": crate::checksum::sha256_text(state),
                "stateSfen": state,
                "ruleProfile": "standard-shogi/v1",
                "buildVersion": "arena-test-v2",
                "provenanceReferences": provenance,
                "candidates": [{
                    "moveUsi": movement,
                    "sampleCount": 2,
                    "sourceDistribution": {"fixture": 2},
                    "blackResults": {"wins": 1, "losses": 1, "draws": 0, "unknown": 0},
                    "whiteResults": {"wins": 0, "losses": 0, "draws": 0, "unknown": 0},
                    "teacherScoreCp": 0,
                    "scoreUncertaintyCp": 0,
                    "teacherDepth": 1,
                    "teacherNodes": 1,
                    "openingClassification": "ibisha",
                    "provenanceReferences": provenance,
                }],
            });
            let checksum = crate::checksum::sha256_bytes(&serde_json::to_vec(&record).unwrap());
            record
                .as_object_mut()
                .unwrap()
                .insert("recordChecksum".to_owned(), serde_json::json!(checksum));
            serde_json::to_writer(&mut gzip, &record).unwrap();
            gzip.write_all(b"\n").unwrap();
        }
        crate::opening::OpeningBook::from_compressed_bytes(&gzip.finish().unwrap()).unwrap()
    }

    fn key_set(values: &[&'static str]) -> BTreeSet<&'static str> {
        values.iter().copied().collect()
    }

    fn object_keys(value: &serde_json::Value) -> BTreeSet<&str> {
        value
            .as_object()
            .expect("JSON object")
            .keys()
            .map(String::as_str)
            .collect()
    }

    fn model_bytes() -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"OSAVAL01");
        for value in [1_u32, 1, 1, 1 << 2, 1, 1, 1, 0, 0, 2] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes.extend_from_slice(&10.0_f32.to_le_bytes());
        for weight in [2.0_f32, 3.0] {
            bytes.extend_from_slice(&1_u32.to_le_bytes());
            bytes.extend_from_slice(&1_u32.to_le_bytes());
            bytes.extend_from_slice(&weight.to_le_bytes());
            bytes.extend_from_slice(&0.0_f32.to_le_bytes());
        }
        let checksum = Sha256::digest(&bytes);
        bytes.extend_from_slice(&checksum);
        bytes
    }
}
