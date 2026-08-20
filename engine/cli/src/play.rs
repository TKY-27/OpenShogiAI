use std::{
    collections::BTreeMap,
    io::{self, BufRead, Write},
    path::{Component, Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};

use open_shogi_core::{
    AnchoredDir, AnchoredFile, CancellationToken, CsaGame, CsaResultValidation, CsaSpecialMove,
    EntryKind, EvaluationConfig, Game, GameEnd, HandPiece, MAX_NEURAL_MODEL_BYTES, Move,
    NeuralEvaluationMode, NeuralEvaluator, NeuralQuantization, Position, RepetitionOutcome,
    SearchConfig, SearchEngine, Side, StableDirectoryIdentity, StableFileIdentity, TimeControl,
    TimeManager, parse_csa_game, parse_sfen, parse_usi_move, repetition_outcome_from_moves,
    to_csa_game, to_sfen, to_usi_move,
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use sha2::{Digest, Sha256};

use crate::{
    args::{BoundedInputLine, next_value, parse_next, path_next, read_bounded_line},
    arena::{
        ArenaConfigSignature, ArenaConfigSignatureOpening, ArenaConfigSignaturePlayer,
        arena_config_signature_bytes,
    },
    checksum::{
        FileArtifact, read_file_artifact, read_open_file_artifact,
        read_open_file_artifact_with_check, read_retained_file_artifact, sha256_bytes,
        sha256_open_file_with_check, sha256_text,
    },
    opening::{OpeningBook, OpeningPolicy, OpeningProfile},
};

const DEFAULT_MAX_PLIES: u32 = 512;
const MAX_NODES_PER_MOVE: u64 = 1_000_000_000;
const MAX_MOVETIME_MS: u64 = 3_600_000;
const MAX_PLIES: u32 = 10_000;
const MAX_REGISTRY_BYTES: u64 = 16 * 1024 * 1024;
const MAX_PROMOTION_POLICY_BYTES: u64 = 64 * 1024;
const MAX_REGISTRY_ARTIFACT_BYTES: u64 = 4 * 1024 * 1024 * 1024;
const MAX_OPENING_BYTES: u64 = 64 * 1024 * 1024;
const MAX_REGISTRY_ITEMS: usize = 10_000;
const MAX_REGISTRY_VERIFICATION_REFERENCES: usize = 10_000;
const MAX_REGISTRY_VERIFICATION_BYTES: u64 = 1024 * 1024 * 1024;
const MAX_REGISTRY_VERIFICATION_DURATION: Duration = Duration::from_secs(30);
const MAX_REGISTRY_JSON_DEPTH: usize = 32;
const MAX_REGISTRY_JSON_NODES: usize = 250_000;
const MAX_PAIRED_EXECUTION_ATTEMPTS: usize = 20_000;
const MAX_PAIRED_COMMAND_ARGUMENTS: usize = 256;
const MAX_PAIRED_COMMAND_ARGUMENT_BYTES: usize = 16 * 1024;
const MAX_PHASE2_ARENA_REPORT_BYTES: u64 = 5 * 1024 * 1024;
const MAX_QUARANTINE_RECORD_BYTES: u64 = 1024 * 1024;
const MAX_PHASE6_COMMAND_RECEIPT_BYTES: u64 = 1024 * 1024;
const MAX_PHASE3_POSITIONS: usize = 250_000;
const MAX_PHASE3_COMPRESSED_BYTES: u64 = 1024 * 1024 * 1024;
const MAX_PHASE3_UNCOMPRESSED_BYTES: u64 = 1024 * 1024 * 1024;
const MAX_PHASE3_LINE_BYTES: u64 = 64 * 1024;
const MAX_PHASE3_GAME_LINE_BYTES: u64 = 32 * 1024 * 1024;
const MAX_PHASE3_CSA_BYTES: usize = 1024 * 1024;
const MAX_AOBAZERO_CSA_LINES: usize = 4_096;
const MAX_AOBAZERO_CSA_LINE_BYTES: usize = 4_096;
const MAX_AOBAZERO_CSA_MOVES: usize = 2_048;
const MAX_AOBAZERO_CSA_COMMENTS: usize = 2_048;
const MAX_AOBAZERO_CSA_COMMENT_BYTES: usize = 262_144;
const MAX_AOBAZERO_CSA_ANNOTATION_BYTES: usize = 524_288;
const MAX_AOBAZERO_CSA_NAME_BYTES: usize = 512;
const MAX_AOBAZERO_CSA_METADATA_ENTRIES: usize = 128;
const MAX_AOBAZERO_CSA_METADATA_VALUE_BYTES: usize = 2_048;
const PHASE6_ARENA_GAMES: u64 = 40;
const PHASE6_ARENA_PAIRS: u64 = 20;
const PHASE6_INITIAL_PAIRS: u64 = 10;
const PHASE6_START_SET_PAIRS: u64 = 10;
const PHASE6_ARENA_SEED: u64 = 20_260_808;
const PHASE6_ARENA_NODES_PER_MOVE: u64 = 500;
const PHASE6_ARENA_MAX_PLIES: u64 = 256;
const PHASE6_ARENA_WORKERS: u64 = 2;
const PHASE6_ARENA_MEMORY_MIB: u64 = 8 * 1024;
const PHASE6_INITIAL_SFEN: &str =
    "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";
const MAX_IDENTIFIER_BYTES: usize = 96;
const MAX_JSON_SAFE_INTEGER: u64 = 9_007_199_254_740_991;
const MAX_HUMAN_CSA_BYTES: u64 = 16 * 1024 * 1024;
const MAX_HUMAN_DECISION_BYTES: u64 = 64 * 1024 * 1024;

struct RegistryStorage {
    root: AnchoredDir,
    root_path: PathBuf,
    registry_relative: PathBuf,
}

#[derive(Clone)]
struct VerifiedModelMetadata {
    architecture_version: u32,
    quantization: NeuralQuantization,
    payload_sha256: String,
}

#[derive(Clone)]
struct ObservedRegistryArtifact {
    sha256: String,
    size: u64,
    model: Option<VerifiedModelMetadata>,
    bytes: Option<Arc<Vec<u8>>>,
}

struct RegistryVerificationBudget {
    references: usize,
    unique_bytes: u64,
    started: Instant,
    maximum_references: usize,
    maximum_unique_bytes: u64,
    maximum_duration: Duration,
}

impl RegistryVerificationBudget {
    fn new() -> Self {
        Self {
            references: 0,
            unique_bytes: 0,
            started: Instant::now(),
            maximum_references: MAX_REGISTRY_VERIFICATION_REFERENCES,
            maximum_unique_bytes: MAX_REGISTRY_VERIFICATION_BYTES,
            maximum_duration: MAX_REGISTRY_VERIFICATION_DURATION,
        }
    }

    #[cfg(test)]
    fn with_limits(maximum_references: usize, maximum_unique_bytes: u64) -> Self {
        Self {
            references: 0,
            unique_bytes: 0,
            started: Instant::now(),
            maximum_references,
            maximum_unique_bytes,
            maximum_duration: MAX_REGISTRY_VERIFICATION_DURATION,
        }
    }

    fn charge_reference(&mut self) -> Result<(), String> {
        self.ensure_time()?;
        self.references = self
            .references
            .checked_add(1)
            .ok_or_else(|| "registry verification reference count overflow".to_owned())?;
        if self.references > self.maximum_references {
            return Err(format!(
                "registry verification exceeds {} artifact references",
                self.maximum_references
            ));
        }
        Ok(())
    }

    fn charge_unique_bytes(&mut self, bytes: u64) -> Result<(), String> {
        self.ensure_time()?;
        self.unique_bytes = self
            .unique_bytes
            .checked_add(bytes)
            .ok_or_else(|| "registry verification byte count overflow".to_owned())?;
        if self.unique_bytes > self.maximum_unique_bytes {
            return Err(format!(
                "registry verification exceeds {} unique artifact bytes",
                self.maximum_unique_bytes
            ));
        }
        Ok(())
    }

    fn ensure_time(&self) -> Result<(), String> {
        if self.started.elapsed() > self.maximum_duration {
            return Err(format!(
                "registry verification exceeded its {} second time budget",
                self.maximum_duration.as_secs()
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PlayProfile {
    Material,
    HandcraftedBaseline,
    HandcraftedExperimental,
    OverallChampion,
    Neural,
    Residual,
    Composite,
    GenerationZero,
    NeuralLineageChampion,
    Challenger,
}

impl PlayProfile {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "material" => Ok(Self::Material),
            "handcrafted" | "handcrafted-baseline" => Ok(Self::HandcraftedBaseline),
            "experimental" | "handcrafted-experimental" => Ok(Self::HandcraftedExperimental),
            "champion" | "overall-champion" => Ok(Self::OverallChampion),
            "neural" => Ok(Self::Neural),
            "residual" => Ok(Self::Residual),
            "composite" => Ok(Self::Composite),
            "generation-0" => Ok(Self::GenerationZero),
            "neural-lineage-champion" => Ok(Self::NeuralLineageChampion),
            "challenger" => Ok(Self::Challenger),
            _ => Err(
                "--profile must be material, handcrafted-baseline, handcrafted-experimental, overall-champion, neural, residual, composite, generation-0, neural-lineage-champion, or challenger"
                    .to_owned(),
            ),
        }
    }

    const fn is_registry_alias(self) -> bool {
        matches!(
            self,
            Self::GenerationZero | Self::NeuralLineageChampion | Self::Challenger
        )
    }

    const fn uses_direct_model(self) -> bool {
        matches!(self, Self::Neural | Self::Residual | Self::Composite)
    }
}

#[derive(Clone, Copy, Debug)]
enum Budget {
    Casual,
    Nodes(u64),
    MoveTime(u64),
    Clock(ClockBudget),
}

#[derive(Clone, Copy, Debug)]
struct ClockBudget {
    black_time: u64,
    white_time: u64,
    byoyomi: Option<u64>,
    black_increment: Option<u64>,
    white_increment: Option<u64>,
}

struct PlayConfig {
    human: Side,
    budget: Budget,
    safety_margin_ms: u64,
    depth: u8,
    initial: Position,
    max_plies: u32,
    output: PathBuf,
    decision_log: PathBuf,
    profile: PlayProfile,
    model_path: Option<PathBuf>,
    registry_path: Option<PathBuf>,
    opening_book_path: Option<PathBuf>,
    opening_max_plies: u32,
    opening_profile: OpeningProfile,
    opening_minimum_samples: u64,
    opening_maximum_teacher_loss_cp: i32,
}

enum HumanTurn {
    Played(Move),
    Ended(CsaSpecialMove, CsaResultValidation),
}

struct ResolvedProfile {
    model_id: String,
    model_artifact_sha256: Option<String>,
    model_payload_sha256: Option<String>,
    architecture_version: Option<u32>,
    quantization: Option<NeuralQuantization>,
    neural: Option<Arc<NeuralEvaluator>>,
    registry_sha256: Option<String>,
    registry_revision: Option<u64>,
}

struct ResolvedOpening {
    book: OpeningBook,
    artifact_sha256: String,
    artifact_size: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct ModelRegistry {
    schema: String,
    revision: u64,
    #[serde(deserialize_with = "deserialize_required_option")]
    champion_model_id: Option<String>,
    #[serde(deserialize_with = "deserialize_required_option")]
    challenger_model_id: Option<String>,
    models: Vec<RegistryModel>,
    generations: Vec<RegistryGeneration>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct RegistryModel {
    model_id: String,
    generation_id: String,
    #[serde(deserialize_with = "deserialize_required_option")]
    parent_model_id: Option<String>,
    artifact: RegistryArtifact,
    evaluator_kind: String,
    architecture_version: String,
    quantization: String,
    #[serde(deserialize_with = "deserialize_required_option")]
    training_run: Option<RegistryArtifact>,
    registered_at: String,
    license_status: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct RegistryGeneration {
    generation_id: String,
    #[serde(deserialize_with = "deserialize_required_option")]
    parent_generation_id: Option<String>,
    champion_model_id: String,
    #[serde(deserialize_with = "deserialize_required_option")]
    challenger_model_id: Option<String>,
    #[serde(deserialize_with = "deserialize_required_option")]
    selfplay_manifest: Option<RegistryArtifact>,
    #[serde(deserialize_with = "deserialize_required_option")]
    teacher_labeling_manifest: Option<RegistryArtifact>,
    #[serde(deserialize_with = "deserialize_required_option")]
    training_run_manifest: Option<RegistryArtifact>,
    #[serde(deserialize_with = "deserialize_required_option")]
    arena_manifest: Option<RegistryArtifact>,
    #[serde(deserialize_with = "deserialize_required_option")]
    promotion_decision: Option<RegistryArtifact>,
    status: String,
    created_at: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct RegistryArtifact {
    path: PathBuf,
    sha256: String,
    size: u64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct PromotionDecision {
    schema: String,
    generation_id: String,
    champion_model_id: String,
    challenger_model_id: String,
    arena_analysis: RegistryArtifact,
    policy: RegistryArtifact,
    policy_sha256: String,
    decision: String,
    weak_evidence: bool,
    reasons: Vec<String>,
    evidence: PromotionEvidence,
    decided_at: String,
    decision_sha256: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct PromotionEvidence {
    games: u64,
    decisive_games: u64,
    score_rate: f64,
    score_wilson95: PromotionWilsonInterval,
    initial_score_rate: f64,
    start_set_score_rate: f64,
    #[serde(deserialize_with = "deserialize_required_option")]
    side_score_gap: Option<f64>,
    illegal_moves: u64,
    crashes: u64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct PromotionWilsonInterval {
    lower: f64,
    upper: f64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct ArenaAnalysisEnvelope {
    schema: String,
    generation_id: String,
    results: RegistryArtifact,
    plan: RegistryArtifact,
    execution: RegistryArtifact,
    champion_model_id: String,
    challenger_model_id: String,
    method: ArenaAnalysisMethod,
    overall: ArenaSummary,
    by_start_group: ArenaStartGroups,
    by_challenger_side: ArenaChallengerSides,
    #[serde(deserialize_with = "deserialize_required_option")]
    side_score_gap: Option<f64>,
    metrics: ArenaAggregateMetrics,
    analysis_sha256: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct ArenaAnalysisMethod {
    pairing: String,
    score: String,
    wilson95: String,
    elo: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct ArenaStartGroups {
    initial: ArenaSummary,
    start_set: ArenaSummary,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct ArenaChallengerSides {
    black: ArenaSummary,
    white: ArenaSummary,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct ArenaSummary {
    games: u64,
    wins: u64,
    losses: u64,
    draws: u64,
    max_plies: u64,
    decisive_games: u64,
    score_rate: f64,
    decisive_win_rate: f64,
    draw_rate: f64,
    score_wilson95: PromotionWilsonInterval,
    #[serde(deserialize_with = "deserialize_required_option")]
    approximate_elo: Option<f64>,
    average_plies: f64,
    illegal_moves: u64,
    crashes: u64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct ArenaAggregateMetrics {
    #[serde(deserialize_with = "deserialize_required_option")]
    champion_inference_calls_per_second: Option<f64>,
    #[serde(deserialize_with = "deserialize_required_option")]
    challenger_inference_calls_per_second: Option<f64>,
    #[serde(deserialize_with = "deserialize_required_option")]
    inference_slowdown_ratio: Option<f64>,
    #[serde(deserialize_with = "deserialize_required_option")]
    champion_search_nodes_per_second: Option<f64>,
    #[serde(deserialize_with = "deserialize_required_option")]
    challenger_search_nodes_per_second: Option<f64>,
    #[serde(deserialize_with = "deserialize_required_option")]
    search_slowdown_ratio: Option<f64>,
    #[serde(deserialize_with = "deserialize_required_option")]
    champion_average_search_depth: Option<f64>,
    #[serde(deserialize_with = "deserialize_required_option")]
    challenger_average_search_depth: Option<f64>,
    raw_totals: ArenaGameMetrics,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct ArenaResults {
    schema: String,
    generation_id: String,
    plan: RegistryArtifact,
    execution: RegistryArtifact,
    champion_model_id: String,
    challenger_model_id: String,
    games: Vec<ArenaResultGame>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct ArenaResultGame {
    game_id: String,
    pair_id: String,
    start_group: String,
    start_position_id: String,
    black_model_id: String,
    white_model_id: String,
    result: String,
    plies: u64,
    illegal_moves: u64,
    crashes: u64,
    metrics: ArenaGameMetrics,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct ArenaGameMetrics {
    #[serde(deserialize_with = "deserialize_required_option")]
    champion_inference_calls: Option<u64>,
    #[serde(deserialize_with = "deserialize_required_option")]
    champion_inference_time_ns: Option<u64>,
    #[serde(deserialize_with = "deserialize_required_option")]
    challenger_inference_calls: Option<u64>,
    #[serde(deserialize_with = "deserialize_required_option")]
    challenger_inference_time_ns: Option<u64>,
    #[serde(deserialize_with = "deserialize_required_option")]
    champion_search_nodes: Option<u64>,
    #[serde(deserialize_with = "deserialize_required_option")]
    champion_search_elapsed_ms: Option<u64>,
    #[serde(deserialize_with = "deserialize_required_option")]
    challenger_search_nodes: Option<u64>,
    #[serde(deserialize_with = "deserialize_required_option")]
    challenger_search_elapsed_ms: Option<u64>,
    #[serde(deserialize_with = "deserialize_required_option")]
    champion_search_depth_sum: Option<u64>,
    #[serde(deserialize_with = "deserialize_required_option")]
    champion_searches: Option<u64>,
    #[serde(deserialize_with = "deserialize_required_option")]
    challenger_search_depth_sum: Option<u64>,
    #[serde(deserialize_with = "deserialize_required_option")]
    challenger_searches: Option<u64>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct PairedArenaPlan {
    schema: String,
    generation_id: String,
    champion: PairedModelSpec,
    challenger: PairedModelSpec,
    engine: RegistryArtifact,
    #[serde(deserialize_with = "deserialize_required_option")]
    engine_build_receipt: Option<RegistryArtifact>,
    model_registry: RegistryArtifact,
    git_commit: String,
    config: RegistryArtifact,
    config_sha256: String,
    start_positions: RegistryArtifact,
    start_position_validation: RegistryArtifact,
    dataset_manifest: RegistryArtifact,
    game_count: u64,
    pair_count: u64,
    normal_start_pairs: u64,
    start_set_pairs: u64,
    seed: u64,
    nodes_per_move: u64,
    max_workers: u64,
    #[serde(rename = "memoryLimitMiB")]
    memory_limit_mib: u64,
    #[serde(rename = "memoryPerWorkerMiB")]
    memory_per_worker_mib: u64,
    jobs: Vec<PairedArenaJob>,
    plan_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct PairedModelSpec {
    model_id: String,
    artifact: RegistryArtifact,
    evaluator_kind: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct EngineBuildReceiptV1 {
    schema: String,
    git_commit: String,
    binary: RegistryArtifact,
    source_tree_sha256: String,
    source_files: u64,
    source_bytes: u64,
    build_command: Vec<String>,
    cargo_version: String,
    rustc_version: String,
    built_at: String,
    receipt_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct EngineBuildReceiptV2 {
    schema: String,
    git_commit: String,
    binary: RegistryArtifact,
    source_tree_sha256: String,
    source_files: u64,
    source_bytes: u64,
    build_command: Vec<String>,
    cargo_tool: EngineBuildToolIdentity,
    rustc_tool: EngineBuildToolIdentity,
    rustc_runtime_tree: EngineBuildRuntimeTree,
    receipt_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct EngineBuildToolIdentity {
    path: String,
    sha256: String,
    size: u64,
    version: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct EngineBuildRuntimeTree {
    tree_sha256: String,
    files: u64,
    bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum EngineBuildReceipt {
    HistoricalV1(EngineBuildReceiptV1),
    ContentAddressedV2(EngineBuildReceiptV2),
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct PairedArenaJob {
    job_id: String,
    pair_index: u64,
    start_group: String,
    start_position_id: String,
    sfen: String,
    seed: u64,
    game_ids: Vec<String>,
    model_a_color_order: Vec<String>,
    output_dir: PathBuf,
    report_path: PathBuf,
    csa_paths: Vec<PathBuf>,
    quarantine_path: PathBuf,
    command: PairedArenaCommand,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct PairedArenaCommand {
    kind: String,
    argv: Vec<String>,
    timeout_seconds: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct PairedArenaExecution {
    schema: String,
    generation_id: String,
    plan: RegistryArtifact,
    plan_sha256: String,
    status: String,
    game_count_planned: u64,
    jobs_planned: u64,
    jobs_completed: u64,
    jobs_quarantined: u64,
    games_completed: u64,
    games_quarantined: u64,
    quarantined_attempts: u64,
    attempts: Vec<PairedArenaAttempt>,
    manifest_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct PairedArenaAttempt {
    job_id: String,
    attempt: u64,
    status: String,
    #[serde(deserialize_with = "deserialize_required_option")]
    return_code: Option<i64>,
    #[serde(deserialize_with = "deserialize_required_option")]
    timed_out: Option<bool>,
    #[serde(deserialize_with = "deserialize_required_option")]
    output_limit_exceeded: Option<bool>,
    #[serde(deserialize_with = "deserialize_required_option")]
    memory_limit_exceeded: Option<bool>,
    #[serde(deserialize_with = "deserialize_required_option")]
    peak_rss_bytes: Option<u64>,
    #[serde(deserialize_with = "deserialize_required_option")]
    rss_measurement: Option<String>,
    #[serde(deserialize_with = "deserialize_required_option")]
    stdout: Option<RegistryArtifact>,
    #[serde(deserialize_with = "deserialize_required_option")]
    stderr: Option<RegistryArtifact>,
    #[serde(deserialize_with = "deserialize_required_option")]
    report: Option<RegistryArtifact>,
    csa: Vec<RegistryArtifact>,
    #[serde(deserialize_with = "deserialize_required_option")]
    quarantine: Option<RegistryArtifact>,
    #[serde(deserialize_with = "deserialize_required_option")]
    failure_category: Option<String>,
    completed_at: String,
    #[serde(deserialize_with = "deserialize_required_option")]
    command_receipt: Option<RegistryArtifact>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct PairedAttemptCommandReceipt {
    schema: String,
    plan_sha256: String,
    job_id: String,
    attempt: u64,
    command: PairedArenaCommand,
    engine: RegistryArtifact,
    #[serde(deserialize_with = "deserialize_required_option")]
    engine_build_receipt: Option<RegistryArtifact>,
    process_receipt: RegistryArtifact,
    result: PairedAttemptResult,
    stdout: RegistryArtifact,
    stderr: RegistryArtifact,
    #[serde(deserialize_with = "deserialize_required_option")]
    report: Option<RegistryArtifact>,
    csa: Vec<RegistryArtifact>,
    #[serde(deserialize_with = "deserialize_required_option")]
    quarantine: Option<RegistryArtifact>,
    #[serde(deserialize_with = "deserialize_required_option")]
    failure_category: Option<String>,
    completed_at: String,
    receipt_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct PairedAttemptResult {
    return_code: i64,
    timed_out: bool,
    output_limit_exceeded: bool,
    memory_limit_exceeded: bool,
    #[serde(deserialize_with = "deserialize_required_option")]
    peak_rss_bytes: Option<u64>,
    rss_measurement: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
#[expect(
    clippy::struct_excessive_bools,
    reason = "the closed producer outcome schema exposes four independent failure flags"
)]
struct Phase6ProcessReceipt {
    schema: String,
    command: PairedArenaCommand,
    command_sha256: String,
    resume: bool,
    #[serde(rename = "memoryLimitMiB")]
    memory_limit_mib: u64,
    #[serde(deserialize_with = "deserialize_required_option")]
    expected_executable: Option<RegistryArtifact>,
    #[serde(deserialize_with = "deserialize_required_option")]
    engine_build_receipt: Option<RegistryArtifact>,
    #[serde(deserialize_with = "deserialize_required_option")]
    runtime_receipt: Option<RegistryArtifact>,
    return_code: i64,
    timed_out: bool,
    output_limit_exceeded: bool,
    memory_limit_exceeded: bool,
    #[serde(deserialize_with = "deserialize_required_option")]
    peak_rss_bytes: Option<u64>,
    rss_measurement: String,
    stdout: RegistryArtifact,
    stderr: RegistryArtifact,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Phase6CommandInvocation<'a> {
    command: &'a PairedArenaCommand,
    resume: bool,
    #[serde(rename = "memoryLimitMiB")]
    memory_limit_mib: u64,
    expected_executable: &'a RegistryArtifact,
    engine_build_receipt: &'a Option<RegistryArtifact>,
    runtime_receipt: Option<&'a RegistryArtifact>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct PairedQuarantineRecord {
    schema: String,
    job_id: String,
    attempt: u64,
    failure_category: String,
    return_code: i64,
    timed_out: bool,
    output_limit_exceeded: bool,
    memory_limit_exceeded: bool,
    #[serde(deserialize_with = "deserialize_required_option")]
    peak_rss_bytes: Option<u64>,
    rss_measurement: String,
    stdout: RegistryArtifact,
    stderr: RegistryArtifact,
    #[serde(deserialize_with = "deserialize_required_option")]
    observed_report: Option<RegistryArtifact>,
    observed_csa: Vec<RegistryArtifact>,
    #[serde(deserialize_with = "deserialize_required_option")]
    failure_detail: Option<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct Phase2ArenaReport {
    schema: String,
    run: Phase2ArenaRun,
    metrics: Phase2ArenaMetrics,
    games: Vec<Phase2ArenaGame>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct Phase2ArenaRun {
    seed: u64,
    game_limit: u64,
    engine: String,
    #[serde(deserialize_with = "deserialize_required_option")]
    git_commit: Option<String>,
    started_at: String,
    #[serde(deserialize_with = "deserialize_required_option")]
    completed_at: Option<String>,
    initial_sfen: String,
    max_plies: u64,
    config_sha256: String,
    budget: Phase2ArenaBudget,
    player_a: Phase2ArenaPlayer,
    player_b: Phase2ArenaPlayer,
    opening: Phase2ArenaOpening,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct Phase2ArenaBudget {
    kind: String,
    value: u64,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct Phase2ArenaPlayer {
    label: String,
    evaluator_kind: String,
    #[serde(deserialize_with = "deserialize_required_option")]
    search_depth: Option<u64>,
    #[serde(deserialize_with = "deserialize_required_option")]
    hash_megabytes: Option<u64>,
    #[serde(deserialize_with = "deserialize_required_option")]
    transposition: Option<bool>,
    #[serde(deserialize_with = "deserialize_required_option")]
    model_artifact_sha256: Option<String>,
    #[serde(deserialize_with = "deserialize_required_option")]
    model_artifact_size: Option<u64>,
    #[serde(deserialize_with = "deserialize_required_option")]
    model_payload_sha256: Option<String>,
    #[serde(deserialize_with = "deserialize_required_option")]
    architecture_version: Option<u64>,
    #[serde(deserialize_with = "deserialize_required_option")]
    quantization: Option<String>,
    opening_enabled: bool,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct Phase2ArenaOpening {
    enabled: bool,
    #[serde(deserialize_with = "deserialize_required_option")]
    artifact_sha256: Option<String>,
    #[serde(deserialize_with = "deserialize_required_option")]
    artifact_size: Option<u64>,
    #[serde(deserialize_with = "deserialize_required_option")]
    max_plies: Option<u64>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct Phase2ArenaMetrics {
    games: u64,
    finished_games: u64,
    player_a_wins: u64,
    player_b_wins: u64,
    search_wins: u64,
    draws: u64,
    nodes_per_second: f64,
    average_depth: f64,
    tt_hit_rate: f64,
    cutoff_rate: f64,
    pruning_rate: f64,
    milliseconds_per_move: f64,
    neural_inference_calls: u64,
    neural_inference_time_ns: u64,
    player_a_search_nodes: u64,
    player_a_search_elapsed_ms: u64,
    player_a_depth_sum: u64,
    player_a_searches: u64,
    player_a_neural_inference_calls: u64,
    player_a_neural_inference_time_ns: u64,
    player_b_search_nodes: u64,
    player_b_search_elapsed_ms: u64,
    player_b_depth_sum: u64,
    player_b_searches: u64,
    player_b_neural_inference_calls: u64,
    player_b_neural_inference_time_ns: u64,
    #[serde(deserialize_with = "deserialize_required_option")]
    peak_memory_bytes: Option<u64>,
    illegal_moves: u64,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct Phase2ArenaGame {
    id: u64,
    black: String,
    white: String,
    result: String,
    moves: u64,
    csa_path: PathBuf,
    csa_sha256: String,
    csa_size: u64,
    neural_inference_calls: u64,
    neural_inference_time_ns: u64,
    player_a_search_nodes: u64,
    player_a_search_elapsed_ms: u64,
    player_a_depth_sum: u64,
    player_a_searches: u64,
    player_a_neural_inference_calls: u64,
    player_a_neural_inference_time_ns: u64,
    player_b_search_nodes: u64,
    player_b_search_elapsed_ms: u64,
    player_b_depth_sum: u64,
    player_b_searches: u64,
    player_b_neural_inference_calls: u64,
    player_b_neural_inference_time_ns: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct Phase6SelfplayConfig {
    schema_version: u64,
    run: Phase6RunConfig,
    resources: Phase6ResourceConfig,
    paths: Phase6PathConfig,
    hard_positions: Phase6HardPositionConfig,
    replay: Phase6ReplayConfig,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct Phase6RunConfig {
    games: u64,
    normal_start_pairs: u64,
    start_set_pairs: u64,
    seed: u64,
    nodes_per_move: u64,
    max_plies: u64,
    search_depth: u64,
    hash_mib: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct Phase6ResourceConfig {
    workers: u64,
    memory_limit_mib: u64,
    memory_per_worker_mib: u64,
    game_timeout_seconds: u64,
    command_timeout_seconds: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct Phase6PathConfig {
    engine_cli: PathBuf,
    output_root: PathBuf,
    start_positions_manifest: PathBuf,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct Phase6HardPositionConfig {
    teacher_drop_cp: u64,
    evaluation_disagreement_cp: u64,
    candidate_gap_cp: u64,
    minimum_search_nodes: u64,
    max_additional_labels: u64,
    teacher_label_limit: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct Phase6ReplayConfig {
    capacity: u64,
    minimum_older_positions: u64,
    dedup_key: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct Phase6StartPositions {
    schema: String,
    dataset_manifest: RegistryArtifact,
    source_positions: RegistryArtifact,
    selection: Phase6StartSelection,
    positions: Vec<Phase6StartPosition>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct Phase6StartSelection {
    seed: u64,
    train_count: u64,
    validation_count: u64,
    require_eligible: bool,
    reset_move_number: bool,
    excluded_cross_split_states: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct Phase6StartPosition {
    position_id: String,
    sfen: String,
    source_game_sha256: String,
    position_index: u64,
    split: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct Phase3DatasetManifest {
    schema: String,
    dataset_id: String,
    source: Phase3DatasetSource,
    config: Phase3NormalizationConfig,
    counts: Phase3DatasetCounts,
    artifacts: BTreeMap<String, Phase3DatasetArtifact>,
    raw_object_sha256: Vec<String>,
    canonical_game_sha256: Vec<String>,
    evidence_snapshots: Vec<Phase3EvidenceSnapshot>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct Phase3DatasetSource {
    source_id: String,
    name: String,
    official_base: String,
    adapter: String,
    license: String,
    license_evidence: Vec<Phase3LicenseEvidence>,
    redistributable: bool,
    machine_learning_allowed: bool,
    last_reviewed: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct Phase3LicenseEvidence {
    url: String,
    local_path: PathBuf,
    quote: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct Phase3NormalizationConfig {
    schema: String,
    dataset_id: String,
    exporter_timeout_seconds: u64,
    max_games: u64,
    max_positions: u64,
    max_raw_bytes: u64,
    split: Phase3SplitPolicy,
    terminal_tail_positions: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct Phase3SplitPolicy {
    schema: String,
    salt: String,
    salt_sha256: String,
    test_basis_points: u64,
    validation_basis_points: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct Phase3DatasetCounts {
    games: u64,
    positions: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct Phase3DatasetArtifact {
    sha256: String,
    size: u64,
    records: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct Phase3EvidenceSnapshot {
    evidence_id: String,
    url: String,
    retrieved_at: String,
    sha256: String,
    size: u64,
    content_type: String,
    object_path: PathBuf,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct Phase3PositionRow {
    schema: String,
    game_id: String,
    canonical_sha256: String,
    raw_sha256: String,
    source_id: String,
    split: String,
    position_index: u64,
    sfen: String,
    #[serde(deserialize_with = "deserialize_required_option")]
    move_usi: Option<String>,
    #[serde(deserialize_with = "deserialize_required_option")]
    next_sfen: Option<String>,
    outcome: String,
    #[serde(deserialize_with = "deserialize_required_option")]
    terminal_reason: Option<String>,
    side_to_move: String,
    full_plies: u64,
    remaining_plies: u64,
    eligible: bool,
    terminal_tail: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct Phase3GameRow {
    schema: String,
    game_id: String,
    canonical_sha256: String,
    raw_object: Phase3RawObject,
    raw_csa: String,
    normalized_csa: String,
    initial_sfen: String,
    usi_moves: Vec<String>,
    ply_count: u64,
    position_count: u64,
    outcome: String,
    #[serde(deserialize_with = "deserialize_required_option")]
    terminal_reason: Option<String>,
    result_validation: String,
    players: Phase3GamePlayers,
    #[serde(deserialize_with = "deserialize_required_option")]
    date: Option<String>,
    #[serde(deserialize_with = "deserialize_required_option")]
    source_date_time: Option<String>,
    #[serde(deserialize_with = "deserialize_required_option")]
    source_time_zone: Option<String>,
    split: String,
    flags: Phase3GameFlags,
    source_id: String,
    url: String,
    retrieved_at: String,
    license_decision: Phase3LicenseDecision,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct Phase3RawObject {
    object_id: String,
    object_path: PathBuf,
    original_filename: String,
    sha256: String,
    size: u64,
    response: Phase3RawResponse,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct Phase3RawResponse {
    #[serde(deserialize_with = "deserialize_required_option")]
    content_type: Option<String>,
    #[serde(deserialize_with = "deserialize_required_option")]
    etag: Option<String>,
    #[serde(deserialize_with = "deserialize_required_option")]
    last_modified: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct Phase3GamePlayers {
    black: Phase3GamePlayer,
    white: Phase3GamePlayer,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct Phase3GamePlayer {
    #[serde(deserialize_with = "deserialize_required_option")]
    name: Option<String>,
    #[serde(deserialize_with = "deserialize_required_option")]
    rating: Option<i64>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct Phase3GameFlags {
    short: bool,
    long: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct Phase3LicenseDecision {
    license: String,
    evidence: Vec<Phase3LicenseEvidence>,
    evidence_snapshots: Vec<Phase3EvidenceSnapshot>,
    redistributable: bool,
    machine_learning_allowed: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ExpectedPhase3Position {
    canonical_sha256: String,
    raw_sha256: String,
    source_id: String,
    split: String,
    position_index: u64,
    sfen: String,
    move_usi: Option<String>,
    next_sfen: Option<String>,
    outcome: String,
    terminal_reason: Option<String>,
    side_to_move: String,
    full_plies: u64,
    remaining_plies: u64,
    eligible: bool,
    terminal_tail: bool,
}

struct ValidatedPhase3Games {
    positions: BTreeMap<(String, u64), ExpectedPhase3Position>,
    position_order: Vec<(String, u64)>,
    game_splits: BTreeMap<String, String>,
    game_raw: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct Phase6StartValidation {
    schema: String,
    start_positions: RegistryArtifact,
    engine: RegistryArtifact,
    #[serde(deserialize_with = "deserialize_required_option")]
    engine_build_receipt: Option<RegistryArtifact>,
    git_commit: String,
    method: String,
    results: Vec<Phase6StartValidationResult>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
#[expect(
    clippy::struct_excessive_bools,
    reason = "the wire schema records four independent process outcome flags"
)]
struct Phase6StartValidationResult {
    position_id: String,
    sfen_sha256: String,
    legal: bool,
    return_code: i64,
    timed_out: bool,
    output_limit_exceeded: bool,
    memory_limit_exceeded: bool,
    #[serde(deserialize_with = "deserialize_required_option")]
    peak_rss_bytes: Option<u64>,
    rss_measurement: String,
    stdout: RegistryArtifact,
    stderr: RegistryArtifact,
    completed_at: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct GenerationPolicy {
    schema_version: u64,
    evidence: EvidencePolicy,
    thresholds: PromotionThresholds,
    performance: PerformancePolicy,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct EvidencePolicy {
    minimum_games: u64,
    minimum_decisive_games: u64,
    minimum_group_games: u64,
    max_illegal: u64,
    max_crashes: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct PromotionThresholds {
    promote_score_rate: PolicyNumber,
    promote_wilson_lower: PolicyNumber,
    reject_score_rate: PolicyNumber,
    reject_wilson_upper: PolicyNumber,
    minimum_group_score_rate: PolicyNumber,
    maximum_side_score_gap: PolicyNumber,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct PerformancePolicy {
    require_metrics: bool,
    maximum_inference_slowdown: PolicyNumber,
    maximum_search_slowdown: PolicyNumber,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
enum PolicyNumber {
    Integer(i64),
    Float(f64),
}

impl Serialize for PolicyNumber {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        // Python's policy parser accepts either TOML integers or floats through
        // `require_number`, then normalizes both to a dataclass `float`.  The semantic
        // policy digest therefore always contains a JSON floating-point value even when
        // the source TOML used an integer spelling.
        serializer.serialize_f64(self.as_f64())
    }
}

impl PolicyNumber {
    #[expect(
        clippy::cast_precision_loss,
        reason = "policy integers are range-checked to at most 100 before this conversion"
    )]
    fn as_f64(&self) -> f64 {
        match self {
            Self::Integer(value) => *value as f64,
            Self::Float(value) => *value,
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct DecisionEvent {
    schema: String,
    ply: usize,
    actor: String,
    model_id: String,
    #[serde(deserialize_with = "deserialize_required_option")]
    model_artifact_sha256: Option<String>,
    #[serde(deserialize_with = "deserialize_required_option")]
    model_payload_sha256: Option<String>,
    config_sha256: String,
    sfen_before: String,
    move_usi: String,
    nodes: u64,
    elapsed_ms: u64,
    depth: u8,
    pv: Vec<String>,
    #[serde(deserialize_with = "deserialize_required_option")]
    score_cp: Option<i32>,
    opening_book: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct HumanPlayConfigRecord {
    schema: String,
    config_sha256: String,
    human_side: String,
    budget_kind: String,
    budget_value: u64,
    #[serde(deserialize_with = "deserialize_required_option")]
    black_time_ms: Option<u64>,
    #[serde(deserialize_with = "deserialize_required_option")]
    white_time_ms: Option<u64>,
    #[serde(deserialize_with = "deserialize_required_option")]
    byoyomi_ms: Option<u64>,
    #[serde(deserialize_with = "deserialize_required_option")]
    black_increment_ms: Option<u64>,
    #[serde(deserialize_with = "deserialize_required_option")]
    white_increment_ms: Option<u64>,
    safety_margin_ms: u64,
    time_control_schema: String,
    depth: u8,
    initial_sfen: String,
    max_plies: u32,
    profile: String,
    model_id: String,
    #[serde(deserialize_with = "deserialize_required_option")]
    model_artifact_sha256: Option<String>,
    #[serde(deserialize_with = "deserialize_required_option")]
    model_payload_sha256: Option<String>,
    #[serde(deserialize_with = "deserialize_required_option")]
    architecture_version: Option<u32>,
    #[serde(deserialize_with = "deserialize_required_option")]
    quantization: Option<String>,
    #[serde(deserialize_with = "deserialize_required_option")]
    registry_sha256: Option<String>,
    #[serde(deserialize_with = "deserialize_required_option")]
    registry_revision: Option<u64>,
    #[serde(deserialize_with = "deserialize_required_option")]
    opening_artifact_sha256: Option<String>,
    #[serde(deserialize_with = "deserialize_required_option")]
    opening_artifact_size: Option<u64>,
    opening_max_plies: u32,
    opening_profile: String,
    opening_minimum_samples: u64,
    opening_maximum_teacher_loss_cp: i32,
    transposition_entries: usize,
    engine_name: String,
    engine_version: String,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct PublicationMarker {
    schema: String,
    config_sha256: String,
    csa_target: String,
    decision_target: String,
    csa_pending: String,
    decision_pending: String,
    csa_sha256: String,
    csa_size: u64,
    decision_sha256: String,
    decision_size: u64,
}

pub fn run(arguments: &[String]) -> Result<(), String> {
    let config = parse_arguments(arguments)?;
    let profile = resolve_profile(&config)?;
    let opening = load_play_opening(&config)?;
    let config_sha256 = play_config_sha256(&config, &profile, opening.as_ref());
    if recover_publication(&config, &config_sha256)? {
        println!(
            "recovered completed CSA and decision log publication ({}, {})",
            config.output.display(),
            config.decision_log.display()
        );
        return Ok(());
    }
    let stdin = io::stdin();
    let mut reader = stdin.lock();
    let stdout = io::stdout();
    let mut writer = stdout.lock();
    run_interactive_resolved(
        &config,
        &profile,
        opening.as_ref(),
        &mut reader,
        &mut writer,
    )
}

#[expect(
    clippy::too_many_lines,
    reason = "the bounded terminal-play option parser keeps profile safety checks explicit"
)]
fn parse_arguments(arguments: &[String]) -> Result<PlayConfig, String> {
    let mut human = Side::Black;
    let mut nodes = None;
    let mut movetime_ms = None;
    let mut black_time_ms = None;
    let mut white_time_ms = None;
    let mut byoyomi_ms = None;
    let mut black_increment_ms = None;
    let mut white_increment_ms = None;
    let mut safety_margin_ms = 50_u64;
    let mut depth = 8_u8;
    let mut sfen = None;
    let mut max_plies = DEFAULT_MAX_PLIES;
    let mut output = PathBuf::from("artifacts/pending_human_review/human-game.csa");
    let mut decision_log = None;
    let mut profile = PlayProfile::OverallChampion;
    let mut model_path = None;
    let mut registry_path = None;
    let mut opening_book_path = None;
    let mut opening_max_plies = 24_u32;
    let mut opening_profile = OpeningProfile::IbishaStrict;
    let mut opening_minimum_samples = 2_u64;
    let mut opening_maximum_teacher_loss_cp = 80_i32;
    let mut seen_options = std::collections::BTreeSet::new();
    let mut index = 0;
    while index < arguments.len() {
        let option = arguments[index].as_str();
        if !seen_options.insert(option) {
            return Err(format!("duplicate play argument: {option}"));
        }
        match option {
            "--human" => {
                human = match next_value(arguments, &mut index, "--human")? {
                    "black" => Side::Black,
                    "white" => Side::White,
                    _ => return Err("--human must be `black` or `white`".to_owned()),
                };
            }
            "--nodes" => nodes = Some(parse_next(arguments, &mut index, "--nodes")?),
            "--movetime-ms" => {
                movetime_ms = Some(parse_next(arguments, &mut index, "--movetime-ms")?);
            }
            "--black-time-ms" => {
                black_time_ms = Some(parse_next(arguments, &mut index, "--black-time-ms")?);
            }
            "--white-time-ms" => {
                white_time_ms = Some(parse_next(arguments, &mut index, "--white-time-ms")?);
            }
            "--byoyomi-ms" => {
                byoyomi_ms = Some(parse_next(arguments, &mut index, "--byoyomi-ms")?);
            }
            "--black-increment-ms" => {
                black_increment_ms = Some(parse_next(
                    arguments,
                    &mut index,
                    "--black-increment-ms",
                )?);
            }
            "--white-increment-ms" => {
                white_increment_ms = Some(parse_next(
                    arguments,
                    &mut index,
                    "--white-increment-ms",
                )?);
            }
            "--safety-margin-ms" => {
                safety_margin_ms =
                    parse_next(arguments, &mut index, "--safety-margin-ms")?;
            }
            "--depth" => depth = parse_next(arguments, &mut index, "--depth")?,
            "--sfen" => sfen = Some(next_value(arguments, &mut index, "--sfen")?.to_owned()),
            "--max-plies" => {
                max_plies = parse_next(arguments, &mut index, "--max-plies")?;
            }
            "--output" => output = path_next(arguments, &mut index, "--output")?,
            "--decision-log" => {
                decision_log = Some(path_next(arguments, &mut index, "--decision-log")?);
            }
            "--profile" => {
                profile = PlayProfile::parse(next_value(arguments, &mut index, "--profile")?)?;
            }
            "--model" => model_path = Some(path_next(arguments, &mut index, "--model")?),
            "--registry" => {
                registry_path = Some(path_next(arguments, &mut index, "--registry")?);
            }
            "--opening-book" => {
                opening_book_path = Some(path_next(arguments, &mut index, "--opening-book")?);
            }
            "--opening-max-plies" => {
                opening_max_plies = parse_next(arguments, &mut index, "--opening-max-plies")?;
            }
            "--opening-profile" => {
                opening_profile =
                    OpeningProfile::parse(next_value(arguments, &mut index, "--opening-profile")?)?;
            }
            "--opening-min-samples" => {
                opening_minimum_samples =
                    parse_next(arguments, &mut index, "--opening-min-samples")?;
            }
            "--opening-max-teacher-loss-cp" => {
                opening_maximum_teacher_loss_cp =
                    parse_next(arguments, &mut index, "--opening-max-teacher-loss-cp")?;
            }
            argument => return Err(format!("unknown play argument: {argument}")),
        }
        index += 1;
    }
    if depth == 0 || depth > 64 {
        return Err("--depth must be 1..=64".to_owned());
    }
    if !(1..=MAX_PLIES).contains(&max_plies) {
        return Err(format!("--max-plies must be 1..={MAX_PLIES}"));
    }
    if !(1..=MAX_PLIES).contains(&opening_max_plies) {
        return Err(format!("--opening-max-plies must be 1..={MAX_PLIES}"));
    }
    if opening_minimum_samples == 0 || opening_minimum_samples > 1_000_000 {
        return Err("--opening-min-samples must be 1..=1000000".to_owned());
    }
    if !(0..=10_000).contains(&opening_maximum_teacher_loss_cp) {
        return Err("--opening-max-teacher-loss-cp must be 0..=10000".to_owned());
    }
    if safety_margin_ms > open_shogi_core::MAX_SAFETY_MARGIN_MS {
        return Err(format!(
            "--safety-margin-ms must be 0..={}",
            open_shogi_core::MAX_SAFETY_MARGIN_MS
        ));
    }
    if profile.uses_direct_model() && model_path.is_none() {
        return Err("direct model-backed profiles require --model".to_owned());
    }
    if !profile.uses_direct_model() && model_path.is_some() {
        return Err("--model is only valid with a direct model-backed profile".to_owned());
    }
    if profile.is_registry_alias() && registry_path.is_none() {
        return Err(
            "generation/neural-lineage-champion/challenger profiles require --registry".to_owned(),
        );
    }
    if !profile.is_registry_alias() && registry_path.is_some() {
        return Err(
            "--registry is only valid with generation/neural-lineage-champion/challenger profiles"
                .to_owned(),
        );
    }
    let has_clock = black_time_ms.is_some()
        || white_time_ms.is_some()
        || byoyomi_ms.is_some()
        || black_increment_ms.is_some()
        || white_increment_ms.is_some();
    if has_clock && (nodes.is_some() || movetime_ms.is_some()) {
        return Err(
            "clock options, --nodes, and --movetime-ms are mutually exclusive".to_owned(),
        );
    }
    let budget = if has_clock {
        let clock = ClockBudget {
            black_time: black_time_ms.unwrap_or(0),
            white_time: white_time_ms.unwrap_or(0),
            byoyomi: byoyomi_ms,
            black_increment: black_increment_ms,
            white_increment: white_increment_ms,
        };
        time_control_for_budget(Budget::Clock(clock), safety_margin_ms).validate()?;
        Budget::Clock(clock)
    } else {
        match (nodes, movetime_ms) {
            (Some(_), Some(_)) => {
                return Err("--nodes and --movetime-ms are mutually exclusive".to_owned());
            }
            (Some(0), _) | (_, Some(0)) => {
                return Err("search budget must be positive".to_owned());
            }
            (Some(value), None) if value <= MAX_NODES_PER_MOVE => Budget::Nodes(value),
            (Some(_), None) => {
                return Err(format!("--nodes must be 1..={MAX_NODES_PER_MOVE}"));
            }
            (None, Some(value)) if value <= MAX_MOVETIME_MS => Budget::MoveTime(value),
            (None, Some(_)) => {
                return Err(format!("--movetime-ms must be 1..={MAX_MOVETIME_MS}"));
            }
            (None, None) => Budget::Casual,
        }
    };
    let initial = match sfen {
        Some(value) => parse_sfen(&value).map_err(|error| format!("invalid --sfen: {error}"))?,
        None => Position::startpos(),
    };
    if initial.move_number() != 1 {
        return Err("--sfen move number must be 1 for canonical CSA output".to_owned());
    }
    let decision_log = decision_log.unwrap_or_else(|| output.with_extension("decisions.jsonl"));
    if decision_log == output {
        return Err("--decision-log must differ from --output".to_owned());
    }
    Ok(PlayConfig {
        human,
        budget,
        safety_margin_ms,
        depth,
        initial,
        max_plies,
        output,
        decision_log,
        profile,
        model_path,
        registry_path,
        opening_book_path,
        opening_max_plies,
        opening_profile,
        opening_minimum_samples,
        opening_maximum_teacher_loss_cp,
    })
}

fn resolve_profile(config: &PlayConfig) -> Result<ResolvedProfile, String> {
    match config.profile {
        PlayProfile::Material => Ok(handcrafted_profile("material")),
        PlayProfile::HandcraftedBaseline => Ok(handcrafted_profile("handcrafted-baseline")),
        PlayProfile::HandcraftedExperimental => Ok(handcrafted_profile("handcrafted-experimental")),
        PlayProfile::OverallChampion => Ok(handcrafted_profile(
            open_shogi_core::OVERALL_CHAMPION_ID,
        )),
        PlayProfile::Neural | PlayProfile::Residual | PlayProfile::Composite => load_neural_profile(
            match config.profile {
                PlayProfile::Neural => "neural-direct",
                PlayProfile::Residual => "residual-direct",
                PlayProfile::Composite => "composite-direct",
                _ => unreachable!(),
            },
            config
                .model_path
                .as_deref()
                .ok_or_else(|| "neural profile lacks --model".to_owned())?,
            None,
            None,
            None,
        ),
        PlayProfile::GenerationZero
        | PlayProfile::NeuralLineageChampion
        | PlayProfile::Challenger => {
            resolve_registry_profile(
                config.profile,
                config
                    .registry_path
                    .as_deref()
                    .ok_or_else(|| "registry profile lacks --registry".to_owned())?,
            )
        }
    }
}

fn handcrafted_profile(model_id: &str) -> ResolvedProfile {
    ResolvedProfile {
        model_id: model_id.to_owned(),
        model_artifact_sha256: None,
        model_payload_sha256: None,
        architecture_version: None,
        quantization: None,
        neural: None,
        registry_sha256: None,
        registry_revision: None,
    }
}

fn load_neural_profile(
    model_id: &str,
    path: &Path,
    expected_sha256: Option<&str>,
    expected_size: Option<u64>,
    expected_architecture: Option<u32>,
) -> Result<ResolvedProfile, String> {
    let artifact = read_file_artifact(
        path,
        u64::try_from(MAX_NEURAL_MODEL_BYTES).unwrap_or(u64::MAX),
    )?;
    load_neural_profile_from_artifact(
        model_id,
        path,
        artifact,
        expected_sha256,
        expected_size,
        expected_architecture,
        None,
        None,
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "registry identity checks remain explicit at the immutable artifact boundary"
)]
fn load_neural_profile_from_artifact(
    model_id: &str,
    path: &Path,
    artifact: FileArtifact,
    expected_sha256: Option<&str>,
    expected_size: Option<u64>,
    expected_architecture: Option<u32>,
    registry_sha256: Option<String>,
    registry_revision: Option<u64>,
) -> Result<ResolvedProfile, String> {
    if let Some(expected) = expected_size
        && artifact.size != expected
    {
        return Err(format!(
            "registry model size mismatch for {}: expected {expected}, found {}",
            path.display(),
            artifact.size
        ));
    }
    let model = NeuralEvaluator::from_bytes(&artifact.bytes)
        .map_err(|error| format!("cannot load neural model {}: {error}", path.display()))?;
    let artifact_sha256 = artifact.sha256;
    if let Some(expected) = expected_sha256
        && artifact_sha256 != expected
    {
        return Err(format!(
            "registry model SHA-256 mismatch for {}",
            path.display()
        ));
    }
    if let Some(expected) = expected_architecture
        && model.identity().architecture_version != expected
    {
        return Err(format!(
            "registry architecture version mismatch for {}",
            path.display()
        ));
    }
    Ok(ResolvedProfile {
        model_id: model_id.to_owned(),
        model_artifact_sha256: Some(artifact_sha256),
        model_payload_sha256: Some(model.identity().sha256_hex()),
        architecture_version: Some(model.identity().architecture_version),
        quantization: Some(model.quantization()),
        neural: Some(Arc::new(model)),
        registry_sha256,
        registry_revision,
    })
}

fn resolve_registry_profile(
    profile: PlayProfile,
    registry_path: &Path,
) -> Result<ResolvedProfile, String> {
    let (registry, storage, registry_sha256) = read_registry(registry_path)?;
    validate_model_registry(&registry)?;
    let models = registry
        .models
        .iter()
        .map(|model| (model.model_id.as_str(), model))
        .collect::<BTreeMap<_, _>>();
    let model_id = match profile {
        PlayProfile::NeuralLineageChampion => registry
            .champion_model_id
            .as_deref()
            .ok_or_else(|| "model registry has no champion".to_owned())?,
        PlayProfile::Challenger => select_challenger_model_id(&registry)?,
        PlayProfile::GenerationZero => registry
            .generations
            .iter()
            .find(|generation| generation.generation_id == "generation-0")
            .map(|generation| generation.champion_model_id.as_str())
            .ok_or_else(|| "model registry has no generation-0".to_owned())?,
        _ => return Err("internal non-registry profile".to_owned()),
    };
    let entry = models
        .get(model_id)
        .ok_or_else(|| format!("model registry does not define {model_id}"))?;
    if entry.evaluator_kind != "neural" {
        return Err("terminal registry profiles currently require a neural model".to_owned());
    }
    validate_sha256(&entry.artifact.sha256)?;
    let path = storage.root_path.join(&entry.artifact.path);
    let artifact = verify_registry_artifacts(&registry, &storage, model_id)?;
    let architecture_version = entry
        .architecture_version
        .parse::<u32>()
        .map_err(|_| "registry architectureVersion must be a decimal wire version".to_owned())?;
    let resolved = load_neural_profile_from_artifact(
        &entry.model_id,
        &path,
        artifact,
        Some(&entry.artifact.sha256),
        Some(entry.artifact.size),
        Some(architecture_version),
        Some(registry_sha256),
        Some(registry.revision),
    )?;
    if quantization_name(
        resolved
            .quantization
            .ok_or_else(|| "registry neural model lacks quantization".to_owned())?,
    ) != entry.quantization
    {
        return Err("registry quantization does not match the model artifact".to_owned());
    }
    Ok(resolved)
}

fn select_challenger_model_id(registry: &ModelRegistry) -> Result<&str, String> {
    if let Some(active_challenger) = registry.challenger_model_id.as_deref() {
        return Ok(active_challenger);
    }
    let latest_generation = registry
        .generations
        .iter()
        .rev()
        .find(|generation| generation.parent_generation_id.is_some())
        .ok_or_else(|| "model registry has no non-initial challenger generation".to_owned())?;
    latest_generation
        .challenger_model_id
        .as_deref()
        .ok_or_else(|| "model registry latest non-initial generation has no challenger".to_owned())
}

fn read_registry(path: &Path) -> Result<(ModelRegistry, RegistryStorage, String), String> {
    let storage = repository_root_and_registry(path)?;
    let file = storage
        .root
        .open_relative_regular(&storage.registry_relative)
        .map_err(|error| format!("cannot open model registry without symlinks: {error}"))?;
    let artifact = read_open_file_artifact(
        file,
        &storage.root_path.join(&storage.registry_relative),
        MAX_REGISTRY_BYTES,
    )?;
    if artifact.bytes.is_empty() {
        return Err(format!(
            "model registry size must be 1..={MAX_REGISTRY_BYTES} bytes"
        ));
    }
    let value = parse_unique_json(&artifact.bytes, "model registry")?;
    let registry: ModelRegistry = deserialize_closed_json(&value, "model registry")?;
    Ok((registry, storage, artifact.sha256))
}

#[expect(
    clippy::too_many_lines,
    reason = "the Rust reader mirrors the closed canonical Python registry contract"
)]
fn validate_model_registry(registry: &ModelRegistry) -> Result<(), String> {
    if registry.schema != "phase6_model_registry/v1" {
        return Err("model registry schema mismatch".to_owned());
    }
    if !(1..=1_000_000_000).contains(&registry.revision) {
        return Err("model registry revision must be 1..=1000000000".to_owned());
    }
    if registry.models.len() > MAX_REGISTRY_ITEMS || registry.generations.len() > MAX_REGISTRY_ITEMS
    {
        return Err("model registry exceeds the 10000-item contract limit".to_owned());
    }
    if let Some(champion) = &registry.champion_model_id {
        validate_identifier(champion, "championModelId")?;
    }
    if let Some(challenger) = &registry.challenger_model_id {
        validate_identifier(challenger, "challengerModelId")?;
    }

    let mut models = BTreeMap::<&str, &RegistryModel>::new();
    let mut model_generations = BTreeMap::<&str, &str>::new();
    for model in &registry.models {
        validate_identifier(&model.model_id, "modelId")?;
        validate_identifier(&model.generation_id, "generationId")?;
        if models.contains_key(model.model_id.as_str()) {
            return Err("model registry contains a duplicate modelId".to_owned());
        }
        if let Some(parent) = &model.parent_model_id {
            validate_identifier(parent, "parentModelId")?;
            if parent == &model.model_id {
                return Err("registry model cannot be its own parent".to_owned());
            }
            if !models.contains_key(parent.as_str()) {
                return Err("registry parentModelId must reference an earlier model".to_owned());
            }
        }
        validate_artifact_ref(&model.artifact)?;
        if model.evaluator_kind != "neural" {
            return Err("registry evaluatorKind must be neural".to_owned());
        }
        if model.architecture_version != "1" {
            return Err("registry architectureVersion must be OSAVAL architecture 1".to_owned());
        }
        if !matches!(model.quantization.as_str(), "float32" | "int8") {
            return Err("registry quantization must be float32 or int8".to_owned());
        }
        if let Some(training_run) = &model.training_run {
            validate_artifact_ref(training_run)?;
        }
        validate_utc_timestamp(&model.registered_at, "registeredAt")?;
        if model.license_status != "pending-review" {
            return Err("registry licenseStatus must be pending-review".to_owned());
        }
        models.insert(&model.model_id, model);
        model_generations.insert(&model.model_id, &model.generation_id);
    }
    if let Some(champion) = &registry.champion_model_id
        && !models.contains_key(champion.as_str())
    {
        return Err("championModelId is not present in models".to_owned());
    }
    if let Some(challenger) = &registry.challenger_model_id
        && !models.contains_key(challenger.as_str())
    {
        return Err("challengerModelId is not present in models".to_owned());
    }
    if registry.champion_model_id.is_some()
        && registry.champion_model_id == registry.challenger_model_id
    {
        return Err("champion and challenger must be different models".to_owned());
    }

    let mut generations = BTreeMap::<&str, &RegistryGeneration>::new();
    let mut initial_generation_seen = false;
    for (generation_index, generation) in registry.generations.iter().enumerate() {
        validate_identifier(&generation.generation_id, "generationId")?;
        if generations.contains_key(generation.generation_id.as_str()) {
            return Err("model registry contains a duplicate generationId".to_owned());
        }
        if let Some(parent) = &generation.parent_generation_id {
            validate_identifier(parent, "parentGenerationId")?;
            if parent == &generation.generation_id {
                return Err("registry generation cannot be its own parent".to_owned());
            }
            if !generations.contains_key(parent.as_str()) {
                return Err(
                    "registry parentGenerationId must reference an earlier generation".to_owned(),
                );
            }
        }
        let expected_parent = generation_index
            .checked_sub(1)
            .map(|index| registry.generations[index].generation_id.as_str());
        if generation.parent_generation_id.as_deref() != expected_parent {
            return Err(
                "registry parentGenerationId must be the immediately preceding generation"
                    .to_owned(),
            );
        }
        validate_identifier(&generation.champion_model_id, "generation championModelId")?;
        if !models.contains_key(generation.champion_model_id.as_str()) {
            return Err("registry generation championModelId is unknown".to_owned());
        }
        if let Some(challenger) = &generation.challenger_model_id {
            validate_identifier(challenger, "generation challengerModelId")?;
            let challenger_generation = model_generations
                .get(challenger.as_str())
                .ok_or_else(|| "registry generation challengerModelId is unknown".to_owned())?;
            if **challenger_generation != generation.generation_id {
                return Err("registry challengerModelId belongs to another generation".to_owned());
            }
        }
        for artifact in generation_artifacts(generation).into_iter().flatten() {
            validate_artifact_ref(artifact)?;
        }
        if !matches!(generation.status.as_str(), "arena" | "complete") {
            return Err("registry generation status must be arena or complete".to_owned());
        }
        validate_utc_timestamp(&generation.created_at, "createdAt")?;
        let champion_model = models
            .get(generation.champion_model_id.as_str())
            .expect("generation champion existence was validated");
        if generation.parent_generation_id.is_none() {
            if initial_generation_seen {
                return Err("model registry must contain exactly one initial generation".to_owned());
            }
            initial_generation_seen = true;
            if generation.challenger_model_id.is_some()
                || generation.status != "complete"
                || champion_model.generation_id != generation.generation_id
                || generation.selfplay_manifest.is_some()
                || generation.teacher_labeling_manifest.is_some()
                || generation.arena_manifest.is_some()
                || generation.promotion_decision.is_some()
                || generation.training_run_manifest != champion_model.training_run
            {
                return Err("registry initial generation violates its lifecycle".to_owned());
            }
        } else {
            let challenger_id = generation.challenger_model_id.as_deref().ok_or_else(|| {
                "registry non-initial generation requires a challenger".to_owned()
            })?;
            let challenger_model = models
                .get(challenger_id)
                .expect("generation challenger existence was validated");
            if challenger_model.parent_model_id.as_deref()
                != Some(generation.champion_model_id.as_str())
                || generation.selfplay_manifest.is_none()
                || generation.teacher_labeling_manifest.is_none()
                || generation.training_run_manifest.is_none()
                || challenger_model.training_run != generation.training_run_manifest
            {
                return Err(
                    "registry generation violates challenger lineage or training evidence"
                        .to_owned(),
                );
            }
            let arena_complete = generation.arena_manifest.is_some();
            let decision_complete = generation.promotion_decision.is_some();
            if arena_complete != decision_complete {
                return Err("registry arena and promotion evidence must appear together".to_owned());
            }
            if (generation.status == "arena" && arena_complete)
                || (generation.status == "complete" && !arena_complete)
            {
                return Err(
                    "registry generation status disagrees with promotion evidence".to_owned(),
                );
            }
        }
        generations.insert(&generation.generation_id, generation);
    }
    if !initial_generation_seen {
        return Err("model registry must contain exactly one initial generation".to_owned());
    }
    if model_generations
        .values()
        .any(|generation| !generations.contains_key(*generation))
    {
        return Err("every model generationId must be present in generations".to_owned());
    }
    let active_generations = registry
        .generations
        .iter()
        .filter(|generation| generation.status == "arena")
        .collect::<Vec<_>>();
    match registry.challenger_model_id.as_deref() {
        None if !active_generations.is_empty() => {
            return Err("registry has an arena generation without an active challenger".to_owned());
        }
        Some(challenger)
            if active_generations.len() != 1
                || active_generations[0].challenger_model_id.as_deref() != Some(challenger) =>
        {
            return Err("active challenger must belong to exactly one arena generation".to_owned());
        }
        None | Some(_) => {}
    }
    if let Some(active) = active_generations.first() {
        if registry.generations.last().map(|generation| &generation.generation_id)
            != Some(&active.generation_id)
        {
            return Err("only the latest registry generation may remain active".to_owned());
        }
        if registry.champion_model_id.as_deref() != Some(active.champion_model_id.as_str()) {
            return Err(
                "registry champion must remain the active generation incumbent".to_owned(),
            );
        }
    }
    if let Some(champion) = registry.champion_model_id.as_deref()
        && !registry.generations.iter().any(|generation| {
            generation.status == "complete"
                && (generation.champion_model_id == champion
                    || generation.challenger_model_id.as_deref() == Some(champion))
        })
    {
        return Err("registry champion has no completed generation evidence".to_owned());
    }
    Ok(())
}

fn generation_artifacts(generation: &RegistryGeneration) -> [Option<&RegistryArtifact>; 5] {
    [
        generation.selfplay_manifest.as_ref(),
        generation.teacher_labeling_manifest.as_ref(),
        generation.training_run_manifest.as_ref(),
        generation.arena_manifest.as_ref(),
        generation.promotion_decision.as_ref(),
    ]
}

fn validate_artifact_ref(artifact: &RegistryArtifact) -> Result<(), String> {
    validate_relative_artifact_path(&artifact.path)?;
    validate_sha256(&artifact.sha256)?;
    if artifact.size > MAX_REGISTRY_ARTIFACT_BYTES {
        return Err("registry artifact exceeds the 4 GiB contract limit".to_owned());
    }
    Ok(())
}

fn verify_registry_artifacts(
    registry: &ModelRegistry,
    storage: &RegistryStorage,
    selected_model_id: &str,
) -> Result<FileArtifact, String> {
    let mut budget = RegistryVerificationBudget::new();
    verify_registry_artifacts_with_budget(registry, storage, selected_model_id, &mut budget)
}

fn verify_registry_artifacts_with_budget(
    registry: &ModelRegistry,
    storage: &RegistryStorage,
    selected_model_id: &str,
    budget: &mut RegistryVerificationBudget,
) -> Result<FileArtifact, String> {
    let selected_index = registry
        .models
        .iter()
        .position(|model| model.model_id == selected_model_id)
        .ok_or_else(|| "selected registry model is not present".to_owned())?;
    let mut selected = None;
    let mut observed_artifacts = BTreeMap::<StableFileIdentity, ObservedRegistryArtifact>::new();
    // Read the selected model first so a later hard-link alias can reuse its parsed metadata
    // without forcing a second read merely to retain the selected bytes.
    let model_order = std::iter::once(selected_index)
        .chain((0..registry.models.len()).filter(|index| *index != selected_index));
    for index in model_order {
        verify_registry_model(
            &registry.models[index],
            index == selected_index,
            storage,
            &mut observed_artifacts,
            budget,
            &mut selected,
        )?;
    }
    let mut completed_decisions = BTreeMap::<String, String>::new();
    for generation in &registry.generations {
        if let Some(reference) = &generation.promotion_decision {
            let bytes = verify_registry_artifact_cached(
                storage,
                reference,
                &mut observed_artifacts,
                budget,
                Some(MAX_REGISTRY_BYTES),
            )?
            .ok_or_else(|| "promotion decision bytes were not retained".to_owned())?;
            let decision = validate_promotion_evidence(
                storage,
                registry,
                generation,
                bytes.as_slice(),
                &mut observed_artifacts,
                budget,
            )?;
            completed_decisions.insert(generation.generation_id.clone(), decision);
        }
        for reference in generation_artifacts(generation)
            .into_iter()
            .take(4)
            .flatten()
        {
            let _ = verify_registry_artifact_cached(
                storage,
                reference,
                &mut observed_artifacts,
                budget,
                None,
            )?;
        }
    }
    validate_registry_champion_transition(registry, &completed_decisions)?;
    budget.ensure_time()?;
    selected.ok_or_else(|| "selected registry model artifact was not verified".to_owned())
}

fn verify_registry_model(
    model: &RegistryModel,
    selected_model: bool,
    storage: &RegistryStorage,
    observed: &mut BTreeMap<StableFileIdentity, ObservedRegistryArtifact>,
    budget: &mut RegistryVerificationBudget,
    selected: &mut Option<FileArtifact>,
) -> Result<(), String> {
    budget.charge_reference()?;
    validate_relative_artifact_path(&model.artifact.path)?;
    let display_path = storage.root_path.join(&model.artifact.path);
    let file = storage
        .root
        .open_relative_regular(&model.artifact.path)
        .map_err(|error| {
            format!(
                "cannot open registry artifact {}: {error}",
                display_path.display()
            )
        })?;
    let identity = file.stable_identity().map_err(|error| {
        format!(
            "cannot bind registry artifact identity {}: {error}",
            display_path.display()
        )
    })?;
    let verified = if let Some(verified) = observed.get(&identity) {
        if selected_model {
            return Err("selected registry model was not the first read of its inode".to_owned());
        }
        verified.clone()
    } else {
        budget.charge_unique_bytes(identity.length())?;
        let artifact = read_open_file_artifact_with_check(
            file,
            &display_path,
            u64::try_from(MAX_NEURAL_MODEL_BYTES).unwrap_or(u64::MAX),
            || budget.ensure_time(),
        )?;
        budget.ensure_time()?;
        verify_observed_artifact(&model.artifact, &artifact.sha256, artifact.size)?;
        let evaluator = NeuralEvaluator::from_bytes(&artifact.bytes).map_err(|error| {
            format!(
                "registry model artifact is not valid OSAVAL01 for {}: {error}",
                model.model_id
            )
        })?;
        let verified = ObservedRegistryArtifact {
            sha256: artifact.sha256.clone(),
            size: artifact.size,
            model: Some(VerifiedModelMetadata {
                architecture_version: evaluator.identity().architecture_version,
                quantization: evaluator.quantization(),
                payload_sha256: evaluator.identity().sha256_hex(),
            }),
            bytes: None,
        };
        if selected_model {
            *selected = Some(artifact);
        }
        observed.insert(identity, verified.clone());
        verified
    };
    verify_observed_artifact(&model.artifact, &verified.sha256, verified.size)?;
    let verified_model = verified.model.as_ref().ok_or_else(|| {
        "registry model aliases an artifact that was not parsed as OSAVAL01".to_owned()
    })?;
    let architecture = model
        .architecture_version
        .parse::<u32>()
        .map_err(|_| "registry architectureVersion must be a decimal wire version".to_owned())?;
    if verified_model.architecture_version != architecture
        || quantization_name(verified_model.quantization) != model.quantization
    {
        return Err(format!(
            "registry metadata disagrees with OSAVAL01 artifact for {}",
            model.model_id
        ));
    }
    if let Some(training_run) = &model.training_run {
        let _ = verify_registry_artifact_cached(storage, training_run, observed, budget, None)?;
    }
    Ok(())
}

fn verify_registry_artifact_cached(
    storage: &RegistryStorage,
    expected: &RegistryArtifact,
    observed: &mut BTreeMap<StableFileIdentity, ObservedRegistryArtifact>,
    budget: &mut RegistryVerificationBudget,
    retained_byte_limit: Option<u64>,
) -> Result<Option<Arc<Vec<u8>>>, String> {
    budget.charge_reference()?;
    validate_relative_artifact_path(&expected.path)?;
    let display_path = storage.root_path.join(&expected.path);
    let file = storage
        .root
        .open_relative_regular(&expected.path)
        .map_err(|error| {
            format!(
                "cannot open registry artifact {}: {error}",
                display_path.display()
            )
        })?;
    let stable_identity = file.stable_identity().map_err(|error| {
        format!(
            "cannot bind registry artifact identity {}: {error}",
            display_path.display()
        )
    })?;
    let identity = if let Some(identity) = observed.get(&stable_identity) {
        if retained_byte_limit.is_some_and(|maximum| identity.size > maximum) {
            return Err(format!(
                "registry artifact {} exceeds its role-specific byte limit",
                expected.path.display()
            ));
        }
        if retained_byte_limit.is_some() && identity.bytes.is_none() {
            return Err(
                "registry artifact aliases a previously verified non-JSON role; refusing a duplicate read"
                    .to_owned(),
            );
        }
        identity.clone()
    } else {
        budget.charge_unique_bytes(stable_identity.length())?;
        let (sha256, size, bytes) = if let Some(maximum_bytes) = retained_byte_limit {
            let artifact = read_open_file_artifact_with_check(
                file,
                &display_path,
                maximum_bytes,
                || budget.ensure_time(),
            )?;
            (
                artifact.sha256,
                artifact.size,
                Some(Arc::new(artifact.bytes)),
            )
        } else {
            let (sha256, size) = sha256_open_file_with_check(
                file,
                &display_path,
                MAX_REGISTRY_ARTIFACT_BYTES,
                || budget.ensure_time(),
            )?;
            (sha256, size, None)
        };
        budget.ensure_time()?;
        let identity = ObservedRegistryArtifact {
            sha256,
            size,
            model: None,
            bytes,
        };
        observed.insert(stable_identity, identity.clone());
        identity
    };
    verify_observed_artifact(expected, &identity.sha256, identity.size)?;
    Ok(identity.bytes)
}

fn validate_promotion_evidence(
    storage: &RegistryStorage,
    registry: &ModelRegistry,
    generation: &RegistryGeneration,
    bytes: &[u8],
    observed: &mut BTreeMap<StableFileIdentity, ObservedRegistryArtifact>,
    budget: &mut RegistryVerificationBudget,
) -> Result<String, String> {
    let value = parse_unique_json(bytes, "registry promotion decision")?;
    let decision: PromotionDecision = deserialize_closed_json(&value, "promotion decision")?;
    if decision.schema != "phase6_promotion_decision/v1" {
        return Err("registry promotion decision schema mismatch".to_owned());
    }
    for (identifier, field) in [
        (&decision.generation_id, "promotion generationId"),
        (&decision.champion_model_id, "promotion championModelId"),
        (&decision.challenger_model_id, "promotion challengerModelId"),
    ] {
        validate_identifier(identifier, field)?;
    }
    let generation_challenger = generation
        .challenger_model_id
        .as_deref()
        .ok_or_else(|| "completed generation lacks its challenger".to_owned())?;
    if decision.generation_id != generation.generation_id
        || decision.champion_model_id != generation.champion_model_id
        || decision.challenger_model_id != generation_challenger
    {
        return Err("promotion decision identity differs from its registry generation".to_owned());
    }
    validate_artifact_ref(&decision.arena_analysis)?;
    validate_artifact_ref(&decision.policy)?;
    if decision.policy.size > MAX_PROMOTION_POLICY_BYTES {
        return Err(format!(
            "promotion policy exceeds the {MAX_PROMOTION_POLICY_BYTES}-byte contract limit"
        ));
    }
    validate_sha256(&decision.policy_sha256)?;
    validate_sha256(&decision.decision_sha256)?;
    if !matches!(
        decision.decision.as_str(),
        "promoted" | "rejected" | "inconclusive"
    ) {
        return Err("registry promotion decision outcome is invalid".to_owned());
    }
    if decision.weak_evidence && decision.decision == "promoted" {
        return Err("weak promotion evidence cannot promote a challenger".to_owned());
    }
    if decision.reasons.is_empty()
        || decision.reasons.len() > 32
        || decision
            .reasons
            .iter()
            .any(|reason| reason.is_empty() || reason.len() > 128)
    {
        return Err("promotion decision reasons are not bounded".to_owned());
    }
    validate_promotion_numeric_evidence(&decision.evidence)?;
    validate_utc_timestamp(&decision.decided_at, "promotion decidedAt")?;
    validate_json_self_hash(&value, "decisionSha256", &decision.decision_sha256)?;

    let policy_bytes =
        verify_registry_artifact_cached(
            storage,
            &decision.policy,
            observed,
            budget,
            Some(MAX_PROMOTION_POLICY_BYTES),
        )?
            .ok_or_else(|| "promotion policy bytes were not retained".to_owned())?;
    let policy = parse_generation_policy(policy_bytes.as_slice())?;
    let semantic_policy_sha256 = python_canonical_sha256(&serde_json::to_value(&policy).map_err(
        |error| format!("cannot encode promotion policy semantics: {error}"),
    )?)?;
    if semantic_policy_sha256 != decision.policy_sha256 {
        return Err("promotion policy semantic digest differs from the decision".to_owned());
    }

    let analysis_bytes =
        verify_registry_artifact_cached(
            storage,
            &decision.arena_analysis,
            observed,
            budget,
            Some(MAX_REGISTRY_BYTES),
        )?
            .ok_or_else(|| "arena analysis bytes were not retained".to_owned())?;
    let analysis = validate_arena_analysis_binding(
        storage,
        registry,
        generation,
        &decision.decision,
        analysis_bytes.as_slice(),
        observed,
        budget,
    )?;
    let expected_decision = derive_promotion_decision(&analysis, &decision, &policy)?;
    if decision != expected_decision {
        return Err("promotion decision is not its deterministic policy result".to_owned());
    }
    Ok(decision.decision)
}

fn validate_promotion_numeric_evidence(evidence: &PromotionEvidence) -> Result<(), String> {
    if evidence.games > 10_000
        || evidence.decisive_games > evidence.games
        || evidence.illegal_moves > 10_000
        || evidence.crashes > 10_000
    {
        return Err("promotion decision counts exceed their bounded contract".to_owned());
    }
    for rate in [
        evidence.score_rate,
        evidence.initial_score_rate,
        evidence.start_set_score_rate,
        evidence.score_wilson95.lower,
        evidence.score_wilson95.upper,
    ] {
        if !rate.is_finite() || !(0.0..=1.0).contains(&rate) {
            return Err("promotion decision rate is outside 0..=1".to_owned());
        }
    }
    if evidence.score_wilson95.lower > evidence.score_wilson95.upper {
        return Err("promotion decision Wilson interval is reversed".to_owned());
    }
    if let Some(gap) = evidence.side_score_gap
        && (!gap.is_finite() || !(0.0..=1.0).contains(&gap))
    {
        return Err("promotion decision sideScoreGap is outside 0..=1".to_owned());
    }
    Ok(())
}

fn validate_arena_analysis_binding(
    storage: &RegistryStorage,
    registry: &ModelRegistry,
    generation: &RegistryGeneration,
    promotion_outcome: &str,
    bytes: &[u8],
    observed: &mut BTreeMap<StableFileIdentity, ObservedRegistryArtifact>,
    budget: &mut RegistryVerificationBudget,
) -> Result<ArenaAnalysisEnvelope, String> {
    let value = parse_unique_json(bytes, "referenced arena analysis")?;
    let analysis: ArenaAnalysisEnvelope =
        deserialize_closed_json(&value, "arena analysis envelope")?;
    if analysis.schema != "phase6_paired_arena_analysis/v1" {
        return Err("referenced arena analysis schema mismatch".to_owned());
    }
    for (identifier, field) in [
        (&analysis.generation_id, "analysis generationId"),
        (&analysis.champion_model_id, "analysis championModelId"),
        (&analysis.challenger_model_id, "analysis challengerModelId"),
    ] {
        validate_identifier(identifier, field)?;
    }
    if analysis.generation_id != generation.generation_id
        || analysis.champion_model_id != generation.champion_model_id
        || generation.challenger_model_id.as_deref() != Some(analysis.challenger_model_id.as_str())
    {
        return Err("arena analysis identity differs from its registry generation".to_owned());
    }
    for artifact in [&analysis.results, &analysis.plan, &analysis.execution] {
        validate_artifact_ref(artifact)?;
    }
    if generation.arena_manifest.as_ref() != Some(&analysis.results) {
        return Err("arena analysis results do not bind the generation arena manifest".to_owned());
    }
    validate_sha256(&analysis.analysis_sha256)?;
    validate_json_self_hash(&value, "analysisSha256", &analysis.analysis_sha256)?;

    let results_bytes =
        verify_registry_artifact_cached(
            storage,
            &analysis.results,
            observed,
            budget,
            Some(MAX_REGISTRY_BYTES),
        )?
            .ok_or_else(|| "arena results bytes were not retained".to_owned())?;
    let results_value = parse_unique_json(results_bytes.as_slice(), "referenced arena results")?;
    let results: ArenaResults = deserialize_closed_json(&results_value, "arena results")?;
    validate_arena_results(&results)?;
    if results.generation_id != analysis.generation_id
        || results.champion_model_id != analysis.champion_model_id
        || results.challenger_model_id != analysis.challenger_model_id
        || results.plan != analysis.plan
        || results.execution != analysis.execution
    {
        return Err("arena analysis identity differs from its referenced results".to_owned());
    }
    validate_and_recollect_arena_results(
        storage,
        registry,
        generation,
        promotion_outcome,
        &results,
        observed,
        budget,
    )?;

    let expected = analyze_arena_results(&results, analysis.results.clone())?;
    if analysis != expected {
        return Err("arena analysis is not the deterministic result analysis".to_owned());
    }
    Ok(analysis)
}

fn validate_and_recollect_arena_results(
    storage: &RegistryStorage,
    registry: &ModelRegistry,
    generation: &RegistryGeneration,
    promotion_outcome: &str,
    observed_results: &ArenaResults,
    observed: &mut BTreeMap<StableFileIdentity, ObservedRegistryArtifact>,
    budget: &mut RegistryVerificationBudget,
) -> Result<(), String> {
    let plan_bytes = verify_registry_artifact_cached(
        storage,
        &observed_results.plan,
        observed,
        budget,
        Some(MAX_REGISTRY_BYTES),
    )?
    .ok_or_else(|| "paired arena plan bytes were not retained".to_owned())?;
    let plan_value = parse_unique_json(plan_bytes.as_slice(), "paired arena plan")?;
    let plan: PairedArenaPlan = deserialize_closed_json(&plan_value, "paired arena plan")?;
    validate_paired_arena_plan(&plan, &plan_value, generation, registry)?;
    validate_paired_plan_artifacts(
        storage,
        &plan,
        registry,
        promotion_outcome,
        observed,
        budget,
    )?;

    let execution_bytes = verify_registry_artifact_cached(
        storage,
        &observed_results.execution,
        observed,
        budget,
        Some(MAX_REGISTRY_BYTES),
    )?
    .ok_or_else(|| "paired arena execution bytes were not retained".to_owned())?;
    let execution_value =
        parse_unique_json(execution_bytes.as_slice(), "paired arena execution")?;
    let execution: PairedArenaExecution =
        deserialize_closed_json(&execution_value, "paired arena execution")?;
    validate_paired_arena_execution(
        storage,
        &execution,
        &execution_value,
        &plan,
        &observed_results.plan,
        observed,
        budget,
    )?;

    let expected = collect_paired_arena_results(
        storage,
        registry,
        &plan,
        &execution,
        observed_results.plan.clone(),
        observed_results.execution.clone(),
        observed,
        budget,
    )?;
    if observed_results != &expected {
        return Err(
            "arena results are not the deterministic plan/execution report collection".to_owned(),
        );
    }
    Ok(())
}

fn validate_paired_arena_plan(
    plan: &PairedArenaPlan,
    raw: &serde_json::Value,
    generation: &RegistryGeneration,
    registry: &ModelRegistry,
) -> Result<(), String> {
    if plan.schema != "phase6_paired_arena_plan/v1" {
        return Err("paired arena plan schema mismatch".to_owned());
    }
    validate_identifier(&plan.generation_id, "paired plan generationId")?;
    let challenger = generation
        .challenger_model_id
        .as_deref()
        .ok_or_else(|| "paired arena generation lacks a challenger".to_owned())?;
    if plan.generation_id != generation.generation_id
        || plan.champion.model_id != generation.champion_model_id
        || plan.challenger.model_id != challenger
    {
        return Err("paired arena plan identity differs from its registry generation".to_owned());
    }
    validate_sha256(&plan.config_sha256)?;
    validate_sha256(&plan.plan_sha256)?;
    validate_json_self_hash(raw, "planSha256", &plan.plan_sha256)?;
    if plan.game_count != PHASE6_ARENA_GAMES
        || plan.pair_count != PHASE6_ARENA_PAIRS
        || plan.game_count != plan.pair_count * 2
        || plan.normal_start_pairs != PHASE6_INITIAL_PAIRS
        || plan.start_set_pairs != PHASE6_START_SET_PAIRS
        || plan.normal_start_pairs + plan.start_set_pairs != plan.pair_count
        || plan.seed != PHASE6_ARENA_SEED
        || plan.nodes_per_move != PHASE6_ARENA_NODES_PER_MOVE
        || plan.max_workers != PHASE6_ARENA_WORKERS
        || plan.memory_limit_mib != PHASE6_ARENA_MEMORY_MIB
        || !(128..=20 * 1024).contains(&plan.memory_per_worker_mib)
        || plan
            .max_workers
            .checked_mul(plan.memory_per_worker_mib)
            .is_none_or(|total| total > plan.memory_limit_mib)
    {
        return Err("paired arena plan differs from the fixed Phase 6 resource contract".to_owned());
    }
    validate_hex_object_id(&plan.git_commit, "paired plan gitCommit")?;
    validate_paired_model_spec(&plan.champion, registry, "paired plan champion")?;
    validate_paired_model_spec(&plan.challenger, registry, "paired plan challenger")?;
    if plan.champion.model_id == plan.challenger.model_id {
        return Err("paired arena plan champion and challenger must differ".to_owned());
    }
    if plan.engine_build_receipt.is_none() {
        return Err("paired arena plan requires an immutable v2 engine build receipt".to_owned());
    }
    validate_content_addressed_engine_artifact(&plan.engine)?;
    for artifact in paired_plan_artifact_references(plan) {
        validate_artifact_ref(artifact)?;
    }
    if plan.jobs.len() != usize::try_from(plan.pair_count).expect("pair count fits usize") {
        return Err("paired arena plan must contain exactly twenty jobs".to_owned());
    }

    let mut run_root = None;
    let mut start_set_ids = std::collections::BTreeSet::new();
    let mut start_set_sfens = std::collections::BTreeSet::new();
    for (index, job) in plan.jobs.iter().enumerate() {
        validate_paired_arena_job(
            plan,
            job,
            index,
            &mut run_root,
            &mut start_set_ids,
            &mut start_set_sfens,
        )?;
    }
    Ok(())
}

fn paired_plan_artifact_references(plan: &PairedArenaPlan) -> Vec<&RegistryArtifact> {
    let mut artifacts = vec![
        &plan.engine,
        &plan.model_registry,
        &plan.config,
        &plan.start_positions,
        &plan.start_position_validation,
        &plan.dataset_manifest,
        &plan.champion.artifact,
        &plan.challenger.artifact,
    ];
    if let Some(receipt) = &plan.engine_build_receipt {
        artifacts.push(receipt);
    }
    artifacts
}

fn validate_paired_model_spec(
    model: &PairedModelSpec,
    registry: &ModelRegistry,
    context: &str,
) -> Result<(), String> {
    validate_identifier(&model.model_id, context)?;
    validate_artifact_ref(&model.artifact)?;
    if model.evaluator_kind != "neural" {
        return Err(format!("{context} must use the neural evaluator"));
    }
    let registered = registry
        .models
        .iter()
        .find(|candidate| candidate.model_id == model.model_id)
        .ok_or_else(|| format!("{context} is absent from the current registry"))?;
    if registered.artifact != model.artifact || registered.evaluator_kind != model.evaluator_kind {
        return Err(format!("{context} differs from its current registry model"));
    }
    Ok(())
}

fn validate_hex_object_id(value: &str, context: &str) -> Result<(), String> {
    if !(7..=64).contains(&value.len())
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(format!(
            "{context} must be a 7..=64 character lowercase hexadecimal object ID"
        ));
    }
    Ok(())
}

fn validate_paired_arena_job(
    plan: &PairedArenaPlan,
    job: &PairedArenaJob,
    index: usize,
    run_root: &mut Option<String>,
    start_set_ids: &mut std::collections::BTreeSet<String>,
    start_set_sfens: &mut std::collections::BTreeSet<String>,
) -> Result<(), String> {
    validate_identifier(&job.job_id, "paired plan jobId")?;
    validate_identifier(&job.start_position_id, "paired plan startPositionId")?;
    let expected_job_id = format!("pair-{index:04}");
    if job.job_id != expected_job_id
        || job.pair_index != u64::try_from(index).expect("job index fits u64")
    {
        return Err("paired-plan job IDs and indices must be contiguous".to_owned());
    }
    let initial = index < usize::try_from(plan.normal_start_pairs).expect("pair count fits usize");
    let expected_group = if initial { "initial" } else { "start_set" };
    if job.start_group != expected_group {
        return Err("paired-plan start groups are not in their canonical partition".to_owned());
    }
    validate_phase6_plan_sfen(&job.sfen)?;
    if initial {
        if job.start_position_id != "standard-initial" || job.sfen != PHASE6_INITIAL_SFEN {
            return Err("initial paired jobs must use the standard initial position".to_owned());
        }
    } else if job.start_position_id == "standard-initial"
        || !start_set_ids.insert(job.start_position_id.clone())
        || !start_set_sfens.insert(job.sfen.clone())
    {
        return Err("start-set paired jobs must use unique non-initial positions".to_owned());
    }
    if job.seed != derive_paired_job_seed(plan.seed, index, &job.start_position_id) {
        return Err("paired-plan job seed disagrees with deterministic derivation".to_owned());
    }
    let first_game = index
        .checked_mul(2)
        .ok_or_else(|| "paired-plan game index overflow".to_owned())?;
    if job.game_ids
        != [
            format!("game-{first_game:06}"),
            format!("game-{:06}", first_game + 1),
        ]
        || job.model_a_color_order != ["black", "white"]
    {
        return Err("paired-plan game IDs or color order are not canonical".to_owned());
    }
    for path in [
        &job.output_dir,
        &job.report_path,
        &job.quarantine_path,
    ]
    .into_iter()
    .chain(job.csa_paths.iter())
    {
        validate_relative_artifact_path(path)?;
    }
    let output = job
        .output_dir
        .to_str()
        .ok_or_else(|| "paired job output path must be UTF-8".to_owned())?;
    let suffix = format!("/jobs/{}", job.job_id);
    let candidate_root = output
        .strip_suffix(&suffix)
        .ok_or_else(|| "paired job output path does not use the jobs layout".to_owned())?;
    let expected_run_suffix = format!("/{}/arena", plan.generation_id);
    if !candidate_root.ends_with(&expected_run_suffix) {
        return Err("paired job output root does not bind generation and arena kind".to_owned());
    }
    match run_root {
        Some(expected) if expected != candidate_root => {
            return Err("paired arena jobs do not share one run root".to_owned());
        }
        None => *run_root = Some(candidate_root.to_owned()),
        Some(_) => {}
    }
    let expected_report_path = format!("{output}/arena-report.json");
    let expected_quarantine_path =
        format!("{candidate_root}/quarantine/{}.json", job.job_id);
    if job.report_path != expected_report_path
        || job.csa_paths
            != [
                PathBuf::from(format!("{output}/games/game-000001.csa")),
                PathBuf::from(format!("{output}/games/game-000002.csa")),
            ]
        || job.quarantine_path != expected_quarantine_path
    {
        return Err("paired job artifact paths disagree with its output layout".to_owned());
    }
    validate_paired_arena_command(plan, job)
}

fn validate_phase6_plan_sfen(value: &str) -> Result<(), String> {
    if !value.is_ascii() || value.contains(['\r', '\n']) {
        return Err("paired-plan SFEN must be canonical ASCII".to_owned());
    }
    let position =
        parse_sfen(value).map_err(|error| format!("invalid paired-plan SFEN: {error}"))?;
    if position.move_number() != 1 || to_sfen(&position) != value {
        return Err("paired-plan SFEN must be canonical with move number 1".to_owned());
    }
    Ok(())
}

fn derive_paired_job_seed(base_seed: u64, index: usize, start_id: &str) -> u64 {
    let digest = sha256_bytes(
        format!("phase6\0{base_seed}\0arena\0{index}\0{start_id}").as_bytes(),
    );
    u64::from_str_radix(&digest[..16], 16).expect("SHA-256 prefix is hexadecimal")
        % (MAX_JSON_SAFE_INTEGER + 1)
}

fn validate_paired_arena_command(
    plan: &PairedArenaPlan,
    job: &PairedArenaJob,
) -> Result<(), String> {
    if job.command.kind != "engine_arena" || !(1..=172_800).contains(&job.command.timeout_seconds) {
        return Err("paired job command kind or timeout is invalid".to_owned());
    }
    let options = paired_arena_command_options(&job.command, &plan.engine)?;
    let expected = [
        ("--games", "2".to_owned()),
        ("--player-a", "neural".to_owned()),
        ("--player-b", "neural".to_owned()),
        ("--nodes", plan.nodes_per_move.to_string()),
        ("--max-plies", PHASE6_ARENA_MAX_PLIES.to_string()),
        ("--seed", job.seed.to_string()),
        ("--sfen", job.sfen.clone()),
        ("--git-commit", plan.git_commit.clone()),
        (
            "--output-dir",
            job.output_dir
                .to_str()
                .ok_or_else(|| "paired job output path must be UTF-8".to_owned())?
                .to_owned(),
        ),
        (
            "--a-model",
            artifact_path_text(&plan.challenger.artifact)?.to_owned(),
        ),
        (
            "--b-model",
            artifact_path_text(&plan.champion.artifact)?.to_owned(),
        ),
    ];
    for (name, value) in expected {
        if options.get(name) != Some(&value) {
            return Err(format!("paired job command {name} differs from its plan"));
        }
    }
    for (left, right, minimum, maximum) in [
        ("--a-depth", "--b-depth", 1_u64, 64_u64),
        ("--a-hash-mb", "--b-hash-mb", 1_u64, 1_024_u64),
    ] {
        let left_value = parse_ascii_decimal_option(&options, left, minimum, maximum)?;
        let right_value = parse_ascii_decimal_option(&options, right, minimum, maximum)?;
        if left_value != right_value {
            return Err("paired job command uses asymmetric player limits".to_owned());
        }
    }
    Ok(())
}

fn paired_arena_command_options(
    command: &PairedArenaCommand,
    engine: &RegistryArtifact,
) -> Result<BTreeMap<String, String>, String> {
    if !(2..=MAX_PAIRED_COMMAND_ARGUMENTS).contains(&command.argv.len()) {
        return Err("paired job command has an invalid argument count".to_owned());
    }
    let mut total_bytes = 0_usize;
    for argument in &command.argv {
        if argument.is_empty() || argument.contains('\0') || argument.len() > 4_096 {
            return Err("paired job command contains an invalid argument".to_owned());
        }
        total_bytes = total_bytes
            .checked_add(argument.len())
            .ok_or_else(|| "paired job command byte count overflow".to_owned())?;
    }
    if total_bytes > MAX_PAIRED_COMMAND_ARGUMENT_BYTES
        || command.argv.first().map(String::as_str) != Some(artifact_path_text(engine)?)
        || command.argv.get(1).map(String::as_str) != Some("arena")
        || !(command.argv.len() - 2).is_multiple_of(2)
        || command.argv.iter().any(|argument| argument == "--resume")
    {
        return Err("paired job command is not one bounded repository arena invocation".to_owned());
    }
    let required = [
        "--games",
        "--player-a",
        "--player-b",
        "--a-depth",
        "--b-depth",
        "--a-hash-mb",
        "--b-hash-mb",
        "--nodes",
        "--max-plies",
        "--seed",
        "--sfen",
        "--git-commit",
        "--output-dir",
        "--a-model",
        "--b-model",
    ];
    let mut options = BTreeMap::new();
    for pair in command.argv[2..].chunks_exact(2) {
        if !required.contains(&pair[0].as_str())
            || options.insert(pair[0].clone(), pair[1].clone()).is_some()
        {
            return Err("paired job command contains an unknown or duplicate option".to_owned());
        }
    }
    if options.len() != required.len() {
        return Err("paired job command lacks an exact required option".to_owned());
    }
    Ok(options)
}

fn artifact_path_text(artifact: &RegistryArtifact) -> Result<&str, String> {
    artifact
        .path
        .to_str()
        .ok_or_else(|| "artifact path must be UTF-8".to_owned())
}

fn parse_ascii_decimal_option(
    options: &BTreeMap<String, String>,
    name: &str,
    minimum: u64,
    maximum: u64,
) -> Result<u64, String> {
    let value = options
        .get(name)
        .ok_or_else(|| format!("paired job command lacks {name}"))?;
    if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(format!(
            "paired job command {name} is not canonical decimal"
        ));
    }
    let parsed = value
        .parse::<u64>()
        .map_err(|_| format!("paired job command {name} is out of range"))?;
    if parsed.to_string() != *value {
        return Err(format!(
            "paired job command {name} is not canonical decimal"
        ));
    }
    if !(minimum..=maximum).contains(&parsed) {
        return Err(format!("paired job command {name} is out of range"));
    }
    Ok(parsed)
}

fn validate_paired_plan_artifacts(
    storage: &RegistryStorage,
    plan: &PairedArenaPlan,
    current_registry: &ModelRegistry,
    promotion_outcome: &str,
    observed: &mut BTreeMap<StableFileIdentity, ObservedRegistryArtifact>,
    budget: &mut RegistryVerificationBudget,
) -> Result<(), String> {
    for artifact in [
        &plan.engine,
        &plan.champion.artifact,
        &plan.challenger.artifact,
    ] {
        let _ = verify_registry_artifact_cached(storage, artifact, observed, budget, None)?;
    }
    let reference = plan
        .engine_build_receipt
        .as_ref()
        .ok_or_else(|| "paired arena plan lacks its v2 engine build receipt".to_owned())?;
    validate_engine_build_receipt_artifact(storage, plan, reference, observed, budget)?;
    let config_bytes = verify_registry_artifact_cached(
        storage,
        &plan.config,
        observed,
        budget,
        Some(MAX_PROMOTION_POLICY_BYTES),
    )?
    .ok_or_else(|| "paired arena config bytes were not retained".to_owned())?;
    let config = parse_phase6_selfplay_config(config_bytes.as_slice())?;
    validate_phase6_plan_config_binding(plan, &config)?;
    validate_phase6_start_artifacts(storage, plan, observed, budget)?;
    let snapshot_bytes = verify_registry_artifact_cached(
        storage,
        &plan.model_registry,
        observed,
        budget,
        Some(MAX_REGISTRY_BYTES),
    )?
    .ok_or_else(|| "paired plan registry snapshot bytes were not retained".to_owned())?;
    let snapshot_value =
        parse_unique_json(snapshot_bytes.as_slice(), "paired plan registry snapshot")?;
    let snapshot: ModelRegistry =
        deserialize_closed_json(&snapshot_value, "paired plan registry snapshot")?;
    validate_model_registry(&snapshot)?;
    if snapshot.champion_model_id.as_deref() != Some(plan.champion.model_id.as_str())
        || snapshot.challenger_model_id.as_deref() != Some(plan.challenger.model_id.as_str())
    {
        return Err("paired plan registry snapshot has different active models".to_owned());
    }
    let active = snapshot
        .generations
        .iter()
        .filter(|generation| generation.status == "arena")
        .collect::<Vec<_>>();
    if active.len() != 1
        || active[0].generation_id != plan.generation_id
        || snapshot.generations.last().map(|generation| &generation.generation_id)
            != Some(&active[0].generation_id)
    {
        return Err("paired plan generation is not the snapshot's final active generation".to_owned());
    }
    validate_paired_model_spec(&plan.champion, &snapshot, "snapshot champion")?;
    validate_paired_model_spec(&plan.challenger, &snapshot, "snapshot challenger")?;
    validate_registry_snapshot_transition(&snapshot, current_registry, promotion_outcome)?;
    Ok(())
}

fn validate_registry_snapshot_transition(
    snapshot: &ModelRegistry,
    current: &ModelRegistry,
    promotion_outcome: &str,
) -> Result<(), String> {
    let target_index = snapshot
        .generations
        .len()
        .checked_sub(1)
        .ok_or_else(|| "paired plan registry snapshot has no generation".to_owned())?;
    if snapshot.models.len() > current.models.len()
        || snapshot.generations.len() > current.generations.len()
        || snapshot.models != current.models[..snapshot.models.len()]
        || snapshot.generations[..target_index] != current.generations[..target_index]
    {
        return Err(
            "paired plan registry snapshot is not immutable current registry history".to_owned(),
        );
    }

    let before = &snapshot.generations[target_index];
    let after = &current.generations[target_index];
    let immutable_generation_fields_match = before.generation_id == after.generation_id
        && before.parent_generation_id == after.parent_generation_id
        && before.champion_model_id == after.champion_model_id
        && before.challenger_model_id == after.challenger_model_id
        && before.selfplay_manifest == after.selfplay_manifest
        && before.teacher_labeling_manifest == after.teacher_labeling_manifest
        && before.training_run_manifest == after.training_run_manifest
        && before.created_at == after.created_at;
    if !immutable_generation_fields_match
        || before.status != "arena"
        || before.arena_manifest.is_some()
        || before.promotion_decision.is_some()
        || after.status != "complete"
        || after.arena_manifest.is_none()
        || after.promotion_decision.is_none()
    {
        return Err(
            "paired plan registry snapshot does not exactly precede generation finalization"
                .to_owned(),
        );
    }

    let later_generations = &current.generations[snapshot.generations.len()..];
    let later_models = &current.models[snapshot.models.len()..];
    if later_models.len() != later_generations.len()
        || later_models
            .iter()
            .zip(later_generations)
            .any(|(model, generation)| {
                generation.challenger_model_id.as_deref() != Some(model.model_id.as_str())
                    || model.generation_id != generation.generation_id
            })
    {
        return Err(
            "current registry is not a sequence of exact append/finalize lifecycle deltas"
                .to_owned(),
        );
    }

    let target_outcome = if promotion_outcome == "promoted" {
        after
            .challenger_model_id
            .as_deref()
            .ok_or_else(|| "promoted generation lacks its challenger".to_owned())?
    } else {
        after.champion_model_id.as_str()
    };
    if let Some(next) = later_generations.first() {
        if next.champion_model_id != target_outcome {
            return Err(
                "generation after the paired plan does not inherit its promotion outcome"
                    .to_owned(),
            );
        }
    } else if current.champion_model_id.as_deref() != Some(target_outcome)
        || current.challenger_model_id.is_some()
    {
        return Err(
            "final registry top-level models disagree with the paired plan outcome".to_owned(),
        );
    }

    let later_finalizations = later_generations
        .iter()
        .filter(|generation| generation.status == "complete")
        .count();
    let revision_delta = 1_usize
        .checked_add(later_generations.len())
        .and_then(|value| value.checked_add(later_finalizations))
        .and_then(|value| u64::try_from(value).ok())
        .ok_or_else(|| "model registry revision delta overflow".to_owned())?;
    if snapshot.revision.checked_add(revision_delta) != Some(current.revision) {
        return Err(
            "current registry revision is not the exact paired-plan lifecycle successor"
                .to_owned(),
        );
    }
    Ok(())
}

fn validate_engine_build_receipt_artifact(
    storage: &RegistryStorage,
    plan: &PairedArenaPlan,
    reference: &RegistryArtifact,
    observed: &mut BTreeMap<StableFileIdentity, ObservedRegistryArtifact>,
    budget: &mut RegistryVerificationBudget,
) -> Result<(), String> {
    let bytes = verify_registry_artifact_cached(
        storage,
        reference,
        observed,
        budget,
        Some(MAX_REGISTRY_BYTES),
    )?
    .ok_or_else(|| "engine build receipt bytes were not retained".to_owned())?;
    let value = parse_unique_json(bytes.as_slice(), "engine build receipt")?;
    let receipt = validate_engine_build_receipt_for_inspection(&value)?;
    let EngineBuildReceipt::ContentAddressedV2(receipt) = receipt else {
        return Err(
            "Phase 6 authorization requires open_shogi_engine_build_receipt/v2".to_owned(),
        );
    };
    if !commit_id_matches(&receipt.git_commit, &plan.git_commit)
        || receipt.binary != plan.engine
    {
        return Err("engine build receipt differs from its paired plan".to_owned());
    }
    let expected_reference_path = PathBuf::from(format!(
        "local/build-receipts/open-shogi-cli/{}.json",
        receipt.receipt_sha256
    ));
    if reference.path != expected_reference_path {
        return Err("engine build receipt path is not content-addressed".to_owned());
    }
    Ok(())
}

fn validate_engine_build_receipt_for_inspection(
    value: &serde_json::Value,
) -> Result<EngineBuildReceipt, String> {
    let schema = value
        .as_object()
        .and_then(|object| object.get("schema"))
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| "engine build receipt lacks a string schema".to_owned())?;
    match schema {
        "open_shogi_engine_build_receipt/v1" => {
            let receipt: EngineBuildReceiptV1 =
                deserialize_closed_json(value, "historical engine build receipt v1")?;
            validate_engine_build_receipt_v1(value, &receipt)?;
            Ok(EngineBuildReceipt::HistoricalV1(receipt))
        }
        "open_shogi_engine_build_receipt/v2" => {
            let receipt: EngineBuildReceiptV2 =
                deserialize_closed_json(value, "engine build receipt v2")?;
            validate_engine_build_receipt_v2(value, &receipt)?;
            Ok(EngineBuildReceipt::ContentAddressedV2(receipt))
        }
        _ => Err("unsupported engine build receipt schema".to_owned()),
    }
}

fn validate_engine_build_receipt_v1(
    value: &serde_json::Value,
    receipt: &EngineBuildReceiptV1,
) -> Result<(), String> {
    if receipt.schema != "open_shogi_engine_build_receipt/v1"
        || receipt.binary.path.as_path() != Path::new("target/release/open-shogi-cli")
        || receipt.binary.size == 0
        || !valid_engine_build_bounds(
            receipt.source_files,
            receipt.source_bytes,
            &receipt.build_command,
            false,
        )
    {
        return Err("historical engine build receipt v1 is invalid".to_owned());
    }
    validate_receipt_commit(&receipt.git_commit)?;
    validate_artifact_ref(&receipt.binary)?;
    validate_sha256(&receipt.source_tree_sha256)?;
    validate_sha256(&receipt.receipt_sha256)?;
    validate_phase3_text(&receipt.cargo_version, 256, "receipt cargoVersion", true)?;
    validate_phase3_text(&receipt.rustc_version, 256, "receipt rustcVersion", true)?;
    validate_utc_timestamp(&receipt.built_at, "engine build receipt builtAt")?;
    validate_json_self_hash(value, "receiptSha256", &receipt.receipt_sha256)
}

fn validate_engine_build_receipt_v2(
    value: &serde_json::Value,
    receipt: &EngineBuildReceiptV2,
) -> Result<(), String> {
    if receipt.schema != "open_shogi_engine_build_receipt/v2"
        || receipt.binary.size == 0
        || !valid_engine_build_bounds(
            receipt.source_files,
            receipt.source_bytes,
            &receipt.build_command,
            true,
        )
    {
        return Err("engine build receipt v2 is invalid".to_owned());
    }
    validate_receipt_commit(&receipt.git_commit)?;
    validate_content_addressed_engine_artifact(&receipt.binary)?;
    validate_sha256(&receipt.source_tree_sha256)?;
    validate_sha256(&receipt.receipt_sha256)?;
    validate_engine_build_tool(&receipt.cargo_tool, "cargoTool")?;
    validate_engine_build_tool(&receipt.rustc_tool, "rustcTool")?;
    validate_engine_build_runtime_tree(&receipt.rustc_runtime_tree)?;
    validate_json_self_hash(value, "receiptSha256", &receipt.receipt_sha256)
}

fn valid_engine_build_bounds(
    source_files: u64,
    source_bytes: u64,
    build_command: &[String],
    offline: bool,
) -> bool {
    (1..=4_096).contains(&source_files)
        && (1..=256 * 1024 * 1024).contains(&source_bytes)
        && if offline {
            build_command
                == [
                    "cargo",
                    "build",
                    "--locked",
                    "--release",
                    "--offline",
                    "--jobs",
                    "4",
                    "-p",
                    "open-shogi-cli",
                ]
        } else {
            build_command
                == [
                "cargo",
                "build",
                "--locked",
                "--release",
                "-p",
                "open-shogi-cli",
            ]
        }
}

fn validate_content_addressed_engine_artifact(
    artifact: &RegistryArtifact,
) -> Result<(), String> {
    validate_artifact_ref(artifact)?;
    let expected = PathBuf::from(format!(
        "local/builds/open-shogi-cli/{}/open-shogi-cli",
        artifact.sha256
    ));
    if artifact.path != expected || artifact.size == 0 {
        return Err("engine binary path is not content-addressed".to_owned());
    }
    Ok(())
}

fn validate_engine_build_tool(tool: &EngineBuildToolIdentity, field: &str) -> Result<(), String> {
    if tool.path.is_empty()
        || tool.path.len() > 4_096
        || tool.path.contains(['\0', '\r', '\n'])
        || !Path::new(&tool.path).is_absolute()
        || !(1..=512 * 1024 * 1024).contains(&tool.size)
    {
        return Err(format!("engine build receipt {field} identity is invalid"));
    }
    validate_sha256(&tool.sha256)?;
    validate_phase3_text(&tool.version, 256, field, true)
}

fn validate_receipt_commit(value: &str) -> Result<(), String> {
    if !matches!(value.len(), 40 | 64)
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err("engine build receipt gitCommit must be a 40- or 64-digit lowercase hex ID"
            .to_owned());
    }
    Ok(())
}

fn validate_engine_build_runtime_tree(tree: &EngineBuildRuntimeTree) -> Result<(), String> {
    validate_sha256(&tree.tree_sha256)?;
    if !(2..=512).contains(&tree.files) || !(1..=1024 * 1024 * 1024).contains(&tree.bytes) {
        return Err("engine build receipt rustcRuntimeTree is outside its bounds".to_owned());
    }
    Ok(())
}

fn commit_id_matches(left: &str, right: &str) -> bool {
    matches!(left.len(), 40 | 64)
        && left
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        && (left == right || ((7..left.len()).contains(&right.len()) && left.starts_with(right)))
}

fn parse_phase6_selfplay_config(bytes: &[u8]) -> Result<Phase6SelfplayConfig, String> {
    let text = std::str::from_utf8(bytes)
        .map_err(|error| format!("paired arena config is not UTF-8: {error}"))?;
    let config: Phase6SelfplayConfig =
        toml::from_str(text).map_err(|error| format!("invalid paired arena config: {error}"))?;
    validate_phase6_selfplay_config(&config)?;
    Ok(config)
}

fn validate_phase6_selfplay_config(config: &Phase6SelfplayConfig) -> Result<(), String> {
    let run = &config.run;
    if config.schema_version != 1
        || run.games != PHASE6_ARENA_GAMES
        || !run.games.is_multiple_of(2)
        || run.normal_start_pairs != PHASE6_INITIAL_PAIRS
        || run.start_set_pairs != PHASE6_START_SET_PAIRS
        || run.normal_start_pairs + run.start_set_pairs != run.games / 2
        || run.seed != PHASE6_ARENA_SEED
        || run.nodes_per_move != PHASE6_ARENA_NODES_PER_MOVE
        || run.max_plies != PHASE6_ARENA_MAX_PLIES
        || !(1..=64).contains(&run.search_depth)
        || !(1..=1_024).contains(&run.hash_mib)
    {
        return Err("paired arena config run differs from the bounded Phase 6 contract".to_owned());
    }
    let resources = &config.resources;
    if resources.workers != PHASE6_ARENA_WORKERS
        || resources.memory_limit_mib != PHASE6_ARENA_MEMORY_MIB
        || !(128..=PHASE6_ARENA_MEMORY_MIB).contains(&resources.memory_per_worker_mib)
        || resources
            .workers
            .checked_mul(resources.memory_per_worker_mib)
            .is_none_or(|memory| memory > resources.memory_limit_mib)
        || !(1..=86_400).contains(&resources.game_timeout_seconds)
        || !(1..=172_800).contains(&resources.command_timeout_seconds)
    {
        return Err("paired arena config resources are outside their bounds".to_owned());
    }
    for path in [
        &config.paths.engine_cli,
        &config.paths.output_root,
        &config.paths.start_positions_manifest,
    ] {
        validate_relative_artifact_path(path)?;
    }
    let hard = &config.hard_positions;
    if !(1..=100_000).contains(&hard.teacher_drop_cp)
        || !(1..=100_000).contains(&hard.evaluation_disagreement_cp)
        || hard.candidate_gap_cp > 100_000
        || !(1..=1_000_000_000).contains(&hard.minimum_search_nodes)
        || hard.max_additional_labels > 10_000
        || hard.teacher_label_limit != 10_000
    {
        return Err("paired arena config hard-position bounds are invalid".to_owned());
    }
    let replay = &config.replay;
    if !(1..=10_000_000).contains(&replay.capacity)
        || replay.minimum_older_positions > replay.capacity
        || replay.dedup_key != "canonical_sfen"
    {
        return Err("paired arena config replay bounds are invalid".to_owned());
    }
    Ok(())
}

fn validate_phase6_plan_config_binding(
    plan: &PairedArenaPlan,
    config: &Phase6SelfplayConfig,
) -> Result<(), String> {
    let semantic = python_canonical_sha256(
        &serde_json::to_value(config)
            .map_err(|error| format!("cannot serialize paired arena config: {error}"))?,
    )?;
    if plan.config_sha256 != semantic
        || plan.game_count != config.run.games
        || plan.pair_count != config.run.games / 2
        || plan.normal_start_pairs != config.run.normal_start_pairs
        || plan.start_set_pairs != config.run.start_set_pairs
        || plan.seed != config.run.seed
        || plan.nodes_per_move != config.run.nodes_per_move
        || plan.max_workers != config.resources.workers
        || plan.memory_limit_mib != config.resources.memory_limit_mib
        || plan.memory_per_worker_mib != config.resources.memory_per_worker_mib
        || plan.start_positions.path != config.paths.start_positions_manifest
    {
        return Err("paired arena plan differs from its semantic config artifact".to_owned());
    }
    let run_root = config
        .paths
        .output_root
        .join(&plan.generation_id)
        .join("arena");
    for (index, job) in plan.jobs.iter().enumerate() {
        let expected_output = run_root
            .join("jobs")
            .join(format!("pair-{index:04}"));
        let options = paired_arena_command_options(&job.command, &plan.engine)?;
        if job.output_dir != expected_output
            || job.command.timeout_seconds != config.resources.game_timeout_seconds
            || parse_ascii_decimal_option(&options, "--a-depth", 1, 64)?
                != config.run.search_depth
            || parse_ascii_decimal_option(&options, "--b-depth", 1, 64)?
                != config.run.search_depth
            || parse_ascii_decimal_option(&options, "--a-hash-mb", 1, 1_024)?
                != config.run.hash_mib
            || parse_ascii_decimal_option(&options, "--b-hash-mb", 1, 1_024)?
                != config.run.hash_mib
        {
            return Err("paired arena job differs from its config artifact".to_owned());
        }
    }
    Ok(())
}

fn validate_phase6_start_artifacts(
    storage: &RegistryStorage,
    plan: &PairedArenaPlan,
    observed: &mut BTreeMap<StableFileIdentity, ObservedRegistryArtifact>,
    budget: &mut RegistryVerificationBudget,
) -> Result<(), String> {
    let starts_bytes = verify_registry_artifact_cached(
        storage,
        &plan.start_positions,
        observed,
        budget,
        Some(MAX_REGISTRY_BYTES),
    )?
    .ok_or_else(|| "paired start-position bytes were not retained".to_owned())?;
    let starts_value = parse_unique_json(starts_bytes.as_slice(), "paired start positions")?;
    let starts: Phase6StartPositions =
        deserialize_closed_json(&starts_value, "paired start positions")?;
    validate_phase6_start_positions(plan, &starts)?;
    let manifest_bytes = verify_registry_artifact_cached(
        storage,
        &starts.dataset_manifest,
        observed,
        budget,
        Some(MAX_REGISTRY_BYTES),
    )?
    .ok_or_else(|| "Phase 3 dataset manifest bytes were not retained".to_owned())?;
    let manifest_value = parse_unique_json(
        manifest_bytes.as_slice(),
        "paired Phase 3 dataset manifest",
    )?;
    let manifest: Phase3DatasetManifest =
        deserialize_closed_json(&manifest_value, "paired Phase 3 dataset manifest")?;
    let _ = verify_registry_artifact_cached(
        storage,
        &starts.source_positions,
        observed,
        budget,
        None,
    )?;
    validate_phase3_dataset_manifest(&starts, &manifest)?;
    verify_phase3_normalization_report(storage, &starts, &manifest, observed, budget)?;
    validate_phase3_start_position_derivation(storage, &starts, &manifest, observed, budget)?;

    let validation_bytes = verify_registry_artifact_cached(
        storage,
        &plan.start_position_validation,
        observed,
        budget,
        Some(MAX_REGISTRY_BYTES),
    )?
    .ok_or_else(|| "paired start-validation bytes were not retained".to_owned())?;
    let validation_value =
        parse_unique_json(validation_bytes.as_slice(), "paired start-position validation")?;
    let validation: Phase6StartValidation =
        deserialize_closed_json(&validation_value, "paired start-position validation")?;
    validate_phase6_start_validation(storage, plan, &starts, &validation, observed, budget)
}

fn verify_phase3_normalization_report(
    storage: &RegistryStorage,
    starts: &Phase6StartPositions,
    manifest: &Phase3DatasetManifest,
    observed: &mut BTreeMap<StableFileIdentity, ObservedRegistryArtifact>,
    budget: &mut RegistryVerificationBudget,
) -> Result<(), String> {
    let parent = starts
        .dataset_manifest
        .path
        .parent()
        .ok_or_else(|| "Phase 3 manifest path has no parent".to_owned())?;
    let recorded = manifest
        .artifacts
        .get("normalization-report.json")
        .ok_or_else(|| "Phase 3 normalization report is absent from its manifest".to_owned())?;
    let reference = RegistryArtifact {
        path: parent.join("normalization-report.json"),
        sha256: recorded.sha256.clone(),
        size: recorded.size,
    };
    validate_artifact_ref(&reference)?;
    let _ = verify_registry_artifact_cached(storage, &reference, observed, budget, None)?;
    Ok(())
}

#[expect(
    clippy::too_many_lines,
    reason = "the validator mirrors the closed Phase 3 production manifest contract"
)]
fn validate_phase3_dataset_manifest(
    starts: &Phase6StartPositions,
    manifest: &Phase3DatasetManifest,
) -> Result<(), String> {
    if manifest.schema != "phase3_dataset_manifest/v1" {
        return Err("unsupported Phase 3 dataset manifest schema".to_owned());
    }
    validate_identifier(&manifest.dataset_id, "Phase 3 datasetId")?;
    let source = &manifest.source;
    validate_identifier(&source.source_id, "Phase 3 sourceId")?;
    if manifest.dataset_id != source.source_id
        && !manifest
            .dataset_id
            .strip_prefix(&source.source_id)
            .is_some_and(|suffix| suffix.starts_with('-'))
    {
        return Err("Phase 3 datasetId is not namespaced by its approved sourceId".to_owned());
    }
    for (value, field) in [
        (&source.name, "source.name"),
        (&source.official_base, "source.officialBase"),
        (&source.adapter, "source.adapter"),
        (&source.license, "source.license"),
        (&source.last_reviewed, "source.lastReviewed"),
    ] {
        validate_phase3_text(value, 4_096, field, false)?;
    }
    if !matches!(source.adapter.as_str(), "aobazero" | "aobazero_csa")
        || (!source.official_base.starts_with("https://")
            && !source.official_base.starts_with("http://"))
        || source.license_evidence.is_empty()
        || !source.machine_learning_allowed
    {
        return Err("Phase 3 dataset source lacks approved license evidence".to_owned());
    }
    for evidence in &source.license_evidence {
        for (value, field) in [
            (&evidence.url, "license evidence URL"),
            (&evidence.quote, "license evidence quote"),
        ] {
            validate_phase3_text(value, 4_096, field, true)?;
        }
        let local_path = evidence
            .local_path
            .to_str()
            .ok_or_else(|| "license evidence local path is not UTF-8".to_owned())?;
        validate_phase3_text(local_path, 4_096, "license evidence local path", true)?;
        validate_phase3_portable_relative_path(&evidence.local_path, 4_096)?;
        if !evidence.url.starts_with("https://") && !evidence.url.starts_with("http://") {
            return Err("Phase 3 license evidence URL is invalid".to_owned());
        }
    }

    let config = &manifest.config;
    if config.schema != "phase3_normalization_config/v1"
        || config.dataset_id != manifest.dataset_id
        || !(1..=3_600).contains(&config.exporter_timeout_seconds)
        || !(1..=100).contains(&config.max_games)
        || !(1..=204_900).contains(&config.max_positions)
        || !(1..=16 * 1024 * 1024).contains(&config.max_raw_bytes)
        || !(1..=2_048).contains(&config.terminal_tail_positions)
    {
        return Err("Phase 3 normalization config is invalid".to_owned());
    }
    let split = &config.split;
    if split.schema != "phase3_game_split/v1"
        || split.salt.is_empty()
        || split.salt.len() > 1_024
        || split.salt.contains('\0')
        || sha256_bytes(split.salt.as_bytes()) != split.salt_sha256
        || split.test_basis_points > 10_000
        || split.validation_basis_points > 10_000
        || split
            .test_basis_points
            .checked_add(split.validation_basis_points)
            .is_none_or(|total| total >= 10_000)
    {
        return Err("Phase 3 split policy is invalid".to_owned());
    }
    validate_sha256(&split.salt_sha256)?;

    let counts = &manifest.counts;
    if !(1..=10_000).contains(&counts.games)
        || !(1..=100_000_000).contains(&counts.positions)
        || counts.positions > u64::try_from(MAX_PHASE3_POSITIONS).expect("bound fits u64")
        || counts.games > config.max_games
        || counts.positions > config.max_positions
    {
        return Err("Phase 3 dataset counts are outside their bounds".to_owned());
    }
    let expected_games = usize::try_from(counts.games).expect("bounded game count fits usize");
    for (hashes, field) in [
        (&manifest.raw_object_sha256, "rawObjectSha256"),
        (&manifest.canonical_game_sha256, "canonicalGameSha256"),
    ] {
        if hashes.len() != expected_games
            || hashes.windows(2).any(|window| window[0] >= window[1])
            || hashes.iter().any(|hash| validate_sha256(hash).is_err())
        {
            return Err(format!("Phase 3 dataset manifest {field} is invalid"));
        }
    }

    if manifest.evidence_snapshots.is_empty() {
        return Err("Phase 3 dataset manifest lacks evidence snapshots".to_owned());
    }
    let mut snapshot_identities = std::collections::BTreeSet::new();
    let mut previous_snapshot_identity = None;
    let mut snapshot_urls = std::collections::BTreeSet::new();
    let mut snapshot_id_definitions = BTreeMap::<&str, (&str, &str)>::new();
    let mut snapshot_url_owners = BTreeMap::<&str, &str>::new();
    let mut snapshot_path_owners = BTreeMap::<&Path, &str>::new();
    for snapshot in &manifest.evidence_snapshots {
        for (value, field) in [
            (&snapshot.evidence_id, "evidence snapshot ID"),
            (&snapshot.url, "evidence snapshot URL"),
            (&snapshot.retrieved_at, "evidence snapshot retrieval time"),
            (&snapshot.content_type, "evidence snapshot content type"),
        ] {
            validate_phase3_text(value, 4_096, field, true)?;
        }
        let object_path = snapshot
            .object_path
            .to_str()
            .ok_or_else(|| "evidence snapshot object path is not UTF-8".to_owned())?;
        validate_phase3_text(object_path, 4_096, "evidence snapshot object path", true)?;
        validate_phase3_portable_relative_path(&snapshot.object_path, 4_096)?;
        validate_utc_timestamp(&snapshot.retrieved_at, "evidence snapshot retrieved_at")?;
        validate_sha256(&snapshot.sha256)?;
        let expected_path = phase3_evidence_object_path(&snapshot.sha256)?;
        let identity = phase3_evidence_identity(snapshot)?;
        let definition = (snapshot.url.as_str(), snapshot.content_type.as_str());
        if snapshot_id_definitions
            .insert(snapshot.evidence_id.as_str(), definition)
            .is_some_and(|previous| previous != definition)
            || snapshot_url_owners
                .insert(snapshot.url.as_str(), snapshot.evidence_id.as_str())
                .is_some_and(|previous| previous != snapshot.evidence_id.as_str())
            || snapshot_path_owners
                .insert(snapshot.object_path.as_path(), snapshot.evidence_id.as_str())
                .is_some_and(|previous| previous != snapshot.evidence_id.as_str())
        {
            return Err(
                "Phase 3 evidence snapshot ID, URL, or object ownership is inconsistent"
                    .to_owned(),
            );
        }
        if (!snapshot.url.starts_with("https://") && !snapshot.url.starts_with("http://"))
            || !(1..=16 * 1024 * 1024).contains(&snapshot.size)
            || snapshot.object_path != expected_path
            || previous_snapshot_identity
                .as_ref()
                .is_some_and(|previous| previous >= &identity)
            || !snapshot_identities.insert(identity.clone())
        {
            return Err("Phase 3 evidence snapshot is invalid".to_owned());
        }
        snapshot_urls.insert(snapshot.url.as_str());
        previous_snapshot_identity = Some(identity);
    }
    if manifest
        .source
        .license_evidence
        .iter()
        .any(|evidence| !snapshot_urls.contains(evidence.url.as_str()))
    {
        return Err("Phase 3 source license evidence lacks a durable snapshot".to_owned());
    }

    let expected_artifacts = [
        ("games-00000.jsonl.gz", counts.games),
        ("normalization-report.json", 1),
        ("positions-00000.jsonl.gz", counts.positions),
    ];
    if manifest.artifacts.len() != expected_artifacts.len()
        || !expected_artifacts
            .iter()
            .all(|(name, _)| manifest.artifacts.contains_key(*name))
    {
        return Err("Phase 3 dataset manifest artifact set is not canonical".to_owned());
    }
    for (name, artifact) in &manifest.artifacts {
        if name.is_empty()
            || artifact.size == 0
            || artifact.size > MAX_REGISTRY_ARTIFACT_BYTES
            || artifact.records > 100_000_000
        {
            return Err("Phase 3 dataset manifest artifact entry is invalid".to_owned());
        }
        validate_sha256(&artifact.sha256)?;
    }
    for (name, expected_records) in expected_artifacts {
        if manifest
            .artifacts
            .get(name)
            .is_none_or(|artifact| artifact.records != expected_records)
        {
            return Err("Phase 3 dataset manifest artifact record count is invalid".to_owned());
        }
    }
    if starts.source_positions.path.parent() != starts.dataset_manifest.path.parent() {
        return Err("Phase 3 positions must be a sibling of their dataset manifest".to_owned());
    }
    let positions_name = starts
        .source_positions
        .path
        .file_name()
        .and_then(std::ffi::OsStr::to_str)
        .ok_or_else(|| "Phase 3 position artifact name is invalid".to_owned())?;
    if positions_name != "positions-00000.jsonl.gz" {
        return Err("Phase 3 position artifact has a non-canonical name".to_owned());
    }
    let recorded = manifest
        .artifacts
        .get(positions_name)
        .ok_or_else(|| "Phase 3 positions are absent from their dataset manifest".to_owned())?;
    let expected_records = u64::try_from(MAX_PHASE3_POSITIONS).expect("bound fits u64");
    if recorded.sha256 != starts.source_positions.sha256
        || recorded.size != starts.source_positions.size
        || recorded.records != counts.positions
        || !(1..=expected_records).contains(&recorded.records)
    {
        return Err("Phase 3 position artifact disagrees with its dataset manifest".to_owned());
    }
    Ok(())
}

fn validate_phase3_text(
    value: &str,
    maximum_characters: usize,
    field: &str,
    reject_nul: bool,
) -> Result<(), String> {
    if value.is_empty()
        || value.chars().count() > maximum_characters
        || (reject_nul && value.contains('\0'))
    {
        return Err(format!("Phase 3 {field} is not a bounded string"));
    }
    Ok(())
}

fn validate_phase3_portable_relative_path(path: &Path, maximum_characters: usize) -> Result<(), String> {
    let value = path
        .to_str()
        .ok_or_else(|| "Phase 3 portable path is not UTF-8".to_owned())?;
    if value.is_empty()
        || value.chars().count() > maximum_characters
        || value.contains(['\\', '\0'])
    {
        return Err("Phase 3 portable path contains a forbidden character".to_owned());
    }
    let parts = path
        .components()
        .map(|component| match component {
            Component::Normal(part) => part
                .to_str()
                .map(str::to_owned)
                .ok_or_else(|| "Phase 3 portable path is not UTF-8".to_owned()),
            _ => Err("Phase 3 portable path is not normalized and relative".to_owned()),
        })
        .collect::<Result<Vec<_>, _>>()?;
    if parts.is_empty() || parts.join("/") != value {
        return Err("Phase 3 portable path is not normalized and relative".to_owned());
    }
    Ok(())
}

type Phase3EvidenceIdentity = (String, String, String, String, u64, String, String);

fn phase3_evidence_identity(
    snapshot: &Phase3EvidenceSnapshot,
) -> Result<Phase3EvidenceIdentity, String> {
    Ok((
        snapshot.evidence_id.clone(),
        snapshot.url.clone(),
        snapshot.retrieved_at.clone(),
        snapshot.sha256.clone(),
        snapshot.size,
        snapshot.content_type.clone(),
        snapshot
            .object_path
            .to_str()
            .ok_or_else(|| "Phase 3 evidence object path is not UTF-8".to_owned())?
            .to_owned(),
    ))
}

fn phase3_evidence_object_path(sha256: &str) -> Result<PathBuf, String> {
    validate_sha256(sha256)?;
    Ok(PathBuf::from(format!(
        "evidence/sha256/{}/{sha256}",
        &sha256[..2]
    )))
}

fn aobazero_line_error(line_number: usize, message: &str) -> String {
    format!("AobaZero CSA line {line_number}: {message}")
}

fn strip_aobazero_line_ending(line: &str, line_number: usize) -> Result<&str, String> {
    let stripped = if let Some(value) = line.strip_suffix("\r\n") {
        value
    } else if let Some(value) = line.strip_suffix('\n') {
        value
    } else if line.ends_with('\r') {
        return Err(aobazero_line_error(
            line_number,
            "bare carriage return is not accepted",
        ));
    } else {
        line
    };
    if stripped.contains(['\r', '\n']) {
        return Err(aobazero_line_error(
            line_number,
            "embedded line ending is not accepted",
        ));
    }
    Ok(stripped)
}

fn is_aobazero_record_control(character: char) -> bool {
    matches!(character, '\u{2028}' | '\u{2029}')
        || (character.is_control() && !matches!(character, '\r' | '\n' | '\t'))
}

fn is_aobazero_unsafe_text_character(character: char) -> bool {
    let scalar = u32::from(character);
    character.is_control()
        || matches!(character, '\u{2028}' | '\u{2029}')
        // Unicode format controls must not become invisible CSA metadata. Rust strings cannot
        // contain surrogate code points; the remaining stable format-control ranges are closed
        // here without adding a second Unicode implementation to the production boundary.
        || matches!(
            scalar,
            0x00ad
                | 0x061c
                | 0x06dd
                | 0x070f
                | 0x180e
                | 0x200b..=0x200f
                | 0x202a..=0x202e
                | 0x2060..=0x2064
                | 0x2066..=0x206f
                | 0xfeff
                | 0xfff9..=0xfffb
                | 0x110bd
                | 0x110cd
                | 0x13430..=0x1343f
                | 0x1bca0..=0x1bca3
                | 0x1d173..=0x1d17a
                | 0xe0001
                | 0xe0020..=0xe007f
        )
        || matches!(scalar, 0x0600..=0x0605 | 0x0890..=0x0891 | 0x08e2)
        || matches!(
            scalar,
            0xe000..=0xf8ff | 0xf0000..=0xffffd | 0x0010_0000..=0x0010_fffd
        )
        || matches!(scalar, 0xfdd0..=0xfdef)
        || scalar & 0xffff >= 0xfffe
}

fn validate_aobazero_safe_text(
    value: &str,
    maximum_bytes: usize,
    line_number: usize,
    description: &str,
) -> Result<(), String> {
    if value.is_empty()
        || value.len() > maximum_bytes
        || value.contains(',')
        || value.chars().any(is_aobazero_unsafe_text_character)
    {
        return Err(aobazero_line_error(
            line_number,
            &format!("{description} is empty, oversized, or contains unsafe text"),
        ));
    }
    Ok(())
}

fn is_aobazero_pi_line(line: &str) -> bool {
    let Some(removals) = line.strip_prefix("PI") else {
        return false;
    };
    removals.as_bytes().chunks_exact(4).remainder().is_empty()
        && removals.as_bytes().chunks_exact(4).all(|entry| {
            matches!(entry[0], b'1'..=b'9')
                && matches!(entry[1], b'1'..=b'9')
                && entry[2].is_ascii_uppercase()
                && entry[3].is_ascii_uppercase()
        })
}

fn is_aobazero_time_line(line: &str) -> bool {
    let Some(value) = line.strip_prefix('T') else {
        return false;
    };
    let (seconds, fraction) = value
        .split_once('.')
        .map_or((value, None), |(seconds, fraction)| {
            (seconds, Some(fraction))
        });
    let seconds_valid = seconds == "0"
        || (seconds
            .as_bytes()
            .first()
            .is_some_and(|first| matches!(first, b'1'..=b'9'))
            && seconds.as_bytes().iter().all(u8::is_ascii_digit));
    seconds_valid
        && fraction.is_none_or(|fraction| {
            (1..=3).contains(&fraction.len())
                && fraction.as_bytes().iter().all(u8::is_ascii_digit)
        })
}

fn is_aobazero_terminal(line: &str) -> bool {
    matches!(
        line,
        "%TORYO"
            | "%CHUDAN"
            | "%SENNICHITE"
            | "%OUTE_SENNICHITE"
            | "%ILLEGAL_MOVE"
            | "%+ILLEGAL_ACTION"
            | "%-ILLEGAL_ACTION"
            | "%TIME_UP"
            | "%JISHOGI"
            | "%KACHI"
            | "%HIKIWAKE"
            | "%MAX_MOVES"
            | "%MATTA"
            | "%TSUMI"
            | "%FUZUMI"
            | "%ERROR"
    )
}

fn parse_aobazero_move(line: &str, line_number: usize) -> Result<(&str, usize), String> {
    let bytes = line.as_bytes();
    if bytes.len() < 7
        || !matches!(bytes[0], b'+' | b'-')
        || !bytes[1..5].iter().all(u8::is_ascii_digit)
        || !bytes[5..7].iter().all(u8::is_ascii_uppercase)
    {
        return Err(aobazero_line_error(
            line_number,
            "move must begin with one exact seven-byte CSA token",
        ));
    }
    let movement = &line[..7];
    let suffix = &bytes[7..];
    if suffix.is_empty() {
        return Ok((movement, 0));
    }
    if !(suffix.starts_with(b",'") || suffix.starts_with(b",v=")) {
        return Err(aobazero_line_error(
            line_number,
            "move annotation is outside the audited comma-attached forms",
        ));
    }
    if !suffix.iter().all(|byte| matches!(byte, 0x20..=0x7e)) {
        return Err(aobazero_line_error(
            line_number,
            "move annotation must contain printable ASCII",
        ));
    }
    Ok((movement, suffix.len()))
}

fn aobazero_comment_timestamp(comment: &str) -> Result<Option<&str>, String> {
    let Some(body) = comment.strip_prefix('\'') else {
        return Ok(None);
    };
    let Some(stem) = body.strip_suffix(".txt") else {
        return Ok(None);
    };
    let Some(timestamp) = stem.get(..15) else {
        return Ok(None);
    };
    let timestamp_bytes = timestamp.as_bytes();
    let suffix = &stem[15..];
    if timestamp_bytes.len() != 15
        || timestamp_bytes[8] != b'_'
        || !timestamp_bytes[..8].iter().all(u8::is_ascii_digit)
        || !timestamp_bytes[9..].iter().all(u8::is_ascii_digit)
        || (!suffix.is_empty()
            && (!suffix.starts_with('_')
                || suffix.len() == 1
                || !suffix.as_bytes()[1..]
                    .iter()
                    .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-'))))
    {
        return Ok(None);
    }

    let parse = |range: std::ops::Range<usize>| -> u32 {
        timestamp[range]
            .parse()
            .expect("timestamp fields contain only bounded ASCII digits")
    };
    let year = parse(0..4);
    let month = parse(4..6);
    let day = parse(6..8);
    let hour = parse(9..11);
    let minute = parse(11..13);
    let second = parse(13..15);
    let leap_year = year.is_multiple_of(4) && (!year.is_multiple_of(100) || year.is_multiple_of(400));
    let days = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if leap_year => 29,
        2 => 28,
        _ => 0,
    };
    if year == 0 || day == 0 || day > days || hour > 23 || minute > 59 || second > 59 {
        return Err("AobaZero CSA source filename comment has an invalid timestamp".to_owned());
    }
    Ok(Some(timestamp))
}

#[expect(
    clippy::too_many_lines,
    reason = "the adapter deliberately mirrors the bounded Python AobaZero envelope state machine"
)]
fn adapt_phase3_aobazero_csa(raw: &str) -> Result<String, String> {
    if raw.is_empty() {
        return Err("AobaZero CSA record is empty".to_owned());
    }
    if raw.len() > MAX_PHASE3_CSA_BYTES {
        return Err("AobaZero CSA record exceeds its byte limit".to_owned());
    }
    if raw.starts_with('\u{feff}') {
        return Err("AobaZero CSA UTF-8 BOM is not accepted".to_owned());
    }
    if raw.chars().any(is_aobazero_record_control) {
        return Err("AobaZero CSA contains an unsupported record control".to_owned());
    }

    let lines = raw.split_inclusive('\n').collect::<Vec<_>>();
    if lines.len() > MAX_AOBAZERO_CSA_LINES {
        return Err("AobaZero CSA exceeds its line limit".to_owned());
    }

    let mut normalized = vec!["V3.0".to_owned()];
    let mut metadata_keys = std::collections::BTreeSet::new();
    let mut black_name = None;
    let mut white_name = None;
    let mut source_timestamp = None;
    let mut saw_version = false;
    let mut saw_non_version_statement = false;
    let mut saw_position = false;
    let mut saw_side = false;
    let mut terminal = false;
    let mut last_was_move = false;
    let mut move_count = 0_usize;
    let mut comment_count = 0_usize;
    let mut comment_bytes = 0_usize;
    let mut annotation_bytes = 0_usize;

    for (index, raw_line) in lines.iter().enumerate() {
        let line_number = index + 1;
        let line = strip_aobazero_line_ending(raw_line, line_number)?;
        if line.len() > MAX_AOBAZERO_CSA_LINE_BYTES {
            return Err(aobazero_line_error(line_number, "line exceeds its byte limit"));
        }
        if line.is_empty() {
            if index + 1 == lines.len() {
                continue;
            }
            return Err(aobazero_line_error(
                line_number,
                "blank lines are not allowed inside a record",
            ));
        }
        if line == "/" {
            return Err(aobazero_line_error(
                line_number,
                "multiple-game separators are not accepted",
            ));
        }
        if terminal {
            if !line.starts_with('\'') {
                return Err(aobazero_line_error(
                    line_number,
                    "only comments may follow the terminal line",
                ));
            }
            comment_count = comment_count.saturating_add(1);
            comment_bytes = comment_bytes.saturating_add(line.len());
            if comment_count > MAX_AOBAZERO_CSA_COMMENTS
                || comment_bytes > MAX_AOBAZERO_CSA_COMMENT_BYTES
            {
                return Err(aobazero_line_error(
                    line_number,
                    "comment budget is exceeded",
                ));
            }
            if let Some(candidate) = aobazero_comment_timestamp(line)? {
                if source_timestamp.is_some_and(|current| current != candidate) {
                    return Err(aobazero_line_error(
                        line_number,
                        "conflicting source filename timestamps are present",
                    ));
                }
                source_timestamp = Some(candidate);
            }
            continue;
        }
        if line.starts_with('\'') {
            comment_count = comment_count.saturating_add(1);
            comment_bytes = comment_bytes.saturating_add(line.len());
            if comment_count > MAX_AOBAZERO_CSA_COMMENTS
                || comment_bytes > MAX_AOBAZERO_CSA_COMMENT_BYTES
            {
                return Err(aobazero_line_error(
                    line_number,
                    "comment budget is exceeded",
                ));
            }
            if let Some(candidate) = aobazero_comment_timestamp(line)? {
                if source_timestamp.is_some_and(|current| current != candidate) {
                    return Err(aobazero_line_error(
                        line_number,
                        "conflicting source filename timestamps are present",
                    ));
                }
                source_timestamp = Some(candidate);
            }
            last_was_move = false;
            continue;
        }
        if line.starts_with('V') {
            if saw_version || saw_non_version_statement {
                return Err(aobazero_line_error(
                    line_number,
                    "version is duplicated or not the first non-comment statement",
                ));
            }
            if line != "V3.0" {
                return Err(aobazero_line_error(
                    line_number,
                    "only CSA V3.0 is accepted",
                ));
            }
            saw_version = true;
            last_was_move = false;
            continue;
        }

        saw_non_version_statement = true;
        if line.starts_with("N+") || line.starts_with("N-") {
            if saw_position || saw_side {
                return Err(aobazero_line_error(
                    line_number,
                    "player name appears after the position",
                ));
            }
            let (side, name) = line.split_at(2);
            validate_aobazero_safe_text(
                name,
                MAX_AOBAZERO_CSA_NAME_BYTES,
                line_number,
                "player name",
            )?;
            let destination = if side == "N+" {
                &mut black_name
            } else {
                &mut white_name
            };
            if destination.replace(name).is_some() {
                return Err(aobazero_line_error(line_number, "player name is duplicated"));
            }
            normalized.push(line.to_owned());
            last_was_move = false;
            continue;
        }
        if let Some(body) = line.strip_prefix('$') {
            if saw_position || saw_side {
                return Err(aobazero_line_error(
                    line_number,
                    "metadata appears after the position",
                ));
            }
            let Some((key, value)) = body.split_once(':') else {
                return Err(aobazero_line_error(
                    line_number,
                    "metadata must contain ':'",
                ));
            };
            if key.is_empty()
                || !key
                    .as_bytes()
                    .iter()
                    .all(|byte| byte.is_ascii_uppercase() || byte.is_ascii_digit() || *byte == b'_')
            {
                return Err(aobazero_line_error(
                    line_number,
                    "metadata key is outside A-Z, digits, and underscore",
                ));
            }
            validate_aobazero_safe_text(
                value,
                MAX_AOBAZERO_CSA_METADATA_VALUE_BYTES,
                line_number,
                "metadata value",
            )?;
            if metadata_keys.len() >= MAX_AOBAZERO_CSA_METADATA_ENTRIES {
                return Err(aobazero_line_error(
                    line_number,
                    "metadata entry limit is exceeded",
                ));
            }
            if !metadata_keys.insert(key) {
                return Err(aobazero_line_error(line_number, "metadata key is duplicated"));
            }
            normalized.push(line.to_owned());
            last_was_move = false;
            continue;
        }
        if line.starts_with("PI") {
            if saw_position || saw_side || !is_aobazero_pi_line(line) {
                return Err(aobazero_line_error(
                    line_number,
                    "PI position is invalid, duplicated, or out of order",
                ));
            }
            saw_position = true;
            normalized.push(line.to_owned());
            last_was_move = false;
            continue;
        }
        if line.starts_with('P') {
            return Err(aobazero_line_error(
                line_number,
                "explicit board rows are not accepted",
            ));
        }
        if matches!(line, "+" | "-") {
            if !saw_position || saw_side {
                return Err(aobazero_line_error(
                    line_number,
                    "side-to-move is missing its PI line or is duplicated",
                ));
            }
            saw_side = true;
            normalized.push(line.to_owned());
            last_was_move = false;
            continue;
        }
        if line.starts_with('%') {
            if !saw_side || !is_aobazero_terminal(line) {
                return Err(aobazero_line_error(
                    line_number,
                    "terminal is before the side line or outside the audited code set",
                ));
            }
            terminal = true;
            normalized.push(line.to_owned());
            last_was_move = true;
            continue;
        }
        if line.starts_with('T') {
            if !last_was_move || !is_aobazero_time_line(line) {
                return Err(aobazero_line_error(
                    line_number,
                    "time must immediately follow one move and use the official numeric form",
                ));
            }
            last_was_move = false;
            continue;
        }
        if line.starts_with('+') || line.starts_with('-') {
            if !saw_side {
                return Err(aobazero_line_error(
                    line_number,
                    "move appears before the side-to-move line",
                ));
            }
            let (movement, annotation_size) = parse_aobazero_move(line, line_number)?;
            move_count = move_count.saturating_add(1);
            annotation_bytes = annotation_bytes.saturating_add(annotation_size);
            if move_count > MAX_AOBAZERO_CSA_MOVES
                || annotation_bytes > MAX_AOBAZERO_CSA_ANNOTATION_BYTES
            {
                return Err(aobazero_line_error(
                    line_number,
                    "move or annotation budget is exceeded",
                ));
            }
            normalized.push(movement.to_owned());
            last_was_move = true;
            continue;
        }
        return Err(aobazero_line_error(
            line_number,
            "statement is outside the selected AobaZero CSA dialect",
        ));
    }

    if !saw_position || !saw_side || !terminal {
        return Err("AobaZero CSA record is incomplete".to_owned());
    }
    Ok(format!("{}\n", normalized.join("\n")))
}

#[expect(
    clippy::too_many_lines,
    reason = "the closed Phase 3 game record binds provenance, canonical CSA, and replay evidence"
)]
fn validate_phase3_game_row(
    row: &Phase3GameRow,
    manifest: &Phase3DatasetManifest,
    context: &str,
) -> Result<Vec<ExpectedPhase3Position>, String> {
    if row.schema != "phase3_game/v1"
        || row.game_id != row.canonical_sha256
        || row.source_id != manifest.source.source_id
        || row.flags.short != (row.ply_count < 20)
        || row.flags.long != (row.ply_count > 512)
        || row.position_count != row.ply_count.saturating_add(1)
        || row.ply_count > 10_000
        || row.usi_moves.len() != usize::try_from(row.ply_count).unwrap_or(usize::MAX)
    {
        return Err(format!("{context} has invalid identity or bounded counts"));
    }
    validate_sha256(&row.game_id)?;
    if sha256_bytes(row.normalized_csa.as_bytes()) != row.canonical_sha256
        || row.normalized_csa.is_empty()
        || row.normalized_csa.len() > MAX_PHASE3_CSA_BYTES
        || row.raw_csa.is_empty()
        || row.raw_csa.len()
            > usize::try_from(manifest.config.max_raw_bytes).unwrap_or(usize::MAX)
        || row.raw_object.size
            != u64::try_from(row.raw_csa.len()).expect("bounded raw CSA length fits u64")
        || sha256_bytes(row.raw_csa.as_bytes()) != row.raw_object.sha256
    {
        return Err(format!("{context} CSA bytes disagree with their identities or bounds"));
    }
    let adapted_raw;
    let raw_csa = if manifest.source.adapter == "aobazero_csa" {
        adapted_raw = adapt_phase3_aobazero_csa(&row.raw_csa)
            .map_err(|error| format!("{context} raw CSA is invalid: {error}"))?;
        adapted_raw.as_str()
    } else {
        row.raw_csa.as_str()
    };
    let raw_parsed = parse_csa_game(raw_csa)
        .map_err(|error| format!("{context} raw CSA is invalid: {error}"))?;
    let normalized_from_raw = to_csa_game(&raw_parsed)
        .map_err(|error| format!("{context} raw CSA cannot be canonicalized: {error}"))?;
    if normalized_from_raw != row.normalized_csa {
        return Err(format!(
            "{context} normalized CSA is not the canonical form of its raw CSA"
        ));
    }
    let parsed = parse_csa_game(&row.normalized_csa)
        .map_err(|error| format!("{context} normalized CSA is invalid: {error}"))?;
    let canonical = to_csa_game(&parsed)
        .map_err(|error| format!("{context} normalized CSA cannot be canonicalized: {error}"))?;
    if canonical != row.normalized_csa || parsed.version != "V3.0" {
        return Err(format!("{context} normalized CSA is not canonical V3.0"));
    }
    let expected_initial_sfen = to_sfen(&raw_parsed.initial_position);
    let expected_moves = raw_parsed
        .moves
        .iter()
        .copied()
        .map(to_usi_move)
        .collect::<Vec<_>>();
    if row.initial_sfen != expected_initial_sfen || row.usi_moves != expected_moves {
        return Err(format!("{context} SFEN or USI sequence differs from canonical CSA"));
    }

    // Phase 3 exports a complete legal history. An ordinary repetition earlier in that history
    // is not a truncation boundary, so replay every move directly on `Position` exactly as the
    // exporter does instead of using `Game`, whose live-play terminal state would stop early.
    let mut replay = raw_parsed.initial_position.clone();
    let mut sfens = vec![to_sfen(&replay)];
    for &movement in &raw_parsed.moves {
        replay
            .make_move(movement)
            .map_err(|error| format!("{context} CSA contains an illegal move: {error}"))?;
        sfens.push(to_sfen(&replay));
    }
    let (expected_outcome, expected_terminal, expected_validation) =
        phase3_terminal_fields(&raw_parsed, &replay, context)?;
    if !phase3_outcome_matches(
        &row.outcome,
        &expected_outcome,
        expected_terminal.as_deref(),
        &expected_validation,
    )
        || row.terminal_reason != expected_terminal
        || row.result_validation != expected_validation
    {
        return Err(format!("{context} terminal fields differ from canonical CSA replay"));
    }
    let expected_black = raw_parsed
        .black_name
        .as_deref()
        .filter(|name| !name.is_empty());
    let expected_white = raw_parsed
        .white_name
        .as_deref()
        .filter(|name| !name.is_empty());
    if row.players.black.name.as_deref() != expected_black
        || row.players.white.name.as_deref() != expected_white
        || row
            .players
            .black
            .rating
            .is_some_and(|rating| !(0..=1_000_000).contains(&rating))
        || row
            .players
            .white
            .rating
            .is_some_and(|rating| !(0..=1_000_000).contains(&rating))
    {
        return Err(format!("{context} player fields differ from canonical CSA"));
    }

    validate_identifier(&row.raw_object.object_id, "Phase 3 raw objectId")?;
    validate_phase3_portable_relative_path(&row.raw_object.object_path, 4_096)?;
    validate_sha256(&row.raw_object.sha256)?;
    let expected_raw_path = PathBuf::from(format!(
        "objects/sha256/{}/{}",
        &row.raw_object.sha256[..2], row.raw_object.sha256
    ));
    if !(1..=manifest.config.max_raw_bytes).contains(&row.raw_object.size)
        || row.raw_object.object_path != expected_raw_path
    {
        return Err(format!("{context} raw object size is outside the configured bound"));
    }
    validate_phase3_text(
        &row.raw_object.original_filename,
        4_096,
        "raw originalFilename",
        true,
    )?;
    for (value, field) in [
        (
            row.raw_object.response.content_type.as_deref(),
            "raw response contentType",
        ),
        (row.raw_object.response.etag.as_deref(), "raw response etag"),
        (
            row.raw_object.response.last_modified.as_deref(),
            "raw response lastModified",
        ),
        (row.date.as_deref(), "game date"),
        (row.source_date_time.as_deref(), "game sourceDateTime"),
        (row.source_time_zone.as_deref(), "game sourceTimeZone"),
    ] {
        if let Some(value) = value {
            validate_phase3_text(value, 4_096, field, true)?;
        }
    }
    validate_phase3_text(&row.url, 4_096, "game URL", true)?;
    if (!row.url.starts_with("https://") && !row.url.starts_with("http://"))
        || !phase3_url_is_under_base(&row.url, &manifest.source.official_base)
    {
        return Err(format!("{context} URL is invalid"));
    }
    validate_utc_timestamp(&row.retrieved_at, "Phase 3 game retrievedAt")?;

    if row.license_decision.license != manifest.source.license
        || row.license_decision.evidence != manifest.source.license_evidence
        || row.license_decision.redistributable != manifest.source.redistributable
        || row.license_decision.machine_learning_allowed
            != manifest.source.machine_learning_allowed
    {
        return Err(format!("{context} license/provenance decision differs from its manifest"));
    }
    let required_snapshot_ids = manifest
        .evidence_snapshots
        .iter()
        .map(|snapshot| snapshot.evidence_id.as_str())
        .collect::<std::collections::BTreeSet<_>>();
    let mut row_snapshot_identities = std::collections::BTreeSet::new();
    let mut row_snapshot_ids = std::collections::BTreeSet::new();
    let mut row_snapshot_urls = std::collections::BTreeSet::new();
    for snapshot in &row.license_decision.evidence_snapshots {
        let identity = phase3_evidence_identity(snapshot)?;
        if snapshot.object_path != phase3_evidence_object_path(&snapshot.sha256)?
            || !manifest.evidence_snapshots.contains(snapshot)
            || !row_snapshot_identities.insert(identity.clone())
            || !row_snapshot_ids.insert(snapshot.evidence_id.as_str())
        {
            return Err(format!("{context} repeats or invents an evidence snapshot"));
        }
        row_snapshot_urls.insert(snapshot.url.as_str());
    }
    if row_snapshot_ids != required_snapshot_ids
        || manifest
        .source
        .license_evidence
        .iter()
        .any(|evidence| !row_snapshot_urls.contains(evidence.url.as_str()))
    {
        return Err(format!("{context} lacks a snapshot for approved license evidence"));
    }
    let expected_split = phase3_game_split(&row.canonical_sha256, &manifest.config.split)?;
    if row.split != expected_split {
        return Err(format!("{context} split is not the salted canonical-hash assignment"));
    }

    let tail_start = row
        .ply_count
        .saturating_sub(manifest.config.terminal_tail_positions);
    let mut expected = Vec::with_capacity(sfens.len());
    for (position_index, sfen) in sfens.iter().enumerate() {
        let index = u64::try_from(position_index).expect("bounded position index fits u64");
        let movement = row.usi_moves.get(position_index).cloned();
        let next_sfen = sfens.get(position_index + 1).cloned();
        let terminal_tail = index >= tail_start;
        let side_to_move = match parse_sfen(sfen)
            .map_err(|error| format!("{context} replay produced invalid SFEN: {error}"))?
            .side_to_move()
        {
            Side::Black => "black",
            Side::White => "white",
        };
        expected.push(ExpectedPhase3Position {
            canonical_sha256: row.canonical_sha256.clone(),
            raw_sha256: row.raw_object.sha256.clone(),
            source_id: row.source_id.clone(),
            split: row.split.clone(),
            position_index: index,
            sfen: sfen.clone(),
            move_usi: movement.clone(),
            next_sfen,
            outcome: row.outcome.clone(),
            terminal_reason: row.terminal_reason.clone(),
            side_to_move: side_to_move.to_owned(),
            full_plies: row.ply_count,
            remaining_plies: row.ply_count - index,
            eligible: movement.is_some() && !terminal_tail,
            terminal_tail,
        });
    }
    Ok(expected)
}

fn phase3_terminal_fields(
    parsed: &CsaGame,
    final_position: &Position,
    context: &str,
) -> Result<(String, Option<String>, String), String> {
    let validation = match parsed.result_validation {
        CsaResultValidation::Verified => "verified",
        CsaResultValidation::ExternalCondition => "external_condition",
        CsaResultValidation::Missing => "missing",
    };
    let terminal = parsed.special_move.as_ref().map(phase3_csa_special_code);
    if terminal.is_none() != (parsed.result_validation == CsaResultValidation::Missing) {
        return Err(format!("{context} terminal and result validation disagree"));
    }
    let Some(special) = parsed.special_move.as_ref() else {
        return Ok(("unknown".to_owned(), None, validation.to_owned()));
    };
    let final_side = final_position.side_to_move();
    let outcome = match special {
        CsaSpecialMove::Resign
        | CsaSpecialMove::IllegalMove
        | CsaSpecialMove::TimeUp
        | CsaSpecialMove::Checkmate => phase3_winner_outcome(final_side.opposite()),
        CsaSpecialMove::BlackIllegalAction => "white_win",
        CsaSpecialMove::WhiteIllegalAction => "black_win",
        CsaSpecialMove::Win => phase3_winner_outcome(final_side),
        CsaSpecialMove::Draw => "draw",
        CsaSpecialMove::Repetition
            if parsed.result_validation == CsaResultValidation::Verified => "draw",
        CsaSpecialMove::PerpetualCheck => {
            match repetition_outcome_from_moves(&parsed.initial_position, &parsed.moves) {
            Ok(Some(RepetitionOutcome::PerpetualCheckLoss(loser))) => {
                phase3_winner_outcome(loser.opposite())
            }
            _ => "unknown",
            }
        }
        _ => "unknown",
    };
    Ok((outcome.to_owned(), terminal, validation.to_owned()))
}

fn phase3_outcome_matches(
    durable: &str,
    replayed: &str,
    terminal: Option<&str>,
    validation: &str,
) -> bool {
    // The approved Phase 3 exporter conservatively recorded two verified ordinary
    // repetitions as unknown. A later rules-aware exporter can prove those draws; no other
    // historical outcome drift is accepted.
    durable == replayed
        || (durable == "unknown"
            && replayed == "draw"
            && terminal == Some("SENNICHITE")
            && validation == "verified")
}

fn phase3_url_is_under_base(url: &str, official_base: &str) -> bool {
    let base = official_base.trim_end_matches('/');
    url == base
        || url
            .strip_prefix(base)
            .is_some_and(|suffix| suffix.starts_with('/'))
}

const fn phase3_winner_outcome(side: Side) -> &'static str {
    match side {
        Side::Black => "black_win",
        Side::White => "white_win",
    }
}

fn phase3_csa_special_code(special: &CsaSpecialMove) -> String {
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
    .to_owned()
}

fn phase3_game_split(hash: &str, policy: &Phase3SplitPolicy) -> Result<String, String> {
    let decoded = decode_sha256_bytes(hash)?;
    let mut hasher = Sha256::new();
    hasher.update(policy.salt.as_bytes());
    hasher.update([0]);
    hasher.update(decoded);
    let digest = hasher.finalize();
    let bucket = u64::from_be_bytes(
        digest[..8]
            .try_into()
            .expect("SHA-256 always has an eight-byte prefix"),
    ) % 10_000;
    if bucket < policy.test_basis_points {
        Ok("test".to_owned())
    } else if bucket < policy.test_basis_points + policy.validation_basis_points {
        Ok("validation".to_owned())
    } else {
        Ok("train".to_owned())
    }
}

fn phase3_game_line_limit(config: &Phase3NormalizationConfig) -> Result<u64, String> {
    // Python's compact JSON writer may expand control characters to six-byte `\\u00xx`
    // escapes. Derive a checked producer-shaped bound from the configured raw payload and the
    // canonical CSA cap, while retaining the approved absolute validation ceiling.
    let raw = config
        .max_raw_bytes
        .checked_mul(6)
        .ok_or_else(|| "Phase 3 game line bound overflow".to_owned())?;
    let canonical = u64::try_from(MAX_PHASE3_CSA_BYTES)
        .expect("CSA bound fits u64")
        .checked_mul(6)
        .ok_or_else(|| "Phase 3 game line bound overflow".to_owned())?;
    raw.checked_add(canonical)
        .and_then(|total| total.checked_add(1024 * 1024))
        .map(|total| total.min(MAX_PHASE3_GAME_LINE_BYTES))
        .ok_or_else(|| "Phase 3 game line bound overflow".to_owned())
}

fn validate_phase3_canonical_json_line(
    line: &[u8],
    value: &serde_json::Value,
    context: &str,
) -> Result<(), String> {
    let payload = line
        .strip_suffix(b"\n")
        .ok_or_else(|| format!("{context} is not LF-terminated"))?;
    let mut canonical = Vec::with_capacity(payload.len());
    write_python_canonical_json(value, &mut canonical)?;
    if payload != canonical {
        return Err(format!(
            "{context} is not Python compact sorted canonical JSON"
        ));
    }
    Ok(())
}

fn decode_sha256_bytes(value: &str) -> Result<[u8; 32], String> {
    validate_sha256(value)?;
    let mut output = [0_u8; 32];
    for (index, pair) in value.as_bytes().chunks_exact(2).enumerate() {
        let digit = |byte: u8| match byte {
            b'0'..=b'9' => Ok(byte - b'0'),
            b'a'..=b'f' => Ok(byte - b'a' + 10),
            _ => Err("invalid SHA-256 hex digit".to_owned()),
        };
        output[index] = digit(pair[0])? << 4 | digit(pair[1])?;
    }
    Ok(output)
}

#[expect(
    clippy::too_many_lines,
    reason = "bounded gzip streaming and full Phase 3 game coverage are validated together"
)]
fn load_validated_phase3_games(
    storage: &RegistryStorage,
    starts: &Phase6StartPositions,
    manifest: &Phase3DatasetManifest,
    observed: &mut BTreeMap<StableFileIdentity, ObservedRegistryArtifact>,
    budget: &mut RegistryVerificationBudget,
) -> Result<ValidatedPhase3Games, String> {
    let game_artifacts = manifest
        .artifacts
        .iter()
        .filter(|(name, _)| name.starts_with("games-") && name.ends_with(".jsonl.gz"))
        .collect::<Vec<_>>();
    let [(name, recorded)] = game_artifacts.as_slice() else {
        return Err("Phase 3 manifest must contain exactly one games JSONL gzip shard".to_owned());
    };
    if recorded.records != manifest.counts.games
        || recorded.size == 0
        || recorded.size > MAX_PHASE3_COMPRESSED_BYTES
    {
        return Err("Phase 3 games artifact disagrees with manifest counts or bounds".to_owned());
    }
    let parent = starts
        .dataset_manifest
        .path
        .parent()
        .ok_or_else(|| "Phase 3 manifest path has no parent".to_owned())?;
    let reference = RegistryArtifact {
        path: parent.join(name),
        sha256: recorded.sha256.clone(),
        size: recorded.size,
    };
    validate_artifact_ref(&reference)?;
    let _ = verify_registry_artifact_cached(storage, &reference, observed, budget, None)?;
    budget.charge_reference()?;

    let display_path = storage.root_path.join(&reference.path);
    let mut file = storage
        .root
        .open_relative_regular(&reference.path)
        .map_err(|error| format!("cannot reopen Phase 3 games {}: {error}", display_path.display()))?;
    let before = file.stable_identity().map_err(|error| {
        format!("cannot bind Phase 3 games identity {}: {error}", display_path.display())
    })?;
    let verified = observed.get(&before).ok_or_else(|| {
        "Phase 3 games changed between artifact verification and parsing".to_owned()
    })?;
    verify_observed_artifact(&reference, &verified.sha256, verified.size)?;

    let canonical_manifest = manifest
        .canonical_game_sha256
        .iter()
        .cloned()
        .collect::<std::collections::BTreeSet<_>>();
    let raw_manifest = manifest
        .raw_object_sha256
        .iter()
        .cloned()
        .collect::<std::collections::BTreeSet<_>>();
    let manifest_evidence = manifest
        .evidence_snapshots
        .iter()
        .map(phase3_evidence_identity)
        .collect::<Result<std::collections::BTreeSet<_>, _>>()?;
    let mut observed_game_evidence = std::collections::BTreeSet::new();
    let mut positions = BTreeMap::new();
    let mut position_order = Vec::new();
    let mut game_splits = BTreeMap::new();
    let mut game_raw = BTreeMap::new();
    let mut raw_to_game = BTreeMap::<String, String>::new();
    let mut previous_game_order = None::<(String, String)>;
    let mut raw_object_ids = std::collections::BTreeSet::new();
    let mut records = 0_usize;
    let mut uncompressed_bytes = 0_u64;
    let game_line_limit = phase3_game_line_limit(&manifest.config)?;

    let compressed_bytes_read = {
        let decoder = flate2::read::MultiGzDecoder::new(file.reader());
        let mut reader = io::BufReader::new(decoder);
        loop {
            budget.ensure_time()?;
            let mut line = Vec::with_capacity(
                usize::try_from(MAX_PHASE3_LINE_BYTES).expect("initial line capacity fits usize"),
            );
            let read = {
                let mut limited = std::io::Read::take(
                    &mut reader,
                    game_line_limit.saturating_add(1),
                );
                limited.read_until(b'\n', &mut line).map_err(|error| {
                    format!("cannot decompress Phase 3 games {}: {error}", display_path.display())
                })?
            };
            if read == 0 {
                break;
            }
            uncompressed_bytes = uncompressed_bytes
                .checked_add(u64::try_from(read).expect("bounded read fits u64"))
                .ok_or_else(|| "Phase 3 game byte count overflow".to_owned())?;
            if uncompressed_bytes > MAX_PHASE3_UNCOMPRESSED_BYTES
                || u64::try_from(line.len()).expect("line length fits u64")
                    > game_line_limit
                || line.last() != Some(&b'\n')
            {
                return Err("Phase 3 games JSONL exceeds its line or byte bound".to_owned());
            }
            records = records
                .checked_add(1)
                .ok_or_else(|| "Phase 3 game count overflow".to_owned())?;
            if records > usize::try_from(manifest.config.max_games).unwrap_or(usize::MAX) {
                return Err("Phase 3 games JSONL exceeds its configured record bound".to_owned());
            }
            let context = format!("Phase 3 game row {records}");
            let value = parse_unique_json(&line, &context)?;
            validate_phase3_canonical_json_line(&line, &value, &context)?;
            let row: Phase3GameRow = deserialize_closed_json(&value, &context)?;
            if !canonical_manifest.contains(&row.canonical_sha256)
                || !raw_manifest.contains(&row.raw_object.sha256)
            {
                return Err("Phase 3 game is absent from its manifest identity lists".to_owned());
            }
            let expected = validate_phase3_game_row(&row, manifest, &context)?;
            for snapshot in &row.license_decision.evidence_snapshots {
                observed_game_evidence.insert(phase3_evidence_identity(snapshot)?);
            }
            let game_order = (
                row.raw_object.object_id.clone(),
                row.raw_object.sha256.clone(),
            );
            if previous_game_order
                .as_ref()
                .is_some_and(|previous| previous >= &game_order)
            {
                return Err(
                    "Phase 3 games are not in canonical raw-object producer order".to_owned(),
                );
            }
            previous_game_order = Some(game_order);
            if !raw_object_ids.insert(row.raw_object.object_id.clone())
                || game_splits
                .insert(row.canonical_sha256.clone(), row.split.clone())
                .is_some()
                || game_raw
                    .insert(row.canonical_sha256.clone(), row.raw_object.sha256.clone())
                    .is_some()
                || raw_to_game
                    .insert(row.raw_object.sha256.clone(), row.canonical_sha256.clone())
                    .is_some()
            {
                return Err("Phase 3 games repeat a canonical or raw object identity".to_owned());
            }
            for expected_position in expected {
                let key = (
                    expected_position.canonical_sha256.clone(),
                    expected_position.position_index,
                );
                position_order.push(key.clone());
                if positions.insert(key, expected_position).is_some() {
                    return Err("Phase 3 games derive a duplicate position identity".to_owned());
                }
            }
        }
        let decoder = reader.into_inner();
        let raw = decoder.into_inner();
        std::io::Seek::stream_position(raw).map_err(|error| {
            format!("cannot inspect compressed Phase 3 games offset {}: {error}", display_path.display())
        })?
    };
    file.verify_stable_read(&before, compressed_bytes_read)
        .map_err(|error| format!("Phase 3 games changed while parsing {}: {error}", display_path.display()))?;
    if u64::try_from(records).expect("bounded game count fits u64") != manifest.counts.games
        || game_splits.keys().cloned().collect::<std::collections::BTreeSet<_>>()
            != canonical_manifest
        || raw_to_game.keys().cloned().collect::<std::collections::BTreeSet<_>>() != raw_manifest
        || observed_game_evidence != manifest_evidence
        || u64::try_from(position_order.len()).expect("bounded position count fits u64")
            != manifest.counts.positions
    {
        return Err("Phase 3 games coverage differs from its dataset manifest".to_owned());
    }
    Ok(ValidatedPhase3Games {
        positions,
        position_order,
        game_splits,
        game_raw,
    })
}

#[expect(
    clippy::too_many_lines,
    reason = "streaming selection keeps Phase 3 coverage and leakage invariants together"
)]
fn validate_phase3_start_position_derivation(
    storage: &RegistryStorage,
    starts: &Phase6StartPositions,
    manifest: &Phase3DatasetManifest,
    observed: &mut BTreeMap<StableFileIdentity, ObservedRegistryArtifact>,
    budget: &mut RegistryVerificationBudget,
) -> Result<(), String> {
    let validated_games =
        load_validated_phase3_games(storage, starts, manifest, observed, budget)?;
    budget.charge_reference()?;
    if starts.source_positions.size > MAX_PHASE3_COMPRESSED_BYTES {
        return Err("Phase 3 positions exceed the compressed-byte limit".to_owned());
    }
    let display_path = storage.root_path.join(&starts.source_positions.path);
    let mut file = storage
        .root
        .open_relative_regular(&starts.source_positions.path)
        .map_err(|error| {
            format!(
                "cannot reopen Phase 3 positions {}: {error}",
                display_path.display()
            )
        })?;
    let before = file.stable_identity().map_err(|error| {
        format!(
            "cannot bind Phase 3 position identity {}: {error}",
            display_path.display()
        )
    })?;
    let verified = observed.get(&before).ok_or_else(|| {
        "Phase 3 positions changed between artifact verification and parsing".to_owned()
    })?;
    verify_observed_artifact(
        &starts.source_positions,
        &verified.sha256,
        verified.size,
    )?;

    let canonical_games = manifest
        .canonical_game_sha256
        .iter()
        .cloned()
        .collect::<std::collections::BTreeSet<_>>();
    let raw_objects = manifest
        .raw_object_sha256
        .iter()
        .cloned()
        .collect::<std::collections::BTreeSet<_>>();
    let mut observed_game_splits = BTreeMap::<String, String>::new();
    let mut observed_game_raw = BTreeMap::<String, String>::new();
    let mut observed_positions = std::collections::BTreeSet::<(String, u64)>::new();
    let mut candidates =
        BTreeMap::<String, BTreeMap<String, (String, u64)>>::new();
    let mut observed_rows = 0_usize;
    let mut uncompressed_bytes = 0_u64;

    let compressed_bytes_read = {
        let decoder = flate2::read::MultiGzDecoder::new(file.reader());
        let mut reader = io::BufReader::new(decoder);
        loop {
            budget.ensure_time()?;
            let mut line = Vec::with_capacity(
                usize::try_from(MAX_PHASE3_LINE_BYTES).expect("line bound fits usize"),
            );
            let read = {
                let mut limited = std::io::Read::take(
                    &mut reader,
                    MAX_PHASE3_LINE_BYTES.saturating_add(1),
                );
                limited.read_until(b'\n', &mut line).map_err(|error| {
                    format!(
                        "cannot decompress Phase 3 positions {}: {error}",
                        display_path.display()
                    )
                })?
            };
            if read == 0 {
                break;
            }
            uncompressed_bytes = uncompressed_bytes
                .checked_add(u64::try_from(read).expect("bounded read length fits u64"))
                .ok_or_else(|| "Phase 3 uncompressed byte count overflow".to_owned())?;
            if uncompressed_bytes > MAX_PHASE3_UNCOMPRESSED_BYTES
                || u64::try_from(line.len()).expect("line length fits u64")
                    > MAX_PHASE3_LINE_BYTES
                || line.last() != Some(&b'\n')
            {
                return Err("Phase 3 JSONL exceeds its line or byte bound".to_owned());
            }
            observed_rows = observed_rows
                .checked_add(1)
                .ok_or_else(|| "Phase 3 row count overflow".to_owned())?;
            if observed_rows > MAX_PHASE3_POSITIONS {
                return Err("Phase 3 JSONL exceeds its record bound".to_owned());
            }
            let context = format!("Phase 3 position row {observed_rows}");
            let value = parse_unique_json(&line, &context)?;
            validate_phase3_canonical_json_line(&line, &value, &context)?;
            let row: Phase3PositionRow = deserialize_closed_json(&value, &context)?;
            let canonical_sfen = validate_phase3_position_row(&row, &context)?;
            if row.source_id != manifest.source.source_id
                || !canonical_games.contains(&row.canonical_sha256)
                || !raw_objects.contains(&row.raw_sha256)
            {
                return Err("Phase 3 position is absent from its approved manifest".to_owned());
            }
            let position_key = (row.canonical_sha256.clone(), row.position_index);
            let expected_order = validated_games
                .position_order
                .get(observed_rows - 1)
                .ok_or_else(|| "Phase 3 positions contain more rows than games derive".to_owned())?;
            if expected_order != &position_key {
                return Err(
                    "Phase 3 positions are not in contiguous canonical game/ply order".to_owned(),
                );
            }
            let expected = validated_games
                .positions
                .get(&position_key)
                .ok_or_else(|| "Phase 3 position is not derived from its canonical game".to_owned())?;
            if !phase3_position_matches(&row, expected) {
                return Err(
                    "Phase 3 position fields differ from canonical game replay".to_owned(),
                );
            }
            if !observed_positions.insert(position_key) {
                return Err("Phase 3 positions repeat a game/position index".to_owned());
            }
            match observed_game_splits.get(&row.canonical_sha256) {
                Some(split) if split != &row.split => {
                    return Err("one Phase 3 game appears in multiple data splits".to_owned());
                }
                None => {
                    observed_game_splits
                        .insert(row.canonical_sha256.clone(), row.split.clone());
                }
                Some(_) => {}
            }
            match observed_game_raw.get(&row.canonical_sha256) {
                Some(raw) if raw != &row.raw_sha256 => {
                    return Err("one Phase 3 game refers to multiple raw objects".to_owned());
                }
                None => {
                    observed_game_raw
                        .insert(row.canonical_sha256.clone(), row.raw_sha256.clone());
                }
                Some(_) => {}
            }
            if row.eligible {
                let origin = (row.canonical_sha256.clone(), row.position_index);
                let by_split = candidates.entry(canonical_sfen).or_default();
                if by_split
                    .get(&row.split)
                    .is_none_or(|previous| origin < *previous)
                {
                    by_split.insert(row.split.clone(), origin);
                }
            }
        }
        let decoder = reader.into_inner();
        let raw = decoder.into_inner();
        std::io::Seek::stream_position(raw).map_err(|error| {
            format!(
                "cannot inspect compressed Phase 3 position offset {}: {error}",
                display_path.display()
            )
        })?
    };
    file.verify_stable_read(&before, compressed_bytes_read)
        .map_err(|error| {
            format!(
                "Phase 3 positions changed while parsing {}: {error}",
                display_path.display()
            )
        })?;
    budget.ensure_time()?;

    if u64::try_from(observed_rows).expect("bounded row count fits u64")
        != manifest.counts.positions
        || observed_rows != validated_games.position_order.len()
    {
        return Err("Phase 3 position count disagrees with its dataset manifest".to_owned());
    }
    let observed_games = observed_game_splits
        .keys()
        .cloned()
        .collect::<std::collections::BTreeSet<_>>();
    let observed_raw = observed_game_raw
        .values()
        .cloned()
        .collect::<std::collections::BTreeSet<_>>();
    if observed_games != canonical_games
        || observed_raw != raw_objects
        || observed_game_splits != validated_games.game_splits
        || observed_game_raw != validated_games.game_raw
    {
        return Err("Phase 3 position coverage differs from its dataset manifest".to_owned());
    }

    let cross_split = candidates
        .values()
        .filter(|origins| origins.len() > 1)
        .count();
    let mut by_split = BTreeMap::<&str, Vec<(String, String, u64)>>::from([
        ("train", Vec::new()),
        ("validation", Vec::new()),
    ]);
    for (sfen, origins) in candidates {
        if origins.len() != 1 {
            continue;
        }
        let (split, (game_id, position_index)) = origins
            .into_iter()
            .next()
            .expect("one candidate origin was checked");
        if let Some(selected_split) = by_split.get_mut(split.as_str()) {
            selected_split.push((sfen, game_id, position_index));
        }
    }

    let mut rebuilt = Vec::new();
    for (split, count) in [
        ("train", starts.selection.train_count),
        ("validation", starts.selection.validation_count),
    ] {
        let candidates = by_split
            .remove(split)
            .expect("both selected Phase 3 splits were initialized");
        let mut ranked = candidates
            .into_iter()
            .map(|(sfen, game_id, position_index)| {
                (
                sha256_bytes(
                    format!(
                        "phase6-start\0{}\0{split}\0{sfen}\0{game_id}\0{position_index}",
                        starts.selection.seed
                    )
                    .as_bytes(),
                ),
                    sfen,
                    game_id,
                    position_index,
                )
            })
            .collect::<Vec<_>>();
        ranked.sort();
        let count = usize::try_from(count).expect("selected position count fits usize");
        if ranked.len() < count {
            return Err(format!(
                "Phase 3 data has only {} safe {split} start positions; {count} required",
                ranked.len()
            ));
        }
        for (_, sfen, game_id, position_index) in ranked.into_iter().take(count) {
            let mut position = Phase6StartPosition {
                position_id: String::new(),
                sfen,
                source_game_sha256: game_id,
                position_index,
                split: split.to_owned(),
            };
            position.position_id = phase6_start_position_identity(&position)?;
            rebuilt.push(position);
        }
    }
    rebuilt.sort_by(|left, right| left.position_id.cmp(&right.position_id));
    if starts.selection.excluded_cross_split_states
        != u64::try_from(cross_split).expect("bounded cross-split count fits u64")
        || rebuilt != starts.positions
    {
        return Err("start-position artifact is not its deterministic Phase 3 selection".to_owned());
    }
    Ok(())
}

fn validate_phase3_position_row(
    row: &Phase3PositionRow,
    context: &str,
) -> Result<String, String> {
    if row.schema != "phase3_position/v1" || row.game_id != row.canonical_sha256 {
        return Err(format!("{context} has invalid game identity or schema"));
    }
    for hash in [&row.game_id, &row.canonical_sha256, &row.raw_sha256] {
        validate_sha256(hash)?;
    }
    validate_identifier(&row.source_id, "Phase 3 position sourceId")?;
    if !matches!(row.split.as_str(), "train" | "validation" | "test")
        || row.position_index > 10_000
        || !matches!(
            row.outcome.as_str(),
            "black_win" | "white_win" | "draw" | "unknown"
        )
        || row.full_plies > 10_000
        || row.remaining_plies > 10_000
        || row.full_plies.checked_sub(row.position_index) != Some(row.remaining_plies)
        || (row.eligible && (row.terminal_tail || row.move_usi.is_none()))
    {
        return Err(format!("{context} violates its bounded position contract"));
    }
    validate_phase3_text(&row.sfen, 1_024, "position SFEN", true)?;
    for (value, maximum, field) in [
        (row.move_usi.as_deref(), 16, "position moveUsi"),
        (row.next_sfen.as_deref(), 1_024, "position nextSfen"),
        (
            row.terminal_reason.as_deref(),
            256,
            "position terminalReason",
        ),
    ] {
        if let Some(value) = value {
            validate_phase3_text(value, maximum, field, true)?;
        }
    }
    if row.move_usi.is_some() != row.next_sfen.is_some() {
        return Err(format!("{context} moveUsi and nextSfen presence disagree"));
    }
    let position = parse_sfen(&row.sfen)
        .map_err(|error| format!("{context}.sfen is illegal: {error}"))?;
    if to_sfen(&position) != row.sfen {
        return Err(format!("{context}.sfen is not canonical"));
    }
    if let (Some(move_usi), Some(next_sfen)) = (&row.move_usi, &row.next_sfen) {
        let movement = parse_usi_move(move_usi)
            .map_err(|error| format!("{context}.moveUsi is malformed: {error}"))?;
        let mut next = position.clone();
        next.make_move(movement)
            .map_err(|error| format!("{context}.moveUsi is illegal: {error}"))?;
        let parsed_next = parse_sfen(next_sfen)
            .map_err(|error| format!("{context}.nextSfen is illegal: {error}"))?;
        if to_sfen(&parsed_next) != *next_sfen || next != parsed_next {
            return Err(format!("{context}.nextSfen does not match legal move application"));
        }
    } else if row.position_index != row.full_plies
        || row.remaining_plies != 0
        || row.eligible
        || !row.terminal_tail
    {
        return Err(format!("{context} final position fields are inconsistent"));
    }
    let fields = row.sfen.split(' ').collect::<Vec<_>>();
    if fields.len() != 4
        || fields.iter().any(|field| field.is_empty())
        || !matches!(fields[1], "b" | "w")
        || !fields[3].bytes().all(|byte| byte.is_ascii_digit())
        || !fields[3].bytes().any(|byte| byte != b'0')
    {
        return Err(format!("{context}.sfen is invalid"));
    }
    let expected_side = if fields[1] == "b" { "black" } else { "white" };
    if row.side_to_move != expected_side {
        return Err(format!("{context}.sideToMove disagrees with SFEN"));
    }
    Ok(format!("{} {} {} 1", fields[0], fields[1], fields[2]))
}

fn phase3_position_matches(
    row: &Phase3PositionRow,
    expected: &ExpectedPhase3Position,
) -> bool {
    row.canonical_sha256 == expected.canonical_sha256
        && row.game_id == expected.canonical_sha256
        && row.raw_sha256 == expected.raw_sha256
        && row.source_id == expected.source_id
        && row.split == expected.split
        && row.position_index == expected.position_index
        && row.sfen == expected.sfen
        && row.move_usi == expected.move_usi
        && row.next_sfen == expected.next_sfen
        && row.outcome == expected.outcome
        && row.terminal_reason == expected.terminal_reason
        && row.side_to_move == expected.side_to_move
        && row.full_plies == expected.full_plies
        && row.remaining_plies == expected.remaining_plies
        && row.eligible == expected.eligible
        && row.terminal_tail == expected.terminal_tail
}

fn validate_phase6_start_positions(
    plan: &PairedArenaPlan,
    starts: &Phase6StartPositions,
) -> Result<(), String> {
    if starts.schema != "phase6_start_positions/v1"
        || starts.dataset_manifest != plan.dataset_manifest
        || starts.selection.seed != PHASE6_ARENA_SEED
        || !(1..=5_000).contains(&starts.selection.train_count)
        || !(1..=5_000).contains(&starts.selection.validation_count)
        || starts
            .selection
            .train_count
            .checked_add(starts.selection.validation_count)
            .is_none_or(|count| count > 10_000)
        || !starts.selection.require_eligible
        || !starts.selection.reset_move_number
        || starts.selection.excluded_cross_split_states > 250_000
        || starts.positions.is_empty()
        || starts.positions.len() > 10_000
    {
        return Err("paired start-position artifact violates its closed contract".to_owned());
    }
    validate_artifact_ref(&starts.dataset_manifest)?;
    validate_artifact_ref(&starts.source_positions)?;
    let mut identifiers = std::collections::BTreeSet::new();
    let mut sfens = std::collections::BTreeSet::new();
    let mut game_splits = BTreeMap::<&str, &str>::new();
    let mut train_count = 0_u64;
    let mut validation_positions = Vec::new();
    for position in &starts.positions {
        validate_identifier(&position.position_id, "start positionId")?;
        validate_sha256(&position.source_game_sha256)?;
        validate_phase6_plan_sfen(&position.sfen)?;
        if position.position_index > 100_000
            || !matches!(position.split.as_str(), "train" | "validation")
            || !identifiers.insert(position.position_id.as_str())
            || !sfens.insert(position.sfen.as_str())
            || position.position_id != phase6_start_position_identity(position)?
        {
            return Err("paired start-position row has an invalid identity".to_owned());
        }
        if let Some(previous) = game_splits.insert(
            position.source_game_sha256.as_str(),
            position.split.as_str(),
        ) && previous != position.split
        {
            return Err("one source game appears in multiple start-position splits".to_owned());
        }
        if position.split == "train" {
            train_count += 1;
        } else {
            validation_positions.push(position);
        }
    }
    if train_count != starts.selection.train_count
        || u64::try_from(validation_positions.len()).expect("bounded length fits u64")
            != starts.selection.validation_count
        || validation_positions.len()
            < usize::try_from(plan.start_set_pairs).expect("pair count fits usize")
    {
        return Err("paired start-position split counts are inconsistent".to_owned());
    }
    let selection_seed = plan.seed ^ 0x0041_5245_4E41;
    validation_positions.sort_by(|left, right| {
        let left_rank = sha256_bytes(format!("{selection_seed}\0{}", left.position_id).as_bytes());
        let right_rank =
            sha256_bytes(format!("{selection_seed}\0{}", right.position_id).as_bytes());
        left_rank
            .cmp(&right_rank)
            .then_with(|| left.position_id.cmp(&right.position_id))
    });
    let observed_jobs = plan
        .jobs
        .iter()
        .skip(usize::try_from(plan.normal_start_pairs).expect("pair count fits usize"));
    for (job, expected) in observed_jobs.zip(validation_positions) {
        if job.start_position_id != expected.position_id || job.sfen != expected.sfen {
            return Err(
                "paired arena start-set jobs are not the deterministic validation selection"
                    .to_owned(),
            );
        }
    }
    Ok(())
}

fn phase6_start_position_identity(position: &Phase6StartPosition) -> Result<String, String> {
    use sha2::{Digest, Sha256};

    let source = decode_sha256(&position.source_game_sha256)?;
    let mut digest = Sha256::new();
    digest.update(b"phase6_start_position/v1\0");
    digest.update(source);
    digest.update(position.position_index.to_be_bytes());
    digest.update(b"\0");
    digest.update(position.sfen.as_bytes());
    Ok(format!("{:x}", digest.finalize()))
}

fn decode_sha256(value: &str) -> Result<[u8; 32], String> {
    validate_sha256(value)?;
    let mut decoded = [0_u8; 32];
    for (index, output) in decoded.iter_mut().enumerate() {
        *output = u8::from_str_radix(&value[index * 2..index * 2 + 2], 16)
            .map_err(|_| "invalid SHA-256 hexadecimal value".to_owned())?;
    }
    Ok(decoded)
}

fn validate_phase6_start_validation(
    storage: &RegistryStorage,
    plan: &PairedArenaPlan,
    starts: &Phase6StartPositions,
    validation: &Phase6StartValidation,
    observed: &mut BTreeMap<StableFileIdentity, ObservedRegistryArtifact>,
    budget: &mut RegistryVerificationBudget,
) -> Result<(), String> {
    if validation.schema != "phase6_start_position_validation/v1"
        || validation.start_positions != plan.start_positions
        || validation.engine != plan.engine
        || validation.engine_build_receipt != plan.engine_build_receipt
        || validation.git_commit != plan.git_commit
        || validation.method != "open-shogi-cli-perft-depth-0"
        || validation.results.len() != starts.positions.len()
    {
        return Err("paired start-position validation differs from its plan".to_owned());
    }
    for artifact in [
        &validation.start_positions,
        &validation.engine,
        &starts.dataset_manifest,
        &starts.source_positions,
    ]
    .into_iter()
    .chain(validation.engine_build_receipt.iter())
    {
        validate_artifact_ref(artifact)?;
    }
    let expected = starts
        .positions
        .iter()
        .map(|position| {
            (
                position.position_id.as_str(),
                sha256_bytes(position.sfen.as_bytes()),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let mut seen = std::collections::BTreeSet::new();
    for result in &validation.results {
        validate_identifier(&result.position_id, "start validation positionId")?;
        validate_sha256(&result.sfen_sha256)?;
        validate_utc_timestamp(&result.completed_at, "start validation completedAt")?;
        let rss_valid = match result.rss_measurement.as_str() {
            "process_tree_ps_rss_sum" => result
                .peak_rss_bytes
                .is_some_and(|peak| peak <= 1024 * 1024 * 1024),
            "process_tree_ps_short_lived_no_sample" => result.peak_rss_bytes == Some(0),
            _ => false,
        };
        if !result.legal
            || result.return_code != 0
            || result.timed_out
            || result.output_limit_exceeded
            || result.memory_limit_exceeded
            || !rss_valid
            || !seen.insert(result.position_id.as_str())
            || expected.get(result.position_id.as_str()) != Some(&result.sfen_sha256)
        {
            return Err("paired start-position legality evidence is invalid".to_owned());
        }
        for artifact in [&result.stdout, &result.stderr] {
            validate_artifact_ref(artifact)?;
            let _ = verify_registry_artifact_cached(storage, artifact, observed, budget, None)?;
        }
    }
    if seen.len() != expected.len() {
        return Err("paired start-position validation omits positions".to_owned());
    }
    Ok(())
}

#[expect(
    clippy::too_many_lines,
    reason = "the closed execution lifecycle and aggregate invariants are validated together"
)]
fn validate_paired_arena_execution(
    storage: &RegistryStorage,
    execution: &PairedArenaExecution,
    raw: &serde_json::Value,
    plan: &PairedArenaPlan,
    plan_reference: &RegistryArtifact,
    observed: &mut BTreeMap<StableFileIdentity, ObservedRegistryArtifact>,
    budget: &mut RegistryVerificationBudget,
) -> Result<(), String> {
    if execution.schema != "phase6_arena_execution_manifest/v1"
        || execution.generation_id != plan.generation_id
        || execution.plan != *plan_reference
        || execution.plan_sha256 != plan.plan_sha256
        || execution.status != "completed"
    {
        return Err("paired arena execution identity or completion status is invalid".to_owned());
    }
    validate_artifact_ref(&execution.plan)?;
    validate_sha256(&execution.plan_sha256)?;
    validate_sha256(&execution.manifest_sha256)?;
    validate_json_self_hash(raw, "manifestSha256", &execution.manifest_sha256)?;
    if !(plan.jobs.len()..=MAX_PAIRED_EXECUTION_ATTEMPTS).contains(&execution.attempts.len()) {
        return Err("paired arena execution attempt count is outside its bound".to_owned());
    }

    let jobs = plan
        .jobs
        .iter()
        .map(|job| (job.job_id.as_str(), job))
        .collect::<BTreeMap<_, _>>();
    let mut seen = std::collections::BTreeSet::new();
    let mut attempts_by_job = BTreeMap::<&str, Vec<&PairedArenaAttempt>>::new();
    let mut previous_order: Option<(&str, u64)> = None;
    let worker_memory_bytes = plan
        .memory_per_worker_mib
        .checked_mul(1024 * 1024)
        .ok_or_else(|| "paired arena worker memory bound overflow".to_owned())?;
    for attempt in &execution.attempts {
        validate_paired_arena_attempt(
            storage,
            attempt,
            plan,
            &jobs,
            worker_memory_bytes,
            observed,
            budget,
        )?;
        if !seen.insert((attempt.job_id.as_str(), attempt.attempt)) {
            return Err("paired arena execution contains a duplicate job attempt".to_owned());
        }
        let current_order = (attempt.job_id.as_str(), attempt.attempt);
        if previous_order.is_some_and(|previous| previous >= current_order) {
            return Err("paired arena attempts are not in canonical job/attempt order".to_owned());
        }
        previous_order = Some(current_order);
        attempts_by_job
            .entry(attempt.job_id.as_str())
            .or_default()
            .push(attempt);
    }
    for attempts in attempts_by_job.values() {
        if attempts
            .iter()
            .map(|attempt| attempt.attempt)
            .ne(1..=u64::try_from(attempts.len()).expect("attempt count fits u64"))
        {
            return Err("paired arena attempt numbers are not contiguous".to_owned());
        }
        let Some((last, preceding)) = attempts.split_last() else {
            return Err("paired arena execution omits a planned job".to_owned());
        };
        if last.status != "completed"
            || preceding
                .iter()
                .any(|attempt| attempt.status != "quarantined")
        {
            return Err(
                "paired arena attempts must end in one completion after only quarantines"
                    .to_owned(),
            );
        }
        for window in attempts.windows(2) {
            if utc_timestamp_key(&window[1].completed_at)?
                < utc_timestamp_key(&window[0].completed_at)?
            {
                return Err("paired arena attempt timestamps are out of order".to_owned());
            }
        }
    }

    let latest = latest_paired_attempts(&execution.attempts);
    if latest.len() != plan.jobs.len()
        || latest
            .values()
            .any(|attempt| attempt.status != "completed")
    {
        return Err("paired arena execution lacks a final completed attempt per job".to_owned());
    }
    for job in &plan.jobs {
        let attempt = latest
            .get(job.job_id.as_str())
            .ok_or_else(|| "paired arena execution omits a planned job".to_owned())?;
        if attempt.report.as_ref().map(|artifact| &artifact.path) != Some(&job.report_path)
            || attempt
                .csa
                .iter()
                .map(|artifact| &artifact.path)
                .ne(job.csa_paths.iter())
        {
            return Err("paired arena execution artifact paths differ from its plan".to_owned());
        }
    }

    let quarantined_attempts = execution
        .attempts
        .iter()
        .filter(|attempt| attempt.status == "quarantined")
        .count();
    let expected_jobs = u64::try_from(plan.jobs.len()).expect("job count fits u64");
    if execution.game_count_planned != plan.game_count
        || execution.jobs_planned != expected_jobs
        || execution.jobs_completed != expected_jobs
        || execution.jobs_quarantined != 0
        || execution.games_completed != plan.game_count
        || execution.games_quarantined != 0
        || execution.quarantined_attempts
            != u64::try_from(quarantined_attempts).expect("attempt count fits u64")
    {
        return Err("paired arena execution aggregate counts are inconsistent".to_owned());
    }
    Ok(())
}

#[expect(
    clippy::too_many_lines,
    reason = "attempt validation keeps every closed lifecycle field adjacent to its artifacts"
)]
fn validate_paired_arena_attempt(
    storage: &RegistryStorage,
    attempt: &PairedArenaAttempt,
    plan: &PairedArenaPlan,
    jobs: &BTreeMap<&str, &PairedArenaJob>,
    worker_memory_bytes: u64,
    observed: &mut BTreeMap<StableFileIdentity, ObservedRegistryArtifact>,
    budget: &mut RegistryVerificationBudget,
) -> Result<(), String> {
    validate_identifier(&attempt.job_id, "paired execution attempt jobId")?;
    let job = jobs
        .get(attempt.job_id.as_str())
        .ok_or_else(|| "paired execution attempt has an unknown job".to_owned())?;
    if !(1..=100).contains(&attempt.attempt) {
        return Err("paired execution attempt has an unknown job or invalid number".to_owned());
    }
    validate_utc_timestamp(&attempt.completed_at, "paired execution completedAt")?;
    if attempt.status == "running" {
        if attempt.return_code.is_some()
            || attempt.timed_out.is_some()
            || attempt.output_limit_exceeded.is_some()
            || attempt.memory_limit_exceeded.is_some()
            || attempt.peak_rss_bytes.is_some()
            || attempt.rss_measurement.is_some()
            || attempt.stdout.is_some()
            || attempt.stderr.is_some()
            || attempt.report.is_some()
            || !attempt.csa.is_empty()
            || attempt.quarantine.is_some()
            || attempt.failure_category.is_some()
            || attempt.command_receipt.is_some()
        {
            return Err("paired execution running attempt is not empty".to_owned());
        }
        return Ok(());
    }
    if !matches!(attempt.status.as_str(), "completed" | "quarantined")
        || attempt.return_code.is_none_or(|value| !(-255..=255).contains(&value))
        || attempt.timed_out.is_none()
        || attempt.output_limit_exceeded.is_none()
        || attempt.memory_limit_exceeded.is_none()
        || attempt.peak_rss_bytes.is_some_and(|value| value > i64::MAX as u64)
        || !matches!(
            attempt.rss_measurement.as_deref(),
            Some(
                "process_tree_ps_rss_sum"
                    | "process_tree_ps_short_lived_no_sample"
                    | "unavailable"
            )
        )
    {
        return Err("paired execution attempt outcome fields are invalid".to_owned());
    }
    let rss_measurement = attempt
        .rss_measurement
        .as_deref()
        .expect("completed/quarantined attempt presence was checked");
    let resource_shape_valid = match rss_measurement {
        "process_tree_ps_rss_sum" => attempt.peak_rss_bytes.is_some_and(|peak| {
            attempt.memory_limit_exceeded == Some(peak > worker_memory_bytes)
        }),
        "process_tree_ps_short_lived_no_sample" => {
            attempt.peak_rss_bytes == Some(0) && attempt.memory_limit_exceeded == Some(false)
        }
        "unavailable" => {
            attempt.peak_rss_bytes.is_none() && attempt.memory_limit_exceeded == Some(true)
        }
        _ => false,
    };
    if !resource_shape_valid {
        return Err("paired execution attempt resource evidence is inconsistent".to_owned());
    }
    let stdout = attempt
        .stdout
        .as_ref()
        .ok_or_else(|| "paired execution attempt lacks stdout evidence".to_owned())?;
    let stderr = attempt
        .stderr
        .as_ref()
        .ok_or_else(|| "paired execution attempt lacks stderr evidence".to_owned())?;
    for artifact in [stdout, stderr] {
        validate_artifact_ref(artifact)?;
        let _ = verify_registry_artifact_cached(storage, artifact, observed, budget, None)?;
    }
    let report_completed_at = if attempt.status == "completed" {
        if attempt.failure_category.is_some()
            || attempt.timed_out != Some(false)
            || attempt.output_limit_exceeded != Some(false)
            || attempt.memory_limit_exceeded != Some(false)
            || attempt.return_code != Some(0)
            || attempt.quarantine.is_some()
            || attempt.csa.len() != 2
            || rss_measurement == "unavailable"
            || attempt
                .peak_rss_bytes
                .is_none_or(|peak| peak > worker_memory_bytes)
        {
            return Err("paired execution completed attempt is internally inconsistent".to_owned());
        }
        let report = attempt
            .report
            .as_ref()
            .ok_or_else(|| "paired execution completed attempt lacks its report".to_owned())?;
        validate_artifact_ref(report)?;
        let report_bytes = verify_registry_artifact_cached(
            storage,
            report,
            observed,
            budget,
            Some(MAX_PHASE2_ARENA_REPORT_BYTES),
        )?
        .ok_or_else(|| "paired arena report bytes were not retained".to_owned())?;
        let report_value = parse_unique_json(report_bytes.as_slice(), "paired arena report")?;
        let parsed_report: Phase2ArenaReport =
            deserialize_closed_json(&report_value, "paired arena report")?;
        if parsed_report.schema != "phase2_arena_report/v2" {
            return Err("paired arena report has an unsupported schema".to_owned());
        }
        let completed_at = parsed_report
            .run
            .completed_at
            .ok_or_else(|| "paired arena report is not complete".to_owned())?;
        validate_utc_timestamp(&completed_at, "paired arena report completedAt")?;
        for csa in &attempt.csa {
            validate_artifact_ref(csa)?;
            let _ = verify_registry_artifact_cached(
                storage,
                csa,
                observed,
                budget,
                Some(MAX_HUMAN_CSA_BYTES),
            )?;
        }
        Some(completed_at)
    } else {
        let expected_failure = if attempt.timed_out == Some(true) {
            "timeout"
        } else if attempt.memory_limit_exceeded == Some(true) {
            "memory_limit"
        } else if attempt.output_limit_exceeded == Some(true) {
            "output_limit"
        } else if attempt.return_code != Some(0) {
            "nonzero_exit"
        } else {
            "missing_or_invalid_artifact"
        };
        if attempt.failure_category.as_deref() != Some(expected_failure)
            || attempt.report.is_some()
            || !attempt.csa.is_empty()
        {
            return Err("paired execution quarantined attempt is internally inconsistent".to_owned());
        }
        let quarantine = attempt
            .quarantine
            .as_ref()
            .ok_or_else(|| "paired execution quarantined attempt lacks evidence".to_owned())?;
        validate_artifact_ref(quarantine)?;
        let quarantine_bytes = verify_registry_artifact_cached(
            storage,
            quarantine,
            observed,
            budget,
            Some(MAX_QUARANTINE_RECORD_BYTES),
        )?
        .ok_or_else(|| "paired quarantine record bytes were not retained".to_owned())?;
        let quarantine_value =
            parse_unique_json(quarantine_bytes.as_slice(), "paired quarantine record")?;
        let record: PairedQuarantineRecord =
            deserialize_closed_json(&quarantine_value, "paired quarantine record")?;
        validate_paired_quarantine_record(storage, &record, attempt, job, observed, budget)?;
        None
    };
    if let Some(report_completed) = report_completed_at.as_deref()
        && utc_timestamp_key(&attempt.completed_at)? < utc_timestamp_key(report_completed)?
    {
        return Err("paired attempt completed before its arena report".to_owned());
    }
    let command_receipt = attempt
        .command_receipt
        .as_ref()
        .ok_or_else(|| "terminal paired attempt lacks its command receipt".to_owned())?;
    validate_paired_attempt_command_receipt(
        storage,
        command_receipt,
        attempt,
        plan,
        job,
        observed,
        budget,
    )?;
    Ok(())
}

fn validate_paired_attempt_command_receipt(
    storage: &RegistryStorage,
    reference: &RegistryArtifact,
    attempt: &PairedArenaAttempt,
    plan: &PairedArenaPlan,
    job: &PairedArenaJob,
    observed: &mut BTreeMap<StableFileIdentity, ObservedRegistryArtifact>,
    budget: &mut RegistryVerificationBudget,
) -> Result<(), String> {
    validate_artifact_ref(reference)?;
    let bytes = verify_registry_artifact_cached(
        storage,
        reference,
        observed,
        budget,
        Some(MAX_PHASE6_COMMAND_RECEIPT_BYTES),
    )?
    .ok_or_else(|| "paired attempt command receipt bytes were not retained".to_owned())?;
    let value = parse_unique_json(bytes.as_slice(), "paired attempt command receipt")?;
    let receipt: PairedAttemptCommandReceipt =
        deserialize_closed_json(&value, "paired attempt command receipt")?;
    if receipt.schema != "phase6_attempt_command_receipt/v1"
        || receipt.plan_sha256 != plan.plan_sha256
        || receipt.job_id != attempt.job_id
        || receipt.attempt != attempt.attempt
        || receipt.command != job.command
        || receipt.engine != plan.engine
        || receipt.engine_build_receipt != plan.engine_build_receipt
        || receipt.result.return_code != attempt.return_code.expect("terminal outcome")
        || receipt.result.timed_out != attempt.timed_out.expect("terminal outcome")
        || receipt.result.output_limit_exceeded
            != attempt.output_limit_exceeded.expect("terminal outcome")
        || receipt.result.memory_limit_exceeded
            != attempt.memory_limit_exceeded.expect("terminal outcome")
        || receipt.result.peak_rss_bytes != attempt.peak_rss_bytes
        || Some(receipt.result.rss_measurement.as_str()) != attempt.rss_measurement.as_deref()
        || attempt.stdout.as_ref() != Some(&receipt.stdout)
        || attempt.stderr.as_ref() != Some(&receipt.stderr)
        || receipt.report != attempt.report
        || receipt.csa != attempt.csa
        || receipt.quarantine != attempt.quarantine
        || receipt.failure_category != attempt.failure_category
        || receipt.completed_at != attempt.completed_at
    {
        return Err("paired attempt differs from its command receipt".to_owned());
    }
    validate_sha256(&receipt.receipt_sha256)?;
    validate_json_self_hash(&value, "receiptSha256", &receipt.receipt_sha256)?;
    validate_phase6_process_receipt(
        storage,
        &receipt.process_receipt,
        &receipt,
        plan,
        observed,
        budget,
    )
}

fn validate_phase6_process_receipt(
    storage: &RegistryStorage,
    reference: &RegistryArtifact,
    attempt_receipt: &PairedAttemptCommandReceipt,
    plan: &PairedArenaPlan,
    observed: &mut BTreeMap<StableFileIdentity, ObservedRegistryArtifact>,
    budget: &mut RegistryVerificationBudget,
) -> Result<(), String> {
    validate_artifact_ref(reference)?;
    let bytes = verify_registry_artifact_cached(
        storage,
        reference,
        observed,
        budget,
        Some(MAX_PHASE6_COMMAND_RECEIPT_BYTES),
    )?
    .ok_or_else(|| "Phase 6 process receipt bytes were not retained".to_owned())?;
    let value = parse_unique_json(bytes.as_slice(), "Phase 6 process receipt")?;
    let receipt: Phase6ProcessReceipt =
        deserialize_closed_json(&value, "Phase 6 process receipt")?;
    let invocation = Phase6CommandInvocation {
        command: &receipt.command,
        resume: receipt.resume,
        memory_limit_mib: receipt.memory_limit_mib,
        expected_executable: &plan.engine,
        engine_build_receipt: &plan.engine_build_receipt,
        runtime_receipt: None,
    };
    let invocation_value = serde_json::to_value(invocation)
        .map_err(|error| format!("cannot canonicalize Phase 6 process invocation: {error}"))?;
    let expected_command_sha256 = python_canonical_sha256(&invocation_value)?;
    if receipt.schema != "phase6_command_receipt/v2"
        || receipt.command != attempt_receipt.command
        || receipt.memory_limit_mib != plan.memory_per_worker_mib
        || receipt.expected_executable.as_ref() != Some(&plan.engine)
        || receipt.engine_build_receipt != plan.engine_build_receipt
        || receipt.runtime_receipt.is_some()
        || receipt.command_sha256 != expected_command_sha256
        || receipt.return_code != attempt_receipt.result.return_code
        || receipt.timed_out != attempt_receipt.result.timed_out
        || receipt.output_limit_exceeded != attempt_receipt.result.output_limit_exceeded
        || receipt.memory_limit_exceeded != attempt_receipt.result.memory_limit_exceeded
        || receipt.peak_rss_bytes != attempt_receipt.result.peak_rss_bytes
        || receipt.rss_measurement != attempt_receipt.result.rss_measurement
        || receipt.stdout != attempt_receipt.stdout
        || receipt.stderr != attempt_receipt.stderr
    {
        return Err("Phase 6 process receipt differs from its attempt evidence".to_owned());
    }
    validate_sha256(&receipt.command_sha256)
}

fn validate_paired_quarantine_record(
    storage: &RegistryStorage,
    record: &PairedQuarantineRecord,
    attempt: &PairedArenaAttempt,
    job: &PairedArenaJob,
    observed: &mut BTreeMap<StableFileIdentity, ObservedRegistryArtifact>,
    budget: &mut RegistryVerificationBudget,
) -> Result<(), String> {
    let expected_stdout = attempt
        .stdout
        .as_ref()
        .ok_or_else(|| "quarantined attempt lacks stdout".to_owned())?;
    let expected_stderr = attempt
        .stderr
        .as_ref()
        .ok_or_else(|| "quarantined attempt lacks stderr".to_owned())?;
    let no_observed_game_artifacts = record.observed_report.is_none() && record.observed_csa.is_empty();
    let complete_observed_game_artifacts = record
        .observed_report
        .as_ref()
        .is_some_and(|report| report.path == job.report_path)
        && record.observed_csa.len() == job.csa_paths.len()
        && record
            .observed_csa
            .iter()
            .map(|artifact| &artifact.path)
            .eq(job.csa_paths.iter());
    let observed_shape_valid = if attempt.failure_category.as_deref()
        == Some("missing_or_invalid_artifact")
    {
        no_observed_game_artifacts || complete_observed_game_artifacts
    } else {
        no_observed_game_artifacts
    };
    if record.schema != "phase6_quarantined_job/v1"
        || record.job_id != attempt.job_id
        || record.attempt != attempt.attempt
        || Some(record.failure_category.as_str()) != attempt.failure_category.as_deref()
        || Some(record.return_code) != attempt.return_code
        || Some(record.timed_out) != attempt.timed_out
        || Some(record.output_limit_exceeded) != attempt.output_limit_exceeded
        || Some(record.memory_limit_exceeded) != attempt.memory_limit_exceeded
        || record.peak_rss_bytes != attempt.peak_rss_bytes
        || Some(record.rss_measurement.as_str()) != attempt.rss_measurement.as_deref()
        || &record.stdout != expected_stdout
        || &record.stderr != expected_stderr
        || record.observed_csa.len() > 2
        || !observed_shape_valid
        || record.failure_detail.as_ref().is_some_and(|detail| {
            detail.is_empty() || detail.len() > 1_024 || detail.contains('\0')
        })
    {
        return Err("paired quarantine record differs from its attempt".to_owned());
    }
    for artifact in record
        .observed_report
        .iter()
        .chain(record.observed_csa.iter())
    {
        validate_artifact_ref(artifact)?;
        let _ = verify_registry_artifact_cached(storage, artifact, observed, budget, None)?;
    }
    Ok(())
}

#[expect(
    clippy::too_many_arguments,
    reason = "arena recollection carries immutable references plus the shared same-FD budget"
)]
fn collect_paired_arena_results(
    storage: &RegistryStorage,
    registry: &ModelRegistry,
    plan: &PairedArenaPlan,
    execution: &PairedArenaExecution,
    plan_reference: RegistryArtifact,
    execution_reference: RegistryArtifact,
    observed: &mut BTreeMap<StableFileIdentity, ObservedRegistryArtifact>,
    budget: &mut RegistryVerificationBudget,
) -> Result<ArenaResults, String> {
    let latest = latest_paired_attempts(&execution.attempts);
    let mut quarantined_by_job = BTreeMap::<&str, u64>::new();
    for attempt in &execution.attempts {
        if attempt.status == "quarantined" {
            let count = quarantined_by_job
                .entry(attempt.job_id.as_str())
                .or_default();
            *count = count
                .checked_add(1)
                .ok_or_else(|| "paired arena quarantine count overflow".to_owned())?;
        }
    }

    let mut games = Vec::with_capacity(plan.jobs.len() * 2);
    for job in &plan.jobs {
        budget.ensure_time()?;
        let attempt = latest
            .get(job.job_id.as_str())
            .ok_or_else(|| format!("arena job lacks a completed attempt: {}", job.job_id))?;
        if attempt.status != "completed" {
            return Err(format!(
                "arena job lacks a completed attempt: {}",
                job.job_id
            ));
        }
        let report_reference = attempt
            .report
            .as_ref()
            .ok_or_else(|| format!("arena job lacks a report: {}", job.job_id))?;
        let report_bytes = verify_registry_artifact_cached(
            storage,
            report_reference,
            observed,
            budget,
            Some(MAX_PHASE2_ARENA_REPORT_BYTES),
        )?
        .ok_or_else(|| "paired arena report bytes were not retained".to_owned())?;
        let report_value = parse_unique_json(
            report_bytes.as_slice(),
            &format!("paired arena report {}", job.job_id),
        )?;
        let report: Phase2ArenaReport =
            deserialize_closed_json(&report_value, "paired arena report")?;
        validate_phase6_pair_report_binding(
            storage,
            &report,
            job,
            plan,
            report_reference,
            &attempt.csa,
            registry,
            observed,
            budget,
        )?;

        let crashes = quarantined_by_job
            .get(job.job_id.as_str())
            .copied()
            .unwrap_or(0);
        for (index, report_game) in report.games.iter().enumerate() {
            let (black_model_id, white_model_id) = if index == 0 {
                (
                    plan.challenger.model_id.clone(),
                    plan.champion.model_id.clone(),
                )
            } else {
                (
                    plan.champion.model_id.clone(),
                    plan.challenger.model_id.clone(),
                )
            };
            games.push(ArenaResultGame {
                game_id: job.game_ids[index].clone(),
                pair_id: job.job_id.clone(),
                start_group: job.start_group.clone(),
                start_position_id: job.start_position_id.clone(),
                black_model_id,
                white_model_id,
                result: report_game.result.clone(),
                plies: report_game.moves,
                illegal_moves: if index == 0 {
                    report.metrics.illegal_moves
                } else {
                    0
                },
                crashes: if index == 0 { crashes } else { 0 },
                metrics: normalized_arena_game_metrics(report_game),
            });
        }
    }

    let result = ArenaResults {
        schema: "phase6_paired_arena_results/v1".to_owned(),
        generation_id: plan.generation_id.clone(),
        plan: plan_reference,
        execution: execution_reference,
        champion_model_id: plan.champion.model_id.clone(),
        challenger_model_id: plan.challenger.model_id.clone(),
        games,
    };
    validate_arena_results(&result)?;
    Ok(result)
}

fn normalized_arena_game_metrics(game: &Phase2ArenaGame) -> ArenaGameMetrics {
    ArenaGameMetrics {
        champion_inference_calls: Some(game.player_b_neural_inference_calls),
        champion_inference_time_ns: Some(game.player_b_neural_inference_time_ns),
        challenger_inference_calls: Some(game.player_a_neural_inference_calls),
        challenger_inference_time_ns: Some(game.player_a_neural_inference_time_ns),
        champion_search_nodes: Some(game.player_b_search_nodes),
        champion_search_elapsed_ms: Some(game.player_b_search_elapsed_ms),
        challenger_search_nodes: Some(game.player_a_search_nodes),
        challenger_search_elapsed_ms: Some(game.player_a_search_elapsed_ms),
        champion_search_depth_sum: Some(game.player_b_depth_sum),
        champion_searches: Some(game.player_b_searches),
        challenger_search_depth_sum: Some(game.player_a_depth_sum),
        challenger_searches: Some(game.player_a_searches),
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "the report binding joins one plan job, its files, and immutable model metadata"
)]
fn validate_phase6_pair_report_binding(
    storage: &RegistryStorage,
    report: &Phase2ArenaReport,
    job: &PairedArenaJob,
    plan: &PairedArenaPlan,
    report_reference: &RegistryArtifact,
    csa_references: &[RegistryArtifact],
    registry: &ModelRegistry,
    observed: &mut BTreeMap<StableFileIdentity, ObservedRegistryArtifact>,
    budget: &mut RegistryVerificationBudget,
) -> Result<(), String> {
    if report.schema != "phase2_arena_report/v2" {
        return Err(format!(
            "paired arena report {} has an unsupported schema",
            job.job_id
        ));
    }
    let options = paired_arena_command_options(&job.command, &plan.engine)?;
    let player_a = expected_phase6_report_player(
        storage,
        &plan.challenger,
        registry,
        parse_ascii_decimal_option(&options, "--a-depth", 1, 64)?,
        parse_ascii_decimal_option(&options, "--a-hash-mb", 1, 1_024)?,
        observed,
        budget,
    )?;
    let player_b = expected_phase6_report_player(
        storage,
        &plan.champion,
        registry,
        parse_ascii_decimal_option(&options, "--b-depth", 1, 64)?,
        parse_ascii_decimal_option(&options, "--b-hash-mb", 1, 1_024)?,
        observed,
        budget,
    )?;
    validate_phase2_report_run(report, job, plan, &player_a, &player_b)?;
    validate_phase2_report_games(
        storage,
        report,
        job,
        csa_references,
        &player_a,
        &player_b,
        observed,
        budget,
    )?;
    validate_phase2_report_metrics(report, &player_a, &player_b)?;
    if report_reference.path != job.report_path {
        return Err(format!(
            "paired arena report {} path differs from its plan",
            job.job_id
        ));
    }
    Ok(())
}

fn expected_phase6_report_player(
    storage: &RegistryStorage,
    planned: &PairedModelSpec,
    registry: &ModelRegistry,
    depth: u64,
    hash_megabytes: u64,
    observed: &BTreeMap<StableFileIdentity, ObservedRegistryArtifact>,
    budget: &mut RegistryVerificationBudget,
) -> Result<Phase2ArenaPlayer, String> {
    let registered = registry
        .models
        .iter()
        .find(|model| model.model_id == planned.model_id)
        .ok_or_else(|| "paired arena report model is absent from the registry".to_owned())?;
    if registered.artifact != planned.artifact || registered.evaluator_kind != "neural" {
        return Err("paired arena report model differs from the registry".to_owned());
    }
    let metadata = cached_model_metadata(storage, &planned.artifact, observed, budget)?;
    let architecture_version = u64::from(metadata.architecture_version);
    let quantization = quantization_name(metadata.quantization).to_owned();
    if registered.architecture_version != architecture_version.to_string()
        || registered.quantization != quantization
    {
        return Err("paired arena model metadata differs from its OSAVAL01 bytes".to_owned());
    }
    Ok(Phase2ArenaPlayer {
        label: format!(
            "search:neural:d{depth}:h{hash_megabytes}:tt-on:book-off:m-{}",
            &planned.artifact.sha256[..12]
        ),
        evaluator_kind: "neural".to_owned(),
        search_depth: Some(depth),
        hash_megabytes: Some(hash_megabytes),
        transposition: Some(true),
        model_artifact_sha256: Some(planned.artifact.sha256.clone()),
        model_artifact_size: Some(planned.artifact.size),
        model_payload_sha256: Some(metadata.payload_sha256),
        architecture_version: Some(architecture_version),
        quantization: Some(quantization),
        opening_enabled: false,
    })
}

fn cached_model_metadata(
    storage: &RegistryStorage,
    artifact: &RegistryArtifact,
    observed: &BTreeMap<StableFileIdentity, ObservedRegistryArtifact>,
    budget: &mut RegistryVerificationBudget,
) -> Result<VerifiedModelMetadata, String> {
    budget.charge_reference()?;
    validate_artifact_ref(artifact)?;
    let display_path = storage.root_path.join(&artifact.path);
    let file = storage
        .root
        .open_relative_regular(&artifact.path)
        .map_err(|error| {
            format!(
                "cannot reopen paired arena model {}: {error}",
                display_path.display()
            )
        })?;
    let identity = file.stable_identity().map_err(|error| {
        format!(
            "cannot rebind paired arena model identity {}: {error}",
            display_path.display()
        )
    })?;
    let verified = observed.get(&identity).ok_or_else(|| {
        "paired arena model changed identity after registry model verification".to_owned()
    })?;
    verify_observed_artifact(artifact, &verified.sha256, verified.size)?;
    verified
        .model
        .clone()
        .ok_or_else(|| "paired arena model identity was not parsed as OSAVAL01".to_owned())
}

fn validate_phase2_report_run(
    report: &Phase2ArenaReport,
    job: &PairedArenaJob,
    plan: &PairedArenaPlan,
    player_a: &Phase2ArenaPlayer,
    player_b: &Phase2ArenaPlayer,
) -> Result<(), String> {
    let run = &report.run;
    validate_utc_timestamp(&run.started_at, "arena report startedAt")?;
    let completed_at = run
        .completed_at
        .as_deref()
        .ok_or_else(|| "paired arena report is not completed".to_owned())?;
    validate_utc_timestamp(completed_at, "arena report completedAt")?;
    if utc_timestamp_key(completed_at)? < utc_timestamp_key(&run.started_at)? {
        return Err("paired arena report completion precedes its start".to_owned());
    }
    validate_sha256(&run.config_sha256)?;
    let signature = ArenaConfigSignature {
        games: run.game_limit,
        seed: run.seed,
        initial_sfen: run.initial_sfen.clone(),
        max_plies: run.max_plies,
        git_commit: run.git_commit.clone(),
        budget_kind: run.budget.kind.clone(),
        budget_value: run.budget.value,
        player_a: phase2_signature_player(&run.player_a),
        player_b: phase2_signature_player(&run.player_b),
        opening: ArenaConfigSignatureOpening {
            enabled: run.opening.enabled,
            artifact_sha256: run.opening.artifact_sha256.clone(),
            artifact_size: run.opening.artifact_size,
            max_plies: run.opening.max_plies,
        },
    };
    let expected_config_sha256 = sha256_bytes(&arena_config_signature_bytes(&signature));
    if run.seed != job.seed
        || run.seed > MAX_JSON_SAFE_INTEGER
        || run.game_limit != 2
        || run.git_commit.as_deref() != Some(plan.git_commit.as_str())
        || run.initial_sfen != job.sfen
        || run.max_plies != PHASE6_ARENA_MAX_PLIES
        || run.budget.kind != "nodes"
        || run.budget.value != PHASE6_ARENA_NODES_PER_MOVE
        || run.player_a != *player_a
        || run.player_b != *player_b
        || run.opening
            != (Phase2ArenaOpening {
                enabled: false,
                artifact_sha256: None,
                artifact_size: None,
                max_plies: None,
            })
        || run.config_sha256 != expected_config_sha256
    {
        return Err(format!(
            "paired arena report {} run differs from its immutable plan",
            job.job_id
        ));
    }
    validate_phase6_plan_sfen(&run.initial_sfen)?;
    let suffix = format!(
        " a={} b={} budget=Nodes({})",
        player_a.label, player_b.label, PHASE6_ARENA_NODES_PER_MOVE
    );
    let prefix = run.engine.strip_suffix(&suffix).ok_or_else(|| {
        format!(
            "paired arena report {} engine identity differs from its players",
            job.job_id
        )
    })?;
    if prefix.is_empty()
        || run.engine.len() > 4_096
        || run.engine.contains(['\r', '\n', '\0'])
    {
        return Err(format!(
            "paired arena report {} engine identity is invalid",
            job.job_id
        ));
    }
    Ok(())
}

fn phase2_signature_player(player: &Phase2ArenaPlayer) -> ArenaConfigSignaturePlayer {
    ArenaConfigSignaturePlayer {
        label: player.label.clone(),
        evaluator_kind: player.evaluator_kind.clone(),
        search_depth: player.search_depth,
        hash_megabytes: player.hash_megabytes,
        transposition: player.transposition,
        model_artifact_sha256: player.model_artifact_sha256.clone(),
        model_artifact_size: player.model_artifact_size,
        model_payload_sha256: player.model_payload_sha256.clone(),
        architecture_version: player.architecture_version,
        quantization: player.quantization.clone(),
        opening_enabled: player.opening_enabled,
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "game validation binds each closed report row to both planned CSA files"
)]
fn validate_phase2_report_games(
    storage: &RegistryStorage,
    report: &Phase2ArenaReport,
    job: &PairedArenaJob,
    csa_references: &[RegistryArtifact],
    player_a: &Phase2ArenaPlayer,
    player_b: &Phase2ArenaPlayer,
    observed: &mut BTreeMap<StableFileIdentity, ObservedRegistryArtifact>,
    budget: &mut RegistryVerificationBudget,
) -> Result<(), String> {
    if report.games.len() != 2 || csa_references.len() != 2 {
        return Err(format!(
            "paired arena report {} must contain exactly two games and CSA files",
            job.job_id
        ));
    }
    for (index, (game, reference)) in report
        .games
        .iter()
        .zip(csa_references)
        .enumerate()
    {
        let expected_id = u64::try_from(index).expect("two-game index fits u64");
        let (expected_black, expected_white) = if index == 0 {
            (&player_a.label, &player_b.label)
        } else {
            (&player_b.label, &player_a.label)
        };
        let expected_relative = PathBuf::from(format!("games/game-{:06}.csa", index + 1));
        if game.id != expected_id
            || &game.black != expected_black
            || &game.white != expected_white
            || !matches!(
                game.result.as_str(),
                "black_win" | "white_win" | "draw" | "max_plies"
            )
            || game.moves > report.run.max_plies
            || game.csa_path != expected_relative
            || reference.path != job.csa_paths[index]
            || game.csa_sha256 != reference.sha256
            || game.csa_size != reference.size
            || game.csa_size == 0
            || game.csa_size > MAX_JSON_SAFE_INTEGER
        {
            return Err(format!(
                "paired arena report {} game {index} differs from its plan or CSA artifact",
                job.job_id
            ));
        }
        validate_sha256(&game.csa_sha256)?;
        let bytes = verify_registry_artifact_cached(
            storage,
            reference,
            observed,
            budget,
            Some(MAX_HUMAN_CSA_BYTES),
        )?
        .ok_or_else(|| "paired arena CSA bytes were not retained".to_owned())?;
        let text = std::str::from_utf8(bytes.as_slice())
            .map_err(|_| "paired arena CSA is not valid UTF-8".to_owned())?;
        let parsed = parse_csa_game(text)
            .map_err(|error| format!("paired arena CSA parse failed: {error}"))?;
        let canonical = to_csa_game(&parsed)
            .map_err(|error| format!("paired arena CSA canonicalization failed: {error}"))?;
        if canonical.as_bytes() != bytes.as_slice() {
            return Err("paired arena CSA is not canonical".to_owned());
        }
        let initial = parse_sfen(&report.run.initial_sfen)
            .map_err(|error| format!("paired arena report initial SFEN is invalid: {error}"))?;
        if parsed.version != "V3.0"
            || !parsed.metadata.is_empty()
            || parsed.initial_position != initial
            || parsed.black_name.as_deref() != Some(game.black.as_str())
            || parsed.white_name.as_deref() != Some(game.white.as_str())
            || parsed.moves.len() != usize::try_from(game.moves).unwrap_or(usize::MAX)
        {
            return Err("paired arena CSA identity differs from its report".to_owned());
        }
        let selections = validate_phase2_csa_terminal(&parsed, game, report.run.max_plies)?;
        let (black_searches, white_searches) = selection_counts(initial.side_to_move(), selections);
        let (first_player_searches, second_player_searches) = if index == 0 {
            (black_searches, white_searches)
        } else {
            (white_searches, black_searches)
        };
        validate_phase2_game_counters(
            game,
            player_a,
            player_b,
            first_player_searches,
            second_player_searches,
        )?;
    }
    Ok(())
}

fn validate_phase2_csa_terminal(
    csa: &CsaGame,
    report_game: &Phase2ArenaGame,
    max_plies: u64,
) -> Result<u64, String> {
    let mut replay = Game::new(csa.initial_position.clone());
    for &movement in &csa.moves {
        if replay.end().is_some() {
            return Err("paired arena CSA continues after a terminal position".to_owned());
        }
        replay
            .play(movement)
            .map_err(|error| format!("paired arena CSA contains an illegal move: {error}"))?;
    }
    let special = csa
        .special_move
        .as_ref()
        .ok_or_else(|| "paired arena CSA lacks a terminal result".to_owned())?;
    let (expected_result, expected_validation, extra_selection) = match (special, replay.end()) {
        (CsaSpecialMove::Checkmate, Some(GameEnd::Checkmate { winner })) => (
            phase2_side_result(winner),
            CsaResultValidation::Verified,
            0,
        ),
        (
            CsaSpecialMove::Repetition,
            Some(GameEnd::Repetition(RepetitionOutcome::NoContest)),
        ) => ("draw", CsaResultValidation::Verified, 0),
        (
            CsaSpecialMove::PerpetualCheck,
            Some(GameEnd::Repetition(RepetitionOutcome::PerpetualCheckLoss(loser))),
        ) => (
            phase2_side_result(loser.opposite()),
            CsaResultValidation::Verified,
            0,
        ),
        (CsaSpecialMove::Resign, None) => (
            phase2_side_result(replay.position().side_to_move().opposite()),
            CsaResultValidation::ExternalCondition,
            1,
        ),
        (CsaSpecialMove::MaxMoves, None)
            if u64::try_from(csa.moves.len()).expect("bounded move count fits u64") == max_plies =>
        {
            ("max_plies", CsaResultValidation::ExternalCondition, 0)
        }
        _ => {
            return Err(
                "paired arena CSA has an unsupported or inconsistent terminal result".to_owned(),
            );
        }
    };
    if csa.result_validation != expected_validation || report_game.result != expected_result {
        return Err("paired arena CSA result differs from its report".to_owned());
    }
    u64::try_from(csa.moves.len())
        .expect("bounded move count fits u64")
        .checked_add(extra_selection)
        .ok_or_else(|| "paired arena search-selection count overflow".to_owned())
}

const fn phase2_side_result(side: Side) -> &'static str {
    match side {
        Side::Black => "black_win",
        Side::White => "white_win",
    }
}

fn selection_counts(initial_side: Side, selections: u64) -> (u64, u64) {
    let first = selections / 2 + selections % 2;
    let second = selections / 2;
    match initial_side {
        Side::Black => (first, second),
        Side::White => (second, first),
    }
}

fn validate_phase2_game_counters(
    game: &Phase2ArenaGame,
    player_a: &Phase2ArenaPlayer,
    player_b: &Phase2ArenaPlayer,
    expected_first_player_searches: u64,
    expected_second_player_searches: u64,
) -> Result<(), String> {
    let counters = [
        game.neural_inference_calls,
        game.neural_inference_time_ns,
        game.player_a_search_nodes,
        game.player_a_search_elapsed_ms,
        game.player_a_depth_sum,
        game.player_a_searches,
        game.player_a_neural_inference_calls,
        game.player_a_neural_inference_time_ns,
        game.player_b_search_nodes,
        game.player_b_search_elapsed_ms,
        game.player_b_depth_sum,
        game.player_b_searches,
        game.player_b_neural_inference_calls,
        game.player_b_neural_inference_time_ns,
    ];
    let first_depth_limit = player_a.search_depth.unwrap_or(0);
    let second_depth_limit = player_b.search_depth.unwrap_or(0);
    if counters.iter().any(|value| *value > MAX_JSON_SAFE_INTEGER)
        || game.player_a_searches != expected_first_player_searches
        || game.player_b_searches != expected_second_player_searches
        || game.player_a_search_nodes
            > PHASE6_ARENA_NODES_PER_MOVE.saturating_mul(game.player_a_searches)
        || game.player_b_search_nodes
            > PHASE6_ARENA_NODES_PER_MOVE.saturating_mul(game.player_b_searches)
        || game.player_a_search_nodes < game.player_a_searches
        || game.player_b_search_nodes < game.player_b_searches
        || game.player_a_depth_sum > first_depth_limit.saturating_mul(game.player_a_searches)
        || game.player_b_depth_sum > second_depth_limit.saturating_mul(game.player_b_searches)
        || game.player_a_neural_inference_calls
            > game
                .player_a_search_nodes
                .saturating_add(game.player_a_searches)
        || game.player_b_neural_inference_calls
            > game
                .player_b_search_nodes
                .saturating_add(game.player_b_searches)
        || (game.player_a_searches == 0 && game.player_a_neural_inference_calls != 0)
        || (game.player_b_searches == 0 && game.player_b_neural_inference_calls != 0)
        || (game.player_a_neural_inference_calls == 0
            && game.player_a_neural_inference_time_ns != 0)
        || (game.player_b_neural_inference_calls == 0
            && game.player_b_neural_inference_time_ns != 0)
        || (player_a.evaluator_kind != "neural"
            && (game.player_a_neural_inference_calls != 0
                || game.player_a_neural_inference_time_ns != 0))
        || (player_b.evaluator_kind != "neural"
            && (game.player_b_neural_inference_calls != 0
                || game.player_b_neural_inference_time_ns != 0))
        || game.neural_inference_calls
            != checked_json_sum(&[
                game.player_a_neural_inference_calls,
                game.player_b_neural_inference_calls,
            ])?
        || game.neural_inference_time_ns
            != checked_json_sum(&[
                game.player_a_neural_inference_time_ns,
                game.player_b_neural_inference_time_ns,
            ])?
    {
        return Err("paired arena report game counters are inconsistent".to_owned());
    }
    Ok(())
}

#[expect(
    clippy::float_cmp,
    reason = "the cross-language wire contract requires exact deterministic decimal ratios"
)]
fn validate_phase2_report_metrics(
    report: &Phase2ArenaReport,
    player_a: &Phase2ArenaPlayer,
    player_b: &Phase2ArenaPlayer,
) -> Result<(), String> {
    let metrics = &report.metrics;
    let counters = phase2_metric_counters(metrics);
    if metrics.games != 2
        || counters.iter().any(|value| *value > MAX_JSON_SAFE_INTEGER)
        || metrics.peak_memory_bytes.is_some()
        || ![
            metrics.nodes_per_second,
            metrics.average_depth,
            metrics.milliseconds_per_move,
        ]
        .iter()
        .all(|value| value.is_finite() && *value >= 0.0)
        || ![metrics.tt_hit_rate, metrics.cutoff_rate, metrics.pruning_rate]
            .iter()
            .all(|value| value.is_finite() && (0.0..=1.0).contains(value))
        || metrics.illegal_moves != 0
        || metrics.player_a_search_nodes
            > PHASE6_ARENA_NODES_PER_MOVE.saturating_mul(metrics.player_a_searches)
        || metrics.player_b_search_nodes
            > PHASE6_ARENA_NODES_PER_MOVE.saturating_mul(metrics.player_b_searches)
        || metrics.player_a_search_nodes < metrics.player_a_searches
        || metrics.player_b_search_nodes < metrics.player_b_searches
        || metrics.player_a_depth_sum
            > player_a
                .search_depth
                .unwrap_or(0)
                .saturating_mul(metrics.player_a_searches)
        || metrics.player_b_depth_sum
            > player_b
                .search_depth
                .unwrap_or(0)
                .saturating_mul(metrics.player_b_searches)
        || metrics.player_a_neural_inference_calls
            > metrics
                .player_a_search_nodes
                .saturating_add(metrics.player_a_searches)
        || metrics.player_b_neural_inference_calls
            > metrics
                .player_b_search_nodes
                .saturating_add(metrics.player_b_searches)
        || (metrics.player_a_searches == 0 && metrics.player_a_neural_inference_calls != 0)
        || (metrics.player_b_searches == 0 && metrics.player_b_neural_inference_calls != 0)
        || (metrics.player_a_neural_inference_calls == 0
            && metrics.player_a_neural_inference_time_ns != 0)
        || (metrics.player_b_neural_inference_calls == 0
            && metrics.player_b_neural_inference_time_ns != 0)
        || (player_a.evaluator_kind != "neural"
            && (metrics.player_a_neural_inference_calls != 0
                || metrics.player_a_neural_inference_time_ns != 0))
        || (player_b.evaluator_kind != "neural"
            && (metrics.player_b_neural_inference_calls != 0
                || metrics.player_b_neural_inference_time_ns != 0))
    {
        return Err("paired arena report aggregate metrics are outside their bounds".to_owned());
    }
    // v2 exposes only the derived TT/cutoff/pruning rates, not their raw denominators.
    // They are therefore range-checked observations and deliberately excluded from every
    // promotion/performance calculation; no unverifiable equality is inferred here.
    validate_phase2_metric_counter_sums(report)?;
    validate_phase2_metric_outcomes(report, player_a, player_b)?;
    let search_nodes = checked_json_sum(&[
        metrics.player_a_search_nodes,
        metrics.player_b_search_nodes,
    ])?;
    let search_elapsed_ms = checked_json_sum(&[
        metrics.player_a_search_elapsed_ms,
        metrics.player_b_search_elapsed_ms,
    ])?;
    let depth_sum = checked_json_sum(&[
        metrics.player_a_depth_sum,
        metrics.player_b_depth_sum,
    ])?;
    let searches = checked_json_sum(&[
        metrics.player_a_searches,
        metrics.player_b_searches,
    ])?;
    if metrics.nodes_per_second != phase2_decimal_ratio(search_nodes, search_elapsed_ms, 1_000)?
        || metrics.average_depth != phase2_decimal_ratio(depth_sum, searches, 1)?
        || metrics.milliseconds_per_move
            != phase2_decimal_ratio(search_elapsed_ms, searches, 1)?
    {
        return Err("paired arena report ratios disagree with raw counters".to_owned());
    }
    Ok(())
}

fn phase2_metric_counters(metrics: &Phase2ArenaMetrics) -> [u64; 21] {
    [
        metrics.finished_games,
        metrics.player_a_wins,
        metrics.player_b_wins,
        metrics.search_wins,
        metrics.draws,
        metrics.neural_inference_calls,
        metrics.neural_inference_time_ns,
        metrics.player_a_search_nodes,
        metrics.player_a_search_elapsed_ms,
        metrics.player_a_depth_sum,
        metrics.player_a_searches,
        metrics.player_a_neural_inference_calls,
        metrics.player_a_neural_inference_time_ns,
        metrics.player_b_search_nodes,
        metrics.player_b_search_elapsed_ms,
        metrics.player_b_depth_sum,
        metrics.player_b_searches,
        metrics.player_b_neural_inference_calls,
        metrics.player_b_neural_inference_time_ns,
        metrics.illegal_moves,
        metrics.games,
    ]
}

#[expect(
    clippy::too_many_lines,
    reason = "all twelve report counters are intentionally checked field by field"
)]
fn validate_phase2_metric_counter_sums(report: &Phase2ArenaReport) -> Result<(), String> {
    let metrics = &report.metrics;
    let expected = [
        (
            metrics.player_a_search_nodes,
            report
                .games
                .iter()
                .map(|game| game.player_a_search_nodes)
                .collect::<Vec<_>>(),
        ),
        (
            metrics.player_a_search_elapsed_ms,
            report
                .games
                .iter()
                .map(|game| game.player_a_search_elapsed_ms)
                .collect(),
        ),
        (
            metrics.player_a_depth_sum,
            report
                .games
                .iter()
                .map(|game| game.player_a_depth_sum)
                .collect(),
        ),
        (
            metrics.player_a_searches,
            report
                .games
                .iter()
                .map(|game| game.player_a_searches)
                .collect(),
        ),
        (
            metrics.player_a_neural_inference_calls,
            report
                .games
                .iter()
                .map(|game| game.player_a_neural_inference_calls)
                .collect(),
        ),
        (
            metrics.player_a_neural_inference_time_ns,
            report
                .games
                .iter()
                .map(|game| game.player_a_neural_inference_time_ns)
                .collect(),
        ),
        (
            metrics.player_b_search_nodes,
            report
                .games
                .iter()
                .map(|game| game.player_b_search_nodes)
                .collect(),
        ),
        (
            metrics.player_b_search_elapsed_ms,
            report
                .games
                .iter()
                .map(|game| game.player_b_search_elapsed_ms)
                .collect(),
        ),
        (
            metrics.player_b_depth_sum,
            report
                .games
                .iter()
                .map(|game| game.player_b_depth_sum)
                .collect(),
        ),
        (
            metrics.player_b_searches,
            report
                .games
                .iter()
                .map(|game| game.player_b_searches)
                .collect(),
        ),
        (
            metrics.player_b_neural_inference_calls,
            report
                .games
                .iter()
                .map(|game| game.player_b_neural_inference_calls)
                .collect(),
        ),
        (
            metrics.player_b_neural_inference_time_ns,
            report
                .games
                .iter()
                .map(|game| game.player_b_neural_inference_time_ns)
                .collect(),
        ),
    ];
    for (observed, values) in expected {
        if observed != checked_json_sum(&values)? {
            return Err("paired arena aggregate counter differs from its games".to_owned());
        }
    }
    if metrics.neural_inference_calls
        != checked_json_sum(&[
            metrics.player_a_neural_inference_calls,
            metrics.player_b_neural_inference_calls,
        ])?
        || metrics.neural_inference_time_ns
            != checked_json_sum(&[
                metrics.player_a_neural_inference_time_ns,
                metrics.player_b_neural_inference_time_ns,
            ])?
    {
        return Err("paired arena aggregate inference counters disagree".to_owned());
    }
    Ok(())
}

fn validate_phase2_metric_outcomes(
    report: &Phase2ArenaReport,
    player_a: &Phase2ArenaPlayer,
    player_b: &Phase2ArenaPlayer,
) -> Result<(), String> {
    let mut finished = 0_u64;
    let mut draws = 0_u64;
    let mut first_player_wins = 0_u64;
    let mut second_player_wins = 0_u64;
    let mut search_wins = 0_u64;
    for (index, game) in report.games.iter().enumerate() {
        if game.result != "max_plies" {
            finished += 1;
        }
        if game.result == "draw" {
            draws += 1;
        }
        let winner = match (index, game.result.as_str()) {
            (0, "black_win") | (1, "white_win") => Some(true),
            (0, "white_win") | (1, "black_win") => Some(false),
            _ => None,
        };
        match winner {
            Some(true) => {
                first_player_wins += 1;
                search_wins += u64::from(player_a.evaluator_kind != "random");
            }
            Some(false) => {
                second_player_wins += 1;
                search_wins += u64::from(player_b.evaluator_kind != "random");
            }
            None => {}
        }
    }
    let metrics = &report.metrics;
    if metrics.finished_games != finished
        || metrics.player_a_wins != first_player_wins
        || metrics.player_b_wins != second_player_wins
        || metrics.search_wins != search_wins
        || metrics.draws != draws
    {
        return Err("paired arena aggregate outcomes disagree with its games".to_owned());
    }
    Ok(())
}

fn checked_json_sum(values: &[u64]) -> Result<u64, String> {
    let total = values.iter().try_fold(0_u64, |total, value| {
        total
            .checked_add(*value)
            .ok_or_else(|| "arena report counter sum overflow".to_owned())
    })?;
    if total > MAX_JSON_SAFE_INTEGER {
        return Err("arena report counter sum exceeds the JSON-safe integer limit".to_owned());
    }
    Ok(total)
}

fn phase2_decimal_ratio(numerator: u64, denominator: u64, multiplier: u64) -> Result<f64, String> {
    if denominator == 0 {
        return Ok(0.0);
    }
    let scaled = u128::from(numerator)
        .checked_mul(u128::from(multiplier))
        .and_then(|value| value.checked_mul(1_000_000))
        .ok_or_else(|| "arena report decimal ratio overflow".to_owned())?
        / u128::from(denominator);
    format!("{}.{:06}", scaled / 1_000_000, scaled % 1_000_000)
        .parse::<f64>()
        .map_err(|_| "arena report decimal ratio is not representable".to_owned())
}

fn utc_timestamp_key(value: &str) -> Result<(u64, u32), String> {
    let bytes = value.as_bytes();
    let digits = |range: std::ops::Range<usize>| -> Result<u64, String> {
        bytes[range]
            .iter()
            .try_fold(0_u64, |number, byte| {
                byte.is_ascii_digit()
                    .then(|| number * 10 + u64::from(*byte - b'0'))
                    .ok_or_else(|| "invalid UTC timestamp digit".to_owned())
            })
    };
    let whole = digits(0..4)? * 10_000_000_000
        + digits(5..7)? * 100_000_000
        + digits(8..10)? * 1_000_000
        + digits(11..13)? * 10_000
        + digits(14..16)? * 100
        + digits(17..19)?;
    let fraction = &value[19..value.len() - 1];
    let micros = if let Some(raw) = fraction.strip_prefix('.') {
        let parsed = raw
            .parse::<u32>()
            .map_err(|_| "invalid UTC timestamp fraction".to_owned())?;
        parsed * 10_u32.pow(u32::try_from(6 - raw.len()).expect("fraction is bounded"))
    } else {
        0
    };
    Ok((whole, micros))
}

fn latest_paired_attempts(
    attempts: &[PairedArenaAttempt],
) -> BTreeMap<&str, &PairedArenaAttempt> {
    let mut latest: BTreeMap<&str, &PairedArenaAttempt> = BTreeMap::new();
    for attempt in attempts {
        match latest.get(attempt.job_id.as_str()) {
            Some(previous) if previous.attempt >= attempt.attempt => {}
            _ => {
                latest.insert(attempt.job_id.as_str(), attempt);
            }
        }
    }
    latest
}

fn validate_json_self_hash(
    value: &serde_json::Value,
    hash_field: &str,
    expected: &str,
) -> Result<(), String> {
    let mut unsigned = value.clone();
    unsigned
        .as_object_mut()
        .ok_or_else(|| "self-hashed JSON artifact must be an object".to_owned())?
        .remove(hash_field)
        .ok_or_else(|| format!("self-hashed JSON artifact lacks {hash_field}"))?;
    if python_canonical_sha256(&unsigned)? != expected {
        return Err(format!(
            "{hash_field} does not match canonical artifact bytes"
        ));
    }
    Ok(())
}

struct UniqueJsonValue(serde_json::Value);

struct UniqueJsonBudget {
    remaining_nodes: usize,
    max_nodes: usize,
    max_depth: usize,
}

struct UniqueJsonSeed<'a> {
    budget: &'a mut UniqueJsonBudget,
    depth: usize,
}

impl<'de> serde::de::DeserializeSeed<'de> for UniqueJsonSeed<'_> {
    type Value = UniqueJsonValue;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        if self.depth > self.budget.max_depth {
            return Err(serde::de::Error::custom(format!(
                "JSON nesting exceeds {} levels",
                self.budget.max_depth
            )));
        }
        self.budget.remaining_nodes = self
            .budget
            .remaining_nodes
            .checked_sub(1)
            .ok_or_else(|| {
                serde::de::Error::custom(format!(
                    "JSON value exceeds {} nodes",
                    self.budget.max_nodes
                ))
            })?;

        deserializer.deserialize_any(UniqueJsonVisitor {
            budget: self.budget,
            depth: self.depth,
        })
    }
}

struct UniqueJsonVisitor<'a> {
    budget: &'a mut UniqueJsonBudget,
    depth: usize,
}

impl<'de> serde::de::Visitor<'de> for UniqueJsonVisitor<'_> {
    type Value = UniqueJsonValue;

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("a bounded JSON value without duplicate object keys")
    }

    fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E> {
        Ok(UniqueJsonValue(serde_json::Value::Bool(value)))
    }

    fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E> {
        Ok(UniqueJsonValue(serde_json::Value::Number(value.into())))
    }

    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E> {
        Ok(UniqueJsonValue(serde_json::Value::Number(value.into())))
    }

    fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        serde_json::Number::from_f64(value)
            .map(serde_json::Value::Number)
            .map(UniqueJsonValue)
            .ok_or_else(|| E::custom("non-finite JSON number"))
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        self.visit_string(value.to_owned())
    }

    fn visit_string<E>(self, value: String) -> Result<Self::Value, E> {
        Ok(UniqueJsonValue(serde_json::Value::String(value)))
    }

    fn visit_none<E>(self) -> Result<Self::Value, E> {
        Ok(UniqueJsonValue(serde_json::Value::Null))
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E> {
        Ok(UniqueJsonValue(serde_json::Value::Null))
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: serde::de::SeqAccess<'de>,
    {
        let mut values = Vec::new();
        while let Some(value) = sequence.next_element_seed(UniqueJsonSeed {
            budget: self.budget,
            depth: self.depth + 1,
        })? {
            values.push(value.0);
        }
        Ok(UniqueJsonValue(serde_json::Value::Array(values)))
    }

    fn visit_map<A>(self, mut object: A) -> Result<Self::Value, A::Error>
    where
        A: serde::de::MapAccess<'de>,
    {
        let mut values = serde_json::Map::new();
        while let Some(key) = object.next_key::<String>()? {
            if values.contains_key(&key) {
                return Err(serde::de::Error::custom(format!(
                    "duplicate JSON key: {key}"
                )));
            }
            let value = object.next_value_seed(UniqueJsonSeed {
                budget: self.budget,
                depth: self.depth + 1,
            })?;
            values.insert(key, value.0);
        }
        Ok(UniqueJsonValue(serde_json::Value::Object(values)))
    }
}

fn parse_unique_json(bytes: &[u8], context: &str) -> Result<serde_json::Value, String> {
    parse_unique_json_with_limits(
        bytes,
        context,
        MAX_REGISTRY_JSON_DEPTH,
        MAX_REGISTRY_JSON_NODES,
    )
}

fn parse_unique_json_with_limits(
    bytes: &[u8],
    context: &str,
    max_depth: usize,
    max_nodes: usize,
) -> Result<serde_json::Value, String> {
    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    let mut budget = UniqueJsonBudget {
        remaining_nodes: max_nodes,
        max_nodes,
        max_depth,
    };
    let value = serde::de::DeserializeSeed::deserialize(
        UniqueJsonSeed {
            budget: &mut budget,
            depth: 0,
        },
        &mut deserializer,
    )
        .map_err(|error| format!("invalid {context}: {error}"))?;
    deserializer
        .end()
        .map_err(|error| format!("invalid {context}: {error}"))?;
    Ok(value.0)
}

fn deserialize_closed_json<T: DeserializeOwned>(
    value: &serde_json::Value,
    context: &str,
) -> Result<T, String> {
    serde_json::from_value(value.clone())
        .map_err(|error| format!("invalid closed {context}: {error}"))
}

fn deserialize_required_option<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::<T>::deserialize(deserializer)
}

fn python_canonical_sha256(value: &serde_json::Value) -> Result<String, String> {
    let mut canonical = Vec::new();
    write_python_canonical_json(value, &mut canonical)?;
    Ok(sha256_bytes(&canonical))
}

fn write_python_canonical_json(
    value: &serde_json::Value,
    output: &mut Vec<u8>,
) -> Result<(), String> {
    match value {
        serde_json::Value::Null => output.extend_from_slice(b"null"),
        serde_json::Value::Bool(true) => output.extend_from_slice(b"true"),
        serde_json::Value::Bool(false) => output.extend_from_slice(b"false"),
        serde_json::Value::Number(number) => {
            let rendered = if let Some(value) = number.as_i64() {
                value.to_string()
            } else if let Some(value) = number.as_u64() {
                value.to_string()
            } else {
                let value = number
                    .as_f64()
                    .filter(|value| value.is_finite())
                    .ok_or_else(|| "canonical JSON contains a non-finite number".to_owned())?;
                python_float_repr(value)
            };
            output.extend_from_slice(rendered.as_bytes());
        }
        serde_json::Value::String(value) => {
            let encoded = serde_json::to_string(value)
                .map_err(|error| format!("cannot canonicalize JSON string: {error}"))?;
            output.extend_from_slice(encoded.as_bytes());
        }
        serde_json::Value::Array(values) => {
            output.push(b'[');
            for (index, item) in values.iter().enumerate() {
                if index != 0 {
                    output.push(b',');
                }
                write_python_canonical_json(item, output)?;
            }
            output.push(b']');
        }
        serde_json::Value::Object(values) => {
            output.push(b'{');
            let mut keys = values.keys().collect::<Vec<_>>();
            keys.sort_unstable();
            for (index, key) in keys.into_iter().enumerate() {
                if index != 0 {
                    output.push(b',');
                }
                let encoded = serde_json::to_string(key)
                    .map_err(|error| format!("cannot canonicalize JSON key: {error}"))?;
                output.extend_from_slice(encoded.as_bytes());
                output.push(b':');
                write_python_canonical_json(&values[key], output)?;
            }
            output.push(b'}');
        }
    }
    Ok(())
}

fn python_float_repr(value: f64) -> String {
    // `serde_json`'s Schubfach formatter chooses the same shortest round-tripping
    // binary64 significand as CPython. The two serializers differ only in when they
    // select exponent notation and in exponent padding, so normalize those surface
    // rules explicitly instead of relying on Rust's `Debug` formatter (which makes a
    // different last-digit choice for some equally short representations).
    let rendered = serde_json::to_string(&value)
        .expect("a previously validated finite binary64 value is JSON serializable");
    let (sign, magnitude) = rendered
        .strip_prefix('-')
        .map_or(("", rendered.as_str()), |magnitude| ("-", magnitude));
    if magnitude == "0.0" {
        return rendered;
    }

    let (digits, decimal_exponent) = if let Some((mantissa, exponent)) = magnitude.split_once('e') {
        let decimal_position = mantissa.find('.').unwrap_or(mantissa.len());
        let digits = mantissa.replace('.', "");
        let exponent = exponent
            .parse::<i32>()
            .expect("serde_json float exponent is decimal");
        (
            digits,
            exponent + i32::try_from(decimal_position).expect("float spelling is short") - 1,
        )
    } else {
        let decimal_position = magnitude.find('.').unwrap_or(magnitude.len());
        let all_digits = magnitude.replace('.', "");
        let first_significant = all_digits
            .bytes()
            .position(|byte| byte != b'0')
            .expect("nonzero float spelling contains a significant digit");
        let decimal_exponent = i32::try_from(decimal_position).expect("float spelling is short")
            - i32::try_from(first_significant).expect("float spelling is short")
            - 1;
        (all_digits[first_significant..].to_owned(), decimal_exponent)
    };

    let magnitude = if (-4..16).contains(&decimal_exponent) {
        if decimal_exponent >= 0 {
            let decimal_position = usize::try_from(decimal_exponent + 1)
                .expect("non-negative decimal exponent fits usize");
            if digits.len() <= decimal_position {
                format!(
                    "{}{}.0",
                    digits,
                    "0".repeat(decimal_position - digits.len())
                )
            } else {
                format!(
                    "{}.{}",
                    &digits[..decimal_position],
                    &digits[decimal_position..]
                )
            }
        } else {
            format!(
                "0.{}{}",
                "0".repeat(usize::try_from(-decimal_exponent - 1).expect("bounded exponent")),
                digits
            )
        }
    } else if digits.len() == 1 {
        format!("{digits}e{decimal_exponent:+03}")
    } else {
        format!(
            "{}.{}e{decimal_exponent:+03}",
            &digits[..1],
            &digits[1..]
        )
    };
    format!("{sign}{magnitude}")
}

fn parse_generation_policy(bytes: &[u8]) -> Result<GenerationPolicy, String> {
    let source = std::str::from_utf8(bytes)
        .map_err(|error| format!("promotion policy is not valid UTF-8: {error}"))?;
    let policy: GenerationPolicy =
        toml::from_str(source).map_err(|error| format!("invalid closed promotion policy: {error}"))?;
    if policy.schema_version != 1 {
        return Err("promotion policy schema_version must be 1".to_owned());
    }
    if policy.evidence.minimum_games != 40
        || policy.evidence.minimum_decisive_games != 12
        || policy.evidence.minimum_group_games != 10
        || policy.evidence.max_illegal != 0
        || policy.evidence.max_crashes != 0
    {
        return Err("promotion policy differs from the fixed Phase 6 evidence contract".to_owned());
    }
    let thresholds = &policy.thresholds;
    for (value, minimum, maximum, name) in [
        (
            thresholds.promote_score_rate.as_f64(),
            0.5,
            1.0,
            "promote_score_rate",
        ),
        (
            thresholds.promote_wilson_lower.as_f64(),
            0.5,
            1.0,
            "promote_wilson_lower",
        ),
        (
            thresholds.reject_score_rate.as_f64(),
            0.0,
            0.5,
            "reject_score_rate",
        ),
        (
            thresholds.reject_wilson_upper.as_f64(),
            0.0,
            0.5,
            "reject_wilson_upper",
        ),
        (
            thresholds.minimum_group_score_rate.as_f64(),
            0.0,
            1.0,
            "minimum_group_score_rate",
        ),
        (
            thresholds.maximum_side_score_gap.as_f64(),
            0.0,
            1.0,
            "maximum_side_score_gap",
        ),
    ] {
        if !value.is_finite() || !(minimum..=maximum).contains(&value) {
            return Err(format!("promotion policy {name} is outside {minimum}..={maximum}"));
        }
    }
    if thresholds.reject_score_rate.as_f64() >= thresholds.promote_score_rate.as_f64() {
        return Err("promotion reject_score_rate must be below promote_score_rate".to_owned());
    }
    for (value, name) in [
        (
            policy.performance.maximum_inference_slowdown.as_f64(),
            "maximum_inference_slowdown",
        ),
        (
            policy.performance.maximum_search_slowdown.as_f64(),
            "maximum_search_slowdown",
        ),
    ] {
        if !value.is_finite() || !(1.0..=100.0).contains(&value) {
            return Err(format!("promotion policy {name} is outside 1..=100"));
        }
    }
    Ok(policy)
}

fn validate_arena_results(results: &ArenaResults) -> Result<(), String> {
    if results.schema != "phase6_paired_arena_results/v1" {
        return Err("referenced arena results schema mismatch".to_owned());
    }
    for (identifier, field) in [
        (&results.generation_id, "arena results generationId"),
        (&results.champion_model_id, "arena results championModelId"),
        (&results.challenger_model_id, "arena results challengerModelId"),
    ] {
        validate_identifier(identifier, field)?;
    }
    if results.champion_model_id == results.challenger_model_id {
        return Err("arena results champion and challenger must differ".to_owned());
    }
    validate_artifact_ref(&results.plan)?;
    validate_artifact_ref(&results.execution)?;
    if !(2..=10_000).contains(&results.games.len()) {
        return Err("arena results games must contain 2..=10000 rows".to_owned());
    }
    let mut game_ids = std::collections::BTreeSet::new();
    let mut pairs = BTreeMap::<&str, Vec<&ArenaResultGame>>::new();
    for game in &results.games {
        for (identifier, field) in [
            (&game.game_id, "arena gameId"),
            (&game.pair_id, "arena pairId"),
            (&game.start_position_id, "arena startPositionId"),
            (&game.black_model_id, "arena blackModelId"),
            (&game.white_model_id, "arena whiteModelId"),
        ] {
            validate_identifier(identifier, field)?;
        }
        if !game_ids.insert(game.game_id.as_str()) {
            return Err("arena results contain a duplicate gameId".to_owned());
        }
        if !matches!(game.start_group.as_str(), "initial" | "start_set") {
            return Err("arena startGroup must be initial or start_set".to_owned());
        }
        if !matches!(
            game.result.as_str(),
            "black_win" | "white_win" | "draw" | "max_plies"
        ) {
            return Err("arena game result is invalid".to_owned());
        }
        if game.plies > 10_000 || game.illegal_moves > 10_000 || game.crashes > 10_000 {
            return Err("arena game counters exceed 10000".to_owned());
        }
        if game.black_model_id == game.white_model_id
            || ![game.black_model_id.as_str(), game.white_model_id.as_str()]
                .contains(&results.champion_model_id.as_str())
            || ![game.black_model_id.as_str(), game.white_model_id.as_str()]
                .contains(&results.challenger_model_id.as_str())
        {
            return Err("arena game does not contain exactly champion and challenger".to_owned());
        }
        validate_arena_game_metrics(&game.metrics)?;
        pairs.entry(&game.pair_id).or_default().push(game);
    }
    for (pair_id, pair) in pairs {
        if pair.len() != 2 {
            return Err(format!("arena pair {pair_id} does not have exactly two games"));
        }
        let first = pair[0];
        let second = pair[1];
        if first.start_group != second.start_group
            || first.start_position_id != second.start_position_id
        {
            return Err(format!("arena pair {pair_id} does not share one start position"));
        }
        if first.black_model_id != second.white_model_id
            || first.white_model_id != second.black_model_id
        {
            return Err(format!("arena pair {pair_id} is not color-swapped"));
        }
    }
    Ok(())
}

fn validate_arena_game_metrics(metrics: &ArenaGameMetrics) -> Result<(), String> {
    for value in [
        metrics.champion_inference_calls,
        metrics.champion_inference_time_ns,
        metrics.challenger_inference_calls,
        metrics.challenger_inference_time_ns,
        metrics.champion_search_nodes,
        metrics.champion_search_elapsed_ms,
        metrics.challenger_search_nodes,
        metrics.challenger_search_elapsed_ms,
        metrics.champion_search_depth_sum,
        metrics.champion_searches,
        metrics.challenger_search_depth_sum,
        metrics.challenger_searches,
    ] {
        if value.is_some_and(|value| value > 1_000_000_000_000_000_000) {
            return Err("arena game metric exceeds 10^18".to_owned());
        }
    }
    Ok(())
}

fn analyze_arena_results(
    results: &ArenaResults,
    results_ref: RegistryArtifact,
) -> Result<ArenaAnalysisEnvelope, String> {
    validate_arena_results(results)?;
    let overall = summarize_arena_games(&results.games, &results.challenger_model_id)?;
    let initial_games = results
        .games
        .iter()
        .filter(|game| game.start_group == "initial")
        .cloned()
        .collect::<Vec<_>>();
    let start_set_games = results
        .games
        .iter()
        .filter(|game| game.start_group == "start_set")
        .cloned()
        .collect::<Vec<_>>();
    let black_games = results
        .games
        .iter()
        .filter(|game| game.black_model_id == results.challenger_model_id)
        .cloned()
        .collect::<Vec<_>>();
    let white_games = results
        .games
        .iter()
        .filter(|game| game.white_model_id == results.challenger_model_id)
        .cloned()
        .collect::<Vec<_>>();
    let initial = summarize_arena_games(&initial_games, &results.challenger_model_id)?;
    let start_set = summarize_arena_games(&start_set_games, &results.challenger_model_id)?;
    let black = summarize_arena_games(&black_games, &results.challenger_model_id)?;
    let white = summarize_arena_games(&white_games, &results.challenger_model_id)?;
    let side_score_gap = if black.games != 0 && white.games != 0 {
        Some(round_twelve((black.score_rate - white.score_rate).abs())?)
    } else {
        None
    };
    let metrics = aggregate_arena_metrics(&results.games)?;
    let mut analysis = ArenaAnalysisEnvelope {
        schema: "phase6_paired_arena_analysis/v1".to_owned(),
        generation_id: results.generation_id.clone(),
        results: results_ref,
        plan: results.plan.clone(),
        execution: results.execution.clone(),
        champion_model_id: results.champion_model_id.clone(),
        challenger_model_id: results.challenger_model_id.clone(),
        method: ArenaAnalysisMethod {
            pairing: "same-start-color-swapped".to_owned(),
            score: "win=1,draw-or-max-plies=0.5,loss=0".to_owned(),
            wilson95: "fractional-score Wilson approximation, z=1.959963984540054".to_owned(),
            elo: "400*log10(score/(1-score)); null at boundary scores".to_owned(),
        },
        overall,
        by_start_group: ArenaStartGroups { initial, start_set },
        by_challenger_side: ArenaChallengerSides { black, white },
        side_score_gap,
        metrics,
        analysis_sha256: String::new(),
    };
    analysis.analysis_sha256 = self_hash_serializable(&analysis, "analysisSha256")?;
    Ok(analysis)
}

fn summarize_arena_games(
    games: &[ArenaResultGame],
    challenger: &str,
) -> Result<ArenaSummary, String> {
    let mut wins = 0_u64;
    let mut losses = 0_u64;
    let mut draws = 0_u64;
    let mut max_plies = 0_u64;
    let mut illegal_moves = 0_u64;
    let mut crashes = 0_u64;
    let mut total_plies = 0_u64;
    for game in games {
        match game.result.as_str() {
            "draw" => draws += 1,
            "max_plies" => {
                draws += 1;
                max_plies += 1;
            }
            "black_win" if game.black_model_id == challenger => wins += 1,
            "white_win" if game.white_model_id == challenger => wins += 1,
            "black_win" | "white_win" => losses += 1,
            _ => return Err("arena game result is invalid".to_owned()),
        }
        illegal_moves = illegal_moves
            .checked_add(game.illegal_moves)
            .ok_or_else(|| "arena illegal-move total overflow".to_owned())?;
        crashes = crashes
            .checked_add(game.crashes)
            .ok_or_else(|| "arena crash total overflow".to_owned())?;
        total_plies = total_plies
            .checked_add(game.plies)
            .ok_or_else(|| "arena ply total overflow".to_owned())?;
    }
    let count = u64::try_from(games.len()).map_err(|_| "arena game count overflow".to_owned())?;
    let decisive = wins + losses;
    let points = u64_to_python_float(wins) + 0.5 * u64_to_python_float(draws);
    let score_rate = if count == 0 {
        0.0
    } else {
        points / u64_to_python_float(count)
    };
    let decisive_win_rate = if decisive == 0 {
        0.0
    } else {
        u64_to_python_float(wins) / u64_to_python_float(decisive)
    };
    let draw_rate = if count == 0 {
        0.0
    } else {
        u64_to_python_float(draws) / u64_to_python_float(count)
    };
    let (lower, upper) = wilson_interval(points, count);
    let approximate_elo = if score_rate > 0.0 && score_rate < 1.0 {
        Some(round_twelve(
            400.0 * (score_rate / (1.0 - score_rate)).log10(),
        )?)
    } else {
        None
    };
    let average_plies = if count == 0 {
        0.0
    } else {
        u64_to_python_float(total_plies) / u64_to_python_float(count)
    };
    if [count, wins, losses, draws, max_plies, decisive, illegal_moves, crashes]
        .into_iter()
        .any(|value| value > 10_000)
    {
        return Err("arena analysis summary counters exceed 10000".to_owned());
    }
    Ok(ArenaSummary {
        games: count,
        wins,
        losses,
        draws,
        max_plies,
        decisive_games: decisive,
        score_rate: round_twelve(score_rate)?,
        decisive_win_rate: round_twelve(decisive_win_rate)?,
        draw_rate: round_twelve(draw_rate)?,
        score_wilson95: PromotionWilsonInterval {
            lower: round_twelve(lower)?,
            upper: round_twelve(upper)?,
        },
        approximate_elo,
        average_plies: round_twelve(average_plies)?,
        illegal_moves,
        crashes,
    })
}

fn wilson_interval(successes: f64, count: u64) -> (f64, f64) {
    if count == 0 {
        return (0.0, 1.0);
    }
    let count = u64_to_python_float(count);
    let z = 1.959_963_984_540_054_f64;
    let proportion = successes / count;
    let denominator = 1.0 + z * z / count;
    let center = (proportion + z * z / (2.0 * count)) / denominator;
    let margin = z
        * (proportion * (1.0 - proportion) / count + z * z / (4.0 * count * count)).sqrt()
        / denominator;
    ((center - margin).max(0.0), (center + margin).min(1.0))
}

fn round_twelve(value: f64) -> Result<f64, String> {
    if !value.is_finite() {
        return Err("arena analysis derived a non-finite number".to_owned());
    }
    format!("{value:.12}")
        .parse::<f64>()
        .map_err(|error| format!("cannot round arena analysis number: {error}"))
}

fn aggregate_arena_metrics(games: &[ArenaResultGame]) -> Result<ArenaAggregateMetrics, String> {
    let raw_totals = ArenaGameMetrics {
        champion_inference_calls: sum_optional_metric(games, |metrics| {
            metrics.champion_inference_calls
        })?,
        champion_inference_time_ns: sum_optional_metric(games, |metrics| {
            metrics.champion_inference_time_ns
        })?,
        challenger_inference_calls: sum_optional_metric(games, |metrics| {
            metrics.challenger_inference_calls
        })?,
        challenger_inference_time_ns: sum_optional_metric(games, |metrics| {
            metrics.challenger_inference_time_ns
        })?,
        champion_search_nodes: sum_optional_metric(games, |metrics| metrics.champion_search_nodes)?,
        champion_search_elapsed_ms: sum_optional_metric(games, |metrics| {
            metrics.champion_search_elapsed_ms
        })?,
        challenger_search_nodes: sum_optional_metric(games, |metrics| {
            metrics.challenger_search_nodes
        })?,
        challenger_search_elapsed_ms: sum_optional_metric(games, |metrics| {
            metrics.challenger_search_elapsed_ms
        })?,
        champion_search_depth_sum: sum_optional_metric(games, |metrics| {
            metrics.champion_search_depth_sum
        })?,
        champion_searches: sum_optional_metric(games, |metrics| metrics.champion_searches)?,
        challenger_search_depth_sum: sum_optional_metric(games, |metrics| {
            metrics.challenger_search_depth_sum
        })?,
        challenger_searches: sum_optional_metric(games, |metrics| metrics.challenger_searches)?,
    };
    let champion_inference = throughput_ns(
        raw_totals.champion_inference_calls,
        raw_totals.champion_inference_time_ns,
    );
    let challenger_inference = throughput_ns(
        raw_totals.challenger_inference_calls,
        raw_totals.challenger_inference_time_ns,
    );
    let champion_search = throughput_ms(
        raw_totals.champion_search_nodes,
        raw_totals.champion_search_elapsed_ms,
    );
    let challenger_search = throughput_ms(
        raw_totals.challenger_search_nodes,
        raw_totals.challenger_search_elapsed_ms,
    );
    Ok(ArenaAggregateMetrics {
        champion_inference_calls_per_second: round_optional(champion_inference)?,
        challenger_inference_calls_per_second: round_optional(challenger_inference)?,
        inference_slowdown_ratio: round_optional(slowdown(
            champion_inference,
            challenger_inference,
        ))?,
        champion_search_nodes_per_second: round_optional(champion_search)?,
        challenger_search_nodes_per_second: round_optional(challenger_search)?,
        search_slowdown_ratio: round_optional(slowdown(champion_search, challenger_search))?,
        champion_average_search_depth: round_optional(ratio(
            raw_totals.champion_search_depth_sum,
            raw_totals.champion_searches,
        ))?,
        challenger_average_search_depth: round_optional(ratio(
            raw_totals.challenger_search_depth_sum,
            raw_totals.challenger_searches,
        ))?,
        raw_totals,
    })
}

fn sum_optional_metric(
    games: &[ArenaResultGame],
    value: impl Fn(&ArenaGameMetrics) -> Option<u64>,
) -> Result<Option<u64>, String> {
    let mut total = 0_u64;
    for game in games {
        let Some(value) = value(&game.metrics) else {
            return Ok(None);
        };
        total = total
            .checked_add(value)
            .ok_or_else(|| "arena metric total overflow".to_owned())?;
        if total > 1_000_000_000_000_000_000 {
            return Err("arena metric total exceeds 10^18".to_owned());
        }
    }
    Ok(Some(total))
}

fn throughput_ns(numerator: Option<u64>, nanoseconds: Option<u64>) -> Option<f64> {
    match (numerator, nanoseconds) {
        (Some(numerator), Some(nanoseconds)) if nanoseconds != 0 => {
            Some(
                u64_to_python_float(numerator) * 1_000_000_000.0
                    / u64_to_python_float(nanoseconds),
            )
        }
        _ => None,
    }
}

fn throughput_ms(numerator: Option<u64>, milliseconds: Option<u64>) -> Option<f64> {
    match (numerator, milliseconds) {
        (Some(numerator), Some(milliseconds)) if milliseconds != 0 => {
            Some(u64_to_python_float(numerator) * 1_000.0 / u64_to_python_float(milliseconds))
        }
        _ => None,
    }
}

fn ratio(numerator: Option<u64>, denominator: Option<u64>) -> Option<f64> {
    match (numerator, denominator) {
        (Some(numerator), Some(denominator)) if denominator != 0 => {
            Some(u64_to_python_float(numerator) / u64_to_python_float(denominator))
        }
        _ => None,
    }
}

fn slowdown(baseline: Option<f64>, challenger: Option<f64>) -> Option<f64> {
    match (baseline, challenger) {
        (Some(baseline), Some(challenger)) if challenger > 0.0 => Some(baseline / challenger),
        _ => None,
    }
}

fn round_optional(value: Option<f64>) -> Result<Option<f64>, String> {
    value.map(round_twelve).transpose()
}

#[expect(
    clippy::cast_precision_loss,
    reason = "Python's arena contract converts bounded integer metrics to binary64 at these exact operations"
)]
fn u64_to_python_float(value: u64) -> f64 {
    value as f64
}

fn self_hash_serializable(value: &impl Serialize, field: &str) -> Result<String, String> {
    let mut value = serde_json::to_value(value)
        .map_err(|error| format!("cannot encode self-hashed artifact: {error}"))?;
    value
        .as_object_mut()
        .ok_or_else(|| "self-hashed artifact must be an object".to_owned())?
        .remove(field);
    python_canonical_sha256(&value)
}

fn derive_promotion_decision(
    analysis: &ArenaAnalysisEnvelope,
    observed: &PromotionDecision,
    policy: &GenerationPolicy,
) -> Result<PromotionDecision, String> {
    let overall = &analysis.overall;
    let mut reasons = Vec::new();
    let mut decision = "inconclusive";
    let mut weak_evidence = false;
    if overall.illegal_moves > policy.evidence.max_illegal {
        decision = "rejected";
        reasons.push("illegal_move_limit_exceeded".to_owned());
    }
    if overall.crashes > policy.evidence.max_crashes {
        decision = "rejected";
        reasons.push("crash_limit_exceeded".to_owned());
    }
    if decision != "rejected" && overall.games < policy.evidence.minimum_games {
        weak_evidence = true;
        reasons.push("insufficient_games".to_owned());
    }
    if decision != "rejected"
        && overall.decisive_games < policy.evidence.minimum_decisive_games
    {
        weak_evidence = true;
        reasons.push("insufficient_decisive_games".to_owned());
    }
    for (name, group) in [
        ("initial", &analysis.by_start_group.initial),
        ("start_set", &analysis.by_start_group.start_set),
    ] {
        if group.games < policy.evidence.minimum_group_games {
            weak_evidence = true;
            reasons.push(format!("insufficient_{name}_games"));
        }
    }
    let (performance_ok, performance_reasons, performance_missing) =
        check_promotion_performance(&analysis.metrics, &policy.performance);
    reasons.extend(performance_reasons);
    weak_evidence |= performance_missing;
    if decision != "rejected" && !performance_ok && !performance_missing {
        decision = "rejected";
    }
    if decision != "rejected" && !weak_evidence {
        if overall.score_rate <= policy.thresholds.reject_score_rate.as_f64()
            && overall.score_wilson95.upper <= policy.thresholds.reject_wilson_upper.as_f64()
        {
            decision = "rejected";
            reasons.push("statistically_supported_regression".to_owned());
        } else if overall.score_rate >= policy.thresholds.promote_score_rate.as_f64()
            && overall.score_wilson95.lower >= policy.thresholds.promote_wilson_lower.as_f64()
            && analysis.by_start_group.initial.score_rate
                >= policy.thresholds.minimum_group_score_rate.as_f64()
            && analysis.by_start_group.start_set.score_rate
                >= policy.thresholds.minimum_group_score_rate.as_f64()
            && analysis.side_score_gap.is_some_and(|gap| {
                gap <= policy.thresholds.maximum_side_score_gap.as_f64()
            })
            && performance_ok
        {
            decision = "promoted";
            reasons.push("all_promotion_gates_passed".to_owned());
        } else {
            reasons.push("evidence_does_not_cross_a_decision_boundary".to_owned());
        }
    } else if decision == "inconclusive" && reasons.is_empty() {
        reasons.push("evidence_does_not_cross_a_decision_boundary".to_owned());
    }
    let mut expected = PromotionDecision {
        schema: "phase6_promotion_decision/v1".to_owned(),
        generation_id: analysis.generation_id.clone(),
        champion_model_id: analysis.champion_model_id.clone(),
        challenger_model_id: analysis.challenger_model_id.clone(),
        arena_analysis: observed.arena_analysis.clone(),
        policy: observed.policy.clone(),
        policy_sha256: observed.policy_sha256.clone(),
        decision: decision.to_owned(),
        weak_evidence,
        reasons,
        evidence: PromotionEvidence {
            games: overall.games,
            decisive_games: overall.decisive_games,
            score_rate: overall.score_rate,
            score_wilson95: overall.score_wilson95.clone(),
            initial_score_rate: analysis.by_start_group.initial.score_rate,
            start_set_score_rate: analysis.by_start_group.start_set.score_rate,
            side_score_gap: analysis.side_score_gap,
            illegal_moves: overall.illegal_moves,
            crashes: overall.crashes,
        },
        decided_at: observed.decided_at.clone(),
        decision_sha256: String::new(),
    };
    expected.decision_sha256 = self_hash_serializable(&expected, "decisionSha256")?;
    Ok(expected)
}

fn check_promotion_performance(
    metrics: &ArenaAggregateMetrics,
    policy: &PerformancePolicy,
) -> (bool, Vec<String>, bool) {
    let (Some(inference), Some(search)) = (
        metrics.inference_slowdown_ratio,
        metrics.search_slowdown_ratio,
    ) else {
        return if policy.require_metrics {
            (
                false,
                vec!["required_performance_metrics_missing".to_owned()],
                true,
            )
        } else {
            (true, Vec::new(), false)
        };
    };
    let mut reasons = Vec::new();
    if inference > policy.maximum_inference_slowdown.as_f64() {
        reasons.push("inference_slowdown_limit_exceeded".to_owned());
    }
    if search > policy.maximum_search_slowdown.as_f64() {
        reasons.push("search_slowdown_limit_exceeded".to_owned());
    }
    (reasons.is_empty(), reasons, false)
}

fn validate_registry_champion_transition(
    registry: &ModelRegistry,
    decisions: &BTreeMap<String, String>,
) -> Result<(), String> {
    let mut completed_outcomes = BTreeMap::<&str, &str>::new();
    for generation in &registry.generations {
        let Some(parent_generation_id) = generation.parent_generation_id.as_deref() else {
            completed_outcomes.insert(
                generation.generation_id.as_str(),
                &generation.champion_model_id,
            );
            continue;
        };
        let expected_incumbent = completed_outcomes
            .get(parent_generation_id)
            .ok_or_else(|| {
                "registry generation parent lacks a verified champion outcome".to_owned()
            })?;
        if generation.champion_model_id != **expected_incumbent {
            return Err(
                "registry generation incumbent disagrees with its parent's promotion decision"
                    .to_owned(),
            );
        }
        if generation.status == "complete" {
            let decision = decisions
                .get(generation.generation_id.as_str())
                .ok_or_else(|| {
                    "completed generation lacks a verified promotion decision".to_owned()
                })?;
            let outcome = if decision == "promoted" {
                generation
                    .challenger_model_id
                    .as_deref()
                    .ok_or_else(|| "promoted generation lacks a challenger".to_owned())?
            } else {
                generation.champion_model_id.as_str()
            };
            completed_outcomes.insert(generation.generation_id.as_str(), outcome);
        }
    }
    let latest = registry
        .generations
        .last()
        .ok_or_else(|| "model registry has no generation history".to_owned())?;
    let expected = if latest.status == "complete" {
        completed_outcomes
            .get(latest.generation_id.as_str())
            .copied()
            .ok_or_else(|| "latest generation lacks a verified champion outcome".to_owned())?
    } else {
        latest.champion_model_id.as_str()
    };
    if registry.champion_model_id.as_deref() != Some(expected) {
        return Err(
            "registry champion transition disagrees with its latest verified generation".to_owned(),
        );
    }
    Ok(())
}

fn verify_observed_artifact(
    expected: &RegistryArtifact,
    sha256: &str,
    size: u64,
) -> Result<(), String> {
    if expected.size != size {
        return Err(format!(
            "registry artifact size mismatch for {}",
            expected.path.display()
        ));
    }
    if expected.sha256 != sha256 {
        return Err(format!(
            "registry artifact SHA-256 mismatch for {}",
            expected.path.display()
        ));
    }
    Ok(())
}

fn repository_root_and_registry(registry_path: &Path) -> Result<RegistryStorage, String> {
    let absolute = if registry_path.is_absolute() {
        registry_path.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|error| format!("cannot inspect current directory: {error}"))?
            .join(registry_path)
    };
    let mut discovered = None;
    for ancestor in absolute.parent().into_iter().flat_map(Path::ancestors) {
        let Ok(root) = AnchoredDir::open_existing(ancestor) else {
            continue;
        };
        let marker = root
            .entry_kind(std::ffi::OsStr::new(".git"))
            .map_err(|error| format!("cannot inspect repository marker: {error}"))?;
        if matches!(marker, Some(EntryKind::File | EntryKind::Directory)) {
            if discovered.is_some() {
                return Err(
                    "model registry path is enclosed by multiple repository roots".to_owned(),
                );
            }
            let relative = absolute
                .strip_prefix(ancestor)
                .map_err(|_| "model registry is outside the repository root".to_owned())?
                .to_path_buf();
            validate_relative_artifact_path(&relative)?;
            root.open_relative_regular(&relative).map_err(|error| {
                format!(
                    "cannot open model registry {} without symlinks: {error}",
                    absolute.display()
                )
            })?;
            discovered = Some(RegistryStorage {
                root,
                root_path: ancestor.to_path_buf(),
                registry_relative: relative,
            });
            continue;
        }
        if marker.is_some() {
            return Err("repository .git marker is not a regular entry".to_owned());
        }
    }
    discovered.ok_or_else(|| "cannot locate repository root from the registry path".to_owned())
}

#[cfg(test)]
fn contained_artifact_path(root: &Path, relative: &Path) -> Result<PathBuf, String> {
    validate_relative_artifact_path(relative)?;
    let path = root.join(relative);
    AnchoredDir::open_existing(root)
        .and_then(|directory| directory.open_relative_regular(relative))
        .map_err(|error| {
            format!(
                "cannot resolve registry artifact {}: {error}",
                path.display()
            )
        })?;
    Ok(path)
}

fn validate_relative_artifact_path(path: &Path) -> Result<(), String> {
    let value = path
        .to_str()
        .ok_or_else(|| "registry artifact path must be UTF-8".to_owned())?;
    if value.is_empty() || value.chars().count() > 1_024 || value.contains(['\\', '\0']) {
        return Err("registry artifact path contains a forbidden character".to_owned());
    }
    let parts = path
        .components()
        .map(|component| match component {
            Component::Normal(part) => part
                .to_str()
                .map(str::to_owned)
                .ok_or_else(|| "registry artifact path must be UTF-8".to_owned()),
            _ => Err(
                "registry artifact path must be a normalized repository-relative path".to_owned(),
            ),
        })
        .collect::<Result<Vec<_>, _>>()?;
    if parts.is_empty() || parts.join("/") != value {
        return Err(
            "registry artifact path must be a normalized repository-relative path".to_owned(),
        );
    }
    Ok(())
}

fn validate_identifier(value: &str, field: &str) -> Result<(), String> {
    let mut bytes = value.bytes();
    let Some(first) = bytes.next() else {
        return Err(format!("{field} is not a safe bounded identifier"));
    };
    if value.len() > MAX_IDENTIFIER_BYTES
        || !first.is_ascii_alphanumeric()
        || !bytes.all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        return Err(format!("{field} is not a safe bounded identifier"));
    }
    Ok(())
}

fn validate_utc_timestamp(value: &str, field: &str) -> Result<(), String> {
    let bytes = value.as_bytes();
    let base_valid = bytes.len() >= 20
        && bytes[4] == b'-'
        && bytes[7] == b'-'
        && bytes[10] == b'T'
        && bytes[13] == b':'
        && bytes[16] == b':'
        && bytes.last() == Some(&b'Z');
    if !base_valid {
        return Err(format!("registry {field} is not an ISO-8601 UTC timestamp"));
    }
    let fractional = &bytes[19..bytes.len() - 1];
    if !(fractional.is_empty()
        || (fractional[0] == b'.'
            && (2..=7).contains(&fractional.len())
            && fractional[1..].iter().all(u8::is_ascii_digit)))
    {
        return Err(format!("registry {field} is not an ISO-8601 UTC timestamp"));
    }
    let digit_positions = [0, 1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18];
    if !digit_positions
        .iter()
        .all(|&index| bytes[index].is_ascii_digit())
    {
        return Err(format!("registry {field} is not an ISO-8601 UTC timestamp"));
    }
    let decimal = |start: usize, length: usize| -> u32 {
        bytes[start..start + length]
            .iter()
            .fold(0, |value, digit| value * 10 + u32::from(*digit - b'0'))
    };
    let year = decimal(0, 4);
    let month = decimal(5, 2);
    let day = decimal(8, 2);
    let hour = decimal(11, 2);
    let minute = decimal(14, 2);
    let second = decimal(17, 2);
    let leap = year.is_multiple_of(4) && (!year.is_multiple_of(100) || year.is_multiple_of(400));
    let maximum_day = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if leap => 29,
        2 => 28,
        _ => 0,
    };
    if year == 0 || day == 0 || day > maximum_day || hour > 23 || minute > 59 || second > 59 {
        return Err(format!("registry {field} is not an ISO-8601 UTC timestamp"));
    }
    Ok(())
}

fn validate_sha256(value: &str) -> Result<(), String> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err("registry SHA-256 must be 64 lowercase hexadecimal characters".to_owned());
    }
    Ok(())
}

fn build_search_engine(config: &PlayConfig, profile: &ResolvedProfile) -> SearchEngine {
    let mut search_config = SearchConfig {
        transposition_entries: 16_384,
        ..SearchConfig::default()
    };
    search_config.evaluation = evaluation_config(config.profile);
    profile.neural.as_ref().map_or_else(
        || SearchEngine::new(search_config),
        |neural| {
            let mode = match config.profile {
                PlayProfile::Residual => NeuralEvaluationMode::Residual,
                PlayProfile::Composite => NeuralEvaluationMode::Composite,
                _ => NeuralEvaluationMode::PureValue,
            };
            SearchEngine::with_neural_mode(search_config, Arc::clone(neural), mode)
        },
    )
}

const fn evaluation_config(profile: PlayProfile) -> EvaluationConfig {
    match profile {
        PlayProfile::Material => EvaluationConfig::material_only(),
        PlayProfile::HandcraftedBaseline => EvaluationConfig::handcrafted_baseline(),
        _ => EvaluationConfig::handcrafted_experimental(),
    }
}

fn play_config_sha256(
    config: &PlayConfig,
    profile: &ResolvedProfile,
    opening: Option<&ResolvedOpening>,
) -> String {
    let clock = clock_budget(config.budget);
    sha256_text(&format!(
        "schema=open_shogi_play_config/v3;time_control_schema={};human={};budget_kind={};budget_value={};black_time_ms={:?};white_time_ms={:?};byoyomi_ms={:?};black_increment_ms={:?};white_increment_ms={:?};safety_margin_ms={};depth={};initial_sfen={};max_plies={};profile={};model_id={};artifact={:?};payload={:?};architecture={:?};quantization={:?};registry_sha256={:?};registry_revision={:?};opening_sha256={:?};opening_size={:?};opening_max_plies={};opening_profile={};opening_minimum_samples={};opening_maximum_teacher_loss_cp={};transposition_entries=16384;engine={}:{}",
        open_shogi_core::TIME_CONTROL_SCHEMA,
        side_name(config.human),
        budget_parts(config.budget).0,
        budget_parts(config.budget).1,
        clock.map(|value| value.black_time),
        clock.map(|value| value.white_time),
        clock.and_then(|value| value.byoyomi),
        clock.and_then(|value| value.black_increment),
        clock.and_then(|value| value.white_increment),
        config.safety_margin_ms,
        config.depth,
        to_sfen(&config.initial),
        config.max_plies,
        profile_name(config.profile),
        profile.model_id,
        profile.model_artifact_sha256,
        profile.model_payload_sha256,
        profile.architecture_version,
        profile.quantization.map(quantization_name),
        profile.registry_sha256,
        profile.registry_revision,
        opening.map(|loaded| loaded.artifact_sha256.as_str()),
        opening.map(|loaded| loaded.artifact_size),
        config.opening_max_plies,
        config.opening_profile.name(),
        config.opening_minimum_samples,
        config.opening_maximum_teacher_loss_cp,
        open_shogi_core::ENGINE_NAME,
        open_shogi_core::ENGINE_VERSION,
    ))
}

fn recorded_play_config_sha256(config: &HumanPlayConfigRecord) -> String {
    sha256_text(&format!(
        "schema=open_shogi_play_config/v3;time_control_schema={};human={};budget_kind={};budget_value={};black_time_ms={:?};white_time_ms={:?};byoyomi_ms={:?};black_increment_ms={:?};white_increment_ms={:?};safety_margin_ms={};depth={};initial_sfen={};max_plies={};profile={};model_id={};artifact={:?};payload={:?};architecture={:?};quantization={:?};registry_sha256={:?};registry_revision={:?};opening_sha256={:?};opening_size={:?};opening_max_plies={};opening_profile={};opening_minimum_samples={};opening_maximum_teacher_loss_cp={};transposition_entries={};engine={}:{}",
        config.time_control_schema,
        config.human_side,
        config.budget_kind,
        config.budget_value,
        config.black_time_ms,
        config.white_time_ms,
        config.byoyomi_ms,
        config.black_increment_ms,
        config.white_increment_ms,
        config.safety_margin_ms,
        config.depth,
        config.initial_sfen,
        config.max_plies,
        config.profile,
        config.model_id,
        config.model_artifact_sha256,
        config.model_payload_sha256,
        config.architecture_version,
        config.quantization.as_deref(),
        config.registry_sha256,
        config.registry_revision,
        config.opening_artifact_sha256.as_deref(),
        config.opening_artifact_size,
        config.opening_max_plies,
        config.opening_profile,
        config.opening_minimum_samples,
        config.opening_maximum_teacher_loss_cp,
        config.transposition_entries,
        config.engine_name,
        config.engine_version,
    ))
}

fn human_play_config_record(
    config: &PlayConfig,
    profile: &ResolvedProfile,
    opening: Option<&ResolvedOpening>,
    config_sha256: &str,
) -> HumanPlayConfigRecord {
    let (budget_kind, budget_value) = budget_parts(config.budget);
    let clock = clock_budget(config.budget);
    HumanPlayConfigRecord {
        schema: "open_shogi_play_config/v3".to_owned(),
        config_sha256: config_sha256.to_owned(),
        human_side: side_name(config.human).to_owned(),
        budget_kind: budget_kind.to_owned(),
        budget_value,
        black_time_ms: clock.map(|value| value.black_time),
        white_time_ms: clock.map(|value| value.white_time),
        byoyomi_ms: clock.and_then(|value| value.byoyomi),
        black_increment_ms: clock.and_then(|value| value.black_increment),
        white_increment_ms: clock.and_then(|value| value.white_increment),
        safety_margin_ms: config.safety_margin_ms,
        time_control_schema: open_shogi_core::TIME_CONTROL_SCHEMA.to_owned(),
        depth: config.depth,
        initial_sfen: to_sfen(&config.initial),
        max_plies: config.max_plies,
        profile: profile_name(config.profile).to_owned(),
        model_id: profile.model_id.clone(),
        model_artifact_sha256: profile.model_artifact_sha256.clone(),
        model_payload_sha256: profile.model_payload_sha256.clone(),
        architecture_version: profile.architecture_version,
        quantization: profile
            .quantization
            .map(quantization_name)
            .map(str::to_owned),
        registry_sha256: profile.registry_sha256.clone(),
        registry_revision: profile.registry_revision,
        opening_artifact_sha256: opening.map(|loaded| loaded.artifact_sha256.clone()),
        opening_artifact_size: opening.map(|loaded| loaded.artifact_size),
        opening_max_plies: config.opening_max_plies,
        opening_profile: config.opening_profile.name().to_owned(),
        opening_minimum_samples: config.opening_minimum_samples,
        opening_maximum_teacher_loss_cp: config.opening_maximum_teacher_loss_cp,
        transposition_entries: 16_384,
        engine_name: open_shogi_core::ENGINE_NAME.to_owned(),
        engine_version: open_shogi_core::ENGINE_VERSION.to_owned(),
    }
}

const fn side_name(side: Side) -> &'static str {
    match side {
        Side::Black => "black",
        Side::White => "white",
    }
}

const fn budget_parts(budget: Budget) -> (&'static str, u64) {
    match budget {
        Budget::Casual => ("casual", open_shogi_core::CASUAL_HARD_MAX_MS),
        Budget::Nodes(value) => ("nodes", value),
        Budget::MoveTime(value) => ("movetime_ms", value),
        Budget::Clock(_) => ("clock", 0),
    }
}

const fn clock_budget(budget: Budget) -> Option<ClockBudget> {
    match budget {
        Budget::Clock(clock) => Some(clock),
        Budget::Casual | Budget::Nodes(_) | Budget::MoveTime(_) => None,
    }
}

fn time_control_for_budget(budget: Budget, safety_margin_ms: u64) -> TimeControl {
    match budget {
        Budget::Casual => TimeControl {
            safety_margin_ms,
            ..TimeControl::casual()
        },
        Budget::Nodes(nodes) => TimeControl {
            nodes: Some(nodes),
            casual: false,
            safety_margin_ms,
            ..TimeControl::casual()
        },
        Budget::MoveTime(movetime_ms) => TimeControl {
            movetime_ms: Some(movetime_ms),
            casual: false,
            safety_margin_ms,
            ..TimeControl::casual()
        },
        Budget::Clock(clock) => TimeControl {
            black_time_ms: Some(clock.black_time),
            white_time_ms: Some(clock.white_time),
            byoyomi_ms: clock.byoyomi,
            black_increment_ms: clock.black_increment,
            white_increment_ms: clock.white_increment,
            casual: false,
            safety_margin_ms,
            ..TimeControl::casual()
        },
    }
}

fn advance_clock_after_move(budget: &mut Budget, side: Side, elapsed_ms: u64) {
    let Budget::Clock(clock) = budget else {
        return;
    };
    let (remaining, increment) = match side {
        Side::Black => (&mut clock.black_time, clock.black_increment),
        Side::White => (&mut clock.white_time, clock.white_increment),
    };
    *remaining = remaining
        .saturating_sub(elapsed_ms)
        .saturating_add(increment.unwrap_or(0));
}

const fn profile_name(profile: PlayProfile) -> &'static str {
    match profile {
        PlayProfile::Material => "material",
        PlayProfile::HandcraftedBaseline => "handcrafted-baseline",
        PlayProfile::HandcraftedExperimental => "handcrafted-experimental",
        PlayProfile::OverallChampion => "overall-champion",
        PlayProfile::Neural => "neural",
        PlayProfile::Residual => "residual",
        PlayProfile::Composite => "composite",
        PlayProfile::GenerationZero => "generation-0",
        PlayProfile::NeuralLineageChampion => "neural-lineage-champion",
        PlayProfile::Challenger => "challenger",
    }
}

const fn quantization_name(value: NeuralQuantization) -> &'static str {
    match value {
        NeuralQuantization::Float32 => "float32",
        NeuralQuantization::Int8 => "int8",
    }
}

fn load_play_opening(config: &PlayConfig) -> Result<Option<ResolvedOpening>, String> {
    let Some(path) = config.opening_book_path.as_deref() else {
        return Ok(None);
    };
    let artifact = read_file_artifact(path, MAX_OPENING_BYTES)?;
    let book = OpeningBook::from_compressed_bytes(&artifact.bytes)?;
    Ok(Some(ResolvedOpening {
        book,
        artifact_sha256: artifact.sha256,
        artifact_size: artifact.size,
    }))
}

#[cfg(test)]
fn run_interactive<R: BufRead, W: Write>(
    config: &PlayConfig,
    reader: &mut R,
    writer: &mut W,
) -> Result<(), String> {
    let profile = resolve_profile(config)?;
    let opening = load_play_opening(config)?;
    run_interactive_resolved(config, &profile, opening.as_ref(), reader, writer)
}

#[expect(
    clippy::too_many_lines,
    reason = "the terminal game loop keeps human, book, search, and audit logging transitions together"
)]
fn run_interactive_resolved<R: BufRead, W: Write>(
    config: &PlayConfig,
    profile: &ResolvedProfile,
    opening: Option<&ResolvedOpening>,
    reader: &mut R,
    writer: &mut W,
) -> Result<(), String> {
    let publication = publication_paths(config)?;
    let publication_storage = PublicationStorage::open(config, &publication, true)?;
    validate_publication_path_identities(&publication_storage)?;
    refuse_publication_conflicts(&publication_storage)?;
    let mut game = Game::new(config.initial.clone());
    let mut engine = build_search_engine(config, profile);
    let config_sha256 = play_config_sha256(config, profile, opening);
    let config_record = human_play_config_record(config, profile, opening, &config_sha256);
    let mut current_budget = config.budget;
    let mut decisions = Vec::new();
    let mut special = None;
    writeln!(
        writer,
        "OpenShogiAI terminal play. Enter a USI move, `moves`, `sfen`, or `resign`."
    )
    .map_err(|error| error.to_string())?;
    writeln!(
        writer,
        "profile {} model {} architecture {} quantization {} config {}",
        profile.model_id,
        profile
            .model_artifact_sha256
            .as_deref()
            .map_or("none", |hash| &hash[..12]),
        profile
            .architecture_version
            .map_or_else(|| "none".to_owned(), |value| value.to_string()),
        profile
            .quantization
            .map_or("handcrafted", quantization_name),
        &config_sha256[..12],
    )
    .map_err(|error| error.to_string())?;
    for _ in 0..config.max_plies {
        render_position(game.position(), writer)?;
        if let Some(end) = game.end() {
            special = Some(special_from_end(end));
            break;
        }
        if game.position().side_to_move() == config.human {
            let sfen_before = to_sfen(game.position());
            match human_turn(&mut game, reader, writer)? {
                HumanTurn::Played(movement) => decisions.push(DecisionEvent {
                    schema: "phase6_human_decision/v1".to_owned(),
                    ply: game.moves().len(),
                    actor: "human".to_owned(),
                    model_id: profile.model_id.clone(),
                    model_artifact_sha256: profile.model_artifact_sha256.clone(),
                    model_payload_sha256: profile.model_payload_sha256.clone(),
                    config_sha256: config_sha256.clone(),
                    sfen_before,
                    move_usi: to_usi_move(movement),
                    nodes: 0,
                    elapsed_ms: 0,
                    depth: 0,
                    pv: Vec::new(),
                    score_cp: None,
                    opening_book: false,
                }),
                HumanTurn::Ended(terminal, validation) => {
                    special = Some((terminal, validation));
                }
            }
            if special.is_some() || game.end().is_some() {
                break;
            }
        } else {
            let sfen_before = to_sfen(game.position());
            if game.position().move_number() <= config.opening_max_plies
                && let Some(choice) = opening.and_then(|loaded| {
                    loaded.book.select_with_policy(
                        game.position(),
                        OpeningPolicy {
                            profile: config.opening_profile,
                            minimum_sample_count: config.opening_minimum_samples,
                            maximum_teacher_loss_cp: config.opening_maximum_teacher_loss_cp,
                        },
                    )
                })
            {
                let side = game.position().side_to_move();
                let score_rate = choice
                    .score_rate
                    .map_or_else(|| "n/a".to_owned(), |value| format!("{value:.3}"));
                writeln!(
                    writer,
                    "AI {} (source=book profile={} count={} score-rate={} teacher={} depth={} nodes={} classification={} provenance={})",
                    to_usi_move(choice.movement),
                    config.opening_profile.name(),
                    choice.count,
                    score_rate,
                    choice.teacher_score_cp.map_or_else(|| "none".to_owned(), |value| value.to_string()),
                    choice.teacher_depth.map_or_else(|| "none".to_owned(), |value| value.to_string()),
                    choice.teacher_nodes.map_or_else(|| "none".to_owned(), |value| value.to_string()),
                    choice.opening_classification,
                    choice.provenance_references.join(","),
                )
                .map_err(|error| error.to_string())?;
                game.play(choice.movement)
                    .map_err(|error| format!("opening returned illegal move: {error}"))?;
                decisions.push(DecisionEvent {
                    schema: "phase6_human_decision/v1".to_owned(),
                    ply: game.moves().len(),
                    actor: "ai".to_owned(),
                    model_id: profile.model_id.clone(),
                    model_artifact_sha256: profile.model_artifact_sha256.clone(),
                    model_payload_sha256: profile.model_payload_sha256.clone(),
                    config_sha256: config_sha256.clone(),
                    sfen_before,
                    move_usi: to_usi_move(choice.movement),
                    nodes: 0,
                    elapsed_ms: 0,
                    depth: 0,
                    pv: vec![to_usi_move(choice.movement)],
                    score_cp: None,
                    opening_book: true,
                });
                advance_clock_after_move(&mut current_budget, side, 0);
                continue;
            }
            let side = game.position().side_to_move();
            let request = time_control_for_budget(current_budget, config.safety_margin_ms);
            let plan = TimeManager::default().plan(
                game.position().side_to_move(),
                request,
                config.depth,
            )?;
            let result =
                engine.search_managed(game.position(), plan, &CancellationToken::new());
            let Some(movement) = result.best_move else {
                special = Some((
                    CsaSpecialMove::Resign,
                    CsaResultValidation::ExternalCondition,
                ));
                break;
            };
            writeln!(
                writer,
                "AI {} (depth {} nodes {} elapsed {}ms score {} pv {})",
                to_usi_move(movement),
                result.depth,
                result.nodes,
                result.elapsed.as_millis(),
                result.score,
                result
                    .pv
                    .iter()
                    .copied()
                    .map(to_usi_move)
                    .collect::<Vec<_>>()
                    .join(" "),
            )
            .map_err(|error| error.to_string())?;
            let move_usi = to_usi_move(movement);
            let pv = result.pv.iter().copied().map(to_usi_move).collect();
            game.play(movement).map_err(|error| {
                format!(
                    "search returned illegal move {}: {error}",
                    to_usi_move(movement)
                )
            })?;
            decisions.push(DecisionEvent {
                schema: "phase6_human_decision/v1".to_owned(),
                ply: game.moves().len(),
                actor: "ai".to_owned(),
                model_id: profile.model_id.clone(),
                model_artifact_sha256: profile.model_artifact_sha256.clone(),
                model_payload_sha256: profile.model_payload_sha256.clone(),
                config_sha256: config_sha256.clone(),
                sfen_before,
                move_usi,
                nodes: result.nodes,
                elapsed_ms: u64::try_from(result.elapsed.as_millis()).unwrap_or(u64::MAX),
                depth: result.depth,
                pv,
                score_cp: Some(result.score),
                opening_book: false,
            });
            advance_clock_after_move(
                &mut current_budget,
                side,
                u64::try_from(result.elapsed.as_millis()).unwrap_or(u64::MAX),
            );
        }
    }
    let (special, validation) = special
        .or_else(|| game.end().map(special_from_end))
        .unwrap_or((
            CsaSpecialMove::MaxMoves,
            CsaResultValidation::ExternalCondition,
        ));
    let csa = encode_record(
        &config.initial,
        game.moves(),
        config.human,
        special,
        validation,
        &config_record,
    )?;
    let decision_log = encode_decision_log(&config_record, &decisions)?;
    publish_human_artifacts_with_storage(
        config,
        &publication,
        &publication_storage,
        &config_sha256,
        csa.as_bytes(),
        &decision_log,
    )?;
    writeln!(writer, "CSA saved to {}", config.output.display()).map_err(|error| error.to_string())
}

fn human_turn<R: BufRead, W: Write>(
    game: &mut Game,
    reader: &mut R,
    writer: &mut W,
) -> Result<HumanTurn, String> {
    loop {
        write!(writer, "your move> ").map_err(|error| error.to_string())?;
        writer.flush().map_err(|error| error.to_string())?;
        let line = match read_bounded_line(reader, 128).map_err(|error| error.to_string())? {
            BoundedInputLine::Line(line) => line,
            BoundedInputLine::TooLong => {
                writeln!(writer, "input too long").map_err(|error| error.to_string())?;
                continue;
            }
            BoundedInputLine::Eof => {
                return Ok(HumanTurn::Ended(
                    CsaSpecialMove::Interrupted,
                    CsaResultValidation::ExternalCondition,
                ));
            }
        };
        match line.trim() {
            "moves" => write_legal_moves(game.position(), writer)?,
            "sfen" => {
                writeln!(writer, "{}", to_sfen(game.position()))
                    .map_err(|error| error.to_string())?;
            }
            "resign" | "quit" => {
                game.resign().map_err(|error| error.to_string())?;
                return Ok(HumanTurn::Ended(
                    CsaSpecialMove::Resign,
                    CsaResultValidation::ExternalCondition,
                ));
            }
            notation => match parse_usi_move(notation) {
                Ok(movement) => match game.play(movement) {
                    Ok(_) => return Ok(HumanTurn::Played(movement)),
                    Err(error) => {
                        writeln!(writer, "illegal move: {error}")
                            .map_err(|io_error| io_error.to_string())?;
                    }
                },
                Err(error) => {
                    writeln!(writer, "invalid command or move: {error}")
                        .map_err(|io_error| io_error.to_string())?;
                }
            },
        }
    }
}

fn write_legal_moves<W: Write>(position: &Position, writer: &mut W) -> Result<(), String> {
    let legal = position
        .legal_moves()
        .into_iter()
        .map(to_usi_move)
        .collect::<Vec<_>>()
        .join(" ");
    writeln!(writer, "{legal}").map_err(|error| error.to_string())
}

fn render_position<W: Write>(position: &Position, writer: &mut W) -> Result<(), String> {
    writeln!(writer, "    9  8  7  6  5  4  3  2  1").map_err(|error| error.to_string())?;
    for rank in 1..=9_u8 {
        write!(writer, "{} ", char::from(b'a' + rank - 1)).map_err(|error| error.to_string())?;
        for file in (1..=9_u8).rev() {
            let square = open_shogi_core::Square::new(file, rank).expect("board coordinate");
            match position.piece_at(square) {
                None => write!(writer, " . ").map_err(|error| error.to_string())?,
                Some(piece) => {
                    let letter = piece.kind.sfen_letter();
                    let letter = match piece.side {
                        Side::Black => letter,
                        Side::White => letter.to_ascii_lowercase(),
                    };
                    let promotion = if piece.kind.is_promoted() { '+' } else { ' ' };
                    write!(writer, "{promotion}{letter}").map_err(|error| error.to_string())?;
                }
            }
        }
        writeln!(writer).map_err(|error| error.to_string())?;
    }
    writeln!(writer, "Black hand: {}", format_hand(position, Side::Black))
        .map_err(|error| error.to_string())?;
    writeln!(writer, "White hand: {}", format_hand(position, Side::White))
        .map_err(|error| error.to_string())?;
    writeln!(writer, "side: {:?}", position.side_to_move()).map_err(|error| error.to_string())
}

fn format_hand(position: &Position, side: Side) -> String {
    let mut pieces = Vec::new();
    for hand_piece in HandPiece::DISPLAY_ORDER {
        let count = position.hand(side).count(hand_piece);
        if count > 0 {
            pieces.push(format!("{}{count}", hand_piece.piece_kind().sfen_letter()));
        }
    }
    if pieces.is_empty() {
        "-".to_owned()
    } else {
        pieces.join(" ")
    }
}

fn special_from_end(end: GameEnd) -> (CsaSpecialMove, CsaResultValidation) {
    match end {
        GameEnd::Checkmate { .. } => (CsaSpecialMove::Checkmate, CsaResultValidation::Verified),
        GameEnd::Repetition(RepetitionOutcome::NoContest) => {
            (CsaSpecialMove::Repetition, CsaResultValidation::Verified)
        }
        GameEnd::Repetition(RepetitionOutcome::PerpetualCheckLoss(_)) => (
            CsaSpecialMove::PerpetualCheck,
            CsaResultValidation::Verified,
        ),
        GameEnd::Resignation { .. } => (
            CsaSpecialMove::Resign,
            CsaResultValidation::ExternalCondition,
        ),
        GameEnd::Impasse(_) => (
            CsaSpecialMove::EnteringKing,
            CsaResultValidation::ExternalCondition,
        ),
        GameEnd::EnteringKing(_) => (CsaSpecialMove::Win, CsaResultValidation::ExternalCondition),
    }
}

fn encode_record(
    initial: &Position,
    moves: &[Move],
    human: Side,
    special: CsaSpecialMove,
    validation: CsaResultValidation,
    config: &HumanPlayConfigRecord,
) -> Result<String, String> {
    let (black_name, white_name) = match human {
        Side::Black => ("human", "OpenShogiAI"),
        Side::White => ("OpenShogiAI", "human"),
    };
    let game = CsaGame {
        version: "V3.0".to_owned(),
        black_name: Some(black_name.to_owned()),
        white_name: Some(white_name.to_owned()),
        metadata: vec![
            ("CONFIG_SHA256".to_owned(), config.config_sha256.clone()),
            ("PROFILE".to_owned(), config.profile.clone()),
            ("MODEL_ID".to_owned(), config.model_id.clone()),
            (
                "MODEL_ARTIFACT_SHA256".to_owned(),
                config
                    .model_artifact_sha256
                    .clone()
                    .unwrap_or_else(|| "none".to_owned()),
            ),
            (
                "MODEL_PAYLOAD_SHA256".to_owned(),
                config
                    .model_payload_sha256
                    .clone()
                    .unwrap_or_else(|| "none".to_owned()),
            ),
            (
                "REGISTRY_SHA256".to_owned(),
                config
                    .registry_sha256
                    .clone()
                    .unwrap_or_else(|| "none".to_owned()),
            ),
            (
                "OPENING_SHA256".to_owned(),
                config
                    .opening_artifact_sha256
                    .clone()
                    .unwrap_or_else(|| "none".to_owned()),
            ),
        ],
        initial_position: initial.clone(),
        moves: moves.to_vec(),
        special_move: Some(special),
        result_validation: validation,
    };
    to_csa_game(&game).map_err(|error| format!("cannot encode CSA: {error}"))
}

fn encode_decision_log(
    config: &HumanPlayConfigRecord,
    events: &[DecisionEvent],
) -> Result<Vec<u8>, String> {
    let mut writer = Vec::new();
    serde_json::to_writer(&mut writer, config)
        .map_err(|error| format!("cannot encode play configuration: {error}"))?;
    writer.push(b'\n');
    for event in events {
        serde_json::to_writer(&mut writer, event)
            .map_err(|error| format!("cannot encode decision log: {error}"))?;
        writer.push(b'\n');
    }
    if writer.len() > 64 * 1024 * 1024 {
        return Err("decision log exceeds the 64 MiB publication limit".to_owned());
    }
    Ok(writer)
}

fn validate_paired_evidence(
    csa: &[u8],
    decision_log: &[u8],
    expected_config_sha256: &str,
) -> Result<(), String> {
    validate_sha256(expected_config_sha256)?;
    let csa_text = std::str::from_utf8(csa)
        .map_err(|_| "human-play CSA evidence is not valid UTF-8".to_owned())?;
    let parsed = open_shogi_core::parse_csa_game(csa_text)
        .map_err(|error| format!("human-play CSA evidence is invalid: {error}"))?;
    let canonical = to_csa_game(&parsed)
        .map_err(|error| format!("cannot canonicalize human-play CSA evidence: {error}"))?;
    if canonical.as_bytes() != csa {
        return Err("human-play CSA evidence is not canonical".to_owned());
    }
    if decision_log.is_empty() || !decision_log.ends_with(b"\n") || decision_log.contains(&b'\r') {
        return Err("human-play decision log must use canonical LF-terminated JSONL".to_owned());
    }
    let decision_text = std::str::from_utf8(decision_log)
        .map_err(|_| "human-play decision log is not valid UTF-8".to_owned())?;
    let mut lines = decision_text.split_terminator('\n');
    let config_line = lines
        .next()
        .ok_or_else(|| "human-play decision log lacks a configuration record".to_owned())?;
    let config: HumanPlayConfigRecord = serde_json::from_str(config_line)
        .map_err(|error| format!("invalid human-play configuration record: {error}"))?;
    let canonical_config = serde_json::to_vec(&config)
        .map_err(|error| format!("cannot canonicalize human-play configuration: {error}"))?;
    if canonical_config != config_line.as_bytes()
        || config.schema != "open_shogi_play_config/v3"
        || config.config_sha256 != expected_config_sha256
        || recorded_play_config_sha256(&config) != expected_config_sha256
    {
        return Err("human-play configuration record identity mismatch".to_owned());
    }
    validate_human_play_config_record(&config)?;
    validate_csa_evidence_identity(&parsed, &config)?;

    let events = lines
        .enumerate()
        .map(|(index, line)| {
            if line.is_empty() {
                return Err("human-play decision log contains an empty record".to_owned());
            }
            let event: DecisionEvent = serde_json::from_str(line)
                .map_err(|error| format!("invalid human-play decision record: {error}"))?;
            let canonical_event = serde_json::to_vec(&event).map_err(|error| {
                format!("cannot canonicalize human-play decision record: {error}")
            })?;
            if canonical_event != line.as_bytes()
                || event.schema != "phase6_human_decision/v1"
                || event.config_sha256 != expected_config_sha256
            {
                return Err("human-play decision record identity mismatch".to_owned());
            }
            if event.ply != index + 1 {
                return Err("human-play decision plies are not contiguous".to_owned());
            }
            Ok(event)
        })
        .collect::<Result<Vec<_>, _>>()?;
    validate_decision_replay(&parsed, &config, &events)?;
    Ok(())
}

#[expect(
    clippy::too_many_lines,
    reason = "the durable play receipt validates every closed identity and compatibility field"
)]
fn validate_human_play_config_record(config: &HumanPlayConfigRecord) -> Result<(), String> {
    validate_sha256(&config.config_sha256)?;
    let clock_fields = [
        config.black_time_ms,
        config.white_time_ms,
        config.byoyomi_ms,
        config.black_increment_ms,
        config.white_increment_ms,
    ];
    let clock_valid = if config.budget_kind == "clock" {
        config.budget_value == 0
            && config.black_time_ms.is_some()
            && config.white_time_ms.is_some()
            && TimeControl {
                black_time_ms: config.black_time_ms,
                white_time_ms: config.white_time_ms,
                byoyomi_ms: config.byoyomi_ms,
                black_increment_ms: config.black_increment_ms,
                white_increment_ms: config.white_increment_ms,
                casual: false,
                safety_margin_ms: config.safety_margin_ms,
                ..TimeControl::casual()
            }
            .validate()
            .is_ok()
    } else {
        config.budget_value > 0 && clock_fields.iter().all(Option::is_none)
    };
    if !matches!(config.human_side.as_str(), "black" | "white")
        || !matches!(
            config.budget_kind.as_str(),
            "casual" | "nodes" | "movetime_ms" | "clock"
        )
        || !clock_valid
        || (config.budget_kind == "casual"
            && config.budget_value != open_shogi_core::CASUAL_HARD_MAX_MS)
        || (config.budget_kind == "nodes" && config.budget_value > MAX_NODES_PER_MOVE)
        || (config.budget_kind == "movetime_ms" && config.budget_value > MAX_MOVETIME_MS)
        || config.safety_margin_ms > open_shogi_core::MAX_SAFETY_MARGIN_MS
        || config.time_control_schema != open_shogi_core::TIME_CONTROL_SCHEMA
        || !(1..=64).contains(&config.depth)
        || !(1..=MAX_PLIES).contains(&config.max_plies)
        || !(1..=MAX_PLIES).contains(&config.opening_max_plies)
        || !matches!(
            config.opening_profile.as_str(),
            "unrestricted" | "ibisha_preferred" | "ibisha_strict"
        )
        || config.opening_minimum_samples == 0
        || config.opening_minimum_samples > 1_000_000
        || !(0..=10_000).contains(&config.opening_maximum_teacher_loss_cp)
        || config.transposition_entries != 16_384
        || config.engine_name != open_shogi_core::ENGINE_NAME
        || config.engine_version != open_shogi_core::ENGINE_VERSION
    {
        return Err("human-play configuration record violates bounded options".to_owned());
    }
    let initial = parse_sfen(&config.initial_sfen)
        .map_err(|error| format!("human-play configuration SFEN is invalid: {error}"))?;
    if initial.move_number() != 1 || to_sfen(&initial) != config.initial_sfen {
        return Err("human-play configuration SFEN is not canonical move one".to_owned());
    }
    let model_fields = [
        config.model_artifact_sha256.is_some(),
        config.model_payload_sha256.is_some(),
        config.architecture_version.is_some(),
        config.quantization.is_some(),
    ];
    let neural = matches!(
        config.profile.as_str(),
        "neural" | "generation-0" | "neural-lineage-champion" | "challenger"
    );
    if neural != model_fields.into_iter().all(std::convert::identity)
        || (!neural && model_fields.into_iter().any(std::convert::identity))
        || !matches!(
            config.profile.as_str(),
            "material"
                | "handcrafted-baseline"
                | "handcrafted-experimental"
                | "overall-champion"
                | "neural"
                | "generation-0"
                | "neural-lineage-champion"
                | "challenger"
        )
    {
        return Err("human-play configuration model identity is inconsistent".to_owned());
    }
    if let Some(hash) = config.model_artifact_sha256.as_deref() {
        validate_sha256(hash)?;
    }
    if let Some(hash) = config.model_payload_sha256.as_deref() {
        validate_sha256(hash)?;
    }
    if neural
        && (config.architecture_version != Some(1)
            || !matches!(config.quantization.as_deref(), Some("float32" | "int8")))
    {
        return Err("human-play neural configuration identity is unsupported".to_owned());
    }
    match config.profile.as_str() {
        "material" if config.model_id == "material" => {}
        "handcrafted-baseline" if config.model_id == "handcrafted-baseline" => {}
        "handcrafted-experimental" if config.model_id == "handcrafted-experimental" => {}
        "overall-champion" if config.model_id == open_shogi_core::OVERALL_CHAMPION_ID => {}
        "neural" if config.model_id == "neural-direct" => {}
        "residual" if config.model_id == "residual-direct" => {}
        "composite" if config.model_id == "composite-direct" => {}
        "generation-0" | "neural-lineage-champion" | "challenger" => {
            validate_identifier(&config.model_id, "human-play modelId")?;
        }
        _ => return Err("human-play profile and modelId disagree".to_owned()),
    }
    let registry_fields = [
        config.registry_sha256.is_some(),
        config.registry_revision.is_some(),
    ];
    let registry_profile = matches!(
        config.profile.as_str(),
        "generation-0" | "neural-lineage-champion" | "challenger"
    );
    if registry_profile != registry_fields.into_iter().all(std::convert::identity)
        || (!registry_profile && registry_fields.into_iter().any(std::convert::identity))
    {
        return Err("human-play registry configuration identity is inconsistent".to_owned());
    }
    if let Some(hash) = config.registry_sha256.as_deref() {
        validate_sha256(hash)?;
    }
    if config
        .registry_revision
        .is_some_and(|revision| !(1..=1_000_000_000).contains(&revision))
        || config.opening_artifact_sha256.is_some() != config.opening_artifact_size.is_some()
    {
        return Err("human-play optional artifact identity is inconsistent".to_owned());
    }
    if let Some(hash) = config.opening_artifact_sha256.as_deref() {
        validate_sha256(hash)?;
        if config
            .opening_artifact_size
            .is_none_or(|size| size == 0 || size > MAX_OPENING_BYTES)
        {
            return Err("human-play opening artifact size is invalid".to_owned());
        }
    }
    Ok(())
}

fn validate_csa_evidence_identity(
    game: &CsaGame,
    config: &HumanPlayConfigRecord,
) -> Result<(), String> {
    let expected_names = match config.human_side.as_str() {
        "black" => ("human", "OpenShogiAI"),
        "white" => ("OpenShogiAI", "human"),
        _ => return Err("human-play configuration has an invalid human side".to_owned()),
    };
    let initial = parse_sfen(&config.initial_sfen)
        .map_err(|error| format!("human-play configuration SFEN is invalid: {error}"))?;
    if game.version != "V3.0"
        || game.initial_position != initial
        || game.black_name.as_deref() != Some(expected_names.0)
        || game.white_name.as_deref() != Some(expected_names.1)
        || game.special_move.is_none()
        || game.moves.len() > usize::try_from(config.max_plies).unwrap_or(usize::MAX)
    {
        return Err("human-play CSA evidence identity mismatch".to_owned());
    }
    let metadata = game
        .metadata
        .iter()
        .map(|(key, value)| (key.as_str(), value.as_str()))
        .collect::<BTreeMap<_, _>>();
    if game.metadata.len() != 7
        || metadata.len() != 7
        || metadata.get("CONFIG_SHA256") != Some(&config.config_sha256.as_str())
        || metadata.get("PROFILE") != Some(&config.profile.as_str())
        || metadata.get("MODEL_ID") != Some(&config.model_id.as_str())
        || metadata.get("MODEL_ARTIFACT_SHA256")
            != Some(&config.model_artifact_sha256.as_deref().unwrap_or("none"))
        || metadata.get("MODEL_PAYLOAD_SHA256")
            != Some(&config.model_payload_sha256.as_deref().unwrap_or("none"))
        || metadata.get("REGISTRY_SHA256")
            != Some(&config.registry_sha256.as_deref().unwrap_or("none"))
        || metadata.get("OPENING_SHA256")
            != Some(&config.opening_artifact_sha256.as_deref().unwrap_or("none"))
    {
        return Err("human-play CSA metadata identity mismatch".to_owned());
    }
    Ok(())
}

fn validate_decision_replay(
    game: &CsaGame,
    config: &HumanPlayConfigRecord,
    events: &[DecisionEvent],
) -> Result<(), String> {
    if events.len() != game.moves.len() {
        return Err("human-play decision count does not match CSA moves".to_owned());
    }
    let human = if config.human_side == "black" {
        Side::Black
    } else {
        Side::White
    };
    let mut replay = Game::new(game.initial_position.clone());
    for (event, &movement) in events.iter().zip(&game.moves) {
        let expected_actor = if replay.position().side_to_move() == human {
            "human"
        } else {
            "ai"
        };
        let parsed_move = parse_usi_move(&event.move_usi)
            .map_err(|error| format!("human-play decision move is invalid: {error}"))?;
        if event.actor != expected_actor
            || event.model_id != config.model_id
            || event.model_artifact_sha256 != config.model_artifact_sha256
            || event.model_payload_sha256 != config.model_payload_sha256
            || event.sfen_before != to_sfen(replay.position())
            || to_usi_move(parsed_move) != event.move_usi
            || parsed_move != movement
            || event.nodes > MAX_JSON_SAFE_INTEGER
            || event.elapsed_ms > MAX_JSON_SAFE_INTEGER
            || (event.actor == "human"
                && (event.nodes != 0
                    || event.elapsed_ms != 0
                    || event.depth != 0
                    || !event.pv.is_empty()
                    || event.score_cp.is_some()
                    || event.opening_book))
            || (event.opening_book
                && (config.opening_artifact_sha256.is_none()
                    || event.actor != "ai"
                    || replay.position().move_number() > config.opening_max_plies
                    || event.nodes != 0
                    || event.elapsed_ms != 0
                    || event.depth != 0
                    || event.score_cp.is_some()
                    || event.pv != [event.move_usi.clone()]))
            || (!event.opening_book
                && event.actor == "ai"
                && (event.score_cp.is_none()
                    || event.depth > config.depth
                    || event.pv.first() != Some(&event.move_usi)
                    || (config.budget_kind == "nodes" && event.nodes > config.budget_value)))
        {
            return Err("human-play decision evidence mismatch".to_owned());
        }
        validate_decision_pv(event, replay.position())?;
        replay
            .play(movement)
            .map_err(|error| format!("human-play decision replay failed: {error}"))?;
    }
    validate_human_play_terminal(game, config, replay.end(), replay.position().side_to_move())
}

fn validate_decision_pv(event: &DecisionEvent, position: &Position) -> Result<(), String> {
    if event.actor != "ai" {
        return Ok(());
    }
    if event.pv.is_empty() || event.pv.len() > usize::try_from(MAX_PLIES).unwrap_or(usize::MAX) {
        return Err("human-play AI principal variation length is invalid".to_owned());
    }
    let mut replay = position.clone();
    for notation in &event.pv {
        let movement = parse_usi_move(notation)
            .map_err(|error| format!("human-play principal variation is invalid: {error}"))?;
        if to_usi_move(movement) != *notation {
            return Err("human-play principal variation is not canonical USI".to_owned());
        }
        replay
            .make_move(movement)
            .map_err(|error| format!("human-play principal variation is illegal: {error}"))?;
    }
    Ok(())
}

fn validate_human_play_terminal(
    game: &CsaGame,
    config: &HumanPlayConfigRecord,
    replayed_end: Option<GameEnd>,
    final_side_to_move: Side,
) -> Result<(), String> {
    let special = game
        .special_move
        .as_ref()
        .ok_or_else(|| "human-play CSA lacks a terminal result".to_owned())?;
    let valid = match special {
        CsaSpecialMove::Checkmate => matches!(replayed_end, Some(GameEnd::Checkmate { .. })),
        CsaSpecialMove::Repetition => matches!(
            replayed_end,
            Some(GameEnd::Repetition(RepetitionOutcome::NoContest))
        ),
        CsaSpecialMove::PerpetualCheck => matches!(
            replayed_end,
            Some(GameEnd::Repetition(RepetitionOutcome::PerpetualCheckLoss(
                _
            )))
        ),
        CsaSpecialMove::Resign => replayed_end.is_none(),
        CsaSpecialMove::Interrupted => {
            let human = if config.human_side == "black" {
                Side::Black
            } else {
                Side::White
            };
            replayed_end.is_none() && final_side_to_move == human
        }
        CsaSpecialMove::MaxMoves => {
            replayed_end.is_none()
                && game.moves.len() == usize::try_from(config.max_plies).unwrap_or(usize::MAX)
        }
        _ => false,
    };
    if !valid {
        return Err("human-play CSA terminal result is inconsistent with replay".to_owned());
    }
    Ok(())
}

struct PublicationPaths {
    marker: PathBuf,
    marker_pending: PathBuf,
    csa_pending: PathBuf,
    decision_pending: PathBuf,
}

struct AnchoredPublicationFile {
    directory: AnchoredDir,
    parent_binding: AnchoredDir,
    name: std::ffi::OsString,
    display: PathBuf,
}

impl AnchoredPublicationFile {
    #[cfg(test)]
    fn from_path(path: &Path, create_parent: bool) -> Result<Self, String> {
        let mut cache = PublicationParentCache::default();
        cache.anchor(path, create_parent)
    }

    fn from_anchored_parent(
        directory: AnchoredDir,
        parent_binding: AnchoredDir,
        name: std::ffi::OsString,
    ) -> Self {
        let display = parent_binding.display().join(&name);
        Self {
            directory,
            parent_binding,
            name,
            display,
        }
    }

    fn kind(&self) -> Result<Option<EntryKind>, String> {
        self.directory
            .entry_kind(&self.name)
            .map_err(|error| format!("cannot inspect {}: {error}", self.display.display()))
    }

    fn read(&self, maximum_size: u64) -> Result<FileArtifact, String> {
        self.read_retained(maximum_size)
            .map(|(artifact, _)| artifact)
    }

    fn read_retained(&self, maximum_size: u64) -> Result<(FileArtifact, AnchoredFile), String> {
        let mut file = self.directory.open_regular(&self.name).map_err(|error| {
            format!(
                "cannot open {} without following links: {error}",
                self.display.display()
            )
        })?;
        let artifact = read_retained_file_artifact(&mut file, &self.display, maximum_size)?;
        Ok((artifact, file))
    }

    fn publish_new_atomic(&self, bytes: &[u8]) -> Result<(), String> {
        self.directory
            .publish_new_atomic(&self.name, bytes)
            .map_err(|error| format!("cannot publish {}: {error}", self.display.display()))
    }

    fn link_new_to(&self, target: &Self) -> Result<(), String> {
        self.directory
            .link_new_to(&self.name, &target.directory, &target.name)
            .map_err(|error| format!("cannot publish {}: {error}", target.display.display()))
    }

    fn remove_retained(&self, file: &AnchoredFile) -> Result<(), String> {
        self.directory
            .remove_anchored_file(file)
            .map_err(|error| format!("cannot remove {}: {error}", self.display.display()))
    }
}

#[derive(Default)]
struct PublicationParentCache {
    by_lexical_parent: BTreeMap<PathBuf, AnchoredDir>,
    by_identity: BTreeMap<StableDirectoryIdentity, AnchoredDir>,
}

impl PublicationParentCache {
    fn anchor(
        &mut self,
        path: &Path,
        create_parent: bool,
    ) -> Result<AnchoredPublicationFile, String> {
        let name = path
            .file_name()
            .ok_or_else(|| format!("publication path has no file name: {}", path.display()))?
            .to_os_string();
        if !name
            .to_str()
            .is_some_and(|name| !name.is_empty() && name.is_ascii())
        {
            return Err(
                "human-play publication file names must be non-empty ASCII for collision-safe identity"
                    .to_owned(),
            );
        }
        let parent = path.parent().unwrap_or_else(|| Path::new("."));
        let lexical_parent = if parent.is_absolute() {
            parent.to_path_buf()
        } else {
            std::env::current_dir()
                .map_err(|error| format!("cannot inspect current directory: {error}"))?
                .join(parent)
        };
        let binding = if let Some(existing) = self.by_lexical_parent.get(&lexical_parent) {
            existing
                .try_clone()
                .map_err(|error| format!("cannot clone publication parent: {error}"))?
        } else {
            let (opened, opened_name) =
                AnchoredDir::open_parent(path, create_parent).map_err(|error| {
                    format!(
                        "cannot anchor publication path {} without symlinks: {error}",
                        path.display()
                    )
                })?;
            if opened_name != name {
                return Err("publication final component changed during anchoring".to_owned());
            }
            self.by_lexical_parent.insert(
                lexical_parent,
                opened
                    .try_clone()
                    .map_err(|error| format!("cannot retain publication parent: {error}"))?,
            );
            opened
        };
        let identity = binding
            .stable_identity()
            .map_err(|error| format!("cannot identify publication parent: {error}"))?;
        let authority = if let Some(existing) = self.by_identity.get(&identity) {
            existing
                .try_clone()
                .map_err(|error| format!("cannot clone publication authority: {error}"))?
        } else {
            self.by_identity.insert(
                identity,
                binding
                    .try_clone()
                    .map_err(|error| format!("cannot retain publication authority: {error}"))?,
            );
            binding
                .try_clone()
                .map_err(|error| format!("cannot clone publication authority: {error}"))?
        };
        Ok(AnchoredPublicationFile::from_anchored_parent(
            authority, binding, name,
        ))
    }
}

struct PublicationStorage {
    output: AnchoredPublicationFile,
    decision: AnchoredPublicationFile,
    marker: AnchoredPublicationFile,
    marker_pending: AnchoredPublicationFile,
    csa_pending: AnchoredPublicationFile,
    decision_pending: AnchoredPublicationFile,
}

impl PublicationStorage {
    fn open(config: &PlayConfig, paths: &PublicationPaths, create: bool) -> Result<Self, String> {
        let mut parents = PublicationParentCache::default();
        Ok(Self {
            output: parents.anchor(&config.output, create)?,
            decision: parents.anchor(&config.decision_log, create)?,
            marker: parents.anchor(&paths.marker, create)?,
            marker_pending: parents.anchor(&paths.marker_pending, create)?,
            csa_pending: parents.anchor(&paths.csa_pending, create)?,
            decision_pending: parents.anchor(&paths.decision_pending, create)?,
        })
    }

    fn files(&self) -> [&AnchoredPublicationFile; 6] {
        [
            &self.output,
            &self.decision,
            &self.marker,
            &self.marker_pending,
            &self.csa_pending,
            &self.decision_pending,
        ]
    }
}

fn publication_paths(config: &PlayConfig) -> Result<PublicationPaths, String> {
    let output_name = config
        .output
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| "CSA output path has no UTF-8 file name".to_owned())?;
    let decision_name = config
        .decision_log
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| "decision-log path has no UTF-8 file name".to_owned())?;
    let marker = config
        .output
        .with_file_name(format!(".{output_name}.publication.json"));
    Ok(PublicationPaths {
        marker_pending: marker_pending_path(&marker)?,
        marker,
        csa_pending: config
            .output
            .with_file_name(format!(".{output_name}.pending-publication")),
        decision_pending: config
            .decision_log
            .with_file_name(format!(".{decision_name}.pending-publication")),
    })
}

fn marker_pending_path(marker: &Path) -> Result<PathBuf, String> {
    let file_name = marker
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| "publication marker has no UTF-8 file name".to_owned())?;
    Ok(marker.with_file_name(format!(".{file_name}.pending-marker")))
}

fn validate_publication_path_identities(storage: &PublicationStorage) -> Result<(), String> {
    let mut identities = std::collections::BTreeSet::new();
    for file in storage.files() {
        file.parent_binding
            .verify_display_identity()
            .map_err(|error| {
                format!(
                    "publication parent identity changed for {}: {error}",
                    file.display.display()
                )
            })?;
        let parent = file
            .directory
            .stable_identity()
            .map_err(|error| format!("cannot identify publication parent: {error}"))?;
        let name = file
            .name
            .to_str()
            .ok_or_else(|| "human-play publication paths must be valid UTF-8".to_owned())?
            .to_lowercase();
        if !identities.insert((parent, name)) {
            return Err("human-play publication paths must be distinct".to_owned());
        }
    }
    Ok(())
}

fn refuse_publication_conflicts(storage: &PublicationStorage) -> Result<(), String> {
    for file in [
        &storage.output,
        &storage.decision,
        &storage.marker,
        &storage.marker_pending,
        &storage.csa_pending,
        &storage.decision_pending,
    ] {
        if file.kind()?.is_some() {
            return Err(format!(
                "refusing to overwrite existing or incomplete human-play artifact {}",
                file.display.display()
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
fn publish_human_artifacts(
    config: &PlayConfig,
    config_sha256: &str,
    csa: &[u8],
    decisions: &[u8],
) -> Result<(), String> {
    let paths = publication_paths(config)?;
    let storage = PublicationStorage::open(config, &paths, true)?;
    publish_human_artifacts_with_storage(config, &paths, &storage, config_sha256, csa, decisions)
}

fn publish_human_artifacts_with_storage(
    config: &PlayConfig,
    paths: &PublicationPaths,
    storage: &PublicationStorage,
    config_sha256: &str,
    csa: &[u8],
    decisions: &[u8],
) -> Result<(), String> {
    if csa.is_empty() || u64::try_from(csa.len()).unwrap_or(u64::MAX) > MAX_HUMAN_CSA_BYTES {
        return Err(format!(
            "CSA publication size must be 1..={MAX_HUMAN_CSA_BYTES} bytes"
        ));
    }
    if decisions.is_empty()
        || u64::try_from(decisions.len()).unwrap_or(u64::MAX) > MAX_HUMAN_DECISION_BYTES
    {
        return Err(format!(
            "decision-log publication size must be 1..={MAX_HUMAN_DECISION_BYTES} bytes"
        ));
    }
    validate_paired_evidence(csa, decisions, config_sha256)?;
    validate_publication_path_identities(storage)?;
    refuse_publication_conflicts(storage)?;
    storage.csa_pending.publish_new_atomic(csa)?;
    if let Err(error) = storage.decision_pending.publish_new_atomic(decisions) {
        return Err(format!(
            "{error}; complete CSA pending file retained for fail-closed recovery"
        ));
    }
    let marker = publication_marker(
        storage,
        config_sha256,
        &FileArtifact {
            bytes: Vec::new(),
            sha256: sha256_bytes(csa),
            size: u64::try_from(csa.len()).unwrap_or(u64::MAX),
        },
        &FileArtifact {
            bytes: Vec::new(),
            sha256: sha256_bytes(decisions),
            size: u64::try_from(decisions.len()).unwrap_or(u64::MAX),
        },
    )?;
    let marker_bytes = serde_json::to_vec(&marker)
        .map_err(|error| format!("cannot encode publication marker: {error}"))?;
    publish_marker_transaction(storage, &marker_bytes)?;
    if !recover_publication_with_storage(config, paths, storage, config_sha256)? {
        return Err("publication commit marker disappeared before recovery".to_owned());
    }
    Ok(())
}

fn recover_publication(config: &PlayConfig, config_sha256: &str) -> Result<bool, String> {
    validate_sha256(config_sha256)?;
    let paths = publication_paths(config)?;
    let storage = PublicationStorage::open(config, &paths, true)?;
    recover_publication_with_storage(config, &paths, &storage, config_sha256)
}

fn recover_publication_with_storage(
    config: &PlayConfig,
    paths: &PublicationPaths,
    storage: &PublicationStorage,
    config_sha256: &str,
) -> Result<bool, String> {
    recover_publication_with_storage_before_cleanup(config, paths, storage, config_sha256, |_| {
        Ok(())
    })
}

fn recover_publication_with_storage_before_cleanup(
    config: &PlayConfig,
    paths: &PublicationPaths,
    storage: &PublicationStorage,
    config_sha256: &str,
    before_cleanup: impl FnOnce(&PublicationStorage) -> Result<(), String>,
) -> Result<bool, String> {
    recover_publication_with_storage_cleanup_hooks(
        config,
        paths,
        storage,
        config_sha256,
        before_cleanup,
        |_| Ok(()),
    )
}

#[expect(
    clippy::too_many_lines,
    reason = "the recovery transaction keeps all publication and final-verification steps adjacent"
)]
fn recover_publication_with_storage_cleanup_hooks(
    _config: &PlayConfig,
    _paths: &PublicationPaths,
    storage: &PublicationStorage,
    config_sha256: &str,
    before_cleanup: impl FnOnce(&PublicationStorage) -> Result<(), String>,
    after_cleanup: impl FnOnce(&PublicationStorage) -> Result<(), String>,
) -> Result<bool, String> {
    if storage.marker.kind()?.is_none() {
        if storage.marker_pending.kind()?.is_some() {
            validate_publication_path_identities(storage)?;
            let (_, pending_marker, pending_file) =
                read_publication_marker(storage, &storage.marker_pending, config_sha256)?;
            storage.marker_pending.link_new_to(&storage.marker)?;
            drop((pending_marker, pending_file));
        } else if storage.csa_pending.kind()?.is_some()
            || storage.decision_pending.kind()?.is_some()
        {
            let both_pending =
                storage.csa_pending.kind()?.is_some() && storage.decision_pending.kind()?.is_some();
            if !both_pending {
                return Err(
                    "incomplete human-play pending pair requires manual inspection".to_owned(),
                );
            }
            validate_publication_path_identities(storage)?;
            let csa = storage.csa_pending.read(MAX_HUMAN_CSA_BYTES)?;
            let decisions = storage.decision_pending.read(MAX_HUMAN_DECISION_BYTES)?;
            validate_paired_evidence(&csa.bytes, &decisions.bytes, config_sha256)?;
            let marker = publication_marker(storage, config_sha256, &csa, &decisions)?;
            let marker_bytes = serde_json::to_vec(&marker)
                .map_err(|error| format!("cannot encode publication marker: {error}"))?;
            publish_marker_transaction(storage, &marker_bytes)?;
        } else {
            return Ok(false);
        }
    }
    validate_publication_path_identities(storage)?;
    let (marker, marker_artifact, marker_file) =
        read_publication_marker(storage, &storage.marker, config_sha256)?;
    let marker_pending_file = if storage.marker_pending.kind()?.is_some() {
        let (_, pending_artifact, pending_file) =
            read_publication_marker(storage, &storage.marker_pending, config_sha256)?;
        if pending_artifact.bytes != marker_artifact.bytes {
            return Err("pending and committed publication markers differ".to_owned());
        }
        Some(pending_file)
    } else {
        None
    };
    let csa_pending_file = recover_one_publication(
        &storage.output,
        &storage.csa_pending,
        &marker.csa_sha256,
        marker.csa_size,
        MAX_HUMAN_CSA_BYTES,
    )?;
    let decision_pending_file = recover_one_publication(
        &storage.decision,
        &storage.decision_pending,
        &marker.decision_sha256,
        marker.decision_size,
        MAX_HUMAN_DECISION_BYTES,
    )?;
    let (csa_artifact, csa_file) = storage.output.read_retained(MAX_HUMAN_CSA_BYTES)?;
    let (decision_artifact, decision_file) = storage
        .decision
        .read_retained(MAX_HUMAN_DECISION_BYTES)?;
    validate_paired_evidence(&csa_artifact.bytes, &decision_artifact.bytes, config_sha256)?;
    let csa_sha256: [u8; 32] = Sha256::digest(&csa_artifact.bytes).into();
    let decision_sha256: [u8; 32] = Sha256::digest(&decision_artifact.bytes).into();
    before_cleanup(storage)?;
    if let Some(file) = csa_pending_file.as_ref() {
        storage.csa_pending.remove_retained(file)?;
    }
    if let Some(file) = decision_pending_file.as_ref() {
        storage.decision_pending.remove_retained(file)?;
    }
    if let Some(file) = marker_pending_file.as_ref() {
        storage.marker_pending.remove_retained(file)?;
    }
    storage.marker.remove_retained(&marker_file)?;
    after_cleanup(storage)?;
    validate_publication_path_identities(storage)?;
    storage
        .output
        .directory
        .sync()
        .map_err(|error| format!("cannot sync final CSA directory: {error}"))?;
    storage
        .decision
        .directory
        .sync()
        .map_err(|error| format!("cannot sync final decision directory: {error}"))?;
    storage
        .output
        .directory
        .verify_retained_link_content(
            &storage.output.name,
            &csa_file,
            &csa_sha256,
            csa_artifact.size,
        )
        .map_err(|error| format!("final CSA changed during publication recovery: {error}"))?;
    storage
        .decision
        .directory
        .verify_retained_link_content(
            &storage.decision.name,
            &decision_file,
            &decision_sha256,
            decision_artifact.size,
        )
        .map_err(|error| format!("final decision log changed during publication recovery: {error}"))?;
    Ok(true)
}

fn publish_marker_transaction(
    storage: &PublicationStorage,
    marker_bytes: &[u8],
) -> Result<(), String> {
    validate_publication_path_identities(storage)?;
    storage.marker_pending.publish_new_atomic(marker_bytes)?;
    validate_publication_path_identities(storage)?;
    storage.marker_pending.link_new_to(&storage.marker)
}

fn publication_marker(
    storage: &PublicationStorage,
    config_sha256: &str,
    csa: &FileArtifact,
    decisions: &FileArtifact,
) -> Result<PublicationMarker, String> {
    Ok(PublicationMarker {
        schema: "phase6_human_publication/v1".to_owned(),
        config_sha256: config_sha256.to_owned(),
        csa_target: anchored_publication_path_text(&storage.output)?,
        decision_target: anchored_publication_path_text(&storage.decision)?,
        csa_pending: anchored_publication_path_text(&storage.csa_pending)?,
        decision_pending: anchored_publication_path_text(&storage.decision_pending)?,
        csa_sha256: csa.sha256.clone(),
        csa_size: csa.size,
        decision_sha256: decisions.sha256.clone(),
        decision_size: decisions.size,
    })
}

fn read_publication_marker(
    storage: &PublicationStorage,
    file: &AnchoredPublicationFile,
    config_sha256: &str,
) -> Result<(PublicationMarker, FileArtifact, AnchoredFile), String> {
    let (artifact, retained) = file.read_retained(64 * 1024)?;
    let marker: PublicationMarker = serde_json::from_slice(&artifact.bytes)
        .map_err(|error| format!("invalid human-play publication marker: {error}"))?;
    let canonical = serde_json::to_vec(&marker)
        .map_err(|error| format!("cannot canonicalize publication marker: {error}"))?;
    if canonical != artifact.bytes {
        return Err("human-play publication marker is not canonical".to_owned());
    }
    if marker.schema != "phase6_human_publication/v1" {
        return Err("human-play publication marker schema mismatch".to_owned());
    }
    validate_sha256(&marker.config_sha256)?;
    if marker.config_sha256 != config_sha256 {
        return Err("human-play publication configuration mismatch".to_owned());
    }
    validate_sha256(&marker.csa_sha256)?;
    validate_sha256(&marker.decision_sha256)?;
    if marker.csa_size == 0
        || marker.csa_size > MAX_HUMAN_CSA_BYTES
        || marker.decision_size == 0
        || marker.decision_size > MAX_HUMAN_DECISION_BYTES
    {
        return Err("human-play publication marker has an invalid artifact size".to_owned());
    }
    if marker.csa_target != anchored_publication_path_text(&storage.output)?
        || marker.decision_target != anchored_publication_path_text(&storage.decision)?
        || marker.csa_pending != anchored_publication_path_text(&storage.csa_pending)?
        || marker.decision_pending != anchored_publication_path_text(&storage.decision_pending)?
    {
        return Err("human-play publication marker path identity mismatch".to_owned());
    }
    Ok((marker, artifact, retained))
}

fn recover_one_publication(
    target: &AnchoredPublicationFile,
    pending: &AnchoredPublicationFile,
    expected_sha256: &str,
    expected_size: u64,
    maximum_size: u64,
) -> Result<Option<AnchoredFile>, String> {
    if expected_size == 0 || expected_size > maximum_size {
        return Err("publication artifact size is outside its defensive limit".to_owned());
    }
    let pending_artifact = if pending.kind()?.is_some() {
        let (artifact, file) = pending.read_retained(expected_size)?;
        if artifact.sha256 != expected_sha256 || artifact.size != expected_size {
            return Err(format!(
                "pending human-play artifact identity mismatch: {}",
                pending.display.display()
            ));
        }
        Some((artifact, file))
    } else {
        None
    };
    if target.kind()?.is_some() {
        let artifact = target.read(expected_size)?;
        if artifact.sha256 != expected_sha256 || artifact.size != expected_size {
            return Err(format!(
                "published human-play artifact identity mismatch: {}",
                target.display.display()
            ));
        }
        return Ok(pending_artifact.map(|(_, file)| file));
    }
    let (_, file) = pending_artifact.ok_or_else(|| {
        format!(
            "pending human-play artifact is missing: {}",
            pending.display.display()
        )
    })?;
    pending.link_new_to(target).map(|()| Some(file))
}

#[cfg(test)]
fn prepare_publication_parent(path: &Path) -> Result<(), String> {
    AnchoredDir::open_parent(path, true)
        .map_err(|error| format!("cannot create publication directory: {error}"))?;
    Ok(())
}

#[cfg(test)]
fn absolute_publication_path(path: &Path) -> Result<PathBuf, String> {
    let (directory, name) = AnchoredDir::open_parent(path, false)
        .map_err(|error| format!("cannot resolve publication directory: {error}"))?;
    Ok(directory.display().join(name))
}

fn anchored_publication_path_text(file: &AnchoredPublicationFile) -> Result<String, String> {
    file.display
        .to_str()
        .map(str::to_owned)
        .ok_or_else(|| "human-play publication path must be valid UTF-8".to_owned())
}

#[cfg(test)]
fn write_pending(path: &Path, bytes: &[u8]) -> Result<(), String> {
    AnchoredPublicationFile::from_path(path, false)?.publish_new_atomic(bytes)
}

#[cfg(test)]
fn publish_new_atomic(path: &Path, bytes: &[u8]) -> Result<(), String> {
    AnchoredPublicationFile::from_path(path, false)?.publish_new_atomic(bytes)
}

#[cfg(test)]
mod tests {
    use std::{
        collections::BTreeMap,
        io::{Cursor, Write},
        path::{Path, PathBuf},
    };

    use open_shogi_core::{
        CsaGame, CsaResultValidation, CsaSpecialMove, Position, Side, to_csa_game, to_sfen,
        to_usi_move,
    };
    use base64::Engine as _;
    use flate2::read::GzDecoder;
    use serde_json::Value;
    use sha2::Digest;

    use super::{
        ArenaAnalysisEnvelope, ArenaResults, Budget, DecisionEvent, GenerationPolicy,
        ModelRegistry, OpeningProfile, Phase3SplitPolicy, Phase6StartPosition, PlayConfig,
        PlayProfile, PromotionDecision,
        PromotionEvidence, PromotionWilsonInterval, PublicationMarker, PublicationPaths,
        PublicationStorage,
        RegistryArtifact, RegistryVerificationBudget,
        absolute_publication_path, advance_clock_after_move, analyze_arena_results,
        contained_artifact_path,
        derive_promotion_decision, deserialize_closed_json, encode_decision_log, encode_record,
        derive_paired_job_seed, handcrafted_profile, human_play_config_record, parse_arguments,
        parse_generation_policy, parse_phase6_selfplay_config, parse_unique_json,
        parse_unique_json_with_limits, phase3_game_split, phase6_start_position_identity,
        play_config_sha256,
        prepare_publication_parent, publication_paths, publish_human_artifacts,
        publish_human_artifacts_with_storage, publish_new_atomic, python_canonical_sha256,
        read_registry, recover_publication,
        recover_publication_with_storage_before_cleanup,
        recover_publication_with_storage_cleanup_hooks, repository_root_and_registry,
        resolve_registry_profile, run_interactive, select_challenger_model_id, sha256_bytes,
        validate_model_registry, validate_paired_evidence, validate_publication_path_identities,
        validate_registry_champion_transition, verify_registry_artifacts,
        verify_registry_artifacts_with_budget, write_pending,
        write_python_canonical_json,
    };

    type Phase3PositionMutation = fn(&mut Vec<serde_json::Value>);

    #[test]
    fn play_arguments_are_bounded() {
        let defaults = parse_arguments(&[]).unwrap();
        assert!(matches!(defaults.budget, Budget::Casual));
        assert_eq!(defaults.profile, PlayProfile::OverallChampion);
        assert_eq!(defaults.safety_margin_ms, 50);
        assert!(parse_arguments(&["--depth".into(), "0".into()]).is_err());
        assert!(parse_arguments(&["--nodes".into(), "1000000001".into()]).is_err());
        assert!(parse_arguments(&["--movetime-ms".into(), "3600001".into()]).is_err());
        assert!(parse_arguments(&["--max-plies".into(), "10001".into()]).is_err());
        assert!(
            parse_arguments(&["--safety-margin-ms".into(), "1001".into()]).is_err()
        );
        assert!(
            parse_arguments(&[
                "--nodes".into(),
                "1".into(),
                "--movetime-ms".into(),
                "1".into()
            ])
            .is_err()
        );
        let clock = parse_arguments(&[
            "--black-time-ms".into(),
            "60000".into(),
            "--white-time-ms".into(),
            "50000".into(),
            "--byoyomi-ms".into(),
            "1000".into(),
            "--black-increment-ms".into(),
            "100".into(),
            "--white-increment-ms".into(),
            "200".into(),
        ])
        .unwrap();
        let Budget::Clock(clock) = clock.budget else {
            panic!("clock options did not select clock mode");
        };
        assert_eq!(clock.black_time, 60_000);
        assert_eq!(clock.white_time, 50_000);
        assert_eq!(clock.byoyomi, Some(1_000));
        assert_eq!(clock.black_increment, Some(100));
        assert_eq!(clock.white_increment, Some(200));
        assert!(
            parse_arguments(&[
                "--black-time-ms".into(),
                "1000".into(),
                "--nodes".into(),
                "1".into(),
            ])
            .is_err()
        );
        assert!(
            parse_arguments(&["--depth".into(), "1".into(), "--depth".into(), "2".into(),])
                .is_err()
        );
        assert!(
            parse_arguments(&[
                "--profile".into(),
                "challenger".into(),
                "--registry".into(),
                "registry.json".into(),
                "--model".into(),
                "ignored.osaval".into(),
            ])
            .is_err()
        );
        assert!(
            parse_arguments(&[
                "--profile".into(),
                "neural-float".into(),
                "--model".into(),
                "model.osaval".into(),
            ])
            .is_err()
        );
    }

    #[test]
    fn native_clock_debits_the_ai_side_and_applies_its_increment() {
        let mut budget = Budget::Clock(super::ClockBudget {
            black_time: 10_000,
            white_time: 20_000,
            byoyomi: Some(1_000),
            black_increment: Some(250),
            white_increment: Some(500),
        });
        advance_clock_after_move(&mut budget, Side::Black, 1_200);
        advance_clock_after_move(&mut budget, Side::White, 2_000);
        let Budget::Clock(clock) = budget else {
            panic!("clock mode changed unexpectedly");
        };
        assert_eq!(clock.black_time, 9_050);
        assert_eq!(clock.white_time, 18_500);
    }

    #[test]
    fn paired_plan_git_commit_is_canonical_lowercase_hexadecimal() {
        assert!(super::validate_hex_object_id("0123456", "test commit").is_ok());
        assert!(super::validate_hex_object_id("012345A", "test commit").is_err());
        assert!(super::validate_hex_object_id("012345g", "test commit").is_err());
        assert!(super::validate_hex_object_id("012345", "test commit").is_err());
    }

    #[test]
    fn paired_command_numbers_have_one_unsigned_decimal_spelling() {
        for (value, accepted) in [
            ("0", true),
            ("64", true),
            ("08", false),
            ("00", false),
            ("+0", false),
            ("-0", false),
            ("18446744073709551615", false),
        ] {
            let options = BTreeMap::from([("--depth".to_owned(), value.to_owned())]);
            assert_eq!(
                super::parse_ascii_decimal_option(&options, "--depth", 0, 64).is_ok(),
                accepted,
                "unexpected canonical-decimal result for {value}"
            );
        }
        let options = BTreeMap::from([("--depth".to_owned(), "65".to_owned())]);
        assert!(super::parse_ascii_decimal_option(&options, "--depth", 0, 64).is_err());
    }

    #[test]
    fn canonical_json_numbers_match_python_spelling_at_format_boundaries() {
        let value = serde_json::json!({
            "values": [
                -0.0,
                1e-7,
                1e-6,
                1e-5,
                1e-4,
                1e15,
                1e16,
                1.234_567_890_123_456_7,
                f64::from_bits(0xc2e9_1477_f34d_5eb4),
                f64::from_bits(0xc2a3_d22b_7888_7320),
                f64::from_bits(0x4301_9a33_f89d_5a32),
                5e-324,
                1.797_693_134_862_315_7e308
            ]
        });
        let mut encoded = Vec::new();
        write_python_canonical_json(&value, &mut encoded).unwrap();

        assert_eq!(
            String::from_utf8(encoded.clone()).unwrap(),
            r#"{"values":[-0.0,1e-07,1e-06,1e-05,0.0001,1000000000000000.0,1e+16,1.2345678901234567,-220605619792629.62,-10896696689721.562,619327826144070.2,5e-324,1.7976931348623157e+308]}"#
        );
        assert_eq!(
            sha256_bytes(&encoded),
            "ba6166f2f972a8a617a07e6d63fa2931026fd8d163c385a7b70781dbb622957d"
        );

        let parsed = parse_unique_json(
            br#"{"value":12308.569780248597}"#,
            "exact float-roundtrip regression",
        )
        .unwrap();
        assert_eq!(
            parsed["value"].as_f64().unwrap().to_bits(),
            12_308.569_780_248_597_f64.to_bits()
        );
        assert_eq!(
            python_canonical_sha256(&parsed).unwrap(),
            "1272f5657d7c813ff5ed21f12af5e0b29d5d1c470dd1579a08ee77eff97e0a6a"
        );
    }

    struct SharedOutcomeFixture {
        name: &'static str,
        results: &'static [u8],
        analysis: &'static [u8],
        decision: &'static [u8],
    }

    const SHARED_FIXTURE_INDEX: &[u8] = include_bytes!(
        "../../../tests/python/selfplay/fixtures/contracts/fixture-index.json"
    );
    const SHARED_POLICY: &[u8] =
        include_bytes!("../../../configs/generation/phase6_promotion.toml");
    const SHARED_REGISTRY_CHAIN: &[u8] = include_bytes!(
        "../../../tests/python/selfplay/fixtures/registry-chain/registry-chain.json.gz.b64"
    );
    const SHARED_OUTCOMES: [SharedOutcomeFixture; 3] = [
        SharedOutcomeFixture {
            name: "promoted",
            results: include_bytes!(
                "../../../tests/python/selfplay/fixtures/contracts/promoted-results.json"
            ),
            analysis: include_bytes!(
                "../../../tests/python/selfplay/fixtures/contracts/promoted-analysis.json"
            ),
            decision: include_bytes!(
                "../../../tests/python/selfplay/fixtures/contracts/promoted-decision.json"
            ),
        },
        SharedOutcomeFixture {
            name: "rejected",
            results: include_bytes!(
                "../../../tests/python/selfplay/fixtures/contracts/rejected-results.json"
            ),
            analysis: include_bytes!(
                "../../../tests/python/selfplay/fixtures/contracts/rejected-analysis.json"
            ),
            decision: include_bytes!(
                "../../../tests/python/selfplay/fixtures/contracts/rejected-decision.json"
            ),
        },
        SharedOutcomeFixture {
            name: "inconclusive",
            results: include_bytes!(
                "../../../tests/python/selfplay/fixtures/contracts/inconclusive-results.json"
            ),
            analysis: include_bytes!(
                "../../../tests/python/selfplay/fixtures/contracts/inconclusive-analysis.json"
            ),
            decision: include_bytes!(
                "../../../tests/python/selfplay/fixtures/contracts/inconclusive-decision.json"
            ),
        },
    ];

    fn assert_shared_canonical_numbers(index: &Value) {
        for number in index["canonicalNumbers"].as_array().unwrap() {
            let input = number["inputJson"].as_str().unwrap();
            let value = parse_unique_json(input.as_bytes(), "shared canonical number").unwrap();
            let mut canonical = Vec::new();
            write_python_canonical_json(&value, &mut canonical).unwrap();
            assert_eq!(
                String::from_utf8(canonical.clone()).unwrap(),
                number["expectedCanonical"].as_str().unwrap(),
                "canonical spelling differs for {}",
                number["name"].as_str().unwrap()
            );
            assert_eq!(
                sha256_bytes(&canonical),
                number["expectedSha256"].as_str().unwrap(),
                "canonical digest differs for {}",
                number["name"].as_str().unwrap()
            );
        }
    }

    fn assert_shared_outcome(
        index: &Value,
        policy: &GenerationPolicy,
        fixture: &SharedOutcomeFixture,
    ) {
        let expected_index = &index["outcomes"][fixture.name];
        for (role, bytes) in [
            ("results", fixture.results),
            ("analysis", fixture.analysis),
            ("decision", fixture.decision),
        ] {
            assert_eq!(
                sha256_bytes(bytes),
                expected_index[role]["sha256"].as_str().unwrap(),
                "raw {} {role} digest differs",
                fixture.name
            );
            assert_eq!(
                bytes.len() as u64,
                expected_index[role]["size"].as_u64().unwrap(),
                "raw {} {role} size differs",
                fixture.name
            );
        }

        let results_value =
            parse_unique_json(fixture.results, "shared Python arena results").unwrap();
        let results: ArenaResults =
            deserialize_closed_json(&results_value, "shared Python arena results").unwrap();
        let analysis_value =
            parse_unique_json(fixture.analysis, "shared Python arena analysis").unwrap();
        let analysis: ArenaAnalysisEnvelope =
            deserialize_closed_json(&analysis_value, "shared Python arena analysis").unwrap();
        let expected_analysis = analyze_arena_results(&results, analysis.results.clone()).unwrap();
        assert_eq!(
            analysis, expected_analysis,
            "{} analysis differs",
            fixture.name
        );

        assert_eq!(analysis.plan.path, PathBuf::from("artifacts/arena-plan.json"));
        assert_eq!(analysis.execution.path, PathBuf::from("artifacts/arena-execution.json"));
        assert!(analysis.plan.size > 0);
        assert!(analysis.execution.size > 0);

        let decision_value =
            parse_unique_json(fixture.decision, "shared Python promotion decision").unwrap();
        let decision: PromotionDecision =
            deserialize_closed_json(&decision_value, "shared Python promotion decision").unwrap();
        assert_eq!(decision.decision, fixture.name);
        assert_eq!(decision.policy.sha256, sha256_bytes(SHARED_POLICY));
        assert_eq!(decision.policy.size, SHARED_POLICY.len() as u64);
        assert_eq!(decision.policy_sha256, index["policySha256"]);
        assert_eq!(
            decision,
            derive_promotion_decision(&analysis, &decision, policy).unwrap(),
            "{} decision differs",
            fixture.name
        );
    }

    #[test]
    fn shared_python_contract_fixtures_match_rust_derivation_and_canonicalization() {
        assert_eq!(
            sha256_bytes(SHARED_FIXTURE_INDEX),
            "fb0be539a9cf7424ad4e37cc12fbcc70490c78508b908757480dfa804453842f"
        );
        let index = parse_unique_json(SHARED_FIXTURE_INDEX, "shared fixture index").unwrap();
        assert_eq!(
            index["schema"],
            "phase6_cross_runtime_contract_fixtures/v1"
        );
        assert_eq!(
            sha256_bytes(SHARED_POLICY),
            index["policy"]["sha256"].as_str().unwrap()
        );
        assert_eq!(
            SHARED_POLICY.len() as u64,
            index["policy"]["size"].as_u64().unwrap()
        );
        let policy = parse_generation_policy(SHARED_POLICY).unwrap();
        assert_eq!(
            python_canonical_sha256(&serde_json::to_value(&policy).unwrap()).unwrap(),
            index["policySha256"].as_str().unwrap()
        );

        assert_shared_canonical_numbers(&index);
        for fixture in &SHARED_OUTCOMES {
            assert_shared_outcome(&index, &policy, fixture);
        }
    }

    fn decode_fixture_base64(bytes: &[u8]) -> Vec<u8> {
        let compact = bytes
            .iter()
            .copied()
            .filter(|byte| !byte.is_ascii_whitespace())
            .collect::<Vec<_>>();
        base64::engine::general_purpose::STANDARD
            .decode(compact)
            .unwrap()
    }

    fn materialize_shared_registry_chain(root: &Path) -> PathBuf {
        assert_eq!(SHARED_REGISTRY_CHAIN.len(), 105_093);
        assert_eq!(
            sha256_bytes(SHARED_REGISTRY_CHAIN),
            "6588f3f309acd63004402b623650291075c4395a505a0af4654a0cec488aea3f"
        );
        let compressed = decode_fixture_base64(SHARED_REGISTRY_CHAIN);
        assert_eq!(compressed.len(), 77_794);
        assert_eq!(
            sha256_bytes(&compressed),
            "f49067c4fea8528d229ad08f53706868abeaa040788278c5ba214c52f08814ab"
        );
        let mut decoder = GzDecoder::new(compressed.as_slice());
        let mut envelope_bytes = Vec::new();
        std::io::Read::read_to_end(&mut decoder, &mut envelope_bytes).unwrap();
        assert_eq!(envelope_bytes.len(), 417_383);
        assert_eq!(
            sha256_bytes(&envelope_bytes),
            "53a6dc9111aac0005efade3f8000d8741479e8d2f442dfbfcae6f949e7b6d176"
        );
        let envelope = parse_unique_json(&envelope_bytes, "shared registry chain").unwrap();
        assert_eq!(
            envelope["schema"],
            "phase6_registry_acceptance_fixture/v1"
        );
        assert_eq!(envelope["expectedDecision"], "inconclusive");
        assert_eq!(envelope["files"].as_array().unwrap().len(), 202);
        assert_eq!(envelope["blobs"].as_object().unwrap().len(), 95);
        let blobs = envelope["blobs"].as_object().unwrap();
        let mut seen = std::collections::BTreeSet::new();
        let mut used = std::collections::BTreeSet::new();
        let mut total = 0_u64;
        for file in envelope["files"].as_array().unwrap() {
            let path = file["path"].as_str().unwrap();
            let sha256 = file["sha256"].as_str().unwrap();
            let size = file["size"].as_u64().unwrap();
            assert!(seen.insert(path));
            assert!(size <= 8 * 1024 * 1024);
            let bytes = decode_fixture_base64(blobs[sha256].as_str().unwrap().as_bytes());
            assert_eq!(bytes.len() as u64, size);
            assert_eq!(sha256_bytes(&bytes), sha256);
            total += size;
            assert!(total <= 16 * 1024 * 1024);
            let destination = root.join(path);
            std::fs::create_dir_all(destination.parent().unwrap()).unwrap();
            std::fs::write(destination, bytes).unwrap();
            used.insert(sha256);
        }
        assert_eq!(used.len(), blobs.len());
        std::fs::create_dir_all(root.join(".git")).unwrap();
        root.join(envelope["registry"]["path"].as_str().unwrap())
    }

    #[test]
    fn phase3_start_selection_is_rebuilt_from_the_verified_gzip_source() {
        let root = temporary_record_path("phase3-start-rebuild").with_extension("");
        let registry_path = materialize_shared_registry_chain(&root);
        let (_, storage, _) = read_registry(&registry_path).unwrap();
        let starts_value = parse_unique_json(
            &std::fs::read(root.join("artifacts/phase6-inputs/start-positions.json")).unwrap(),
            "shared start positions",
        )
        .unwrap();
        let mut starts: super::Phase6StartPositions =
            deserialize_closed_json(&starts_value, "shared start positions").unwrap();
        let manifest_value = parse_unique_json(
            &std::fs::read(root.join("data/phase3/manifest.json")).unwrap(),
            "shared Phase 3 manifest",
        )
        .unwrap();
        let mut manifest: super::Phase3DatasetManifest =
            deserialize_closed_json(&manifest_value, "shared Phase 3 manifest").unwrap();

        let selected = starts.positions.first().unwrap().clone();
        let source_path = root.join(&starts.source_positions.path);
        let compressed = std::fs::read(&source_path).unwrap();
        let mut decoder = GzDecoder::new(compressed.as_slice());
        let mut source = Vec::new();
        std::io::Read::read_to_end(&mut decoder, &mut source).unwrap();
        let mut changed_source = Vec::new();
        let mut changed = false;
        for line in source.split(|byte| *byte == b'\n').filter(|line| !line.is_empty()) {
            let mut row = parse_unique_json(line, "shared Phase 3 row").unwrap();
            if row["canonicalSha256"].as_str() == Some(selected.source_game_sha256.as_str())
                && row["positionIndex"].as_u64() == Some(selected.position_index)
            {
                row["eligible"] = serde_json::Value::Bool(false);
                changed = true;
            }
            write_python_canonical_json(&row, &mut changed_source).unwrap();
            changed_source.push(b'\n');
        }
        assert!(changed, "fixture did not contain the selected source position");
        let mut encoder = flate2::write::GzEncoder::new(
            Vec::new(),
            flate2::Compression::default(),
        );
        std::io::Write::write_all(&mut encoder, &changed_source).unwrap();
        let changed_compressed = encoder.finish().unwrap();
        std::fs::write(&source_path, &changed_compressed).unwrap();
        starts.source_positions.sha256 = sha256_bytes(&changed_compressed);
        starts.source_positions.size = changed_compressed.len() as u64;
        let artifact_name = starts
            .source_positions
            .path
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .to_owned();
        let recorded = manifest.artifacts.get_mut(&artifact_name).unwrap();
        recorded.sha256.clone_from(&starts.source_positions.sha256);
        recorded.size = starts.source_positions.size;

        super::validate_phase3_dataset_manifest(&starts, &manifest).unwrap();
        let mut observed = BTreeMap::new();
        let mut budget = RegistryVerificationBudget::with_limits(10_000, 100_000_000);
        super::verify_registry_artifact_cached(
            &storage,
            &starts.source_positions,
            &mut observed,
            &mut budget,
            None,
        )
        .unwrap();
        let error = super::validate_phase3_start_position_derivation(
            &storage,
            &starts,
            &manifest,
            &mut observed,
            &mut budget,
        )
        .unwrap_err();
        assert!(
            error.contains("deterministic Phase 3 selection")
                || error.contains("safe train start positions")
                || error.contains("safe validation start positions")
                || error.contains("position fields differ from canonical game replay"),
            "unexpected error: {error}"
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    fn read_fixture_phase3_rows(
        root: &Path,
        dataset: &RegistryArtifact,
    ) -> (super::Phase3DatasetManifest, Vec<super::Phase3GameRow>) {
        let manifest_value = parse_unique_json(
            &std::fs::read(root.join(&dataset.path)).unwrap(),
            "fixture Phase 3 manifest",
        )
        .unwrap();
        let manifest: super::Phase3DatasetManifest =
            deserialize_closed_json(&manifest_value, "fixture Phase 3 manifest").unwrap();
        let games_path = dataset.path.parent().unwrap().join("games-00000.jsonl.gz");
        let compressed = std::fs::read(root.join(games_path)).unwrap();
        let mut decoder = GzDecoder::new(compressed.as_slice());
        let mut jsonl = Vec::new();
        std::io::Read::read_to_end(&mut decoder, &mut jsonl).unwrap();
        let rows = jsonl
            .split(|byte| *byte == b'\n')
            .filter(|line| !line.is_empty())
            .map(|line| {
                let value = parse_unique_json(line, "fixture Phase 3 game").unwrap();
                deserialize_closed_json(&value, "fixture Phase 3 game").unwrap()
            })
            .collect();
        (manifest, rows)
    }

    fn mutated_phase3_position_error(label: &str, mutate: Phase3PositionMutation) -> String {
        let root = temporary_record_path(label).with_extension("");
        let (engine, engine_receipt) = write_fixture_engine_build(&root, "abcdef0");
        let selected = fixture_start_positions();
        let (starts_reference, _, dataset) =
            write_start_evidence(&root, &engine, &engine_receipt, &selected);
        let starts_value = parse_unique_json(
            &std::fs::read(root.join(starts_reference.path)).unwrap(),
            "fixture starts",
        )
        .unwrap();
        let mut starts: super::Phase6StartPositions =
            deserialize_closed_json(&starts_value, "fixture starts").unwrap();
        let manifest_value = parse_unique_json(
            &std::fs::read(root.join(&dataset.path)).unwrap(),
            "fixture manifest",
        )
        .unwrap();
        let mut manifest: super::Phase3DatasetManifest =
            deserialize_closed_json(&manifest_value, "fixture manifest").unwrap();
        let compressed = std::fs::read(root.join(&starts.source_positions.path)).unwrap();
        let mut decoder = GzDecoder::new(compressed.as_slice());
        let mut jsonl = Vec::new();
        std::io::Read::read_to_end(&mut decoder, &mut jsonl).unwrap();
        let mut rows = jsonl
            .split(|byte| *byte == b'\n')
            .filter(|line| !line.is_empty())
            .map(|line| parse_unique_json(line, "fixture position").unwrap())
            .collect::<Vec<_>>();
        mutate(&mut rows);
        let changed = gzip_fixture_jsonl(&rows);
        std::fs::write(root.join(&starts.source_positions.path), &changed).unwrap();
        starts.source_positions.sha256 = sha256_bytes(&changed);
        starts.source_positions.size = changed.len() as u64;
        let recorded = manifest
            .artifacts
            .get_mut("positions-00000.jsonl.gz")
            .unwrap();
        recorded.sha256.clone_from(&starts.source_positions.sha256);
        recorded.size = starts.source_positions.size;

        let storage = super::RegistryStorage {
            root: open_shogi_core::AnchoredDir::open_existing(&root).unwrap(),
            root_path: root.clone(),
            registry_relative: PathBuf::from("artifacts/unused-registry.json"),
        };
        let mut observed = BTreeMap::new();
        let mut budget = RegistryVerificationBudget::with_limits(10_000, 100_000_000);
        super::verify_registry_artifact_cached(
            &storage,
            &starts.source_positions,
            &mut observed,
            &mut budget,
            None,
        )
        .unwrap();
        let error = super::validate_phase3_start_position_derivation(
            &storage,
            &starts,
            &manifest,
            &mut observed,
            &mut budget,
        )
        .unwrap_err();
        std::fs::remove_dir_all(root).unwrap();
        error
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "the provenance regression covers the complete AobaZero adaptation boundary"
    )]
    fn phase3_raw_csa_must_canonicalize_to_the_retained_normalized_game() {
        let root = temporary_record_path("phase3-raw-canonical").with_extension("");
        let (engine, engine_receipt) = write_fixture_engine_build(&root, "abcdef0");
        let starts = fixture_start_positions();
        let (_, _, dataset) =
            write_start_evidence(&root, &engine, &engine_receipt, &starts);
        let (manifest, rows) = read_fixture_phase3_rows(&root, &dataset);
        assert_eq!(rows.len(), 20);

        let valid = rows[0].clone();
        assert!(valid.raw_csa.starts_with("' fixture acquisition comment"));
        assert!(!valid.raw_csa.lines().any(|line| line == "V3.0"));
        assert!(valid.raw_csa.lines().any(|line| line.contains(",v=")));
        assert!(valid.raw_csa.lines().any(|line| line.contains(",'")));
        assert!(valid.raw_csa.lines().any(|line| line == "T0.001"));
        super::validate_phase3_game_row(&valid, &manifest, "fixture raw CSA").unwrap();

        let mut wrong_size = valid.clone();
        wrong_size.raw_object.size += 1;
        assert!(
            super::validate_phase3_game_row(&wrong_size, &manifest, "fixture raw size").is_err()
        );

        let mut wrong_hash = valid.clone();
        wrong_hash.raw_object.sha256 = "0".repeat(64);
        wrong_hash.raw_object.object_path =
            PathBuf::from(format!("objects/sha256/00/{}", wrong_hash.raw_object.sha256));
        assert!(
            super::validate_phase3_game_row(&wrong_hash, &manifest, "fixture raw hash").is_err()
        );

        let mut wrong_canonical_hash = valid.clone();
        wrong_canonical_hash.canonical_sha256 = "0".repeat(64);
        wrong_canonical_hash.game_id = wrong_canonical_hash.canonical_sha256.clone();
        assert!(
            super::validate_phase3_game_row(
                &wrong_canonical_hash,
                &manifest,
                "fixture canonical hash"
            )
            .is_err()
        );

        let mut mismatched = valid.clone();
        mismatched.raw_csa.clone_from(&rows[1].raw_csa);
        mismatched.raw_object.sha256 = sha256_bytes(mismatched.raw_csa.as_bytes());
        mismatched.raw_object.size = mismatched.raw_csa.len() as u64;
        mismatched.raw_object.object_path = PathBuf::from(format!(
            "objects/sha256/{}/{}",
            &mismatched.raw_object.sha256[..2], mismatched.raw_object.sha256
        ));
        assert!(
            super::validate_phase3_game_row(&mismatched, &manifest, "fixture raw mismatch")
                .unwrap_err()
                .contains("canonical form of its raw CSA")
        );

        let mut invalid = valid;
        invalid.raw_csa = "not a CSA record\n".to_owned();
        invalid.raw_object.sha256 = sha256_bytes(invalid.raw_csa.as_bytes());
        invalid.raw_object.size = invalid.raw_csa.len() as u64;
        invalid.raw_object.object_path = PathBuf::from(format!(
            "objects/sha256/{}/{}",
            &invalid.raw_object.sha256[..2], invalid.raw_object.sha256
        ));
        assert!(
            super::validate_phase3_game_row(&invalid, &manifest, "fixture invalid raw")
                .unwrap_err()
                .contains("raw CSA is invalid")
        );

        for (name, raw) in [
            (
                "version after name",
                "N+Black\nV3.0\nPI\n+\n%TORYO\n".to_owned(),
            ),
            (
                "metadata after position",
                "PI\n$EVENT:late\n+\n%TORYO\n".to_owned(),
            ),
            (
                "player after position",
                "PI\nN+late\n+\n%TORYO\n".to_owned(),
            ),
            (
                "explicit position",
                "P1-KY-KE-GI-KI-OU-KI-GI-KE-KY\n+\n%TORYO\n".to_owned(),
            ),
            (
                "move before side",
                "PI\n+7776FU\n%TORYO\n".to_owned(),
            ),
            (
                "time after comment",
                "PI\n+\n+7776FU\n' interruption\nT1\n%TORYO\n".to_owned(),
            ),
            (
                "unknown annotation",
                "PI\n+\n+7776FU,T1\n%TORYO\n".to_owned(),
            ),
            (
                "non-printable annotation",
                format!("PI\n+\n+7776FU,v=bad{}\n%TORYO\n", '\u{7f}'),
            ),
            (
                "content after terminal",
                "PI\n+\n%TORYO\n+7776FU\n".to_owned(),
            ),
            (
                "unknown terminal",
                "PI\n+\n%UNKNOWN\n".to_owned(),
            ),
        ] {
            let mut rejected = rows[0].clone();
            rejected.raw_csa = raw;
            rejected.raw_object.sha256 = sha256_bytes(rejected.raw_csa.as_bytes());
            rejected.raw_object.size = rejected.raw_csa.len() as u64;
            rejected.raw_object.object_path = PathBuf::from(format!(
                "objects/sha256/{}/{}",
                &rejected.raw_object.sha256[..2], rejected.raw_object.sha256
            ));
            let error = super::validate_phase3_game_row(&rejected, &manifest, name).unwrap_err();
            assert!(error.contains("raw CSA is invalid"), "{name}: {error}");
        }
        let oversized_raw = format!(
            "'{}\nPI\n+\n%TORYO\n",
            "x".repeat(super::MAX_PHASE3_CSA_BYTES)
        );
        assert!(super::adapt_phase3_aobazero_csa(&oversized_raw).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn phase3_replay_accepts_legal_history_after_ordinary_repetition_and_classifies_perpetual_check() {
        let root = temporary_record_path("phase3-continued-repetition").with_extension("");
        let (engine, engine_receipt) = write_fixture_engine_build(&root, "abcdef0");
        let starts = fixture_start_positions();
        let (_, _, dataset) = write_start_evidence(&root, &engine, &engine_receipt, &starts);
        let (mut manifest, rows) = read_fixture_phase3_rows(&root, &dataset);
        manifest.source.adapter = "aobazero".to_owned();

        let initial = open_shogi_core::parse_sfen("4k4/5R3/9/9/9/9/9/9/K8 b - 1").unwrap();
        let mut moves = Vec::new();
        for _ in 0..3 {
            moves.extend(
                ["9i8h", "5a6a", "8h9i", "6a5a"]
                    .map(|notation| open_shogi_core::parse_usi_move(notation).unwrap()),
            );
        }
        for _ in 0..3 {
            moves.extend(
                ["4b5b", "5a4a", "5b4b", "4a5a"]
                    .map(|notation| open_shogi_core::parse_usi_move(notation).unwrap()),
            );
        }
        let normalized = to_csa_game(&CsaGame {
            version: "V3.0".to_owned(),
            black_name: None,
            white_name: None,
            metadata: Vec::new(),
            initial_position: initial.clone(),
            moves: moves.clone(),
            special_move: Some(CsaSpecialMove::PerpetualCheck),
            result_validation: CsaResultValidation::Verified,
        })
        .unwrap();
        let raw = format!("' continued history after ordinary repetition\n{normalized}");
        let canonical_sha256 = sha256_bytes(normalized.as_bytes());
        let raw_sha256 = sha256_bytes(raw.as_bytes());
        let mut row = rows[0].clone();
        row.game_id.clone_from(&canonical_sha256);
        row.canonical_sha256 = canonical_sha256;
        row.raw_csa = raw;
        row.normalized_csa = normalized;
        row.raw_object.sha256.clone_from(&raw_sha256);
        row.raw_object.size = row.raw_csa.len() as u64;
        row.raw_object.object_path = PathBuf::from(format!(
            "objects/sha256/{}/{raw_sha256}",
            &raw_sha256[..2]
        ));
        row.initial_sfen = to_sfen(&initial);
        row.usi_moves = moves.iter().copied().map(to_usi_move).collect();
        row.ply_count = moves.len() as u64;
        row.position_count = row.ply_count + 1;
        row.flags.short = false;
        row.flags.long = false;
        row.outcome = "white_win".to_owned();
        row.terminal_reason = Some("OUTE_SENNICHITE".to_owned());
        row.result_validation = "verified".to_owned();
        row.players.black.name = None;
        row.players.white.name = None;
        row.split = super::phase3_game_split(&row.canonical_sha256, &manifest.config.split).unwrap();

        let expected = super::validate_phase3_game_row(
            &row,
            &manifest,
            "continued repetition fixture",
        )
        .unwrap();
        assert_eq!(expected.len(), moves.len() + 1);
        assert_eq!(expected.last().unwrap().outcome, "white_win");
        assert_eq!(
            expected.last().unwrap().terminal_reason.as_deref(),
            Some("OUTE_SENNICHITE")
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn phase3_accepts_only_the_historical_verified_repetition_outcome_upgrade() {
        assert!(super::phase3_outcome_matches(
            "unknown",
            "draw",
            Some("SENNICHITE"),
            "verified"
        ));
        assert!(super::phase3_outcome_matches(
            "draw",
            "draw",
            Some("SENNICHITE"),
            "verified"
        ));
        assert!(!super::phase3_outcome_matches(
            "unknown",
            "draw",
            Some("HIKIWAKE"),
            "verified"
        ));
        assert!(!super::phase3_outcome_matches(
            "unknown",
            "draw",
            Some("SENNICHITE"),
            "external_condition"
        ));
        assert!(!super::phase3_outcome_matches(
            "unknown",
            "black_win",
            Some("SENNICHITE"),
            "verified"
        ));
    }

    #[test]
    fn phase3_position_stream_rejects_illegal_noncanonical_gapped_permuted_and_false_final_rows() {
        let cases: [(&str, Phase3PositionMutation); 9] = [
            ("phase3-position-illegal", |rows| {
                rows[0]["moveUsi"] = serde_json::json!("9a9b");
            }),
            ("phase3-position-wrong-next", |rows| {
                rows[0]["nextSfen"] = rows[0]["sfen"].clone();
            }),
            ("phase3-position-split", |rows| {
                rows[0]["split"] = if rows[0]["split"] == "train" {
                    serde_json::json!("validation")
                } else {
                    serde_json::json!("train")
                };
            }),
            ("phase3-position-gap", |rows| {
                rows[1]["positionIndex"] = serde_json::json!(2);
                rows[1]["remainingPlies"] = serde_json::json!(1);
            }),
            ("phase3-position-duplicate", |rows| {
                rows.insert(1, rows[0].clone());
            }),
            ("phase3-position-final", |rows| {
                let final_index = rows
                    .iter()
                    .position(|row| row["moveUsi"].is_null())
                    .unwrap();
                rows[final_index]["nextSfen"] = rows[final_index]["sfen"].clone();
            }),
            ("phase3-position-raw-permutation", |rows| {
                let first = rows[0]["rawSha256"].clone();
                rows[0]["rawSha256"] = rows[4]["rawSha256"].clone();
                rows[4]["rawSha256"] = first;
            }),
            ("phase3-position-terminal", |rows| {
                rows[0]["terminalReason"] = serde_json::json!("TORYO");
            }),
            ("phase3-position-noncanonical", |rows| {
                let sfen = rows[0]["sfen"].as_str().unwrap();
                let prefix = sfen.rsplit_once(' ').unwrap().0;
                rows[0]["sfen"] = serde_json::json!(format!("{prefix} 01"));
            }),
        ];
        for (label, mutate) in cases {
            let error = mutated_phase3_position_error(label, mutate);
            assert!(!error.is_empty(), "{label} unexpectedly validated");
        }
    }

    #[test]
    fn phase3_jsonl_rows_require_python_compact_sorted_canonical_spelling() {
        for raw in [
            b"{\"b\":1,\"a\":2}\n".as_slice(),
            b"{\"a\": 1}\n",
            b"{\"a\":1e+00}\n",
            b"{\"a\":\"\\u0061\"}\n",
        ] {
            let value = parse_unique_json(raw, "noncanonical Phase 3 row").unwrap();
            assert!(
                super::validate_phase3_canonical_json_line(
                    raw,
                    &value,
                    "noncanonical Phase 3 row"
                )
                .is_err()
            );
        }
        let canonical = b"{\"a\":1,\"b\":2}\n";
        let value = parse_unique_json(canonical, "canonical Phase 3 row").unwrap();
        super::validate_phase3_canonical_json_line(
            canonical,
            &value,
            "canonical Phase 3 row",
        )
        .unwrap();
    }

    #[test]
    fn phase3_game_stream_accepts_a_valid_row_larger_than_the_position_line_limit() {
        let root = temporary_record_path("phase3-large-game-line").with_extension("");
        let (engine, engine_receipt) = write_fixture_engine_build(&root, "abcdef0");
        let selected = fixture_start_positions();
        let (starts_reference, _, dataset) =
            write_start_evidence(&root, &engine, &engine_receipt, &selected);
        let starts_value = parse_unique_json(
            &std::fs::read(root.join(starts_reference.path)).unwrap(),
            "fixture starts",
        )
        .unwrap();
        let starts: super::Phase6StartPositions =
            deserialize_closed_json(&starts_value, "fixture starts").unwrap();
        let manifest_value = parse_unique_json(
            &std::fs::read(root.join(&dataset.path)).unwrap(),
            "fixture manifest",
        )
        .unwrap();
        let mut manifest: super::Phase3DatasetManifest =
            deserialize_closed_json(&manifest_value, "fixture manifest").unwrap();
        let games_path = dataset.path.parent().unwrap().join("games-00000.jsonl.gz");
        let compressed = std::fs::read(root.join(&games_path)).unwrap();
        let mut decoder = GzDecoder::new(compressed.as_slice());
        let mut jsonl = Vec::new();
        std::io::Read::read_to_end(&mut decoder, &mut jsonl).unwrap();
        let mut rows = jsonl
            .split(|byte| *byte == b'\n')
            .filter(|line| !line.is_empty())
            .map(|line| parse_unique_json(line, "fixture game").unwrap())
            .collect::<Vec<_>>();
        let old_raw_sha256 = rows[0]["rawObject"]["sha256"]
            .as_str()
            .unwrap()
            .to_owned();
        let old_raw = rows[0]["rawCsa"].as_str().unwrap();
        let comment_body = "x".repeat(900);
        let mut comment_prefix = String::new();
        for index in 0..80 {
            comment_prefix.push_str("' ");
            comment_prefix.push_str(&comment_body);
            comment_prefix.push(' ');
            comment_prefix.push_str(&index.to_string());
            comment_prefix.push('\n');
        }
        let enlarged_raw = format!("{comment_prefix}{old_raw}");
        let enlarged_sha256 = sha256_bytes(enlarged_raw.as_bytes());
        rows[0]["rawCsa"] = serde_json::json!(enlarged_raw);
        rows[0]["rawObject"]["sha256"] = serde_json::json!(enlarged_sha256);
        rows[0]["rawObject"]["size"] =
            serde_json::json!(rows[0]["rawCsa"].as_str().unwrap().len());
        rows[0]["rawObject"]["objectPath"] = serde_json::json!(format!(
            "objects/sha256/{}/{}",
            &enlarged_sha256[..2], enlarged_sha256
        ));
        let mut rendered_row = Vec::new();
        write_python_canonical_json(&rows[0], &mut rendered_row).unwrap();
        assert!(rendered_row.len() as u64 > super::MAX_PHASE3_LINE_BYTES);
        assert!(rendered_row.len() as u64 <= super::phase3_game_line_limit(&manifest.config).unwrap());

        manifest
            .raw_object_sha256
            .retain(|sha256| sha256 != &old_raw_sha256);
        manifest.raw_object_sha256.push(enlarged_sha256);
        manifest.raw_object_sha256.sort();
        let changed = gzip_fixture_jsonl(&rows);
        std::fs::write(root.join(&games_path), &changed).unwrap();
        let recorded = manifest
            .artifacts
            .get_mut("games-00000.jsonl.gz")
            .unwrap();
        recorded.sha256 = sha256_bytes(&changed);
        recorded.size = changed.len() as u64;

        let storage = super::RegistryStorage {
            root: open_shogi_core::AnchoredDir::open_existing(&root).unwrap(),
            root_path: root.clone(),
            registry_relative: PathBuf::from("artifacts/unused-registry.json"),
        };
        super::load_validated_phase3_games(
            &storage,
            &starts,
            &manifest,
            &mut BTreeMap::new(),
            &mut RegistryVerificationBudget::with_limits(10_000, 100_000_000),
        )
        .unwrap();
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn phase3_manifest_rejects_split_caps_artifact_and_source_contract_mutations() {
        let root = temporary_record_path("phase3-manifest-contract").with_extension("");
        let (engine, engine_receipt) = write_fixture_engine_build(&root, "abcdef0");
        let selected = fixture_start_positions();
        let (starts_reference, _, dataset) =
            write_start_evidence(&root, &engine, &engine_receipt, &selected);
        let starts_value = parse_unique_json(
            &std::fs::read(root.join(starts_reference.path)).unwrap(),
            "fixture starts",
        )
        .unwrap();
        let starts: super::Phase6StartPositions =
            deserialize_closed_json(&starts_value, "fixture starts").unwrap();
        let (manifest, _) = read_fixture_phase3_rows(&root, &dataset);
        super::validate_phase3_dataset_manifest(&starts, &manifest).unwrap();

        let mut bad_salt = manifest.clone();
        bad_salt.config.split.salt_sha256 = "0".repeat(64);
        assert!(super::validate_phase3_dataset_manifest(&starts, &bad_salt).is_err());

        let mut over_cap = manifest.clone();
        over_cap.config.max_games = over_cap.counts.games - 1;
        assert!(super::validate_phase3_dataset_manifest(&starts, &over_cap).is_err());

        let mut missing_report = manifest.clone();
        missing_report.artifacts.remove("normalization-report.json");
        assert!(super::validate_phase3_dataset_manifest(&starts, &missing_report).is_err());

        let mut extra_artifact = manifest.clone();
        extra_artifact.artifacts.insert(
            "extra.json".to_owned(),
            extra_artifact.artifacts["normalization-report.json"].clone(),
        );
        assert!(super::validate_phase3_dataset_manifest(&starts, &extra_artifact).is_err());

        let mut wrong_namespace = manifest.clone();
        wrong_namespace.dataset_id = "unrelated-dataset".to_owned();
        wrong_namespace.config.dataset_id = wrong_namespace.dataset_id.clone();
        assert!(super::validate_phase3_dataset_manifest(&starts, &wrong_namespace).is_err());

        let mut wrong_adapter = manifest;
        wrong_adapter.source.adapter = "generic_csa".to_owned();
        assert!(super::validate_phase3_dataset_manifest(&starts, &wrong_adapter).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "one provenance test exercises the complete evidence-catalog invariant surface"
    )]
    fn phase3_evidence_catalog_is_complete_unique_and_manifest_ordered() {
        let root = temporary_record_path("phase3-evidence-catalog").with_extension("");
        let (engine, engine_receipt) = write_fixture_engine_build(&root, "abcdef0");
        let starts = fixture_start_positions();
        let (starts_reference, _, dataset) =
            write_start_evidence(&root, &engine, &engine_receipt, &starts);
        let starts_value = parse_unique_json(
            &std::fs::read(root.join(starts_reference.path)).unwrap(),
            "fixture starts",
        )
        .unwrap();
        let starts: super::Phase6StartPositions =
            deserialize_closed_json(&starts_value, "fixture starts").unwrap();
        let (mut manifest, rows) = read_fixture_phase3_rows(&root, &dataset);
        let first = manifest.evidence_snapshots[0].clone();
        let mut later_retrieval = first.clone();
        later_retrieval.retrieved_at = "2026-08-09T00:00:01Z".to_owned();
        let second_sha256 = sha256_bytes(b"second fixture evidence");
        let second_id: super::Phase3EvidenceSnapshot = serde_json::from_value(serde_json::json!({
            "evidence_id": "fixture-evidence-second",
            "url": "https://example.invalid/fixture/evidence-second",
            "retrieved_at": "2026-08-09T00:00:00Z",
            "sha256": second_sha256,
            "size": 23,
            "content_type": "text/plain",
            "object_path": format!(
                "evidence/sha256/{}/{second_sha256}",
                &second_sha256[..2]
            )
        }))
        .unwrap();
        manifest.evidence_snapshots.push(later_retrieval.clone());
        manifest.evidence_snapshots.push(second_id.clone());
        manifest.evidence_snapshots.sort_by(|left, right| {
            super::phase3_evidence_identity(left)
                .unwrap()
                .cmp(&super::phase3_evidence_identity(right).unwrap())
        });
        super::validate_phase3_dataset_manifest(&starts, &manifest).unwrap();

        let mut complete = rows[0].clone();
        complete.license_decision.evidence_snapshots = vec![first.clone(), second_id.clone()];
        complete
            .license_decision
            .evidence_snapshots
            .sort_by(|left, right| {
                super::phase3_evidence_identity(left)
                    .unwrap()
                    .cmp(&super::phase3_evidence_identity(right).unwrap())
            });
        super::validate_phase3_game_row(&complete, &manifest, "complete evidence").unwrap();

        let mut later = complete.clone();
        later.license_decision.evidence_snapshots =
            vec![later_retrieval.clone(), second_id.clone()];
        later
            .license_decision
            .evidence_snapshots
            .sort_by(|left, right| {
                super::phase3_evidence_identity(left)
                    .unwrap()
                    .cmp(&super::phase3_evidence_identity(right).unwrap())
            });
        super::validate_phase3_game_row(&later, &manifest, "later evidence retrieval").unwrap();

        let mut missing_id = complete.clone();
        missing_id
            .license_decision
            .evidence_snapshots
            .retain(|snapshot| snapshot.evidence_id != second_id.evidence_id);
        assert!(
            super::validate_phase3_game_row(&missing_id, &manifest, "missing evidence ID")
                .is_err()
        );

        let mut duplicate_id = complete.clone();
        duplicate_id
            .license_decision
            .evidence_snapshots
            .push(later_retrieval);
        duplicate_id
            .license_decision
            .evidence_snapshots
            .sort_by(|left, right| {
                super::phase3_evidence_identity(left)
                    .unwrap()
                    .cmp(&super::phase3_evidence_identity(right).unwrap())
            });
        assert!(
            super::validate_phase3_game_row(&duplicate_id, &manifest, "duplicate evidence ID")
                .is_err()
        );

        let mut reordered = complete;
        reordered.license_decision.evidence_snapshots.reverse();
        super::validate_phase3_game_row(&reordered, &manifest, "reordered evidence").unwrap();

        let mut invented = later;
        invented.license_decision.evidence_snapshots[0].retrieved_at =
            "2026-08-09T00:00:02Z".to_owned();
        assert!(
            super::validate_phase3_game_row(&invented, &manifest, "invented evidence tuple")
                .is_err()
        );

        let mut duplicate_tuple = manifest.clone();
        duplicate_tuple
            .evidence_snapshots
            .push(duplicate_tuple.evidence_snapshots[0].clone());
        duplicate_tuple.evidence_snapshots.sort_by(|left, right| {
            super::phase3_evidence_identity(left)
                .unwrap()
                .cmp(&super::phase3_evidence_identity(right).unwrap())
        });
        assert!(super::validate_phase3_dataset_manifest(&starts, &duplicate_tuple).is_err());

        for field in ["url", "contentType"] {
            let mut drifted = manifest.clone();
            let later = drifted
                .evidence_snapshots
                .iter_mut()
                .find(|snapshot| {
                    snapshot.evidence_id == first.evidence_id
                        && snapshot.retrieved_at != first.retrieved_at
                })
                .unwrap();
            match field {
                "url" => later.url.push_str("-drift"),
                "contentType" => later.content_type = "application/octet-stream".to_owned(),
                _ => unreachable!(),
            }
            drifted.evidence_snapshots.sort_by(|left, right| {
                super::phase3_evidence_identity(left)
                    .unwrap()
                    .cmp(&super::phase3_evidence_identity(right).unwrap())
            });
            assert!(
                super::validate_phase3_dataset_manifest(&starts, &drifted).is_err(),
                "evidence ID {field} drift must fail closed"
            );
        }

        let mut shared_url = manifest.clone();
        shared_url
            .evidence_snapshots
            .iter_mut()
            .find(|snapshot| snapshot.evidence_id == second_id.evidence_id)
            .unwrap()
            .url
            .clone_from(&first.url);
        shared_url.evidence_snapshots.sort_by(|left, right| {
            super::phase3_evidence_identity(left)
                .unwrap()
                .cmp(&super::phase3_evidence_identity(right).unwrap())
        });
        assert!(super::validate_phase3_dataset_manifest(&starts, &shared_url).is_err());

        let mut shared_object = manifest.clone();
        let collided = shared_object
            .evidence_snapshots
            .iter_mut()
            .find(|snapshot| snapshot.evidence_id == second_id.evidence_id)
            .unwrap();
        collided.sha256.clone_from(&first.sha256);
        collided.size = first.size;
        collided.object_path.clone_from(&first.object_path);
        shared_object.evidence_snapshots.sort_by(|left, right| {
            super::phase3_evidence_identity(left)
                .unwrap()
                .cmp(&super::phase3_evidence_identity(right).unwrap())
        });
        assert!(super::validate_phase3_dataset_manifest(&starts, &shared_object).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn phase3_games_cover_every_manifest_evidence_retrieval_version() {
        let root = temporary_record_path("phase3-evidence-union").with_extension("");
        let (engine, engine_receipt) = write_fixture_engine_build(&root, "abcdef0");
        let selected = fixture_start_positions();
        let (starts_reference, _, dataset) =
            write_start_evidence(&root, &engine, &engine_receipt, &selected);
        let starts_value = parse_unique_json(
            &std::fs::read(root.join(starts_reference.path)).unwrap(),
            "fixture starts",
        )
        .unwrap();
        let starts: super::Phase6StartPositions =
            deserialize_closed_json(&starts_value, "fixture starts").unwrap();
        let manifest_value = parse_unique_json(
            &std::fs::read(root.join(&dataset.path)).unwrap(),
            "fixture manifest",
        )
        .unwrap();
        let mut manifest: super::Phase3DatasetManifest =
            deserialize_closed_json(&manifest_value, "fixture manifest").unwrap();
        let first = manifest.evidence_snapshots[0].clone();
        let mut later = first.clone();
        later.retrieved_at = "2026-08-09T00:00:01Z".to_owned();
        manifest.evidence_snapshots.push(later.clone());
        manifest.evidence_snapshots.sort_by(|left, right| {
            super::phase3_evidence_identity(left)
                .unwrap()
                .cmp(&super::phase3_evidence_identity(right).unwrap())
        });
        super::validate_phase3_dataset_manifest(&starts, &manifest).unwrap();

        let games_path = dataset.path.parent().unwrap().join("games-00000.jsonl.gz");
        let compressed = std::fs::read(root.join(&games_path)).unwrap();
        let mut decoder = GzDecoder::new(compressed.as_slice());
        let mut jsonl = Vec::new();
        std::io::Read::read_to_end(&mut decoder, &mut jsonl).unwrap();
        let mut rows = jsonl
            .split(|byte| *byte == b'\n')
            .filter(|line| !line.is_empty())
            .map(|line| parse_unique_json(line, "fixture game").unwrap())
            .collect::<Vec<_>>();
        rows[0]["licenseDecision"]["evidenceSnapshots"][0]["retrieved_at"] =
            serde_json::Value::String(later.retrieved_at);
        let changed = gzip_fixture_jsonl(&rows);
        std::fs::write(root.join(&games_path), &changed).unwrap();
        let recorded = manifest
            .artifacts
            .get_mut("games-00000.jsonl.gz")
            .unwrap();
        recorded.sha256 = sha256_bytes(&changed);
        recorded.size = changed.len() as u64;

        let storage = super::RegistryStorage {
            root: open_shogi_core::AnchoredDir::open_existing(&root).unwrap(),
            root_path: root.clone(),
            registry_relative: PathBuf::from("artifacts/unused-registry.json"),
        };
        super::load_validated_phase3_games(
            &storage,
            &starts,
            &manifest,
            &mut BTreeMap::new(),
            &mut RegistryVerificationBudget::with_limits(10_000, 100_000_000),
        )
        .unwrap();
        drop(storage);

        rows[0]["licenseDecision"]["evidenceSnapshots"][0]["retrieved_at"] =
            serde_json::Value::String(first.retrieved_at);
        let missing_version = gzip_fixture_jsonl(&rows);
        std::fs::write(root.join(&games_path), &missing_version).unwrap();
        let recorded = manifest
            .artifacts
            .get_mut("games-00000.jsonl.gz")
            .unwrap();
        recorded.sha256 = sha256_bytes(&missing_version);
        recorded.size = missing_version.len() as u64;
        let storage = super::RegistryStorage {
            root: open_shogi_core::AnchoredDir::open_existing(&root).unwrap(),
            root_path: root.clone(),
            registry_relative: PathBuf::from("artifacts/unused-registry.json"),
        };
        let error = super::load_validated_phase3_games(
            &storage,
            &starts,
            &manifest,
            &mut BTreeMap::new(),
            &mut RegistryVerificationBudget::with_limits(10_000, 100_000_000),
        )
        .err()
        .expect("omitted manifest evidence retrieval must fail closed");
        assert!(error.contains("coverage differs"), "unexpected error: {error}");
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn phase3_games_reject_duplicate_raw_object_ids_even_when_hashes_differ() {
        let root = temporary_record_path("phase3-duplicate-object-id").with_extension("");
        let (engine, engine_receipt) = write_fixture_engine_build(&root, "abcdef0");
        let selected = fixture_start_positions();
        let (starts_reference, _, dataset) =
            write_start_evidence(&root, &engine, &engine_receipt, &selected);
        let starts_value = parse_unique_json(
            &std::fs::read(root.join(starts_reference.path)).unwrap(),
            "fixture starts",
        )
        .unwrap();
        let starts: super::Phase6StartPositions =
            deserialize_closed_json(&starts_value, "fixture starts").unwrap();
        let manifest_value = parse_unique_json(
            &std::fs::read(root.join(&dataset.path)).unwrap(),
            "fixture manifest",
        )
        .unwrap();
        let mut manifest: super::Phase3DatasetManifest =
            deserialize_closed_json(&manifest_value, "fixture manifest").unwrap();
        let games_path = dataset.path.parent().unwrap().join("games-00000.jsonl.gz");
        let compressed = std::fs::read(root.join(&games_path)).unwrap();
        let mut decoder = GzDecoder::new(compressed.as_slice());
        let mut jsonl = Vec::new();
        std::io::Read::read_to_end(&mut decoder, &mut jsonl).unwrap();
        let mut rows = jsonl
            .split(|byte| *byte == b'\n')
            .filter(|line| !line.is_empty())
            .map(|line| parse_unique_json(line, "fixture game").unwrap())
            .collect::<Vec<_>>();
        rows[1]["rawObject"]["objectId"] = rows[0]["rawObject"]["objectId"].clone();
        let changed = gzip_fixture_jsonl(&rows);
        std::fs::write(root.join(&games_path), &changed).unwrap();
        let recorded = manifest
            .artifacts
            .get_mut("games-00000.jsonl.gz")
            .unwrap();
        recorded.sha256 = sha256_bytes(&changed);
        recorded.size = changed.len() as u64;

        let storage = super::RegistryStorage {
            root: open_shogi_core::AnchoredDir::open_existing(&root).unwrap(),
            root_path: root.clone(),
            registry_relative: PathBuf::from("artifacts/unused-registry.json"),
        };
        let mut observed = BTreeMap::new();
        let mut budget = RegistryVerificationBudget::with_limits(10_000, 100_000_000);
        let error = super::load_validated_phase3_games(
            &storage,
            &starts,
            &manifest,
            &mut observed,
            &mut budget,
        )
        .err()
        .expect("duplicate raw objectId must fail closed");
        assert!(
            error.contains("producer order") || error.contains("repeat"),
            "unexpected error: {error}"
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    fn historical_engine_receipt_fixture() -> serde_json::Value {
        self_hashed_value(
            serde_json::json!({
                "schema": "open_shogi_engine_build_receipt/v1",
                "gitCommit": "a".repeat(40),
                "binary": {
                    "path": "target/release/open-shogi-cli",
                    "sha256": sha256_bytes(b"historical engine"),
                    "size": 17
                },
                "sourceTreeSha256": sha256_bytes(b"historical source"),
                "sourceFiles": 1,
                "sourceBytes": 1,
                "buildCommand": [
                    "cargo", "build", "--locked", "--release", "-p", "open-shogi-cli"
                ],
                "cargoVersion": "cargo historical",
                "rustcVersion": "rustc historical",
                "builtAt": "2026-08-09T00:00:00Z"
            }),
            "receiptSha256",
        )
    }

    #[test]
    fn historical_v1_engine_receipt_is_inspectable_but_malformed_commits_fail_closed() {
        let receipt = historical_engine_receipt_fixture();
        assert!(matches!(
            super::validate_engine_build_receipt_for_inspection(&receipt).unwrap(),
            super::EngineBuildReceipt::HistoricalV1(_)
        ));
        let mut malformed = receipt;
        malformed["gitCommit"] = serde_json::json!("g".repeat(40));
        refresh_fixture_self_hash(&mut malformed, "receiptSha256");
        assert!(super::validate_engine_build_receipt_for_inspection(&malformed).is_err());
    }

    #[test]
    fn v2_engine_receipt_rejects_rehashed_tool_and_runtime_contract_mutations() {
        let root = temporary_record_path("v2-receipt-mutations").with_extension("");
        let (_, reference) = write_fixture_engine_build(&root, "abcdef0");
        let valid: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join(reference.path)).unwrap()).unwrap();
        assert!(matches!(
            super::validate_engine_build_receipt_for_inspection(&valid).unwrap(),
            super::EngineBuildReceipt::ContentAddressedV2(_)
        ));

        let mut mutations = Vec::new();
        let mut relative_tool = valid.clone();
        relative_tool["cargoTool"]["path"] = serde_json::json!("relative/cargo");
        mutations.push(("relative tool path", relative_tool));
        let mut oversized_tool = valid.clone();
        oversized_tool["cargoTool"]["size"] = serde_json::json!(513_u64 * 1024 * 1024);
        mutations.push(("513 MiB tool", oversized_tool));
        let mut oversized_version = valid.clone();
        oversized_version["rustcTool"]["version"] = serde_json::json!("v".repeat(257));
        mutations.push(("257-byte tool version", oversized_version));
        let mut one_runtime_file = valid.clone();
        one_runtime_file["rustcRuntimeTree"]["files"] = serde_json::json!(1);
        mutations.push(("one-file runtime tree", one_runtime_file));
        let mut missing_offline = valid.clone();
        missing_offline["buildCommand"] = serde_json::json!([
            "cargo", "build", "--locked", "--release", "--jobs", "4", "-p",
            "open-shogi-cli"
        ]);
        mutations.push(("non-offline v2 build", missing_offline));
        let mut missing_jobs = valid.clone();
        missing_jobs["buildCommand"] = serde_json::json!([
            "cargo", "build", "--locked", "--release", "--offline", "-p",
            "open-shogi-cli"
        ]);
        mutations.push(("unbounded v2 build", missing_jobs));
        let mut wrong_jobs = valid.clone();
        wrong_jobs["buildCommand"] = serde_json::json!([
            "cargo", "build", "--locked", "--release", "--offline", "--jobs", "5",
            "-p", "open-shogi-cli"
        ]);
        mutations.push(("wrong v2 job limit", wrong_jobs));
        let mut reordered_jobs = valid.clone();
        reordered_jobs["buildCommand"] = serde_json::json!([
            "cargo", "build", "--locked", "--release", "--jobs", "4", "--offline",
            "-p", "open-shogi-cli"
        ]);
        mutations.push(("reordered v2 build", reordered_jobs));
        for (name, mut mutation) in mutations {
            refresh_fixture_self_hash(&mut mutation, "receiptSha256");
            assert!(
                super::validate_engine_build_receipt_for_inspection(&mutation).is_err(),
                "{name} must fail closed after self-hash refresh"
            );
        }
        std::fs::remove_dir_all(root).unwrap();
    }

    fn paired_report_fixture(
        root: &Path,
    ) -> (
        super::Phase2ArenaReport,
        super::PairedArenaJob,
        Vec<RegistryArtifact>,
    ) {
        let challenger = RegistryArtifact {
            path: PathBuf::from("weights/challenger.osaval"),
            sha256: "1".repeat(64),
            size: 1,
        };
        let champion = RegistryArtifact {
            path: PathBuf::from("weights/champion.osaval"),
            sha256: "2".repeat(64),
            size: 1,
        };
        let job = serde_json::json!({
            "jobId": "pair-0000",
            "pairIndex": 0,
            "startGroup": "initial",
            "startPositionId": "standard-initial",
            "sfen": super::PHASE6_INITIAL_SFEN,
            "seed": 7,
            "gameIds": ["game-000000", "game-000001"],
            "modelAColorOrder": ["black", "white"],
            "outputDir": "artifacts/generation-1/arena/jobs/pair-0000",
            "reportPath": "artifacts/generation-1/arena/jobs/pair-0000/arena-report.json",
            "csaPaths": [
                "artifacts/generation-1/arena/jobs/pair-0000/games/game-000001.csa",
                "artifacts/generation-1/arena/jobs/pair-0000/games/game-000002.csa"
            ],
            "quarantinePath": "artifacts/generation-1/arena/quarantine/pair-0000.json",
            "command": {
                "kind": "engine_arena",
                "argv": ["fixture-engine", "arena"],
                "timeoutSeconds": 1
            }
        });
        let (report_reference, csa) =
            write_pair_report(root, &job, &challenger, &champion, &"3".repeat(64));
        let report_value = parse_unique_json(
            &std::fs::read(root.join(report_reference.path)).unwrap(),
            "fixture pair report",
        )
        .unwrap();
        let report = deserialize_closed_json(&report_value, "fixture pair report").unwrap();
        let job = serde_json::from_value(job).unwrap();
        (report, job, csa)
    }

    fn validate_fixture_report_games(
        root: &Path,
        report: &super::Phase2ArenaReport,
        job: &super::PairedArenaJob,
        csa: &[RegistryArtifact],
    ) -> Result<(), String> {
        let storage = super::RegistryStorage {
            root: open_shogi_core::AnchoredDir::open_existing(root).unwrap(),
            root_path: root.to_path_buf(),
            registry_relative: PathBuf::from("artifacts/unused-registry.json"),
        };
        super::validate_phase2_report_games(
            &storage,
            report,
            job,
            csa,
            &report.run.player_a,
            &report.run.player_b,
            &mut BTreeMap::new(),
            &mut RegistryVerificationBudget::with_limits(10_000, 100_000_000),
        )
    }

    #[test]
    fn paired_report_csa_replay_rejects_resigned_garbage_and_semantic_mismatches() {
        let root = temporary_record_path("paired-csa-replay").with_extension("");
        let (report, job, csa) = paired_report_fixture(&root);
        validate_fixture_report_games(&root, &report, &job, &csa).unwrap();

        let mut wrong_result = report.clone();
        wrong_result.games[0].result = "black_win".to_owned();
        assert!(validate_fixture_report_games(&root, &wrong_result, &job, &csa).is_err());

        let mut wrong_moves = report.clone();
        wrong_moves.games[0].moves += 1;
        assert!(validate_fixture_report_games(&root, &wrong_moves, &job, &csa).is_err());

        let mut wrong_initial = report.clone();
        let mut after_move = open_shogi_core::Position::startpos();
        after_move
            .make_move(after_move.legal_moves().into_iter().next().unwrap())
            .unwrap();
        wrong_initial.run.initial_sfen = reset_fixture_move_number(&to_sfen(&after_move));
        assert!(validate_fixture_report_games(&root, &wrong_initial, &job, &csa).is_err());

        let mut legal_mismatch = report.clone();
        let initial = open_shogi_core::Position::startpos();
        let movement = initial.legal_moves().into_iter().next().unwrap();
        let replacement = to_csa_game(&CsaGame {
            version: "V3.0".to_owned(),
            black_name: Some(report.games[0].black.clone()),
            white_name: Some(report.games[0].white.clone()),
            metadata: Vec::new(),
            initial_position: initial,
            moves: vec![movement],
            special_move: Some(CsaSpecialMove::Resign),
            result_validation: CsaResultValidation::ExternalCondition,
        })
        .unwrap();
        std::fs::write(root.join(&csa[0].path), replacement.as_bytes()).unwrap();
        let mut resigned_csa = csa.clone();
        resigned_csa[0].sha256 = sha256_bytes(replacement.as_bytes());
        resigned_csa[0].size = replacement.len() as u64;
        legal_mismatch.games[0].csa_sha256.clone_from(&resigned_csa[0].sha256);
        legal_mismatch.games[0].csa_size = resigned_csa[0].size;
        assert!(
            validate_fixture_report_games(&root, &legal_mismatch, &job, &resigned_csa).is_err(),
            "a legal but different re-signed CSA must not launder report semantics"
        );

        let garbage = b"not CSA\n";
        std::fs::write(root.join(&csa[0].path), garbage).unwrap();
        let mut garbage_refs = csa;
        garbage_refs[0].sha256 = sha256_bytes(garbage);
        garbage_refs[0].size = garbage.len() as u64;
        let mut garbage_report = report;
        garbage_report.games[0].csa_sha256.clone_from(&garbage_refs[0].sha256);
        garbage_report.games[0].csa_size = garbage_refs[0].size;
        assert!(
            validate_fixture_report_games(&root, &garbage_report, &job, &garbage_refs).is_err()
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn paired_report_metrics_reject_impossible_counters_but_ignore_unverifiable_rates() {
        let root = temporary_record_path("paired-counter-contract").with_extension("");
        let (report, job, csa) = paired_report_fixture(&root);
        super::validate_phase2_report_metrics(
            &report,
            &report.run.player_a,
            &report.run.player_b,
        )
        .unwrap();

        let mut zero_nodes = report.clone();
        zero_nodes.games[0].player_a_search_nodes = 0;
        assert!(validate_fixture_report_games(&root, &zero_nodes, &job, &csa).is_err());
        zero_nodes.metrics.player_a_search_nodes = 0;
        assert!(
            super::validate_phase2_report_metrics(
                &zero_nodes,
                &zero_nodes.run.player_a,
                &zero_nodes.run.player_b,
            )
            .is_err()
        );

        let mut illegal = report.clone();
        illegal.metrics.illegal_moves = 1;
        assert!(
            super::validate_phase2_report_metrics(
                &illegal,
                &illegal.run.player_a,
                &illegal.run.player_b,
            )
            .is_err()
        );
        let mut peak = report.clone();
        peak.metrics.peak_memory_bytes = Some(1);
        assert!(
            super::validate_phase2_report_metrics(
                &peak,
                &peak.run.player_a,
                &peak.run.player_b,
            )
            .is_err()
        );

        for (tt, cutoff, pruning) in [(1.0, 0.5, 0.25), (0.0, 1.0, 1.0)] {
            let mut observation = report.clone();
            observation.metrics.tt_hit_rate = tt;
            observation.metrics.cutoff_rate = cutoff;
            observation.metrics.pruning_rate = pruning;
            super::validate_phase2_report_metrics(
                &observation,
                &observation.run.player_a,
                &observation.run.player_b,
            )
            .unwrap();
            assert_eq!(
                super::phase2_metric_counters(&report.metrics),
                super::phase2_metric_counters(&observation.metrics),
                "range-only rates must not alter any retained performance/decision counter"
            );
            assert_eq!(report.games, observation.games);
        }
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn paired_report_config_digest_is_recomputed_from_language_neutral_signature_bytes() {
        let root = temporary_record_path("paired-config-digest").with_extension("");
        write_finalized_registry_fixture(&root);
        let (plan, _) = read_fixture_plan_and_execution(&root);
        let job = &plan.jobs[0];
        let report_value = parse_unique_json(
            &std::fs::read(root.join(&job.report_path)).unwrap(),
            "fixture pair report",
        )
        .unwrap();
        let report: super::Phase2ArenaReport =
            deserialize_closed_json(&report_value, "fixture pair report").unwrap();
        super::validate_phase2_report_run(
            &report,
            job,
            &plan,
            &report.run.player_a,
            &report.run.player_b,
        )
        .unwrap();

        let mut changed = report;
        changed.run.config_sha256 = "0".repeat(64);
        assert!(
            super::validate_phase2_report_run(
                &changed,
                job,
                &plan,
                &changed.run.player_a,
                &changed.run.player_b,
            )
            .is_err()
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn paired_attempt_lifecycle_preserves_quarantine_history_and_rejects_running_or_retry_completion() {
        let root = temporary_record_path("paired-attempt-lifecycle").with_extension("");
        write_finalized_registry_fixture(&root);
        let (plan, execution) = read_fixture_plan_and_execution(&root);
        validate_fixture_execution_value(&root, &plan, &execution).unwrap();

        let original = execution["attempts"][0].clone();
        let completed_second = completed_attempt_with_number(&root, &original, 2);
        let quarantine = quarantined_attempt_fixture(&root, &plan, &original);

        let mut recovered = execution.clone();
        let attempts = recovered["attempts"].as_array_mut().unwrap();
        attempts[0] = quarantine;
        attempts.insert(1, completed_second.clone());
        recovered["quarantinedAttempts"] = serde_json::json!(1);
        refresh_fixture_self_hash(&mut recovered, "manifestSha256");
        validate_fixture_execution_value(&root, &plan, &recovered).unwrap();

        let mut completed_twice = execution.clone();
        completed_twice["attempts"]
            .as_array_mut()
            .unwrap()
            .insert(1, completed_second);
        refresh_fixture_self_hash(&mut completed_twice, "manifestSha256");
        let completion_error =
            validate_fixture_execution_value(&root, &plan, &completed_twice).unwrap_err();
        assert!(
            completion_error.contains("one completion after only quarantines"),
            "unexpected error: {completion_error}"
        );

        let mut running = execution.clone();
        let running_attempt = &mut running["attempts"][0];
        running_attempt["status"] = serde_json::json!("running");
        for field in [
            "returnCode",
            "timedOut",
            "outputLimitExceeded",
            "memoryLimitExceeded",
            "peakRssBytes",
            "rssMeasurement",
            "stdout",
            "stderr",
            "report",
            "quarantine",
            "failureCategory",
            "commandReceipt",
        ] {
            running_attempt[field] = serde_json::Value::Null;
        }
        running_attempt["csa"] = serde_json::json!([]);
        refresh_fixture_self_hash(&mut running, "manifestSha256");
        let running_error = validate_fixture_execution_value(&root, &plan, &running).unwrap_err();
        assert!(
            running_error.contains("must end in one completion")
                || running_error.contains("final completed attempt"),
            "unexpected error: {running_error}"
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "one closed-contract test mutates every outer and nested receipt link"
    )]
    fn paired_attempt_command_receipts_bind_every_command_outcome_and_artifact_link() {
        let root = temporary_record_path("paired-command-receipt").with_extension("");
        write_finalized_registry_fixture(&root);
        let (plan, execution) = read_fixture_plan_and_execution(&root);
        let original = execution["attempts"][0].clone();
        validate_fixture_attempt_value(&root, &plan, &original).unwrap();

        let mut missing = original.clone();
        missing["commandReceipt"] = serde_json::Value::Null;
        assert!(validate_fixture_attempt_value(&root, &plan, &missing).is_err());

        let receipt_reference: RegistryArtifact =
            serde_json::from_value(original["commandReceipt"].clone()).unwrap();
        let receipt = parse_unique_json(
            &std::fs::read(root.join(&receipt_reference.path)).unwrap(),
            "fixture attempt command receipt",
        )
        .unwrap();
        for field in [
            "plan",
            "job",
            "attempt",
            "command",
            "engine",
            "build_receipt",
            "outcome",
            "stdout",
            "report",
            "csa",
            "failure",
            "completed_at",
        ] {
            let mut changed_receipt = receipt.clone();
            match field {
                "plan" => changed_receipt["planSha256"] = serde_json::json!("0".repeat(64)),
                "job" => changed_receipt["jobId"] = serde_json::json!("pair-9999"),
                "attempt" => changed_receipt["attempt"] = serde_json::json!(2),
                "command" => {
                    changed_receipt["command"]["argv"][0] = serde_json::json!("other-engine");
                }
                "engine" => {
                    changed_receipt["engine"]["sha256"] = serde_json::json!("0".repeat(64));
                }
                "build_receipt" => {
                    changed_receipt["engineBuildReceipt"] = serde_json::Value::Null;
                }
                "outcome" => changed_receipt["result"]["returnCode"] = serde_json::json!(1),
                "stdout" => {
                    changed_receipt["stdout"]["sha256"] = serde_json::json!("0".repeat(64));
                }
                "report" => {
                    changed_receipt["report"]["sha256"] = serde_json::json!("0".repeat(64));
                }
                "csa" => {
                    changed_receipt["csa"][0]["sha256"] = serde_json::json!("0".repeat(64));
                }
                "failure" => changed_receipt["failureCategory"] = serde_json::json!("timeout"),
                "completed_at" => {
                    changed_receipt["completedAt"] = serde_json::json!("2026-08-09T00:00:02Z");
                }
                _ => unreachable!(),
            }
            refresh_fixture_self_hash(&mut changed_receipt, "receiptSha256");
            let changed_reference = write_fixture_json(
                &root,
                &format!("artifacts/receipt-tamper-{field}.json"),
                &changed_receipt,
            );
            let mut changed_attempt = original.clone();
            changed_attempt["commandReceipt"] = serde_json::to_value(changed_reference).unwrap();
            assert!(
                validate_fixture_attempt_value(&root, &plan, &changed_attempt).is_err(),
                "attempt receipt {field} tamper was accepted"
            );
        }

        let process_reference: RegistryArtifact =
            serde_json::from_value(receipt["processReceipt"].clone()).unwrap();
        let process_receipt = parse_unique_json(
            &std::fs::read(root.join(process_reference.path)).unwrap(),
            "fixture process receipt",
        )
        .unwrap();
        for field in [
            "schema",
            "command",
            "command_sha",
            "executable",
            "build_receipt",
            "runtime_receipt",
            "outcome",
            "stdout",
        ] {
            let mut changed_process = process_receipt.clone();
            match field {
                "schema" => changed_process["schema"] = serde_json::json!("phase6_command_receipt/v1"),
                "command" => {
                    changed_process["command"]["argv"][0] = serde_json::json!("other-engine");
                }
                "command_sha" => {
                    changed_process["commandSha256"] = serde_json::json!("0".repeat(64));
                }
                "executable" => {
                    changed_process["expectedExecutable"]["sha256"] =
                        serde_json::json!("0".repeat(64));
                }
                "build_receipt" => changed_process["engineBuildReceipt"] = serde_json::Value::Null,
                "runtime_receipt" => {
                    changed_process["runtimeReceipt"] = serde_json::to_value(&plan.engine).unwrap();
                }
                "outcome" => changed_process["returnCode"] = serde_json::json!(1),
                "stdout" => {
                    changed_process["stdout"]["sha256"] = serde_json::json!("0".repeat(64));
                }
                _ => unreachable!(),
            }
            let changed_process_reference = write_fixture_json(
                &root,
                &format!("artifacts/process-receipt-tamper-{field}.json"),
                &changed_process,
            );
            let mut changed_receipt = receipt.clone();
            changed_receipt["processReceipt"] =
                serde_json::to_value(changed_process_reference).unwrap();
            refresh_fixture_self_hash(&mut changed_receipt, "receiptSha256");
            let changed_reference = write_fixture_json(
                &root,
                &format!("artifacts/attempt-process-tamper-{field}.json"),
                &changed_receipt,
            );
            let mut changed_attempt = original.clone();
            changed_attempt["commandReceipt"] = serde_json::to_value(changed_reference).unwrap();
            assert!(
                validate_fixture_attempt_value(&root, &plan, &changed_attempt).is_err(),
                "process receipt {field} tamper was accepted"
            );
        }
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn paired_attempt_resource_and_failure_evidence_matches_the_producer_contract() {
        let root = temporary_record_path("paired-resource-evidence").with_extension("");
        write_finalized_registry_fixture(&root);
        let (plan, execution) = read_fixture_plan_and_execution(&root);
        let completed = execution["attempts"][0].clone();
        validate_fixture_attempt_value(&root, &plan, &completed).unwrap();

        let mut unavailable_completion = completed.clone();
        unavailable_completion["rssMeasurement"] = serde_json::json!("unavailable");
        unavailable_completion["peakRssBytes"] = serde_json::Value::Null;
        unavailable_completion["memoryLimitExceeded"] = serde_json::json!(true);
        assert!(
            validate_fixture_attempt_value(&root, &plan, &unavailable_completion).is_err()
        );

        let mut nonzero_short_lived = completed.clone();
        nonzero_short_lived["peakRssBytes"] = serde_json::json!(1);
        assert!(validate_fixture_attempt_value(&root, &plan, &nonzero_short_lived).is_err());

        let mut over_limit_sum = completed.clone();
        over_limit_sum["rssMeasurement"] = serde_json::json!("process_tree_ps_rss_sum");
        over_limit_sum["peakRssBytes"] =
            serde_json::json!(plan.memory_per_worker_mib * 1024 * 1024 + 1);
        assert!(validate_fixture_attempt_value(&root, &plan, &over_limit_sum).is_err());

        let mut before_report = completed.clone();
        before_report["completedAt"] = serde_json::json!("2026-08-09T00:00:00Z");
        assert!(validate_fixture_attempt_value(&root, &plan, &before_report).is_err());

        let quarantine = quarantined_attempt_fixture(&root, &plan, &completed);
        validate_fixture_attempt_value(&root, &plan, &quarantine).unwrap();

        let mut false_sum_limit = quarantine.clone();
        false_sum_limit["rssMeasurement"] = serde_json::json!("process_tree_ps_rss_sum");
        false_sum_limit["peakRssBytes"] = serde_json::json!(1);
        false_sum_limit["memoryLimitExceeded"] = serde_json::json!(true);
        assert!(validate_fixture_attempt_value(&root, &plan, &false_sum_limit).is_err());

        let mut false_unavailable = quarantine.clone();
        false_unavailable["rssMeasurement"] = serde_json::json!("unavailable");
        false_unavailable["peakRssBytes"] = serde_json::Value::Null;
        false_unavailable["memoryLimitExceeded"] = serde_json::json!(false);
        assert!(validate_fixture_attempt_value(&root, &plan, &false_unavailable).is_err());

        let mut wrong_precedence = quarantine;
        wrong_precedence["failureCategory"] = serde_json::json!("memory_limit");
        assert!(validate_fixture_attempt_value(&root, &plan, &wrong_precedence).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn quarantine_observed_artifacts_are_all_or_none_and_only_for_missing_artifacts() {
        let root = temporary_record_path("quarantine-observed-artifacts").with_extension("");
        write_finalized_registry_fixture(&root);
        let (plan, execution) = read_fixture_plan_and_execution(&root);
        let completed = execution["attempts"][0].clone();
        let timeout = quarantined_attempt_fixture(&root, &plan, &completed);
        let quarantine_reference: RegistryArtifact =
            serde_json::from_value(timeout["quarantine"].clone()).unwrap();
        let timeout_record = parse_unique_json(
            &std::fs::read(root.join(quarantine_reference.path)).unwrap(),
            "fixture timeout quarantine",
        )
        .unwrap();
        validate_fixture_quarantine_record_value(&root, &plan, &timeout, &timeout_record).unwrap();

        let mut timeout_with_artifact = timeout_record.clone();
        timeout_with_artifact["observedReport"] = completed["report"].clone();
        timeout_with_artifact["observedCsa"] = completed["csa"].clone();
        assert!(
            validate_fixture_quarantine_record_value(
                &root,
                &plan,
                &timeout,
                &timeout_with_artifact
            )
            .is_err()
        );

        let mut missing_attempt = timeout.clone();
        missing_attempt["returnCode"] = serde_json::json!(0);
        missing_attempt["timedOut"] = serde_json::json!(false);
        missing_attempt["failureCategory"] = serde_json::json!("missing_or_invalid_artifact");
        let mut missing_none = timeout_record.clone();
        missing_none["returnCode"] = serde_json::json!(0);
        missing_none["timedOut"] = serde_json::json!(false);
        missing_none["failureCategory"] = serde_json::json!("missing_or_invalid_artifact");
        validate_fixture_quarantine_record_value(&root, &plan, &missing_attempt, &missing_none)
            .unwrap();

        let mut missing_all = missing_none.clone();
        missing_all["observedReport"] = completed["report"].clone();
        missing_all["observedCsa"] = completed["csa"].clone();
        validate_fixture_quarantine_record_value(&root, &plan, &missing_attempt, &missing_all)
            .unwrap();

        let mut partial = missing_all.clone();
        partial["observedCsa"].as_array_mut().unwrap().pop();
        assert!(
            validate_fixture_quarantine_record_value(&root, &plan, &missing_attempt, &partial)
                .is_err()
        );

        let mut wrong_path = missing_all;
        wrong_path["observedReport"]["path"] = serde_json::json!("artifacts/wrong-report.json");
        assert!(
            validate_fixture_quarantine_record_value(&root, &plan, &missing_attempt, &wrong_path)
                .is_err()
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn start_validation_success_requires_bounded_measured_rss() {
        let root = temporary_record_path("start-validation-rss").with_extension("");
        write_finalized_registry_fixture(&root);
        let (plan, _) = read_fixture_plan_and_execution(&root);
        let validation = parse_unique_json(
            &std::fs::read(root.join(&plan.start_position_validation.path)).unwrap(),
            "fixture start validation",
        )
        .unwrap();
        validate_fixture_start_validation_value(&root, &plan, &validation).unwrap();

        let mut unavailable = validation.clone();
        unavailable["results"][0]["rssMeasurement"] = serde_json::json!("unavailable");
        unavailable["results"][0]["peakRssBytes"] = serde_json::Value::Null;
        assert!(validate_fixture_start_validation_value(&root, &plan, &unavailable).is_err());

        for peak in [serde_json::Value::Null, serde_json::json!(1)] {
            let mut invalid_short = validation.clone();
            invalid_short["results"][0]["peakRssBytes"] = peak;
            assert!(
                validate_fixture_start_validation_value(&root, &plan, &invalid_short).is_err()
            );
        }

        let mut over_limit = validation;
        over_limit["results"][0]["rssMeasurement"] =
            serde_json::json!("process_tree_ps_rss_sum");
        over_limit["results"][0]["peakRssBytes"] =
            serde_json::json!(1024_u64 * 1024 * 1024 + 1);
        assert!(validate_fixture_start_validation_value(&root, &plan, &over_limit).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn durable_engine_receipt_is_closed_self_hashed_and_bound_to_its_plan() {
        let root = temporary_record_path("durable-engine-receipt").with_extension("");
        let registry_path = materialize_shared_registry_chain(&root);
        let (_, storage, _) = read_registry(&registry_path).unwrap();
        let plan_value = parse_unique_json(
            &std::fs::read(root.join("artifacts/arena-plan.json")).unwrap(),
            "shared arena plan",
        )
        .unwrap();
        let mut plan: super::PairedArenaPlan =
            deserialize_closed_json(&plan_value, "shared arena plan").unwrap();
        let (engine, mut reference) =
            write_fixture_engine_build(&root, &plan.git_commit);
        plan.engine = engine;
        plan.engine_build_receipt = Some(reference.clone());
        let mut observed = BTreeMap::new();
        let mut budget = RegistryVerificationBudget::with_limits(10_000, 100_000_000);
        super::validate_engine_build_receipt_artifact(
            &storage,
            &plan,
            &reference,
            &mut observed,
            &mut budget,
        )
        .unwrap();

        let historical_reference = write_fixture_json(
            &root,
            "local/build-receipts/open-shogi-cli.json",
            &historical_engine_receipt_fixture(),
        );
        let historical_error = super::validate_engine_build_receipt_artifact(
            &storage,
            &plan,
            &historical_reference,
            &mut BTreeMap::new(),
            &mut RegistryVerificationBudget::with_limits(10_000, 100_000_000),
        )
        .unwrap_err();
        assert!(
            historical_error.contains("requires open_shogi_engine_build_receipt/v2"),
            "unexpected error: {historical_error}"
        );

        let receipt_path = reference.path.clone();
        let mut receipt: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join(&receipt_path)).unwrap()).unwrap();
        receipt["unexpected"] = serde_json::Value::Bool(true);
        let mut unsigned = receipt.as_object().unwrap().clone();
        unsigned.remove("receiptSha256");
        receipt["receiptSha256"] = serde_json::Value::String(
            python_canonical_sha256(&serde_json::Value::Object(unsigned)).unwrap(),
        );
        let mut changed_bytes = serde_json::to_vec(&receipt).unwrap();
        changed_bytes.push(b'\n');
        std::fs::write(root.join(&receipt_path), &changed_bytes).unwrap();
        reference.sha256 = sha256_bytes(&changed_bytes);
        reference.size = changed_bytes.len() as u64;
        plan.engine_build_receipt = Some(reference.clone());
        let error = super::validate_engine_build_receipt_artifact(
            &storage,
            &plan,
            &reference,
            &mut BTreeMap::new(),
            &mut RegistryVerificationBudget::with_limits(10_000, 100_000_000),
        )
        .unwrap_err();
        assert!(error.contains("unknown field"), "unexpected error: {error}");
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn shared_python_registry_chain_is_accepted_and_report_tampering_is_rejected() {
        let root = temporary_record_path("shared-registry-chain").with_extension("");
        let registry_path = materialize_shared_registry_chain(&root);

        let resolved =
            resolve_registry_profile(PlayProfile::NeuralLineageChampion, &registry_path).unwrap();
        assert_eq!(resolved.model_id, "champion-v0");

        let report = std::fs::read_dir(root.join("artifacts/phase6/generation-0001/arena/jobs"))
            .unwrap()
            .map(Result::unwrap)
            .map(|entry| entry.path().join("arena-report.json"))
            .find(|path| path.is_file())
            .unwrap();
        let original = std::fs::read(&report).unwrap();
        let mut changed = original.clone();
        let needle = b"\"black_win\"";
        let index = changed
            .windows(needle.len())
            .position(|window| window == needle)
            .unwrap();
        changed[index + 1..index + 6].copy_from_slice(b"white");
        assert_eq!(changed.len(), original.len());
        std::fs::write(&report, changed).unwrap();
        let Err(error) =
            resolve_registry_profile(PlayProfile::NeuralLineageChampion, &registry_path)
        else {
            panic!("tampered shared registry chain was accepted");
        };
        assert!(
            error.contains("registry artifact SHA-256 mismatch"),
            "unexpected error: {error}"
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn evidence_json_rejects_duplicate_keys_recursively() {
        for raw in [
            br#"{"decision":"rejected","decision":"promoted"}"#.as_slice(),
            br#"{"evidence":{"games":40,"games":2}}"#,
            br#"{"overall":{"scoreWilson95":{"lower":0.1,"lower":0.9}}}"#,
            br#"{"games":[{"metrics":{"championSearches":1,"championSearches":2}}]}"#,
        ] {
            let error = parse_unique_json(raw, "test evidence").unwrap_err();
            assert!(error.contains("duplicate JSON key"), "{error}");
        }
    }

    #[test]
    fn evidence_json_rejects_excessive_nesting_and_node_counts() {
        let depth_error = parse_unique_json_with_limits(b"[[[[0]]]]", "test evidence", 2, 16)
            .unwrap_err();
        assert!(depth_error.contains("nesting exceeds 2 levels"), "{depth_error}");

        let node_error =
            parse_unique_json_with_limits(b"[0,1,2,3]", "test evidence", 4, 4).unwrap_err();
        assert!(node_error.contains("value exceeds 4 nodes"), "{node_error}");
    }

    #[test]
    fn policy_semantic_hash_matches_python_for_float_and_integer_numbers() {
        let standard = parse_generation_policy(include_bytes!(
            "../../../configs/generation/phase6_promotion.toml"
        ))
        .unwrap();
        let integer_policy = parse_generation_policy(
            br"schema_version=1
[evidence]
minimum_games=40
minimum_decisive_games=12
minimum_group_games=10
max_illegal=0
max_crashes=0
[thresholds]
promote_score_rate=1
promote_wilson_lower=1
reject_score_rate=0
reject_wilson_upper=0
minimum_group_score_rate=0
maximum_side_score_gap=1
[performance]
require_metrics=false
maximum_inference_slowdown=1
maximum_search_slowdown=1
",
        )
        .unwrap();

        assert_eq!(
            python_canonical_sha256(&serde_json::to_value(standard).unwrap()).unwrap(),
            "b067a02a0ac5d68959eca8bb3b779ab4d410c39738b486ac921710314e791a02"
        );
        assert_eq!(
            python_canonical_sha256(&serde_json::to_value(integer_policy).unwrap()).unwrap(),
            "d31ff6cd03775453e18c7e45527369583ec1fd4c489fdeecd91b24beaa24488f"
        );
    }

    #[test]
    fn promotion_artifact_structures_are_closed_and_deterministically_rederived() {
        let root = temporary_record_path("strict-promotion-contract").with_extension("");
        let _ = write_finalized_registry_fixture(&root);

        let mut results_value = parse_unique_json(
            &std::fs::read(root.join("artifacts/arena.json")).unwrap(),
            "test results",
        )
        .unwrap();
        let results: ArenaResults =
            deserialize_closed_json(&results_value, "test results").unwrap();
        let results_ref = RegistryArtifact {
            path: PathBuf::from("artifacts/arena.json"),
            sha256: sha256_bytes(&std::fs::read(root.join("artifacts/arena.json")).unwrap()),
            size: std::fs::metadata(root.join("artifacts/arena.json"))
                .unwrap()
                .len(),
        };
        let expected_analysis = analyze_arena_results(&results, results_ref).unwrap();

        let mut analysis_value = parse_unique_json(
            &std::fs::read(root.join("artifacts/arena-analysis.json")).unwrap(),
            "test analysis",
        )
        .unwrap();
        let analysis: ArenaAnalysisEnvelope =
            deserialize_closed_json(&analysis_value, "test analysis").unwrap();
        assert_eq!(analysis, expected_analysis);

        let mut decision_value = parse_unique_json(
            &std::fs::read(root.join("artifacts/promotion-rejected.json")).unwrap(),
            "test decision",
        )
        .unwrap();
        let decision: PromotionDecision =
            deserialize_closed_json(&decision_value, "test decision").unwrap();
        let policy = parse_generation_policy(
            &std::fs::read(root.join("artifacts/policy.toml")).unwrap(),
        )
        .unwrap();
        assert_eq!(
            decision,
            derive_promotion_decision(&analysis, &decision, &policy).unwrap()
        );

        let mut missing_results_nullable = results_value.clone();
        missing_results_nullable["games"][0]["metrics"]
            .as_object_mut()
            .unwrap()
            .remove("championInferenceCalls");
        assert!(
            deserialize_closed_json::<ArenaResults>(
                &missing_results_nullable,
                "test results missing nullable field",
            )
            .is_err()
        );
        let mut missing_analysis_nullable = analysis_value.clone();
        missing_analysis_nullable
            .as_object_mut()
            .unwrap()
            .remove("sideScoreGap");
        assert!(
            deserialize_closed_json::<ArenaAnalysisEnvelope>(
                &missing_analysis_nullable,
                "test analysis missing nullable field",
            )
            .is_err()
        );
        let mut missing_decision_nullable = decision_value.clone();
        missing_decision_nullable["evidence"]
            .as_object_mut()
            .unwrap()
            .remove("sideScoreGap");
        assert!(
            deserialize_closed_json::<PromotionDecision>(
                &missing_decision_nullable,
                "test decision missing nullable field",
            )
            .is_err()
        );

        results_value["games"][0]["metrics"]["unexpected"] = serde_json::json!(0);
        assert!(deserialize_closed_json::<ArenaResults>(&results_value, "test results").is_err());
        analysis_value["overall"]["unexpected"] = serde_json::json!(0);
        assert!(
            deserialize_closed_json::<ArenaAnalysisEnvelope>(&analysis_value, "test analysis")
                .is_err()
        );
        decision_value["evidence"]["unexpected"] = serde_json::json!(0);
        assert!(
            deserialize_closed_json::<PromotionDecision>(&decision_value, "test decision").is_err()
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn resignation_saves_a_canonical_record() {
        let path = temporary_record_path("resign");
        let config = test_config(&path);
        let mut input = Cursor::new(b"resign\n");
        let mut output = Vec::new();
        run_interactive(&config, &mut input, &mut output).unwrap();
        let rendered = String::from_utf8(output).unwrap();
        assert!(rendered.contains("Black hand: -"));
        assert!(rendered.contains("White hand: -"));
        let record = std::fs::read_to_string(path).unwrap();
        assert!(record.starts_with("'CSA encoding=UTF-8\nV3.0\n"));
        assert!(record.contains("$CONFIG_SHA256:"));
        assert!(record.contains("%TORYO\n"));
        let decisions = std::fs::read_to_string(&config.decision_log).unwrap();
        assert!(decisions.starts_with("{\"schema\":\"open_shogi_play_config/v3\""));
        std::fs::remove_file(&config.output).unwrap();
        std::fs::remove_file(&config.decision_log).unwrap();
    }

    #[test]
    fn overlong_no_newline_human_input_is_drained_safely() {
        let path = temporary_record_path("overlong");
        let config = test_config(&path);
        let mut input = Cursor::new(vec![b'x'; 129]);
        let mut output = Vec::new();
        run_interactive(&config, &mut input, &mut output).unwrap();
        assert!(
            String::from_utf8(output)
                .unwrap()
                .contains("input too long")
        );
        std::fs::remove_file(&config.output).unwrap();
        std::fs::remove_file(&config.decision_log).unwrap();
    }

    #[test]
    fn paired_publication_preserves_an_existing_destination() {
        let path = temporary_record_path("existing");
        let config = test_config(&path);
        std::fs::write(&path, "sentinel").unwrap();
        let (config_sha256, csa, decisions) = paired_evidence(&config);
        assert!(publish_human_artifacts(&config, &config_sha256, &csa, &decisions).is_err());
        assert_eq!(std::fs::read_to_string(path).unwrap(), "sentinel");
    }

    #[test]
    fn paired_publication_recovers_after_only_the_csa_was_linked() {
        let path = temporary_record_path("recover");
        let config = test_config(&path);
        prepare_publication_parent(&config.output).unwrap();
        prepare_publication_parent(&config.decision_log).unwrap();
        let paths = publication_paths(&config).unwrap();
        let (config_sha256, csa, decisions) = paired_evidence(&config);
        write_pending(&paths.csa_pending, &csa).unwrap();
        write_pending(&paths.decision_pending, &decisions).unwrap();
        let marker = PublicationMarker {
            schema: "phase6_human_publication/v1".into(),
            config_sha256: config_sha256.clone(),
            csa_target: absolute_publication_path(&config.output)
                .unwrap()
                .to_string_lossy()
                .into_owned(),
            decision_target: absolute_publication_path(&config.decision_log)
                .unwrap()
                .to_string_lossy()
                .into_owned(),
            csa_pending: absolute_publication_path(&paths.csa_pending)
                .unwrap()
                .to_string_lossy()
                .into_owned(),
            decision_pending: absolute_publication_path(&paths.decision_pending)
                .unwrap()
                .to_string_lossy()
                .into_owned(),
            csa_sha256: sha256_bytes(&csa),
            csa_size: csa.len() as u64,
            decision_sha256: sha256_bytes(&decisions),
            decision_size: decisions.len() as u64,
        };
        publish_new_atomic(&paths.marker, &serde_json::to_vec(&marker).unwrap()).unwrap();
        std::fs::hard_link(&paths.csa_pending, &config.output).unwrap();

        assert!(recover_publication(&config, &config_sha256).unwrap());
        assert_eq!(std::fs::read(&config.output).unwrap(), csa);
        assert_eq!(std::fs::read(&config.decision_log).unwrap(), decisions);
        assert!(!paths.marker.exists());
        assert!(!paths.csa_pending.exists());
        assert!(!paths.decision_pending.exists());
    }

    #[test]
    fn paired_publication_cleanup_preserves_a_replaced_pending_file() {
        let path = temporary_record_path("cleanup-replacement");
        let config = test_config(&path);
        prepare_publication_parent(&config.output).unwrap();
        prepare_publication_parent(&config.decision_log).unwrap();
        let paths = publication_paths(&config).unwrap();
        let (config_sha256, csa, decisions) = paired_evidence(&config);
        write_pending(&paths.csa_pending, &csa).unwrap();
        write_pending(&paths.decision_pending, &decisions).unwrap();
        let marker = PublicationMarker {
            schema: "phase6_human_publication/v1".into(),
            config_sha256: config_sha256.clone(),
            csa_target: absolute_publication_path(&config.output)
                .unwrap()
                .to_string_lossy()
                .into_owned(),
            decision_target: absolute_publication_path(&config.decision_log)
                .unwrap()
                .to_string_lossy()
                .into_owned(),
            csa_pending: absolute_publication_path(&paths.csa_pending)
                .unwrap()
                .to_string_lossy()
                .into_owned(),
            decision_pending: absolute_publication_path(&paths.decision_pending)
                .unwrap()
                .to_string_lossy()
                .into_owned(),
            csa_sha256: sha256_bytes(&csa),
            csa_size: csa.len() as u64,
            decision_sha256: sha256_bytes(&decisions),
            decision_size: decisions.len() as u64,
        };
        publish_new_atomic(&paths.marker, &serde_json::to_vec(&marker).unwrap()).unwrap();
        let storage = PublicationStorage::open(&config, &paths, false).unwrap();

        let result = recover_publication_with_storage_before_cleanup(
            &config,
            &paths,
            &storage,
            &config_sha256,
            |_| {
                std::fs::remove_file(&paths.csa_pending).map_err(|error| error.to_string())?;
                std::fs::write(&paths.csa_pending, b"foreign replacement")
                    .map_err(|error| error.to_string())
            },
        );

        assert!(result.is_err());
        assert_eq!(
            std::fs::read(&paths.csa_pending).unwrap(),
            b"foreign replacement"
        );
        assert_eq!(std::fs::read(&config.output).unwrap(), csa);
        assert_eq!(std::fs::read(&config.decision_log).unwrap(), decisions);
    }

    #[test]
    fn paired_publication_rejects_final_path_substitution_before_or_after_cleanup() {
        for (label, replace_decision, after_cleanup) in [
            ("final-csa-before-cleanup", false, false),
            ("final-csa-after-cleanup", false, true),
            ("final-decision-before-cleanup", true, false),
            ("final-decision-after-cleanup", true, true),
        ] {
            let path = temporary_record_path(label);
            let config = test_config(&path);
            let (config_sha256, csa, decisions) = paired_evidence(&config);
            let (paths, storage) = prepare_publication_recovery_fixture(
                &config,
                &config_sha256,
                &csa,
                &decisions,
            );
            let target = if replace_decision {
                config.decision_log.clone()
            } else {
                config.output.clone()
            };
            let replacement = target.with_extension("foreign-replacement");
            let foreign = if replace_decision {
                b"foreign decision".as_slice()
            } else {
                b"foreign CSA".as_slice()
            };
            std::fs::write(&replacement, foreign).unwrap();

            let result = recover_publication_with_storage_cleanup_hooks(
                &config,
                &paths,
                &storage,
                &config_sha256,
                |_| {
                    if !after_cleanup {
                        substitute_publication_target(&target, &replacement)?;
                    }
                    Ok(())
                },
                |_| {
                    if after_cleanup {
                        substitute_publication_target(&target, &replacement)?;
                    }
                    Ok(())
                },
            );

            assert!(result.is_err(), "{label}");
            assert_eq!(std::fs::read(&target).unwrap(), foreign, "{label}");
            assert!(!replacement.exists(), "{label}");
        }
    }

    #[test]
    fn paired_publication_reconstructs_marker_from_two_durable_pending_files() {
        let path = temporary_record_path("recover-before-marker");
        let config = test_config(&path);
        prepare_publication_parent(&config.output).unwrap();
        prepare_publication_parent(&config.decision_log).unwrap();
        let paths = publication_paths(&config).unwrap();
        let (config_sha256, csa, decisions) = paired_evidence(&config);
        write_pending(&paths.csa_pending, &csa).unwrap();
        write_pending(&paths.decision_pending, &decisions).unwrap();
        assert!(!paths.marker.exists());

        assert!(recover_publication(&config, &config_sha256).unwrap());
        assert_eq!(std::fs::read(&config.output).unwrap(), csa);
        assert_eq!(std::fs::read(&config.decision_log).unwrap(), decisions);
        assert!(!paths.marker.exists());
        assert!(!paths.csa_pending.exists());
        assert!(!paths.decision_pending.exists());
    }

    #[test]
    fn paired_publication_recovers_a_durable_uncommitted_marker() {
        let path = temporary_record_path("paired-marker-recovery");
        let config = test_config(&path);
        prepare_publication_parent(&config.output).unwrap();
        prepare_publication_parent(&config.decision_log).unwrap();
        let paths = publication_paths(&config).unwrap();
        let (config_sha256, csa, decisions) = paired_evidence(&config);
        write_pending(&paths.csa_pending, &csa).unwrap();
        write_pending(&paths.decision_pending, &decisions).unwrap();
        let marker = PublicationMarker {
            schema: "phase6_human_publication/v1".into(),
            config_sha256: config_sha256.clone(),
            csa_target: absolute_publication_path(&config.output)
                .unwrap()
                .to_string_lossy()
                .into_owned(),
            decision_target: absolute_publication_path(&config.decision_log)
                .unwrap()
                .to_string_lossy()
                .into_owned(),
            csa_pending: absolute_publication_path(&paths.csa_pending)
                .unwrap()
                .to_string_lossy()
                .into_owned(),
            decision_pending: absolute_publication_path(&paths.decision_pending)
                .unwrap()
                .to_string_lossy()
                .into_owned(),
            csa_sha256: sha256_bytes(&csa),
            csa_size: csa.len() as u64,
            decision_sha256: sha256_bytes(&decisions),
            decision_size: decisions.len() as u64,
        };
        write_pending(&paths.marker_pending, &serde_json::to_vec(&marker).unwrap()).unwrap();

        assert!(recover_publication(&config, &"f".repeat(64)).is_err());
        assert!(!paths.marker.exists());
        assert!(!config.output.exists());
        assert!(!config.decision_log.exists());
        assert!(recover_publication(&config, &config_sha256).unwrap());
        assert_eq!(std::fs::read(&config.output).unwrap(), csa);
        assert_eq!(std::fs::read(&config.decision_log).unwrap(), decisions);
        assert!(!paths.marker.exists());
        assert!(!paths.marker_pending.exists());
        assert!(!paths.csa_pending.exists());
        assert!(!paths.decision_pending.exists());
    }

    #[test]
    fn paired_publication_rejects_colliding_derived_paths_before_play() {
        let path = temporary_record_path("paired-collision");
        let mut config = test_config(&path);
        let output_name = config.output.file_name().unwrap().to_string_lossy();
        config.decision_log = config
            .output
            .with_file_name(format!(".{output_name}.publication.json"));
        let mut input = Cursor::new(b"resign\n");
        let mut output = Vec::new();

        assert!(run_interactive(&config, &mut input, &mut output).is_err());
        assert!(!config.output.exists());
        assert!(!config.decision_log.exists());
    }

    #[test]
    fn human_play_digest_covers_side_initial_position_and_limits() {
        let path = temporary_record_path("digest");
        let mut config = test_config(&path);
        let profile = handcrafted_profile("handcrafted-experimental");
        let original = play_config_sha256(&config, &profile, None);
        config.human = Side::White;
        assert_ne!(original, play_config_sha256(&config, &profile, None));
        config.human = Side::Black;
        config.max_plies += 1;
        assert_ne!(original, play_config_sha256(&config, &profile, None));
        config.max_plies -= 1;
        config.initial = open_shogi_core::parse_sfen(
            "lnsgkgsnl/1r5b1/ppppppppp/9/9/P8/1PPPPPPPP/1B5R1/LNSGKGSNL w - 2",
        )
        .unwrap();
        assert_ne!(original, play_config_sha256(&config, &profile, None));
    }

    #[test]
    fn paired_evidence_rejects_an_external_terminal_the_play_loop_never_emits() {
        let path = temporary_record_path("unsupported-terminal");
        let config = test_config(&path);
        let (config_sha256, csa, decisions) = paired_evidence(&config);
        let csa = String::from_utf8(csa)
            .unwrap()
            .replace("%CHUDAN\n", "%HIKIWAKE\n")
            .into_bytes();

        assert!(validate_paired_evidence(&csa, &decisions, &config_sha256).is_err());
    }

    #[test]
    fn paired_evidence_rejects_interruption_during_the_ai_turn() {
        let path = temporary_record_path("interrupted-ai-turn");
        let config = test_config(&path);
        let profile = handcrafted_profile("handcrafted-experimental");
        let config_sha256 = play_config_sha256(&config, &profile, None);
        let record = human_play_config_record(&config, &profile, None, &config_sha256);
        let movement = open_shogi_core::parse_usi_move("7g7f").unwrap();
        let csa = encode_record(
            &config.initial,
            &[movement],
            config.human,
            open_shogi_core::CsaSpecialMove::Interrupted,
            open_shogi_core::CsaResultValidation::ExternalCondition,
            &record,
        )
        .unwrap();
        let event = DecisionEvent {
            schema: "phase6_human_decision/v1".to_owned(),
            ply: 1,
            actor: "human".to_owned(),
            model_id: record.model_id.clone(),
            model_artifact_sha256: None,
            model_payload_sha256: None,
            config_sha256: config_sha256.clone(),
            sfen_before: open_shogi_core::to_sfen(&config.initial),
            move_usi: "7g7f".to_owned(),
            nodes: 0,
            elapsed_ms: 0,
            depth: 0,
            pv: Vec::new(),
            score_cp: None,
            opening_book: false,
        };
        let decisions = encode_decision_log(&record, &[event]).unwrap();

        assert!(validate_paired_evidence(csa.as_bytes(), &decisions, &config_sha256).is_err());
    }

    #[test]
    fn rust_registry_validation_matches_core_python_relationships() {
        let canonical = canonical_registry();
        validate_model_registry(&canonical).unwrap();

        let mut bad_revision = canonical_registry_value();
        bad_revision["revision"] = serde_json::json!(0);
        assert!(validate_model_registry(&registry_from_value(bad_revision)).is_err());

        let mut bad_status = canonical_registry_value();
        bad_status["generations"][0]["status"] = serde_json::json!("unknown");
        assert!(validate_model_registry(&registry_from_value(bad_status)).is_err());

        let mut bad_parent = canonical_registry_value();
        bad_parent["models"][0]["parentModelId"] = serde_json::json!("missing");
        assert!(validate_model_registry(&registry_from_value(bad_parent)).is_err());

        let mut bad_generation = canonical_registry_value();
        bad_generation["models"][0]["generationId"] = serde_json::json!("missing");
        assert!(validate_model_registry(&registry_from_value(bad_generation)).is_err());

        let mut bad_timestamp = canonical_registry_value();
        bad_timestamp["models"][0]["registeredAt"] = serde_json::json!("2026-02-30T00:00:00Z");
        assert!(validate_model_registry(&registry_from_value(bad_timestamp)).is_err());

        let mut bad_identifier = canonical_registry_value();
        bad_identifier["championModelId"] = serde_json::json!(".hidden");
        assert!(validate_model_registry(&registry_from_value(bad_identifier)).is_err());

        let mut bad_license = canonical_registry_value();
        bad_license["models"][0]["licenseStatus"] = serde_json::json!("project-generated");
        assert!(validate_model_registry(&registry_from_value(bad_license)).is_err());

        let active = active_registry_value();
        validate_model_registry(&registry_from_value(active.clone())).unwrap();

        let mut promoted = active.clone();
        promoted["championModelId"] = serde_json::json!("challenger-v1");
        promoted["challengerModelId"] = serde_json::Value::Null;
        promoted["generations"][1]["status"] = serde_json::json!("complete");
        promoted["generations"][1]["arenaManifest"] = artifact_value("artifacts/arena.json", "e");
        promoted["generations"][1]["promotionDecision"] =
            artifact_value("artifacts/promotion.json", "f");
        validate_model_registry(&registry_from_value(promoted)).unwrap();

        let mut legacy_status = active.clone();
        legacy_status["generations"][1]["status"] = serde_json::json!("training");
        assert!(validate_model_registry(&registry_from_value(legacy_status)).is_err());

        let mut forward_model_parent = active.clone();
        forward_model_parent["models"]
            .as_array_mut()
            .unwrap()
            .swap(0, 1);
        assert!(validate_model_registry(&registry_from_value(forward_model_parent)).is_err());

        let mut forward_generation_parent = active.clone();
        forward_generation_parent["generations"]
            .as_array_mut()
            .unwrap()
            .swap(0, 1);
        assert!(validate_model_registry(&registry_from_value(forward_generation_parent)).is_err());

        let mut missing_training_evidence = active.clone();
        missing_training_evidence["generations"][1]["trainingRunManifest"] =
            serde_json::Value::Null;
        assert!(validate_model_registry(&registry_from_value(missing_training_evidence)).is_err());

        let mut incomplete_promotion_pair = active.clone();
        incomplete_promotion_pair["generations"][1]["arenaManifest"] =
            serde_json::json!(artifact_value("artifacts/arena.json", "e"));
        assert!(validate_model_registry(&registry_from_value(incomplete_promotion_pair)).is_err());

        let mut active_without_top_level_challenger = active;
        active_without_top_level_challenger["challengerModelId"] = serde_json::Value::Null;
        assert!(
            validate_model_registry(&registry_from_value(active_without_top_level_challenger))
                .is_err()
        );
    }

    #[test]
    fn paired_plan_registry_snapshot_requires_exact_lifecycle_successors() {
        let snapshot = registry_from_value(active_registry_value());
        let current = registry_from_value(finalized_registry_value(false, "rejected"));
        super::validate_registry_snapshot_transition(&snapshot, &current, "rejected").unwrap();

        let mut changed_history = finalized_registry_value(false, "rejected");
        changed_history["generations"][1]["selfplayManifest"]["path"] =
            serde_json::json!("artifacts/replaced-selfplay.json");
        let error = super::validate_registry_snapshot_transition(
            &snapshot,
            &registry_from_value(changed_history),
            "rejected",
        )
        .unwrap_err();
        assert!(error.contains("exactly precede generation finalization"));

        let mut skipped_revision = finalized_registry_value(false, "rejected");
        skipped_revision["revision"] = serde_json::json!(4);
        let error = super::validate_registry_snapshot_transition(
            &snapshot,
            &registry_from_value(skipped_revision),
            "rejected",
        )
        .unwrap_err();
        assert!(error.contains("exact paired-plan lifecycle successor"));

        let mut later = finalized_registry_value(false, "rejected");
        append_completed_generation(&mut later, "generation-2", "challenger-v2", "2");
        later["revision"] = serde_json::json!(5);
        later["generations"][2]["parentGenerationId"] = serde_json::json!("generation-1");
        super::validate_registry_snapshot_transition(
            &snapshot,
            &registry_from_value(later),
            "rejected",
        )
        .unwrap();

        let promoted = registry_from_value(finalized_registry_value(true, "promoted"));
        super::validate_registry_snapshot_transition(&snapshot, &promoted, "promoted").unwrap();
        assert!(
            super::validate_registry_snapshot_transition(&snapshot, &promoted, "rejected")
                .is_err()
        );
    }

    #[test]
    fn challenger_alias_prefers_the_active_challenger() {
        let mut value = finalized_registry_value(false, "rejected");
        append_completed_generation(&mut value, "generation-2", "challenger-v2", "2");
        value["generations"][2]["arenaManifest"] = serde_json::Value::Null;
        value["generations"][2]["promotionDecision"] = serde_json::Value::Null;
        value["generations"][2]["status"] = serde_json::json!("arena");
        value["challengerModelId"] = serde_json::json!("challenger-v2");
        let registry = registry_from_value(value);
        validate_model_registry(&registry).unwrap();

        assert_eq!(
            select_challenger_model_id(&registry).unwrap(),
            "challenger-v2"
        );
    }

    #[test]
    fn challenger_alias_reselects_the_latest_finalized_challenger_for_every_decision() {
        for (decision, promoted) in [
            ("promoted", true),
            ("rejected", false),
            ("inconclusive", false),
        ] {
            let registry = registry_from_value(finalized_registry_value(promoted, decision));
            validate_model_registry(&registry).unwrap();

            assert_eq!(
                select_challenger_model_id(&registry).unwrap(),
                "challenger-v1",
                "unexpected challenger for {decision}"
            );
        }
    }

    #[test]
    fn challenger_alias_falls_back_to_the_latest_completed_generation() {
        let mut value = finalized_registry_value(false, "rejected");
        append_completed_generation(&mut value, "generation-2", "challenger-v2", "2");
        let registry = registry_from_value(value);
        validate_model_registry(&registry).unwrap();

        assert_eq!(
            select_challenger_model_id(&registry).unwrap(),
            "challenger-v2"
        );
    }

    #[test]
    fn challenger_alias_rejects_an_initial_only_registry() {
        let registry = canonical_registry();
        validate_model_registry(&registry).unwrap();

        assert_eq!(
            select_challenger_model_id(&registry).unwrap_err(),
            "model registry has no non-initial challenger generation"
        );
    }

    #[test]
    fn finalized_challenger_resolution_preserves_registry_artifact_identity_checks() {
        let root = temporary_record_path("registry-challenger-identity").with_extension("");
        let (registry_path, expected_registry_sha256, expected_model_sha256) =
            write_finalized_registry_fixture(&root);

        let resolved = resolve_registry_profile(PlayProfile::Challenger, &registry_path).unwrap();
        assert_eq!(resolved.model_id, "challenger-v1");
        assert_eq!(
            resolved.model_artifact_sha256.as_deref(),
            Some(expected_model_sha256.as_str())
        );
        assert_eq!(
            resolved.registry_sha256.as_deref(),
            Some(expected_registry_sha256.as_str())
        );
        assert_eq!(resolved.registry_revision, Some(3));

        std::fs::write(root.join("weights/challenger.osaval"), b"tampered").unwrap();
        let Err(error) = resolve_registry_profile(PlayProfile::Challenger, &registry_path) else {
            panic!("tampered challenger artifact was accepted");
        };
        assert!(
            error.contains("registry artifact size mismatch"),
            "unexpected error: {error}"
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn registry_verification_budget_bounds_references_and_unique_bytes() {
        let mut references = RegistryVerificationBudget::with_limits(2, 100);
        references.charge_reference().unwrap();
        references.charge_reference().unwrap();
        assert!(references.charge_reference().is_err());

        let mut bytes = RegistryVerificationBudget::with_limits(10, 4);
        bytes.charge_unique_bytes(4).unwrap();
        assert!(bytes.charge_unique_bytes(1).is_err());
    }

    #[test]
    fn registry_resolution_reuses_stable_identity_for_hard_link_aliases() {
        let root = temporary_record_path("registry-hardlink-dedupe").with_extension("");
        let (registry_path, _, _) = write_finalized_registry_fixture(&root);
        let (separate_registry, separate_storage, _) = read_registry(&registry_path).unwrap();
        validate_model_registry(&separate_registry).unwrap();
        let mut separate_budget = RegistryVerificationBudget::with_limits(10_000, 100_000_000);
        verify_registry_artifacts_with_budget(
            &separate_registry,
            &separate_storage,
            "champion-v0",
            &mut separate_budget,
        )
        .unwrap();
        let model_size = std::fs::metadata(root.join("weights/champion.osaval"))
            .unwrap()
            .len();

        std::fs::remove_file(root.join("weights/challenger.osaval")).unwrap();
        std::fs::hard_link(
            root.join("weights/champion.osaval"),
            root.join("weights/challenger.osaval"),
        )
        .unwrap();

        let (registry, storage, _) = read_registry(&registry_path).unwrap();
        validate_model_registry(&registry).unwrap();
        let mut budget = RegistryVerificationBudget::with_limits(10_000, 100_000_000);

        let artifact =
            verify_registry_artifacts_with_budget(&registry, &storage, "champion-v0", &mut budget)
                .unwrap();

        assert_eq!(artifact.sha256, registry.models[0].artifact.sha256);
        assert_eq!(separate_budget.unique_bytes - budget.unique_bytes, model_size);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn rejected_decision_cannot_transition_the_top_level_champion() {
        let root = temporary_record_path("registry-rejected-transition").with_extension("");
        let (registry_path, _, _) = write_finalized_registry_fixture(&root);
        let mut registry: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&registry_path).unwrap()).unwrap();
        registry["championModelId"] = serde_json::json!("challenger-v1");
        std::fs::write(&registry_path, serde_json::to_vec(&registry).unwrap()).unwrap();

        let Err(error) =
            resolve_registry_profile(PlayProfile::NeuralLineageChampion, &registry_path)
        else {
            panic!("invalid champion transition was accepted");
        };

        assert!(
            error.contains("final registry top-level models disagree with the paired plan outcome"),
            "unexpected error: {error}"
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn rejected_challenger_cannot_be_laundered_as_a_later_generation_incumbent() {
        let mut value = finalized_registry_value(false, "rejected");
        append_completed_generation(&mut value, "generation-2", "challenger-v2", "2");
        value["models"][2]["parentModelId"] = serde_json::json!("challenger-v1");
        value["generations"][2]["parentGenerationId"] = serde_json::json!("generation-1");
        value["generations"][2]["championModelId"] = serde_json::json!("challenger-v1");
        value["championModelId"] = serde_json::json!("challenger-v1");
        let registry = registry_from_value(value);
        validate_model_registry(&registry).unwrap();
        let decisions = BTreeMap::from([
            ("generation-1".to_owned(), "rejected".to_owned()),
            ("generation-2".to_owned(), "rejected".to_owned()),
        ]);

        let error = validate_registry_champion_transition(&registry, &decisions).unwrap_err();

        assert!(
            error.contains("parent's promotion decision"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn later_generation_accepts_every_legitimate_parent_outcome() {
        for (parent_decision, promoted, incumbent) in [
            ("promoted", true, "challenger-v1"),
            ("rejected", false, "champion-v0"),
            ("inconclusive", false, "champion-v0"),
        ] {
            let mut value = finalized_registry_value(promoted, parent_decision);
            append_completed_generation(&mut value, "generation-2", "challenger-v2", "2");
            value["models"][2]["parentModelId"] = serde_json::json!(incumbent);
            value["generations"][2]["parentGenerationId"] = serde_json::json!("generation-1");
            value["generations"][2]["championModelId"] = serde_json::json!(incumbent);
            value["championModelId"] = serde_json::json!(incumbent);
            let registry = registry_from_value(value);
            validate_model_registry(&registry).unwrap();
            let decisions = BTreeMap::from([
                (
                    "generation-1".to_owned(),
                    parent_decision.to_owned(),
                ),
                ("generation-2".to_owned(), "inconclusive".to_owned()),
            ]);

            validate_registry_champion_transition(&registry, &decisions).unwrap();
        }
    }

    #[test]
    fn a_second_initial_generation_cannot_bypass_promotion_evidence() {
        let mut value = finalized_registry_value(false, "rejected");
        let second_model = serde_json::json!({
            "modelId": "unreviewed-root",
            "generationId": "generation-root-2",
            "parentModelId": null,
            "artifact": artifact_value("weights/unreviewed.osaval", "9"),
            "evaluatorKind": "neural",
            "architectureVersion": "1",
            "quantization": "float32",
            "trainingRun": null,
            "registeredAt": "2026-08-13T00:00:00Z",
            "licenseStatus": "pending-review"
        });
        value["models"].as_array_mut().unwrap().push(second_model);
        value["generations"]
            .as_array_mut()
            .unwrap()
            .push(serde_json::json!({
                "generationId": "generation-root-2",
                "parentGenerationId": null,
                "championModelId": "unreviewed-root",
                "challengerModelId": null,
                "selfplayManifest": null,
                "teacherLabelingManifest": null,
                "trainingRunManifest": null,
                "arenaManifest": null,
                "promotionDecision": null,
                "status": "complete",
                "createdAt": "2026-08-13T00:00:00Z"
            }));
        value["championModelId"] = serde_json::json!("unreviewed-root");
        let registry = registry_from_value(value);

        let error = validate_model_registry(&registry).unwrap_err();

        assert!(
            error.contains("exactly one initial generation")
                || error.contains("immediately preceding generation")
        );
    }

    #[test]
    fn an_empty_registry_cannot_omit_its_initial_generation() {
        let mut value = canonical_registry_value();
        value["championModelId"] = serde_json::Value::Null;
        value["models"] = serde_json::json!([]);
        value["generations"] = serde_json::json!([]);
        let registry = registry_from_value(value);

        let error = validate_model_registry(&registry).unwrap_err();

        assert!(error.contains("exactly one initial generation"));
    }

    #[test]
    fn active_generation_incumbent_must_remain_the_top_level_champion() {
        let mut value = finalized_registry_value(true, "promoted");
        append_completed_generation(&mut value, "generation-2", "challenger-v2", "2");
        value["models"][2]["parentModelId"] = serde_json::json!("challenger-v1");
        value["generations"][2]["parentGenerationId"] = serde_json::json!("generation-1");
        value["generations"][2]["championModelId"] = serde_json::json!("challenger-v1");
        value["generations"][2]["arenaManifest"] = serde_json::Value::Null;
        value["generations"][2]["promotionDecision"] = serde_json::Value::Null;
        value["generations"][2]["status"] = serde_json::json!("arena");
        value["championModelId"] = serde_json::json!("champion-v0");
        value["challengerModelId"] = serde_json::json!("challenger-v2");
        let registry = registry_from_value(value);

        let error = validate_model_registry(&registry).unwrap_err();

        assert!(
            error.contains("active generation incumbent"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn promotion_decision_and_analysis_must_bind_the_registry_generation() {
        let root = temporary_record_path("registry-promotion-binding").with_extension("");
        let (registry_path, _, _) = write_finalized_registry_fixture(&root);
        let promotion_path = root.join("artifacts/promotion-rejected.json");
        let mut promotion: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&promotion_path).unwrap()).unwrap();
        promotion.as_object_mut().unwrap().remove("decisionSha256");
        promotion["generationId"] = serde_json::json!("generation-2");
        let promotion_bytes = self_hashed_json(promotion, "decisionSha256");
        std::fs::write(&promotion_path, &promotion_bytes).unwrap();
        let mut registry: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&registry_path).unwrap()).unwrap();
        registry["generations"][1]["promotionDecision"]["sha256"] =
            serde_json::json!(sha256_bytes(&promotion_bytes));
        registry["generations"][1]["promotionDecision"]["size"] =
            serde_json::json!(promotion_bytes.len());
        std::fs::write(&registry_path, serde_json::to_vec(&registry).unwrap()).unwrap();

        let Err(error) =
            resolve_registry_profile(PlayProfile::NeuralLineageChampion, &registry_path)
        else {
            panic!("mismatched promotion identity was accepted");
        };

        assert!(
            error.contains("identity differs"),
            "unexpected error: {error}"
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn promotion_analysis_results_must_equal_the_generation_arena_reference() {
        let root = temporary_record_path("registry-analysis-binding").with_extension("");
        let (registry_path, _, _) = write_finalized_registry_fixture(&root);
        let analysis_path = root.join("artifacts/arena-analysis.json");
        let mut analysis: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&analysis_path).unwrap()).unwrap();
        analysis.as_object_mut().unwrap().remove("analysisSha256");
        analysis["results"] = analysis["plan"].clone();
        let analysis_bytes = self_hashed_json(analysis, "analysisSha256");
        std::fs::write(&analysis_path, &analysis_bytes).unwrap();

        let promotion_path = root.join("artifacts/promotion-rejected.json");
        let mut promotion: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&promotion_path).unwrap()).unwrap();
        promotion.as_object_mut().unwrap().remove("decisionSha256");
        promotion["arenaAnalysis"]["sha256"] = serde_json::json!(sha256_bytes(&analysis_bytes));
        promotion["arenaAnalysis"]["size"] = serde_json::json!(analysis_bytes.len());
        let promotion_bytes = self_hashed_json(promotion, "decisionSha256");
        std::fs::write(&promotion_path, &promotion_bytes).unwrap();

        let mut registry: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&registry_path).unwrap()).unwrap();
        registry["generations"][1]["promotionDecision"]["sha256"] =
            serde_json::json!(sha256_bytes(&promotion_bytes));
        registry["generations"][1]["promotionDecision"]["size"] =
            serde_json::json!(promotion_bytes.len());
        std::fs::write(&registry_path, serde_json::to_vec(&registry).unwrap()).unwrap();

        let Err(error) =
            resolve_registry_profile(PlayProfile::NeuralLineageChampion, &registry_path)
        else {
            panic!("mismatched arena analysis was accepted");
        };

        assert!(
            error.contains("arena manifest"),
            "unexpected error: {error}"
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn registry_resolution_rejects_a_checksum_correct_corrupt_unselected_model() {
        let root = temporary_record_path("registry-unselected-corrupt").with_extension("");
        let (registry_path, _, _) = write_finalized_registry_fixture(&root);
        let corrupt = b"not-an-osaval-model";
        std::fs::write(root.join("weights/challenger.osaval"), corrupt).unwrap();
        let mut registry: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&registry_path).unwrap()).unwrap();
        registry["models"][1]["artifact"]["sha256"] = serde_json::json!(sha256_bytes(corrupt));
        registry["models"][1]["artifact"]["size"] = serde_json::json!(corrupt.len());
        std::fs::write(&registry_path, serde_json::to_vec(&registry).unwrap()).unwrap();

        let Err(error) =
            resolve_registry_profile(PlayProfile::NeuralLineageChampion, &registry_path)
        else {
            panic!("corrupt unselected model was accepted");
        };
        assert!(error.contains("not valid OSAVAL01"));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn repository_root_is_derived_from_registry_not_current_directory() {
        let root = temporary_record_path("registry-root").with_extension("");
        std::fs::create_dir_all(root.join(".git")).unwrap();
        std::fs::create_dir_all(root.join("weights")).unwrap();
        let registry_path = root.join("weights/model-registry.json");
        std::fs::write(&registry_path, b"{}").unwrap();

        let storage = repository_root_and_registry(&registry_path).unwrap();
        assert_eq!(storage.root_path, root.canonicalize().unwrap());
        assert_eq!(
            storage.root_path.join(&storage.registry_relative),
            registry_path.canonicalize().unwrap()
        );
    }

    #[test]
    fn nested_repository_markers_make_the_registry_root_ambiguous() {
        let root = temporary_record_path("registry-ambiguous-root").with_extension("");
        let nested = root.join("nested");
        std::fs::create_dir_all(root.join(".git")).unwrap();
        std::fs::create_dir_all(nested.join(".git")).unwrap();
        std::fs::create_dir_all(nested.join("weights")).unwrap();
        let registry_path = nested.join("weights/model-registry.json");
        std::fs::write(&registry_path, b"{}").unwrap();

        let error = repository_root_and_registry(&registry_path)
            .err()
            .expect("nested repository roots must fail closed");

        assert!(error.contains("multiple repository roots"), "{error}");
        std::fs::remove_dir_all(root).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn registry_root_swap_cannot_redirect_artifact_reads_outside_the_anchored_root() {
        use std::os::unix::fs::symlink;

        let root = temporary_record_path("registry-root-swap").with_extension("");
        let (registry_path, _, model_sha256) = write_finalized_registry_fixture(&root);
        let (registry, storage, _) = read_registry(&registry_path).unwrap();
        let moved = root.with_extension("anchored");
        let external = root.with_extension("external");
        std::fs::create_dir_all(external.join("weights")).unwrap();
        std::fs::create_dir_all(external.join("artifacts")).unwrap();
        std::fs::write(external.join("weights/champion.osaval"), b"outside").unwrap();
        std::fs::rename(&root, &moved).unwrap();
        symlink(&external, &root).unwrap();

        let artifact = verify_registry_artifacts(&registry, &storage, "champion-v0").unwrap();
        assert_eq!(artifact.sha256, model_sha256);
        assert_ne!(artifact.bytes, b"outside");
    }

    #[cfg(unix)]
    #[test]
    fn human_publication_parent_swap_is_rejected_without_external_writes() {
        use std::os::unix::fs::symlink;

        let root = temporary_record_path("human-root-swap").with_extension("");
        std::fs::create_dir_all(&root).unwrap();
        let config = test_config(&root.join("game.csa"));
        let paths = publication_paths(&config).unwrap();
        let storage = PublicationStorage::open(&config, &paths, false).unwrap();
        let (config_sha256, csa, decisions) = paired_evidence(&config);
        let moved = root.with_extension("anchored");
        let external = root.with_extension("external");
        std::fs::create_dir_all(&external).unwrap();
        std::fs::rename(&root, &moved).unwrap();
        symlink(&external, &root).unwrap();

        assert!(
            publish_human_artifacts_with_storage(
                &config,
                &paths,
                &storage,
                &config_sha256,
                &csa,
                &decisions,
            )
            .is_err()
        );
        assert_eq!(std::fs::read_dir(external).unwrap().count(), 0);
    }

    #[test]
    fn human_publication_real_directory_replacement_is_rejected() {
        let root = temporary_record_path("human-real-root-swap").with_extension("");
        std::fs::create_dir_all(&root).unwrap();
        let config = test_config(&root.join("game.csa"));
        let paths = publication_paths(&config).unwrap();
        let storage = PublicationStorage::open(&config, &paths, false).unwrap();
        let (config_sha256, csa, decisions) = paired_evidence(&config);
        let moved = root.with_extension("anchored");
        std::fs::rename(&root, &moved).unwrap();
        std::fs::create_dir(&root).unwrap();

        let result = publish_human_artifacts_with_storage(
            &config,
            &paths,
            &storage,
            &config_sha256,
            &csa,
            &decisions,
        );

        assert!(result.is_err());
        assert_eq!(std::fs::read_dir(&root).unwrap().count(), 0);
    }

    #[test]
    fn human_publication_rejects_case_aliases_within_one_directory_identity() {
        let root = temporary_record_path("human-case-alias").with_extension("");
        std::fs::create_dir_all(&root).unwrap();
        let mut config = test_config(&root.join("game.csa"));
        config.decision_log = root.join("GAME.CSA");
        let paths = publication_paths(&config).unwrap();
        let storage = PublicationStorage::open(&config, &paths, false).unwrap();

        let error = validate_publication_path_identities(&storage).unwrap_err();

        assert_eq!(error, "human-play publication paths must be distinct");
    }

    #[cfg(unix)]
    #[test]
    fn registry_artifacts_reject_final_and_intermediate_symlinks() {
        use std::os::unix::fs::symlink;

        let root = temporary_record_path("registry-symlink").with_extension("");
        std::fs::create_dir_all(root.join("real")).unwrap();
        std::fs::write(root.join("real/model.osaval"), b"model").unwrap();
        symlink(
            root.join("real/model.osaval"),
            root.join("model-link.osaval"),
        )
        .unwrap();
        symlink(root.join("real"), root.join("directory-link")).unwrap();

        assert!(contained_artifact_path(&root, Path::new("model-link.osaval")).is_err());
        assert!(contained_artifact_path(&root, Path::new("directory-link/model.osaval")).is_err());
    }

    fn canonical_registry() -> ModelRegistry {
        registry_from_value(canonical_registry_value())
    }

    fn registry_from_value(value: serde_json::Value) -> ModelRegistry {
        serde_json::from_value(value).unwrap()
    }

    fn canonical_registry_value() -> serde_json::Value {
        serde_json::json!({
            "schema": "phase6_model_registry/v1",
            "revision": 1,
            "championModelId": "champion-v0",
            "challengerModelId": null,
            "models": [{
                "modelId": "champion-v0",
                "generationId": "generation-0",
                "parentModelId": null,
                "artifact": {"path": "weights/champion.osaval", "sha256": "a".repeat(64), "size": 0},
                "evaluatorKind": "neural",
                "architectureVersion": "1",
                "quantization": "int8",
                "trainingRun": null,
                "registeredAt": "2026-08-08T00:00:00Z",
                "licenseStatus": "pending-review"
            }],
            "generations": [{
                "generationId": "generation-0",
                "parentGenerationId": null,
                "championModelId": "champion-v0",
                "challengerModelId": null,
                "selfplayManifest": null,
                "teacherLabelingManifest": null,
                "trainingRunManifest": null,
                "arenaManifest": null,
                "promotionDecision": null,
                "status": "complete",
                "createdAt": "2026-08-08T00:00:00Z"
            }]
        })
    }

    fn active_registry_value() -> serde_json::Value {
        let mut value = canonical_registry_value();
        value["revision"] = serde_json::json!(2);
        value["challengerModelId"] = serde_json::json!("challenger-v1");
        value["models"]
            .as_array_mut()
            .unwrap()
            .push(serde_json::json!({
                "modelId": "challenger-v1",
                "generationId": "generation-1",
                "parentModelId": "champion-v0",
                "artifact": artifact_value("weights/challenger.osaval", "b"),
                "evaluatorKind": "neural",
                "architectureVersion": "1",
                "quantization": "float32",
                "trainingRun": artifact_value("artifacts/training.json", "c"),
                "registeredAt": "2026-08-09T00:00:00Z",
                "licenseStatus": "pending-review"
            }));
        value["generations"]
            .as_array_mut()
            .unwrap()
            .push(serde_json::json!({
                "generationId": "generation-1",
                "parentGenerationId": "generation-0",
                "championModelId": "champion-v0",
                "challengerModelId": "challenger-v1",
                "selfplayManifest": artifact_value("artifacts/selfplay.json", "d"),
                "teacherLabelingManifest": artifact_value("artifacts/teacher.json", "e"),
                "trainingRunManifest": artifact_value("artifacts/training.json", "c"),
                "arenaManifest": null,
                "promotionDecision": null,
                "status": "arena",
                "createdAt": "2026-08-09T00:00:00Z"
            }));
        value
    }

    fn finalized_registry_value(promoted: bool, decision: &str) -> serde_json::Value {
        let mut value = active_registry_value();
        value["revision"] = serde_json::json!(3);
        value["championModelId"] = if promoted {
            serde_json::json!("challenger-v1")
        } else {
            serde_json::json!("champion-v0")
        };
        value["challengerModelId"] = serde_json::Value::Null;
        value["generations"][1]["status"] = serde_json::json!("complete");
        value["generations"][1]["arenaManifest"] = artifact_value("artifacts/arena.json", "f");
        value["generations"][1]["promotionDecision"] =
            artifact_value(&format!("artifacts/promotion-{decision}.json"), "1");
        value
    }

    fn append_completed_generation(
        value: &mut serde_json::Value,
        generation_id: &str,
        challenger_id: &str,
        hash_digit: &str,
    ) {
        let parent_generation_id = value["generations"]
            .as_array()
            .and_then(|generations| generations.last())
            .and_then(|generation| generation["generationId"].as_str())
            .expect("fixture has a preceding generation")
            .to_owned();
        let training_path = format!("artifacts/{generation_id}-training.json");
        value["models"]
            .as_array_mut()
            .unwrap()
            .push(serde_json::json!({
                "modelId": challenger_id,
                "generationId": generation_id,
                "parentModelId": "champion-v0",
                "artifact": artifact_value(
                    &format!("weights/{challenger_id}.osaval"),
                    hash_digit,
                ),
                "evaluatorKind": "neural",
                "architectureVersion": "1",
                "quantization": "float32",
                "trainingRun": artifact_value(&training_path, hash_digit),
                "registeredAt": "2026-08-10T00:00:00Z",
                "licenseStatus": "pending-review"
            }));
        value["generations"]
            .as_array_mut()
            .unwrap()
            .push(serde_json::json!({
                "generationId": generation_id,
                "parentGenerationId": parent_generation_id,
                "championModelId": "champion-v0",
                "challengerModelId": challenger_id,
                "selfplayManifest": artifact_value(
                    &format!("artifacts/{generation_id}-selfplay.json"),
                    hash_digit,
                ),
                "teacherLabelingManifest": artifact_value(
                    &format!("artifacts/{generation_id}-teacher.json"),
                    hash_digit,
                ),
                "trainingRunManifest": artifact_value(&training_path, hash_digit),
                "arenaManifest": artifact_value(
                    &format!("artifacts/{generation_id}-arena.json"),
                    hash_digit,
                ),
                "promotionDecision": artifact_value(
                    &format!("artifacts/{generation_id}-promotion.json"),
                    hash_digit,
                ),
                "status": "complete",
                "createdAt": "2026-08-10T00:00:00Z"
            }));
    }

    fn write_finalized_registry_fixture(root: &Path) -> (PathBuf, String, String) {
        std::fs::create_dir_all(root.join(".git")).unwrap();
        std::fs::create_dir_all(root.join("weights")).unwrap();
        std::fs::create_dir_all(root.join("artifacts")).unwrap();
        let model = test_model_bytes();
        let model_sha256 = sha256_bytes(&model);
        std::fs::write(root.join("weights/champion.osaval"), &model).unwrap();
        std::fs::write(root.join("weights/challenger.osaval"), &model).unwrap();
        for path in [
            "artifacts/training.json",
            "artifacts/selfplay.json",
            "artifacts/teacher.json",
            "artifacts/arena.json",
        ] {
            std::fs::write(root.join(path), []).unwrap();
        }

        let mut value = finalized_registry_value(false, "rejected");
        value["models"][0]["quantization"] = serde_json::json!("float32");
        let model_artifact = serde_json::json!({"sha256": model_sha256, "size": model.len()});
        for (index, path) in ["weights/champion.osaval", "weights/challenger.osaval"]
            .into_iter()
            .enumerate()
        {
            value["models"][index]["artifact"] = serde_json::json!({
                "path": path,
                "sha256": model_artifact["sha256"],
                "size": model_artifact["size"],
            });
        }
        value["models"][1]["trainingRun"] = empty_artifact("artifacts/training.json");
        value["generations"][1]["selfplayManifest"] = empty_artifact("artifacts/selfplay.json");
        value["generations"][1]["teacherLabelingManifest"] =
            empty_artifact("artifacts/teacher.json");
        value["generations"][1]["trainingRunManifest"] = empty_artifact("artifacts/training.json");
        value["generations"][1]["arenaManifest"] = empty_artifact("artifacts/arena.json");
        write_promotion_evidence_fixture(root, &mut value);

        let encoded = serde_json::to_vec(&value).unwrap();
        let registry_sha256 = sha256_bytes(&encoded);
        let registry_path = root.join("artifacts/model-registry.json");
        std::fs::write(&registry_path, encoded).unwrap();
        (registry_path, registry_sha256, model_sha256)
    }

    fn empty_artifact(path: &str) -> serde_json::Value {
        serde_json::json!({
            "path": path,
            "sha256": sha256_bytes(&[]),
            "size": 0,
        })
    }

    fn write_fixture_artifact(root: &Path, path: &str, bytes: &[u8]) -> RegistryArtifact {
        let destination = root.join(path);
        std::fs::create_dir_all(destination.parent().unwrap()).unwrap();
        std::fs::write(&destination, bytes).unwrap();
        RegistryArtifact {
            path: PathBuf::from(path),
            sha256: sha256_bytes(bytes),
            size: bytes.len() as u64,
        }
    }

    fn write_fixture_engine_build(
        root: &Path,
        git_commit: &str,
    ) -> (RegistryArtifact, RegistryArtifact) {
        let engine_bytes = b"fixture open-shogi-cli";
        let engine_sha256 = sha256_bytes(engine_bytes);
        let engine_path = format!(
            "local/builds/open-shogi-cli/{engine_sha256}/open-shogi-cli"
        );
        let engine = write_fixture_artifact(root, &engine_path, engine_bytes);
        let receipt_commit = format!(
            "{git_commit}{}",
            "0".repeat(40_usize.saturating_sub(git_commit.len()))
        );
        let mut receipt = serde_json::json!({
            "schema": "open_shogi_engine_build_receipt/v2",
            "gitCommit": receipt_commit,
            "binary": engine,
            "sourceTreeSha256": sha256_bytes(b"fixture source tree"),
            "sourceFiles": 1,
            "sourceBytes": 1,
            "buildCommand": [
                "cargo", "build", "--locked", "--release", "--offline", "--jobs", "4",
                "-p", "open-shogi-cli"
            ],
            "cargoTool": {
                "path": "/fixture/bin/cargo",
                "sha256": sha256_bytes(b"fixture cargo"),
                "size": 1,
                "version": "cargo fixture"
            },
            "rustcTool": {
                "path": "/fixture/bin/rustc",
                "sha256": sha256_bytes(b"fixture rustc"),
                "size": 1,
                "version": "rustc fixture"
            },
            "rustcRuntimeTree": {
                "treeSha256": sha256_bytes(b"fixture rustc runtime tree"),
                "files": 2,
                "bytes": 2
            }
        });
        let receipt_sha256 = python_canonical_sha256(&receipt).unwrap();
        receipt["receiptSha256"] = serde_json::json!(receipt_sha256);
        let receipt_path = format!(
            "local/build-receipts/open-shogi-cli/{receipt_sha256}.json"
        );
        let receipt_reference = write_fixture_json(root, &receipt_path, &receipt);
        (engine, receipt_reference)
    }

    fn write_fixture_json(
        root: &Path,
        path: &str,
        value: &serde_json::Value,
    ) -> RegistryArtifact {
        let mut bytes = serde_json::to_vec(value).unwrap();
        bytes.push(b'\n');
        write_fixture_artifact(root, path, &bytes)
    }

    fn self_hashed_value(mut value: serde_json::Value, field: &str) -> serde_json::Value {
        let digest = python_canonical_sha256(&value).unwrap();
        value[field] = serde_json::json!(digest);
        value
    }

    fn refresh_fixture_self_hash(value: &mut serde_json::Value, field: &str) {
        let mut unsigned = value.as_object().unwrap().clone();
        unsigned.remove(field);
        value[field] = serde_json::json!(
            python_canonical_sha256(&serde_json::Value::Object(unsigned)).unwrap()
        );
    }

    fn active_registry_snapshot(registry: &serde_json::Value) -> serde_json::Value {
        let mut snapshot = registry.clone();
        snapshot["revision"] = serde_json::json!(2);
        snapshot["championModelId"] = serde_json::json!("champion-v0");
        snapshot["challengerModelId"] = serde_json::json!("challenger-v1");
        snapshot["generations"][1]["status"] = serde_json::json!("arena");
        snapshot["generations"][1]["arenaManifest"] = serde_json::Value::Null;
        snapshot["generations"][1]["promotionDecision"] = serde_json::Value::Null;
        snapshot
    }

    #[derive(Clone)]
    struct FixturePhase3Game {
        object_id: String,
        raw_csa: String,
        normalized_csa: String,
        raw_sha256: String,
        canonical_sha256: String,
        split: String,
        usi_moves: Vec<String>,
        sfens: Vec<String>,
    }

    fn fixture_phase3_games() -> Vec<FixturePhase3Game> {
        let split_policy = Phase3SplitPolicy {
            schema: "phase3_game_split/v1".to_owned(),
            salt: "fixture".to_owned(),
            salt_sha256: sha256_bytes(b"fixture"),
            test_basis_points: 0,
            validation_basis_points: 5_000,
        };
        let initial = Position::startpos();
        let first_moves = initial.legal_moves().into_iter().take(20).collect::<Vec<_>>();
        assert_eq!(first_moves.len(), 20);
        first_moves
            .into_iter()
            .enumerate()
            .map(|(index, first)| {
                let desired_split = if index < 10 { "train" } else { "validation" };
                let mut replay = initial.clone();
                replay.make_move(first).unwrap();
                let second = replay.legal_moves().into_iter().next().unwrap();
                replay.make_move(second).unwrap();
                let third = replay.legal_moves().into_iter().next().unwrap();
                let moves = vec![first, second, third];
                let (normalized_csa, canonical_sha256, split, nonce) = (0..10_000)
                    .find_map(|nonce| {
                        let game = CsaGame {
                            version: "V3.0".to_owned(),
                            black_name: None,
                            white_name: None,
                            metadata: vec![(
                                "FIXTURE_ID".to_owned(),
                                format!("{index:03}-{nonce:04}"),
                            )],
                            initial_position: initial.clone(),
                            moves: moves.clone(),
                            special_move: Some(CsaSpecialMove::MaxMoves),
                            result_validation: CsaResultValidation::ExternalCondition,
                        };
                        let csa = to_csa_game(&game).unwrap();
                        let hash = sha256_bytes(csa.as_bytes());
                        let assigned = phase3_game_split(&hash, &split_policy).unwrap();
                        (assigned == desired_split).then_some((csa, hash, assigned, nonce))
                    })
                    .expect("fixture nonce must produce the requested split");
                let mut raw_lines = vec![
                    format!("' fixture acquisition comment {index}"),
                    format!("$FIXTURE_ID:{index:03}-{nonce:04}"),
                    "PI".to_owned(),
                    "+".to_owned(),
                ];
                let mut raw_replay = initial.clone();
                for (move_index, movement) in moves.iter().copied().enumerate() {
                    let notation = open_shogi_core::to_csa_move(&raw_replay, movement).unwrap();
                    raw_lines.push(match move_index {
                        0 => format!("{notation},v=fixture"),
                        1 => format!("{notation},' fixture principal variation"),
                        _ => notation,
                    });
                    if move_index == 0 {
                        raw_lines.push("T0.001".to_owned());
                    }
                    raw_replay.make_move(movement).unwrap();
                }
                raw_lines.push("%MAX_MOVES".to_owned());
                raw_lines.push("' retained trailing comment".to_owned());
                let raw_csa = format!("{}\n", raw_lines.join("\n"));
                let raw_sha256 = sha256_bytes(raw_csa.as_bytes());
                let mut position = initial.clone();
                let mut sfens = vec![to_sfen(&position)];
                for movement in &moves {
                    position.make_move(*movement).unwrap();
                    sfens.push(to_sfen(&position));
                }
                FixturePhase3Game {
                    object_id: format!("fixture-game-{index:03}"),
                    raw_csa,
                    normalized_csa,
                    raw_sha256,
                    canonical_sha256,
                    split,
                    usi_moves: moves.into_iter().map(to_usi_move).collect(),
                    sfens,
                }
            })
            .collect()
    }

    fn reset_fixture_move_number(sfen: &str) -> String {
        let mut fields = sfen.split(' ');
        let board = fields.next().unwrap();
        let side = fields.next().unwrap();
        let hands = fields.next().unwrap();
        assert!(fields.next().is_some());
        assert!(fields.next().is_none());
        format!("{board} {side} {hands} 1")
    }

    fn fixture_start_positions() -> Vec<Phase6StartPosition> {
        let mut starts = fixture_phase3_games()
            .into_iter()
            .map(|game| {
                let mut row = Phase6StartPosition {
                    position_id: String::new(),
                    sfen: reset_fixture_move_number(&game.sfens[1]),
                    source_game_sha256: game.canonical_sha256,
                    position_index: 1,
                    split: game.split,
                };
                row.position_id = phase6_start_position_identity(&row).unwrap();
                row
            })
            .collect::<Vec<_>>();
        starts.sort_by(|left, right| left.position_id.cmp(&right.position_id));
        starts
    }

    fn gzip_fixture_jsonl(rows: &[serde_json::Value]) -> Vec<u8> {
        let mut jsonl = Vec::new();
        for row in rows {
            write_python_canonical_json(row, &mut jsonl).unwrap();
            jsonl.push(b'\n');
        }
        let mut encoder = flate2::write::GzEncoder::new(
            Vec::new(),
            flate2::Compression::default(),
        );
        encoder.write_all(&jsonl).unwrap();
        encoder.finish().unwrap()
    }

    #[expect(
        clippy::too_many_lines,
        reason = "the test fixture materializes one complete closed Phase 3 source chain"
    )]
    fn write_start_evidence(
        root: &Path,
        engine: &RegistryArtifact,
        engine_build_receipt: &RegistryArtifact,
        starts: &[Phase6StartPosition],
    ) -> (RegistryArtifact, RegistryArtifact, RegistryArtifact) {
        let games = fixture_phase3_games();
        let expected_starts = fixture_start_positions();
        assert_eq!(starts, expected_starts);
        let evidence_bytes = b"fixture license evidence";
        let evidence_sha256 = sha256_bytes(evidence_bytes);
        let evidence_snapshot = serde_json::json!({
            "evidence_id": "fixture-evidence",
            "url": "https://example.invalid/fixture/license",
            "retrieved_at": "2026-08-09T00:00:00Z",
            "sha256": evidence_sha256,
            "size": evidence_bytes.len(),
            "content_type": "text/plain",
            "object_path": format!(
                "evidence/sha256/{}/{evidence_sha256}",
                &evidence_sha256[..2]
            )
        });
        let license_evidence = serde_json::json!([{
            "url": "https://example.invalid/fixture/license",
            "local_path": "evidence/license.txt",
            "quote": "fixture license evidence"
        }]);
        let game_rows = games
            .iter()
            .map(|game| {
                serde_json::json!({
                    "schema": "phase3_game/v1",
                    "gameId": game.canonical_sha256,
                    "canonicalSha256": game.canonical_sha256,
                    "rawObject": {
                        "objectId": game.object_id,
                        "objectPath": format!(
                            "objects/sha256/{}/{}",
                            &game.raw_sha256[..2], game.raw_sha256
                        ),
                        "originalFilename": format!("{}.csa", game.object_id),
                        "sha256": game.raw_sha256,
                        "size": game.raw_csa.len(),
                        "response": {
                            "contentType": "application/x-csa",
                            "etag": null,
                            "lastModified": null
                        }
                    },
                    "rawCsa": game.raw_csa,
                    "normalizedCsa": game.normalized_csa,
                    "initialSfen": game.sfens[0],
                    "usiMoves": game.usi_moves,
                    "plyCount": 3,
                    "positionCount": 4,
                    "outcome": "unknown",
                    "terminalReason": "MAX_MOVES",
                    "resultValidation": "external_condition",
                    "players": {
                        "black": {"name": null, "rating": null},
                        "white": {"name": null, "rating": null}
                    },
                    "date": null,
                    "sourceDateTime": null,
                    "sourceTimeZone": null,
                    "split": game.split,
                    "flags": {"short": true, "long": false},
                    "sourceId": "fixture-source",
                    "url": format!(
                        "https://example.invalid/fixture/games/{}",
                        game.object_id
                    ),
                    "retrievedAt": "2026-08-09T00:00:00Z",
                    "licenseDecision": {
                        "license": "fixture-license",
                        "evidence": license_evidence,
                        "evidenceSnapshots": [evidence_snapshot],
                        "redistributable": false,
                        "machineLearningAllowed": true
                    }
                })
            })
            .collect::<Vec<_>>();
        let mut position_rows = Vec::with_capacity(games.len() * 4);
        for game in &games {
            for (index, sfen) in game.sfens.iter().enumerate() {
                let move_usi = game.usi_moves.get(index).cloned();
                let next_sfen = game.sfens.get(index + 1).cloned();
                let terminal_tail = index >= 2;
                position_rows.push(serde_json::json!({
                    "schema": "phase3_position/v1",
                    "gameId": game.canonical_sha256,
                    "canonicalSha256": game.canonical_sha256,
                    "rawSha256": game.raw_sha256,
                    "sourceId": "fixture-source",
                    "split": game.split,
                    "positionIndex": index,
                    "sfen": sfen,
                    "moveUsi": move_usi,
                    "nextSfen": next_sfen,
                    "outcome": "unknown",
                    "terminalReason": "MAX_MOVES",
                    "sideToMove": if sfen.split(' ').nth(1) == Some("b") {
                        "black"
                    } else {
                        "white"
                    },
                    "fullPlies": 3,
                    "remainingPlies": 3 - index,
                    "eligible": move_usi.is_some() && !terminal_tail,
                    "terminalTail": terminal_tail
                }));
            }
        }
        let game_bytes = gzip_fixture_jsonl(&game_rows);
        let position_bytes = gzip_fixture_jsonl(&position_rows);
        let games_ref = write_fixture_artifact(
            root,
            "artifacts/phase3/games-00000.jsonl.gz",
            &game_bytes,
        );
        let source = write_fixture_artifact(
            root,
            "artifacts/phase3/positions-00000.jsonl.gz",
            &position_bytes,
        );
        let report = write_fixture_json(
            root,
            "artifacts/phase3/normalization-report.json",
            &serde_json::json!({"schema": "phase3_normalization_report/v1"}),
        );
        let mut raw_hashes = games
            .iter()
            .map(|game| game.raw_sha256.clone())
            .collect::<Vec<_>>();
        raw_hashes.sort();
        let mut canonical_hashes = games
            .iter()
            .map(|game| game.canonical_sha256.clone())
            .collect::<Vec<_>>();
        canonical_hashes.sort();
        let dataset = write_fixture_json(
            root,
            "artifacts/phase3/manifest.json",
            &serde_json::json!({
                "schema": "phase3_dataset_manifest/v1",
                "datasetId": "fixture-source-dataset",
                "source": {
                    "sourceId": "fixture-source",
                    "name": "fixture",
                    "officialBase": "https://example.invalid/fixture",
                    "adapter": "aobazero_csa",
                    "license": "fixture-license",
                    "licenseEvidence": license_evidence,
                    "redistributable": false,
                    "machineLearningAllowed": true,
                    "lastReviewed": "2026-08-09"
                },
                "config": {
                    "schema": "phase3_normalization_config/v1",
                    "datasetId": "fixture-source-dataset",
                    "exporterTimeoutSeconds": 1,
                    "maxGames": 20,
                    "maxPositions": 80,
                    "maxRawBytes": 1_048_576,
                    "split": {
                        "schema": "phase3_game_split/v1",
                        "salt": "fixture",
                        "saltSha256": sha256_bytes(b"fixture"),
                        "testBasisPoints": 0,
                        "validationBasisPoints": 5000
                    },
                    "terminalTailPositions": 1
                },
                "counts": {"games": 20, "positions": 80},
                "artifacts": {
                    "games-00000.jsonl.gz": {
                        "sha256": games_ref.sha256,
                        "size": games_ref.size,
                        "records": 20
                    },
                    "normalization-report.json": {
                        "sha256": report.sha256,
                        "size": report.size,
                        "records": 1
                    },
                    "positions-00000.jsonl.gz": {
                        "sha256": source.sha256,
                        "size": source.size,
                        "records": 80
                    }
                },
                "rawObjectSha256": raw_hashes,
                "canonicalGameSha256": canonical_hashes,
                "evidenceSnapshots": [evidence_snapshot]
            }),
        );
        let starts_value = serde_json::json!({
            "schema": "phase6_start_positions/v1",
            "datasetManifest": dataset,
            "sourcePositions": source,
            "selection": {
                "seed": 20_260_808,
                "trainCount": 10,
                "validationCount": 10,
                "requireEligible": true,
                "resetMoveNumber": true,
                "excludedCrossSplitStates": 1
            },
            "positions": starts,
        });
        let starts_ref =
            write_fixture_json(root, "artifacts/start-positions.json", &starts_value);
        let stdout = write_fixture_artifact(root, "artifacts/start.stdout.log", b"");
        let stderr = write_fixture_artifact(root, "artifacts/start.stderr.log", b"");
        let rows = starts
            .iter()
            .map(|position| {
                serde_json::json!({
                    "positionId": position.position_id,
                    "sfenSha256": sha256_bytes(position.sfen.as_bytes()),
                    "legal": true,
                    "returnCode": 0,
                    "timedOut": false,
                    "outputLimitExceeded": false,
                    "memoryLimitExceeded": false,
                    "peakRssBytes": 0,
                    "rssMeasurement": "process_tree_ps_short_lived_no_sample",
                    "stdout": stdout,
                    "stderr": stderr,
                    "completedAt": "2026-08-09T00:00:00Z"
                })
            })
            .collect::<Vec<_>>();
        let validation = serde_json::json!({
            "schema": "phase6_start_position_validation/v1",
            "startPositions": starts_ref,
            "engine": engine,
            "engineBuildReceipt": engine_build_receipt,
            "gitCommit": "abcdef0",
            "method": "open-shogi-cli-perft-depth-0",
            "results": rows,
        });
        let validation_ref =
            write_fixture_json(root, "artifacts/start-validation.json", &validation);
        (starts_ref, validation_ref, dataset)
    }

    fn phase6_fixture_config() -> &'static [u8] {
        br#"schema_version = 1

[run]
games = 40
normal_start_pairs = 10
start_set_pairs = 10
seed = 20260808
nodes_per_move = 500
max_plies = 256
search_depth = 6
hash_mib = 64

[resources]
workers = 2
memory_limit_mib = 8192
memory_per_worker_mib = 3072
game_timeout_seconds = 1800
command_timeout_seconds = 3600

[paths]
engine_cli = "artifacts/open-shogi-cli"
output_root = "artifacts"
start_positions_manifest = "artifacts/start-positions.json"

[hard_positions]
teacher_drop_cp = 150
evaluation_disagreement_cp = 200
candidate_gap_cp = 40
minimum_search_nodes = 500
max_additional_labels = 500
teacher_label_limit = 10000

[replay]
capacity = 100000
minimum_older_positions = 1000
dedup_key = "canonical_sfen"
"#
    }

    fn report_player(
        artifact: &RegistryArtifact,
        payload_sha256: &str,
    ) -> serde_json::Value {
        serde_json::json!({
            "label": format!(
                "search:neural:d6:h64:tt-on:book-off:m-{}",
                &artifact.sha256[..12]
            ),
            "evaluatorKind": "neural",
            "searchDepth": 6,
            "hashMegabytes": 64,
            "transposition": true,
            "modelArtifactSha256": artifact.sha256,
            "modelArtifactSize": artifact.size,
            "modelPayloadSha256": payload_sha256,
            "architectureVersion": 1,
            "quantization": "float32",
            "openingEnabled": false
        })
    }

    #[expect(
        clippy::too_many_lines,
        reason = "the closed v2 report fixture lists every required aggregate and game counter"
    )]
    fn write_pair_report(
        root: &Path,
        job: &serde_json::Value,
        challenger: &RegistryArtifact,
        champion: &RegistryArtifact,
        payload_sha256: &str,
    ) -> (RegistryArtifact, Vec<RegistryArtifact>) {
        let player_a = report_player(challenger, payload_sha256);
        let player_b = report_player(champion, payload_sha256);
        let player_a_parsed: super::Phase2ArenaPlayer =
            serde_json::from_value(player_a.clone()).unwrap();
        let player_b_parsed: super::Phase2ArenaPlayer =
            serde_json::from_value(player_b.clone()).unwrap();
        let initial_sfen = job["sfen"].as_str().unwrap();
        let initial = open_shogi_core::parse_sfen(initial_sfen).unwrap();
        let mut csa = Vec::new();
        let mut games = Vec::new();
        let mut total_first_player_searches = 0_u64;
        let mut total_second_player_searches = 0_u64;
        for (index, winner) in [Side::White, Side::Black].into_iter().enumerate() {
            let (black, white) = if index == 0 {
                (
                    player_a["label"].as_str().unwrap(),
                    player_b["label"].as_str().unwrap(),
                )
            } else {
                (
                    player_b["label"].as_str().unwrap(),
                    player_a["label"].as_str().unwrap(),
                )
            };
            let mut replay = initial.clone();
            let mut moves = Vec::new();
            if replay.side_to_move() == winner {
                let movement = replay.legal_moves().into_iter().next().unwrap();
                replay.make_move(movement).unwrap();
                moves.push(movement);
            }
            assert_eq!(replay.side_to_move(), winner.opposite());
            let text = to_csa_game(&CsaGame {
                version: "V3.0".to_owned(),
                black_name: Some(black.to_owned()),
                white_name: Some(white.to_owned()),
                metadata: Vec::new(),
                initial_position: initial.clone(),
                moves: moves.clone(),
                special_move: Some(CsaSpecialMove::Resign),
                result_validation: CsaResultValidation::ExternalCondition,
            })
            .unwrap();
            let reference = write_fixture_artifact(
                root,
                job["csaPaths"][index].as_str().unwrap(),
                text.as_bytes(),
            );
            let selections = u64::try_from(moves.len()).unwrap() + 1;
            let (black_searches, white_searches) =
                super::selection_counts(initial.side_to_move(), selections);
            let (first_player_searches, second_player_searches) = if index == 0 {
                (black_searches, white_searches)
            } else {
                (white_searches, black_searches)
            };
            total_first_player_searches += first_player_searches;
            total_second_player_searches += second_player_searches;
            games.push(serde_json::json!({
                "id": index,
                "black": black,
                "white": white,
                "result": if winner == Side::Black { "black_win" } else { "white_win" },
                "moves": moves.len(),
                "csaPath": format!("games/game-{:06}.csa", index + 1),
                "csaSha256": reference.sha256,
                "csaSize": reference.size,
                "neuralInferenceCalls": 0,
                "neuralInferenceTimeNs": 0,
                "playerASearchNodes": first_player_searches,
                "playerASearchElapsedMs": 0,
                "playerADepthSum": first_player_searches,
                "playerASearches": first_player_searches,
                "playerANeuralInferenceCalls": 0,
                "playerANeuralInferenceTimeNs": 0,
                "playerBSearchNodes": second_player_searches,
                "playerBSearchElapsedMs": 0,
                "playerBDepthSum": second_player_searches,
                "playerBSearches": second_player_searches,
                "playerBNeuralInferenceCalls": 0,
                "playerBNeuralInferenceTimeNs": 0
            }));
            csa.push(reference);
        }
        assert_eq!(total_first_player_searches, 2);
        assert_eq!(total_second_player_searches, 1);
        let signature = super::ArenaConfigSignature {
            games: 2,
            seed: job["seed"].as_u64().unwrap(),
            initial_sfen: initial_sfen.to_owned(),
            max_plies: 256,
            git_commit: Some("abcdef0".to_owned()),
            budget_kind: "nodes".to_owned(),
            budget_value: 500,
            player_a: super::phase2_signature_player(&player_a_parsed),
            player_b: super::phase2_signature_player(&player_b_parsed),
            opening: super::ArenaConfigSignatureOpening {
                enabled: false,
                artifact_sha256: None,
                artifact_size: None,
                max_plies: None,
            },
        };
        let config_sha256 = sha256_bytes(&super::arena_config_signature_bytes(&signature));
        let report = serde_json::json!({
            "schema": "phase2_arena_report/v2",
            "run": {
                "seed": job["seed"],
                "gameLimit": 2,
                "engine": format!(
                    "OpenShogiAI 0.0.0 a={} b={} budget=Nodes(500)",
                    player_a["label"].as_str().unwrap(),
                    player_b["label"].as_str().unwrap()
                ),
                "gitCommit": "abcdef0",
                "startedAt": "2026-08-09T00:00:00Z",
                "completedAt": "2026-08-09T00:00:01Z",
                "initialSfen": job["sfen"],
                "maxPlies": 256,
                "configSha256": config_sha256,
                "budget": {"kind": "nodes", "value": 500},
                "playerA": player_a,
                "playerB": player_b,
                "opening": {
                    "enabled": false,
                    "artifactSha256": null,
                    "artifactSize": null,
                    "maxPlies": null
                }
            },
            "metrics": {
                "games": 2,
                "finishedGames": 2,
                "playerAWins": 0,
                "playerBWins": 2,
                "searchWins": 2,
                "draws": 0,
                "nodesPerSecond": 0.0,
                "averageDepth": 1.0,
                "ttHitRate": 0.0,
                "cutoffRate": 0.0,
                "pruningRate": 0.0,
                "millisecondsPerMove": 0.0,
                "neuralInferenceCalls": 0,
                "neuralInferenceTimeNs": 0,
                "playerASearchNodes": total_first_player_searches,
                "playerASearchElapsedMs": 0,
                "playerADepthSum": total_first_player_searches,
                "playerASearches": total_first_player_searches,
                "playerANeuralInferenceCalls": 0,
                "playerANeuralInferenceTimeNs": 0,
                "playerBSearchNodes": total_second_player_searches,
                "playerBSearchElapsedMs": 0,
                "playerBDepthSum": total_second_player_searches,
                "playerBSearches": total_second_player_searches,
                "playerBNeuralInferenceCalls": 0,
                "playerBNeuralInferenceTimeNs": 0,
                "peakMemoryBytes": null,
                "illegalMoves": 0
            },
            "games": games
        });
        let reference = write_fixture_json(
            root,
            job["reportPath"].as_str().unwrap(),
            &report,
        );
        (reference, csa)
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "the fixture binds one terminal attempt to every retained process artifact"
    )]
    fn write_completed_attempt_command_receipt(
        root: &Path,
        job: &serde_json::Value,
        plan_sha256: &str,
        engine: &RegistryArtifact,
        engine_build_receipt: &RegistryArtifact,
        stdout: &RegistryArtifact,
        stderr: &RegistryArtifact,
        report: &RegistryArtifact,
        csa: &[RegistryArtifact],
    ) -> RegistryArtifact {
        let invocation = serde_json::json!({
            "command": job["command"],
            "resume": false,
            "memoryLimitMiB": 3072,
            "expectedExecutable": engine,
            "engineBuildReceipt": engine_build_receipt,
            "runtimeReceipt": null
        });
        let command_sha256 = python_canonical_sha256(&invocation).unwrap();
        let process_receipt = serde_json::json!({
            "schema": "phase6_command_receipt/v2",
            "command": job["command"],
            "commandSha256": command_sha256,
            "resume": false,
            "memoryLimitMiB": 3072,
            "expectedExecutable": engine,
            "engineBuildReceipt": engine_build_receipt,
            "runtimeReceipt": null,
            "returnCode": 0,
            "timedOut": false,
            "outputLimitExceeded": false,
            "memoryLimitExceeded": false,
            "peakRssBytes": 0,
            "rssMeasurement": "process_tree_ps_short_lived_no_sample",
            "stdout": stdout,
            "stderr": stderr
        });
        let job_id = job["jobId"].as_str().unwrap();
        let process_reference = write_fixture_json(
            root,
            &format!(
                "artifacts/generation-1/arena/evidence/{job_id}/attempt-001.process.json"
            ),
            &process_receipt,
        );
        let attempt_receipt = self_hashed_value(
            serde_json::json!({
                "schema": "phase6_attempt_command_receipt/v1",
                "planSha256": plan_sha256,
                "jobId": job_id,
                "attempt": 1,
                "command": job["command"],
                "engine": engine,
                "engineBuildReceipt": engine_build_receipt,
                "processReceipt": process_reference,
                "result": {
                    "returnCode": 0,
                    "timedOut": false,
                    "outputLimitExceeded": false,
                    "memoryLimitExceeded": false,
                    "peakRssBytes": 0,
                    "rssMeasurement": "process_tree_ps_short_lived_no_sample"
                },
                "stdout": stdout,
                "stderr": stderr,
                "report": report,
                "csa": csa,
                "quarantine": null,
                "failureCategory": null,
                "completedAt": "2026-08-09T00:00:01Z"
            }),
            "receiptSha256",
        );
        write_fixture_json(
            root,
            &format!(
                "artifacts/generation-1/arena/evidence/{job_id}/attempt-001.receipt.json"
            ),
            &attempt_receipt,
        )
    }

    fn read_fixture_plan_and_execution(
        root: &Path,
    ) -> (super::PairedArenaPlan, serde_json::Value) {
        let plan_value = parse_unique_json(
            &std::fs::read(root.join("artifacts/plan.json")).unwrap(),
            "fixture paired plan",
        )
        .unwrap();
        let plan = deserialize_closed_json(&plan_value, "fixture paired plan").unwrap();
        let execution = parse_unique_json(
            &std::fs::read(root.join("artifacts/execution.json")).unwrap(),
            "fixture paired execution",
        )
        .unwrap();
        (plan, execution)
    }

    fn validate_fixture_execution_value(
        root: &Path,
        plan: &super::PairedArenaPlan,
        execution_value: &serde_json::Value,
    ) -> Result<(), String> {
        let execution: super::PairedArenaExecution =
            deserialize_closed_json(execution_value, "fixture paired execution")?;
        let plan_reference = execution.plan.clone();
        let storage = super::RegistryStorage {
            root: open_shogi_core::AnchoredDir::open_existing(root)
                .map_err(|error| error.to_string())?,
            root_path: root.to_path_buf(),
            registry_relative: PathBuf::from("artifacts/unused-registry.json"),
        };
        super::validate_paired_arena_execution(
            &storage,
            &execution,
            execution_value,
            plan,
            &plan_reference,
            &mut BTreeMap::new(),
            &mut RegistryVerificationBudget::with_limits(10_000, 100_000_000),
        )
    }

    fn validate_fixture_attempt_value(
        root: &Path,
        plan: &super::PairedArenaPlan,
        attempt_value: &serde_json::Value,
    ) -> Result<(), String> {
        let attempt: super::PairedArenaAttempt =
            deserialize_closed_json(attempt_value, "fixture paired attempt")?;
        let jobs = plan
            .jobs
            .iter()
            .map(|job| (job.job_id.as_str(), job))
            .collect::<BTreeMap<_, _>>();
        let worker_memory_bytes = plan
            .memory_per_worker_mib
            .checked_mul(1024 * 1024)
            .unwrap();
        let storage = super::RegistryStorage {
            root: open_shogi_core::AnchoredDir::open_existing(root)
                .map_err(|error| error.to_string())?,
            root_path: root.to_path_buf(),
            registry_relative: PathBuf::from("artifacts/unused-registry.json"),
        };
        super::validate_paired_arena_attempt(
            &storage,
            &attempt,
            plan,
            &jobs,
            worker_memory_bytes,
            &mut BTreeMap::new(),
            &mut RegistryVerificationBudget::with_limits(10_000, 100_000_000),
        )
    }

    fn validate_fixture_start_validation_value(
        root: &Path,
        plan: &super::PairedArenaPlan,
        validation_value: &serde_json::Value,
    ) -> Result<(), String> {
        let starts_value = parse_unique_json(
            &std::fs::read(root.join(&plan.start_positions.path))
                .map_err(|error| error.to_string())?,
            "fixture start positions",
        )?;
        let starts: super::Phase6StartPositions =
            deserialize_closed_json(&starts_value, "fixture start positions")?;
        let validation: super::Phase6StartValidation =
            deserialize_closed_json(validation_value, "fixture start validation")?;
        let storage = super::RegistryStorage {
            root: open_shogi_core::AnchoredDir::open_existing(root)
                .map_err(|error| error.to_string())?,
            root_path: root.to_path_buf(),
            registry_relative: PathBuf::from("artifacts/unused-registry.json"),
        };
        super::validate_phase6_start_validation(
            &storage,
            plan,
            &starts,
            &validation,
            &mut BTreeMap::new(),
            &mut RegistryVerificationBudget::with_limits(10_000, 100_000_000),
        )
    }

    fn validate_fixture_quarantine_record_value(
        root: &Path,
        plan: &super::PairedArenaPlan,
        attempt_value: &serde_json::Value,
        record_value: &serde_json::Value,
    ) -> Result<(), String> {
        let attempt: super::PairedArenaAttempt =
            deserialize_closed_json(attempt_value, "fixture quarantined attempt")?;
        let record: super::PairedQuarantineRecord =
            deserialize_closed_json(record_value, "fixture quarantine record")?;
        let job = plan
            .jobs
            .iter()
            .find(|job| job.job_id == attempt.job_id)
            .ok_or_else(|| "fixture quarantine job is absent".to_owned())?;
        let storage = super::RegistryStorage {
            root: open_shogi_core::AnchoredDir::open_existing(root)
                .map_err(|error| error.to_string())?,
            root_path: root.to_path_buf(),
            registry_relative: PathBuf::from("artifacts/unused-registry.json"),
        };
        super::validate_paired_quarantine_record(
            &storage,
            &record,
            &attempt,
            job,
            &mut BTreeMap::new(),
            &mut RegistryVerificationBudget::with_limits(10_000, 100_000_000),
        )
    }

    fn completed_attempt_with_number(
        root: &Path,
        original: &serde_json::Value,
        attempt_number: u64,
    ) -> serde_json::Value {
        let original_reference: RegistryArtifact =
            serde_json::from_value(original["commandReceipt"].clone()).unwrap();
        let mut receipt = parse_unique_json(
            &std::fs::read(root.join(original_reference.path)).unwrap(),
            "fixture completed command receipt",
        )
        .unwrap();
        receipt["attempt"] = serde_json::json!(attempt_number);
        refresh_fixture_self_hash(&mut receipt, "receiptSha256");
        let job_id = original["jobId"].as_str().unwrap();
        let reference = write_fixture_json(
            root,
            &format!(
                "artifacts/generation-1/arena/evidence/{job_id}/attempt-{attempt_number:03}.receipt.json"
            ),
            &receipt,
        );
        let mut changed = original.clone();
        changed["attempt"] = serde_json::json!(attempt_number);
        changed["commandReceipt"] = serde_json::to_value(reference).unwrap();
        changed
    }

    #[expect(
        clippy::too_many_lines,
        reason = "the fixture mirrors the complete quarantined attempt and receipt schemas"
    )]
    fn quarantined_attempt_fixture(
        root: &Path,
        plan: &super::PairedArenaPlan,
        original: &serde_json::Value,
    ) -> serde_json::Value {
        let job_id = original["jobId"].as_str().unwrap();
        let job = plan
            .jobs
            .iter()
            .find(|job| job.job_id == job_id)
            .unwrap();
        let stdout: RegistryArtifact =
            serde_json::from_value(original["stdout"].clone()).unwrap();
        let stderr: RegistryArtifact =
            serde_json::from_value(original["stderr"].clone()).unwrap();
        let quarantine = write_fixture_json(
            root,
            &format!("artifacts/generation-1/arena/quarantine/{job_id}-attempt-001.json"),
            &serde_json::json!({
                "schema": "phase6_quarantined_job/v1",
                "jobId": job_id,
                "attempt": 1,
                "failureCategory": "timeout",
                "returnCode": -15,
                "timedOut": true,
                "outputLimitExceeded": false,
                "memoryLimitExceeded": false,
                "peakRssBytes": 0,
                "rssMeasurement": "process_tree_ps_short_lived_no_sample",
                "stdout": stdout,
                "stderr": stderr,
                "observedReport": null,
                "observedCsa": [],
                "failureDetail": "fixture timeout"
            }),
        );
        let invocation = serde_json::json!({
            "command": job.command,
            "resume": false,
            "memoryLimitMiB": plan.memory_per_worker_mib,
            "expectedExecutable": plan.engine,
            "engineBuildReceipt": plan.engine_build_receipt,
            "runtimeReceipt": null
        });
        let process_receipt = serde_json::json!({
            "schema": "phase6_command_receipt/v2",
            "command": job.command,
            "commandSha256": python_canonical_sha256(&invocation).unwrap(),
            "resume": false,
            "memoryLimitMiB": plan.memory_per_worker_mib,
            "expectedExecutable": plan.engine,
            "engineBuildReceipt": plan.engine_build_receipt,
            "runtimeReceipt": null,
            "returnCode": -15,
            "timedOut": true,
            "outputLimitExceeded": false,
            "memoryLimitExceeded": false,
            "peakRssBytes": 0,
            "rssMeasurement": "process_tree_ps_short_lived_no_sample",
            "stdout": stdout,
            "stderr": stderr
        });
        let process_reference = write_fixture_json(
            root,
            &format!(
                "artifacts/generation-1/arena/evidence/{job_id}/attempt-001-quarantine.process.json"
            ),
            &process_receipt,
        );
        let attempt_receipt = self_hashed_value(
            serde_json::json!({
                "schema": "phase6_attempt_command_receipt/v1",
                "planSha256": plan.plan_sha256,
                "jobId": job_id,
                "attempt": 1,
                "command": job.command,
                "engine": plan.engine,
                "engineBuildReceipt": plan.engine_build_receipt,
                "processReceipt": process_reference,
                "result": {
                    "returnCode": -15,
                    "timedOut": true,
                    "outputLimitExceeded": false,
                    "memoryLimitExceeded": false,
                    "peakRssBytes": 0,
                    "rssMeasurement": "process_tree_ps_short_lived_no_sample"
                },
                "stdout": stdout,
                "stderr": stderr,
                "report": null,
                "csa": [],
                "quarantine": quarantine,
                "failureCategory": "timeout",
                "completedAt": "2026-08-09T00:00:00Z"
            }),
            "receiptSha256",
        );
        let command_receipt = write_fixture_json(
            root,
            &format!(
                "artifacts/generation-1/arena/evidence/{job_id}/attempt-001-quarantine.receipt.json"
            ),
            &attempt_receipt,
        );
        serde_json::json!({
            "jobId": job_id,
            "attempt": 1,
            "status": "quarantined",
            "returnCode": -15,
            "timedOut": true,
            "outputLimitExceeded": false,
            "memoryLimitExceeded": false,
            "peakRssBytes": 0,
            "rssMeasurement": "process_tree_ps_short_lived_no_sample",
            "stdout": stdout,
            "stderr": stderr,
            "report": null,
            "csa": [],
            "quarantine": quarantine,
            "failureCategory": "timeout",
            "completedAt": "2026-08-09T00:00:00Z",
            "commandReceipt": command_receipt
        })
    }

    #[expect(
        clippy::too_many_lines,
        reason = "one complete registry fixture deliberately materializes every evidence link"
    )]
    fn arena_results_fixture(root: &Path, registry: &serde_json::Value) -> ArenaResults {
        let champion: RegistryArtifact =
            serde_json::from_value(registry["models"][0]["artifact"].clone()).unwrap();
        let challenger: RegistryArtifact =
            serde_json::from_value(registry["models"][1]["artifact"].clone()).unwrap();
        let model_bytes = std::fs::read(root.join(&champion.path)).unwrap();
        let payload_sha256 = model_bytes[model_bytes.len() - 32..].iter().fold(
            String::with_capacity(64),
            |mut encoded, byte| {
                use std::fmt::Write as _;
                write!(encoded, "{byte:02x}").unwrap();
                encoded
            },
        );
        let (engine, engine_build_receipt) = write_fixture_engine_build(root, "abcdef0");
        let mut starts = fixture_start_positions();
        for split in ["train", "validation"] {
            let mut ranked = starts
                .iter()
                .filter(|position| position.split == split)
                .map(|position| {
                    (
                        sha256_bytes(
                            format!(
                                "phase6-start\0{}\0{split}\0{}\0{}\0{}",
                                super::PHASE6_ARENA_SEED,
                                position.sfen,
                                position.source_game_sha256,
                                position.position_index
                            )
                            .as_bytes(),
                        ),
                        position.clone(),
                    )
                })
                .collect::<Vec<_>>();
            ranked.sort_by(|left, right| left.0.cmp(&right.0));
            let mut selected = ranked
                .into_iter()
                .map(|(_, position)| position)
                .collect::<Vec<_>>();
            starts.retain(|position| position.split != split);
            starts.append(&mut selected);
        }
        starts.sort_by(|left, right| left.position_id.cmp(&right.position_id));
        let (starts_ref, validation_ref, dataset_ref) =
            write_start_evidence(root, &engine, &engine_build_receipt, &starts);
        let config_bytes = phase6_fixture_config();
        let config_ref = write_fixture_artifact(root, "configs/phase6.toml", config_bytes);
        let config = parse_phase6_selfplay_config(config_bytes).unwrap();
        let config_sha256 =
            python_canonical_sha256(&serde_json::to_value(config).unwrap()).unwrap();
        let snapshot = active_registry_snapshot(registry);
        let snapshot_ref =
            write_fixture_json(root, "artifacts/model-registry-snapshot.json", &snapshot);

        let mut selected = starts
            .iter()
            .filter(|position| position.split == "validation")
            .collect::<Vec<_>>();
        let selection_seed = 0x0135_27c8_u64 ^ 0x0041_5245_4E41;
        selected.sort_by_key(|position| {
            (
                sha256_bytes(
                    format!("{selection_seed}\0{}", position.position_id).as_bytes(),
                ),
                position.position_id.clone(),
            )
        });
        let mut jobs = Vec::new();
        for index in 0..20 {
            let job_id = format!("pair-{index:04}");
            let (start_group, start_id, sfen) = if index < 10 {
                (
                    "initial",
                    "standard-initial".to_owned(),
                    super::PHASE6_INITIAL_SFEN.to_owned(),
                )
            } else {
                let position = selected[index - 10];
                (
                    "start_set",
                    position.position_id.clone(),
                    position.sfen.clone(),
                )
            };
            let seed = derive_paired_job_seed(20_260_808, index, &start_id);
            let output = format!("artifacts/generation-1/arena/jobs/{job_id}");
            jobs.push(serde_json::json!({
                "jobId": job_id,
                "pairIndex": index,
                "startGroup": start_group,
                "startPositionId": start_id,
                "sfen": sfen,
                "seed": seed,
                "gameIds": [format!("game-{:06}", index * 2), format!("game-{:06}", index * 2 + 1)],
                "modelAColorOrder": ["black", "white"],
                "outputDir": output,
                "reportPath": format!("{output}/arena-report.json"),
                "csaPaths": [format!("{output}/games/game-000001.csa"), format!("{output}/games/game-000002.csa")],
                "quarantinePath": format!("artifacts/generation-1/arena/quarantine/{job_id}.json"),
                "command": {
                    "kind": "engine_arena",
                    "argv": [
                        engine.path, "arena", "--games", "2", "--player-a", "neural",
                        "--player-b", "neural", "--a-depth", "6", "--b-depth", "6",
                        "--a-hash-mb", "64", "--b-hash-mb", "64", "--nodes", "500",
                        "--max-plies", "256", "--seed", seed.to_string(), "--sfen", sfen,
                        "--git-commit", "abcdef0", "--output-dir", output,
                        "--a-model", challenger.path, "--b-model", champion.path
                    ],
                    "timeoutSeconds": 1800
                }
            }));
        }
        let plan = self_hashed_value(
            serde_json::json!({
                "schema": "phase6_paired_arena_plan/v1",
                "generationId": "generation-1",
                "champion": {"modelId": "champion-v0", "artifact": champion, "evaluatorKind": "neural"},
                "challenger": {"modelId": "challenger-v1", "artifact": challenger, "evaluatorKind": "neural"},
                "engine": engine,
                "engineBuildReceipt": engine_build_receipt,
                "modelRegistry": snapshot_ref,
                "gitCommit": "abcdef0",
                "config": config_ref,
                "configSha256": config_sha256,
                "startPositions": starts_ref,
                "startPositionValidation": validation_ref,
                "datasetManifest": dataset_ref,
                "gameCount": 40,
                "pairCount": 20,
                "normalStartPairs": 10,
                "startSetPairs": 10,
                "seed": 20_260_808,
                "nodesPerMove": 500,
                "maxWorkers": 2,
                "memoryLimitMiB": 8192,
                "memoryPerWorkerMiB": 3072,
                "jobs": jobs
            }),
            "planSha256",
        );
        let plan_ref = write_fixture_json(root, "artifacts/plan.json", &plan);
        let stdout = write_fixture_artifact(root, "artifacts/arena.stdout.log", b"");
        let stderr = write_fixture_artifact(root, "artifacts/arena.stderr.log", b"");
        let mut attempts = Vec::new();
        for job in plan["jobs"].as_array().unwrap() {
            let (report, csa) =
                write_pair_report(root, job, &challenger, &champion, &payload_sha256);
            let command_receipt = write_completed_attempt_command_receipt(
                root,
                job,
                plan["planSha256"].as_str().unwrap(),
                &engine,
                &engine_build_receipt,
                &stdout,
                &stderr,
                &report,
                &csa,
            );
            attempts.push(serde_json::json!({
                "jobId": job["jobId"],
                "attempt": 1,
                "status": "completed",
                "returnCode": 0,
                "timedOut": false,
                "outputLimitExceeded": false,
                "memoryLimitExceeded": false,
                "peakRssBytes": 0,
                "rssMeasurement": "process_tree_ps_short_lived_no_sample",
                "stdout": stdout,
                "stderr": stderr,
                "report": report,
                "csa": csa,
                "quarantine": null,
                "failureCategory": null,
                "commandReceipt": command_receipt,
                "completedAt": "2026-08-09T00:00:01Z"
            }));
        }
        let execution = self_hashed_value(
            serde_json::json!({
                "schema": "phase6_arena_execution_manifest/v1",
                "generationId": "generation-1",
                "plan": plan_ref,
                "planSha256": plan["planSha256"],
                "status": "completed",
                "gameCountPlanned": 40,
                "jobsPlanned": 20,
                "jobsCompleted": 20,
                "jobsQuarantined": 0,
                "gamesCompleted": 40,
                "gamesQuarantined": 0,
                "quarantinedAttempts": 0,
                "attempts": attempts
            }),
            "manifestSha256",
        );
        let execution_ref =
            write_fixture_json(root, "artifacts/execution.json", &execution);
        let games = plan["jobs"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|job| {
                let initial = open_shogi_core::parse_sfen(job["sfen"].as_str().unwrap()).unwrap();
                [
                    ("challenger-v1", "champion-v0", "white_win"),
                    ("champion-v0", "challenger-v1", "black_win"),
                ]
                .into_iter()
                .enumerate()
                .map(move |(index, (black, white, result))| {
                    let winner = if index == 0 { Side::White } else { Side::Black };
                    let plies = u64::from(initial.side_to_move() == winner);
                    let (black_searches, white_searches) =
                        super::selection_counts(initial.side_to_move(), plies + 1);
                    let (challenger_searches, champion_searches) = if index == 0 {
                        (black_searches, white_searches)
                    } else {
                        (white_searches, black_searches)
                    };
                    serde_json::json!({
                        "gameId": job["gameIds"][index],
                        "pairId": job["jobId"],
                        "startGroup": job["startGroup"],
                        "startPositionId": job["startPositionId"],
                        "blackModelId": black,
                        "whiteModelId": white,
                        "result": result,
                        "plies": plies,
                        "illegalMoves": 0,
                        "crashes": 0,
                        "metrics": {
                            "championInferenceCalls": 0,
                            "championInferenceTimeNs": 0,
                            "challengerInferenceCalls": 0,
                            "challengerInferenceTimeNs": 0,
                            "championSearchNodes": champion_searches,
                            "championSearchElapsedMs": 0,
                            "challengerSearchNodes": challenger_searches,
                            "challengerSearchElapsedMs": 0,
                            "championSearchDepthSum": champion_searches,
                            "championSearches": champion_searches,
                            "challengerSearchDepthSum": challenger_searches,
                            "challengerSearches": challenger_searches
                        }
                    })
                })
            })
            .collect::<Vec<_>>();
        serde_json::from_value(serde_json::json!({
            "schema": "phase6_paired_arena_results/v1",
            "generationId": "generation-1",
            "plan": plan_ref,
            "execution": execution_ref,
            "championModelId": "champion-v0",
            "challengerModelId": "challenger-v1",
            "games": games
        }))
        .unwrap()
    }

    fn write_promotion_evidence_fixture(root: &Path, registry: &mut serde_json::Value) {
        let results = arena_results_fixture(root, registry);
        let mut results_bytes = serde_json::to_vec(&results).unwrap();
        results_bytes.push(b'\n');
        std::fs::write(root.join("artifacts/arena.json"), &results_bytes).unwrap();
        let results_ref = RegistryArtifact {
            path: PathBuf::from("artifacts/arena.json"),
            sha256: sha256_bytes(&results_bytes),
            size: results_bytes.len() as u64,
        };
        registry["generations"][1]["arenaManifest"] =
            serde_json::to_value(&results_ref).unwrap();

        let analysis_path = "artifacts/arena-analysis.json";
        let analysis = analyze_arena_results(&results, results_ref).unwrap();
        let mut analysis_bytes = serde_json::to_vec(&analysis).unwrap();
        analysis_bytes.push(b'\n');
        std::fs::write(root.join(analysis_path), &analysis_bytes).unwrap();
        let analysis_ref = RegistryArtifact {
            path: PathBuf::from(analysis_path),
            sha256: sha256_bytes(&analysis_bytes),
            size: analysis_bytes.len() as u64,
        };

        let policy_bytes = include_bytes!("../../../configs/generation/phase6_promotion.toml");
        std::fs::write(root.join("artifacts/policy.toml"), policy_bytes).unwrap();
        let policy_ref = RegistryArtifact {
            path: PathBuf::from("artifacts/policy.toml"),
            sha256: sha256_bytes(policy_bytes),
            size: policy_bytes.len() as u64,
        };
        let policy = parse_generation_policy(policy_bytes).unwrap();
        let policy_sha256 = python_canonical_sha256(&serde_json::to_value(&policy).unwrap()).unwrap();
        let observed = PromotionDecision {
            schema: String::new(),
            generation_id: String::new(),
            champion_model_id: String::new(),
            challenger_model_id: String::new(),
            arena_analysis: analysis_ref,
            policy: policy_ref,
            policy_sha256,
            decision: String::new(),
            weak_evidence: false,
            reasons: Vec::new(),
            evidence: PromotionEvidence {
                games: 0,
                decisive_games: 0,
                score_rate: 0.0,
                score_wilson95: PromotionWilsonInterval {
                    lower: 0.0,
                    upper: 0.0,
                },
                initial_score_rate: 0.0,
                start_set_score_rate: 0.0,
                side_score_gap: None,
                illegal_moves: 0,
                crashes: 0,
            },
            decided_at: "2026-08-09T01:00:00Z".to_owned(),
            decision_sha256: String::new(),
        };
        let decision = derive_promotion_decision(&analysis, &observed, &policy).unwrap();
        assert_eq!(decision.decision, "rejected");
        let promotion_path = "artifacts/promotion-rejected.json";
        let mut promotion = serde_json::to_vec(&decision).unwrap();
        promotion.push(b'\n');
        std::fs::write(root.join(promotion_path), &promotion).unwrap();
        registry["generations"][1]["promotionDecision"] = serde_json::json!({
            "path": promotion_path,
            "sha256": sha256_bytes(&promotion),
            "size": promotion.len(),
        });
    }

    fn self_hashed_json(mut value: serde_json::Value, field: &str) -> Vec<u8> {
        let canonical = serde_json::to_vec(&value).unwrap();
        value[field] = serde_json::json!(sha256_bytes(&canonical));
        let mut encoded = serde_json::to_vec(&value).unwrap();
        encoded.push(b'\n');
        encoded
    }

    fn test_model_bytes() -> Vec<u8> {
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
        let checksum = sha2::Sha256::digest(&bytes);
        bytes.extend_from_slice(&checksum);
        bytes
    }

    fn artifact_value(path: &str, hash_digit: &str) -> serde_json::Value {
        serde_json::json!({"path": path, "sha256": hash_digit.repeat(64), "size": 0})
    }

    fn temporary_record_path(label: &str) -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().canonicalize().unwrap().join(format!(
            "open-shogi-human-{label}-{}-{nonce}.csa",
            std::process::id()
        ))
    }

    fn test_config(path: &Path) -> PlayConfig {
        PlayConfig {
            human: Side::Black,
            budget: Budget::Nodes(1),
            safety_margin_ms: 50,
            depth: 1,
            initial: Position::startpos(),
            max_plies: 2,
            output: path.to_path_buf(),
            decision_log: path.with_extension("decisions.jsonl"),
            profile: PlayProfile::HandcraftedExperimental,
            model_path: None,
            registry_path: None,
            opening_book_path: None,
            opening_max_plies: 24,
            opening_profile: OpeningProfile::IbishaStrict,
            opening_minimum_samples: 2,
            opening_maximum_teacher_loss_cp: 80,
        }
    }

    fn paired_evidence(config: &PlayConfig) -> (String, Vec<u8>, Vec<u8>) {
        let profile = handcrafted_profile("handcrafted-experimental");
        let config_sha256 = play_config_sha256(config, &profile, None);
        let record = human_play_config_record(config, &profile, None, &config_sha256);
        let csa = encode_record(
            &config.initial,
            &[],
            config.human,
            open_shogi_core::CsaSpecialMove::Interrupted,
            open_shogi_core::CsaResultValidation::ExternalCondition,
            &record,
        )
        .unwrap()
        .into_bytes();
        let decisions = encode_decision_log(&record, &[]).unwrap();
        (config_sha256, csa, decisions)
    }

    fn prepare_publication_recovery_fixture(
        config: &PlayConfig,
        config_sha256: &str,
        csa: &[u8],
        decisions: &[u8],
    ) -> (PublicationPaths, PublicationStorage) {
        prepare_publication_parent(&config.output).unwrap();
        prepare_publication_parent(&config.decision_log).unwrap();
        let paths = publication_paths(config).unwrap();
        write_pending(&paths.csa_pending, csa).unwrap();
        write_pending(&paths.decision_pending, decisions).unwrap();
        let storage = PublicationStorage::open(config, &paths, false).unwrap();
        let csa_artifact = storage.csa_pending.read(super::MAX_HUMAN_CSA_BYTES).unwrap();
        let decision_artifact = storage
            .decision_pending
            .read(super::MAX_HUMAN_DECISION_BYTES)
            .unwrap();
        let marker = super::publication_marker(
            &storage,
            config_sha256,
            &csa_artifact,
            &decision_artifact,
        )
        .unwrap();
        publish_new_atomic(&paths.marker, &serde_json::to_vec(&marker).unwrap()).unwrap();
        (paths, storage)
    }

    fn substitute_publication_target(target: &Path, replacement: &Path) -> Result<(), String> {
        std::fs::remove_file(target).map_err(|error| error.to_string())?;
        std::fs::rename(replacement, target).map_err(|error| error.to_string())
    }
}
