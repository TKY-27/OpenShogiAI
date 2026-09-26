//! Asynchronous USI engine-session control.

use std::{
    io::{self, BufRead},
    sync::Arc,
    thread::{self, JoinHandle},
};

use open_shogi_core::{
    CancellationToken, EvaluationConfig, NeuralEvaluationMode, NeuralEvaluator, NeuralQuantization,
    OpeningBookV2, OpeningPolicy, OpeningProfile, Osaval02Evaluator, Osaval02Quantization,
    Position, RuntimeProfile, SearchConfig, SearchEngine, SearchResult, SearchTermination,
    TimeManager, parse_sfen, parse_usi_move, to_usi_move,
};

pub use crate::lifecycle::ProtocolSink;
use crate::{
    GoParameters, UsiCommand,
    clocks::time_control,
    engine_id_line,
    lifecycle::{
        BoundedLine, CompletionGate, OutputAuthority, WriterSink, format_search_info,
        read_bounded_line, sanitized,
    },
    parse_command,
    parser::MAX_GO_DEPTH,
};

const MAX_HASH_MEGABYTES: usize = 1_024;
const MAX_DEPTH: u8 = MAX_GO_DEPTH;
const DEFAULT_HASH_MEGABYTES: usize = 32;
const DEFAULT_SAFETY_MARGIN_MS: u64 = 50;
const DEFAULT_OPENING_MAX_PLIES: u32 = 40;
const DEFAULT_OPENING_MINIMUM_SAMPLES: u64 = 2;
const DEFAULT_OPENING_MAXIMUM_TEACHER_LOSS_CP: i32 = 80;

/// Static evaluator selected through the USI `ModelKind` option.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ModelKind {
    /// Use the built-in evaluator that passed the overall-champion evidence gate.
    #[default]
    Handcrafted,
    /// Require an `OSAVAL01` model containing `f32` weights.
    NeuralFloat,
    /// Require an `OSAVAL01` model containing symmetric per-layer `i8` weights.
    NeuralQuantized,
    /// Require an `OSAVAL02` Phase 10R model containing `f32` weights.
    Osaval02Float,
    /// Require an `OSAVAL02` Phase 10R model containing `i8` weights.
    Osaval02Quantized,
}

impl ModelKind {
    const fn usi_value(self) -> &'static str {
        match self {
            Self::Handcrafted => "overall-champion",
            Self::NeuralFloat => "neural-float",
            Self::NeuralQuantized => "neural-quantized",
            Self::Osaval02Float => "osaval02-float",
            Self::Osaval02Quantized => "osaval02-quantized",
        }
    }

    fn parse(value: Option<&str>) -> Result<Self, String> {
        match value {
            Some("overall-champion" | "handcrafted") => Ok(Self::Handcrafted),
            Some("neural-float") => Ok(Self::NeuralFloat),
            Some("neural-quantized") => Ok(Self::NeuralQuantized),
            Some("osaval02-float") => Ok(Self::Osaval02Float),
            Some("osaval02-quantized") => Ok(Self::Osaval02Quantized),
            _ => Err(
                "ModelKind must be `overall-champion`, `neural-float`, `neural-quantized`, `osaval02-float`, or `osaval02-quantized`".to_owned(),
            ),
        }
    }

    const fn required_quantization(self) -> Option<NeuralQuantization> {
        match self {
            Self::NeuralFloat => Some(NeuralQuantization::Float32),
            Self::NeuralQuantized => Some(NeuralQuantization::Int8),
            Self::Handcrafted | Self::Osaval02Float | Self::Osaval02Quantized => None,
        }
    }

    const fn required_osaval02_quantization(self) -> Option<Osaval02Quantization> {
        match self {
            Self::Osaval02Float => Some(Osaval02Quantization::Float32),
            Self::Osaval02Quantized => Some(Osaval02Quantization::Int8),
            _ => None,
        }
    }

    const fn is_osaval02(self) -> bool {
        self.required_osaval02_quantization().is_some()
    }
}

enum LoadedModel {
    Neural(NeuralEvaluator),
    Osaval02(Box<Osaval02Evaluator>),
}

impl LoadedModel {
    fn artifact_sha256(&self) -> String {
        match self {
            Self::Neural(model) => model.identity().sha256_hex(),
            Self::Osaval02(model) => model.identity().artifact_sha256.clone(),
        }
    }
}

/// Mutable engine options exposed through USI.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UsiOptions {
    pub hash_megabytes: usize,
    pub max_depth: u8,
    pub safety_margin_ms: u64,
    pub evaluation: EvaluationConfig,
    pub model_kind: ModelKind,
    pub model_semantics: NeuralEvaluationMode,
    pub model_path: String,
    pub expected_model_sha256: String,
    pub runtime_profile: RuntimeProfile,
    pub opening_book_path: String,
    pub opening_profile: OpeningProfile,
    pub opening_max_plies: u32,
    pub opening_minimum_samples: u64,
    pub opening_maximum_teacher_loss_cp: i32,
}

impl Default for UsiOptions {
    fn default() -> Self {
        Self {
            hash_megabytes: DEFAULT_HASH_MEGABYTES,
            max_depth: 8,
            safety_margin_ms: DEFAULT_SAFETY_MARGIN_MS,
            evaluation: EvaluationConfig::default(),
            model_kind: ModelKind::Handcrafted,
            model_semantics: NeuralEvaluationMode::PureValue,
            model_path: String::new(),
            expected_model_sha256: String::new(),
            runtime_profile: RuntimeProfile::Standard,
            opening_book_path: String::new(),
            opening_profile: OpeningProfile::Unrestricted,
            opening_max_plies: DEFAULT_OPENING_MAX_PLIES,
            opening_minimum_samples: DEFAULT_OPENING_MINIMUM_SAMPLES,
            opening_maximum_teacher_loss_cp: DEFAULT_OPENING_MAXIMUM_TEACHER_LOSS_CP,
        }
    }
}

struct ActiveSearch {
    cancellation: CancellationToken,
    completion_gate: Option<Arc<CompletionGate>>,
    handle: JoinHandle<()>,
}

/// One transactional USI session with at most one active worker.
pub struct UsiSession {
    position: Position,
    options: UsiOptions,
    output_authority: Arc<OutputAuthority>,
    active: Option<ActiveSearch>,
    neural_evaluator: Option<Arc<NeuralEvaluator>>,
    osaval02_evaluator: Option<Arc<Osaval02Evaluator>>,
    pending_model: Option<(ModelKind, String)>,
    opening_book: Option<Arc<OpeningBookV2>>,
    sink: Arc<dyn ProtocolSink>,
    /// A held `go ponder` request. Invariant: set only while no search is active.
    pending_ponder: Option<GoParameters>,
}

impl UsiSession {
    #[must_use]
    pub fn new(sink: Arc<dyn ProtocolSink>) -> Self {
        Self {
            position: Position::startpos(),
            options: UsiOptions::default(),
            output_authority: Arc::new(OutputAuthority::default()),
            active: None,
            neural_evaluator: None,
            osaval02_evaluator: None,
            pending_model: None,
            opening_book: None,
            sink,
            pending_ponder: None,
        }
    }

    #[must_use]
    pub const fn position(&self) -> &Position {
        &self.position
    }

    #[must_use]
    pub const fn options(&self) -> &UsiOptions {
        &self.options
    }

    /// Processes one command. Returns `false` after `quit`.
    pub fn process_line(&mut self, line: &str) -> bool {
        let command = match parse_command(line.trim_end_matches(['\r', '\n'])) {
            Ok(command) => command,
            Err(error) => {
                self.send_error(&error.to_string());
                return true;
            }
        };
        match command {
            UsiCommand::Usi => self.identify(),
            UsiCommand::IsReady => match self.ensure_model_ready() {
                Ok(()) => self.sink.send("readyok"),
                Err(error) => {
                    self.send_error(&error);
                    return false;
                }
            },
            UsiCommand::SetOption { name, value } => {
                if let Err(error) = self.set_option(&name, value.as_deref()) {
                    self.send_error(&error);
                }
            }
            UsiCommand::NewGame => {
                self.cancel_active(true);
                self.pending_ponder = None;
                self.position = Position::startpos();
            }
            UsiCommand::PositionStartpos { moves } => {
                if let Err(error) = self.set_position(Position::startpos(), &moves) {
                    self.send_error(&error);
                }
            }
            UsiCommand::PositionSfen { sfen, moves } => {
                let result = parse_sfen(&sfen)
                    .map_err(|error| format!("invalid SFEN: {error}"))
                    .and_then(|position| self.set_position(position, &moves));
                if let Err(error) = result {
                    self.send_error(&error);
                }
            }
            UsiCommand::Go(parameters) => {
                if parameters.ponder {
                    // No ponder search exists: hold the request instead of answering
                    // the predicted position early or reinterpreting it as a search.
                    self.cancel_active(true);
                    self.pending_ponder = Some(parameters);
                    self.send_error(
                        "go ponder held without pondering; ponderhit starts the search and stop discards it (USI_Ponder must stay false)",
                    );
                } else if let Err(error) = self.start_search(&parameters) {
                    self.send_error(&error);
                    return false;
                }
            }
            UsiCommand::GoMate { .. } => self.sink.send("checkmate notimplemented"),
            UsiCommand::PonderHit(clocks) => match self.pending_ponder.take() {
                Some(mut pending) => {
                    pending.apply_clock_override(&clocks);
                    if let Err(error) = self.start_search(&pending) {
                        self.send_error(&error);
                        return false;
                    }
                }
                None => self.send_error("ponderhit without a held go ponder"),
            },
            UsiCommand::Stop => {
                if self.pending_ponder.take().is_some() {
                    // USI requires an answer to stop; the GUI discards it in this state.
                    self.send_error("discarded go ponder without ponderhit");
                    self.sink.send("bestmove resign");
                } else {
                    self.cancel_active(false);
                }
            }
            UsiCommand::Quit => {
                self.cancel_active(true);
                self.pending_ponder = None;
                return false;
            }
            UsiCommand::GameOver { .. } => {
                self.cancel_active(true);
                self.pending_ponder = None;
            }
        }
        true
    }

    fn identify(&self) {
        self.sink.send(&engine_id_line());
        self.sink.send("id author OpenShogiAI contributors");
        self.sink.send(&format!(
            "option name USI_Hash type spin default {DEFAULT_HASH_MEGABYTES} min 1 max {MAX_HASH_MEGABYTES}"
        ));
        // Advertise the real default so a GUI never invents USI_Ponder=true for us.
        self.sink
            .send("option name USI_Ponder type check default false");
        self.sink
            .send("option name Threads type spin default 1 min 1 max 1");
        self.sink
            .send("option name MaxDepth type spin default 8 min 1 max 64");
        self.sink.send(
            "option name ModelKind type combo default overall-champion var overall-champion var neural-float var neural-quantized var osaval02-float var osaval02-quantized",
        );
        self.sink.send(
            "option name ModelSemantics type combo default pure-value var pure-value var residual var composite-50-50",
        );
        self.sink.send(
            "option name RuntimeProfile type combo default standard var standard var pure_learned",
        );
        self.sink.send(&format!(
            "option name TimeSafetyMarginMs type spin default {DEFAULT_SAFETY_MARGIN_MS} min 0 max {}",
            open_shogi_core::MAX_SAFETY_MARGIN_MS
        ));
        self.sink
            .send("option name ModelPath type filename default <empty>");
        self.sink
            .send("option name ExpectedModelSha256 type string default <empty>");
        for name in evaluation_option_names() {
            self.sink
                .send(&format!("option name Eval{name} type check default true"));
        }
        self.sink.send("usiok");
    }

    #[expect(
        clippy::too_many_lines,
        reason = "the closed USI option table keeps validation and transactional updates together"
    )]
    fn set_option(&mut self, name: &str, value: Option<&str>) -> Result<(), String> {
        self.cancel_active(true);
        match name {
            "USI_Hash" | "Hash" => {
                let value = parse_usize_option(value, "USI_Hash")?;
                if !(1..=MAX_HASH_MEGABYTES).contains(&value) {
                    return Err(format!("USI_Hash must be 1..={MAX_HASH_MEGABYTES}"));
                }
                self.options.hash_megabytes = value;
            }
            "USI_Ponder" => {
                let enabled = parse_bool_option(value, "USI_Ponder")?;
                if enabled {
                    return Err("pondering is not implemented; keep USI_Ponder false".to_owned());
                }
            }
            "Threads" => {
                let threads = parse_usize_option(value, "Threads")?;
                if threads != 1 {
                    return Err("only one search worker is supported".to_owned());
                }
            }
            "MaxDepth" => {
                let value = parse_u8_option(value, "MaxDepth")?;
                if !(1..=MAX_DEPTH).contains(&value) {
                    return Err(format!("MaxDepth must be 1..={MAX_DEPTH}"));
                }
                self.options.max_depth = value;
            }
            "TimeSafetyMarginMs" => {
                let value = parse_u64_option(value, "TimeSafetyMarginMs")?;
                if value > open_shogi_core::MAX_SAFETY_MARGIN_MS {
                    return Err(format!(
                        "TimeSafetyMarginMs must be 0..={}",
                        open_shogi_core::MAX_SAFETY_MARGIN_MS
                    ));
                }
                self.options.safety_margin_ms = value;
            }
            "ModelKind" => {
                let model_kind = ModelKind::parse(value)?;
                let model_path = self
                    .pending_model
                    .as_ref()
                    .map_or_else(|| self.options.model_path.clone(), |(_, path)| path.clone());
                self.apply_model_candidate(model_kind, model_path)?;
            }
            "ModelSemantics" => {
                let semantics = match value {
                    Some("pure-value") => NeuralEvaluationMode::PureValue,
                    Some("residual") => NeuralEvaluationMode::Residual,
                    Some("composite-50-50") => NeuralEvaluationMode::Composite,
                    _ => {
                        return Err(
                            "ModelSemantics must be pure-value, residual, or composite-50-50"
                                .to_owned(),
                        );
                    }
                };
                if (self.options.model_kind.is_osaval02()
                    || self.options.runtime_profile == RuntimeProfile::PureLearned)
                    && semantics != NeuralEvaluationMode::PureValue
                {
                    return Err(
                        "OSAVAL02 only supports pure-value semantics; score blending is forbidden"
                            .to_owned(),
                    );
                }
                self.options.model_semantics = semantics;
            }
            "RuntimeProfile" => {
                let profile = RuntimeProfile::parse(
                    value.ok_or_else(|| "RuntimeProfile requires a value".to_owned())?,
                )?;
                self.options.runtime_profile = profile;
                if profile == RuntimeProfile::PureLearned {
                    self.options.model_semantics = NeuralEvaluationMode::PureValue;
                    self.options.evaluation = EvaluationConfig::disabled();
                    self.options.opening_profile = OpeningProfile::Unrestricted;
                    self.options.opening_book_path.clear();
                    self.opening_book = None;
                }
            }
            "ModelPath" => {
                let path = value.ok_or_else(|| "ModelPath requires a value".to_owned())?;
                if path.is_empty() {
                    return Err("ModelPath must not be empty".to_owned());
                }
                let model_kind = self
                    .pending_model
                    .as_ref()
                    .map_or(self.options.model_kind, |(kind, _)| *kind);
                self.apply_model_candidate(model_kind, path.to_owned())?;
            }
            "ExpectedModelSha256" => {
                let hash =
                    value.ok_or_else(|| "ExpectedModelSha256 requires a value".to_owned())?;
                if hash.len() != 64
                    || !hash
                        .bytes()
                        .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
                {
                    return Err(
                        "ExpectedModelSha256 must be 64 lowercase hexadecimal characters"
                            .to_owned(),
                    );
                }
                hash.clone_into(&mut self.options.expected_model_sha256);
            }
            "OpeningBookPath" => {
                return Err("opening books are disabled for play; offline training only".to_owned());
            }
            "OpeningProfile" => {
                let profile = OpeningProfile::parse(
                    value.ok_or_else(|| "OpeningProfile requires a value".to_owned())?,
                )?;
                if self.options.runtime_profile == RuntimeProfile::PureLearned
                    && profile != OpeningProfile::Unrestricted
                {
                    return Err("pure_learned requires unrestricted opening style".to_owned());
                }
                self.options.opening_profile = profile;
            }
            "OpeningMaxPlies" => {
                let value = parse_u32_option(value, "OpeningMaxPlies")?;
                if !(1..=40).contains(&value) {
                    return Err("OpeningMaxPlies must be 1..=40".to_owned());
                }
                self.options.opening_max_plies = value;
            }
            "OpeningMinSamples" => {
                let value = parse_u64_option(value, "OpeningMinSamples")?;
                if !(1..=1_000_000).contains(&value) {
                    return Err("OpeningMinSamples must be 1..=1000000".to_owned());
                }
                self.options.opening_minimum_samples = value;
            }
            "OpeningMaxTeacherLossCp" => {
                let value = parse_i32_option(value, "OpeningMaxTeacherLossCp")?;
                if !(0..=10_000).contains(&value) {
                    return Err("OpeningMaxTeacherLossCp must be 0..=10000".to_owned());
                }
                self.options.opening_maximum_teacher_loss_cp = value;
            }
            option if option.starts_with("Eval") => {
                let enabled = parse_bool_option(value, option)?;
                if self.options.runtime_profile == RuntimeProfile::PureLearned && enabled {
                    return Err("pure_learned prohibits handcrafted evaluation terms".to_owned());
                }
                set_evaluation_option(&mut self.options.evaluation, &option[4..], enabled)?;
            }
            _ => return Err(format!("unknown option `{name}`")),
        }
        Ok(())
    }

    fn set_position(&mut self, mut position: Position, moves: &[String]) -> Result<(), String> {
        self.cancel_active(true);
        self.pending_ponder = None;
        for (index, notation) in moves.iter().enumerate() {
            let movement = parse_usi_move(notation)
                .map_err(|error| format!("move {} is malformed: {error}", index + 1))?;
            position
                .make_move(movement)
                .map_err(|error| format!("move {} is illegal: {error}", index + 1))?;
        }
        self.position = position;
        Ok(())
    }

    fn start_search(&mut self, parameters: &GoParameters) -> Result<(), String> {
        self.cancel_active(true);
        self.pending_ponder = None;
        self.ensure_model_ready()?;
        if self.position.move_number() <= self.options.opening_max_plies
            && let Some(choice) = self.opening_book.as_ref().and_then(|book| {
                book.select(
                    &self.position,
                    OpeningPolicy {
                        profile: self.options.opening_profile,
                        minimum_sample_count: self.options.opening_minimum_samples,
                        maximum_teacher_loss_cp: self.options.opening_maximum_teacher_loss_cp,
                    },
                )
            })
        {
            self.output_authority.advance();
            self.sink.send(&format!(
                "info string source book profile {} samples {} teacher_cp {} teacher_depth {} teacher_nodes {} classification {}",
                self.options.opening_profile.name(),
                choice.sample_count,
                choice.teacher_score_cp,
                choice.teacher_depth,
                choice.teacher_nodes,
                choice.opening_classification,
            ));
            self.sink
                .send(&format!("bestmove {}", to_usi_move(choice.movement)));
            return Ok(());
        }
        let generation = self.output_authority.advance();
        let cancellation = CancellationToken::new();
        let worker_cancellation = cancellation.clone();
        let output_authority = Arc::clone(&self.output_authority);
        let sink = Arc::clone(&self.sink);
        let position = self.position.clone();
        let config = search_config(&self.options);
        let neural_evaluator = self.neural_evaluator.clone();
        let osaval02_evaluator = self.osaval02_evaluator.clone();
        let model_semantics = self.options.model_semantics;
        let runtime_profile = self.options.runtime_profile;
        let expected_model_sha256 = self.options.expected_model_sha256.clone();
        let plan = TimeManager::default().plan_for_position(
            &position,
            time_control(parameters, self.options.safety_margin_ms),
            if parameters.infinite {
                MAX_DEPTH
            } else {
                self.options.max_depth
            },
        )?;
        let completion_gate = parameters
            .infinite
            .then(|| Arc::new(CompletionGate::default()));
        let worker_gate = completion_gate.clone();
        let handle = thread::spawn(move || {
            let mut engine = build_search_engine(
                config,
                runtime_profile,
                &expected_model_sha256,
                osaval02_evaluator,
                neural_evaluator,
                model_semantics,
            );
            let callback_sink = Arc::clone(&sink);
            let callback_authority = Arc::clone(&output_authority);
            let result = engine.search_managed_with_callback(
                &position,
                plan,
                &worker_cancellation,
                move |info| {
                    callback_authority.send_if_current(generation, callback_sink.as_ref(), || {
                        format_search_info(info)
                    });
                },
            );
            if runtime_profile == RuntimeProfile::PureLearned {
                let proof = engine.runtime_proof(result.stats, expected_model_sha256);
                output_authority
                    .send_if_current(generation, sink.as_ref(), || format_runtime_proof(&proof));
            }
            complete_search(
                &result,
                worker_gate,
                output_authority.as_ref(),
                sink.as_ref(),
                generation,
            );
        });
        self.active = Some(ActiveSearch {
            cancellation,
            completion_gate,
            handle,
        });
        Ok(())
    }

    fn ensure_model_ready(&mut self) -> Result<(), String> {
        if self.options.runtime_profile == RuntimeProfile::PureLearned {
            if !self.options.model_kind.is_osaval02() {
                return Err("pure_learned requires an OSAVAL02 ModelKind".to_owned());
            }
            if self.options.expected_model_sha256.is_empty() {
                return Err("pure_learned requires ExpectedModelSha256".to_owned());
            }
            if !self.options.opening_book_path.is_empty() || self.opening_book.is_some() {
                return Err("pure_learned prohibits opening books".to_owned());
            }
        }
        if self.options.model_kind == ModelKind::Handcrafted {
            self.neural_evaluator = None;
            self.osaval02_evaluator = None;
            return Ok(());
        }
        if self.neural_evaluator.is_some() || self.osaval02_evaluator.is_some() {
            return self.validate_runtime_profile_model();
        }
        self.load_configured_model()?;
        self.validate_runtime_profile_model()
    }

    fn validate_runtime_profile_model(&self) -> Result<(), String> {
        if self.options.runtime_profile != RuntimeProfile::PureLearned {
            return Ok(());
        }
        let actual = self
            .osaval02_evaluator
            .as_ref()
            .ok_or_else(|| "pure_learned requires a loaded OSAVAL02 model".to_owned())?
            .identity()
            .artifact_sha256
            .as_str();
        if actual != self.options.expected_model_sha256 {
            return Err("pure_learned model SHA-256 mismatch".to_owned());
        }
        Ok(())
    }

    fn load_configured_model(&mut self) -> Result<(), String> {
        let result = Self::try_load_model(&self.options);
        match result {
            Ok(evaluator) => {
                self.sink.send(&format!(
                    "info string model loaded {} {}",
                    self.options.model_kind.usi_value(),
                    evaluator.artifact_sha256()
                ));
                self.set_loaded_model(evaluator);
                Ok(())
            }
            Err(error) => {
                self.neural_evaluator = None;
                self.osaval02_evaluator = None;
                Err(error)
            }
        }
    }

    fn apply_model_candidate(
        &mut self,
        model_kind: ModelKind,
        model_path: String,
    ) -> Result<(), String> {
        if model_kind == ModelKind::Handcrafted {
            self.options.model_kind = model_kind;
            self.options.model_path = model_path;
            self.neural_evaluator = None;
            self.osaval02_evaluator = None;
            self.pending_model = None;
            return Ok(());
        }
        if model_path.is_empty() {
            self.options.model_kind = model_kind;
            self.options.model_path = model_path;
            self.neural_evaluator = None;
            self.osaval02_evaluator = None;
            self.pending_model = None;
            return Ok(());
        }
        let mut candidate = self.options.clone();
        candidate.model_kind = model_kind;
        candidate.model_path.clone_from(&model_path);
        match Self::try_load_model(&candidate) {
            Ok(evaluator) => {
                self.sink.send(&format!(
                    "info string model loaded {} {}",
                    candidate.model_kind.usi_value(),
                    evaluator.artifact_sha256()
                ));
                self.options.model_kind = model_kind;
                self.options.model_path = model_path;
                self.set_loaded_model(evaluator);
                self.pending_model = None;
                Ok(())
            }
            Err(error) if self.neural_evaluator.is_some() || self.osaval02_evaluator.is_some() => {
                self.pending_model = Some((model_kind, model_path));
                Err(error)
            }
            Err(error) => {
                self.options.model_kind = model_kind;
                self.options.model_path = model_path;
                self.neural_evaluator = None;
                self.osaval02_evaluator = None;
                self.pending_model = None;
                Err(error)
            }
        }
    }

    fn set_loaded_model(&mut self, evaluator: LoadedModel) {
        match evaluator {
            LoadedModel::Neural(model) => {
                self.neural_evaluator = Some(Arc::new(model));
                self.osaval02_evaluator = None;
            }
            LoadedModel::Osaval02(model) => {
                self.neural_evaluator = None;
                self.osaval02_evaluator = Some(Arc::from(model));
            }
        }
    }

    fn try_load_model(options: &UsiOptions) -> Result<LoadedModel, String> {
        if options.model_path.is_empty() {
            return Err("ModelPath is required for neural evaluation".to_owned());
        }
        if let Some(required) = options.model_kind.required_quantization() {
            let evaluator = NeuralEvaluator::load_file(&options.model_path)
                .map_err(|error| format!("failed to load neural model: {error}"))?;
            if evaluator.quantization() != required {
                return Err(format!(
                    "ModelKind {} requires {:?} weights, but the model contains {:?}",
                    options.model_kind.usi_value(),
                    required,
                    evaluator.quantization()
                ));
            }
            return Ok(LoadedModel::Neural(evaluator));
        }
        if let Some(required) = options.model_kind.required_osaval02_quantization() {
            let evaluator = Osaval02Evaluator::load_file(&options.model_path)
                .map_err(|error| format!("failed to load OSAVAL02 model: {error}"))?;
            if evaluator.quantization() != required {
                return Err(format!(
                    "ModelKind {} requires {:?} weights, but the model contains {:?}",
                    options.model_kind.usi_value(),
                    required,
                    evaluator.quantization()
                ));
            }
            return Ok(LoadedModel::Osaval02(Box::new(evaluator)));
        }
        Err("handcrafted evaluation does not load a model".to_owned())
    }

    fn cancel_active(&mut self, suppress_output: bool) {
        if suppress_output {
            self.output_authority.advance();
        }
        let Some(active) = self.active.take() else {
            return;
        };
        active.cancellation.cancel();
        if let Some(gate) = &active.completion_gate {
            gate.release();
        }
        let _ = active.handle.join();
    }

    fn send_error(&self, message: &str) {
        self.sink
            .send(&format!("info string error {}", sanitized(message)));
    }
}

fn complete_search(
    result: &SearchResult,
    worker_gate: Option<Arc<CompletionGate>>,
    output_authority: &OutputAuthority,
    sink: &dyn ProtocolSink,
    generation: u64,
) {
    if let Some(gate) = worker_gate
        && !gate.wait()
    {
        return;
    }
    if result.termination == SearchTermination::EvaluationError {
        output_authority.send_if_current(generation, sink, || {
            format!(
                "info string error OSAVAL02 strict inference failed after {} calls",
                result.stats.neural_inference_calls
            )
        });
    }
    // Deliberate divergence from the pure adapter: the handcrafted profile answers
    // a strict-inference failure with the protocol-required `resign` instead of a
    // process-fatal error, because this session may carry a loaded neural model
    // whose failure is recoverable at the GUI level. The pure adapter fails closed.
    let bestmove = if result.termination == SearchTermination::EvaluationError {
        "resign".to_owned()
    } else {
        result
            .best_move
            .map_or_else(|| "resign".to_owned(), to_usi_move)
    };
    output_authority.send_if_current(generation, sink, || format!("bestmove {bestmove}"));
}

impl Drop for UsiSession {
    fn drop(&mut self) {
        self.cancel_active(true);
    }
}

/// Runs a USI session over process standard input and output.
///
/// # Errors
///
/// Returns an input error if standard input cannot be read.
pub fn run_stdio() -> io::Result<()> {
    let sink: Arc<dyn ProtocolSink> = Arc::new(WriterSink::new(io::stdout()));
    let stdin = io::stdin();
    run_protocol(&mut stdin.lock(), sink)
}

fn run_protocol<R: BufRead>(reader: &mut R, sink: Arc<dyn ProtocolSink>) -> io::Result<()> {
    let mut session = UsiSession::new(sink);
    loop {
        match read_bounded_line(reader, 4_096)? {
            BoundedLine::Line(line) => {
                if !session.process_line(&line) {
                    return Ok(());
                }
            }
            BoundedLine::TooLong => session.send_error("command exceeds 4096-byte limit"),
            BoundedLine::Eof => return Ok(()),
        }
    }
}

fn search_config(options: &UsiOptions) -> SearchConfig {
    SearchConfig {
        evaluation: options.evaluation,
        runtime_profile: options.runtime_profile,
        transposition_entries: hash_entries(options.hash_megabytes),
        ..SearchConfig::default()
    }
}

/// Builds the search worker's engine. The pure-learned profile forces the strict
/// OSAVAL02-only runtime; readiness already guaranteed a loaded, hash-matched evaluator.
fn build_search_engine(
    config: SearchConfig,
    runtime_profile: RuntimeProfile,
    expected_model_sha256: &str,
    osaval02_evaluator: Option<Arc<Osaval02Evaluator>>,
    neural_evaluator: Option<Arc<NeuralEvaluator>>,
    model_semantics: NeuralEvaluationMode,
) -> SearchEngine {
    if runtime_profile == RuntimeProfile::PureLearned {
        return SearchEngine::with_pure_learned(
            config,
            osaval02_evaluator.expect("profile readiness guarantees OSAVAL02"),
            expected_model_sha256,
        )
        .expect("profile readiness guarantees model identity");
    }
    osaval02_evaluator.map_or_else(
        || {
            neural_evaluator.map_or_else(
                || SearchEngine::new(config),
                |evaluator| SearchEngine::with_neural_mode(config, evaluator, model_semantics),
            )
        },
        |evaluator| SearchEngine::with_osaval02(config, evaluator),
    )
}

fn format_runtime_proof(proof: &open_shogi_core::RuntimeProofCounters) -> String {
    format!(
        "info string runtime_proof profile={} schema={} learned_eval_calls={} handcrafted_eval_calls={} residual_eval_calls={} composite_eval_calls={} book_hits={} teacher_calls={} fallback_count={} model_sha256={} evaluator_profile_schema_hash={} valid={}",
        proof.profile,
        proof.profile_schema,
        proof.learned_eval_calls,
        proof.handcrafted_eval_calls,
        proof.residual_eval_calls,
        proof.composite_eval_calls,
        proof.book_hits,
        proof.teacher_calls,
        proof.fallback_count,
        proof.model_sha256,
        proof.evaluator_profile_schema_hash,
        proof.valid_pure_learned(),
    )
}

fn hash_entries(megabytes: usize) -> usize {
    SearchEngine::transposition_entries_for_megabytes(megabytes).max(1)
}

fn evaluation_option_names() -> [&'static str; 8] {
    [
        "Material",
        "Hands",
        "Promotions",
        "KingSafety",
        "Mobility",
        "Check",
        "EnemyCamp",
        "Tempo",
    ]
}

fn set_evaluation_option(
    config: &mut EvaluationConfig,
    name: &str,
    enabled: bool,
) -> Result<(), String> {
    match name {
        "Material" => config.material = enabled,
        "Hands" => config.hands = enabled,
        "Promotions" => config.promotions = enabled,
        "KingSafety" => config.king_safety = enabled,
        "Mobility" => config.mobility = enabled,
        "Check" => config.check = enabled,
        "EnemyCamp" => config.enemy_camp = enabled,
        "Tempo" => config.tempo = enabled,
        _ => return Err(format!("unknown evaluation option `{name}`")),
    }
    Ok(())
}

fn parse_usize_option(value: Option<&str>, name: &str) -> Result<usize, String> {
    value
        .ok_or_else(|| format!("{name} requires a value"))?
        .parse()
        .map_err(|_| format!("{name} must be an integer"))
}

fn parse_u8_option(value: Option<&str>, name: &str) -> Result<u8, String> {
    value
        .ok_or_else(|| format!("{name} requires a value"))?
        .parse()
        .map_err(|_| format!("{name} must be an integer"))
}

fn parse_u32_option(value: Option<&str>, name: &str) -> Result<u32, String> {
    value
        .ok_or_else(|| format!("{name} requires a value"))?
        .parse()
        .map_err(|_| format!("{name} must be an integer"))
}

fn parse_u64_option(value: Option<&str>, name: &str) -> Result<u64, String> {
    value
        .ok_or_else(|| format!("{name} requires a value"))?
        .parse()
        .map_err(|_| format!("{name} must be an integer"))
}

fn parse_i32_option(value: Option<&str>, name: &str) -> Result<i32, String> {
    value
        .ok_or_else(|| format!("{name} requires a value"))?
        .parse()
        .map_err(|_| format!("{name} must be an integer"))
}

fn parse_bool_option(value: Option<&str>, name: &str) -> Result<bool, String> {
    match value {
        Some("true") => Ok(true),
        Some("false") => Ok(false),
        _ => Err(format!("{name} must be `true` or `false`")),
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        io::Cursor,
        path::PathBuf,
        sync::{
            Arc,
            atomic::{AtomicU64, Ordering},
        },
        thread,
        time::Duration,
    };

    use flate2::{Compression, write::GzEncoder};
    use open_shogi_core::{
        MATE_SCORE, NeuralQuantization, Position, SearchInfo, SearchStats, to_sfen,
    };
    use sha2::{Digest, Sha256};
    use std::io::Write as _;

    use super::{ModelKind, UsiSession, format_search_info, run_protocol};
    use crate::{GoParameters, clocks::time_control, lifecycle::test_support::MemorySink};

    static NEXT_MODEL_FILE: AtomicU64 = AtomicU64::new(0);

    struct TemporaryModel(PathBuf);

    impl TemporaryModel {
        fn write(bytes: &[u8]) -> Self {
            let sequence = NEXT_MODEL_FILE.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().canonicalize().unwrap().join(format!(
                "open-shogi-usi-model-{}-{sequence}.osaval",
                std::process::id()
            ));
            fs::write(&path, bytes).expect("write temporary model");
            Self(path)
        }

        fn display(&self) -> String {
            self.0.to_string_lossy().into_owned()
        }
    }

    impl Drop for TemporaryModel {
        fn drop(&mut self) {
            let _ = fs::remove_file(&self.0);
        }
    }

    fn append_u32(bytes: &mut Vec<u8>, value: u32) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn append_f32(bytes: &mut Vec<u8>, value: f32) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn model_bytes(quantization: NeuralQuantization) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"OSAVAL01");
        for value in [
            1,
            1,
            1,
            1 << 2,
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
        for (weight, quantized_weight) in [(2.0_f32, 4_u8), (3.0, 6_u8)] {
            append_u32(&mut bytes, 1);
            append_u32(&mut bytes, 1);
            match quantization {
                NeuralQuantization::Float32 => append_f32(&mut bytes, weight),
                NeuralQuantization::Int8 => {
                    append_f32(&mut bytes, 0.5);
                    bytes.push(quantized_weight);
                }
            }
            append_f32(&mut bytes, 0.0);
        }
        let checksum = Sha256::digest(bytes.as_slice());
        bytes.extend_from_slice(&checksum);
        bytes
    }

    fn opening_book_bytes() -> Vec<u8> {
        let provenance = ["1".repeat(64), "2".repeat(64)];
        let mut record = serde_json::json!({
            "schema": open_shogi_core::OPENING_BOOK_SCHEMA,
            "stateKey": "eb5bc2ef917ec96fe2172f96d7060ec4f39322caf929fc1177d2c9fc8b937ebc",
            "stateSfen": "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b -",
            "ruleProfile": "standard-shogi/v1",
            "buildVersion": "usi-fixture",
            "provenanceReferences": provenance,
            "candidates": [{
                "moveUsi": "2g2f", "sampleCount": 2,
                "sourceDistribution": {"aobazero-no-noise": 2},
                "blackResults": {"wins": 1, "losses": 1, "draws": 0, "unknown": 0},
                "whiteResults": {"wins": 0, "losses": 0, "draws": 0, "unknown": 0},
                "teacherScoreCp": 20, "scoreUncertaintyCp": null,
                "teacherDepth": 8, "teacherNodes": 25000,
                "openingClassification": "ibisha",
                "provenanceReferences": provenance
            }]
        });
        let checksum = format!("{:x}", Sha256::digest(serde_json::to_vec(&record).unwrap()));
        record
            .as_object_mut()
            .unwrap()
            .insert("recordChecksum".to_owned(), serde_json::json!(checksum));
        let mut encoder = GzEncoder::new(Vec::new(), Compression::fast());
        writeln!(encoder, "{}", serde_json::to_string(&record).unwrap()).unwrap();
        encoder.finish().unwrap()
    }

    #[test]
    fn handshake_has_identity_options_and_terminator() {
        let sink = Arc::new(MemorySink::default());
        let mut session = UsiSession::new(sink.clone());
        assert!(session.process_line("usi"));
        let lines = sink.lines();
        assert!(lines.first().unwrap().starts_with("id name OpenShogiAI "));
        assert!(lines.iter().any(|line| {
            line.starts_with("option name USI_Hash type spin default 32 min 1 max 1024")
        }));
        assert!(
            lines
                .iter()
                .any(|line| line == "option name USI_Ponder type check default false")
        );
        assert!(
            lines
                .iter()
                .any(|line| line == "option name Threads type spin default 1 min 1 max 1")
        );
        assert!(
            !lines
                .iter()
                .any(|line| line.starts_with("option name Hash "))
        );
        assert!(lines.iter().any(|line| {
            line == "option name ModelKind type combo default overall-champion var overall-champion var neural-float var neural-quantized var osaval02-float var osaval02-quantized"
        }));
        assert!(lines.iter().any(|line| {
            line == "option name RuntimeProfile type combo default standard var standard var pure_learned"
        }));
        assert!(lines.iter().any(|line| {
            line == "option name TimeSafetyMarginMs type spin default 50 min 0 max 1000"
        }));
        assert!(
            !lines
                .iter()
                .any(|line| line.starts_with("option name Opening"))
        );
        assert!(
            lines
                .iter()
                .any(|line| line == "option name ModelPath type filename default <empty>")
        );
        assert_eq!(lines.last().unwrap(), "usiok");
    }

    #[test]
    fn model_selection_loads_matching_model_and_preserves_failed_request() {
        let model = TemporaryModel::write(&model_bytes(NeuralQuantization::Float32));
        let sink = Arc::new(MemorySink::default());
        let mut session = UsiSession::new(sink.clone());

        assert!(session.process_line(&format!(
            "setoption name ModelPath value {}",
            model.display()
        )));
        assert!(session.process_line("setoption name ModelKind value neural-float"));
        assert_eq!(session.options().model_kind, ModelKind::NeuralFloat);
        assert_eq!(
            session
                .neural_evaluator
                .as_ref()
                .expect("loaded evaluator")
                .quantization(),
            NeuralQuantization::Float32
        );
        assert!(
            sink.lines()
                .iter()
                .any(|line| line.starts_with("info string model loaded neural-float "))
        );

        assert!(session.process_line("setoption name ModelKind value neural-quantized"));
        assert_eq!(session.options().model_kind, ModelKind::NeuralFloat);
        assert_eq!(session.options().model_path, model.display());
        assert_eq!(
            session
                .neural_evaluator
                .as_ref()
                .expect("previous valid evaluator is preserved")
                .quantization(),
            NeuralQuantization::Float32
        );
        assert!(sink.lines().iter().any(|line| {
            line.starts_with("info string error ") && line.contains("requires Int8 weights")
        }));
        assert!(session.process_line("isready"));
        assert_eq!(sink.lines().last().unwrap(), "readyok");

        let quantized = TemporaryModel::write(&model_bytes(NeuralQuantization::Int8));
        assert!(session.process_line(&format!(
            "setoption name ModelPath value {}",
            quantized.display()
        )));
        assert!(session.process_line("setoption name ModelKind value neural-quantized"));
        assert_eq!(session.options().model_kind, ModelKind::NeuralQuantized);
        assert_eq!(
            session
                .neural_evaluator
                .as_ref()
                .expect("loaded quantized evaluator")
                .quantization(),
            NeuralQuantization::Int8
        );
    }

    #[test]
    fn corrupt_model_and_missing_path_fail_closed_before_readyok() {
        let mut corrupt_bytes = model_bytes(NeuralQuantization::Float32);
        corrupt_bytes[60] ^= 1;
        let corrupt = TemporaryModel::write(&corrupt_bytes);
        let sink = Arc::new(MemorySink::default());
        let mut session = UsiSession::new(sink.clone());

        assert!(session.process_line(&format!(
            "setoption name ModelPath value {}",
            corrupt.display()
        )));
        assert!(session.process_line("setoption name ModelKind value neural-float"));
        assert_eq!(session.options().model_kind, ModelKind::NeuralFloat);
        assert!(session.neural_evaluator.is_none());
        assert!(
            sink.lines().iter().any(|line| {
                line.starts_with("info string error ") && line.contains("checksum")
            })
        );

        let empty_sink = Arc::new(MemorySink::default());
        let mut empty_session = UsiSession::new(empty_sink.clone());
        assert!(empty_session.process_line("setoption name ModelKind value neural-quantized"));
        assert!(!empty_session.process_line("isready"));
        assert_eq!(
            empty_session.options().model_kind,
            ModelKind::NeuralQuantized
        );
        let lines = empty_sink.lines();
        assert!(lines[0].contains("ModelPath is required"));
        assert!(!lines.iter().any(|line| line == "readyok"));
        assert!(!lines.iter().any(|line| line.starts_with("bestmove ")));
    }

    #[test]
    fn pure_learned_rejects_missing_hash_and_non_osaval02_before_game_start() {
        let sink = Arc::new(MemorySink::default());
        let mut session = UsiSession::new(sink.clone());
        assert!(session.process_line("setoption name RuntimeProfile value pure_learned"));
        assert!(!session.process_line("isready"));
        assert!(
            sink.lines()
                .iter()
                .any(|line| { line.contains("pure_learned requires an OSAVAL02 ModelKind") })
        );

        let model = TemporaryModel::write(&model_bytes(NeuralQuantization::Float32));
        let sink = Arc::new(MemorySink::default());
        let mut session = UsiSession::new(sink.clone());
        assert!(session.process_line("setoption name RuntimeProfile value pure_learned"));
        assert!(session.process_line(&format!(
            "setoption name ModelPath value {}",
            model.display()
        )));
        assert!(session.process_line("setoption name ModelKind value osaval02-float"));
        assert!(session.process_line(&format!(
            "setoption name ExpectedModelSha256 value {}",
            "a".repeat(64)
        )));
        assert!(!session.process_line("isready"));
        let lines = sink.lines();
        assert!(
            lines
                .iter()
                .any(|line| line.contains("failed to load OSAVAL02 model"))
        );
        assert!(!lines.iter().any(|line| line == "readyok"));
    }

    #[cfg(unix)]
    #[test]
    fn model_path_symlink_fails_closed_before_readyok_or_go() {
        use std::os::unix::fs::symlink;

        let model = TemporaryModel::write(&model_bytes(NeuralQuantization::Float32));
        let link_path = model.0.with_extension("link.osaval");
        symlink(&model.0, &link_path).expect("create model symlink");
        let sink = Arc::new(MemorySink::default());
        let mut session = UsiSession::new(sink.clone());

        assert!(session.process_line("setoption name ModelKind value neural-float"));
        assert!(session.process_line(&format!(
            "setoption name ModelPath value {}",
            link_path.to_string_lossy()
        )));
        assert!(session.neural_evaluator.is_none());
        assert!(!session.process_line("isready"));
        let lines = sink.lines();
        assert!(!lines.iter().any(|line| line == "readyok"));
        assert!(!lines.iter().any(|line| line.starts_with("bestmove ")));
        assert!(lines.iter().any(|line| {
            line.starts_with("info string error ") && line.contains("not a regular file")
        }));
        drop(lines);
        fs::remove_file(link_path).expect("remove model symlink");
    }

    #[cfg(unix)]
    #[test]
    fn model_path_ancestor_symlink_fails_closed_before_readyok() {
        use std::os::unix::fs::symlink;

        let root = std::env::temp_dir().canonicalize().unwrap().join(format!(
            "open-shogi-usi-model-ancestor-{}-{}",
            std::process::id(),
            NEXT_MODEL_FILE.fetch_add(1, Ordering::Relaxed)
        ));
        let external = root.with_extension("external");
        fs::create_dir_all(&root).expect("create model parent fixture");
        fs::create_dir_all(&external).expect("create external model fixture");
        fs::write(
            external.join("model.osaval"),
            model_bytes(NeuralQuantization::Float32),
        )
        .expect("write external model");
        symlink(&external, root.join("models")).expect("create ancestor symlink");
        let path = root.join("models/model.osaval");
        let sink = Arc::new(MemorySink::default());
        let mut session = UsiSession::new(sink.clone());

        assert!(session.process_line("setoption name ModelKind value neural-float"));
        assert!(session.process_line(&format!(
            "setoption name ModelPath value {}",
            path.to_string_lossy()
        )));
        assert!(session.neural_evaluator.is_none());
        assert!(!session.process_line("isready"));
        assert!(!sink.lines().iter().any(|line| line == "readyok"));
    }

    #[test]
    fn loaded_neural_snapshot_survives_source_removal() {
        let model = TemporaryModel::write(&model_bytes(NeuralQuantization::Float32));
        let sink = Arc::new(MemorySink::default());
        let mut session = UsiSession::new(sink.clone());

        assert!(session.process_line(&format!(
            "setoption name ModelPath value {}",
            model.display()
        )));
        assert!(session.process_line("setoption name ModelKind value neural-float"));
        fs::remove_file(&model.0).expect("remove loaded source model");
        assert!(session.process_line("isready"));
        assert!(sink.lines().iter().any(|line| line == "readyok"));
    }

    #[test]
    fn incomplete_neural_setup_can_be_corrected_before_readiness() {
        let sink = Arc::new(MemorySink::default());
        let mut session = UsiSession::new(sink.clone());
        assert!(session.process_line("setoption name ModelKind value neural-float"));

        assert!(session.active.is_none());
        assert!(
            !sink
                .lines()
                .iter()
                .any(|line| line.starts_with("bestmove "))
        );

        let model = TemporaryModel::write(&model_bytes(NeuralQuantization::Float32));
        assert!(session.process_line(&format!(
            "setoption name ModelPath value {}",
            model.display()
        )));
        assert!(session.process_line("isready"));
        assert!(sink.lines().iter().any(|line| line == "readyok"));
    }

    #[test]
    fn initial_invalid_neural_configuration_terminates_the_protocol_boundedly() {
        let sink = Arc::new(MemorySink::default());
        let mut input =
            Cursor::new(b"setoption name ModelKind value neural-float\nisready\nusi\n".to_vec());
        run_protocol(&mut input, sink.clone()).unwrap();
        let lines = sink.lines();
        assert!(lines.iter().any(|line| {
            line.starts_with("info string error ") && line.contains("ModelPath is required")
        }));
        assert!(!lines.iter().any(|line| line == "readyok"));
        assert!(!lines.iter().any(|line| line == "usiok"));
    }

    #[test]
    fn a_bad_position_is_transactional() {
        let sink = Arc::new(MemorySink::default());
        let mut session = UsiSession::new(sink.clone());
        let before = to_sfen(session.position());
        assert!(session.process_line("position startpos moves 7g7f 7g7f"));
        assert_eq!(to_sfen(session.position()), before);
        assert!(
            sink.lines()
                .iter()
                .any(|line| line.starts_with("info string error"))
        );
    }

    #[test]
    fn valid_position_and_options_commit() {
        let sink = Arc::new(MemorySink::default());
        let mut session = UsiSession::new(sink);
        assert!(session.process_line("position startpos moves 7g7f 3c3d"));
        assert_ne!(session.position(), &Position::startpos());
        assert!(session.process_line("setoption name USI_Hash value 64"));
        assert_eq!(session.options().hash_megabytes, 64);
        assert!(session.process_line("setoption name USI_Hash value 1025"));
        assert_eq!(session.options().hash_megabytes, 64);
        assert!(session.process_line("setoption name Hash value 32"));
        assert_eq!(session.options().hash_megabytes, 32);
        assert!(session.process_line("setoption name EvalMobility value false"));
        assert!(!session.options().evaluation.mobility);
    }

    #[test]
    fn ready_errors_and_quit_are_protocol_safe() {
        let sink = Arc::new(MemorySink::default());
        let mut session = UsiSession::new(sink.clone());
        assert!(session.process_line("isready"));
        assert!(session.process_line("unknown"));
        assert!(!session.process_line("quit"));
        let lines = sink.lines();
        assert_eq!(lines[0], "readyok");
        assert!(lines[1].starts_with("info string error "));
        assert!(!lines[1].contains('\n'));
    }

    #[test]
    fn info_formats_centipawn_and_signed_mate_scores() {
        let info = |score| SearchInfo {
            best_move: None,
            score,
            depth: 3,
            seldepth: 4,
            nodes: 5,
            elapsed: Duration::from_millis(1),
            nps: 5_000,
            pv: Vec::new(),
            root_moves: Vec::new(),
            stats: SearchStats::default(),
        };
        assert!(format_search_info(&info(42)).contains("score cp 42"));
        assert!(format_search_info(&info(MATE_SCORE - 3)).contains("score mate 2"),);
        assert!(format_search_info(&info(-MATE_SCORE + 3)).contains("score mate -2"),);
        assert!(format_search_info(&info(-MATE_SCORE)).contains("score mate 0"));
        assert!(!format_search_info(&info(-MATE_SCORE)).contains("mate -0"));
    }

    #[test]
    fn asynchronous_search_emits_exactly_one_bestmove() {
        let sink = Arc::new(MemorySink::default());
        let mut session = UsiSession::new(sink.clone());
        assert!(session.process_line("go nodes 50 depth 2"));
        for _ in 0..100 {
            if sink
                .lines()
                .iter()
                .any(|line| line.starts_with("bestmove "))
            {
                break;
            }
            thread::sleep(Duration::from_millis(5));
        }
        assert!(session.process_line("stop"));
        let lines = sink.lines();
        assert_eq!(
            lines
                .iter()
                .filter(|line| line.starts_with("bestmove "))
                .count(),
            1
        );
        assert!(lines.iter().any(|line| line.starts_with("info depth ")));
    }

    #[test]
    fn valid_opening_book_is_rejected_for_play() {
        let book = TemporaryModel::write(&opening_book_bytes());
        let sink = Arc::new(MemorySink::default());
        let mut session = UsiSession::new(sink.clone());
        assert!(session.process_line(&format!(
            "setoption name OpeningBookPath value {}",
            book.display()
        )));
        assert!(session.opening_book.is_none());
        let lines = sink.lines();
        assert!(
            lines
                .iter()
                .any(|line| line.contains("opening books are disabled"))
        );
        assert!(!lines.iter().any(|line| line.starts_with("bestmove ")));
    }

    #[test]
    fn rejected_go_resource_limit_never_spawns_a_worker() {
        let sink = Arc::new(MemorySink::default());
        let mut session = UsiSession::new(sink.clone());

        assert!(session.process_line("go nodes 1000000001"));
        assert!(session.active.is_none());
        assert!(
            sink.lines()
                .iter()
                .any(|line| line.starts_with("info string error "))
        );
        assert!(
            !sink
                .lines()
                .iter()
                .any(|line| line.starts_with("bestmove "))
        );
    }

    #[test]
    fn clock_allocation_handles_byoyomi_only_and_zero_clock() {
        let manager = open_shogi_core::TimeManager::default();
        let byoyomi = manager
            .plan(
                open_shogi_core::Side::Black,
                time_control(
                    &GoParameters {
                        byoyomi_ms: Some(500),
                        ..GoParameters::default()
                    },
                    super::DEFAULT_SAFETY_MARGIN_MS,
                ),
                8,
            )
            .unwrap();
        assert_eq!(
            byoyomi.allocated_hard_limit,
            Some(Duration::from_millis(500))
        );
        assert_eq!(byoyomi.hard_limit, Some(Duration::from_millis(450)));

        let zero_clock = manager
            .plan(
                open_shogi_core::Side::Black,
                time_control(
                    &GoParameters {
                        black_time_ms: Some(0),
                        ..GoParameters::default()
                    },
                    super::DEFAULT_SAFETY_MARGIN_MS,
                ),
                8,
            )
            .unwrap();
        assert_eq!(zero_clock.allocated_hard_limit, Some(Duration::ZERO));
        assert_eq!(zero_clock.hard_limit, Some(Duration::ZERO));
    }

    #[test]
    fn fischer_clocks_receive_the_increment_as_spendable_time() {
        for base in [0_u64, 60_000] {
            let plan = open_shogi_core::TimeManager::default()
                .plan_for_position(
                    &Position::startpos(),
                    time_control(
                        &GoParameters {
                            black_time_ms: Some(base),
                            white_time_ms: Some(base),
                            black_increment_ms: Some(1_000),
                            white_increment_ms: Some(1_000),
                            ..GoParameters::default()
                        },
                        super::DEFAULT_SAFETY_MARGIN_MS,
                    ),
                    8,
                )
                .unwrap();
            assert!(plan.mode == open_shogi_core::TimeControlMode::Clock);
            assert!(
                plan.hard_limit.unwrap_or_default() > Duration::ZERO,
                "base {base} ms with a positive increment must leave usable time"
            );
            assert!(
                plan.allocated_hard_limit.unwrap_or_default()
                    <= Duration::from_millis(base + 1_000),
                "the increment is credited once, never duplicated"
            );
        }
    }

    #[test]
    fn ponder_negotiation_holds_go_ponder_until_ponderhit() {
        let sink = Arc::new(MemorySink::default());
        let mut session = UsiSession::new(sink.clone());
        assert!(session.process_line("setoption name USI_Ponder value true"));
        assert!(
            sink.lines()
                .iter()
                .any(|line| line.contains("pondering is not implemented"))
        );
        assert!(session.process_line("setoption name USI_Ponder value false"));
        assert!(session.process_line("setoption name Threads value 1"));
        assert!(session.process_line("setoption name Threads value 8"));
        assert!(
            sink.lines()
                .iter()
                .any(|line| line.contains("only one search worker"))
        );

        assert!(session.process_line("go ponder"));
        assert!(session.pending_ponder.is_some());
        assert!(session.process_line("stop"));
        assert!(session.pending_ponder.is_none());
        assert!(session.process_line("go ponder"));
        assert!(session.pending_ponder.is_some());
        assert!(session.process_line("stop"));
        assert!(session.process_line("ponderhit"));
        assert!(
            sink.lines()
                .iter()
                .any(|line| line.contains("ponderhit without a held go ponder"))
        );
        assert!(
            sink.lines()
                .iter()
                .filter(|line| line.starts_with("bestmove "))
                .count()
                == 2,
            "each discarded go ponder answers stop so the GUI never stalls"
        );
    }

    #[test]
    fn go_mate_is_declined_without_leaving_the_protocol() {
        let sink = Arc::new(MemorySink::default());
        let mut session = UsiSession::new(sink.clone());
        assert!(session.process_line("go mate 5000"));
        assert!(session.process_line("isready"));
        assert!(session.active.is_none());
        let lines = sink.lines();
        assert!(lines.iter().any(|line| line == "checkmate notimplemented"));
        assert!(lines.iter().any(|line| line == "readyok"));
    }

    #[test]
    fn infinite_terminal_search_waits_for_stop_before_bestmove() {
        let sink = Arc::new(MemorySink::default());
        let mut session = UsiSession::new(sink.clone());
        assert!(session.process_line("position sfen 3lkl3/3pRp3/4G4/9/9/9/9/9/K8 w - 1"));
        assert!(session.process_line("go infinite"));
        thread::sleep(Duration::from_millis(25));
        assert!(
            !sink
                .lines()
                .iter()
                .any(|line| line.starts_with("bestmove "))
        );
        assert!(session.process_line("stop"));
        assert_eq!(
            sink.lines()
                .iter()
                .filter(|line| line.starts_with("bestmove "))
                .count(),
            1
        );
    }

    #[test]
    fn protocol_continues_after_draining_an_overlong_line() {
        let mut input = vec![b'x'; 4_097];
        input.extend_from_slice(b"\nusi\nquit\n");
        let mut reader = Cursor::new(input);
        let sink = Arc::new(MemorySink::default());
        run_protocol(&mut reader, sink.clone()).unwrap();
        let lines = sink.lines();
        assert!(
            lines
                .iter()
                .any(|line| line == "info string error command exceeds 4096-byte limit")
        );
        assert!(lines.iter().any(|line| line == "usiok"));
    }
}
