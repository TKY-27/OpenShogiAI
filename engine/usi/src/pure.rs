//! Pure-only USI session. The model is validated before the protocol starts and stays
//! immutable; only the transposition budget is negotiable. One search worker, ponder
//! never enabled, and every fatal failure is reported instead of becoming a move.

use std::{
    io::{self, BufRead},
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{
        Arc,
        mpsc::{self, Sender},
    },
    thread::{self, JoinHandle},
};

use open_shogi_core::{
    CancellationToken, Position, PurePlayingEvaluator, SearchConfig, SearchEngine, SearchOutcome,
    SearchTermination, TimeManager, TimePlan, is_mate_score, parse_sfen, parse_usi_move,
    to_usi_move,
};

use crate::{
    GoParameters, UsiCommand,
    clocks::time_control,
    engine_id_line,
    lifecycle::{
        BoundedLine, CompletionGate, OutputAuthority, ProtocolSink, WriterSink, format_search_info,
        format_search_stats, read_bounded_line, sanitized,
    },
    parse_command,
};

const MAX_LINE_BYTES: usize = 4_096;
const DEFAULT_HASH_MEGABYTES: usize = 32;
const MAX_HASH_MEGABYTES: usize = 1_024;
const DEFAULT_SAFETY_MARGIN_MS: u64 = 50;
const MODEL_SHORT_HASH_BYTES: usize = 8;

/// Typed protocol options of the pure session. Malformed values keep the previous
/// valid state; the model itself has no option surface at all.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PureOptions {
    hash_megabytes: usize,
}

impl Default for PureOptions {
    fn default() -> Self {
        Self {
            hash_megabytes: DEFAULT_HASH_MEGABYTES,
        }
    }
}

enum Event {
    Line(String),
    TooLong,
    Eof,
    /// A fatal worker failure tagged with its output generation. The session owner
    /// must observe it even while blocked on input; superseded generations are stale
    /// and ignored so they cannot tear down a healthy replacement search.
    Failed {
        generation: u64,
        message: String,
    },
    /// A fatal transport failure; never stale, always ends the session.
    ReaderFailed(String),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Flow {
    Continue,
    Stop,
}

#[derive(Clone, Copy)]
enum FinishPolicy {
    /// `stop`: the outstanding completion belongs to the accepted request and is
    /// published exactly once.
    Complete,
    /// New position, new game, game over, setoption, superseding `go`, quit, EOF,
    /// or failure: the completion is stale and must never reach the peer.
    Supersede,
}

struct PureWorker {
    cancellation: CancellationToken,
    completion_gate: Option<Arc<CompletionGate>>,
    handle: JoinHandle<()>,
}

/// Generation-guarded completion publisher owned by one search worker.
struct Completion {
    generation: u64,
    gate: Option<Arc<CompletionGate>>,
    authority: Arc<OutputAuthority>,
    sink: Arc<dyn ProtocolSink>,
}

impl Completion {
    fn line(&self, make_line: impl FnOnce() -> String) {
        self.authority
            .send_if_current(self.generation, self.sink.as_ref(), make_line);
    }

    /// `go infinite` retains its completed result but publishes only after the
    /// session releases the gate on `stop`.
    fn wait_for_release(&self) -> bool {
        self.gate.iter().all(|gate| gate.wait())
    }
}

struct PureSession<'a> {
    model: &'a PurePlayingEvaluator,
    model_hash: &'a str,
    options: PureOptions,
    output: Arc<dyn ProtocolSink>,
    authority: Arc<OutputAuthority>,
    events: mpsc::Receiver<Event>,
    event_sender: Sender<Event>,
    position: Position,
    history_initial: Position,
    history_moves: Vec<open_shogi_core::Move>,
    worker: Option<PureWorker>,
    /// A held `go ponder` request. Invariant: set only while no worker is active.
    pending_ponder: Option<GoParameters>,
}

/// Run USI with an immutable validated pure model and no alternate evaluator.
///
/// # Errors
/// Rejects wrong model identity, fatal worker or output failures, and stdin I/O errors.
/// Recoverable protocol problems are answered with bounded `info string` diagnostics.
pub fn run_pure_stdio(model: &PurePlayingEvaluator, hash: &str) -> Result<(), String> {
    // Fail-closed readiness through the exact constructor used for play.
    model.search_engine(SearchConfig::default(), hash)?;
    let output: Arc<dyn ProtocolSink> = Arc::new(WriterSink::new(io::stdout()));
    let (event_sender, events) = mpsc::channel();
    // The stdin lock lives inside the reader thread: `StdinLock` is not `Send`.
    let reader_sender = event_sender.clone();
    spawn_guarded_reader(
        move || {
            let mut reader = io::stdin().lock();
            read_events(&mut reader, &reader_sender);
        },
        event_sender.clone(),
    );
    session_loop(event_sender, events, output, model, hash)
}

/// Buffered-input variant of the session loop, used by protocol tests to drive
/// deterministic line sequences.
#[cfg(test)]
fn run_session<R: BufRead + Send + 'static>(
    reader: R,
    output: Arc<dyn ProtocolSink>,
    model: &PurePlayingEvaluator,
    hash: &str,
) -> Result<(), String> {
    let (event_sender, events) = mpsc::channel();
    let reader_sender = event_sender.clone();
    spawn_guarded_reader(
        move || {
            let mut reader = reader;
            read_events(&mut reader, &reader_sender);
        },
        event_sender.clone(),
    );
    session_loop(event_sender, events, output, model, hash)
}

fn session_loop(
    event_sender: Sender<Event>,
    events: mpsc::Receiver<Event>,
    output: Arc<dyn ProtocolSink>,
    model: &PurePlayingEvaluator,
    hash: &str,
) -> Result<(), String> {
    let mut session = PureSession {
        model,
        model_hash: hash,
        options: PureOptions::default(),
        output,
        authority: Arc::new(OutputAuthority::default()),
        events,
        event_sender,
        position: Position::startpos(),
        history_initial: Position::startpos(),
        history_moves: Vec::new(),
        worker: None,
        pending_ponder: None,
    };
    loop {
        let event = session
            .events
            .recv()
            .map_err(|_| "USI input channel closed".to_owned())?;
        match event {
            Event::Line(line) => {
                if session.handle_line(&line)? == Flow::Stop {
                    return Ok(());
                }
            }
            Event::TooLong => session.diagnose("command exceeds 4096-byte limit"),
            Event::Eof => {
                // EOF is a transport close, so terminate any otherwise infinite search.
                session.finish(FinishPolicy::Supersede)?;
                return Ok(());
            }
            Event::Failed {
                generation,
                message,
            } => {
                if session.authority.current() == generation {
                    session.diagnose(&message);
                    session.finish(FinishPolicy::Supersede)?;
                    return Err(message);
                }
            }
            Event::ReaderFailed(message) => {
                session.diagnose(&message);
                session.finish(FinishPolicy::Supersede)?;
                return Err(message);
            }
        }
    }
}

fn read_events<R: BufRead>(mut reader: R, event_sender: &Sender<Event>) {
    loop {
        let event = match read_bounded_line(&mut reader, MAX_LINE_BYTES) {
            Ok(BoundedLine::Line(line)) => Event::Line(line),
            Ok(BoundedLine::TooLong) => Event::TooLong,
            Ok(BoundedLine::Eof) => {
                let _ = event_sender.send(Event::Eof);
                return;
            }
            Err(error) => {
                let _ =
                    event_sender.send(Event::ReaderFailed(format!("USI input failed: {error}")));
                return;
            }
        };
        if event_sender.send(event).is_err() {
            return;
        }
    }
}

/// Spawns the reader with a panic guard so a reader-thread crash still ends the
/// session instead of leaving it blocked on the event channel forever.
fn spawn_guarded_reader<F>(read_loop: F, event_sender: Sender<Event>)
where
    F: FnOnce() + Send + 'static,
{
    thread::spawn(move || {
        let outcome = catch_unwind(AssertUnwindSafe(read_loop));
        if outcome.is_err() {
            let _ = event_sender.send(Event::ReaderFailed("USI reader thread panicked".to_owned()));
        }
    });
}

impl PureSession<'_> {
    fn handle_line(&mut self, line: &str) -> Result<Flow, String> {
        // Accept CRLF peers; `read_bounded_line` only strips the newline.
        let line = line.trim_end_matches('\r');
        if line.trim().is_empty() {
            return Ok(Flow::Continue);
        }
        match parse_command(line) {
            Ok(command) => self.handle_command(command),
            Err(error) => {
                let message = format!("USI protocol error: {error}");
                self.diagnose(&message);
                Ok(Flow::Continue)
            }
        }
    }

    fn handle_command(&mut self, command: UsiCommand) -> Result<Flow, String> {
        match command {
            UsiCommand::Usi => {
                self.send(&identity_line(self.model_hash));
                self.send("id author OpenShogiAI contributors");
                self.send(&format!(
                    "info string profile pure_learned model_format {} model_sha256 {}",
                    self.model.format(),
                    self.model_hash
                ));
                for line in option_lines() {
                    self.send(&line);
                }
                self.send("usiok");
            }
            UsiCommand::IsReady => {
                // Model identity was verified before the protocol started and is immutable.
                self.send("readyok");
            }
            UsiCommand::SetOption { name, value } => {
                self.finish(FinishPolicy::Supersede)?;
                self.set_option(&name, value.as_deref());
            }
            UsiCommand::NewGame => {
                self.finish(FinishPolicy::Supersede)?;
                self.pending_ponder = None;
                self.position = Position::startpos();
                self.history_initial = self.position.clone();
                self.history_moves.clear();
            }
            UsiCommand::PositionStartpos { moves } => {
                self.finish(FinishPolicy::Supersede)?;
                self.pending_ponder = None;
                self.set_position(Position::startpos(), &moves);
            }
            UsiCommand::PositionSfen { sfen, moves } => {
                self.finish(FinishPolicy::Supersede)?;
                self.pending_ponder = None;
                match parse_sfen(&sfen).map_err(|error| error.to_string()) {
                    Ok(initial) => self.set_position(initial, &moves),
                    Err(message) => self.diagnose(&message),
                }
            }
            UsiCommand::Go(parameters) => {
                if parameters.ponder {
                    // No ponder search exists: hold the request instead of answering the
                    // predicted position early or reinterpreting it as a timed search.
                    self.finish(FinishPolicy::Supersede)?;
                    self.pending_ponder = Some(parameters);
                    self.diagnose(
                        "go ponder held without pondering; ponderhit starts the search and stop discards it (USI_Ponder must stay false)",
                    );
                } else {
                    self.start_search(&parameters)?;
                }
            }
            UsiCommand::PonderHit(clocks) => match self.pending_ponder.take() {
                Some(mut pending) => {
                    pending.apply_clock_override(&clocks);
                    self.start_search(&pending)?;
                }
                None => self.diagnose("ponderhit without a held go ponder"),
            },
            UsiCommand::GoMate { .. } => {
                self.send("checkmate notimplemented");
            }
            UsiCommand::Stop => {
                if self.pending_ponder.take().is_some() {
                    // USI requires an answer to stop; the GUI discards it in this state.
                    self.diagnose("discarded go ponder without ponderhit");
                    self.send("bestmove resign");
                } else {
                    self.finish(FinishPolicy::Complete)?;
                }
            }
            UsiCommand::GameOver { .. } => {
                self.finish(FinishPolicy::Supersede)?;
                self.pending_ponder = None;
            }
            UsiCommand::Quit => {
                self.finish(FinishPolicy::Supersede)?;
                self.pending_ponder = None;
                return Ok(Flow::Stop);
            }
        }
        Ok(Flow::Continue)
    }

    fn set_option(&mut self, name: &str, value: Option<&str>) {
        let applied = match name {
            "USI_Hash" | "Hash" => parse_hash_megabytes(value).map(Some),
            "USI_Ponder" => match value {
                Some("false") => Ok(None),
                Some("true") => {
                    Err("pondering is not implemented; keep USI_Ponder false".to_owned())
                }
                _ => Err("USI_Ponder must be `true` or `false`".to_owned()),
            },
            "Threads" => match value {
                Some("1") => Ok(None),
                _ => Err("only one search worker is supported".to_owned()),
            },
            "RuntimeProfile" => match value {
                Some("pure_learned") => Ok(None),
                _ => Err(
                    "pure-only USI only accepts RuntimeProfile=pure_learned; model identity is immutable"
                        .to_owned(),
                ),
            },
            _ => Err(format!(
                "unknown option `{}`; accepted: USI_Hash, USI_Ponder, Threads, RuntimeProfile",
                sanitized(name)
            )),
        };
        match applied {
            Ok(Some(hash_megabytes)) => self.options.hash_megabytes = hash_megabytes,
            Ok(None) => {}
            Err(message) => self.diagnose(&message),
        }
    }

    /// Parses a new position into temporaries and commits only on full success; a
    /// legal move prefix never replaces the current valid state.
    fn set_position(&mut self, initial: Position, moves: &[String]) {
        match apply_moves(initial, moves) {
            Ok((initial, position, history)) => {
                self.history_initial = initial;
                self.history_moves = history;
                self.position = position;
            }
            Err(message) => self.diagnose(&message),
        }
    }

    fn start_search(&mut self, parameters: &GoParameters) -> Result<(), String> {
        self.finish(FinishPolicy::Supersede)?;
        self.pending_ponder = None;
        let plan = match TimeManager::default().plan_for_position(
            &self.position,
            time_control(parameters, DEFAULT_SAFETY_MARGIN_MS),
            open_shogi_core::MAX_TIME_CONTROL_DEPTH,
        ) {
            Ok(plan) => plan,
            Err(message) => {
                self.diagnose(&message);
                return Ok(());
            }
        };
        let entries = hash_entries(self.options.hash_megabytes);
        // One fresh engine per accepted request: the transposition table is rebuilt
        // with the negotiated budget and never carries state across games or moves.
        let mut engine = self
            .model
            .search_engine(
                SearchConfig {
                    transposition_entries: entries,
                    ..SearchConfig::default()
                },
                self.model_hash,
            )
            .map_err(|error| format!("pure search engine construction failed: {error}"))?;
        engine
            .set_pure_history(&self.history_initial, &self.history_moves)
            .map_err(|error| format!("pure history binding failed: {error}"))?;
        self.send(&format!(
            "info string hash {}MiB transposition_entries {entries} workers 1",
            self.options.hash_megabytes
        ));
        let generation = self.authority.advance();
        let cancellation = CancellationToken::new();
        let worker_cancellation = cancellation.clone();
        let completion_gate = parameters
            .infinite
            .then(|| Arc::new(CompletionGate::default()));
        let completion = Completion {
            generation,
            gate: completion_gate.clone(),
            authority: Arc::clone(&self.authority),
            sink: Arc::clone(&self.output),
        };
        let event_sender = self.event_sender.clone();
        let model_hash = self.model_hash.to_owned();
        let root = self.position.clone();
        let worker_generation = completion.generation;
        let handle = thread::spawn(move || {
            let outcome = catch_unwind(AssertUnwindSafe(|| {
                #[cfg(test)]
                if let Some(message) = injected_worker_failure() {
                    return Err(message);
                }
                run_search(
                    &mut engine,
                    &root,
                    plan,
                    &worker_cancellation,
                    &model_hash,
                    &completion,
                )
            }));
            let report = |message| {
                let _ = event_sender.send(Event::Failed {
                    generation: worker_generation,
                    message,
                });
            };
            match outcome {
                Ok(Ok(())) => {}
                Ok(Err(message)) => report(message),
                Err(_) => report("pure search worker panicked".to_owned()),
            }
        });
        self.worker = Some(PureWorker {
            cancellation,
            completion_gate,
            handle,
        });
        Ok(())
    }

    fn finish(&mut self, policy: FinishPolicy) -> Result<(), String> {
        let Some(worker) = self.worker.take() else {
            return Ok(());
        };
        if matches!(policy, FinishPolicy::Supersede) {
            self.authority.advance();
        }
        // Release a retained infinite-search completion before cancelling so its
        // worker always wakes up and exits.
        if let Some(gate) = &worker.completion_gate {
            gate.release();
        }
        worker.cancellation.cancel();
        worker
            .handle
            .join()
            .map_err(|_| "pure search worker panicked".to_owned())
    }

    fn send(&self, line: &str) {
        self.output.send(line);
    }

    fn diagnose(&self, message: &str) {
        self.send(&format!("info string error {}", sanitized(message)));
    }
}

fn run_search(
    engine: &mut SearchEngine,
    root: &Position,
    plan: TimePlan,
    cancellation: &CancellationToken,
    model_hash: &str,
    completion: &Completion,
) -> Result<(), String> {
    let result = engine.search_managed_with_callback(root, plan, cancellation, |info| {
        completion.line(|| format_search_info(info));
    });
    if result.termination == SearchTermination::EvaluationError {
        return Err("pure-only inference failed; no bestmove is available".to_owned());
    }
    let proof = engine.runtime_proof(result.stats, model_hash);
    if !proof.valid_pure_search(&result) {
        return Err("pure runtime proof failed".to_owned());
    }
    if !completion.wait_for_release() {
        return Ok(());
    }
    let proof_json = serde_json::to_string(&proof).map_err(|error| error.to_string())?;
    let outcome_json = serde_json::to_string(&result.outcome).map_err(|error| error.to_string())?;
    let score = score_field(result.score, result.outcome);
    // A rule score is reported as its own diagnostic line, never embedded in the
    // search-statistics line.
    let rule_score_line = score
        .strip_prefix('\n')
        .map(ToOwned::to_owned)
        .unwrap_or_default();
    let score_suffix = if rule_score_line.is_empty() {
        score
    } else {
        String::new()
    };
    let pv = result
        .pv
        .iter()
        .copied()
        .map(to_usi_move)
        .collect::<Vec<_>>()
        .join(" ");
    completion.line(|| {
        let mut line = format!(
            "{}{}",
            format_search_stats(
                result.depth,
                result.seldepth,
                result.nodes,
                result.nps,
                result.elapsed.as_millis(),
            ),
            score_suffix
        );
        if !pv.is_empty() {
            line.push_str(" pv ");
            line.push_str(&pv);
        }
        line
    });
    if !rule_score_line.is_empty() {
        completion.line(move || rule_score_line);
    }
    completion.line(|| format!("info string search_outcome {outcome_json}"));
    completion.line(|| format!("info string pure-proof {proof_json}"));
    completion.line(|| {
        format!(
            "bestmove {}",
            result
                .best_move
                .map_or_else(|| "resign".to_owned(), to_usi_move)
        )
    });
    Ok(())
}

fn identity_line(model_hash: &str) -> String {
    let short = model_hash
        .get(..MODEL_SHORT_HASH_BYTES)
        .unwrap_or(model_hash);
    format!("{} pure_learned {short}", engine_id_line())
}

fn option_lines() -> [String; 4] {
    [
        format!(
            "option name USI_Hash type spin default {DEFAULT_HASH_MEGABYTES} min 1 max {MAX_HASH_MEGABYTES}"
        ),
        "option name USI_Ponder type check default false".to_owned(),
        "option name Threads type spin default 1 min 1 max 1".to_owned(),
        "option name RuntimeProfile type combo default pure_learned var pure_learned".to_owned(),
    ]
}

fn parse_hash_megabytes(value: Option<&str>) -> Result<usize, String> {
    let value = value
        .ok_or_else(|| "USI_Hash requires a value".to_owned())?
        .parse::<usize>()
        .map_err(|_| "USI_Hash must be an integer".to_owned())?;
    if !(1..=MAX_HASH_MEGABYTES).contains(&value) {
        return Err(format!("USI_Hash must be 1..={MAX_HASH_MEGABYTES}"));
    }
    Ok(value)
}

fn hash_entries(megabytes: usize) -> usize {
    SearchEngine::transposition_entries_for_megabytes(megabytes).max(1)
}

fn apply_moves(
    initial: Position,
    moves: &[String],
) -> Result<(Position, Position, Vec<open_shogi_core::Move>), String> {
    let mut position = initial.clone();
    let mut history = Vec::new();
    for movement in moves {
        let parsed = parse_usi_move(movement).map_err(|error| error.to_string())?;
        position
            .make_move(parsed)
            .map_err(|error| error.to_string())?;
        history.push(parsed);
    }
    Ok((initial, position, history))
}

fn score_field(score: i32, outcome: SearchOutcome) -> String {
    use open_shogi_core::SearchOutcome;
    if !outcome.has_score() {
        return String::new();
    }
    // Only the root checkmate outcome certifies the kind of terminal. A searched
    // mate-range value can also originate in a descendant no-legal-move loss or
    // perpetual check; its magnitude alone is not a checkmate proof. Progress lines
    // (shared `format_search_info`) may still label searched mate distances as
    // `mate N`; this completion line deliberately stays conservative.
    if outcome == SearchOutcome::Checkmate {
        // ShogiHome parses `0`/`+0` as a won, mating position and `-0` as already
        // mated; a checkmated root is always the losing side.
        return " score mate -0".to_owned();
    }
    if is_mate_score(score)
        || matches!(
            outcome,
            SearchOutcome::NoLegalMoves | SearchOutcome::PerpetualCheck
        )
    {
        return format!("\ninfo string rule_score {score}");
    }
    format!(" score cp {score}")
}

#[cfg(test)]
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};

/// Harmless failure seam for worker-notification tests; never compiled into shipped builds.
#[cfg(test)]
static INJECTED_WORKER_FAILURE: AtomicBool = AtomicBool::new(false);

#[cfg(test)]
fn injected_worker_failure() -> Option<String> {
    INJECTED_WORKER_FAILURE
        .load(AtomicOrdering::SeqCst)
        .then(|| "injected pure search worker failure".to_owned())
}

#[cfg(test)]
mod tests {
    use std::{
        io::{BufReader, Cursor, Write as _},
        os::unix::net::UnixStream,
        path::PathBuf,
        sync::{Arc, atomic::Ordering::SeqCst},
        thread::{self, JoinHandle},
        time::{Duration, Instant},
    };

    use open_shogi_core::{MATE_SCORE, Position, SearchOutcome};
    use sha2::{Digest, Sha256};

    use super::{
        DEFAULT_HASH_MEGABYTES, INJECTED_WORKER_FAILURE, MAX_LINE_BYTES, PureOptions, apply_moves,
        hash_entries, run_session, score_field,
    };
    use crate::lifecycle::{ProtocolSink, test_support::MemorySink};

    static NEXT_FIXTURE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    struct TemporaryModel(PathBuf);

    impl TemporaryModel {
        fn write(bytes: &[u8]) -> (Self, String) {
            let sequence = NEXT_FIXTURE.fetch_add(1, SeqCst);
            let path = std::env::temp_dir().canonicalize().unwrap().join(format!(
                "open-shogi-usi-pure-model-{}-{sequence}.osaval03",
                std::process::id()
            ));
            std::fs::write(&path, bytes).expect("write temporary model");
            let hash = format!("{:x}", Sha256::digest(bytes));
            (Self(path), hash)
        }

        fn display(&self) -> String {
            self.0.to_string_lossy().into_owned()
        }
    }

    impl Drop for TemporaryModel {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    /// Minimal valid OSAVAL03 artifact (frozen zero weights) for protocol tests.
    fn osaval03_fixture() -> Vec<u8> {
        const HEADER_BYTES: usize = 44;
        const WIDTH: usize = open_shogi_core::PHASE10V_ACCUMULATOR_WIDTH;
        let payload_bytes = (open_shogi_core::PHASE10V_FEATURE_COUNT * WIDTH
            + WIDTH
            + 3 * WIDTH * open_shogi_core::PHASE10V_HIDDEN_WIDTH
            + open_shogi_core::PHASE10V_HIDDEN_WIDTH
            + open_shogi_core::PHASE10V_HIDDEN_WIDTH * open_shogi_core::PHASE10V_HEAD_COUNT
            + open_shogi_core::PHASE10V_HEAD_COUNT)
            * 4;
        let mut bytes = open_shogi_core::PHASE10V_MODEL_MAGIC.to_vec();
        for value in [3_u32, 1, 8_427, u32::try_from(WIDTH).unwrap(), 16, 4] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes.extend_from_slice(&71_u64.to_le_bytes());
        bytes.extend_from_slice(&1_f32.to_le_bytes());
        bytes.resize(HEADER_BYTES + payload_bytes, 0);
        let checksum = Sha256::digest(&bytes);
        bytes.extend_from_slice(&checksum);
        bytes
    }

    struct LoadedModel {
        _temporary: TemporaryModel,
        hash: String,
        engine: open_shogi_core::PurePlayingEvaluator,
    }

    fn loaded_model() -> LoadedModel {
        let bytes = osaval03_fixture();
        let (temporary, hash) = TemporaryModel::write(&bytes);
        let engine = open_shogi_core::PurePlayingEvaluator::load_file(
            "OSAVAL03",
            temporary.display(),
            &hash,
        )
        .expect("fixture model loads");
        LoadedModel {
            _temporary: temporary,
            hash,
            engine,
        }
    }

    /// Runs the session over a pipe so tests can wait for asynchronous completions
    /// before sending the next command.
    struct PipedSession {
        writer: Option<UnixStream>,
        sink: Arc<MemorySink>,
        result: JoinHandle<Result<(), String>>,
    }

    impl PipedSession {
        fn start(model: &open_shogi_core::PurePlayingEvaluator, hash: &str) -> Self {
            let (writer, reader) = UnixStream::pair().expect("session pipe");
            let sink = Arc::new(MemorySink::default());
            let output: Arc<dyn ProtocolSink> = Arc::clone(&sink) as _;
            let result = {
                let model = model.clone();
                let hash = hash.to_owned();
                thread::spawn(move || run_session(BufReader::new(reader), output, &model, &hash))
            };
            Self {
                writer: Some(writer),
                sink,
                result,
            }
        }

        fn send(&self, line: &str) {
            let mut writer = self.writer.as_ref().expect("open session pipe");
            writer
                .write_all(format!("{line}\n").as_bytes())
                .expect("write protocol line");
            writer.flush().expect("flush protocol line");
        }

        fn wait_for(&self, prefix: &str) -> bool {
            self.wait_for_within(prefix, Duration::from_secs(10))
        }

        fn wait_for_within(&self, prefix: &str, window: Duration) -> bool {
            let deadline = Instant::now() + window;
            while Instant::now() < deadline {
                if self
                    .sink
                    .lines()
                    .iter()
                    .any(|line| line.starts_with(prefix))
                {
                    return true;
                }
                thread::sleep(Duration::from_millis(5));
            }
            false
        }

        fn bestmove_count(&self) -> usize {
            self.sink
                .lines()
                .iter()
                .filter(|line| line.starts_with("bestmove "))
                .count()
        }

        fn wait_for_bestmove_count(&self, target: usize) -> bool {
            let deadline = Instant::now() + Duration::from_secs(10);
            while Instant::now() < deadline {
                if self.bestmove_count() >= target {
                    return true;
                }
                thread::sleep(Duration::from_millis(5));
            }
            false
        }

        /// Closes the pipe and asserts the session ended cleanly.
        fn finish(mut self) -> Result<(), String> {
            self.writer.take();
            self.result.join().expect("session thread does not panic")
        }
    }

    /// Fully buffered input for sessions with no asynchronous waiting.
    fn drive(
        model: &open_shogi_core::PurePlayingEvaluator,
        hash: &str,
        input: &str,
    ) -> (Arc<MemorySink>, Result<(), String>) {
        let sink = Arc::new(MemorySink::default());
        let output: Arc<dyn ProtocolSink> = Arc::clone(&sink) as _;
        let result = run_session(Cursor::new(input.as_bytes().to_vec()), output, model, hash);
        (sink, result)
    }

    #[test]
    fn handshake_identifies_pure_profile_and_negotiated_options() {
        let model = loaded_model();
        let (sink, result) = drive(&model.engine, &model.hash, "usi\nisready\nquit\n");
        result.expect("session completes");
        let lines = sink.lines();
        let identity = lines
            .iter()
            .find(|line| line.starts_with("id name "))
            .expect("identity line")
            .clone();
        assert!(identity.contains("pure_learned"));
        assert!(identity.contains(&model.hash[..8]));
        assert!(
            lines
                .iter()
                .any(|line| line == "option name USI_Hash type spin default 32 min 1 max 1024")
        );
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
        assert!(lines.iter().any(|line| line
            == "option name RuntimeProfile type combo default pure_learned var pure_learned"));
        assert!(lines.iter().any(|line| {
            line.starts_with("info string profile pure_learned model_format OSAVAL03 model_sha256 ")
                && line.contains(&model.hash)
        }));
        assert!(lines.iter().any(|line| line == "usiok"));
        assert!(lines.iter().any(|line| line == "readyok"));
        assert_eq!(lines.last().unwrap(), "readyok");
    }

    #[test]
    fn setoption_updates_hash_and_tolerates_unknown_or_rejected_settings() {
        let model = loaded_model();
        let input = concat!(
            "setoption name USI_Hash value 64\n",
            "setoption name USI_Hash value 1025\n",
            "setoption name USI_Hash value notanumber\n",
            "setoption name Hash value 128\n",
            "setoption name USI_Ponder value true\n",
            "setoption name USI_Ponder value false\n",
            "setoption name Threads value 1\n",
            "setoption name Threads value 4\n",
            "setoption name RuntimeProfile value standard\n",
            "setoption name EvaluationAttack value 50\n",
            "setoption name ClearNet value true\n",
            "isready\nquit\n",
        );
        let (sink, result) = drive(&model.engine, &model.hash, input);
        result.expect("recoverable options never end the session");
        let lines = sink.lines();
        assert!(lines.iter().any(|line| line == "readyok"));
        assert!(
            lines
                .iter()
                .any(|line| line.contains("USI_Hash must be 1..=1024"))
        );
        assert!(
            lines
                .iter()
                .any(|line| line.contains("USI_Hash must be an integer"))
        );
        assert!(
            lines
                .iter()
                .any(|line| line.contains("pondering is not implemented"))
        );
        assert!(
            lines
                .iter()
                .any(|line| line.contains("only one search worker"))
        );
        assert!(
            lines
                .iter()
                .any(|line| line.contains("RuntimeProfile=pure_learned"))
        );
        assert!(
            lines
                .iter()
                .any(|line| line.contains("unknown option `EvaluationAttack`"))
        );
        assert!(
            lines
                .iter()
                .any(|line| line.contains("unknown option `ClearNet`"))
        );
    }

    #[test]
    fn bad_position_is_transactional_and_protocol_survives_unknown_input() {
        let model = loaded_model();
        let input = concat!(
            "\n",
            "nonsense command\n",
            "position startpos moves 7g7f 7g7e\n",
            "isready\n",
            "quit\n",
        );
        let (sink, result) = drive(&model.engine, &model.hash, input);
        result.expect("session survives malformed input");
        let lines = sink.lines();
        assert!(lines.iter().any(|line| line == "readyok"));
        assert!(lines.iter().any(|line| line.contains("not legal")));
        assert!(
            lines
                .iter()
                .any(|line| line.contains("unknown or malformed"))
        );
    }

    #[test]
    fn go_mate_answers_notimplemented_and_stays_usable() {
        let model = loaded_model();
        let input = "go mate 5000\ngo mate infinite\nisready\nquit\n";
        let (sink, result) = drive(&model.engine, &model.hash, input);
        result.expect("session completes");
        let lines = sink.lines();
        assert_eq!(
            lines
                .iter()
                .filter(|line| line.as_str() == "checkmate notimplemented")
                .count(),
            2
        );
        assert!(lines.iter().any(|line| line == "readyok"));
    }

    #[test]
    fn crlf_peers_receive_a_normal_handshake() {
        let model = loaded_model();
        let input = "usi\r\nisready\r\nquit\r\n";
        let (sink, result) = drive(&model.engine, &model.hash, input);
        result.expect("CRLF line endings are tolerated");
        let lines = sink.lines();
        assert!(lines.iter().any(|line| line == "usiok"));
        assert!(lines.iter().any(|line| line == "readyok"));
        assert!(
            !lines
                .iter()
                .any(|line| line.contains("forbidden control character"))
        );
    }

    #[test]
    fn overlong_lines_are_diagnosed_without_ending_the_session() {
        let model = loaded_model();
        let mut input = "x".repeat(MAX_LINE_BYTES + 1);
        input.push_str("\nisready\nquit\n");
        let (sink, result) = drive(&model.engine, &model.hash, &input);
        result.expect("overlong input is bounded and recoverable");
        let lines = sink.lines();
        assert!(lines.iter().any(|line| line == "readyok"));
        assert!(
            lines
                .iter()
                .any(|line| line.contains("exceeds 4096-byte limit"))
        );
    }

    #[test]
    fn finite_search_publishes_progress_and_a_single_completion() {
        let model = loaded_model();
        let session = PipedSession::start(&model.engine, &model.hash);
        session.send("setoption name USI_Hash value 16");
        session.send("position startpos moves 7g7f 3c3d");
        session.send("go nodes 200 depth 3");
        assert!(session.wait_for("bestmove "), "completion arrives");
        let lines = session.sink.lines();
        assert!(lines.iter().any(|line| line.starts_with("info depth ")));
        assert!(lines.iter().any(|line| line.contains(" pure-proof ")));
        assert!(
            lines
                .iter()
                .any(|line| line.starts_with("info string hash 16MiB "))
        );
        let bestmove = lines
            .iter()
            .find(|line| line.starts_with("bestmove "))
            .expect("completion recorded")
            .clone();
        let movement = bestmove.split(' ').nth(1).expect("move token").to_owned();
        drop(lines);
        assert_eq!(session.bestmove_count(), 1);
        assert_ne!(movement, "resign");
        session.finish().expect("clean exit");
    }

    #[test]
    fn applied_hash_budget_matches_the_advertised_option() {
        let model = loaded_model();
        let session = PipedSession::start(&model.engine, &model.hash);
        session.send("setoption name USI_Hash value 16");
        session.send("position startpos");
        session.send("go movetime 100");
        assert!(session.wait_for("bestmove "));
        let expected = format!(
            "info string hash 16MiB transposition_entries {} workers 1",
            hash_entries(16)
        );
        assert!(session.sink.lines().contains(&expected));
        session.finish().expect("clean exit");
    }

    #[test]
    fn infinite_search_at_terminal_root_waits_for_stop() {
        let model = loaded_model();
        let session = PipedSession::start(&model.engine, &model.hash);
        session.send("position sfen 3lkl3/3pRp3/4G4/9/9/9/9/9/K8 w - 1");
        session.send("go infinite");
        assert!(
            !session.wait_for_within("bestmove ", Duration::from_millis(500)),
            "an infinite search never completes before stop"
        );
        session.send("stop");
        assert!(session.wait_for("bestmove "));
        assert_eq!(session.bestmove_count(), 1);
        assert!(
            session
                .sink
                .lines()
                .iter()
                .any(|line| line.contains("search_outcome"))
        );
        session.finish().expect("clean exit");
    }

    #[test]
    fn superseded_infinite_search_publishes_nothing() {
        let model = loaded_model();
        let session = PipedSession::start(&model.engine, &model.hash);
        session.send("position startpos");
        session.send("go infinite");
        session.send("position startpos moves 7g7f");
        session.send("go movetime 100");
        assert!(session.wait_for("bestmove "));
        assert_eq!(
            session.bestmove_count(),
            1,
            "the superseded infinite request never publishes its completion"
        );
        session.finish().expect("clean exit");
    }

    #[test]
    fn idle_and_repeated_stop_produce_no_completion() {
        let model = loaded_model();
        let session = PipedSession::start(&model.engine, &model.hash);
        session.send("position startpos");
        session.send("stop");
        session.send("stop");
        assert!(!session.wait_for_within("bestmove ", Duration::from_millis(300)));
        session.send("go movetime 100");
        assert!(session.wait_for("bestmove "));
        assert_eq!(session.bestmove_count(), 1);
        session.finish().expect("clean exit");
    }

    #[test]
    fn go_ponder_is_held_then_started_by_ponderhit_with_early_ponder_clocks() {
        let model = loaded_model();
        let session = PipedSession::start(&model.engine, &model.hash);
        session.send("position startpos");
        session.send("go ponder");
        session.send("stop");
        assert!(session.wait_for("bestmove "));
        {
            let lines = session.sink.lines();
            assert!(
                lines
                    .iter()
                    .any(|line| line.contains("go ponder held without pondering"))
            );
            assert!(
                lines
                    .iter()
                    .any(|line| line.contains("discarded go ponder without ponderhit"))
            );
        }
        session.send("position startpos");
        session.send("go ponder");
        session.send("ponderhit btime 30000 wtime 30000 binc 500 winc 500");
        assert!(
            session.wait_for_bestmove_count(2),
            "ponderhit completion arrives"
        );
        assert_eq!(
            session.bestmove_count(),
            2,
            "one ignored discard, one real completion"
        );
        assert!(
            session
                .sink
                .lines()
                .iter()
                .any(|line| line.starts_with("info string hash 32MiB "))
        );
        session.finish().expect("clean exit");
    }

    #[test]
    fn clocked_go_receives_increment_time() {
        let model = loaded_model();
        let session = PipedSession::start(&model.engine, &model.hash);
        session.send("position startpos");
        session.send("go btime 0 wtime 0 binc 1000 winc 1000");
        assert!(
            session.wait_for("bestmove "),
            "a zero base clock with a positive increment still yields a move"
        );
        session.finish().expect("clean exit");
    }

    #[test]
    fn worker_failure_is_reported_promptly_without_a_move() {
        let model = loaded_model();
        let session = PipedSession::start(&model.engine, &model.hash);
        session.send("position startpos");
        INJECTED_WORKER_FAILURE.store(true, SeqCst);
        session.send("go movetime 100");
        assert!(
            session.wait_for("info string error"),
            "the failure is observable without further input"
        );
        INJECTED_WORKER_FAILURE.store(false, SeqCst);
        assert_eq!(
            session.bestmove_count(),
            0,
            "a failed search never fabricates a move"
        );
        let error = session
            .finish()
            .expect_err("the injected failure is fatal for the session");
        assert!(error.contains("injected"));
    }

    #[test]
    fn searched_rule_wins_are_not_reported_as_checkmate_and_mated_roots_keep_the_losing_sign() {
        assert_eq!(score_field(123, SearchOutcome::Evaluated), " score cp 123");
        assert_eq!(
            score_field(MATE_SCORE - 1, SearchOutcome::Evaluated),
            "\ninfo string rule_score 29999"
        );
        assert_eq!(
            score_field(-MATE_SCORE, SearchOutcome::Checkmate),
            " score mate -0"
        );
        assert_eq!(
            score_field(-MATE_SCORE, SearchOutcome::NoLegalMoves),
            "\ninfo string rule_score -30000"
        );
        assert_eq!(score_field(0, SearchOutcome::CancelledBeforeEvaluation), "");
    }

    #[test]
    fn positions_require_legal_moves() {
        assert!(apply_moves(Position::startpos(), &["7g7f".into(), "3c3d".into()]).is_ok());
        assert!(apply_moves(Position::startpos(), &["7g7e".into()]).is_err());
    }

    #[test]
    fn hash_option_bounds_match_the_advertised_spin_range() {
        let options = PureOptions::default();
        assert_eq!(options.hash_megabytes, DEFAULT_HASH_MEGABYTES);
        assert_eq!(super::parse_hash_megabytes(Some("1")).unwrap(), 1);
        assert_eq!(super::parse_hash_megabytes(Some("1024")).unwrap(), 1_024);
        assert!(super::parse_hash_megabytes(Some("0")).is_err());
        assert!(super::parse_hash_megabytes(Some("1025")).is_err());
        assert!(super::parse_hash_megabytes(None).is_err());
        assert!(super::parse_hash_megabytes(Some("32.5")).is_err());
    }
}
