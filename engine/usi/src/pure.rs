//! Pure-only USI session. The model is validated before the protocol starts.
use crate::{UsiCommand, engine_id_line, parse_command};
use open_shogi_core::{
    CancellationToken, Position, PurePlayingEvaluator, SearchConfig, SearchEngine, TimeControl,
    TimeManager, parse_sfen, parse_usi_move, to_usi_move,
};
use std::{
    io::{self, BufRead, Write},
    sync::{Arc, Mutex},
    thread::{self, JoinHandle},
};

type Worker = (CancellationToken, JoinHandle<Result<(), String>>);

/// Run USI with an immutable validated pure model and no alternate evaluator.
///
/// # Errors
/// Rejects wrong model identity, unsupported options, invalid positions and I/O failures.
pub fn run_pure_stdio(model: &PurePlayingEvaluator, hash: &str) -> Result<(), String> {
    // Check the exact same constructor used by CLI and Wasm before advertising readiness.
    model.search_engine(SearchConfig::default(), hash)?;
    let output = Arc::new(Mutex::new(io::stdout()));
    let mut position = Position::startpos();
    let mut history_initial = position.clone();
    let mut history_moves = Vec::new();
    let mut worker: Option<Worker> = None;
    let mut input = io::stdin().lock();
    let result = (|| {
        loop {
            let Some(command) = read_command(&mut input)? else {
                break;
            };
            match command {
                UsiCommand::Usi => emit(
                    &output,
                    &format!(
                        "{}\nid author OpenShogiAI contributors\noption name RuntimeProfile type combo default pure_learned var pure_learned\nusiok",
                        engine_id_line()
                    ),
                )?,
                UsiCommand::IsReady => emit(&output, "readyok")?,
                UsiCommand::SetOption { name, value } => {
                    if name != "RuntimeProfile" || value.as_deref() != Some("pure_learned") {
                        return Err("pure-only USI only accepts RuntimeProfile=pure_learned; model identity is immutable".to_owned());
                    }
                }
                UsiCommand::PositionStartpos { moves } => {
                    finish(&mut worker, true)?;
                    history_initial = Position::startpos();
                    (position, history_moves) = apply_moves(history_initial.clone(), &moves)?;
                }
                UsiCommand::PositionSfen { sfen, moves } => {
                    finish(&mut worker, true)?;
                    history_initial = parse_sfen(&sfen).map_err(|error| error.to_string())?;
                    (position, history_moves) = apply_moves(history_initial.clone(), &moves)?;
                }
                UsiCommand::NewGame => {
                    finish(&mut worker, true)?;
                    position = Position::startpos();
                    history_initial = position.clone();
                    history_moves.clear();
                }
                UsiCommand::Go(parameters) => {
                    finish(&mut worker, true)?;
                    let plan = TimeManager::default().plan(
                        position.side_to_move(),
                        TimeControl {
                            black_time_ms: parameters.black_time_ms,
                            white_time_ms: parameters.white_time_ms,
                            byoyomi_ms: parameters.byoyomi_ms,
                            black_increment_ms: parameters.black_increment_ms,
                            white_increment_ms: parameters.white_increment_ms,
                            movetime_ms: parameters.movetime_ms,
                            nodes: parameters.nodes,
                            depth: parameters.depth,
                            infinite: parameters.infinite,
                            casual: false,
                            safety_margin_ms: 50,
                        },
                        64,
                    )?;
                    let mut engine = model.search_engine(SearchConfig::default(), hash)?;
                    engine.set_pure_history(&history_initial, &history_moves)?;
                    let cancellation = CancellationToken::new();
                    let token = cancellation.clone();
                    let root = position.clone();
                    let sink = Arc::clone(&output);
                    let model_hash = hash.to_owned();
                    worker = Some((
                        cancellation,
                        thread::spawn(move || {
                            let result = engine.search_managed(&root, plan, &token);
                            emit_result(&engine, &result, &model_hash, &sink)
                        }),
                    ));
                }
                UsiCommand::Stop | UsiCommand::GameOver { .. } => finish(&mut worker, true)?,
                UsiCommand::Quit => {
                    finish(&mut worker, true)?;
                    break;
                }
            }
        }
        // EOF is a transport close, so terminate any otherwise infinite search.
        finish(&mut worker, true)
    })();
    finish(&mut worker, true)?;
    result
}

fn apply_moves(
    mut position: Position,
    moves: &[String],
) -> Result<(Position, Vec<open_shogi_core::Move>), String> {
    let mut history = Vec::new();
    for movement in moves {
        let parsed = parse_usi_move(movement).map_err(|error| error.to_string())?;
        position
            .make_move(parsed)
            .map_err(|error| error.to_string())?;
        history.push(parsed);
    }
    Ok((position, history))
}

fn finish(worker: &mut Option<Worker>, cancel: bool) -> Result<(), String> {
    if let Some((token, handle)) = worker.take() {
        if cancel {
            token.cancel();
        }
        handle
            .join()
            .map_err(|_| "USI search worker panicked".to_owned())??;
    }
    Ok(())
}

fn emit(output: &Arc<Mutex<io::Stdout>>, message: &str) -> Result<(), String> {
    let mut sink = output
        .lock()
        .map_err(|_| "USI output lock poisoned".to_owned())?;
    writeln!(sink, "{message}")
        .and_then(|()| sink.flush())
        .map_err(|error| error.to_string())
}

fn score_field(score: i32, outcome: open_shogi_core::SearchOutcome) -> String {
    use open_shogi_core::SearchOutcome;
    if !outcome.has_score() {
        return String::new();
    }
    // Only the root checkmate outcome certifies the kind of terminal. A searched
    // mate-range value can also originate in a descendant no-legal-move loss or
    // perpetual check; its magnitude alone is not a checkmate proof.
    if outcome == SearchOutcome::Checkmate {
        return " score mate 0".to_owned();
    }
    if open_shogi_core::is_mate_score(score)
        || matches!(
            outcome,
            SearchOutcome::NoLegalMoves | SearchOutcome::PerpetualCheck
        )
    {
        return format!("\ninfo string rule_score {score}");
    }
    format!(" score cp {score}")
}

fn emit_result(
    engine: &SearchEngine,
    result: &open_shogi_core::SearchResult,
    hash: &str,
    sink: &Arc<Mutex<io::Stdout>>,
) -> Result<(), String> {
    if result.termination == open_shogi_core::SearchTermination::EvaluationError {
        return Err("pure-only inference failed; no bestmove is available".to_owned());
    }
    let proof = engine.runtime_proof(result.stats, hash);
    if !proof.valid_pure_search(result) {
        return Err("pure runtime proof failed".to_owned());
    }
    let proof_json = serde_json::to_string(&proof).map_err(|error| error.to_string())?;
    let outcome_json = serde_json::to_string(&result.outcome).map_err(|error| error.to_string())?;
    let score = score_field(result.score, result.outcome);
    emit(
        sink,
        &format!(
            "info depth {} nodes {}{}\ninfo string search_outcome {}\ninfo string pure-proof {}\nbestmove {}",
            result.depth,
            result.nodes,
            score,
            outcome_json,
            proof_json,
            result
                .best_move
                .map_or_else(|| "resign".to_owned(), to_usi_move)
        ),
    )
}

fn read_command(input: &mut impl BufRead) -> Result<Option<UsiCommand>, String> {
    let mut bytes = Vec::new();
    // Bound allocation even when a peer sends a line without a newline.
    let read = std::io::Read::take(input, 4098)
        .read_until(b'\n', &mut bytes)
        .map_err(|error| error.to_string())?;
    if read == 0 {
        return Ok(None);
    }
    if bytes.len() > 4097 {
        return Err("USI line exceeds limit".to_owned());
    }
    let line = std::str::from_utf8(&bytes)
        .map_err(|error| error.to_string())?
        .trim_end_matches(['\r', '\n']);
    parse_command(line)
        .map(Some)
        .map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    use super::{apply_moves, score_field};
    use open_shogi_core::{MATE_SCORE, Position, SearchOutcome};
    #[test]
    fn searched_rule_wins_are_not_reported_as_checkmate() {
        assert_eq!(score_field(123, SearchOutcome::Evaluated), " score cp 123");
        // A nonterminal root can reach a no-legal-move child at one ply.
        assert_eq!(
            score_field(MATE_SCORE - 1, SearchOutcome::Evaluated),
            "\ninfo string rule_score 29999"
        );
        assert_eq!(
            score_field(-MATE_SCORE + 3, SearchOutcome::Evaluated),
            "\ninfo string rule_score -29997"
        );
        assert_eq!(
            score_field(-MATE_SCORE, SearchOutcome::Checkmate),
            " score mate 0"
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
}
