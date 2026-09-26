//! Immutable single-player JSON-lines transport for the separately reviewed Arena coordinator.
use open_shogi_core::{
    CancellationToken, Game, PurePlayingEvaluator, SearchConfig, SearchEngine, SearchLimits,
    parse_sfen, parse_usi_move, to_sfen, to_usi_move,
};
use serde::Deserialize;
use std::{
    collections::BTreeMap,
    io::{self, BufRead, Write},
    time::{Duration, Instant},
};

const HARD_DEADLINE_SAFETY_MARGIN: Duration = Duration::from_millis(5);

#[derive(Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
struct Request {
    initial_sfen: String,
    moves: Vec<String>,
    depth: u8,
    nodes: Option<u64>,
    movetime_ms: Option<u64>,
    hard_timeout_ms: u64,
    hash_mb: usize,
}

pub fn run(arguments: &[String]) -> Result<(), String> {
    let mut options = BTreeMap::new();
    let (pairs, remainder) = arguments.as_chunks::<2>();
    for pair in pairs {
        if !matches!(
            pair[0].as_str(),
            "--model" | "--model-sha256" | "--model-format" | "--profile"
        ) || options.insert(pair[0].as_str(), pair[1].as_str()).is_some()
        {
            return Err("unknown or duplicate arena-player option".into());
        }
    }
    if !remainder.is_empty() {
        return Err("option requires a value".into());
    }
    let model = match options.get("--profile").copied() {
        Some("pure_learned") => {
            if !cfg!(feature = "pure-only") {
                return Err("pure player requires a certified pure-only build".into());
            }
            let get = |key| {
                options
                    .get(key)
                    .copied()
                    .ok_or_else(|| format!("required option {key}"))
            };
            Some(PurePlayingEvaluator::load_file(
                get("--model-format")?,
                get("--model")?,
                get("--model-sha256")?,
            )?)
        }
        #[cfg(feature = "handcrafted")]
        Some("handcrafted_experimental") if options.len() == 1 => None,
        _ => return Err("unsupported arena-player profile for this artifact".into()),
    };
    let mut input = io::stdin().lock();
    let mut output = io::stdout().lock();
    let ready = serde_json::json!({"schema":"open_shogiai_phase10u_arena_player_ready/v1", "ready":true, "adapter_format":model.as_ref().map_or("HANDCRAFTED", PurePlayingEvaluator::format), "model_format":model.as_ref().map_or("HANDCRAFTED", PurePlayingEvaluator::format), "model_sha256":model.as_ref().map(PurePlayingEvaluator::artifact_sha256), "profile":options.get("--profile"), "compiled_evaluators":open_shogi_core::COMPILED_EVALUATORS});
    serde_json::to_writer(&mut output, &ready).map_err(|e| e.to_string())?;
    writeln!(output)
        .and_then(|()| output.flush())
        .map_err(|e| e.to_string())?;
    loop {
        let mut bytes = Vec::new();
        let read = std::io::Read::take(&mut input, 16385)
            .read_until(b'\n', &mut bytes)
            .map_err(|e| e.to_string())?;
        if read == 0 {
            return Ok(());
        }
        if bytes.len() > 16384 {
            return Err("arena-player request exceeds byte limit".into());
        }
        let started = Instant::now();
        let request: Request = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
        let response = search(model.as_ref(), &request, started)?;
        serde_json::to_writer(&mut output, &response).map_err(|e| e.to_string())?;
        writeln!(output)
            .and_then(|()| output.flush())
            .map_err(|e| e.to_string())?;
    }
}

fn search(
    model: Option<&PurePlayingEvaluator>,
    request: &Request,
    started: Instant,
) -> Result<serde_json::Value, String> {
    if !(1..=64).contains(&request.depth)
        || request.nodes == Some(0)
        || request.movetime_ms == Some(0)
        || request.hard_timeout_ms == 0
        || request.hard_timeout_ms > 60000
        || request.hash_mb != 32
        || request.moves.len() > 500
        || (request.nodes.is_none() && request.movetime_ms.is_none())
    {
        return Err("invalid or unbounded arena-player limits".into());
    }
    if request
        .movetime_ms
        .is_some_and(|ms| ms > request.hard_timeout_ms)
    {
        return Err("movetime exceeds hard timeout".into());
    }
    let initial = parse_sfen(&request.initial_sfen).map_err(|e| e.to_string())?;
    let mut game = Game::new(initial.clone());
    let mut moves = Vec::new();
    for token in &request.moves {
        let movement = parse_usi_move(token).map_err(|e| e.to_string())?;
        game.play(movement).map_err(|e| e.to_string())?;
        moves.push(movement);
    }
    // The standard evaluator's search does not consume exact repetition history. Preserve
    // its existing rejection rather than returning a move for an already adjudicated game.
    if model.is_none() && matches!(game.end(), Some(open_shogi_core::GameEnd::Repetition(_))) {
        return Err("cannot search a terminal game".into());
    }
    let config = search_config(request.hash_mb);
    let mut engine = match model {
        Some(model) => model.search_engine(config, model.artifact_sha256())?,
        #[cfg(feature = "handcrafted")]
        None => SearchEngine::new(config),
        #[cfg(feature = "pure-only")]
        None => return Err("pure-only player requires a model".into()),
    };
    if model.is_some() {
        engine.set_pure_history(&initial, &moves)?;
    }
    let setup_elapsed = started.elapsed();
    let hard_limit = Duration::from_millis(request.hard_timeout_ms);
    let full_budget = Duration::from_millis(request.movetime_ms.unwrap_or(request.hard_timeout_ms));
    let (remaining, soft_budget_exhausted_in_setup, hard_deadline_safety_margin) = search_budget(
        setup_elapsed,
        request.movetime_ms.map(Duration::from_millis),
        hard_limit,
    )?;
    let search_started = started.elapsed();
    let result = engine.search(
        game.position(),
        SearchLimits {
            max_depth: request.depth,
            max_nodes: request.nodes,
            movetime: Some(remaining),
        },
        &CancellationToken::new(),
    );
    let elapsed = started.elapsed();
    if result.termination == open_shogi_core::SearchTermination::EvaluationError {
        return Err("pure inference failed".into());
    }
    let proof = engine.runtime_proof(
        result.stats,
        model.map_or("", PurePlayingEvaluator::artifact_sha256),
    );
    if model.is_some() && !proof.valid_pure_search(&result) {
        return Err("pure runtime proof failed".into());
    }
    let deadline = serde_json::json!({"clock":"monotonic", "scope":"request_parse_replay_setup_search", "requested_movetime_ms":request.movetime_ms, "hard_timeout_ms":request.hard_timeout_ms, "setup_elapsed_ns":setup_elapsed.as_nanos(), "search_start_ns":search_started.as_nanos(), "search_budget_ns":remaining.as_nanos(), "search_elapsed_ns":result.elapsed.as_nanos(), "elapsed_ns":elapsed.as_nanos(), "soft_budget_ns":request.movetime_ms.map(|ms| u128::from(ms) * 1_000_000), "hard_budget_ns":hard_limit.as_nanos(), "soft_budget_exhausted_in_setup":soft_budget_exhausted_in_setup, "hard_deadline_safety_margin_ns":hard_deadline_safety_margin.as_nanos(), "soft_compliant":request.movetime_ms.map(|_| elapsed <= full_budget), "hard_compliant":elapsed <= hard_limit, "compliant":elapsed <= hard_limit});
    if elapsed > hard_limit {
        return Err(format!("per-search hard deadline exceeded: {deadline}"));
    }
    Ok(
        serde_json::json!({"schema":"open_shogiai_arena_player/v1", "model_format":model.map_or("HANDCRAFTED", PurePlayingEvaluator::format), "model_sha256":model.map_or("", PurePlayingEvaluator::artifact_sha256), "sfen":to_sfen(game.position()), "final_sfen":to_sfen(game.position()), "requested_controls":request, "best_move":result.best_move.map(to_usi_move), "score":result.outcome.has_score().then_some(result.score), "outcome":result.outcome, "depth":result.depth, "nodes":result.nodes, "termination":format!("{:?}",result.termination), "proof":proof, "deadline":deadline, "timing":deadline, "threads":1, "hash_mb":request.hash_mb, "transposition_entries":config.transposition_entries, "depth_limit":request.depth, "node_limit":request.nodes}),
    )
}

fn search_budget(
    setup_elapsed: Duration,
    soft_budget: Option<Duration>,
    hard_budget: Duration,
) -> Result<(Duration, bool, Duration), String> {
    let hard_remaining = hard_budget
        .checked_sub(setup_elapsed)
        .filter(|duration| !duration.is_zero())
        .ok_or("deadline exhausted during setup")?;
    match soft_budget {
        Some(soft) if setup_elapsed >= soft => Ok((
            hard_remaining
                .checked_sub(HARD_DEADLINE_SAFETY_MARGIN)
                .filter(|duration| !duration.is_zero())
                .ok_or("deadline exhausted during setup")?,
            true,
            HARD_DEADLINE_SAFETY_MARGIN,
        )),
        Some(soft) => Ok((
            soft.checked_sub(setup_elapsed)
                .filter(|duration| !duration.is_zero())
                .ok_or("deadline exhausted during setup")?,
            false,
            Duration::ZERO,
        )),
        None => Ok((hard_remaining, false, Duration::ZERO)),
    }
}

fn search_config(hash_mb: usize) -> SearchConfig {
    SearchConfig {
        #[cfg(feature = "handcrafted")]
        evaluation: open_shogi_core::EvaluationConfig::handcrafted_experimental(),
        transposition_entries: hash_mb * 1024 * 1024
            / SearchEngine::transposition_entry_size_bytes(),
        ..SearchConfig::default()
    }
}

#[cfg(test)]
mod tests {
    use super::Request;
    #[cfg(feature = "handcrafted")]
    #[test]
    fn pure_transport_returns_rule_terminal_without_weakening_normal_proof() {
        use std::{io::Read, sync::Arc};
        let mut bytes = Vec::new();
        flate2::read::GzDecoder::new(
            &include_bytes!("../../../tests/fixtures/osaval02/pure-history.osaval02.gz")[..],
        )
        .read_to_end(&mut bytes)
        .unwrap();
        let evaluator = open_shogi_core::Osaval02Evaluator::from_bytes(&bytes).unwrap();
        let model = open_shogi_core::PurePlayingEvaluator::Osaval02(Arc::new(evaluator));
        let mut request = Request {
            initial_sfen: "4k4/3P1P3/5K3/9/9/9/9/9/9 b - 1".into(),
            moves: vec!["4c5c".into()],
            depth: 64,
            nodes: Some(2000),
            movetime_ms: None,
            hard_timeout_ms: 30000,
            hash_mb: 32,
        };
        let terminal = super::search(Some(&model), &request, std::time::Instant::now()).unwrap();
        assert_eq!(terminal["outcome"], "no_legal_moves");
        assert_eq!(terminal["best_move"], serde_json::Value::Null);
        assert_eq!(terminal["score"], -open_shogi_core::MATE_SCORE);
        assert_eq!(terminal["nodes"], 0);
        assert_eq!(terminal["proof"]["learned_eval_calls"], 0);
        request.initial_sfen = "4k4/9/9/9/9/9/9/9/4K4 b - 1".into();
        request.moves.clear();
        request.depth = 1;
        request.nodes = Some(8);
        let normal = super::search(Some(&model), &request, std::time::Instant::now()).unwrap();
        assert_eq!(normal["outcome"], "evaluated");
        assert!(normal["proof"]["learned_eval_calls"].as_u64().unwrap() > 0);
        for response in [&terminal, &normal] {
            for counter in [
                "handcrafted_eval_calls",
                "residual_eval_calls",
                "composite_eval_calls",
                "book_hits",
                "teacher_calls",
                "fallback_count",
            ] {
                assert_eq!(response["proof"][counter], 0);
            }
        }
    }
    #[cfg(feature = "handcrafted")]
    #[test]
    fn experimental_player_uses_the_named_evaluator_configuration() {
        assert_eq!(
            super::search_config(32).evaluation,
            open_shogi_core::EvaluationConfig::handcrafted_experimental()
        );
        assert_eq!(
            super::search_config(32).runtime_profile,
            open_shogi_core::RuntimeProfile::Standard
        );
    }

    #[cfg(feature = "handcrafted")]
    #[test]
    fn request_deadline_includes_setup_and_exact_controls() {
        let request = Request {
            initial_sfen: open_shogi_core::to_sfen(&open_shogi_core::Position::startpos()),
            moves: vec![],
            depth: 1,
            nodes: Some(8),
            movetime_ms: None,
            hard_timeout_ms: 30000,
            hash_mb: 32,
        };
        let result = super::search(None, &request, std::time::Instant::now()).unwrap();
        let timing = &result["timing"];
        assert_eq!(timing["hard_compliant"], true);
        assert_eq!(timing["soft_compliant"], serde_json::Value::Null);
        assert!(
            timing["elapsed_ns"].as_u64().unwrap()
                >= timing["setup_elapsed_ns"].as_u64().unwrap()
                    + timing["search_elapsed_ns"].as_u64().unwrap()
        );
        assert_eq!(result["requested_controls"]["nodes"], 8);
        assert!(
            super::search(
                None,
                &request,
                std::time::Instant::now()
                    .checked_sub(std::time::Duration::from_secs(31))
                    .unwrap()
            )
            .is_err()
        );
    }

    #[test]
    fn protocol_rejects_unknown_fields_and_wrong_types() {
        assert!(serde_json::from_str::<Request>(r#"{"initial_sfen":"x","moves":[],"depth":2,"hard_timeout_ms":100,"hash_mb":32,"fallback":true}"#).is_err());
        assert!(
            serde_json::from_str::<Request>(
                r#"{"initial_sfen":"x","moves":[],"depth":2,"hard_timeout_ms":"100","hash_mb":32}"#
            )
            .is_err()
        );
    }

    #[test]
    fn setup_soft_overrun_uses_only_remaining_hard_budget() {
        let (remaining, exhausted, margin) = super::search_budget(
            std::time::Duration::from_millis(101),
            Some(std::time::Duration::from_millis(100)),
            std::time::Duration::from_secs(1),
        )
        .unwrap();
        assert_eq!(remaining, std::time::Duration::from_millis(894));
        assert!(exhausted);
        assert_eq!(margin, std::time::Duration::from_millis(5));
    }

    #[test]
    fn setup_hard_overrun_stays_closed() {
        let error = super::search_budget(
            std::time::Duration::from_secs(1),
            Some(std::time::Duration::from_millis(100)),
            std::time::Duration::from_secs(1),
        )
        .unwrap_err();
        assert_eq!(error, "deadline exhausted during setup");
    }
}
