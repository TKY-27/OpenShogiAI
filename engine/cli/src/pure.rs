//! Strict format-selected pure command boundary; development tools are not declared in this build.
use std::collections::BTreeMap;

use open_shogi_core::{
    CancellationToken, Position, PurePlayingEvaluator, SearchConfig, SearchLimits, parse_sfen,
    to_usi_move,
};

pub fn run(arguments: &[String]) -> Result<(), String> {
    if arguments.first().is_some_and(|s| s == "arena-player") {
        return crate::arena_player::run(&arguments[1..]);
    }
    let command = arguments.first().map_or("help", String::as_str);
    if matches!(command, "help" | "--help") && arguments.len() <= 1 {
        println!(
            "open-shogi-cli pure|usi --model FILE --model-sha256 SHA256 [--model-format OSAVAL02|OSAT10A1|OSAVAL03] --profile pure_learned [--sfen SFEN] [--depth N] [--nodes N] [--history-json JSON]"
        );
        return Ok(());
    }
    if !matches!(command, "pure" | "usi") {
        return Err("pure-only build accepts only pure, usi and help".to_owned());
    }
    let options = options(&arguments[1..])?;
    let required = |name| {
        options
            .get(name)
            .copied()
            .ok_or_else(|| format!("required option {name}"))
    };
    if required("--profile")? != "pure_learned" {
        return Err("profile must be pure_learned".to_owned());
    }
    let hash = required("--model-sha256")?;
    let model = PurePlayingEvaluator::load_file(
        options.get("--model-format").copied().unwrap_or("OSAT10A1"),
        required("--model")?,
        hash,
    )?;
    if command == "usi" {
        if options.keys().any(|key| {
            matches!(
                *key,
                "--sfen" | "--nodes" | "--depth" | "--history-json" | "--leaf-trace-limit"
            )
        }) {
            return Err(
                "USI search limits and history must be supplied through go and position".to_owned(),
            );
        }
        // Readiness and the per-search engines are built inside the session; building
        // one here would only be a duplicate transposition allocation.
        return open_shogi_usi::run_pure_stdio(&model, hash);
    }
    let mut engine = model.search_engine(SearchConfig::default(), hash)?;
    let position = options.get("--sfen").map_or_else(
        || Ok(Position::startpos()),
        |sfen| parse_sfen(sfen).map_err(|error| error.to_string()),
    )?;
    let depth = options.get("--depth").map_or(Ok(2), |value| {
        value.parse::<u8>().map_err(|error| error.to_string())
    })?;
    let nodes = options.get("--nodes").map_or(Ok(64), |value| {
        value.parse::<u64>().map_err(|error| error.to_string())
    })?;
    if !(1..=64).contains(&depth) || nodes == 0 {
        return Err("depth must be 1..64 and nodes positive".to_owned());
    }
    // Standalone diagnostic facts never replace the search engine's authoritative game history.
    let history = options.get("--history-json").map_or_else(
        || Ok(open_shogi_core::Osaval02History::default()),
        |json| parse_history(json),
    )?;
    let trace_limit = options.get("--leaf-trace-limit").map_or(Ok(0), |value| {
        value.parse::<usize>().map_err(|error| error.to_string())
    })?;
    if trace_limit > 10_000 || (trace_limit > 0 && model.format() != "OSAVAL03") {
        return Err("leaf trace requires OSAVAL03 and a limit no larger than 10000".into());
    }
    engine.set_leaf_trace_limit(trace_limit);
    let result = engine.search(
        &position,
        SearchLimits {
            max_depth: depth,
            max_nodes: Some(nodes),
            movetime: None,
        },
        &CancellationToken::new(),
    );
    if result.termination == open_shogi_core::SearchTermination::EvaluationError {
        return Err("pure-only inference failed; no result is available".to_owned());
    }
    let proof = engine.runtime_proof(result.stats, hash);
    if !proof.valid_pure_search(&result) {
        return Err("pure runtime proof failed".to_owned());
    }
    let inference = if result.outcome == open_shogi_core::SearchOutcome::Evaluated {
        model.infer(&position, history)?
    } else {
        serde_json::Value::Null
    };
    println!(
        "{}",
        serde_json::json!({"schema":"open_shogiai_phase10t_pure_runtime/v1", "compiled_evaluators":open_shogi_core::COMPILED_EVALUATORS, "profile":"pure_learned", "model_sha256":hash, "history":history, "model_format":model.format(), "sfen":open_shogi_core::to_sfen(&position), "cp":inference["cp"], "wdl_logits":inference.get("wdl_logits"), "inference":inference, "best_move":result.best_move.map(to_usi_move), "score":result.outcome.has_score().then_some(result.score),"outcome":result.outcome,"depth":result.depth,"nodes":result.nodes,"proof":proof,"leaf_trace":engine.take_leaf_trace()})
    );
    Ok(())
}

fn options(arguments: &[String]) -> Result<BTreeMap<&str, &str>, String> {
    let mut options = BTreeMap::new();
    let mut chunks = arguments.chunks_exact(2);
    for pair in &mut chunks {
        if !matches!(
            pair[0].as_str(),
            "--model"
                | "--model-format"
                | "--model-sha256"
                | "--profile"
                | "--sfen"
                | "--depth"
                | "--nodes"
                | "--history-json"
                | "--leaf-trace-limit"
        ) {
            return Err(format!("unknown option {}", pair[0]));
        }
        if options.insert(pair[0].as_str(), pair[1].as_str()).is_some() {
            return Err(format!("duplicate option {}", pair[0]));
        }
    }
    if !chunks.remainder().is_empty() {
        return Err("option requires a value".to_owned());
    }
    Ok(options)
}

fn parse_history(json: &str) -> Result<open_shogi_core::Osaval02History, String> {
    if json.len() > 1024 {
        return Err("history JSON exceeds byte limit".to_owned());
    }
    serde_json::from_str(json).map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    use super::{parse_history, run};
    #[test]
    fn history_json_rejects_unknown_fields_types_and_oversize() {
        assert!(parse_history(r#"{"available":true,"repetition_count":2}"#).is_err());
        assert!(parse_history(r#"{"available":"yes"}"#).is_err());
        assert!(parse_history(&" ".repeat(1025)).is_err());
        assert!(
            parse_history(r#"{"available":true,"repetitionCount":2,"continuousCheckByUs":true}"#)
                .is_ok()
        );
    }

    #[test]
    fn missing_model_and_wrong_profile_fail_closed() {
        assert!(run(&["pure".into()]).is_err());
        assert!(run(&["pure".into(), "--profile".into(), "standard".into()]).is_err());
        assert!(run(&["arena".into()]).is_err());
        assert!(
            run(&[
                "pure".into(),
                "--profile".into(),
                "pure_learned".into(),
                "--model".into(),
                "/does/not/exist".into()
            ])
            .is_err()
        );
    }
}
