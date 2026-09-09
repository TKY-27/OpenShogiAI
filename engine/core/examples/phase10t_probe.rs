//! Bounded synthetic-position runtime probe; this is not an Arena strength estimate.
use std::{env, fs, sync::Arc, time::Duration};

use open_shogi_core::{
    CancellationToken, Osaval02Evaluator, Osaval02History, Position, SearchConfig, SearchEngine,
    SearchLimits, overall_champion_evaluation, parse_usi_move, to_sfen,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let paths: Vec<String> = env::args().skip(1).collect();
    if paths.is_empty() || paths.len() > 2 {
        return Err("usage: phase10t_probe MODEL [MODEL]".into());
    }
    let mut models = vec![("handcrafted-overall-champion".to_owned(), None)];
    let mut identities = Vec::new();
    for path in paths {
        let evaluator = Arc::new(Osaval02Evaluator::load_file(&path)?);
        identities.push(json!({"path": path, "identity": evaluator.identity()}));
        models.push((path, Some(evaluator)));
    }
    let sequences = [
        "",
        "7g7f 3c3d",
        "7g7f 8c8d 2g2f 8d8e 8h7g 3a4b",
        "7g7f 3c3d 8h2b+ 3a2b",
    ];
    let mut runs = Vec::<Value>::new();
    let mut static_outputs = Vec::<Value>::new();
    for (position_id, sequence) in sequences.iter().enumerate() {
        let mut position = Position::startpos();
        for movement in sequence.split_whitespace() {
            position.make_move(parse_usi_move(movement)?)?;
        }
        for (name, model) in &models {
            if let Some(model) = model {
                static_outputs.push(json!({"model": name, "position_id": position_id,
                    "inference": model.infer(&position, Osaval02History::default())?}));
            }
            for (budget, limits) in [
                (
                    "equal_nodes_256",
                    SearchLimits {
                        max_depth: 32,
                        max_nodes: Some(256),
                        movetime: None,
                    },
                ),
                (
                    "equal_time_100ms",
                    SearchLimits {
                        max_depth: 32,
                        max_nodes: None,
                        movetime: Some(Duration::from_millis(100)),
                    },
                ),
            ] {
                for repetition in 0..3 {
                    let config = SearchConfig {
                        evaluation: overall_champion_evaluation(),
                        ..SearchConfig::default()
                    };
                    let mut engine = match model {
                        Some(model) => SearchEngine::with_pure_learned(
                            config,
                            Arc::clone(model),
                            &model.identity().artifact_sha256,
                        )?,
                        None => SearchEngine::new(config),
                    };
                    let result = engine.search(&position, limits, &CancellationToken::new());
                    runs.push(json!({"model": name, "position_id": position_id, "sfen": to_sfen(&position),
                        "budget": budget, "repetition": repetition, "nodes": result.nodes,
                        "elapsed_ns": result.elapsed.as_nanos(), "nps": result.nps, "completed_depth": result.depth,
                        "seldepth": result.seldepth, "score_cp": result.score, "qnodes": result.stats.qnodes,
                        "learned_calls": result.stats.learned_eval_calls, "handcrafted_calls": result.stats.handcrafted_eval_calls,
                        "inference_ns": result.stats.neural_inference_time.as_nanos(),
                        "inference_errors": result.stats.osaval02_inference_errors, "fallback_count": result.stats.fallback_count,
                        "residual_calls": result.stats.residual_eval_calls, "composite_calls": result.stats.composite_eval_calls,
                        "termination": format!("{:?}", result.termination)}));
                }
            }
        }
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "schema": "open_shogiai_phase10t_runtime_probe/v1", "purpose": "synthetic runtime diagnostic; no strength inference",
            "source_sha256": format!("{:x}", Sha256::digest(fs::read(file!())?)),
            "platform": {"os": env::consts::OS, "arch": env::consts::ARCH},
            "policy_ablation": "unavailable: existing API has no independent policy toggle; move-ordering switch is confounded",
            "model_loading_timed": false, "fresh_search_engine_per_run": true,
            "identities": identities, "runs": runs, "static_outputs": static_outputs,
        }))?
    );
    Ok(())
}
