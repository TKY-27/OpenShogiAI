//! Bounded parity, inference-cost and actual-search-leaf evidence; never an Arena run.
use open_shogi_core::{
    CancellationToken, Phase10VEvaluator, Position, SearchConfig, SearchEngine, SearchLimits,
    parse_sfen, parse_usi_move, to_sfen, to_usi_move,
};
use serde_json::json;
use std::{
    env,
    hint::black_box,
    sync::Arc,
    time::{Duration, Instant},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().skip(1).collect();
    let Some(path) = args.first() else {
        return Err("usage: phase10v_probe MODEL [--infer SFEN]".into());
    };
    let model = Arc::new(Phase10VEvaluator::load_file(path)?);
    if args.get(1).is_some_and(|value| value == "--infer") {
        if args.len() != 3 {
            return Err("--infer requires one quoted SFEN".into());
        }
        let position = parse_sfen(&args[2])?;
        let result = model.infer(&position);
        println!(
            "{}",
            json!({"cp": result.cp, "wdl_logits": result.wdl_logits,
            "model_sha256": model.identity().artifact_sha256})
        );
        return Ok(());
    }
    if args.len() != 1 {
        return Err("usage: phase10v_probe MODEL [--infer SFEN]".into());
    }
    let mut runs = Vec::new();
    let mut inference = Vec::new();
    for sequence in ["", "7g7f 3c3d", "7g7f 3c3d 8h2b+ 3a2b"] {
        let mut position = Position::startpos();
        for movement in sequence.split_whitespace() {
            position.make_move(parse_usi_move(movement)?)?;
        }
        let state = model.accumulator(&position)?;
        for _ in 0..16 {
            black_box(model.infer_accumulator(black_box(&state))?);
        }
        let started = Instant::now();
        for _ in 0..1000 {
            black_box(model.infer_accumulator(black_box(&state))?);
        }
        let ns = started.elapsed().as_nanos() / 1000;
        let started = Instant::now();
        for _ in 0..100 {
            black_box(model.infer(black_box(&position)));
        }
        let full_ns = started.elapsed().as_nanos() / 100;
        inference.push(
            json!({"sfen": to_sfen(&position), "cached_inference_ns": ns, "full_inference_ns": full_ns,
            "cp": model.infer_accumulator(&state)?.cp}),
        );
        for (budget, limits) in [
            (
                "nodes_128",
                SearchLimits {
                    max_depth: 16,
                    max_nodes: Some(128),
                    movetime: None,
                },
            ),
            (
                "clock_50ms",
                SearchLimits {
                    max_depth: 16,
                    max_nodes: None,
                    movetime: Some(Duration::from_millis(50)),
                },
            ),
        ] {
            let mut engine = SearchEngine::with_phase10v(
                SearchConfig::default(),
                model.clone(),
                &model.identity().artifact_sha256,
            )?;
            engine.set_leaf_trace_limit(32);
            let result = engine.search(&position, limits, &CancellationToken::new());
            let legal = result
                .best_move
                .is_some_and(|m| position.legal_moves().contains(&m));
            let proof = engine.runtime_proof(result.stats, &model.identity().artifact_sha256);
            if !legal || !proof.valid_pure_learned() {
                return Err("short-search legality or runtime proof failed".into());
            }
            runs.push(
                json!({"sfen":to_sfen(&position),"budget":budget,"nodes":result.nodes,
                "elapsed_ns":result.elapsed.as_nanos(),"nps":result.nps,"score_cp":result.score,
                "depth":result.depth,"best_move":result.best_move.map(to_usi_move),
                "inference_ns":result.stats.neural_inference_time.as_nanos(),"proof":proof,
                "leaves":engine.take_leaf_trace()}),
            );
        }
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({"schema":"open_shogiai_phase10v_runtime_probe/v1",
        "purpose":"bounded runtime and real search-leaf diagnostic; no playing-strength claim",
        "model_sha256":model.identity().artifact_sha256,"width":model.identity().accumulator_width,
        "build_class":if cfg!(feature="pure-only") {"pure-only"} else {"development"},
        "inference":inference,"runs":runs}))?
    );
    Ok(())
}
