//! Small source-game generator and JSON-lines probe for the computation prototype.
use open_shogi_core::{
    CancellationToken, ComputationModel, Game, Phase10VEvaluator, Position, RandomMoveSelector,
    SearchConfig, SearchEngine, SearchInfo, SearchLimits, TimeControl, TimeManager,
    computation_features, parse_sfen, parse_usi_move, to_sfen, to_usi_move,
};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    env,
    io::{self, BufRead},
    sync::Arc,
    time::Duration,
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    #[serde(default)]
    moves: Vec<String>,
    sfen: Option<String>,
    #[serde(default = "default_depth")]
    depth: u8,
    nodes: Option<u64>,
    movetime_ms: Option<u64>,
    black_time_ms: Option<u64>,
    white_time_ms: Option<u64>,
    byoyomi_ms: Option<u64>,
    black_increment_ms: Option<u64>,
    white_increment_ms: Option<u64>,
    #[serde(default)]
    control: bool,
}
fn default_depth() -> u8 {
    5
}

fn engine(model: &Arc<Phase10VEvaluator>) -> Result<SearchEngine, String> {
    SearchEngine::with_phase10v(
        SearchConfig {
            transposition_entries: SearchEngine::transposition_entries_for_megabytes(2),
            quiescence_depth: 4,
            ..SearchConfig::default()
        },
        Arc::clone(model),
        &model.identity().artifact_sha256,
    )
}

fn sample(model: &Arc<Phase10VEvaluator>, games: u32, count: u32, seed: u64) -> Result<(), String> {
    if games > 32 || count > 12 {
        return Err("small generator is limited to 32 games x 12 samples".into());
    }
    for game_id in 0..games {
        let mut game = Game::startpos();
        let mut random = RandomMoveSelector::new(seed + u64::from(game_id));
        let mut moves = Vec::new();
        for ply in 0..(24 + 7 * count) {
            if game.end().is_some() {
                break;
            }
            if ply >= 24 && (ply - 24) % 7 == 0 {
                println!(
                    "{}",
                    json!({"game_id":game_id,"seed":seed + u64::from(game_id),
                    "moves":moves,"sfen":to_sfen(game.position()),"source":"self-generated-legal-game/v1"})
                );
            }
            let movement = if ply % 3 == 0 {
                random.select(game.position())
            } else {
                let mut search = engine(model)?;
                search.set_pure_history(&Position::startpos(), game.moves())?;
                let result = search.search(
                    game.position(),
                    SearchLimits {
                        max_depth: 1,
                        max_nodes: Some(128),
                        movetime: None,
                    },
                    &CancellationToken::new(),
                );
                if result.stats.learned_eval_calls == 0 || result.stats.handcrafted_eval_calls != 0
                {
                    return Err(
                        "generated nonterminal trajectory lacks pure learned inference".into(),
                    );
                }
                result.best_move
            };
            let Some(movement) = movement else {
                break;
            };
            moves.push(to_usi_move(movement));
            game.play(movement).map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

fn info_value(position: &Position, info: &SearchInfo, previous: Option<&SearchInfo>) -> Value {
    json!({"depth":info.depth,"score":info.score,"best_move":info.best_move.map(to_usi_move),
        "nodes":info.nodes,"elapsed_ms":info.elapsed.as_secs_f64()*1000.0,
        "roots":info.root_moves.iter().map(|r|json!({
            "move":to_usi_move(r.movement),"score":r.score,"nodes":r.nodes,
            "features":computation_features(position,info,previous,r)
        })).collect::<Vec<_>>()})
}

fn probe(
    model: &Arc<Phase10VEvaluator>,
    controller: Option<&Arc<ComputationModel>>,
    request: &Request,
) -> Result<Value, String> {
    let initial = request
        .sfen
        .as_deref()
        .map_or_else(|| Ok(Position::startpos()), parse_sfen)
        .map_err(|e| e.to_string())?;
    let mut game = Game::new(initial.clone());
    for movement in &request.moves {
        game.play(parse_usi_move(movement).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    }
    let mut search = engine(model)?;
    search.set_pure_history(&initial, game.moves())?;
    if request.control {
        search.set_computation_model(Arc::clone(
            controller.ok_or("controller requested but not loaded")?,
        ))?;
    }
    let mut iterations = Vec::new();
    let mut previous = None;
    let mut callback = |info: &SearchInfo| {
        iterations.push(info_value(game.position(), info, previous.as_ref()));
        previous = Some(info.clone());
    };
    let mut time_hard_limit_ms = None;
    let result = if request.black_time_ms.is_some() || request.white_time_ms.is_some() {
        let plan = TimeManager::default().plan_for_position(
            game.position(),
            TimeControl {
                black_time_ms: request.black_time_ms,
                white_time_ms: request.white_time_ms,
                byoyomi_ms: request.byoyomi_ms,
                black_increment_ms: request.black_increment_ms,
                white_increment_ms: request.white_increment_ms,
                depth: Some(request.depth),
                nodes: request.nodes,
                casual: false,
                safety_margin_ms: 100,
                ..TimeControl::casual()
            },
            64,
        )?;
        time_hard_limit_ms = plan.hard_limit.map(|limit| limit.as_secs_f64() * 1000.0);
        search.search_managed_with_callback(
            game.position(),
            plan,
            &CancellationToken::new(),
            &mut callback,
        )
    } else {
        search.search_with_callback(
            game.position(),
            SearchLimits {
                max_depth: request.depth,
                max_nodes: request.nodes,
                movetime: request.movetime_ms.map(Duration::from_millis),
            },
            &CancellationToken::new(),
            &mut callback,
        )
    };
    let proof = search.runtime_proof(result.stats, &model.identity().artifact_sha256);
    Ok(
        json!({"schema":"open_shogiai_core_probe/v1","sfen":to_sfen(game.position()),
        "perspective":format!("{:?}",game.position().side_to_move()),
        "leaf_sha256":model.identity().artifact_sha256,"best_move":result.best_move.map(to_usi_move),
        "score":result.score,"depth":result.depth,"seldepth":result.seldepth,"nodes":result.nodes,
        "outcome":result.outcome,"time_hard_limit_ms":time_hard_limit_ms,
        "elapsed_ms":result.elapsed.as_secs_f64()*1000.0,"termination":format!("{:?}",result.termination),
        "time_target_ms":search.managed_target_ms(),
        "legal":result.best_move.is_some_and(|m|game.position().legal_moves().contains(&m)),
        "proof":proof,"iterations":iterations,"compute_control":search.compute_control_summary(),
        "game_end":format!("{:?}",game.end()),"scores_are_alpha_beta_observations":true}),
    )
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() < 3 {
        return Err(
            "usage: core_probe LEAF SHA (generate GAMES SAMPLES SEED | probe [CONTROLLER SHA])"
                .into(),
        );
    }
    let model = Arc::new(Phase10VEvaluator::load_file(&args[0])?);
    if model.identity().artifact_sha256 != args[1] {
        return Err("leaf SHA-256 mismatch".into());
    }
    if args[2] == "generate" {
        if args.len() != 6 {
            return Err("generate requires games, samples, seed".into());
        }
        sample(&model, args[3].parse()?, args[4].parse()?, args[5].parse()?)?;
    } else if args[2] == "probe" {
        let controller = if args.len() == 5 {
            Some(Arc::new(ComputationModel::from_bytes(
                &std::fs::read(&args[3])?,
                &args[4],
                &args[1],
            )?))
        } else if args.len() == 3 {
            None
        } else {
            return Err("probe controller requires path and SHA".into());
        };
        for line in io::stdin().lock().lines() {
            let request = serde_json::from_str::<Request>(&line?)?;
            println!("{}", probe(&model, controller.as_ref(), &request)?);
        }
    } else {
        return Err("unknown probe command".into());
    }
    Ok(())
}
