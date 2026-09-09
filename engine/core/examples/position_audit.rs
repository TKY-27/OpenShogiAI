//! Stateful legal replay and real-model accumulator audit for offline data generation.
use open_shogi_core::{
    Game, Phase10VEvaluator, Position, parse_sfen, parse_usi_move, to_sfen, to_usi_move,
};
use serde::Deserialize;
use serde_json::json;
use std::{
    env,
    io::{self, BufRead, Write},
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    reset: Option<String>,
    movement: Option<String>,
    #[serde(default)]
    successors: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    if args.len() != 2 {
        return Err("usage: position_audit MODEL SHA256".into());
    }
    let model = Phase10VEvaluator::load_file(&args[0])?;
    if model.identity().artifact_sha256 != args[1] {
        return Err("model identity mismatch".into());
    }
    let mut game = Game::new(Position::startpos());
    let mut accumulated = model.accumulator(game.position())?;
    for line in io::stdin().lock().lines() {
        let request: Request = serde_json::from_str(&line?)?;
        if request.reset.is_some() && request.movement.is_some() {
            return Err("reset and move conflict".into());
        }
        if let Some(sfen) = request.reset {
            game = Game::new(parse_sfen(&sfen)?);
            accumulated = model.accumulator(game.position())?;
        }
        if let Some(usi) = request.movement {
            let movement = parse_usi_move(&usi)?;
            game.play(movement)?;
            model.update_accumulator(&mut accumulated, movement, game.position())?;
        }
        let position = game.position();
        let parent = model.accumulator(position)?;
        if accumulated != parent {
            return Err("trajectory incremental/full mismatch".into());
        }
        let mut successors = Vec::new();
        if request.successors && game.end().is_none() {
            for movement in position.legal_moves() {
                let mut child_game = game.clone();
                child_game.play(movement)?;
                let child = child_game.position();
                let mut incremental = parent.clone();
                model.update_accumulator(&mut incremental, movement, child)?;
                let full = model.accumulator(child)?;
                // Accumulator equality includes both exact vectors, board and model identity.
                if incremental != full {
                    return Err("incremental/full accumulator mismatch".into());
                }
                let predicted = model.infer_accumulator(&incremental)?;
                let refreshed = model.infer_accumulator(&full)?;
                // This diagnostic requires bit equality, including signed zero.
                if predicted.cp != refreshed.cp
                    || predicted
                        .wdl_logits
                        .iter()
                        .zip(refreshed.wdl_logits.iter())
                        .any(|(left, right)| left.to_bits() != right.to_bits())
                {
                    return Err("incremental/full output mismatch".into());
                }
                successors.push(json!({"move":to_usi_move(movement),"sfen":to_sfen(child),
                    "child_cp":predicted.cp,"child_wdl_logits":predicted.wdl_logits,
                    "terminal":format!("{:?}",child_game.end())}));
            }
        }
        println!(
            "{}",
            json!({"sfen":to_sfen(position),"model_sha256":model.identity().artifact_sha256,
            "cp":model.infer_accumulator(&parent)?.cp,"terminal":format!("{:?}",game.end()),
            "audited_successors":successors.len(),"successors":successors,"incremental_full_equal":true})
        );
        io::stdout().flush()?;
    }
    Ok(())
}
