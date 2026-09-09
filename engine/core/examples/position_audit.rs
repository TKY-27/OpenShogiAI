//! Stateful legal replay and real-model accumulator audit for offline data generation.
use open_shogi_core::{
    EnteringKingDeclaration, EnteringKingRule, Game, GameEnd, Phase10VEvaluator, Position, Side,
    parse_sfen, parse_usi_move, to_sfen, to_usi_move,
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

/// The pinned offline teacher implements CSA's asymmetric 28/27-point declaration.
/// This invokes the existing rules API; it never alters a nonterminal model score.
fn teacher_declaration(game: &Game) -> serde_json::Value {
    let side = game.position().side_to_move();
    let required_points = if side == Side::Black { 28 } else { 27 };
    let result = game.entering_king_declaration(
        side,
        EnteringKingRule::Custom {
            minimum_camp_pieces: 10,
            win_points: required_points,
            no_contest_points: required_points,
        },
    );
    let (outcome, points) = match result {
        EnteringKingDeclaration::Win { points, .. } => ("win", Some(points)),
        EnteringKingDeclaration::NoContest { points, .. } => ("no_contest", Some(points)),
        EnteringKingDeclaration::InvalidLoss { .. } => ("invalid_loss", None),
        EnteringKingDeclaration::Unavailable { .. } => ("unavailable", None),
        EnteringKingDeclaration::Disabled => ("disabled", None),
    };
    json!({"rule":"csa_28_27", "side":if side == Side::Black {"black"} else {"white"},
        "minimum_camp_pieces":10, "required_points":required_points,
        "result":outcome, "points":points})
}

fn teacher_resign_eligible(game: &Game) -> bool {
    matches!(
        game.end(),
        Some(GameEnd::Checkmate { .. } | GameEnd::NoLegalMoves { .. })
    )
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
                    "terminal":format!("{:?}",child_game.end()),
                    "teacher_declaration":teacher_declaration(&child_game),
                    "teacher_resign_eligible":teacher_resign_eligible(&child_game)}));
            }
        }
        println!(
            "{}",
            json!({"sfen":to_sfen(position),"model_sha256":model.identity().artifact_sha256,
            "cp":model.infer_accumulator(&parent)?.cp,"terminal":format!("{:?}",game.end()),
            "teacher_declaration":teacher_declaration(&game),
            "teacher_resign_eligible":teacher_resign_eligible(&game),
            "audited_successors":successors.len(),"successors":successors,"incremental_full_equal":true})
        );
        io::stdout().flush()?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csa_declaration_obeys_both_point_boundaries() {
        for (sfen, valid, threshold) in [
            ("1p7/KRRBBPPPP/NN7/9/9/9/9/9/8k b 2P 1", true, 28),
            ("1p7/KRRBBPPPP/NN7/9/9/9/9/9/8k b P 1", false, 28),
            ("K8/9/9/9/9/9/nn7/krrbbpppp/1P7 w p 2", true, 27),
            ("K8/9/9/9/9/9/nn7/krrbbpppp/1P7 w - 2", false, 27),
        ] {
            let game = Game::new(parse_sfen(sfen).expect("valid declaration fixture"));
            let result = teacher_declaration(&game);
            assert_eq!(result["result"] == "win", valid, "{sfen}");
            assert_eq!(result["required_points"], threshold);
            assert!(!teacher_resign_eligible(&game));
        }
    }

    #[test]
    fn declaration_requires_camp_king_ten_pieces_and_no_check() {
        for sfen in [
            // Only nine camp pieces, despite sufficient points in hand.
            "1p7/KRRBBPPPP/N8/9/9/9/9/9/8k b 3P 1",
            // The king remains outside the opponent's camp.
            "1p7/1RRBBPPPP/NN7/K8/9/9/9/9/8k b 2P 1",
            // The opposing lance checks the declaring king.
            "l8/KRRBBPPPP/NN7/9/9/9/9/9/8k b 2P 1",
        ] {
            let game = Game::new(parse_sfen(sfen).expect("valid negative fixture"));
            assert_ne!(teacher_declaration(&game)["result"], "win", "{sfen}");
        }
        assert_ne!(teacher_declaration(&Game::startpos())["result"], "win");
        assert!(!teacher_resign_eligible(&Game::startpos()));
    }
}
