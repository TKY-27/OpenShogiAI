//! Independent rules-only replay oracle: no search or evaluator is instantiated.
use open_shogi_core::{
    Game, GameEnd, RepetitionOutcome, Side, parse_sfen, parse_usi_move, to_sfen,
};
use serde::Deserialize;
use serde_json::json;
use std::io::{self, Read};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    initial_sfen: String,
    moves: Vec<String>,
    max_plies: usize,
}
fn side(side: Side) -> &'static str {
    if side == Side::Black {
        "black"
    } else {
        "white"
    }
}
fn run() -> Result<(), String> {
    let mut input = String::new();
    io::stdin()
        .take(2_000_001)
        .read_to_string(&mut input)
        .map_err(|e| e.to_string())?;
    if input.len() > 2_000_000 {
        return Err("oversized replay".into());
    }
    let request: Request = serde_json::from_str(&input).map_err(|e| e.to_string())?;
    if request.max_plies == 0
        || request.max_plies > 10000
        || request.moves.len() > request.max_plies
    {
        return Err("invalid ply cap".into());
    }
    let mut game = Game::new(parse_sfen(&request.initial_sfen).map_err(|e| e.to_string())?);
    for notation in &request.moves {
        game.play(parse_usi_move(notation).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    }
    let (termination, result) = match game.end() {
        Some(GameEnd::Checkmate { winner }) => ("checkmate", side(winner)),
        Some(GameEnd::NoLegalMoves { loser }) => ("no_legal_moves", side(loser.opposite())),
        Some(GameEnd::Repetition(RepetitionOutcome::NoContest)) => ("repetition", "draw"),
        Some(GameEnd::Repetition(RepetitionOutcome::PerpetualCheckLoss(loser))) => {
            ("perpetual_check", side(loser.opposite()))
        }
        None if request.moves.len() == request.max_plies => ("max_plies", "excluded"),
        None => ("ongoing", "ongoing"),
        _ => return Err("unsupported adjudication".into()),
    };
    println!(
        "{}",
        json!({"final_sfen":to_sfen(game.position()), "termination":termination,"result":result,"max_plies_reached":request.moves.len()==request.max_plies,"side_to_move":side(game.position().side_to_move())})
    );
    Ok(())
}
fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(2);
    }
}
