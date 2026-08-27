//! Emit authoritative legal moves and bounded history facts for a replayed game.
//!
//! This example is intentionally line-oriented so the Phase 10R preparation command can
//! stream one complete game at a time without materialising the approved population in Python.

use std::io::{self, BufRead, Write};

use open_shogi_core::{Position, Side, parse_sfen, parse_usi_move, to_sfen, to_usi_move};
use serde::{Deserialize, Serialize};

const MAX_INPUT_LINE_BYTES: usize = 4 * 1024 * 1024;
const MAX_MOVES: usize = 10_000;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct Request {
    initial_sfen: String,
    usi_moves: Vec<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct HistoryOutput {
    available: bool,
    repetition_count: u8,
    continuous_check_by_us: bool,
    continuous_check_by_them: bool,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct PositionOutput {
    event: &'static str,
    position_index: usize,
    sfen: String,
    legal_moves: Vec<String>,
    history: HistoryOutput,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct DoneOutput {
    event: &'static str,
    position_count: usize,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let stdin = io::stdin();
    let mut reader = stdin.lock();
    let stdout = io::stdout();
    let mut writer = stdout.lock();
    let mut line = String::new();
    loop {
        line.clear();
        let bytes = reader
            .read_line(&mut line)
            .map_err(|error| format!("cannot read request: {error}"))?;
        if bytes == 0 {
            break;
        }
        if bytes > MAX_INPUT_LINE_BYTES || !line.ends_with('\n') {
            return Err(format!(
                "request line exceeds {MAX_INPUT_LINE_BYTES} bytes or is not newline terminated"
            ));
        }
        let request: Request = serde_json::from_str(line.trim_end_matches(['\r', '\n']))
            .map_err(|error| format!("invalid request JSON: {error}"))?;
        if request.usi_moves.len() > MAX_MOVES {
            return Err(format!("request contains more than {MAX_MOVES} moves"));
        }
        emit_game(&mut writer, &request)?;
    }
    Ok(())
}

fn emit_game(writer: &mut impl Write, request: &Request) -> Result<(), String> {
    let mut position = parse_sfen(&request.initial_sfen)
        .map_err(|error| format!("invalid initial SFEN: {error}"))?;
    let mut positions = vec![position.clone()];
    let mut gave_check = Vec::with_capacity(request.usi_moves.len());
    for (index, notation) in request.usi_moves.iter().enumerate() {
        emit_position(writer, index, &position, &positions, &gave_check)?;
        let movement = parse_usi_move(notation)
            .map_err(|error| format!("invalid USI move at index {index}: {error}"))?;
        position
            .make_move(movement)
            .map_err(|error| format!("illegal USI move at index {index}: {error}"))?;
        gave_check.push(position.is_in_check(position.side_to_move()));
        positions.push(position.clone());
    }
    emit_position(
        writer,
        request.usi_moves.len(),
        &position,
        &positions,
        &gave_check,
    )?;
    write_json_line(
        writer,
        &DoneOutput {
            event: "done",
            position_count: positions.len(),
        },
    )
}

fn emit_position(
    writer: &mut impl Write,
    index: usize,
    position: &Position,
    positions: &[Position],
    gave_check: &[bool],
) -> Result<(), String> {
    let legal_moves = position
        .legal_moves()
        .into_iter()
        .map(to_usi_move)
        .collect();
    let history = history_facts(index, position, positions, gave_check);
    write_json_line(
        writer,
        &PositionOutput {
            event: "position",
            position_index: index,
            sfen: to_sfen(position),
            legal_moves,
            history,
        },
    )
}

fn history_facts(
    index: usize,
    position: &Position,
    positions: &[Position],
    gave_check: &[bool],
) -> HistoryOutput {
    let occurrences = positions
        .iter()
        .enumerate()
        .filter_map(|(position_index, previous)| {
            previous.same_state(position).then_some(position_index)
        })
        .collect::<Vec<_>>();
    let repetition_count = u8::try_from(occurrences.len().min(4)).expect("bounded count");
    let (continuous_check_by_us, continuous_check_by_them) = if occurrences.len() >= 4 {
        let start = occurrences[occurrences.len() - 4];
        let finish = index;
        let black = all_checks_for_side(start, finish, Side::Black, positions, gave_check);
        let white = all_checks_for_side(start, finish, Side::White, positions, gave_check);
        (black, white)
    } else {
        (false, false)
    };
    HistoryOutput {
        available: true,
        repetition_count,
        continuous_check_by_us,
        continuous_check_by_them,
    }
}

fn all_checks_for_side(
    start: usize,
    finish: usize,
    side: Side,
    positions: &[Position],
    gave_check: &[bool],
) -> bool {
    let mut has_move = false;
    let all_checks = (start..finish)
        .filter(|move_index| positions[*move_index].side_to_move() == side)
        .all(|move_index| {
            has_move = true;
            gave_check[move_index]
        });
    has_move && all_checks
}

fn write_json_line(writer: &mut impl Write, value: &impl Serialize) -> Result<(), String> {
    serde_json::to_writer(&mut *writer, value).map_err(|error| error.to_string())?;
    writer.write_all(b"\n").map_err(|error| error.to_string())?;
    writer.flush().map_err(|error| error.to_string())
}
