use std::{fs::File, io::Read};

use open_shogi_core::{
    Game, Position, Side, parse_csa_game, parse_sfen, perft, perft_divide, to_usi_move,
};

use crate::args::{next_value, parse_next};

const DEFAULT_RANDOM_GAMES: u64 = 100;
const DEFAULT_MAX_PLIES: u32 = 512;
const DEFAULT_SEED: u64 = 0x4F53_4149_5F50_4831;
const MAX_CSA_BYTES: usize = 1_048_576;
const MAX_CLI_PERFT_DEPTH: u8 = 5;
const MAX_RANDOM_GAMES: u64 = 10_000;
const MAX_RANDOM_PLIES: u32 = 10_000;
const MAX_RANDOM_GAME_PLIES: u64 = 1_000_000;

pub fn run_perft(arguments: &[String]) -> Result<(), String> {
    let mut depth = None;
    let mut sfen = None;
    let mut divide = false;
    let mut index = 0;
    while index < arguments.len() {
        match arguments[index].as_str() {
            "--depth" => depth = Some(parse_next::<u8>(arguments, &mut index, "--depth")?),
            "--sfen" => sfen = Some(next_value(arguments, &mut index, "--sfen")?.to_owned()),
            "--divide" => divide = true,
            argument => return Err(format!("unknown perft argument: {argument}")),
        }
        index += 1;
    }
    let depth = depth.ok_or_else(|| "perft requires --depth <0..5>".to_owned())?;
    if depth > MAX_CLI_PERFT_DEPTH {
        return Err(format!(
            "--depth must not exceed the defensive CLI limit of {MAX_CLI_PERFT_DEPTH}"
        ));
    }
    let position = match sfen {
        Some(value) => parse_sfen(&value).map_err(|error| format!("invalid SFEN: {error}"))?,
        None => Position::startpos(),
    };
    if divide {
        for (movement, nodes) in
            perft_divide(&position, depth).map_err(|error| format!("perft failed: {error}"))?
        {
            println!("{} {nodes}", to_usi_move(movement));
        }
    }
    let result = perft(&position, depth).map_err(|error| format!("perft failed: {error}"))?;
    println!(
        "depth {depth} nodes {} captures {} promotions {} drops {} checks {} checkmates {}",
        result.nodes,
        result.captures,
        result.promotions,
        result.drops,
        result.checks,
        result.checkmates
    );
    Ok(())
}

pub fn run_random_games(arguments: &[String]) -> Result<(), String> {
    let mut games = DEFAULT_RANDOM_GAMES;
    let mut max_plies = DEFAULT_MAX_PLIES;
    let mut seed = DEFAULT_SEED;
    let mut index = 0;
    while index < arguments.len() {
        match arguments[index].as_str() {
            "--games" => games = parse_next(arguments, &mut index, "--games")?,
            "--max-plies" => max_plies = parse_next(arguments, &mut index, "--max-plies")?,
            "--seed" => seed = parse_next(arguments, &mut index, "--seed")?,
            argument => return Err(format!("unknown random-games argument: {argument}")),
        }
        index += 1;
    }
    validate_random_game_bounds(games, max_plies)?;

    let mut random = seed;
    let mut completed = 0_u64;
    let mut capped = 0_u64;
    let mut total_plies = 0_u64;
    for _ in 0..games {
        let mut game = Game::startpos();
        for ply in 0..max_plies {
            let legal_moves = game.position().legal_moves();
            if legal_moves.is_empty() || game.end().is_some() {
                completed = completed.saturating_add(1);
                break;
            }
            let random_value = splitmix64(&mut random);
            let move_index = usize::try_from(
                random_value % u64::try_from(legal_moves.len()).expect("move count fits u64"),
            )
            .expect("reduced move index fits usize");
            game.play(legal_moves[move_index])
                .map_err(|error| format!("generated move was rejected: {error}"))?;
            game.position()
                .validate()
                .map_err(|error| format!("position invariant failed: {error}"))?;
            if game.position().zobrist_hash() != game.position().recompute_zobrist() {
                return Err("incremental and recomputed Zobrist hashes differ".to_owned());
            }
            if material_count(game.position()) != 40 {
                return Err("total material count changed during a random game".to_owned());
            }
            total_plies = total_plies.saturating_add(1);
            if ply + 1 == max_plies {
                capped = capped.saturating_add(1);
            }
        }
    }
    println!(
        "random-games seed={seed} games={games} completed={completed} capped={capped} plies={total_plies} illegal_moves=0"
    );
    Ok(())
}

fn validate_random_game_bounds(games: u64, max_plies: u32) -> Result<(), String> {
    if !(1..=MAX_RANDOM_GAMES).contains(&games) {
        return Err(format!("--games must be 1..={MAX_RANDOM_GAMES}"));
    }
    if !(1..=MAX_RANDOM_PLIES).contains(&max_plies) {
        return Err(format!("--max-plies must be 1..={MAX_RANDOM_PLIES}"));
    }
    let work = games
        .checked_mul(u64::from(max_plies))
        .ok_or_else(|| "random-games work estimate overflowed".to_owned())?;
    if work > MAX_RANDOM_GAME_PLIES {
        return Err(format!(
            "random-games games*max-plies must not exceed {MAX_RANDOM_GAME_PLIES}"
        ));
    }
    Ok(())
}

pub fn run_validate_csa(arguments: &[String]) -> Result<(), String> {
    if arguments.len() != 1 {
        return Err("validate-csa requires exactly one file path".to_owned());
    }
    let contents = read_csa_file(&arguments[0])?;
    let game = parse_csa_game(&contents).map_err(|error| format!("invalid CSA: {error}"))?;
    println!(
        "valid CSA version={} moves={} result={:?} result_validation={:?}",
        game.version,
        game.moves.len(),
        game.special_move,
        game.result_validation
    );
    Ok(())
}

fn read_csa_file(path: &str) -> Result<String, String> {
    let file = File::open(path).map_err(|error| format!("cannot open {path}: {error}"))?;
    let mut bytes = Vec::new();
    file.take(u64::try_from(MAX_CSA_BYTES + 1).expect("CSA byte limit fits in u64"))
        .read_to_end(&mut bytes)
        .map_err(|error| format!("cannot read {path}: {error}"))?;
    if bytes.len() > MAX_CSA_BYTES {
        return Err(format!(
            "CSA file exceeds the {MAX_CSA_BYTES}-byte defensive limit"
        ));
    }
    decode_csa_bytes(&bytes).map(str::to_owned)
}

fn decode_csa_bytes(bytes: &[u8]) -> Result<&str, String> {
    let raw_first_line = bytes
        .split(|byte| *byte == b'\n')
        .next()
        .unwrap_or_default();
    let first_line = raw_first_line.strip_suffix(b"\r").unwrap_or(raw_first_line);
    let declared_utf8 = first_line == b"'CSA encoding=UTF-8";
    if first_line.starts_with(b"'CSA encoding=") && !declared_utf8 {
        return Err(
            "unsupported CSA encoding declaration; convert the record to UTF-8 first".to_owned(),
        );
    }
    let text = std::str::from_utf8(bytes)
        .map_err(|_| "CSA is not valid UTF-8; convert SHIFT_JIS records before validation")?;
    if !declared_utf8 && !text.is_ascii() {
        return Err(
            "non-ASCII CSA requires an explicit `'CSA encoding=UTF-8` declaration".to_owned(),
        );
    }
    Ok(text)
}

fn material_count(position: &Position) -> u16 {
    let board_count = position
        .board()
        .iter()
        .filter(|piece| piece.is_some())
        .count();
    u16::try_from(board_count).expect("board count fits u16")
        + position.hand(Side::Black).total()
        + position.hand(Side::White).total()
}

pub fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use super::{
        MAX_RANDOM_GAMES, MAX_RANDOM_PLIES, decode_csa_bytes, run_random_games,
        validate_random_game_bounds,
    };

    #[test]
    fn zero_random_games_is_rejected() {
        assert!(run_random_games(&["--games".to_owned(), "0".to_owned()]).is_err());
    }

    #[test]
    fn random_game_work_is_explicitly_bounded() {
        assert!(validate_random_game_bounds(1_000, 1_000).is_ok());
        assert!(validate_random_game_bounds(MAX_RANDOM_GAMES + 1, 1).is_err());
        assert!(validate_random_game_bounds(1, MAX_RANDOM_PLIES + 1).is_err());
        assert!(validate_random_game_bounds(u64::MAX, u32::MAX).is_err());
        assert!(validate_random_game_bounds(1_001, 1_000).is_err());
    }

    #[test]
    fn csa_encoding_boundary_is_explicit() {
        assert!(decode_csa_bytes(b"V3.0\nPI\n+\n").is_ok());
        assert!(decode_csa_bytes("'棋譜\nV3.0\nPI\n+\n".as_bytes()).is_err());
        assert!(decode_csa_bytes(b"'CSA encoding=SHIFT_JIS\nV3.0\nPI\n+\n").is_err());
        assert!(decode_csa_bytes("'CSA encoding=UTF-8\n'棋譜\nV3.0\nPI\n+\n".as_bytes()).is_ok());
    }
}
