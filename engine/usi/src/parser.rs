//! Bounded, side-effect-free USI command parsing.

use std::{error::Error, fmt};

const MAX_COMMAND_BYTES: usize = 4_096;
const MAX_TOKENS: usize = 512;
const MAX_OPTION_NAME_BYTES: usize = 128;
const MAX_OPTION_VALUE_BYTES: usize = 512;
const MAX_POSITION_MOVES: usize = 500;
pub(crate) const MAX_GO_CLOCK_MS: u64 = 7 * 24 * 60 * 60 * 1_000;
pub(crate) const MAX_GO_MOVETIME_MS: u64 = 60 * 60 * 1_000;
pub(crate) const MAX_GO_BYOYOMI_MS: u64 = 60 * 60 * 1_000;
pub(crate) const MAX_GO_INCREMENT_MS: u64 = 60 * 60 * 1_000;
pub(crate) const MAX_GO_NODES: u64 = 1_000_000_000;
pub(crate) const MAX_GO_DEPTH: u8 = 64;

/// Search controls accepted by the USI `go` command.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct GoParameters {
    pub black_time_ms: Option<u64>,
    pub white_time_ms: Option<u64>,
    pub byoyomi_ms: Option<u64>,
    pub black_increment_ms: Option<u64>,
    pub white_increment_ms: Option<u64>,
    pub movetime_ms: Option<u64>,
    pub nodes: Option<u64>,
    pub depth: Option<u8>,
    pub infinite: bool,
}

/// A parsed USI command with no engine-side effects.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum UsiCommand {
    Usi,
    IsReady,
    SetOption { name: String, value: Option<String> },
    NewGame,
    PositionStartpos { moves: Vec<String> },
    PositionSfen { sfen: String, moves: Vec<String> },
    Go(GoParameters),
    Stop,
    Quit,
    GameOver { result: String },
}

/// A bounded parser error suitable for a USI `info string` response.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UsiParseError(String);

impl UsiParseError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for UsiParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for UsiParseError {}

/// Parses one complete USI input line.
///
/// # Errors
///
/// Returns a bounded syntax error for malformed, unknown, or oversized commands.
pub fn parse_command(input: &str) -> Result<UsiCommand, UsiParseError> {
    if input.len() > MAX_COMMAND_BYTES {
        return Err(UsiParseError::new("command exceeds 4096-byte limit"));
    }
    if input
        .bytes()
        .any(|byte| byte == b'\0' || byte == b'\r' || byte == b'\n')
    {
        return Err(UsiParseError::new(
            "command contains a forbidden control character",
        ));
    }
    let tokens = input.split_ascii_whitespace().collect::<Vec<_>>();
    if tokens.len() > MAX_TOKENS {
        return Err(UsiParseError::new("command has too many tokens"));
    }
    let Some(command) = tokens.first().copied() else {
        return Err(UsiParseError::new("empty command"));
    };
    match command {
        "usi" if tokens.len() == 1 => Ok(UsiCommand::Usi),
        "isready" if tokens.len() == 1 => Ok(UsiCommand::IsReady),
        "usinewgame" if tokens.len() == 1 => Ok(UsiCommand::NewGame),
        "stop" if tokens.len() == 1 => Ok(UsiCommand::Stop),
        "quit" if tokens.len() == 1 => Ok(UsiCommand::Quit),
        "setoption" => parse_setoption(&tokens),
        "position" => parse_position(&tokens),
        "go" => parse_go(&tokens),
        "gameover" => parse_gameover(&tokens),
        _ => Err(UsiParseError::new("unknown or malformed USI command")),
    }
}

fn parse_setoption(tokens: &[&str]) -> Result<UsiCommand, UsiParseError> {
    if tokens.get(1) != Some(&"name") {
        return Err(UsiParseError::new("setoption requires `name`"));
    }
    let value_index = tokens[2..]
        .iter()
        .position(|token| *token == "value")
        .map(|index| index + 2);
    let name_end = value_index.unwrap_or(tokens.len());
    if name_end == 2 {
        return Err(UsiParseError::new("setoption name is empty"));
    }
    let name = tokens[2..name_end].join(" ");
    if name.len() > MAX_OPTION_NAME_BYTES {
        return Err(UsiParseError::new("setoption name is too long"));
    }
    let value = if let Some(index) = value_index {
        let value = tokens[index + 1..].join(" ");
        if value.len() > MAX_OPTION_VALUE_BYTES {
            return Err(UsiParseError::new("setoption value is too long"));
        }
        Some(value)
    } else {
        None
    };
    Ok(UsiCommand::SetOption { name, value })
}

fn parse_position(tokens: &[&str]) -> Result<UsiCommand, UsiParseError> {
    match tokens.get(1).copied() {
        Some("startpos") => {
            let moves = parse_position_moves(tokens, 2)?;
            Ok(UsiCommand::PositionStartpos { moves })
        }
        Some("sfen") => {
            if tokens.len() < 6 {
                return Err(UsiParseError::new(
                    "position sfen requires four SFEN fields",
                ));
            }
            let sfen = tokens[2..6].join(" ");
            let moves = parse_position_moves(tokens, 6)?;
            Ok(UsiCommand::PositionSfen { sfen, moves })
        }
        _ => Err(UsiParseError::new("position requires `startpos` or `sfen`")),
    }
}

fn parse_position_moves(tokens: &[&str], index: usize) -> Result<Vec<String>, UsiParseError> {
    if index == tokens.len() {
        return Ok(Vec::new());
    }
    if tokens.get(index) != Some(&"moves") {
        return Err(UsiParseError::new("unexpected token after position"));
    }
    let moves = &tokens[index + 1..];
    if moves.len() > MAX_POSITION_MOVES {
        return Err(UsiParseError::new("position has more than 500 moves"));
    }
    Ok(moves
        .iter()
        .map(|movement| (*movement).to_owned())
        .collect())
}

fn parse_go(tokens: &[&str]) -> Result<UsiCommand, UsiParseError> {
    let mut parameters = GoParameters::default();
    let mut index = 1;
    while index < tokens.len() {
        let key = tokens[index];
        match key {
            "infinite" => {
                if parameters.infinite {
                    return Err(UsiParseError::new("duplicate go infinite"));
                }
                parameters.infinite = true;
                index += 1;
            }
            "ponder" => {
                return Err(UsiParseError::new("ponder is not supported"));
            }
            "btime" => set_u64(
                &mut parameters.black_time_ms,
                tokens,
                &mut index,
                "btime",
                true,
                MAX_GO_CLOCK_MS,
            )?,
            "wtime" => set_u64(
                &mut parameters.white_time_ms,
                tokens,
                &mut index,
                "wtime",
                true,
                MAX_GO_CLOCK_MS,
            )?,
            "byoyomi" => set_u64(
                &mut parameters.byoyomi_ms,
                tokens,
                &mut index,
                "byoyomi",
                true,
                MAX_GO_BYOYOMI_MS,
            )?,
            "binc" => set_u64(
                &mut parameters.black_increment_ms,
                tokens,
                &mut index,
                "binc",
                true,
                MAX_GO_INCREMENT_MS,
            )?,
            "winc" => set_u64(
                &mut parameters.white_increment_ms,
                tokens,
                &mut index,
                "winc",
                true,
                MAX_GO_INCREMENT_MS,
            )?,
            "movetime" => set_u64(
                &mut parameters.movetime_ms,
                tokens,
                &mut index,
                "movetime",
                false,
                MAX_GO_MOVETIME_MS,
            )?,
            "nodes" => set_u64(
                &mut parameters.nodes,
                tokens,
                &mut index,
                "nodes",
                false,
                MAX_GO_NODES,
            )?,
            "depth" => {
                if parameters.depth.is_some() {
                    return Err(UsiParseError::new("duplicate go depth"));
                }
                let raw = next_number(tokens, &mut index, "depth")?;
                let depth = u8::try_from(raw)
                    .map_err(|_| UsiParseError::new("go depth exceeds the supported limit"))?;
                if !(1..=MAX_GO_DEPTH).contains(&depth) {
                    return Err(UsiParseError::new("go depth must be 1..=64"));
                }
                parameters.depth = Some(depth);
            }
            _ => return Err(UsiParseError::new(format!("unsupported go token `{key}`"))),
        }
    }
    if parameters.infinite
        && (parameters.movetime_ms.is_some()
            || parameters.nodes.is_some()
            || parameters.depth.is_some())
    {
        return Err(UsiParseError::new(
            "go infinite cannot be combined with a fixed search limit",
        ));
    }
    Ok(UsiCommand::Go(parameters))
}

fn set_u64(
    slot: &mut Option<u64>,
    tokens: &[&str],
    index: &mut usize,
    name: &str,
    allow_zero: bool,
    maximum: u64,
) -> Result<(), UsiParseError> {
    if slot.is_some() {
        return Err(UsiParseError::new(format!("duplicate go {name}")));
    }
    let value = next_number(tokens, index, name)?;
    if !allow_zero && value == 0 {
        return Err(UsiParseError::new(format!("go {name} must be positive")));
    }
    if value > maximum {
        return Err(UsiParseError::new(format!(
            "go {name} exceeds the defensive limit of {maximum}"
        )));
    }
    *slot = Some(value);
    Ok(())
}

fn next_number(tokens: &[&str], index: &mut usize, name: &str) -> Result<u64, UsiParseError> {
    *index += 1;
    let value = tokens
        .get(*index)
        .ok_or_else(|| UsiParseError::new(format!("go {name} requires a value")))?
        .parse::<u64>()
        .map_err(|_| UsiParseError::new(format!("go {name} has an invalid value")))?;
    *index += 1;
    Ok(value)
}

fn parse_gameover(tokens: &[&str]) -> Result<UsiCommand, UsiParseError> {
    if tokens.len() != 2 || !matches!(tokens[1], "win" | "lose" | "draw") {
        return Err(UsiParseError::new(
            "gameover requires `win`, `lose`, or `draw`",
        ));
    }
    Ok(UsiCommand::GameOver {
        result: tokens[1].to_owned(),
    })
}

#[cfg(test)]
mod tests {
    use super::{
        GoParameters, MAX_GO_BYOYOMI_MS, MAX_GO_CLOCK_MS, MAX_GO_DEPTH, MAX_GO_INCREMENT_MS,
        MAX_GO_MOVETIME_MS, MAX_GO_NODES, UsiCommand, parse_command,
    };

    #[test]
    fn parses_supported_position_shapes() {
        assert_eq!(
            parse_command("position startpos moves 7g7f 3c3d").unwrap(),
            UsiCommand::PositionStartpos {
                moves: vec!["7g7f".to_owned(), "3c3d".to_owned()]
            }
        );
        assert!(matches!(
            parse_command("position sfen lnsgkgsnl/9/9/9/9/9/9/9/LNSGKGSNL b - 1"),
            Ok(UsiCommand::PositionSfen { .. })
        ));
    }

    #[test]
    fn parses_all_practical_go_limits() {
        assert_eq!(
            parse_command("go btime 1000 wtime 2000 byoyomi 100 binc 5 winc 6 depth 7 nodes 8")
                .unwrap(),
            UsiCommand::Go(GoParameters {
                black_time_ms: Some(1000),
                white_time_ms: Some(2000),
                byoyomi_ms: Some(100),
                black_increment_ms: Some(5),
                white_increment_ms: Some(6),
                movetime_ms: None,
                nodes: Some(8),
                depth: Some(7),
                infinite: false,
            })
        );
    }

    #[test]
    fn malformed_and_unbounded_inputs_are_rejected() {
        assert!(parse_command("go nodes 0").is_err());
        assert!(parse_command("go nodes 1 nodes 2").is_err());
        assert!(parse_command("go infinite depth 2").is_err());
        assert!(parse_command(&"x".repeat(4097)).is_err());
        assert!(parse_command("position startpos nope 7g7f").is_err());
        assert!(parse_command("gameover maybe").is_err());
    }

    #[test]
    fn go_resource_limits_are_closed_at_the_documented_boundaries() {
        for (name, maximum) in [
            ("btime", MAX_GO_CLOCK_MS),
            ("wtime", MAX_GO_CLOCK_MS),
            ("byoyomi", MAX_GO_BYOYOMI_MS),
            ("binc", MAX_GO_INCREMENT_MS),
            ("winc", MAX_GO_INCREMENT_MS),
            ("movetime", MAX_GO_MOVETIME_MS),
            ("nodes", MAX_GO_NODES),
        ] {
            assert!(parse_command(&format!("go {name} {maximum}")).is_ok());
            assert!(parse_command(&format!("go {name} {}", maximum + 1)).is_err());
            assert!(parse_command(&format!("go {name} {}", u64::MAX)).is_err());
        }
        assert!(parse_command(&format!("go depth {MAX_GO_DEPTH}")).is_ok());
        assert!(parse_command(&format!("go depth {}", MAX_GO_DEPTH + 1)).is_err());
        assert!(parse_command("go depth 255").is_err());
    }

    #[test]
    fn setoption_preserves_spaces_without_controls() {
        assert_eq!(
            parse_command("setoption name Evaluation Material value false").unwrap(),
            UsiCommand::SetOption {
                name: "Evaluation Material".to_owned(),
                value: Some("false".to_owned())
            }
        );
        assert!(parse_command("setoption name Hash\0value 1").is_err());
    }
}
