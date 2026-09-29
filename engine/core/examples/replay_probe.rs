//! Diagnostic replay probe: replays a USI move list through a validated pure model,
//! then searches the resulting root under several budgets with per-iteration evidence.
//!
//! Development tool; keep arguments bounded. Not part of the USI surface.

use std::process::exit;

use open_shogi_core::{
    CancellationToken, Position, PurePlayingEvaluator, SearchConfig, SearchEngine, SearchLimits,
    TimeControl, TimeManager, parse_sfen, parse_usi_move, to_sfen, to_usi_move,
};
use std::sync::Arc;

fn die(message: impl std::fmt::Display) -> ! {
    eprintln!("{message}");
    exit(1)
}

struct Arguments {
    model: String,
    hash: String,
    format: String,
    moves: String,
    depths: Vec<u8>,
    movetimes: Vec<u64>,
    nodes: Option<u64>,
    workers: usize,
    initial: Option<String>,
}

fn parse_list(value: &str) -> Vec<u64> {
    value
        .split(',')
        .filter_map(|item| item.trim().parse().ok())
        .collect()
}

fn arguments() -> Arguments {
    let mut options = std::env::args().skip(1);
    let mut parsed = Arguments {
        model: String::new(),
        hash: String::new(),
        format: "OSAVAL03".to_owned(),
        moves: String::new(),
        depths: Vec::new(),
        movetimes: Vec::new(),
        nodes: None,
        workers: 1,
        initial: None,
    };
    while let Some(name) = options.next() {
        let mut value = || {
            options
                .next()
                .unwrap_or_else(|| die(format!("missing {name}")))
        };
        match name.as_str() {
            "--model" => parsed.model = value(),
            "--model-sha256" => parsed.hash = value(),
            "--model-format" => parsed.format = value(),
            "--moves" => parsed.moves = value(),
            "--initial" => parsed.initial = Some(value()),
            "--depths" => {
                parsed.depths = parse_list(&value())
                    .into_iter()
                    .map(|depth| u8::try_from(depth).unwrap_or(u8::MAX))
                    .collect();
            }
            "--movetimes" => parsed.movetimes = parse_list(&value()),
            "--nodes" => parsed.nodes = parse_list(&value()).first().copied(),
            "--workers" => {
                parsed.workers =
                    usize::try_from(parse_list(&value()).first().copied().unwrap_or(1))
                        .unwrap_or(1);
            }
            other => die(format!("unknown option {other}")),
        }
    }
    if parsed.model.is_empty() || parsed.hash.is_empty() || parsed.moves.is_empty() {
        die("--model, --model-sha256 and --moves are required");
    }
    parsed
}

fn main() {
    let arguments = arguments();
    let model = Arc::new(
        PurePlayingEvaluator::load_file(&arguments.format, &arguments.model, &arguments.hash)
            .unwrap_or_else(|error| die(error)),
    );
    let initial = match &arguments.initial {
        Some(sfen) => parse_sfen(sfen).unwrap_or_else(|error| die(error)),
        None => Position::startpos(),
    };
    let mut root = initial.clone();
    let mut moves = Vec::new();
    for token in arguments.moves.split_whitespace() {
        let movement = parse_usi_move(token).unwrap_or_else(|error| die(error));
        root.make_move(movement)
            .unwrap_or_else(|error| die(format!("illegal {token}: {error}")));
        moves.push(movement);
    }
    let mut engine = model
        .search_engine(
            SearchConfig {
                transposition_entries: SearchEngine::transposition_entries_for_megabytes(256),
                ..SearchConfig::default()
            },
            &arguments.hash,
        )
        .unwrap_or_else(|error| die(error));
    engine
        .set_pure_history(&initial, &moves)
        .unwrap_or_else(|error| die(error));
    println!("# root sfen {}", to_sfen(&root));
    let inference = model
        .infer(&root, open_shogi_core::Osaval02History::default())
        .unwrap();
    println!("# static cp {}", inference["cp"]);

    for depth in &arguments.depths {
        let result = if arguments.workers > 1 {
            let request = TimeControl {
                depth: Some(*depth),
                nodes: arguments.nodes,
                casual: false,
                ..TimeControl::casual()
            };
            let plan = TimeManager::default()
                .plan(root.side_to_move(), request, 64)
                .unwrap_or_else(|error| die(error));
            engine.search_parallel_managed_with_callback(
                &root,
                plan,
                &CancellationToken::new(),
                arguments.workers,
                |_| {},
            )
        } else {
            engine.search(
                &root,
                SearchLimits {
                    max_depth: *depth,
                    max_nodes: arguments.nodes.or(Some(200_000_000)),
                    movetime: None,
                },
                &CancellationToken::new(),
            )
        };
        report(&format!("depth<{depth}"), &result);
    }
    for movetime in &arguments.movetimes {
        let request = TimeControl {
            movetime_ms: Some(*movetime),
            casual: false,
            ..TimeControl::casual()
        };
        let plan = TimeManager::default()
            .plan(root.side_to_move(), request, 64)
            .unwrap_or_else(|error| die(error));
        let result = engine.search_parallel_managed_with_callback(
            &root,
            plan,
            &CancellationToken::new(),
            arguments.workers,
            |info| {
                println!(
                    "#  it depth {} seldepth {} score {} best {} nodes {} nps {} ms {}",
                    info.depth,
                    info.seldepth,
                    info.score,
                    info.best_move.map_or_else(|| "-".to_owned(), to_usi_move),
                    info.nodes,
                    info.nps,
                    info.elapsed.as_millis(),
                );
            },
        );
        report(&format!("movetime<{movetime}"), &result);
    }
}

fn report(label: &str, result: &open_shogi_core::SearchResult) {
    println!(
        "{} termination {:?} outcome {:?} depth {} seldepth {} score {} best {} nodes {} ms {}",
        label,
        result.termination,
        result.outcome,
        result.depth,
        result.seldepth,
        result.score,
        result.best_move.map_or_else(|| "-".to_owned(), to_usi_move),
        result.nodes,
        result.elapsed.as_millis(),
    );
    println!(
        "  pv {}",
        result
            .pv
            .iter()
            .copied()
            .map(to_usi_move)
            .collect::<Vec<_>>()
            .join(" ")
    );
    let mut stats = result.root_moves.clone();
    stats.sort_by(|left, right| right.score.cmp(&left.score));
    for stat in stats.iter().take(8) {
        println!(
            "  cand {} score {} depth {} nodes {} pv {}",
            to_usi_move(stat.movement),
            stat.score,
            stat.depth,
            stat.nodes,
            stat.pv
                .iter()
                .copied()
                .map(to_usi_move)
                .collect::<Vec<_>>()
                .join(" "),
        );
    }
}
