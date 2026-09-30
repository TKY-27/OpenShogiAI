//! Diagnostic replay probe: replays a USI move list through a validated pure model,
//! then searches the resulting root under several budgets with per-iteration evidence.
//!
//! Development tool; keep arguments bounded. Not part of the USI surface.

use std::process::exit;

use open_shogi_core::{
    CancellationToken, Move, PieceKind, Position, PurePlayingEvaluator, SearchConfig, SearchEngine,
    SearchLimits, TimeControl, TimeManager, parse_sfen, parse_usi_move, to_sfen, to_usi_move,
};
use std::sync::Arc;

fn die(message: impl std::fmt::Display) -> ! {
    eprintln!("{message}");
    exit(1)
}

#[expect(
    clippy::struct_excessive_bools,
    reason = "each diagnostic rung toggle stays independently addressable on the CLI"
)]
struct Arguments {
    model: String,
    hash: String,
    format: String,
    hash_mb: usize,
    moves: String,
    depths: Vec<u8>,
    movetimes: Vec<u64>,
    nodes: Option<u64>,
    workers: usize,
    initial: Option<String>,
    /// Diagnostic rung toggles (PVS, aspiration, TT, ordering, quiescence).
    disable_pvs: bool,
    disable_aspiration: bool,
    disable_tt: bool,
    disable_ordering: bool,
    disable_quiescence: bool,
    /// Walk the whole game, comparing the incremental accumulator chain against a full
    /// refresh at every position (Case C evidence).
    parity: bool,
    /// Print the search statistics block (qsearch, TT, inference counters).
    stats: bool,
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
        hash_mb: 256,
        moves: String::new(),
        depths: Vec::new(),
        movetimes: Vec::new(),
        nodes: None,
        workers: 1,
        initial: None,
        parity: false,
        stats: false,
        disable_pvs: false,
        disable_aspiration: false,
        disable_tt: false,
        disable_ordering: false,
        disable_quiescence: false,
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
            "--hash-mb" => {
                parsed.hash_mb = parse_list(&value())
                    .first()
                    .copied()
                    .and_then(|mb| usize::try_from(mb).ok())
                    .unwrap_or(256);
            }
            "--moves" => parsed.moves = value(),
            "--initial" => parsed.initial = Some(value()),
            "--parity" => parsed.parity = true,
            "--stats" => parsed.stats = true,
            "--no-pvs" => parsed.disable_pvs = true,
            "--no-aspiration" => parsed.disable_aspiration = true,
            "--no-tt" => parsed.disable_tt = true,
            "--no-ordering" => parsed.disable_ordering = true,
            "--no-quiescence" => parsed.disable_quiescence = true,
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

#[expect(
    clippy::too_many_lines,
    reason = "the closed option table and mode dispatch stay one readable unit"
)]
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
    if arguments.parity {
        parity_walk(&model, &initial, &moves);
        return;
    }
    let mut engine = model
        .search_engine(
            SearchConfig {
                transposition_entries: SearchEngine::transposition_entries_for_megabytes(
                    arguments.hash_mb,
                ),
                enable_pvs: !arguments.disable_pvs,
                enable_aspiration: !arguments.disable_aspiration,
                enable_transposition_table: !arguments.disable_tt,
                enable_move_ordering: !arguments.disable_ordering,
                enable_quiescence: !arguments.disable_quiescence,
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
        report(&format!("depth<{depth}"), &result, arguments.stats);
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
        report(&format!("movetime<{movetime}"), &result, arguments.stats);
    }
}

/// Case C evidence: at every position of the game, advance the chained incremental
/// accumulator (never reset to a refresh) and compare it against an independent
/// full feature-scan refresh of the same model, plus the snapshot/restore contract the
/// search's undo path relies on. Exits non-zero on any divergence beyond the
/// implementation-defined tolerances.
///
/// # Panics
///
/// Dies via [`exit`](std::process::exit) with status 1 on rejected inputs and on any
/// parity divergence.
#[expect(
    clippy::too_many_lines,
    reason = "the chained walk, branch checks and tolerance gates stay one coherent unit"
)]
fn parity_walk(model: &PurePlayingEvaluator, initial: &Position, moves: &[Move]) {
    let PurePlayingEvaluator::Osaval03(evaluator) = model else {
        die("parity walk requires the OSAVAL03 format");
    };
    let mut position = initial.clone();
    let mut chained = match evaluator.accumulator(&position) {
        Ok(state) => state,
        Err(error) => die(format!("start position accumulator failed: {error}")),
    };
    let mut worst_drift: f64 = 0.0;
    let mut worst_output_drift: f32 = 0.0;
    let mut worst_ply: usize = 0;
    let mut mismatches: Vec<String> = Vec::new();
    let mut king_moves = 0_usize;
    let mut promotions = 0_usize;
    let mut captures = 0_usize;
    let mut drops = 0_usize;
    let mut branch_checks = 0_usize;
    for (ply, movement) in moves.iter().enumerate() {
        let after = {
            let mut next = position.clone();
            next.make_move(*movement)
                .unwrap_or_else(|error| die(format!("illegal move {ply}: {error}")));
            next
        };
        if matches!(movement, Move::Drop { .. }) {
            drops += 1;
        } else {
            let is_capture = position.piece_at(movement.destination()).is_some();
            if is_capture {
                captures += 1;
            }
            if let Move::Normal { from, promote, .. } = movement {
                if position
                    .piece_at(*from)
                    .is_some_and(|piece| piece.kind == PieceKind::King)
                {
                    king_moves += 1;
                }
                if *promote {
                    promotions += 1;
                }
            }
        }
        let before_state = chained.clone();
        let parent = match evaluator.update_accumulator(&mut chained, *movement, &after) {
            Ok(parent) => parent,
            Err(error) => die(format!("incremental update rejected move {ply}: {error:?}")),
        };
        // The returned snapshot must restore the parent bit-exactly: the search's
        // undo path replaces its state with it instead of inverting float operations.
        if parent != before_state {
            mismatches.push(format!(
                "ply {ply} move {}: parent snapshot is not bit-exact",
                to_usi_move(*movement)
            ));
        }
        // Branching from the same parent with an unrelated sibling must leave the
        // chain state untouched — the search's alpha/beta restore semantics.
        if let Some(sibling) = position
            .legal_moves()
            .into_iter()
            .find(|candidate| candidate != movement)
        {
            let mut branched = before_state.clone();
            let mut sibling_after = position.clone();
            sibling_after
                .make_move(sibling)
                .unwrap_or_else(|error| die(format!("illegal sibling at {ply}: {error}")));
            let sibling_parent =
                match evaluator.update_accumulator(&mut branched, sibling, &sibling_after) {
                    Ok(parent) => parent,
                    Err(error) => die(format!("sibling update rejected at {ply}: {error:?}")),
                };
            if sibling_parent != before_state || branched == before_state {
                mismatches.push(format!(
                    "ply {ply}: branching from the parent disturbed the chain state"
                ));
            }
            // The branch's own state must also agree with an independent refresh.
            if let Ok(sibling_refresh) = evaluator.accumulator(&sibling_after) {
                match evaluator.accumulator_parity(&branched, &sibling_refresh) {
                    Ok(sibling_parity) => {
                        if !sibling_parity.within_tolerance() {
                            mismatches.push(format!(
                                "ply {ply}: sibling branch drifted {:.3e}",
                                sibling_parity.max_value_drift,
                            ));
                        }
                    }
                    Err(error) => die(format!("sibling parity failed at {ply}: {error:?}")),
                }
            }
            branch_checks += 1;
        }
        let refreshed = match evaluator.accumulator(&after) {
            Ok(state) => state,
            Err(error) => die(format!("refresh failed at {ply}: {error:?}")),
        };
        let parity = match evaluator.accumulator_parity(&chained, &refreshed) {
            Ok(parity) => parity,
            Err(error) => die(format!("parity comparison failed at {ply}: {error:?}")),
        };
        let incremental_output = evaluator
            .infer_accumulator(&chained)
            .unwrap_or_else(|error| die(format!("infer failed at {ply}: {error:?}")));
        let refreshed_output = evaluator
            .infer_accumulator(&refreshed)
            .unwrap_or_else(|error| die(format!("infer failed at {ply}: {error:?}")));
        let cp_drift = i64::from(incremental_output.cp) - i64::from(refreshed_output.cp);
        let logit_drift = incremental_output
            .wdl_logits
            .iter()
            .zip(refreshed_output.wdl_logits.iter())
            .map(|(left, right)| (left - right).abs())
            .fold(0.0_f32, f32::max);
        if logit_drift > worst_output_drift {
            worst_output_drift = logit_drift;
        }
        if parity.max_value_drift > worst_drift {
            worst_drift = parity.max_value_drift;
            worst_ply = ply + 1;
        }
        if !parity.within_tolerance() || cp_drift.abs() > 1 || logit_drift > 1.0e-3 {
            mismatches.push(format!(
                "ply {ply} move {}: positions_match {} kings_match {} max_drift {:.3e} \
                 cp {cp_drift} wdl_drift {logit_drift:.3e}",
                to_usi_move(*movement),
                parity.positions_match,
                parity.kings_match,
                parity.max_value_drift,
            ));
        }
        position = after;
    }
    println!(
        "# parity positions {} exact {} king_moves {king_moves} promotions {promotions} \
         captures {captures} drops {drops} branch_checks {branch_checks} \
         worst_value_drift {worst_drift:.3e} worst_output_drift {worst_output_drift:.3e} at ply {worst_ply}",
        moves.len(),
        moves.len() - mismatches.len(),
    );
    for mismatch in &mismatches {
        eprintln!("# mismatch {mismatch}");
    }
    if !mismatches.is_empty() {
        die(format!(
            "incremental/refresh parity failed at {} of {} positions",
            mismatches.len(),
            moves.len()
        ));
    }
}

fn report(label: &str, result: &open_shogi_core::SearchResult, stats: bool) {
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
    if stats {
        let s = &result.stats;
        println!(
            "  stats tt_probes {} tt_hits {} tt_collisions {} beta_cutoffs {} qnodes {} learned_evals {} infer_errors {} fallbacks {}",
            s.tt_probes,
            s.tt_hits,
            s.tt_collisions,
            s.beta_cutoffs,
            s.qnodes,
            s.learned_eval_calls,
            s.osaval02_inference_errors,
            s.fallback_count,
        );
    }
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
