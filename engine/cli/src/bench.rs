use std::{fmt::Write as _, time::Duration};

use open_shogi_core::{
    CancellationToken, Position, SearchConfig, SearchEngine, SearchLimits, parse_usi_move,
};

use crate::args::parse_next;

const BENCH_SCHEMA: &str = "phase2_benchmark/v1";
const DEFAULT_NODES: u64 = 25_000;
const MAX_NODES: u64 = 1_000_000_000;

pub fn run(arguments: &[String]) -> Result<(), String> {
    let mut depth = 8_u8;
    let mut nodes = None;
    let mut index = 0;
    while index < arguments.len() {
        match arguments[index].as_str() {
            "--depth" => depth = parse_next(arguments, &mut index, "--depth")?,
            "--nodes" => nodes = Some(parse_next(arguments, &mut index, "--nodes")?),
            argument => return Err(format!("unknown bench argument: {argument}")),
        }
        index += 1;
    }
    if depth == 0 || depth > 64 {
        return Err("--depth must be 1..=64".to_owned());
    }
    let nodes = nodes.unwrap_or(DEFAULT_NODES);
    if nodes == 0 || nodes > MAX_NODES {
        return Err(format!("--nodes must be 1..={MAX_NODES}"));
    }
    let positions = fixed_positions()?;
    let mut engine = SearchEngine::new(SearchConfig {
        transposition_entries: 16_384,
        ..SearchConfig::default()
    });
    let mut total_nodes = 0_u64;
    let mut elapsed = Duration::ZERO;
    let mut depth_sum = 0_u64;
    let mut tt_probes = 0_u64;
    let mut tt_hits = 0_u64;
    let mut cutoffs = 0_u64;
    let mut candidate_moves = 0_u64;
    let mut pruned_moves = 0_u64;
    let mut details = String::new();
    for (index, position) in positions.iter().enumerate() {
        let result = engine.search(
            position,
            SearchLimits {
                max_depth: depth,
                max_nodes: Some(nodes),
                movetime: None,
            },
            &CancellationToken::new(),
        );
        total_nodes = total_nodes.saturating_add(result.nodes);
        elapsed = elapsed.saturating_add(result.elapsed);
        depth_sum = depth_sum.saturating_add(u64::from(result.depth));
        tt_probes = tt_probes.saturating_add(result.stats.tt_probes);
        tt_hits = tt_hits.saturating_add(result.stats.tt_hits);
        cutoffs = cutoffs.saturating_add(result.stats.beta_cutoffs);
        candidate_moves = candidate_moves.saturating_add(result.stats.candidate_moves);
        pruned_moves = pruned_moves.saturating_add(result.stats.pruned_moves);
        if index > 0 {
            details.push(',');
        }
        write!(
            details,
            "{{\"id\":{},\"depth\":{},\"nodes\":{},\"elapsedMilliseconds\":{},\"nps\":{}}}",
            index,
            result.depth,
            result.nodes,
            result.elapsed.as_millis(),
            result.nps
        )
        .expect("writing to String cannot fail");
    }
    let elapsed_micros = u64::try_from(elapsed.as_micros()).unwrap_or(u64::MAX);
    let nps = decimal_ratio_scaled(total_nodes, 1_000_000, elapsed_micros);
    let count = u64::try_from(positions.len()).expect("position count fits u64");
    let average_depth = decimal_ratio(depth_sum, count);
    let tt_hit_rate = decimal_ratio(tt_hits, tt_probes);
    let cutoff_rate = decimal_ratio(cutoffs, total_nodes);
    let pruning_rate = decimal_ratio(pruned_moves, candidate_moves);
    let milliseconds_per_position = decimal_ratio(
        u64::try_from(elapsed.as_micros()).unwrap_or(u64::MAX),
        count.saturating_mul(1_000),
    );
    println!(
        "{{\"schema\":\"{BENCH_SCHEMA}\",\"positions\":{},\"nodes\":{total_nodes},\"nodesPerSecond\":{nps},\"averageDepth\":{average_depth},\"ttHitRate\":{tt_hit_rate},\"cutoffRate\":{cutoff_rate},\"pruningRate\":{pruning_rate},\"millisecondsPerPosition\":{milliseconds_per_position},\"peakMemoryBytes\":null,\"runs\":[{details}]}}",
        positions.len(),
    );
    Ok(())
}

fn fixed_positions() -> Result<Vec<Position>, String> {
    let sequences: &[&[&str]] = &[
        &[],
        &["7g7f", "3c3d", "2g2f", "8c8d"],
        &["7g7f", "8c8d", "2g2f", "8d8e", "8h7g", "3a4b"],
    ];
    sequences
        .iter()
        .map(|sequence| {
            let mut position = Position::startpos();
            for notation in *sequence {
                let movement = parse_usi_move(notation)
                    .map_err(|error| format!("invalid built-in bench move: {error}"))?;
                position
                    .make_move(movement)
                    .map_err(|error| format!("illegal built-in bench move: {error}"))?;
            }
            Ok(position)
        })
        .collect()
}

fn decimal_ratio(numerator: u64, denominator: u64) -> String {
    decimal_ratio_scaled(numerator, 1, denominator)
}

fn decimal_ratio_scaled(numerator: u64, multiplier: u64, denominator: u64) -> String {
    if denominator == 0 {
        return "0.000000".to_owned();
    }
    let numerator = u128::from(numerator).saturating_mul(u128::from(multiplier));
    let scaled = numerator.saturating_mul(1_000_000) / u128::from(denominator);
    format!("{}.{:06}", scaled / 1_000_000, scaled % 1_000_000)
}

#[cfg(test)]
mod tests {
    use super::{fixed_positions, run};

    #[test]
    fn fixed_positions_are_valid_and_distinct() {
        let positions = fixed_positions().unwrap();
        assert_eq!(positions.len(), 3);
        for position in &positions {
            position.validate().unwrap();
        }
        assert!(positions[0] != positions[1]);
    }

    #[test]
    fn invalid_bench_limits_are_rejected() {
        assert!(run(&["--nodes".into(), "0".into()]).is_err());
        assert!(run(&["--depth".into(), "65".into()]).is_err());
    }
}
