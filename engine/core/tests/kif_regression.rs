//! Regression fixtures from the attached 2026-09-28 R4 match (see local QA notes).
//! The positions pin mechanical search invariants; they do not encode expectations
//! about the learned evaluation.

use std::time::Duration;

use open_shogi_core::{
    CancellationToken, SearchConfig, SearchEngine, SearchLimits, SearchTermination, TimeControl,
    TimeManager, parse_sfen,
};

/// Ply 70 root: the first major sacrifice decision of the match.
const PLY70_SFEN: &str =
    "3g3n1/+L1s2kg2/3rppsp1/6p2/1nP4P1/pP2P3L/3G1PPS1/PKGB5/1N4RN1 w BS2L6P 70";
/// Ply 118 root: deep material deficit where the match engine answered instantly.
const PLY118_SFEN: &str = "3+N3k1/9/+L6G1/2P3RP1/9/1P1GPB3/K4PPS1/9/1N4RN1 w B2G3S3L12Pn 118";

#[test]
fn incident_positions_replay_and_search_legally() {
    for sfen in [PLY70_SFEN, PLY118_SFEN] {
        let position = parse_sfen(sfen).expect("fixture parses");
        assert!(!position.legal_moves().is_empty(), "{sfen} has legal moves");
        let mut engine = SearchEngine::new(SearchConfig::default());
        let result = engine.search(
            &position,
            SearchLimits {
                max_depth: 3,
                max_nodes: Some(200_000),
                movetime: None,
            },
            &CancellationToken::new(),
        );
        assert_eq!(result.termination, SearchTermination::Completed);
        assert!(position.is_legal_move(result.best_move.expect("best move")));
        assert!(!result.root_moves.is_empty());
    }
}

#[test]
fn incident_positions_survive_parallel_stress() {
    let plan = TimeManager::default()
        .plan(
            open_shogi_core::Side::White,
            TimeControl {
                black_time_ms: Some(5_000),
                white_time_ms: Some(5_000),
                casual: false,
                ..TimeControl::casual()
            },
            64,
        )
        .expect("clock plan");
    for round in 0..6 {
        for sfen in [PLY70_SFEN, PLY118_SFEN] {
            let position = parse_sfen(sfen).expect("fixture parses");
            let mut engine = SearchEngine::new(SearchConfig {
                transposition_entries: 4_096,
                ..SearchConfig::default()
            });
            let cancellation = CancellationToken::new();
            if round % 3 == 2 {
                let stopper = cancellation.clone();
                std::thread::spawn(move || {
                    std::thread::sleep(Duration::from_millis(30));
                    stopper.cancel();
                });
            }
            let result = engine.search_parallel_managed_with_callback(
                &position,
                plan,
                &cancellation,
                4,
                |_| {},
            );
            if round % 3 == 2 {
                assert_eq!(
                    result.termination,
                    SearchTermination::Cancelled,
                    "round {round} {sfen} elapsed {:?} nodes {}",
                    result.elapsed,
                    result.nodes
                );
            }
            // Every completed round must end in a usable, legal move.
            assert!(position.is_legal_move(result.best_move.expect("best move")));
        }
    }
}
