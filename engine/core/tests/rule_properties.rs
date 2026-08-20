use open_shogi_core::{Position, Side, parse_sfen, to_sfen};
use proptest::prelude::*;

fn next_random(state: &mut u64) -> u64 {
    *state = state
        .wrapping_add(0x9E37_79B9_7F4A_7C15)
        .rotate_left(17)
        .wrapping_mul(0xBF58_476D_1CE4_E5B9);
    *state
}

fn material_count(position: &Position) -> u16 {
    let board = position
        .board()
        .iter()
        .filter(|piece| piece.is_some())
        .count();
    u16::try_from(board).expect("board count fits u16")
        + position.hand(Side::Black).total()
        + position.hand(Side::White).total()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    #[test]
    fn random_legal_sequences_preserve_core_invariants(seed: u64, requested_plies in 1usize..80) {
        let mut random_state = seed;
        let mut position = Position::startpos();
        let initial = position.clone();
        let mut undos = Vec::new();

        for _ in 0..requested_plies {
            let moves = position.legal_moves();
            if moves.is_empty() {
                break;
            }
            let move_index = usize::try_from(next_random(&mut random_state))
                .unwrap_or(0) % moves.len();
            let mv = moves[move_index];
            let before = position.clone();
            let undo = position.make_move(mv).expect("generated move is legal");

            prop_assert!(!position.is_in_check(before.side_to_move()));
            prop_assert_eq!(position.zobrist_hash(), position.recompute_zobrist());
            prop_assert_eq!(material_count(&position), 40);

            let canonical_sfen = to_sfen(&position);
            let reparsed = parse_sfen(&canonical_sfen).expect("serialized legal state parses");
            prop_assert!(position.same_state(&reparsed));

            position.unmake_move(undo.clone());
            prop_assert_eq!(&position, &before);
            let replay_undo = position.make_move(mv).expect("move remains legal after undo");
            prop_assert_eq!(position.zobrist_hash(), position.recompute_zobrist());
            undos.push(replay_undo);
        }

        while let Some(undo) = undos.pop() {
            position.unmake_move(undo);
        }
        prop_assert_eq!(position, initial);
    }
}
