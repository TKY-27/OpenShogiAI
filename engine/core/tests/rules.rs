use open_shogi_core::{
    BOARD_SQUARES, Hand, HandPiece, Move, Piece, PieceKind, Position, Side, Square, parse_sfen,
    parse_usi_move,
};
use std::collections::BTreeSet;

fn square(text: &str) -> Square {
    let file = text.as_bytes()[0] - b'0';
    let rank = text.as_bytes()[1] - b'a' + 1;
    Square::new(file, rank).expect("fixture square")
}

fn central_piece_position(kind: PieceKind) -> Position {
    let mut board = [None; BOARD_SQUARES];
    if kind != PieceKind::King {
        board[square("9h").index()] = Some(Piece::new(Side::Black, PieceKind::King));
    }
    board[square("1b").index()] = Some(Piece::new(Side::White, PieceKind::King));
    board[square("5e").index()] = Some(Piece::new(Side::Black, kind));
    Position::from_parts(board, [Hand::default(); 2], Side::Black, 1).expect("fixture position")
}

#[test]
fn standard_position_has_hand_audited_root_moves() {
    let position = Position::startpos();
    let moves = position.legal_moves();

    // Derived directly from the official movement rules: 9 pawn, 2 lance, 4 silver,
    // 6 gold, 3 king, and 6 rook moves.
    assert_eq!(moves.len(), 30);
    assert!(moves.contains(&parse_usi_move("7g7f").expect("notation")));
    assert!(moves.contains(&parse_usi_move("2h7h").expect("notation")));
}

#[test]
fn every_piece_kind_has_the_official_movement_footprint() {
    let expectations = [
        (PieceKind::Pawn, 1),
        (PieceKind::Lance, 4),
        (PieceKind::Knight, 2),
        (PieceKind::Silver, 5),
        (PieceKind::Gold, 6),
        (PieceKind::Bishop, 16),
        (PieceKind::Rook, 16),
        (PieceKind::King, 8),
        (PieceKind::PromotedPawn, 6),
        (PieceKind::PromotedLance, 6),
        (PieceKind::PromotedKnight, 6),
        (PieceKind::PromotedSilver, 6),
        (PieceKind::Horse, 20),
        (PieceKind::Dragon, 20),
    ];

    for (kind, expected_destinations) in expectations {
        let position = central_piece_position(kind);
        let destinations: BTreeSet<Square> = position
            .legal_moves()
            .into_iter()
            .filter_map(|mv| match mv {
                Move::Normal { from, to, .. } if from == square("5e") => Some(to),
                _ => None,
            })
            .collect();
        assert_eq!(
            destinations.len(),
            expected_destinations,
            "unexpected movement footprint for {kind:?}"
        );
    }
}

#[test]
fn promotion_is_optional_on_zone_entry_but_forced_on_last_rank() {
    let optional = parse_sfen("4k4/9/9/4P4/9/9/9/9/K8 b - 1").expect("fixture");
    let optional_plain = parse_usi_move("5d5c").expect("notation");
    let optional_promoted = parse_usi_move("5d5c+").expect("notation");
    assert!(optional.legal_moves().contains(&optional_plain));
    assert!(optional.legal_moves().contains(&optional_promoted));

    let forced = parse_sfen("8k/4P4/9/9/9/9/9/9/K8 b - 1").expect("fixture");
    let plain = parse_usi_move("5b5a").expect("notation");
    let promoted = parse_usi_move("5b5a+").expect("notation");
    assert!(!forced.legal_moves().contains(&plain));
    assert!(forced.legal_moves().contains(&promoted));

    let white_forced = parse_sfen("8K/9/9/9/9/9/9/4p4/k8 w - 1").expect("fixture");
    assert!(
        !white_forced
            .legal_moves()
            .contains(&parse_usi_move("5h5i").expect("notation"))
    );
    assert!(
        white_forced
            .legal_moves()
            .contains(&parse_usi_move("5h5i+").expect("notation"))
    );
}

#[test]
fn knight_and_lance_cannot_be_dropped_without_future_mobility() {
    let position = parse_sfen("4k4/9/9/9/9/9/9/9/4K4 b NL 1").expect("fixture");
    for mv in position.legal_moves() {
        match mv {
            Move::Drop {
                piece: HandPiece::Lance,
                to,
            } => assert_ne!(to.rank(), 1),
            Move::Drop {
                piece: HandPiece::Knight,
                to,
            } => assert!(to.rank() > 2),
            _ => {}
        }
    }
}

#[test]
fn every_hand_piece_can_be_dropped_on_a_live_empty_square() {
    let position = parse_sfen("4k4/9/9/9/9/9/9/9/4K4 b RBGSNLP 1").expect("fixture");
    for piece in HandPiece::ALL {
        assert!(
            position.legal_moves().contains(&Move::Drop {
                piece,
                to: square("5e"),
            }),
            "missing representative drop for {piece:?}"
        );
    }
}

#[test]
fn nifu_counts_only_unpromoted_pawns() {
    let unpromoted = parse_sfen("4k4/9/9/9/4P4/9/9/9/4K4 b P 1").expect("fixture");
    assert!(!unpromoted.legal_moves().contains(&Move::Drop {
        piece: HandPiece::Pawn,
        to: square("5d"),
    }));

    let promoted = parse_sfen("4k4/9/9/9/4+P4/9/9/9/4K4 b P 1").expect("fixture");
    assert!(promoted.legal_moves().contains(&Move::Drop {
        piece: HandPiece::Pawn,
        to: square("5d"),
    }));
}

#[test]
fn pawn_drop_mate_is_excluded() {
    let position = parse_sfen("3lkl3/3p1p3/4G4/9/9/9/9/9/K8 b P 1").expect("fixture");
    let pawn_drop_mate = Move::Drop {
        piece: HandPiece::Pawn,
        to: square("5b"),
    };

    assert!(!position.legal_moves().contains(&pawn_drop_mate));
}

#[test]
fn checked_side_without_a_legal_response_is_checkmated() {
    let position = parse_sfen("3lkl3/3pRp3/4G4/9/9/9/9/9/K8 w - 1").expect("fixture");
    assert!(position.is_in_check(Side::White));
    assert!(position.is_checkmate());
    assert!(position.legal_moves().is_empty());
}

#[test]
fn a_pinned_piece_cannot_expose_its_king() {
    let position = parse_sfen("4r3k/9/9/9/9/9/9/4G4/4K4 b - 1").expect("fixture");

    assert!(
        !position
            .legal_moves()
            .contains(&parse_usi_move("5h4g").expect("notation"))
    );
    assert!(
        position
            .legal_moves()
            .contains(&parse_usi_move("5h5g").expect("notation"))
    );
}

#[test]
fn double_check_allows_only_king_moves() {
    let position = parse_sfen("4r3k/9/9/9/8b/9/9/9/4K4 b - 1").expect("fixture");
    assert!(position.is_in_check(Side::Black));

    for mv in position.legal_moves() {
        let Move::Normal { from, .. } = mv else {
            panic!("a drop cannot answer this double check");
        };
        assert_eq!(from, square("5i"));
    }
}

#[test]
fn make_and_unmake_restore_capture_hand_hash_and_move_number() {
    let mut position = parse_sfen("4k4/9/9/9/4p4/4P4/9/9/4K4 b - 17").expect("fixture");
    let original = position.clone();
    let mv = parse_usi_move("5f5e").expect("notation");

    let undo = position.make_move(mv).expect("legal capture");
    assert_eq!(position.hand(Side::Black).count(HandPiece::Pawn), 1);
    assert_eq!(position.zobrist_hash(), position.recompute_zobrist());
    position.unmake_move(undo);

    assert_eq!(position, original);
}

#[test]
fn move_number_overflow_is_rejected_without_mutating_the_position() {
    let mut position = parse_sfen("4k4/9/9/9/9/9/4P4/9/4K4 b - 4294967295").expect("fixture");
    let original = position.clone();

    assert_eq!(
        position.make_move(parse_usi_move("5g5f").expect("notation")),
        Err(open_shogi_core::IllegalMove::MoveNumberOverflow)
    );
    assert_eq!(position, original);
}

#[test]
fn capturing_a_promoted_piece_adds_its_base_kind_to_hand() {
    let mut position = parse_sfen("4k4/9/9/9/4+p4/4S4/9/9/4K4 b - 1").expect("fixture");
    position
        .make_move(parse_usi_move("5f5e").expect("notation"))
        .expect("legal capture");

    assert_eq!(position.hand(Side::Black).count(HandPiece::Pawn), 1);
    assert_eq!(
        position
            .piece_at(square("5e"))
            .expect("capturing piece")
            .kind,
        PieceKind::Silver
    );
}

#[test]
fn legal_moves_never_capture_the_opposing_king() {
    let position = parse_sfen("4k4/4R4/9/9/9/9/9/9/4K4 b - 1").expect("fixture");
    assert!(position.is_in_check(Side::White));
    assert!(
        !position
            .legal_moves()
            .iter()
            .any(|mv| mv.destination() == square("5a"))
    );
}
