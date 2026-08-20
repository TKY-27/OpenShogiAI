use open_shogi_core::{
    CsaGame, CsaResultValidation, CsaSpecialMove, Position, Side, Square, parse_csa_game,
    parse_sfen, parse_usi_move, to_csa_game, to_csa_move, to_sfen, to_usi_move,
};

const STARTPOS_SFEN: &str = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";

#[test]
fn start_position_sfen_round_trips_canonically() {
    let position = parse_sfen(STARTPOS_SFEN).expect("valid SFEN");
    assert_eq!(position, Position::startpos());
    assert_eq!(to_sfen(&position), STARTPOS_SFEN);
}

#[test]
fn sfen_hands_and_promotions_round_trip_canonically() {
    let source = "4k4/9/4+P4/9/9/4+r4/9/9/4K4 w Rbgs2n4p 81";
    let position = parse_sfen(source).expect("valid SFEN");
    let canonical = to_sfen(&position);
    let reparsed = parse_sfen(&canonical).expect("canonical output parses");

    assert_eq!(reparsed, position);
    assert_eq!(reparsed.side_to_move(), Side::White);
}

#[test]
fn malformed_sfen_is_rejected_without_guessing() {
    for source in [
        "",
        "9/9/9/9/9/9/9/9/9 b - 1",
        "4k4/9/9/9/9/9/9/9/4K3 b - 1",
        "4k4/9/9/9/9/9/9/9/4K4 x - 1",
        "4k4/9/9/9/9/9/9/9/4K4 b K 1",
        "4k4/9/9/9/9/9/9/9/4K4 b - 0",
        "18/9/9/9/4k4/9/9/9/4K4 b - 1",
    ] {
        assert!(
            parse_sfen(source).is_err(),
            "accepted malformed SFEN: {source}"
        );
    }
}

#[test]
fn usi_normal_promotion_and_drop_moves_round_trip() {
    for notation in ["7g7f", "2b3c+", "P*5e"] {
        let mv = parse_usi_move(notation).expect("valid USI move");
        assert_eq!(to_usi_move(mv), notation);
    }
}

#[test]
fn csa_move_uses_destination_piece_code() {
    let position = parse_sfen("k8/4P4/9/9/9/9/9/9/4K4 b - 1").expect("fixture");
    let mv = parse_usi_move("5b5a+").expect("notation");
    assert_eq!(
        to_csa_move(&position, mv).expect("CSA formatting"),
        "+5251TO"
    );
}

#[test]
fn csa_move_writer_rejects_a_positionally_illegal_move() {
    let position = Position::startpos();
    let illegal = parse_usi_move("7g7e").expect("notation");
    assert!(to_csa_move(&position, illegal).is_err());
}

#[test]
fn csa_game_round_trips_through_canonical_writer() {
    let source = "\
'CSA encoding=UTF-8
V3.0
N+Sente
N-Gote
$EVENT:OpenShogiAI fixture
PI
+
+7776FU
T1
-3334FU
T2
%TORYO
";
    let parsed = parse_csa_game(source).expect("valid CSA");
    assert_eq!(parsed.moves.len(), 2);
    assert_eq!(parsed.special_move, Some(CsaSpecialMove::Resign));
    assert_eq!(
        parsed.result_validation,
        CsaResultValidation::ExternalCondition
    );

    let encoded = to_csa_game(&parsed).expect("canonical CSA");
    assert!(encoded.starts_with("'CSA encoding=UTF-8\nV3.0\n"));
    let reparsed = parse_csa_game(&encoded).expect("writer output parses");
    assert_eq!(reparsed.initial_position, parsed.initial_position);
    assert_eq!(reparsed.moves, parsed.moves);
    assert_eq!(reparsed.special_move, parsed.special_move);
}

#[test]
fn csa_v3_accepts_milliseconds_and_terminal_time() {
    let source = "V3.0\nPI\n+\n+7776FU\nT15.123\n-3334FU\n%TORYO\nT0.250\n";
    let parsed = parse_csa_game(source).expect("CSA V3 time syntax");

    assert_eq!(parsed.moves.len(), 2);
    assert_eq!(parsed.special_move, Some(CsaSpecialMove::Resign));
}

#[test]
fn csa_v3_accepts_one_to_three_fractional_time_digits() {
    for time in ["T1.1", "T1.12", "T1.123"] {
        let source = format!("V3.0\nPI\n+\n+7776FU\n{time}\n%CHUDAN\n");
        parse_csa_game(&source).expect("CSA V3 fractional time");
    }
}

#[test]
fn csa_v3_accepts_move_and_time_multi_statements() {
    let source = "V3.0\nPI\n+\n+7776FU,T1,-3334FU,T2.5\n%CHUDAN\n";
    let parsed = parse_csa_game(source).expect("CSA V3 multi-statement line");
    assert_eq!(parsed.moves.len(), 2);
}

#[test]
fn csa_v3_accepts_general_multi_statements() {
    let parsed = parse_csa_game("V3.0,PI,+\n%CHUDAN\n").expect("CSA general multi-statement line");
    assert_eq!(parsed.initial_position, Position::startpos());
}

#[test]
fn csa_pi_removals_create_a_valid_handicap_position() {
    let source = "V3.0\nPI82HI22KA\n+\n%CHUDAN\n";
    let parsed = parse_csa_game(source).expect("CSA PI removal syntax");

    assert_eq!(
        parsed
            .initial_position
            .piece_at(Square::new(8, 2).expect("square")),
        None
    );
    assert_eq!(
        parsed
            .initial_position
            .piece_at(Square::new(2, 2).expect("square")),
        None
    );
}

#[test]
fn csa_max_moves_has_a_typed_terminal_value() {
    let parsed = parse_csa_game("V3.0\nPI\n+\n%MAX_MOVES\n").expect("CSA terminal");
    assert_eq!(parsed.special_move, Some(CsaSpecialMove::MaxMoves));
    assert_eq!(
        parsed.result_validation,
        CsaResultValidation::ExternalCondition
    );
}

#[test]
fn csa_rejects_unverified_checkmate_and_perpetual_check_results() {
    assert!(parse_csa_game("V3.0\nPI\n+\n%TSUMI\n").is_err());
    assert!(parse_csa_game("V3.0\nPI\n+\n%OUTE_SENNICHITE\n").is_err());
}

#[test]
fn csa_early_repetition_agreement_is_preserved_as_an_external_condition() {
    let parsed = parse_csa_game("V3.0\nPI\n+\n%SENNICHITE\n").expect("agreement-dependent result");
    assert_eq!(
        parsed.result_validation,
        CsaResultValidation::ExternalCondition
    );
}

#[test]
fn csa_verifies_checkmate_from_the_replayed_position() {
    let game = CsaGame {
        version: "V3.0".to_owned(),
        black_name: None,
        white_name: None,
        metadata: Vec::new(),
        initial_position: parse_sfen("3lkl3/3pRp3/4G4/9/9/9/9/9/K8 w - 1")
            .expect("checkmate fixture"),
        moves: Vec::new(),
        special_move: Some(CsaSpecialMove::Checkmate),
        result_validation: CsaResultValidation::Verified,
    };
    let encoded = to_csa_game(&game).expect("canonical CSA");
    let parsed = parse_csa_game(&encoded).expect("verified CSA checkmate");
    assert_eq!(parsed.result_validation, CsaResultValidation::Verified);
}

#[test]
fn csa_verifies_repetition_from_replayed_history() {
    let mut moves = Vec::new();
    for _ in 0..3 {
        moves.extend(
            ["5i6h", "5a6b", "6h5i", "6b5a"]
                .map(|notation| parse_usi_move(notation).expect("repetition move")),
        );
    }
    let game = CsaGame {
        version: "V3.0".to_owned(),
        black_name: None,
        white_name: None,
        metadata: Vec::new(),
        initial_position: Position::startpos(),
        moves,
        special_move: Some(CsaSpecialMove::Repetition),
        result_validation: CsaResultValidation::Verified,
    };
    let parsed =
        parse_csa_game(&to_csa_game(&game).expect("canonical CSA")).expect("verified repetition");
    assert_eq!(parsed.result_validation, CsaResultValidation::Verified);
}

#[test]
fn csa_replay_can_continue_after_a_broken_repetition_claim() {
    let mut moves = Vec::new();
    for _ in 0..3 {
        moves.extend(
            ["5i6h", "5a6b", "6h5i", "6b5a"]
                .map(|notation| parse_usi_move(notation).expect("repetition move")),
        );
    }
    moves.push(parse_usi_move("7g7f").expect("breaking move"));
    let game = CsaGame {
        version: "V3.0".to_owned(),
        black_name: None,
        white_name: None,
        metadata: Vec::new(),
        initial_position: Position::startpos(),
        moves,
        special_move: Some(CsaSpecialMove::Interrupted),
        result_validation: CsaResultValidation::ExternalCondition,
    };
    let parsed =
        parse_csa_game(&to_csa_game(&game).expect("canonical CSA")).expect("continued CSA replay");
    assert_eq!(parsed.moves.len(), 13);
    assert_eq!(
        parsed.result_validation,
        CsaResultValidation::ExternalCondition
    );
}

#[test]
fn csa_rejects_position_declarations_after_all_remaining_or_mixed_with_pi() {
    for source in [
        "V3.0\nP+00AL\nPI\n+\n",
        "V3.0\nP+59OU00AL\nP-51OU\n+\n",
        "V3.0\nP+00AL59OU\nP-51OU\n+\n",
    ] {
        assert!(
            parse_csa_game(source).is_err(),
            "accepted invalid CSA: {source}"
        );
    }
}

#[test]
fn csa_requires_version_before_record_sections() {
    assert!(parse_csa_game("PI\nV3.0\n+\n").is_err());
    assert!(parse_csa_game("V2.2\nPI\n+\n").is_err());
    assert!(parse_csa_game("V3.999\nPI\n+\n").is_err());
}

#[test]
fn csa_illegal_record_reports_failure() {
    let source = "V3.0\nPI\n+\n+7775FU\n";
    assert!(parse_csa_game(source).is_err());
}

#[test]
fn csa_writer_rejects_an_illegal_move_sequence() {
    let game = CsaGame {
        version: "V3.0".to_owned(),
        black_name: None,
        white_name: None,
        metadata: Vec::new(),
        initial_position: Position::startpos(),
        moves: vec![parse_usi_move("7g7e").expect("notation")],
        special_move: None,
        result_validation: CsaResultValidation::Missing,
    };
    assert!(to_csa_game(&game).is_err());
}

#[test]
fn csa_writer_rejects_a_terminal_validation_inconsistent_with_replay() {
    let game = CsaGame {
        version: "V3.0".to_owned(),
        black_name: None,
        white_name: None,
        metadata: Vec::new(),
        initial_position: Position::startpos(),
        moves: Vec::new(),
        special_move: Some(CsaSpecialMove::Checkmate),
        result_validation: CsaResultValidation::Verified,
    };
    assert!(to_csa_game(&game).is_err());
}

#[test]
fn csa_writer_rejects_metadata_keys_that_cannot_round_trip() {
    let game = CsaGame {
        version: "V3.0".to_owned(),
        black_name: None,
        white_name: None,
        metadata: vec![("A:B".to_owned(), "value".to_owned())],
        initial_position: Position::startpos(),
        moves: Vec::new(),
        special_move: None,
        result_validation: CsaResultValidation::Missing,
    };
    assert!(to_csa_game(&game).is_err());
}

#[test]
fn csa_writer_rejects_unescaped_multi_statement_delimiters_in_text() {
    let name_with_comma = CsaGame {
        version: "V3.0".to_owned(),
        black_name: Some("A,B".to_owned()),
        white_name: None,
        metadata: Vec::new(),
        initial_position: Position::startpos(),
        moves: Vec::new(),
        special_move: None,
        result_validation: CsaResultValidation::Missing,
    };
    let metadata_with_comma = CsaGame {
        black_name: None,
        metadata: vec![("EVENT".to_owned(), "one,two".to_owned())],
        ..name_with_comma.clone()
    };
    assert!(to_csa_game(&name_with_comma).is_err());
    assert!(to_csa_game(&metadata_with_comma).is_err());
}

#[test]
fn csa_writer_rejects_an_unrepresentable_initial_move_number() {
    let game = CsaGame {
        version: "V3.0".to_owned(),
        black_name: None,
        white_name: None,
        metadata: Vec::new(),
        initial_position: parse_sfen(
            "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 50",
        )
        .expect("SFEN fixture"),
        moves: Vec::new(),
        special_move: None,
        result_validation: CsaResultValidation::Missing,
    };
    assert!(to_csa_game(&game).is_err());
}
