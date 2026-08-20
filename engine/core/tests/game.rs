use open_shogi_core::{
    EnteringKingDeclaration, EnteringKingRule, Game, GameEnd, ImpasseCondition, ImpasseOutcome,
    RepetitionOutcome, Side, parse_sfen, parse_usi_move,
};

#[test]
fn fourfold_repetition_uses_board_hands_and_side_not_move_number() {
    let mut game = Game::startpos();
    for _ in 0..3 {
        for notation in ["5i6h", "5a6b", "6h5i", "6b5a"] {
            game.play(parse_usi_move(notation).expect("notation"))
                .expect("legal cycle");
        }
    }

    assert_eq!(
        game.end(),
        Some(GameEnd::Repetition(RepetitionOutcome::NoContest))
    );
}

#[test]
fn continuous_check_repetition_is_a_loss_for_the_checker() {
    let position = parse_sfen("4k4/5R3/9/9/9/9/9/9/K8 b - 1").expect("fixture");
    let mut game = Game::new(position);
    for _ in 0..3 {
        for notation in ["4b5b", "5a4a", "5b4b", "4a5a"] {
            game.play(parse_usi_move(notation).expect("notation"))
                .expect("legal checking cycle");
        }
    }

    assert_eq!(
        game.end(),
        Some(GameEnd::Repetition(RepetitionOutcome::PerpetualCheckLoss(
            Side::Black
        )))
    );
}

#[test]
fn official_entering_king_profile_is_explicit_and_configurable() {
    let position = parse_sfen("RBG1KGS+N+L/RBGS+N4/9/9/9/9/9/9/4k4 b G2S2N3L18P 1")
        .expect("declaration fixture");
    let game = Game::new(position);

    assert_eq!(
        game.entering_king_declaration(Side::Black, EnteringKingRule::Disabled),
        EnteringKingDeclaration::Disabled
    );
    assert!(matches!(
        game.entering_king_declaration(Side::Black, EnteringKingRule::JsaOfficial2025),
        EnteringKingDeclaration::Win {
            side: Side::Black,
            ..
        }
    ));
}

#[test]
fn entering_king_declaration_requires_the_declarers_active_turn() {
    let position = parse_sfen("4K4/9/9/9/9/9/9/9/4k4 b - 1").expect("fixture");
    let game = Game::new(position);
    let permissive = EnteringKingRule::Custom {
        minimum_camp_pieces: 0,
        win_points: 0,
        no_contest_points: 0,
    };

    assert_eq!(
        game.entering_king_declaration(Side::White, permissive),
        EnteringKingDeclaration::Unavailable { side: Side::White }
    );
}

#[test]
fn entering_king_declaration_is_unavailable_after_move_five_hundred() {
    let position = parse_sfen("4K4/9/9/9/9/9/9/9/4k4 b - 501").expect("fixture");
    let game = Game::new(position);
    let permissive = EnteringKingRule::Custom {
        minimum_camp_pieces: 0,
        win_points: 0,
        no_contest_points: 0,
    };

    assert_eq!(
        game.entering_king_declaration(Side::Black, permissive),
        EnteringKingDeclaration::Unavailable { side: Side::Black }
    );
}

#[test]
fn mutual_impasse_uses_all_board_and_hand_points() {
    let position = parse_sfen("4K4/9/9/9/9/9/9/9/4k4 b RB2G2S2N2L9Prb2g2s2n2l9p 1")
        .expect("mutual impasse fixture");
    let game = Game::new(position);

    assert_eq!(
        game.adjudicate_impasse(ImpasseCondition::NoMatingProspect),
        Some(ImpasseOutcome::NoContest {
            black_points: 27,
            white_points: 27,
        })
    );
    assert_eq!(
        game.adjudicate_impasse(ImpasseCondition::NotEstablished),
        None
    );
}

#[test]
fn impasse_requires_at_least_one_entering_king() {
    let game = Game::startpos();
    assert_eq!(
        game.adjudicate_impasse(ImpasseCondition::NoMatingProspect),
        None
    );
    assert!(!game.five_hundred_move_no_contest(ImpasseCondition::NoMatingProspect));
}

#[test]
fn game_undo_restores_the_initial_position_and_clears_terminal_state() {
    let mut game = Game::startpos();
    let initial = game.position().clone();
    game.play(parse_usi_move("7g7f").expect("notation"))
        .expect("legal move");
    assert!(game.undo());
    assert_eq!(game.position(), &initial);
    assert_eq!(game.end(), None);
    assert!(!game.undo());
}

#[test]
fn game_created_from_checkmate_is_already_terminal() {
    let position = parse_sfen("3lkl3/3pRp3/4G4/9/9/9/9/9/K8 w - 1").expect("checkmate fixture");
    let mut game = Game::new(position);

    assert_eq!(
        game.end(),
        Some(GameEnd::Checkmate {
            winner: Side::Black,
        })
    );
    assert!(game.resign().is_err());
}
