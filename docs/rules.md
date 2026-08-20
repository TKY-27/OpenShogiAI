# Shogi rules profile

## Authority and scope

Phase 1 follows the Japan Shogi Association (JSA) official game rules published for
professional and amateur competition and reviewed on 2026-07-29. The deterministic core
implements board legality; tournament operations such as clocks, restarts, and colours after
a no-contest remain adapter responsibilities.

USI and SFEN follow the ShogiDokoro-maintained de facto protocol description. CSA records
follow the Computer Shogi Association record format version 3.0. These notation documents do
not override JSA game legality.

The engine is an independent implementation. No existing shogi engine or shogi library source
was consulted or copied.

## Coordinates and canonical state

- The board has nine files (`9` through `1` when scanning left-to-right from Black's view) and
  nine ranks (`a` through `i`, or `1` through `9` internally).
- Black/sente moves toward decreasing ranks. White/gote moves toward increasing ranks.
- A position consists of the board, both hands, and the side to move. The SFEN move number is
  metadata and is not part of position identity for repetition.
- SFEN output scans rank `a` to `i`, file `9` to `1`. Hand output uses `R B G S N L P` order,
  Black before White. Parsers validate syntax and semantics; serializers emit one canonical
  representation.
- Zobrist hashes are deterministic acceleration keys. Repetition is confirmed against full
  position state so a hash collision cannot adjudicate a game.

## Movement and promotion

| Piece | Unpromoted movement | Promoted movement |
| --- | --- | --- |
| King | One square in any direction | Not applicable |
| Rook | Any unobstructed distance orthogonally | Rook plus one square diagonally |
| Bishop | Any unobstructed distance diagonally | Bishop plus one square orthogonally |
| Gold | Forward, forward diagonals, sideways, and directly backward | Not applicable |
| Silver | Forward, forward diagonals, and backward diagonals | Gold |
| Knight | Two forward and one sideways, jumping intervening squares | Gold |
| Lance | Any unobstructed distance directly forward | Gold |
| Pawn | One square directly forward | Gold |

Rook, bishop, silver, knight, lance, and pawn may promote when either the origin or
destination is in the opponent's three-rank camp. Promotion is optional except that a pawn or
lance reaching the last rank and a knight reaching either of the last two ranks must promote.
Promoted pieces never promote again.

A captured piece returns to its unpromoted kind in the capturer's hand. Kings cannot enter a
hand. A drop places an unpromoted hand piece on an empty square and cannot promote immediately.

## Legal move pipeline

The core first generates movement- and occupancy-correct candidates, then filters them by the
following rules:

1. A move may not capture the opposing king. Victory is represented by checkmate.
2. The moving side's king must exist and may not remain or move into attack.
3. A pawn may not be dropped on a file already containing an unpromoted pawn of that side
   (`nifu`). Promoted pawns do not count.
4. Pawns and lances may not be dropped on their last rank; knights may not be dropped on their
   last two ranks.
5. A pawn drop that gives check is illegal when the opponent has no complete legal response
   (`uchifuzume`). A pawn moved from the board may checkmate normally.

Attack detection is separate from capture application and includes the opposing king's
adjacent squares. Pins, discovered checks, and double checks consequently require no special
exception: only candidates that leave the king safe survive.

## Checkmate and recorded endings

Checkmate is a checked side having no legal reply. The core also records resignation as a
separate external decision. CSA special endings remain distinct typed values rather than being
silently mapped to checkmate. Replay verifies deterministic checkmate and repetition terminal
codes. Clock, resignation, agreement, declaration, move-limit, and other tournament-dependent
codes are retained with an explicit external-condition status rather than presented as
rules-proven outcomes.

## Repetition and perpetual check

The initial state and every state after a move are retained by `Game`. Four occurrences of the
same board, both hands, and side to move produce repetition.

For the interval from the first through fourth relevant occurrence, each side's moves are
examined. If every move by one side gave check, that checking side loses by the
continuous-check rule. Otherwise the result is a repetition no-contest. When a longer history
contains more occurrences, the latest four occurrences ending at the current state are used;
this algorithmic choice is fixed by tests because the official rule defines the result but not
an implementation procedure. A CSA record may legally continue after a repetition claim would
have become available; record replay therefore retains history without forcibly terminating and
evaluates repetition only at the terminal statement being validated.

The JSA rules also permit both players to agree to restart during a repeating sequence before
the fourth occurrence. Because CSA text alone cannot prove that agreement, such a
`SENNICHITE` record is retained as an external-condition result rather than rejected or marked
as replay-verified.

## Entering king and impasse

Entering-king adjudication is a selectable rule profile rather than an implicit legal-move
side effect.

The `JsaOfficial2025` declaration profile requires:

1. the game to be active, the declaring side to have the move, and fewer than 500 moves to
   have been completed;
2. the declaring king to be in the enemy camp;
3. at least ten other declaring-side pieces in the enemy camp;
4. the declaring king not to be in check; and
5. points counted from declaring-side pieces in the enemy camp plus pieces in hand.

Rooks and bishops count five points, other non-king pieces one. At least 31 points is a win;
24–30 is a no-contest/replay; failure of any condition is a declaration loss. A configurable
profile boundary permits tournament-specific 27-point rules without changing move legality.

The JSA mutual-agreement 24-point impasse rule and the 500-move no-contest are competition
adjudications. Both apply only after at least one king has entered and the players or arbiter
have established that neither side has a reasonable mating prospect. Because the core cannot
infer that human/tournament judgment, callers must pass an explicit `ImpasseCondition`.
Phase 1 never automatically terminates a game based on an inferred agreement. The entering-
king condition for the 500-move boundary is evaluated on the stored position immediately after
move 500; a delayed end to continuous check does not shift that boundary.

## Perft policy

No authoritative, versioned shogi perft corpus was located during Phase 1 research. The
project therefore does not claim community perft constants as independent truth. `perft`
traverses this implementation deterministically and is tested against:

- the exact generated root move set for hand-auditable fixtures;
- make/unmake and hash invariants at every traversed edge; and
- focused legal-move fixtures derived from the official movement and illegality rules.

Any future external perft baseline must record its source, rule profile, position, and whether
pawn-drop mate and repetition are included before a value is accepted.

Perft counters are exact: overflow returns an error instead of saturating. The CLI applies a
depth-five runtime limit to avoid accidentally launching an impractical exhaustive traversal.
