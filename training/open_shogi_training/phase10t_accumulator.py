"""Independent, random-only Phase 10T accumulator correctness/cost prototype.

This is not an engine evaluator or trained model. Feature-set differencing provides a
reference oracle for future move-local updates, not an O(1) native implementation.
Rule adjudication, mate values, score calibration and quantization are out of scope.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from open_shogi_training.phase10r_model import HistoryFacts, ParsedPosition, parse_sfen

BOARD_FEATURES = 17 * 17 * 28
KING_OFFSET = BOARD_FEATURES
HAND_OFFSET = KING_OFFSET + 81
HISTORY_OFFSET = HAND_OFFSET + 2 * 7 * 18
FEATURE_COUNT = HISTORY_OFFSET + 8
ACCUMULATOR_WIDTH = 128
HIDDEN_WIDTH = 32
SEED = 20260907
DEFAULT_HISTORY = HistoryFacts()


def king_square(position: ParsedPosition, perspective: int) -> int:
    return next(p.square for p in position.board if p and p.kind == 7 and p.side == perspective)


def feature_ids(
    position: ParsedPosition, perspective: int, history: HistoryFacts = DEFAULT_HISTORY
) -> frozenset[int]:
    """No hashing/collisions: rotate White 180 degrees and encode relative ownership."""
    if perspective not in (0, 1):
        raise ValueError("perspective must be Black or White")
    history.validate()
    king = king_square(position, perspective)
    orient = lambda square: 80 - square if perspective else square  # noqa: E731
    kr, kc = divmod(orient(king), 9)
    result = {KING_OFFSET + orient(king)}
    for piece in position.board:
        if piece is not None:
            rank, column = divmod(orient(piece.square), 9)
            offset = (rank - kr + 8) * 17 + column - kc + 8
            result.add(offset * 28 + (piece.side ^ perspective) * 14 + piece.kind)
    for side, hand in enumerate(position.hands):
        for kind, count in enumerate(hand):
            for ordinal in range(count):
                result.add(HAND_OFFSET + ((side ^ perspective) * 7 + kind) * 18 + ordinal)
    result.add(HISTORY_OFFSET + int(history.available))
    result.add(HISTORY_OFFSET + 2 + history.repetition_count - 1)
    checks = (history.continuous_check_by_us, history.continuous_check_by_them)
    own_check, other_check = checks if perspective == position.side_to_move else checks[::-1]
    if own_check:
        result.add(HISTORY_OFFSET + 6)
    if other_check:
        result.add(HISTORY_OFFSET + 7)
    return frozenset(result)


@dataclass(frozen=True)
class State:
    position: ParsedPosition
    features: tuple[frozenset[int], frozenset[int]]
    values: np.ndarray


@dataclass(frozen=True)
class Update:
    """Undo retains exact parent values; it never reverses floating-point additions."""

    before: State
    after: State
    rebuilt: tuple[bool, bool]
    vector_additions: int


class RandomAccumulator:
    """Random weights only; every non-terminal output depends on learned parameters."""

    def __init__(self, seed: int = SEED) -> None:
        rng = np.random.default_rng(seed)

        def random(shape: tuple[int, ...]) -> np.ndarray:
            return rng.normal(0, 0.01, shape).astype(np.float32)

        self.table = random((FEATURE_COUNT, ACCUMULATOR_WIDTH))
        self.bias = random((ACCUMULATOR_WIDTH,))
        self.hidden_weight = random((2 * ACCUMULATOR_WIDTH, HIDDEN_WIDTH))
        self.hidden_bias = random((HIDDEN_WIDTH,))
        self.head_weight = random((HIDDEN_WIDTH, 4))
        self.head_bias = random((4,))

    def _sum(self, features: frozenset[int]) -> np.ndarray:
        value = self.bias.copy()
        for index in sorted(features):
            value += self.table[index]
        return value

    def refresh(self, sfen: str, history: HistoryFacts = DEFAULT_HISTORY) -> State:
        position = parse_sfen(sfen)
        features = tuple(feature_ids(position, side, history) for side in (0, 1))
        return State(position, features, np.stack([self._sum(f) for f in features]))

    def update(self, before: State, sfen: str, history: HistoryFacts = DEFAULT_HISTORY) -> Update:
        position = parse_sfen(sfen)
        features = tuple(feature_ids(position, side, history) for side in (0, 1))
        values = before.values.copy()
        rebuilt = tuple(
            king_square(before.position, side) != king_square(position, side) for side in (0, 1)
        )
        additions = 0
        for side in (0, 1):
            if rebuilt[side]:
                values[side] = self._sum(features[side])
                additions += len(features[side])
            else:
                removed = before.features[side] - features[side]
                added = features[side] - before.features[side]
                for index in sorted(removed):
                    values[side] -= self.table[index]
                for index in sorted(added):
                    values[side] += self.table[index]
                additions += len(removed) + len(added)
        return Update(before, State(position, features, values), rebuilt, additions)

    @staticmethod
    def unmake(update: Update) -> State:
        return update.before

    def evaluate(self, state: State) -> tuple[float, np.ndarray]:
        """Return uncalibrated learned score and W/D/L logits, both STM-relative."""
        stm = state.position.side_to_move
        inputs = np.maximum(state.values[[stm, 1 - stm]].reshape(-1), 0)
        hidden = np.maximum(inputs @ self.hidden_weight + self.hidden_bias, 0)
        output = hidden @ self.head_weight + self.head_bias
        return float(output[0]), output[1:].copy()

    def costs(self) -> dict[str, int]:
        parameters = sum(
            array.size
            for array in (
                self.table,
                self.bias,
                self.hidden_weight,
                self.hidden_bias,
                self.head_weight,
                self.head_bias,
            )
        )
        return {
            "features": FEATURE_COUNT,
            "parameters": parameters,
            "float32_parameter_bytes": parameters * 4,
            "two_accumulator_float32_bytes": 2 * ACCUMULATOR_WIDTH * 4,
            "head_multiply_accumulates_per_leaf": 256 * 32 + 32 * 4,
            "scalar_additions_per_feature_delta": ACCUMULATOR_WIDTH,
        }


# Constructed semantic fixtures only, with both kings present; no corpus/holdout reads.
TRANSITIONS = {
    "quiet": ("4k4/9/9/9/4P4/9/9/9/4K4 b - 1", "4k4/9/9/4P4/9/9/9/9/4K4 w - 2"),
    "capture": ("4k4/9/9/4p4/4P4/9/9/9/4K4 b - 1", "4k4/9/9/4P4/9/9/9/9/4K4 w P 2"),
    "promotion": ("4k4/9/9/4P4/9/9/9/9/4K4 b - 1", "4k4/9/4+P4/9/9/9/9/9/4K4 w - 2"),
    "drop": ("4k4/9/9/9/9/9/9/9/4K4 b P 1", "4k4/9/9/9/4P4/9/9/9/4K4 w - 2"),
    "king": ("4k4/9/9/9/4P4/9/9/9/4K4 b - 1", "4k4/9/9/9/4P4/9/9/4K4/9 w - 2"),
}


def probe() -> dict[str, object]:
    model = RandomAccumulator()
    transitions = {}
    for name, (before_sfen, after_sfen) in TRANSITIONS.items():
        before = model.refresh(before_sfen)
        update = model.update(before, after_sfen)
        expected = model.refresh(after_sfen)
        transitions[name] = {
            "maximum_absolute_error": float(np.max(np.abs(update.after.values - expected.values))),
            "feature_sets_equal": update.after.features == expected.features,
            "exact_unmake": bool(np.array_equal(model.unmake(update).values, before.values)),
            "rebuilt_perspectives": list(update.rebuilt),
            "delta_vector_additions": update.vector_additions,
            "full_refresh_vector_additions": sum(map(len, expected.features)),
        }
    return {
        "schema": "open_shogiai_phase10t_accumulator_probe/v1",
        "seed": SEED,
        "weights": "random-only-untrained",
        "native_runtime_or_strength_proven": False,
        "cost_scope": "arithmetic counts exclude feature extraction, memory traffic and search",
        "costs": model.costs(),
        "transitions": transitions,
    }


if __name__ == "__main__":
    print(json.dumps(probe(), indent=2, sort_keys=True))
