"""Typed, observed-only Alpha-Beta leaf targets for Phase 10V.

All scalar labels and factual outcomes are from the evaluated position's side to move.
Candidate teacher scores are from the parent side; successor predictions must be negated.
Mate labels stay symbolic and never enter the scalar centipawn objective.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from open_shogi_training.phase10r_model import parse_sfen
from open_shogi_training.phase10u_execution import successor_sfen

TARGET_SCHEMA = "open_shogiai_phase10v_leaf_targets/v1"
LEAF_KINDS = frozenset(
    {
        "quiescence_leaf",
        "pv_leaf",
        "quiet_middlegame",
        "tactical_middlegame",
        "mate_boundary",
        "endgame",
        "crossplay_leaf",
        "approved_external",
        "teacher_candidate",
    }
)
MAX_CP = 20_000
MAX_OBSERVED_CP = 28_999
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_MOVE = re.compile(r"(?:[1-9][a-i][1-9][a-i]\+?|[PLNSGBR]\*[1-9][a-i])\Z")


@dataclass(frozen=True, slots=True)
class TeacherScore:
    kind: str
    value: int

    def __post_init__(self) -> None:
        if self.kind not in {"cp", "mate"} or type(self.value) is not int:
            raise ValueError("teacher score must be a typed integer cp or mate")
        if (self.kind == "cp" and abs(self.value) > MAX_OBSERVED_CP) or (
            self.kind == "mate" and (self.value == 0 or abs(self.value) > 2**31 - 1)
        ):
            raise ValueError("score outside scalar/mate namespace")

    @property
    def order(self) -> tuple[int, int]:
        if self.kind == "cp":
            return 1, self.value
        return (2, -self.value) if self.value > 0 else (0, -self.value)

    def reversed(self) -> TeacherScore:
        return TeacherScore(self.kind, -self.value)


@dataclass(frozen=True, slots=True)
class CandidateTarget:
    move: str
    child_sfen: str
    score: TeacherScore  # Parent side-to-move perspective, including mate sign.


@dataclass(frozen=True, slots=True)
class Phase10VExample:
    sfen: str
    score: TeacherScore | None
    wdl: int | None = None  # 0 loss, 1 draw, 2 win; factual outcome only.
    candidates: tuple[CandidateTarget, ...] = ()
    source: str = "diagnostic"
    source_game_id: str = "diagnostic"
    split: str = "train"
    leaf_kind: str = "quiet_middlegame"
    provenance_sha256: str = ""

    @property
    def cp(self) -> float | None:
        # Preserve the exact observed score in .score; search-range saturation is explicit.
        return (
            float(max(-MAX_CP, min(MAX_CP, self.score.value)))
            if self.score and self.score.kind == "cp"
            else None
        )

    @property
    def ranking_pairs(self) -> tuple[tuple[int, int], ...]:
        pairs = []
        for left in range(len(self.candidates)):
            for right in range(left + 1, len(self.candidates)):
                a, b = self.candidates[left].score.order, self.candidates[right].score.order
                if a != b:
                    pairs.append((left, right) if a > b else (right, left))
        return tuple(pairs)

    def validate(self, *, production: bool = True) -> None:
        parent = parse_sfen(self.sfen)
        if self.score is not None and not isinstance(self.score, TeacherScore):
            raise ValueError("untyped primary score")
        if self.wdl is not None and (type(self.wdl) is not int or self.wdl not in (0, 1, 2)):
            raise ValueError("invalid factual WDL")
        if self.split not in {"train", "validation"} or self.leaf_kind not in LEAF_KINDS:
            raise ValueError("unapproved split or leaf distribution category")
        if len(self.candidates) > 3 or len({c.move for c in self.candidates}) != len(
            self.candidates
        ):
            raise ValueError("only unique observed MultiPV candidates are permitted")
        for candidate in self.candidates:
            if not _MOVE.fullmatch(candidate.move) or not isinstance(candidate.score, TeacherScore):
                raise ValueError("malformed observed candidate")
            child = parse_sfen(candidate.child_sfen)
            if (
                child.side_to_move == parent.side_to_move
                or child.move_number != parent.move_number + 1
            ):
                raise ValueError("candidate successor must be exactly one ply from parent")
            replayed = parse_sfen(successor_sfen(self.sfen, candidate.move))
            if child.canonical_state != replayed.canonical_state:
                raise ValueError("candidate child differs from replayed move application")
        if production and (
            not self.source
            or self.source == "diagnostic"
            or not self.source_game_id
            or not _HASH.fullmatch(self.provenance_sha256)
        ):
            raise ValueError("production examples require source, game lineage and provenance hash")
        if self.score is None and self.wdl is None and not self.ranking_pairs:
            raise ValueError("row has neither a typed score nor an observed auxiliary target")


def observed_targets(example: Phase10VExample) -> dict[str, Any]:
    """Explicit masks, with no padding, pseudo-cp for mate, or fabricated candidate."""
    return {
        "schema": TARGET_SCHEMA,
        "value_cp": example.cp,
        "observed_value_cp": (
            example.score.value if example.score and example.score.kind == "cp" else None
        ),
        "scalar_target_saturated": bool(
            example.score and example.score.kind == "cp" and abs(example.score.value) > MAX_CP
        ),
        "value_mask": example.cp is not None,
        "wdl": example.wdl,
        "wdl_mask": example.wdl is not None,
        "mate_metadata": (
            {"kind": "mate", "value": example.score.value}
            if example.score and example.score.kind == "mate"
            else None
        ),
        "observed_candidate_count": len(example.candidates),
        "candidate_mask": [i < len(example.candidates) for i in range(3)],
        "ranking_pairs": [list(pair) for pair in example.ranking_pairs],
        "ranking_mask": bool(example.ranking_pairs),
        "policy_mask": False,
    }


def root_score_from_child(child_score: float) -> float:
    if not math.isfinite(child_score):
        raise ValueError("nonfinite child prediction")
    return -child_score


def example_from_mapping(row: Mapping[str, Any], *, expected_split: str) -> Phase10VExample:
    if row.get("schema") != TARGET_SCHEMA or row.get("split") != expected_split:
        raise ValueError("target schema or split mismatch; final holdout is forbidden")
    if row.get("score_perspective") != "side_to_move":
        raise ValueError("explicit side-to-move score perspective required")
    if row.get("wdl") is not None and row.get("wdl_source") != "factual_outcome":
        raise ValueError("WDL must be an observed factual outcome")
    provenance = row.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("approved") is not True:
        raise ValueError("approved source/leaf provenance required")
    if provenance.get("original_split") != expected_split:
        raise ValueError("original source split cannot be reassigned")
    if provenance.get("leaf_kind") in {"quiescence_leaf", "pv_leaf", "crossplay_leaf"}:
        for key in ("search_receipt_sha256", "root_position_sha256", "engine_model_sha256"):
            if not isinstance(provenance.get(key), str) or not _HASH.fullmatch(provenance[key]):
                raise ValueError(f"search leaf missing {key}")
    candidates = []
    for candidate in row.get("candidates", []):
        if candidate.get("score_perspective") != "parent_side_to_move":
            raise ValueError("candidate score perspective must be parent_side_to_move")
        if not _HASH.fullmatch(str(candidate.get("replay_receipt_sha256", ""))):
            raise ValueError("candidate requires legal successor replay receipt")
        candidates.append(
            CandidateTarget(
                candidate["move"], candidate["child_sfen"], TeacherScore(**candidate["score"])
            )
        )
    encoded = json.dumps(provenance, sort_keys=True, separators=(",", ":")).encode()
    example = Phase10VExample(
        sfen=row["sfen"],
        score=TeacherScore(**row["score"]) if row.get("score") else None,
        wdl=row.get("wdl"),
        candidates=tuple(candidates),
        source=provenance["source"],
        source_game_id=provenance["source_game_id"],
        split=row["split"],
        leaf_kind=provenance["leaf_kind"],
        provenance_sha256=hashlib.sha256(encoded).hexdigest(),
    )
    example.validate()
    return example


def iter_examples(path: Path, *, expected_split: str = "train") -> Iterator[Phase10VExample]:
    """Stream an explicitly approved split, rejecting malformed rows with line context."""
    if expected_split not in {"train", "validation"}:
        raise ValueError("only train and development validation are accessible")
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield example_from_mapping(json.loads(line), expected_split=expected_split)
            except (ValueError, KeyError, TypeError) as error:
                raise ValueError(f"{path}:{number}: {error}") from error
