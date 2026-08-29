from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from open_shogi_training.phase10r_model import VARIANT_PAIR
from open_shogi_training.phase10r_run import _parser
from open_shogi_training.phase10r_teacher_binding import (
    Phase10RTeacherBindingError,
    _canonical_sfen,
    _fit_positive_affine,
    _score_mapping,
)

INITIAL_SFEN = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"


def test_teacher_binding_commands_have_the_frozen_cli_shape() -> None:
    prepare = _parser().parse_args(["prepare-teacher-binding", "--root", ".", "--scale", "1m"])
    calibrate = _parser().parse_args(
        [
            "calibrate-teacher",
            "--root",
            ".",
            "--scale",
            "1m",
            "--variant",
            "sparse-pair-policy-wdl",
            "--resume",
        ]
    )

    assert prepare.command == "prepare-teacher-binding"
    assert prepare.scale == "1m"
    assert calibrate.command == "calibrate-teacher"
    assert calibrate.resume is True
    assert calibrate.variant == "sparse-pair-policy-wdl"


def test_canonical_sfen_and_score_namespace_are_strict() -> None:
    assert _canonical_sfen(INITIAL_SFEN) == INITIAL_SFEN

    with pytest.raises(Phase10RTeacherBindingError, match="zero mate distance"):
        _score_mapping({"kind": "mate", "value": 0}, "score")

    with pytest.raises(Phase10RTeacherBindingError, match="non-mate namespace"):
        _score_mapping({"kind": "cp", "value": 29_000}, "score")


def test_stage4_affine_fit_is_positive_and_finite(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("open_shogi_training.phase10r_teacher_binding.STAGE4_CP_ROWS", 2)

    class StubModel:
        variant_id = VARIANT_PAIR

        def forward_example(self, example: SimpleNamespace) -> dict[str, torch.Tensor]:
            raw_logit = float(example.raw_targets["raw_logit"])
            return {"values": torch.tensor([0.0, 0.0, raw_logit])}

    model = StubModel()
    examples = [
        SimpleNamespace(
                raw_targets={
                    "raw_logit": -1.0,
                    "teacher_binding": {
                        "score": {"kind": "cp", "value": 100},
                        "candidates": [{}],
                    },
                }
            ),
            SimpleNamespace(
                raw_targets={
                    "raw_logit": 1.0,
                    "teacher_binding": {
                        "score": {"kind": "cp", "value": 300},
                        "candidates": [{}],
                    },
                }
            ),
    ]

    scale, bias, details = _fit_positive_affine(model, examples, VARIANT_PAIR)

    assert scale > 0.0
    assert bias == pytest.approx(200.0, abs=1.0)
    assert details["fit_rows"] == 2
