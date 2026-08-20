"""Phase 4 black-box USI teacher labeling."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from open_shogi_training.labeling.config import TeacherConfig, load_teacher_config
from open_shogi_training.labeling.schema import (
    LABEL_SCHEMA,
    PARSER_VERSION,
    SCORE_POV,
    iter_teacher_labels,
    validate_label_record,
    validate_score,
)
from open_shogi_training.labeling.selection import SelectionResult, select_positions
from open_shogi_training.labeling.usi import (
    USICandidate,
    USIEngine,
    USIScore,
    USISearchResult,
)

if TYPE_CHECKING:
    from open_shogi_training.labeling.pipeline import LabelAuditResult, LabelingResult


def __getattr__(name: str) -> Any:
    """Load pipeline exports lazily so low-level artifact helpers stay acyclic."""

    if name in {"LabelAuditResult", "LabelingResult", "audit_labeling_output", "run_labeling"}:
        from open_shogi_training.labeling import pipeline

        return getattr(pipeline, name)
    raise AttributeError(name)


__all__ = [
    "LABEL_SCHEMA",
    "PARSER_VERSION",
    "SCORE_POV",
    "LabelAuditResult",
    "LabelingResult",
    "SelectionResult",
    "TeacherConfig",
    "USICandidate",
    "USIEngine",
    "USIScore",
    "USISearchResult",
    "audit_labeling_output",
    "iter_teacher_labels",
    "load_teacher_config",
    "run_labeling",
    "select_positions",
    "validate_label_record",
    "validate_score",
]
