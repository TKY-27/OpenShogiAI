"""Closed, file-level rights registry for the Phase 10R data foundation.

This registry is intentionally separate from the frozen Phase 3/10 registry.
Phase 10R may inventory many public artifacts, but an artifact is never
trainable merely because it appears in a public URL.  Every entry carries an
explicit decision and the four permission dimensions required by the data
foundation plan.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final
from urllib.parse import urlsplit

import yaml

from open_shogi_training.data.registry import UniqueSafeLoader

PHASE10R_REGISTRY_SCHEMA: Final = 1
DECISION_STATES: Final = frozenset(
    {
        "approved",
        "approved_local_only",
        "pending_permission",
        "denied",
        "reserved_holdout",
    }
)
PERMISSION_FIELDS: Final = (
    "training_permission",
    "derived_weights_permission",
    "redistribution_permission",
    "commercial_use_permission",
)
_IDENTIFIER_RE = re.compile(r"[a-z0-9][a-z0-9_.-]{0,127}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_TOP_LEVEL_KEYS = frozenset(
    {"schema_version", "registry_id", "as_of", "policy", "sources", "artifacts"}
)
_POLICY_KEYS = frozenset(
    {
        "data_root_env",
        "default_data_root",
        "minimum_free_bytes",
        "max_single_download_bytes",
        "max_parallel_downloads",
        "max_requests_per_second",
        "raw_data_ignored",
        "normalized_data_ignored",
        "checkpoint_policy",
    }
)
_SOURCE_KEYS = frozenset(
    {
        "source_id",
        "publisher",
        "official_page",
        "state",
        "reason",
        "rights_scope",
        "artifact_ids",
    }
)
_ARTIFACT_KEYS = frozenset(
    {
        "artifact_id",
        "source_id",
        "artifact_kind",
        "exactness",
        "exact_artifact_name",
        "source_revision",
        "source_url",
        "official_page",
        "publisher",
        "license",
        "license_evidence",
        "state",
        "training_permission",
        "derived_weights_permission",
        "redistribution_permission",
        "commercial_use_permission",
        "attribution",
        "size_bytes",
        "sha256",
        "format",
        "compression",
        "availability",
        "reason",
        "holdout_role",
        "sample_plan",
    }
)


class Phase10RRegistryError(ValueError):
    """Raised when the Phase 10R registry is incomplete or ambiguous."""


@dataclass(frozen=True, slots=True)
class Phase10RSource:
    source_id: str
    publisher: str
    official_page: str
    state: str
    reason: str
    rights_scope: str
    artifact_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Phase10RArtifact:
    artifact_id: str
    source_id: str
    artifact_kind: str
    exactness: str
    exact_artifact_name: str
    source_revision: str
    source_url: str
    official_page: str
    publisher: str
    license: str
    license_evidence: tuple[dict[str, str], ...]
    state: str
    training_permission: str
    derived_weights_permission: str
    redistribution_permission: str
    commercial_use_permission: str
    attribution: str
    size_bytes: int | None
    sha256: str | None
    format: str
    compression: str
    availability: str
    reason: str
    holdout_role: str
    sample_plan: str

    @property
    def permissions(self) -> dict[str, str]:
        return {field: getattr(self, field) for field in PERMISSION_FIELDS}

    @property
    def trainable(self) -> bool:
        return self.state == "approved" and self.training_permission == "approved"

    @property
    def local_only(self) -> bool:
        return self.state == "approved_local_only"

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "source_id": self.source_id,
            "artifact_kind": self.artifact_kind,
            "exactness": self.exactness,
            "exact_artifact_name": self.exact_artifact_name,
            "source_revision": self.source_revision,
            "source_url": self.source_url,
            "official_page": self.official_page,
            "publisher": self.publisher,
            "license": self.license,
            "license_evidence": [dict(item) for item in self.license_evidence],
            "state": self.state,
            **self.permissions,
            "attribution": self.attribution,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "format": self.format,
            "compression": self.compression,
            "availability": self.availability,
            "reason": self.reason,
            "holdout_role": self.holdout_role,
            "sample_plan": self.sample_plan,
        }


@dataclass(frozen=True, slots=True)
class Phase10RRegistry:
    registry_id: str
    as_of: str
    policy: dict[str, Any]
    sources: tuple[Phase10RSource, ...]
    artifacts: tuple[Phase10RArtifact, ...]
    sha256: str

    def source(self, source_id: str) -> Phase10RSource:
        for source in self.sources:
            if source.source_id == source_id:
                return source
        raise Phase10RRegistryError(f"unknown Phase 10R source: {source_id}")

    def artifact(self, artifact_id: str) -> Phase10RArtifact:
        for artifact in self.artifacts:
            if artifact.artifact_id == artifact_id:
                return artifact
        raise Phase10RRegistryError(f"unknown Phase 10R artifact: {artifact_id}")

    def approved_artifacts(self) -> tuple[Phase10RArtifact, ...]:
        return tuple(artifact for artifact in self.artifacts if artifact.trainable)

    def by_state(self) -> dict[str, int]:
        counts = {state: 0 for state in sorted(DECISION_STATES)}
        for artifact in self.artifacts:
            counts[artifact.state] += 1
        return counts

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PHASE10R_REGISTRY_SCHEMA,
            "registry_id": self.registry_id,
            "as_of": self.as_of,
            "policy": self.policy,
            "sources": [
                {
                    "source_id": source.source_id,
                    "publisher": source.publisher,
                    "official_page": source.official_page,
                    "state": source.state,
                    "reason": source.reason,
                    "rights_scope": source.rights_scope,
                    "artifact_ids": list(source.artifact_ids),
                }
                for source in self.sources
            ],
            "artifacts": [artifact.as_dict() for artifact in self.artifacts],
        }


def load_phase10r_registry(path: Path) -> Phase10RRegistry:
    """Load and validate a closed Phase 10R registry."""

    raw = path.read_bytes()
    if len(raw) > 4 * 1024 * 1024:
        raise Phase10RRegistryError("Phase 10R registry exceeds 4 MiB")
    try:
        payload = yaml.load(raw.decode("utf-8"), Loader=UniqueSafeLoader)
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise Phase10RRegistryError(f"invalid Phase 10R YAML: {error}") from error
    if not isinstance(payload, Mapping):
        raise Phase10RRegistryError("Phase 10R registry must be a mapping")
    _require_keys(payload, _TOP_LEVEL_KEYS, "registry")
    if payload.get("schema_version") != PHASE10R_REGISTRY_SCHEMA:
        raise Phase10RRegistryError("unsupported Phase 10R registry schema")
    registry_id = _identifier(payload.get("registry_id"), "registry_id")
    as_of = _string(payload.get("as_of"), "as_of")
    policy = _mapping(payload.get("policy"), "policy")
    _require_keys(policy, _POLICY_KEYS, "policy")
    for key in ("data_root_env", "default_data_root", "checkpoint_policy"):
        _string(policy.get(key), f"policy.{key}")
    for key in ("minimum_free_bytes", "max_single_download_bytes", "max_parallel_downloads"):
        value = policy.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise Phase10RRegistryError(f"policy.{key} must be a positive integer")
    requests = policy.get("max_requests_per_second")
    if not isinstance(requests, (float, int)) or isinstance(requests, bool) or requests <= 0:
        raise Phase10RRegistryError("policy.max_requests_per_second must be positive")
    for key in ("raw_data_ignored", "normalized_data_ignored"):
        if policy.get(key) is not True:
            raise Phase10RRegistryError(f"policy.{key} must be true")

    sources_raw = payload.get("sources")
    artifacts_raw = payload.get("artifacts")
    if not isinstance(sources_raw, Sequence) or isinstance(sources_raw, (str, bytes)):
        raise Phase10RRegistryError("sources must be a sequence")
    if not isinstance(artifacts_raw, Sequence) or isinstance(artifacts_raw, (str, bytes)):
        raise Phase10RRegistryError("artifacts must be a sequence")

    sources = tuple(_parse_source(item, index) for index, item in enumerate(sources_raw))
    artifacts = tuple(_parse_artifact(item, index) for index, item in enumerate(artifacts_raw))
    source_ids = {source.source_id for source in sources}
    artifact_ids = {artifact.artifact_id for artifact in artifacts}
    if len(source_ids) != len(sources):
        raise Phase10RRegistryError("duplicate source_id")
    if len(artifact_ids) != len(artifacts):
        raise Phase10RRegistryError("duplicate artifact_id")
    for source in sources:
        if source.state not in DECISION_STATES:
            raise Phase10RRegistryError(f"invalid source state: {source.state}")
        if any(artifact_id not in artifact_ids for artifact_id in source.artifact_ids):
            raise Phase10RRegistryError(f"source {source.source_id} references unknown artifact")
    for artifact in artifacts:
        if artifact.source_id not in source_ids:
            raise Phase10RRegistryError(
                f"artifact {artifact.artifact_id} references unknown source"
            )
        if artifact.state not in DECISION_STATES:
            raise Phase10RRegistryError(f"invalid artifact state: {artifact.state}")
        if artifact.state == "approved" and artifact.training_permission != "approved":
            raise Phase10RRegistryError(
                f"approved artifact {artifact.artifact_id} lacks approved training permission"
            )
        if (
            artifact.state in {"denied", "reserved_holdout"}
            and artifact.training_permission == "approved"
        ):
            raise Phase10RRegistryError(
                f"non-trainable artifact {artifact.artifact_id} has approved training permission"
            )
    listed = {artifact_id for source in sources for artifact_id in source.artifact_ids}
    if listed != artifact_ids:
        missing = sorted(artifact_ids - listed)
        extra = sorted(listed - artifact_ids)
        raise Phase10RRegistryError(
            f"source/artifact index mismatch: missing={missing}, extra={extra}"
        )
    canonical = yaml.safe_dump(
        {
            "schema_version": PHASE10R_REGISTRY_SCHEMA,
            "registry_id": registry_id,
            "as_of": as_of,
            "policy": policy,
            "sources": [
                {
                    "source_id": source.source_id,
                    "publisher": source.publisher,
                    "official_page": source.official_page,
                    "state": source.state,
                    "reason": source.reason,
                    "rights_scope": source.rights_scope,
                    "artifact_ids": list(source.artifact_ids),
                }
                for source in sources
            ],
            "artifacts": [artifact.as_dict() for artifact in artifacts],
        },
        allow_unicode=True,
        sort_keys=True,
    ).encode("utf-8")
    return Phase10RRegistry(
        registry_id=registry_id,
        as_of=as_of,
        policy=dict(policy),
        sources=sources,
        artifacts=artifacts,
        sha256=hashlib.sha256(canonical).hexdigest(),
    )


def _parse_source(value: Any, index: int) -> Phase10RSource:
    mapping = _mapping(value, f"sources[{index}]")
    _require_keys(mapping, _SOURCE_KEYS, f"sources[{index}]")
    artifact_ids = mapping.get("artifact_ids")
    if not isinstance(artifact_ids, Sequence) or isinstance(artifact_ids, (str, bytes)):
        raise Phase10RRegistryError(f"sources[{index}].artifact_ids must be a sequence")
    return Phase10RSource(
        source_id=_identifier(mapping.get("source_id"), f"sources[{index}].source_id"),
        publisher=_string(mapping.get("publisher"), f"sources[{index}].publisher"),
        official_page=_url(mapping.get("official_page"), f"sources[{index}].official_page"),
        state=_decision(mapping.get("state"), f"sources[{index}].state"),
        reason=_string(mapping.get("reason"), f"sources[{index}].reason"),
        rights_scope=_string(mapping.get("rights_scope"), f"sources[{index}].rights_scope"),
        artifact_ids=tuple(
            _identifier(item, f"sources[{index}].artifact_ids[{item_index}]")
            for item_index, item in enumerate(artifact_ids)
        ),
    )


def _parse_artifact(value: Any, index: int) -> Phase10RArtifact:
    mapping = _mapping(value, f"artifacts[{index}]")
    _require_keys(mapping, _ARTIFACT_KEYS, f"artifacts[{index}]")
    evidence_raw = mapping.get("license_evidence")
    if not isinstance(evidence_raw, Sequence) or isinstance(evidence_raw, (str, bytes)):
        raise Phase10RRegistryError(f"artifacts[{index}].license_evidence must be a sequence")
    evidence: list[dict[str, str]] = []
    for evidence_index, item in enumerate(evidence_raw):
        evidence_mapping = _mapping(item, f"artifacts[{index}].license_evidence[{evidence_index}]")
        if set(evidence_mapping) != {"url", "quote"}:
            raise Phase10RRegistryError(
                f"artifacts[{index}].license_evidence[{evidence_index}] requires only url and quote"
            )
        evidence.append(
            {
                "url": _url(evidence_mapping.get("url"), "license evidence url"),
                "quote": _string(evidence_mapping.get("quote"), "license evidence quote"),
            }
        )
    size = mapping.get("size_bytes")
    if size is not None and (not isinstance(size, int) or isinstance(size, bool) or size < 0):
        raise Phase10RRegistryError(
            f"artifacts[{index}].size_bytes must be null or non-negative integer"
        )
    sha256 = mapping.get("sha256")
    if sha256 is not None and (not isinstance(sha256, str) or not _SHA256_RE.fullmatch(sha256)):
        raise Phase10RRegistryError(
            f"artifacts[{index}].sha256 must be a lowercase SHA-256 or null"
        )
    return Phase10RArtifact(
        artifact_id=_identifier(mapping.get("artifact_id"), f"artifacts[{index}].artifact_id"),
        source_id=_identifier(mapping.get("source_id"), f"artifacts[{index}].source_id"),
        artifact_kind=_string(mapping.get("artifact_kind"), "artifact_kind"),
        exactness=_string(mapping.get("exactness"), "exactness"),
        exact_artifact_name=_string(mapping.get("exact_artifact_name"), "exact_artifact_name"),
        source_revision=_string(mapping.get("source_revision"), "source_revision"),
        source_url=_url(mapping.get("source_url"), "source_url"),
        official_page=_url(mapping.get("official_page"), "official_page"),
        publisher=_string(mapping.get("publisher"), "publisher"),
        license=_string(mapping.get("license"), "license"),
        license_evidence=tuple(evidence),
        state=_decision(mapping.get("state"), "state"),
        training_permission=_decision(mapping.get("training_permission"), "training_permission"),
        derived_weights_permission=_decision(
            mapping.get("derived_weights_permission"), "derived_weights_permission"
        ),
        redistribution_permission=_decision(
            mapping.get("redistribution_permission"), "redistribution_permission"
        ),
        commercial_use_permission=_decision(
            mapping.get("commercial_use_permission"), "commercial_use_permission"
        ),
        attribution=_string(mapping.get("attribution"), "attribution"),
        size_bytes=size,
        sha256=sha256,
        format=_string(mapping.get("format"), "format"),
        compression=_string(mapping.get("compression"), "compression"),
        availability=_string(mapping.get("availability"), "availability"),
        reason=_string(mapping.get("reason"), "reason"),
        holdout_role=_string(mapping.get("holdout_role"), "holdout_role"),
        sample_plan=_string(mapping.get("sample_plan"), "sample_plan"),
    )


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Phase10RRegistryError(f"{field} must be a mapping")
    return value


def _require_keys(value: Mapping[str, Any], allowed: frozenset[str], field: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise Phase10RRegistryError(f"{field} has unknown keys: {unknown}")


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise Phase10RRegistryError(f"{field} must be a non-empty string")
    return value


def _identifier(value: Any, field: str) -> str:
    result = _string(value, field)
    if not _IDENTIFIER_RE.fullmatch(result):
        raise Phase10RRegistryError(f"{field} is not a safe identifier")
    return result


def _decision(value: Any, field: str) -> str:
    result = _string(value, field)
    if result not in DECISION_STATES:
        raise Phase10RRegistryError(f"{field} must be one of {sorted(DECISION_STATES)}")
    return result


def _url(value: Any, field: str) -> str:
    result = _string(value, field)
    parts = urlsplit(result)
    if (
        parts.scheme not in {"http", "https"}
        or not parts.netloc
        or parts.username
        or parts.password
    ):
        raise Phase10RRegistryError(f"{field} must be an HTTP(S) URL without credentials")
    return result


__all__ = [
    "DECISION_STATES",
    "PERMISSION_FIELDS",
    "PHASE10R_REGISTRY_SCHEMA",
    "Phase10RArtifact",
    "Phase10RRegistry",
    "Phase10RRegistryError",
    "Phase10RSource",
    "load_phase10r_registry",
]
