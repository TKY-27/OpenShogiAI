"""Strict, bounded loading for the audited data-source registry."""

from __future__ import annotations

import re
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Any, Final
from urllib.parse import unquote, urlsplit

import yaml
from yaml.composer import ComposerError
from yaml.constructor import ConstructorError
from yaml.events import AliasEvent
from yaml.nodes import MappingNode

REGISTRY_SCHEMA_VERSION: Final = 1
CATALOG_SCHEMA_VERSION: Final = 1
MAX_REGISTRY_BYTES: Final = 256 * 1024
MAX_CATALOG_BYTES: Final = 512 * 1024
MAX_CATALOG_OBJECTS: Final = 100
MAX_YAML_NODES: Final = 10_000
MAX_STRING_LENGTH: Final = 2_048
MAX_EVIDENCE_QUOTE_LENGTH: Final = 240
MAX_OBJECT_BYTES_LIMIT: Final = 16 * 1024 * 1024
MAX_CONCURRENCY: Final = 4
MAX_REQUESTS_PER_SECOND: Final = 2.0

_SOURCE_ID_RE = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}\Z")
_OBJECT_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_HOST_RE = re.compile(
    r"(?:localhost|(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)*"
    r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)\Z"
)

_ROOT_KEYS = frozenset({"schema_version", "sources"})
_SOURCE_KEYS = frozenset(
    {
        "source_id",
        "name",
        "official_base",
        "enabled",
        "approved",
        "license",
        "license_evidence",
        "robots_checked",
        "terms_checked",
        "max_requests_per_second",
        "concurrency",
        "allowed_paths",
        "denied_paths",
        "redistributable",
        "machine_learning_allowed",
        "last_reviewed",
        "allowed_hosts",
        "allow_insecure_http",
        "max_object_bytes",
        "user_agent",
        "catalog",
        "evidence_catalog",
        "adapter",
        "robots_url",
        "robots_policy",
    }
)
_EVIDENCE_KEYS = frozenset({"url", "local_path", "quote"})
_CATALOG_ROOT_KEYS = frozenset({"schema_version", "source_id", "objects"})
_CATALOG_OBJECT_KEYS = frozenset({"object_id", "url", "filename", "data_format", "compression"})
_EVIDENCE_CATALOG_ROOT_KEYS = frozenset({"schema_version", "source_id", "evidence"})
_EVIDENCE_OBJECT_KEYS = frozenset({"evidence_id", "url", "max_bytes", "sha256"})


class RegistryError(ValueError):
    """Raised when a registry or catalog violates its closed schema."""


class UniqueSafeLoader(yaml.SafeLoader):
    """SafeLoader variant that rejects aliases, duplicate keys, and node bombs."""

    _composed_nodes: int

    def __init__(self, stream: str) -> None:
        super().__init__(stream)
        self._composed_nodes = 0

    def compose_node(self, parent: Any, index: Any) -> Any:
        if self.check_event(AliasEvent):
            event = self.peek_event()
            raise ComposerError(
                "while composing YAML",
                None,
                "aliases are not allowed",
                event.start_mark,
            )
        self._composed_nodes += 1
        if self._composed_nodes > MAX_YAML_NODES:
            event = self.peek_event()
            raise ComposerError(
                "while composing YAML",
                None,
                f"document exceeds {MAX_YAML_NODES} nodes",
                event.start_mark,
            )
        return super().compose_node(parent, index)


def _construct_unique_mapping(
    loader: UniqueSafeLoader, node: MappingNode, *, deep: bool = False
) -> dict[Any, Any]:
    if not isinstance(node, MappingNode):
        raise ConstructorError(
            None,
            None,
            f"expected a mapping node, found {node.id}",
            node.start_mark,
        )

    seen: set[Hashable] = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, Hashable):
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable key",
                key_node.start_mark,
            )
        if key in seen:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        seen.add(key)
    return loader.construct_mapping(node, deep=deep)


UniqueSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


@dataclass(frozen=True, slots=True)
class LicenseEvidence:
    """Pinned evidence establishing the rights decision for one source."""

    url: str
    local_path: str
    quote: str

    def as_dict(self) -> dict[str, str]:
        return {"url": self.url, "local_path": self.local_path, "quote": self.quote}


@dataclass(frozen=True, slots=True)
class CatalogObject:
    """One explicitly enumerated object; catalog entries are never guessed."""

    object_id: str
    url: str
    filename: str
    data_format: str
    compression: str


@dataclass(frozen=True, slots=True)
class EvidenceObject:
    """One exact external page retained as a hashed local evidence snapshot."""

    evidence_id: str
    url: str
    max_bytes: int
    sha256: str | None = None


@dataclass(frozen=True, slots=True)
class DataSource:
    """Validated rights and operational policy for a data source."""

    source_id: str
    name: str
    official_base: str
    enabled: bool
    approved: bool
    license: str
    license_evidence: tuple[LicenseEvidence, ...]
    robots_checked: date
    terms_checked: date
    max_requests_per_second: float
    concurrency: int
    allowed_paths: tuple[str, ...]
    denied_paths: tuple[str, ...]
    redistributable: bool
    machine_learning_allowed: bool
    last_reviewed: date
    allowed_hosts: tuple[str, ...]
    allow_insecure_http: bool
    max_object_bytes: int
    user_agent: str
    catalog_path: str | None
    evidence_catalog_path: str | None
    adapter: str
    robots_url: str
    robots_policy: str
    catalog: tuple[CatalogObject, ...]
    evidence_catalog: tuple[EvidenceObject, ...]

    def validate_url(self, url: str, *, require_catalog_entry: bool = False) -> None:
        """Reject URLs outside the exact audited scheme, host, and path policy."""

        _validate_source_url(self, url)
        if require_catalog_entry and url not in {item.url for item in self.catalog}:
            raise RegistryError(f"URL is not present in the audited catalog: {url}")


@dataclass(frozen=True, slots=True)
class SourceRegistry:
    """Closed collection of audited source policies."""

    schema_version: int
    sources: tuple[DataSource, ...]

    def get(self, source_id: str) -> DataSource:
        for source in self.sources:
            if source.source_id == source_id:
                return source
        raise RegistryError(f"unknown source_id: {source_id}")


def load_source_registry(path: Path) -> SourceRegistry:
    """Load a bounded registry and each referenced bounded static catalog."""

    raw = _load_bounded_yaml(path, MAX_REGISTRY_BYTES)
    root = _require_mapping(raw, "registry")
    _require_exact_keys(root, _ROOT_KEYS, "registry")
    schema_version = _require_int(root, "schema_version")
    if schema_version != REGISTRY_SCHEMA_VERSION:
        raise RegistryError(f"unsupported registry schema_version: {schema_version}")

    raw_sources = _require_list(root, "sources")
    if not raw_sources:
        raise RegistryError("registry.sources must not be empty")

    project_root = path.resolve().parent.parent
    sources: list[DataSource] = []
    source_ids: set[str] = set()
    for index, raw_source in enumerate(raw_sources):
        source = _parse_source(
            raw_source,
            context=f"registry.sources[{index}]",
            project_root=project_root,
        )
        if source.source_id in source_ids:
            raise RegistryError(f"duplicate source_id: {source.source_id}")
        source_ids.add(source.source_id)
        sources.append(source)
    return SourceRegistry(schema_version=schema_version, sources=tuple(sources))


def _parse_source(raw: Any, *, context: str, project_root: Path) -> DataSource:
    table = _require_mapping(raw, context)
    _require_exact_keys(table, _SOURCE_KEYS, context)

    source_id = _require_string(table, "source_id", pattern=_SOURCE_ID_RE)
    name = _require_string(table, "name")
    official_base = _require_url(table, "official_base")
    enabled = _require_bool(table, "enabled")
    approved = _require_bool(table, "approved")
    license_name = _require_string(table, "license")
    license_evidence = _parse_evidence(
        _require_list(table, "license_evidence"),
        context=f"{context}.license_evidence",
        project_root=project_root,
    )
    robots_checked = _require_date(table, "robots_checked")
    terms_checked = _require_date(table, "terms_checked")
    max_requests_per_second = _require_float(table, "max_requests_per_second")
    concurrency = _require_int(table, "concurrency")
    allowed_paths = _require_path_prefixes(table, "allowed_paths")
    denied_paths = _require_path_prefixes(table, "denied_paths")
    redistributable = _require_bool(table, "redistributable")
    machine_learning_allowed = _require_bool(table, "machine_learning_allowed")
    last_reviewed = _require_date(table, "last_reviewed")
    allowed_hosts = _require_hosts(table, "allowed_hosts")
    allow_insecure_http = _require_bool(table, "allow_insecure_http")
    max_object_bytes = _require_int(table, "max_object_bytes")
    user_agent = _require_string(table, "user_agent")
    catalog_path = _require_optional_relative_path(table, "catalog")
    evidence_catalog_path = _require_optional_relative_path(table, "evidence_catalog")
    adapter = _require_string(table, "adapter")
    robots_url = _require_url(table, "robots_url")
    robots_policy = _require_string(table, "robots_policy")

    if not 0 < max_requests_per_second <= MAX_REQUESTS_PER_SECOND:
        raise RegistryError(
            f"{context}.max_requests_per_second must be in (0, {MAX_REQUESTS_PER_SECOND}]"
        )
    if not 1 <= concurrency <= MAX_CONCURRENCY:
        raise RegistryError(f"{context}.concurrency must be in [1, {MAX_CONCURRENCY}]")
    if not 1 <= max_object_bytes <= MAX_OBJECT_BYTES_LIMIT:
        raise RegistryError(f"{context}.max_object_bytes must be in [1, {MAX_OBJECT_BYTES_LIMIT}]")
    if len(user_agent) < 10 or "openshogi" not in user_agent.casefold():
        raise RegistryError(f"{context}.user_agent must explicitly identify OpenShogiAI")
    if robots_policy not in {"allowed", "denied", "unverified"}:
        raise RegistryError(f"{context}.robots_policy is invalid")
    if robots_checked > last_reviewed or terms_checked > last_reviewed:
        raise RegistryError(f"{context} check dates must not be after last_reviewed")
    if last_reviewed > date.today():
        raise RegistryError(f"{context}.last_reviewed must not be in the future")

    official_parts = urlsplit(official_base)
    if official_parts.hostname not in allowed_hosts:
        raise RegistryError(f"{context}.official_base host is not in allowed_hosts")
    if official_parts.scheme == "http" and not allow_insecure_http:
        raise RegistryError(f"{context}.official_base uses HTTP without explicit approval")
    robots_parts = urlsplit(robots_url)
    if robots_parts.hostname not in allowed_hosts:
        raise RegistryError(f"{context}.robots_url host is not in allowed_hosts")
    if robots_parts.scheme == "http" and not allow_insecure_http:
        raise RegistryError(f"{context}.robots_url uses HTTP without explicit approval")

    if approved:
        if not enabled:
            raise RegistryError(f"{context}: approved sources must be enabled")
        if license_name.casefold() in {"unknown", "unverified", "pending"}:
            raise RegistryError(f"{context}: approved sources require a decided license")
        if not license_evidence:
            raise RegistryError(f"{context}: approved sources require license evidence")
        if robots_policy != "allowed":
            raise RegistryError(f"{context}: approved sources require allowed robots policy")
        if not machine_learning_allowed:
            raise RegistryError(f"{context}: approved sources must permit machine learning")
        if adapter != "aobazero_csa" or catalog_path is None:
            raise RegistryError(f"{context}: approved sources require a static catalog")
        if evidence_catalog_path is None:
            raise RegistryError(f"{context}: approved sources require an evidence catalog")
    elif enabled:
        raise RegistryError(f"{context}: unapproved sources must remain disabled")

    catalog: tuple[CatalogObject, ...] = ()
    if catalog_path is not None:
        catalog = _load_catalog(
            _resolve_project_relative(
                project_root,
                catalog_path,
                context=f"{context}.catalog",
            ),
            source_id=source_id,
            source_policy=(
                official_base,
                allowed_hosts,
                allow_insecure_http,
                allowed_paths,
                denied_paths,
            ),
        )
    evidence_catalog: tuple[EvidenceObject, ...] = ()
    if evidence_catalog_path is not None:
        evidence_catalog = _load_evidence_catalog(
            _resolve_project_relative(
                project_root,
                evidence_catalog_path,
                context=f"{context}.evidence_catalog",
            ),
            source_id=source_id,
            allow_insecure_http=allow_insecure_http,
        )
    if approved:
        evidence_urls = {item.url for item in evidence_catalog}
        missing_evidence = [item.url for item in license_evidence if item.url not in evidence_urls]
        if missing_evidence:
            raise RegistryError(
                f"{context}: license evidence URLs are absent from evidence catalog"
            )
        unpinned = [
            item.evidence_id
            for item in evidence_catalog
            if item.url != robots_url and item.sha256 is None
        ]
        if unpinned:
            raise RegistryError(f"{context}: non-robots evidence must pin SHA-256: {unpinned}")

    source = DataSource(
        source_id=source_id,
        name=name,
        official_base=official_base,
        enabled=enabled,
        approved=approved,
        license=license_name,
        license_evidence=license_evidence,
        robots_checked=robots_checked,
        terms_checked=terms_checked,
        max_requests_per_second=max_requests_per_second,
        concurrency=concurrency,
        allowed_paths=allowed_paths,
        denied_paths=denied_paths,
        redistributable=redistributable,
        machine_learning_allowed=machine_learning_allowed,
        last_reviewed=last_reviewed,
        allowed_hosts=allowed_hosts,
        allow_insecure_http=allow_insecure_http,
        max_object_bytes=max_object_bytes,
        user_agent=user_agent,
        catalog_path=catalog_path,
        evidence_catalog_path=evidence_catalog_path,
        adapter=adapter,
        robots_url=robots_url,
        robots_policy=robots_policy,
        catalog=catalog,
        evidence_catalog=evidence_catalog,
    )
    source.validate_url(robots_url)
    for item in catalog:
        source.validate_url(item.url)
    return source


def _parse_evidence(
    raw_items: list[Any], *, context: str, project_root: Path
) -> tuple[LicenseEvidence, ...]:
    evidence: list[LicenseEvidence] = []
    for index, raw_item in enumerate(raw_items):
        item_context = f"{context}[{index}]"
        table = _require_mapping(raw_item, item_context)
        _require_exact_keys(table, _EVIDENCE_KEYS, item_context)
        url = _require_url(table, "url")
        local_path = _require_string(table, "local_path")
        quote = _require_string(
            table,
            "quote",
            maximum_length=MAX_EVIDENCE_QUOTE_LENGTH,
        )
        evidence_path = _resolve_project_relative(
            project_root,
            local_path,
            context=f"{item_context}.local_path",
        )
        if not evidence_path.is_file():
            raise RegistryError(f"{item_context}.local_path does not exist: {local_path}")
        evidence.append(LicenseEvidence(url=url, local_path=local_path, quote=quote))
    return tuple(evidence)


def _load_catalog(
    path: Path,
    *,
    source_id: str,
    source_policy: tuple[str, tuple[str, ...], bool, tuple[str, ...], tuple[str, ...]],
) -> tuple[CatalogObject, ...]:
    raw = _load_bounded_yaml(path, MAX_CATALOG_BYTES)
    root = _require_mapping(raw, f"catalog {path}")
    _require_exact_keys(root, _CATALOG_ROOT_KEYS, f"catalog {path}")
    schema_version = _require_int(root, "schema_version")
    if schema_version != CATALOG_SCHEMA_VERSION:
        raise RegistryError(f"unsupported catalog schema_version: {schema_version}")
    if _require_string(root, "source_id", pattern=_SOURCE_ID_RE) != source_id:
        raise RegistryError(f"catalog {path} source_id does not match registry")

    raw_objects = _require_list(root, "objects")
    if not 1 <= len(raw_objects) <= MAX_CATALOG_OBJECTS:
        raise RegistryError(
            f"catalog {path} must contain between 1 and {MAX_CATALOG_OBJECTS} objects"
        )

    official_base, hosts, allow_http, allowed_paths, denied_paths = source_policy
    objects: list[CatalogObject] = []
    object_ids: set[str] = set()
    urls: set[str] = set()
    filenames: set[str] = set()
    for index, raw_object in enumerate(raw_objects):
        context = f"catalog.objects[{index}]"
        table = _require_mapping(raw_object, context)
        _require_exact_keys(table, _CATALOG_OBJECT_KEYS, context)
        item = CatalogObject(
            object_id=_require_string(table, "object_id", pattern=_OBJECT_ID_RE),
            url=_require_url(table, "url"),
            filename=_require_filename(table, "filename"),
            data_format=_require_string(table, "data_format"),
            compression=_require_string(table, "compression"),
        )
        if item.object_id in object_ids:
            raise RegistryError(f"duplicate catalog object_id: {item.object_id}")
        if item.url in urls:
            raise RegistryError(f"duplicate catalog URL: {item.url}")
        if item.filename in filenames:
            raise RegistryError(f"duplicate catalog filename: {item.filename}")
        object_ids.add(item.object_id)
        urls.add(item.url)
        filenames.add(item.filename)
        _validate_url_policy(
            item.url,
            official_base=official_base,
            allowed_hosts=hosts,
            allow_insecure_http=allow_http,
            allowed_paths=allowed_paths,
            denied_paths=denied_paths,
        )
        if PurePosixPath(urlsplit(item.url).path).name != item.filename:
            raise RegistryError(f"{context}.filename must match the URL path")
        objects.append(item)
    return tuple(objects)


def _load_evidence_catalog(
    path: Path,
    *,
    source_id: str,
    allow_insecure_http: bool,
) -> tuple[EvidenceObject, ...]:
    raw = _load_bounded_yaml(path, MAX_CATALOG_BYTES)
    root = _require_mapping(raw, f"evidence catalog {path}")
    _require_exact_keys(root, _EVIDENCE_CATALOG_ROOT_KEYS, f"evidence catalog {path}")
    schema_version = _require_int(root, "schema_version")
    if schema_version != CATALOG_SCHEMA_VERSION:
        raise RegistryError(f"unsupported evidence catalog schema_version: {schema_version}")
    if _require_string(root, "source_id", pattern=_SOURCE_ID_RE) != source_id:
        raise RegistryError(f"evidence catalog {path} source_id does not match registry")
    raw_items = _require_list(root, "evidence")
    if not 1 <= len(raw_items) <= 16:
        raise RegistryError("evidence catalog must contain between 1 and 16 exact URLs")

    items: list[EvidenceObject] = []
    ids: set[str] = set()
    urls: set[str] = set()
    for index, raw_item in enumerate(raw_items):
        context = f"evidence[{index}]"
        table = _require_mapping(raw_item, context)
        _require_exact_keys(table, _EVIDENCE_OBJECT_KEYS, context)
        item = EvidenceObject(
            evidence_id=_require_string(table, "evidence_id", pattern=_OBJECT_ID_RE),
            url=_require_url(table, "url"),
            max_bytes=_require_int(table, "max_bytes"),
            sha256=_require_optional_sha256(table, "sha256"),
        )
        if item.evidence_id in ids or item.url in urls:
            raise RegistryError(f"{context} duplicates an evidence id or URL")
        if not 1 <= item.max_bytes <= 1024 * 1024:
            raise RegistryError(f"{context}.max_bytes must be in [1, 1048576]")
        parts = urlsplit(item.url)
        if parts.query or parts.fragment:
            raise RegistryError(f"{context}.url must not contain query or fragment")
        if parts.hostname is None:
            raise RegistryError(f"{context}.url must have a host")
        if parts.scheme == "http" and not allow_insecure_http:
            raise RegistryError(f"{context}.url uses unapproved insecure HTTP")
        ids.add(item.evidence_id)
        urls.add(item.url)
        items.append(item)
    return tuple(items)


def _validate_source_url(source: DataSource, url: str) -> None:
    _validate_url_policy(
        url,
        official_base=source.official_base,
        allowed_hosts=source.allowed_hosts,
        allow_insecure_http=source.allow_insecure_http,
        allowed_paths=source.allowed_paths,
        denied_paths=source.denied_paths,
    )


def _validate_url_policy(
    url: str,
    *,
    official_base: str,
    allowed_hosts: tuple[str, ...],
    allow_insecure_http: bool,
    allowed_paths: tuple[str, ...],
    denied_paths: tuple[str, ...],
) -> None:
    parts = urlsplit(url)
    if parts.scheme not in {"https", "http"}:
        raise RegistryError(f"unsupported URL scheme: {parts.scheme!r}")
    if parts.username is not None or parts.password is not None:
        raise RegistryError("URL credentials are forbidden")
    if parts.fragment:
        raise RegistryError("URL fragments are forbidden")
    if parts.query:
        raise RegistryError("URL queries are forbidden for audited objects")
    if parts.hostname is None or parts.hostname.casefold() not in allowed_hosts:
        raise RegistryError(f"URL host is not approved: {parts.hostname!r}")
    if parts.scheme == "http" and not allow_insecure_http:
        raise RegistryError("insecure HTTP is not approved for this source")

    base_parts = urlsplit(official_base)
    if parts.scheme != base_parts.scheme:
        raise RegistryError("URL scheme differs from the audited official base")
    if _effective_port(parts) != _effective_port(base_parts):
        raise RegistryError("URL port differs from the audited official base")

    path = _canonical_url_path(parts.path)
    if any(_path_matches_prefix(path, prefix) for prefix in denied_paths):
        raise RegistryError(f"URL path is explicitly denied: {path}")
    if not any(_path_matches_prefix(path, prefix) for prefix in allowed_paths):
        raise RegistryError(f"URL path is outside allowed paths: {path}")


def _canonical_url_path(raw_path: str) -> str:
    try:
        decoded = unquote(raw_path, errors="strict")
    except UnicodeDecodeError as error:
        raise RegistryError("URL path is not valid UTF-8") from error
    if not decoded.startswith("/"):
        raise RegistryError("URL path must be absolute")
    if "\\" in decoded or "\x00" in decoded or any(ord(char) < 32 for char in decoded):
        raise RegistryError("URL path contains forbidden characters")
    parts = decoded.split("/")
    if any(part in {".", ".."} for part in parts):
        raise RegistryError("URL path traversal is forbidden")
    return decoded


def _path_matches_prefix(path: str, prefix: str) -> bool:
    if prefix == "/":
        return True
    return path == prefix.rstrip("/") or path.startswith(prefix)


def _effective_port(parts: Any) -> int:
    try:
        if parts.port is not None:
            return parts.port
    except ValueError as error:
        raise RegistryError("URL contains an invalid port") from error
    return 443 if parts.scheme == "https" else 80


def _load_bounded_yaml(path: Path, maximum_bytes: int) -> Any:
    try:
        size = path.stat().st_size
    except OSError as error:
        raise RegistryError(f"cannot inspect YAML file {path}: {error}") from error
    if size > maximum_bytes:
        raise RegistryError(f"YAML file exceeds {maximum_bytes} bytes: {path}")
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise RegistryError(f"cannot read YAML file {path}: {error}") from error
    if len(content.encode("utf-8")) > maximum_bytes:
        raise RegistryError(f"YAML file exceeds {maximum_bytes} bytes: {path}")
    try:
        return yaml.load(content, Loader=UniqueSafeLoader)
    except yaml.YAMLError as error:
        raise RegistryError(f"invalid YAML in {path}: {error}") from error


def _require_mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RegistryError(f"{context} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise RegistryError(f"{context} keys must be strings")
    return value


def _require_exact_keys(table: Mapping[str, Any], expected: frozenset[str], context: str) -> None:
    actual = frozenset(table)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        raise RegistryError(f"{context} keys mismatch; missing={missing}, unknown={unknown}")


def _require_string(
    table: Mapping[str, Any],
    key: str,
    *,
    pattern: re.Pattern[str] | None = None,
    maximum_length: int = MAX_STRING_LENGTH,
) -> str:
    value = table.get(key)
    if not isinstance(value, str) or not value or len(value) > maximum_length:
        raise RegistryError(f"{key} must be a non-empty string of at most {maximum_length} chars")
    if pattern is not None and pattern.fullmatch(value) is None:
        raise RegistryError(f"{key} has an invalid format")
    return value


def _require_url(table: Mapping[str, Any], key: str) -> str:
    value = _require_string(table, key)
    parts = urlsplit(value)
    if parts.scheme not in {"http", "https"} or parts.hostname is None:
        raise RegistryError(f"{key} must be an absolute HTTP(S) URL")
    if parts.username is not None or parts.password is not None:
        raise RegistryError(f"{key} must not include credentials")
    if parts.query or parts.fragment:
        raise RegistryError(f"{key} must not include query or fragment")
    try:
        _ = parts.port
    except ValueError as error:
        raise RegistryError(f"{key} contains an invalid port") from error
    return value


def _require_bool(table: Mapping[str, Any], key: str) -> bool:
    value = table.get(key)
    if not isinstance(value, bool):
        raise RegistryError(f"{key} must be a boolean")
    return value


def _require_int(table: Mapping[str, Any], key: str) -> int:
    value = table.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise RegistryError(f"{key} must be an integer")
    return value


def _require_float(table: Mapping[str, Any], key: str) -> float:
    value = table.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise RegistryError(f"{key} must be numeric")
    return float(value)


def _require_list(table: Mapping[str, Any], key: str) -> list[Any]:
    value = table.get(key)
    if not isinstance(value, list):
        raise RegistryError(f"{key} must be a list")
    return value


def _require_date(table: Mapping[str, Any], key: str) -> date:
    raw = _require_string(table, key, maximum_length=10)
    try:
        parsed = date.fromisoformat(raw)
    except ValueError as error:
        raise RegistryError(f"{key} must be an ISO-8601 calendar date") from error
    if parsed.isoformat() != raw:
        raise RegistryError(f"{key} must use canonical YYYY-MM-DD format")
    return parsed


def _require_hosts(table: Mapping[str, Any], key: str) -> tuple[str, ...]:
    raw_hosts = _require_list(table, key)
    if not raw_hosts:
        raise RegistryError(f"{key} must not be empty")
    hosts: list[str] = []
    for raw_host in raw_hosts:
        if not isinstance(raw_host, str):
            raise RegistryError(f"{key} entries must be strings")
        host = raw_host.casefold()
        if _HOST_RE.fullmatch(host) is None:
            raise RegistryError(f"{key} contains an invalid host: {raw_host!r}")
        if host in hosts:
            raise RegistryError(f"{key} contains a duplicate host: {host}")
        hosts.append(host)
    return tuple(hosts)


def _require_path_prefixes(table: Mapping[str, Any], key: str) -> tuple[str, ...]:
    raw_paths = _require_list(table, key)
    if not raw_paths:
        raise RegistryError(f"{key} must not be empty")
    paths: list[str] = []
    for raw_path in raw_paths:
        if not isinstance(raw_path, str):
            raise RegistryError(f"{key} entries must be strings")
        path = _canonical_url_path(raw_path)
        if path != "/" and not path.endswith("/"):
            raise RegistryError(f"{key} prefixes must end with '/': {path}")
        if path in paths:
            raise RegistryError(f"{key} contains a duplicate path: {path}")
        paths.append(path)
    return tuple(paths)


def _require_optional_relative_path(table: Mapping[str, Any], key: str) -> str | None:
    value = table.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise RegistryError(f"{key} must be null or a non-empty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise RegistryError(f"{key} must be a normalized project-relative path")
    return value


def _require_optional_sha256(table: Mapping[str, Any], key: str) -> str | None:
    value = table.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise RegistryError(f"{key} must be null or 64 lowercase hexadecimal characters")
    return value


def _require_filename(table: Mapping[str, Any], key: str) -> str:
    value = _require_string(table, key, pattern=_OBJECT_ID_RE)
    path = PurePosixPath(value)
    if path.name != value or value in {".", ".."}:
        raise RegistryError(f"{key} must be a safe basename")
    return value


def _resolve_project_relative(project_root: Path, relative: str, *, context: str) -> Path:
    posix = PurePosixPath(relative)
    if posix.is_absolute() or ".." in posix.parts or "." in posix.parts:
        raise RegistryError(f"{context} must be a normalized project-relative path")
    resolved_root = project_root.resolve()
    candidate = resolved_root / Path(*posix.parts)
    current = resolved_root
    for part in posix.parts:
        current /= part
        if current.is_symlink():
            raise RegistryError(f"{context} must not traverse a symlink")
    resolved = candidate.resolve()
    if not resolved.is_relative_to(resolved_root):
        raise RegistryError(f"{context} escapes the project root")
    return resolved
