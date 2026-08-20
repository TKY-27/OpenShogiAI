"""Fail-closed, resumable acquisition for exact audited catalogs."""

from __future__ import annotations

import hashlib
import os
import re
import stat
import tempfile
import threading
import time
import urllib.error
import urllib.request
import urllib.robotparser
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from io import BufferedIOBase
from pathlib import Path
from typing import Final, Protocol
from urllib.parse import urljoin

from open_shogi_training.data.manifest import (
    CompletedObject,
    EvidenceSnapshot,
    ManifestError,
    ManifestStore,
    PartialObject,
    utc_now,
)
from open_shogi_training.data.registry import (
    MAX_CATALOG_OBJECTS,
    CatalogObject,
    DataSource,
    EvidenceObject,
    RegistryError,
)

MAX_SAMPLE_FILES: Final = 100
MAX_ROBOTS_BYTES: Final = 64 * 1024
MAX_RETRY_AFTER_SECONDS: Final = 30.0
DEFAULT_TIMEOUT_SECONDS: Final = 30.0
DEFAULT_RETRY_ATTEMPTS: Final = 3
DEFAULT_BACKOFF_SECONDS: Final = 0.5
READ_CHUNK_BYTES: Final = 64 * 1024
_RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})
_CONTENT_RANGE_RE = re.compile(r"bytes ([0-9]+)-([0-9]+)/([0-9]+)\Z")
_STRONG_ETAG_RE = re.compile(r'"[!#-~]*"\Z')
_KIFU_LOAD_RE = re.compile(r'Kifu[.]load[(]\s*"([^"\r\n]+)"\s*[)]\s*;')


class AcquisitionError(RuntimeError):
    """Raised when a rights, transport, integrity, or storage gate fails."""


class TransportError(AcquisitionError):
    """Raised when an HTTP request fails before receiving a response."""


class HttpResponse(Protocol):
    status: int
    url: str
    headers: Mapping[str, str]
    body: BufferedIOBase

    def close(self) -> None: ...


class HttpTransport(Protocol):
    def request(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout: float,
        validate_redirect: Callable[[str], None],
    ) -> HttpResponse: ...


@dataclass(frozen=True, slots=True)
class AcquisitionPlan:
    """Read-only sample plan; constructing it performs no network or disk writes."""

    source_id: str
    sample_only: bool
    limit: int
    objects: tuple[CatalogObject, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "mode": "dry-run",
            "source_id": self.source_id,
            "sample_only": self.sample_only,
            "limit": self.limit,
            "count": len(self.objects),
            "objects": [
                {
                    "object_id": item.object_id,
                    "url": item.url,
                    "filename": item.filename,
                    "data_format": item.data_format,
                    "compression": item.compression,
                }
                for item in self.objects
            ],
        }


@dataclass(frozen=True, slots=True)
class DownloadOutcome:
    """One catalog object's acquisition result."""

    status: str
    record: CompletedObject


@dataclass(slots=True)
class _UrllibResponse:
    status: int
    url: str
    headers: Mapping[str, str]
    body: BufferedIOBase

    def close(self) -> None:
        self.body.close()


class _AnchorCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.casefold() != "a":
            return
        for name, value in attrs:
            if name.casefold() == "href" and value is not None:
                self.hrefs.append(value)


class _PolicyRedirectHandler(urllib.request.HTTPRedirectHandler):
    def __init__(self, validate_redirect: Callable[[str], None]) -> None:
        self._validate_redirect = validate_redirect
        super().__init__()

    def redirect_request(
        self,
        request: urllib.request.Request,
        file_pointer: BufferedIOBase,
        code: int,
        message: str,
        headers: Mapping[str, str],
        new_url: str,
    ) -> urllib.request.Request | None:
        self._validate_redirect(new_url)
        return super().redirect_request(
            request,
            file_pointer,
            code,
            message,
            headers,
            new_url,
        )


class UrlLibTransport:
    """urllib transport with redirect validation before following each Location."""

    def request(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout: float,
        validate_redirect: Callable[[str], None],
    ) -> HttpResponse:
        request = urllib.request.Request(url, headers=dict(headers), method="GET")
        opener = urllib.request.build_opener(_PolicyRedirectHandler(validate_redirect))
        try:
            response = opener.open(request, timeout=timeout)
        except urllib.error.HTTPError as error:
            return _UrllibResponse(
                status=error.code,
                url=error.geturl(),
                headers=error.headers,
                body=error,
            )
        except (OSError, urllib.error.URLError) as error:
            raise TransportError(f"request failed for {url}: {error}") from error
        return _UrllibResponse(
            status=response.status,
            url=response.geturl(),
            headers=response.headers,
            body=response,
        )


class _RateLimiter:
    def __init__(
        self,
        requests_per_second: float,
        *,
        sleep: Callable[[float], None],
        monotonic: Callable[[], float],
    ) -> None:
        self._interval = 1.0 / requests_per_second
        self._sleep = sleep
        self._monotonic = monotonic
        self._next_request = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            current = self._monotonic()
            delay = self._next_request - current
            if delay > 0:
                self._sleep(delay)
                current = self._monotonic()
            self._next_request = max(current, self._next_request) + self._interval


def plan_acquisition(
    source: DataSource,
    *,
    limit: int = MAX_SAMPLE_FILES,
    sample_only: bool = True,
) -> AcquisitionPlan:
    """Create an exact, local-only plan after checking all static rights gates."""

    _require_source_approval(source)
    if not sample_only:
        raise AcquisitionError("Phase 3 acquisition is sample-only")
    if not 1 <= limit <= MAX_SAMPLE_FILES:
        raise AcquisitionError(f"limit must be between 1 and {MAX_SAMPLE_FILES}")
    if len(source.catalog) > MAX_CATALOG_OBJECTS:
        raise AcquisitionError("source catalog exceeds the hard sample limit")
    objects = source.catalog[:limit]
    if not objects:
        raise AcquisitionError("approved source has no catalog objects")
    for item in objects:
        source.validate_url(item.url, require_catalog_entry=True)
    return AcquisitionPlan(
        source_id=source.source_id,
        sample_only=True,
        limit=limit,
        objects=objects,
    )


class Downloader:
    """Download only enumerated objects under the source's closed policy."""

    def __init__(
        self,
        source: DataSource,
        output_root: Path,
        *,
        transport: HttpTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        now: Callable[[], str] = utc_now,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        retry_attempts: int = DEFAULT_RETRY_ATTEMPTS,
        backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
    ) -> None:
        if timeout_seconds <= 0 or timeout_seconds > 120:
            raise ValueError("timeout_seconds must be in (0, 120]")
        if not 1 <= retry_attempts <= 5:
            raise ValueError("retry_attempts must be in [1, 5]")
        if backoff_seconds < 0 or backoff_seconds > 10:
            raise ValueError("backoff_seconds must be in [0, 10]")
        self.source = source
        self.output_root = output_root
        if output_root.exists():
            root_status = output_root.lstat()
            if not stat.S_ISDIR(root_status.st_mode) or stat.S_ISLNK(root_status.st_mode):
                raise AcquisitionError("output root must be a non-symlink directory")
        self.manifest = ManifestStore(output_root)
        self._transport = transport or UrlLibTransport()
        self._sleep = sleep
        self._now = now
        self._timeout_seconds = timeout_seconds
        self._retry_attempts = retry_attempts
        self._backoff_seconds = backoff_seconds
        self._rate_limiter = _RateLimiter(
            source.max_requests_per_second,
            sleep=sleep,
            monotonic=monotonic,
        )
        self._request_slots = threading.BoundedSemaphore(source.concurrency)
        self._download_slots = threading.BoundedSemaphore(source.concurrency)
        self._robots_lock = threading.Lock()
        self._robots_parser: urllib.robotparser.RobotFileParser | None = None
        self._evidence_lock = threading.Lock()
        self._evidence_snapshots: tuple[EvidenceSnapshot, ...] | None = None
        self._run_mode_lock = threading.Lock()
        self._run_mode: str | None = None

    def acquire_catalog(
        self,
        *,
        limit: int = MAX_SAMPLE_FILES,
        sample_only: bool = True,
    ) -> tuple[DownloadOutcome, ...]:
        plan = plan_acquisition(self.source, limit=limit, sample_only=sample_only)
        with self._run_mode_lock:
            if self._run_mode is not None:
                raise AcquisitionError(
                    "each Downloader instance supports one acquisition mode; create a fresh "
                    "instance before acquire_catalog so live evidence is refreshed"
                )
            self._run_mode = "catalog"
        self._ensure_evidence_snapshots()
        self._ensure_robots_allows(plan.objects)
        with ThreadPoolExecutor(
            max_workers=self.source.concurrency,
            thread_name_prefix="source-download",
        ) as executor:
            return tuple(executor.map(self._download_catalog_item, plan.objects))

    def download(
        self,
        item: CatalogObject,
        *,
        refresh: bool = False,
    ) -> DownloadOutcome:
        with self._run_mode_lock:
            if self._run_mode == "catalog":
                raise AcquisitionError(
                    "direct download is unavailable after acquire_catalog starts; create a "
                    "fresh Downloader instance"
                )
            self._run_mode = "standalone"
        with self._download_slots:
            return self._download_with_slot(item, refresh=refresh)

    def _download_catalog_item(self, item: CatalogObject) -> DownloadOutcome:
        with self._download_slots:
            return self._download_with_slot(item, refresh=False)

    def _download_with_slot(
        self,
        item: CatalogObject,
        *,
        refresh: bool,
    ) -> DownloadOutcome:
        _require_source_approval(self.source)
        self.source.validate_url(item.url, require_catalog_entry=True)
        if item not in self.source.catalog:
            raise AcquisitionError(f"catalog object does not match audited entry: {item.object_id}")
        evidence_snapshots = self._ensure_evidence_snapshots()
        self._ensure_robots_allows((item,))

        try:
            snapshot = self.manifest.snapshot()
        except ManifestError as error:
            raise AcquisitionError(str(error)) from error
        completed = snapshot.completed_by_url().get(item.url)
        if completed is not None:
            self._validate_completed_record(item, completed)
        if completed is not None and not refresh:
            return DownloadOutcome(status="skipped-url", record=completed)

        partial = snapshot.latest_partial_by_url().get(item.url)
        return self._download_response(
            item,
            completed=completed,
            partial=partial,
            evidence_snapshots=evidence_snapshots,
        )

    def _download_response(
        self,
        item: CatalogObject,
        *,
        completed: CompletedObject | None,
        partial: PartialObject | None,
        evidence_snapshots: tuple[EvidenceSnapshot, ...],
    ) -> DownloadOutcome:
        headers = self._base_headers()
        existing_partial, resume_validator, partial_sha256 = self._usable_partial(item, partial)
        requested_offset = 0
        if existing_partial is not None and resume_validator is not None:
            requested_offset = existing_partial.stat().st_size
            headers["Range"] = f"bytes={requested_offset}-"
            headers["If-Range"] = resume_validator
        elif completed is not None:
            if completed.etag is not None:
                headers["If-None-Match"] = completed.etag
            elif completed.last_modified is not None:
                headers["If-Modified-Since"] = completed.last_modified

        response = self._open_with_retry(
            item.url,
            headers=headers,
            validator=lambda candidate: self._validate_exact_catalog_url(item, candidate),
        )
        try:
            self.source.validate_url(response.url)
            if response.status == 304:
                if completed is None:
                    raise AcquisitionError("received 304 without a completed local object")
                self._verify_completed_object(completed)
                return DownloadOutcome(status="not-modified", record=completed)
            if response.status == 416 and requested_offset:
                response.close()
                response = self._open_with_retry(
                    item.url,
                    headers=self._base_headers(),
                    validator=lambda candidate: self._validate_exact_catalog_url(item, candidate),
                )
                self.source.validate_url(response.url)
                requested_offset = 0
                existing_partial = None
                partial_sha256 = None
            if response.status not in {200, 206}:
                raise AcquisitionError(f"unexpected HTTP status {response.status} for {item.url}")

            append = response.status == 206
            if append and requested_offset == 0:
                raise AcquisitionError("unsolicited partial response is forbidden")
            expected_final_size: int | None = None
            if append:
                expected_final_size = self._validate_partial_response(
                    response,
                    requested_offset=requested_offset,
                    resume_validator=resume_validator,
                )
            elif _header(response.headers, "Content-Range") is not None:
                raise AcquisitionError("HTTP 200 response must not include Content-Range")

            working_descriptor, working = self._create_working_file()
            working_status = os.fstat(working_descriptor)
            etag = _safe_header(response.headers, "ETag")
            last_modified = _safe_header(response.headers, "Last-Modified")
            content_type = _safe_content_type(response.headers)
            try:
                try:
                    self._write_response(
                        response,
                        working_descriptor,
                        existing_partial=existing_partial if append else None,
                        expected_partial_sha256=partial_sha256 if append else None,
                        requested_offset=requested_offset if append else 0,
                    )
                    if (
                        expected_final_size is not None
                        and os.fstat(working_descriptor).st_size != expected_final_size
                    ):
                        raise AcquisitionError(
                            "resumed object size disagrees with Content-Range total"
                        )
                except BaseException:
                    self._record_partial(
                        item,
                        working_descriptor,
                        working,
                        working_status,
                        etag=etag,
                        last_modified=last_modified,
                    )
                    raise

                digest, size = _hash_descriptor(
                    working_descriptor,
                    self.source.max_object_bytes,
                )
                if completed is not None and digest != completed.sha256:
                    raise AcquisitionError(f"immutable catalog URL changed content: {item.url}")
                object_path, was_deduplicated = self._install_immutable(
                    working_descriptor,
                    working,
                    working_status,
                    digest,
                    size,
                )
                self._remove_saved_partial(item)
            finally:
                os.close(working_descriptor)
                _unlink_if_same(working, working_status)

            record = CompletedObject(
                source_id=self.source.source_id,
                object_id=item.object_id,
                url=item.url,
                retrieved_at=self._now(),
                sha256=digest,
                size=size,
                content_type=content_type,
                etag=etag,
                last_modified=last_modified,
                object_path=object_path,
                original_filename=item.filename,
                data_format=item.data_format,
                compression=item.compression,
                license=self.source.license,
                license_evidence=self.source.license_evidence,
                evidence_snapshots=evidence_snapshots,
                redistributable=self.source.redistributable,
                machine_learning_allowed=self.source.machine_learning_allowed,
            )
            if completed is not None:
                self._verify_completed_object(completed)
                return DownloadOutcome(status="refreshed-identical", record=completed)
            self.manifest.append_completed(record)
            return DownloadOutcome(
                status="deduplicated-content" if was_deduplicated else "downloaded",
                record=record,
            )
        except (ManifestError, RegistryError) as error:
            raise AcquisitionError(str(error)) from error
        finally:
            response.close()

    def _base_headers(self) -> dict[str, str]:
        return {
            "User-Agent": self.source.user_agent,
            "Accept": "application/octet-stream,text/plain;q=0.9",
            "Accept-Encoding": "identity",
            "Connection": "close",
        }

    def _ensure_evidence_snapshots(self) -> tuple[EvidenceSnapshot, ...]:
        with self._evidence_lock:
            if self._evidence_snapshots is not None:
                return self._evidence_snapshots

            evidence_items = list(self.source.evidence_catalog)
            robots_items = [item for item in evidence_items if item.url == self.source.robots_url]
            if len(robots_items) != 1:
                raise AcquisitionError("evidence catalog must contain the exact robots URL once")
            ordered = robots_items + [
                item for item in evidence_items if item.url != self.source.robots_url
            ]
            snapshots: list[EvidenceSnapshot] = []
            for item in ordered:
                if snapshots:
                    self._initialize_robots_from_snapshots(tuple(snapshots))
                    parser = self._robots_parser
                    if item.url.startswith(self.source.official_base) and (
                        parser is None or not parser.can_fetch(self.source.user_agent, item.url)
                    ):
                        raise AcquisitionError(f"robots.txt denies evidence URL: {item.url}")
                snapshots.append(self._fetch_evidence_snapshot(item))
            result = tuple(snapshots)
            self._verify_evidence_snapshots(result)
            self._verify_evidence_scope(result)
            self._evidence_snapshots = result
            self._initialize_robots_from_snapshots(result)
            return result

    def _fetch_evidence_snapshot(self, item: EvidenceObject) -> EvidenceSnapshot:
        evidence_id = item.evidence_id
        url = item.url
        maximum_bytes = item.max_bytes

        def validate_exact(candidate: str) -> None:
            if candidate != url:
                raise RegistryError(
                    f"evidence redirect or final URL differs from exact catalog: {candidate}"
                )

        response = self._open_with_retry(
            url,
            headers={
                "User-Agent": self.source.user_agent,
                "Accept": "text/plain,text/html;q=0.9,*/*;q=0.1",
                "Accept-Encoding": "identity",
                "Connection": "close",
            },
            validator=validate_exact,
        )
        try:
            if response.status != 200:
                raise AcquisitionError(f"evidence {evidence_id} returned HTTP {response.status}")
            content_encoding = _safe_header(response.headers, "Content-Encoding")
            if content_encoding is not None and content_encoding.casefold() != "identity":
                raise AcquisitionError("encoded evidence responses are forbidden")
            content_length = _content_length(response.headers)
            if content_length is not None and content_length > maximum_bytes:
                raise AcquisitionError(f"evidence {evidence_id} exceeds its byte cap")
            raw = _read_bounded(response.body, maximum_bytes)
            if not raw:
                raise AcquisitionError(f"evidence {evidence_id} is empty")
            if content_length is not None and content_length != len(raw):
                raise AcquisitionError(f"evidence {evidence_id} ended before Content-Length")
            content_type = _safe_content_type(response.headers)
        finally:
            response.close()

        digest = hashlib.sha256(raw).hexdigest()
        relative = Path("evidence") / "sha256" / digest[:2] / digest
        destination = self.output_root / relative
        _ensure_storage_directory(self.output_root, relative.parent)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix="evidence-",
            suffix=".tmp",
            dir=destination.parent,
        )
        temporary = Path(temporary_name)
        try:
            view = memoryview(raw)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("evidence write made no progress")
                view = view[written:]
            os.fsync(descriptor)
        except OSError as error:
            raise AcquisitionError(f"cannot write evidence snapshot: {error}") from error
        finally:
            os.close(descriptor)
        try:
            try:
                os.link(temporary, destination)
                _fsync_directory(destination.parent)
            except FileExistsError:
                existing_digest, existing_size = _hash_file(destination, maximum_bytes)
                if existing_digest != digest or existing_size != len(raw):
                    raise AcquisitionError(f"evidence object collision at {destination}") from None
        finally:
            temporary.unlink(missing_ok=True)
        return EvidenceSnapshot(
            evidence_id=evidence_id,
            url=url,
            retrieved_at=self._now(),
            sha256=digest,
            size=len(raw),
            content_type=content_type,
            object_path=relative.as_posix(),
        )

    def _verify_evidence_snapshots(self, snapshots: tuple[EvidenceSnapshot, ...]) -> None:
        expected = {item.evidence_id: item for item in self.source.evidence_catalog}
        if set(expected) != {item.evidence_id for item in snapshots}:
            raise AcquisitionError("evidence snapshots do not match the exact catalog")
        if len(snapshots) != len(expected):
            raise AcquisitionError("evidence snapshots contain duplicate identifiers")
        for snapshot in snapshots:
            item = expected[snapshot.evidence_id]
            if snapshot.url != item.url:
                raise AcquisitionError("evidence snapshot URL differs from catalog")
            path = _resolve_existing_storage_file(self.output_root, snapshot.object_path)
            try:
                digest, size = _hash_file(path, item.max_bytes)
            except OSError as error:
                raise AcquisitionError(f"cannot verify evidence snapshot: {error}") from error
            if digest != snapshot.sha256 or size != snapshot.size:
                raise AcquisitionError("evidence snapshot failed integrity verification")
            if item.sha256 is not None and snapshot.sha256 != item.sha256:
                raise AcquisitionError(
                    f"evidence {item.evidence_id} differs from its audited SHA-256"
                )

    def _verify_evidence_scope(self, snapshots: tuple[EvidenceSnapshot, ...]) -> None:
        by_url = {snapshot.url: snapshot for snapshot in snapshots}
        for evidence in self.source.license_evidence:
            snapshot = by_url.get(evidence.url)
            if snapshot is None:
                raise AcquisitionError(f"license evidence URL was not snapshotted: {evidence.url}")
            path = _resolve_existing_storage_file(self.output_root, snapshot.object_path)
            try:
                text = _read_file_bounded(path, 1024 * 1024).decode("utf-8")
            except (OSError, UnicodeError) as error:
                raise AcquisitionError(f"cannot read license evidence snapshot: {error}") from error
            if evidence.quote not in text:
                raise AcquisitionError("pinned license quotation is absent from snapshot")

        sample = next(
            (snapshot for snapshot in snapshots if snapshot.evidence_id == "official-sample-index"),
            None,
        )
        if sample is not None:
            path = _resolve_existing_storage_file(self.output_root, sample.object_path)
            try:
                index = _read_file_bounded(path, 1024 * 1024).decode("utf-8")
            except (OSError, UnicodeError) as error:
                raise AcquisitionError(f"cannot read sample index snapshot: {error}") from error
            parser = _AnchorCollector()
            try:
                parser.feed(index)
                parser.close()
            except (ValueError, AssertionError) as error:
                raise AcquisitionError("official sample index is invalid HTML") from error
            references = (*parser.hrefs, *_KIFU_LOAD_RE.findall(index))
            referenced_csa_urls = {
                urljoin(sample.url, reference)
                for reference in references
                if urljoin(sample.url, reference).casefold().endswith(".csa")
            }
            catalog_urls = {item.url for item in self.source.catalog}
            missing = sorted(catalog_urls - referenced_csa_urls)
            if missing:
                raise AcquisitionError(
                    "official sample index does not reference every audited catalog object; "
                    f"missing={missing[:3]}"
                )

    def _initialize_robots_from_snapshots(self, snapshots: tuple[EvidenceSnapshot, ...]) -> None:
        if self._robots_parser is not None:
            return
        snapshot = next(
            (item for item in snapshots if item.url == self.source.robots_url),
            None,
        )
        if snapshot is None:
            return
        path = _resolve_existing_storage_file(self.output_root, snapshot.object_path)
        try:
            raw = _read_file_bounded(path, MAX_ROBOTS_BYTES)
            text = raw.decode("utf-8")
        except (OSError, UnicodeError) as error:
            raise AcquisitionError(f"cannot read robots evidence snapshot: {error}") from error
        if len(raw) > MAX_ROBOTS_BYTES:
            raise AcquisitionError("robots evidence snapshot exceeds byte cap")
        parser = urllib.robotparser.RobotFileParser()
        parser.set_url(self.source.robots_url)
        parser.parse(text.splitlines())
        self._robots_parser = parser

    def _snapshotted_robots_bytes(self) -> bytes | None:
        snapshots = self._evidence_snapshots
        if snapshots is None:
            return None
        snapshot = next(
            (item for item in snapshots if item.url == self.source.robots_url),
            None,
        )
        if snapshot is None:
            return None
        path = _resolve_existing_storage_file(self.output_root, snapshot.object_path)
        try:
            raw = _read_file_bounded(path, MAX_ROBOTS_BYTES)
        except OSError as error:
            raise AcquisitionError(f"cannot read robots evidence snapshot: {error}") from error
        if len(raw) > MAX_ROBOTS_BYTES:
            raise AcquisitionError("robots evidence snapshot exceeds byte cap")
        return raw

    def _ensure_robots_allows(self, objects: Sequence[CatalogObject]) -> None:
        _require_source_approval(self.source)
        if self.source.robots_policy != "allowed":
            raise AcquisitionError("source robots policy is not approved")
        with self._robots_lock:
            if self._robots_parser is None:
                raw = self._snapshotted_robots_bytes()
                if raw is None:
                    response = self._open_with_retry(
                        self.source.robots_url,
                        headers={
                            "User-Agent": self.source.user_agent,
                            "Accept": "text/plain",
                            "Accept-Encoding": "identity",
                            "Connection": "close",
                        },
                    )
                    try:
                        self.source.validate_url(response.url)
                        if response.status != 200:
                            raise AcquisitionError(f"robots.txt returned HTTP {response.status}")
                        raw = _read_bounded(response.body, MAX_ROBOTS_BYTES)
                    finally:
                        response.close()
                try:
                    text = raw.decode("utf-8")
                except UnicodeDecodeError as error:
                    raise AcquisitionError("robots.txt is not valid UTF-8") from error
                parser = urllib.robotparser.RobotFileParser()
                parser.set_url(self.source.robots_url)
                parser.parse(text.splitlines())
                self._robots_parser = parser
            parser = self._robots_parser
            if parser is None:
                raise AcquisitionError("robots policy was not initialized")
            for item in objects:
                if not parser.can_fetch(self.source.user_agent, item.url):
                    raise AcquisitionError(f"robots.txt denies catalog URL: {item.url}")

    def _open_with_retry(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        validator: Callable[[str], None] | None = None,
    ) -> HttpResponse:
        validate = validator or self.source.validate_url
        validate(url)
        last_transport_error: TransportError | None = None
        for attempt in range(self._retry_attempts):
            try:
                self._rate_limiter.wait()
                with self._request_slots:
                    response = self._transport.request(
                        url,
                        headers=headers,
                        timeout=self._timeout_seconds,
                        validate_redirect=validate,
                    )
            except (RegistryError, TransportError) as error:
                if isinstance(error, RegistryError):
                    raise AcquisitionError(f"redirect policy rejected request: {error}") from error
                last_transport_error = error
                if attempt + 1 >= self._retry_attempts:
                    break
                self._sleep(self._backoff_delay(attempt, None))
                continue
            try:
                validate(response.url)
            except RegistryError as error:
                response.close()
                raise AcquisitionError(f"redirect policy rejected final URL: {error}") from error

            if response.status not in _RETRYABLE_STATUSES:
                return response
            try:
                retry_after = _safe_header(response.headers, "Retry-After")
            finally:
                response.close()
            if attempt + 1 >= self._retry_attempts:
                raise AcquisitionError(
                    f"HTTP {response.status} after {self._retry_attempts} attempts"
                )
            self._sleep(self._backoff_delay(attempt, retry_after))

        if last_transport_error is not None:
            raise AcquisitionError(
                f"request failed after {self._retry_attempts} attempts: {last_transport_error}"
            ) from last_transport_error
        raise AcquisitionError("request failed without a response")

    def _backoff_delay(self, attempt: int, retry_after: str | None) -> float:
        parsed = _parse_retry_after(retry_after)
        if parsed is not None:
            return min(parsed, MAX_RETRY_AFTER_SECONDS)
        return min(
            self._backoff_seconds * (2**attempt),
            MAX_RETRY_AFTER_SECONDS,
        )

    def _usable_partial(
        self,
        item: CatalogObject,
        partial: PartialObject | None,
    ) -> tuple[Path | None, str | None, str | None]:
        if partial is None:
            return None, None, None
        if (
            partial.source_id != self.source.source_id
            or partial.object_id != item.object_id
            or partial.url != item.url
        ):
            return None, None, None
        validator = _resume_validator(partial.etag, partial.last_modified)
        if validator is None:
            return None, None, None
        try:
            path = _resolve_existing_storage_file(self.output_root, partial.partial_path)
        except ManifestError as error:
            raise AcquisitionError(str(error)) from error
        expected_path = self._saved_partial_path(item).relative_to(self.output_root).as_posix()
        if partial.partial_path != expected_path:
            return None, None, None
        try:
            digest, size = _hash_file(path, self.source.max_object_bytes)
        except (AcquisitionError, OSError):
            return None, None, None
        if (
            size != partial.size
            or digest != partial.sha256
            or not 0 < size < self.source.max_object_bytes
        ):
            return None, None, None
        return path, validator, partial.sha256

    def _validate_partial_response(
        self,
        response: HttpResponse,
        *,
        requested_offset: int,
        resume_validator: str | None,
    ) -> int:
        if resume_validator is None:
            raise AcquisitionError("Range resume requires a strong ETag validator")
        content_range = _safe_header(response.headers, "Content-Range")
        if content_range is None:
            raise AcquisitionError("HTTP 206 response is missing Content-Range")
        match = _CONTENT_RANGE_RE.fullmatch(content_range)
        if match is None:
            raise AcquisitionError("HTTP 206 Content-Range is malformed")
        start, end, total = (int(value) for value in match.groups())
        if start != requested_offset or end < start or end + 1 != total:
            raise AcquisitionError("HTTP 206 Content-Range does not match requested offset")
        if total > self.source.max_object_bytes:
            raise AcquisitionError("HTTP 206 total exceeds the source object size cap")
        content_length = _content_length(response.headers)
        if content_length is not None and content_length != end - start + 1:
            raise AcquisitionError("Content-Length disagrees with Content-Range")

        response_etag = _safe_header(response.headers, "ETag")
        if response_etag != resume_validator:
            raise AcquisitionError("resumed response ETag does not match saved validator")
        return total

    def _validate_exact_catalog_url(self, item: CatalogObject, candidate: str) -> None:
        self.source.validate_url(candidate)
        if candidate != item.url:
            raise RegistryError(
                f"catalog redirects must preserve the exact audited URL: {candidate}"
            )

    def _create_working_file(self) -> tuple[int, Path]:
        partial_directory = self.output_root / ".partials"
        _ensure_storage_directory(self.output_root, Path(".partials"))
        descriptor, name = tempfile.mkstemp(
            prefix="download-",
            suffix=".tmp",
            dir=partial_directory,
        )
        return descriptor, Path(name)

    def _write_response(
        self,
        response: HttpResponse,
        working_descriptor: int,
        *,
        existing_partial: Path | None,
        expected_partial_sha256: str | None,
        requested_offset: int,
    ) -> None:
        content_encoding = _safe_header(response.headers, "Content-Encoding")
        if content_encoding is not None and content_encoding.casefold() not in {
            "identity",
        }:
            raise AcquisitionError("encoded HTTP responses are forbidden")
        content_length = _content_length(response.headers)
        expected_total = requested_offset + (content_length or 0)
        if content_length is not None and (
            content_length < 0 or expected_total > self.source.max_object_bytes
        ):
            raise AcquisitionError("Content-Length exceeds the source object size cap")

        try:
            os.ftruncate(working_descriptor, 0)
            os.lseek(working_descriptor, 0, os.SEEK_SET)
            copied = 0
            if existing_partial is not None:
                copied, copied_sha256 = _copy_bounded_to_descriptor(
                    existing_partial,
                    working_descriptor,
                    maximum_bytes=self.source.max_object_bytes,
                )
                if copied_sha256 != expected_partial_sha256:
                    os.ftruncate(working_descriptor, 0)
                    os.fsync(working_descriptor)
                    raise AcquisitionError("saved partial hash changed before resume")
            elif expected_partial_sha256 is not None:
                raise AcquisitionError("partial SHA-256 is present without a saved partial")
            if copied != requested_offset:
                raise AcquisitionError("saved partial size changed before resume")
        except OSError as error:
            raise AcquisitionError(f"cannot initialize response file: {error}") from error

        bytes_written = copied
        try:
            while True:
                chunk = response.body.read(READ_CHUNK_BYTES)
                if not chunk:
                    break
                if not isinstance(chunk, bytes):
                    raise AcquisitionError("HTTP body returned non-byte content")
                bytes_written += len(chunk)
                if bytes_written > self.source.max_object_bytes:
                    raise AcquisitionError("response body exceeds the source object size cap")
                _write_all(working_descriptor, chunk)
            os.fsync(working_descriptor)
        except OSError as error:
            raise AcquisitionError(f"cannot write response body: {error}") from error

        if content_length is not None and bytes_written - requested_offset != content_length:
            raise AcquisitionError("response ended before Content-Length bytes were received")
        if bytes_written <= 0:
            raise AcquisitionError("empty source objects are forbidden")

    def _record_partial(
        self,
        item: CatalogObject,
        working_descriptor: int,
        working: Path,
        working_status: os.stat_result,
        *,
        etag: str | None,
        last_modified: str | None,
    ) -> None:
        size = os.fstat(working_descriptor).st_size
        if not 0 < size <= self.source.max_object_bytes:
            return
        validator = _resume_validator(etag, last_modified)
        if validator is None:
            return
        digest, hashed_size = _hash_descriptor(
            working_descriptor,
            self.source.max_object_bytes,
        )
        if hashed_size != size:
            raise AcquisitionError("partial file size changed while hashing")
        destination = self._saved_partial_path(item)
        _ensure_storage_directory(self.output_root, Path(".partials"))
        try:
            os.replace(working, destination)
            installed_status = destination.lstat()
            if not _same_file(installed_status, working_status):
                _unlink_if_same(destination, installed_status)
                raise AcquisitionError("working file was replaced before partial persistence")
            _fsync_directory(destination.parent)
            relative = destination.relative_to(self.output_root).as_posix()
            self.manifest.append_partial(
                PartialObject(
                    source_id=self.source.source_id,
                    object_id=item.object_id,
                    url=item.url,
                    recorded_at=self._now(),
                    partial_path=relative,
                    size=size,
                    sha256=digest,
                    etag=etag,
                    last_modified=last_modified,
                )
            )
        except (OSError, ManifestError) as error:
            raise AcquisitionError(f"cannot persist partial download: {error}") from error

    def _saved_partial_path(self, item: CatalogObject) -> Path:
        key = hashlib.sha256(item.url.encode("utf-8")).hexdigest()
        return self.output_root / ".partials" / f"{key}.part"

    def _remove_saved_partial(self, item: CatalogObject) -> None:
        partial = self._saved_partial_path(item)
        if partial.exists():
            partial.unlink()
            _fsync_directory(partial.parent)

    def _install_immutable(
        self,
        working_descriptor: int,
        working: Path,
        working_status: os.stat_result,
        digest: str,
        size: int,
    ) -> tuple[str, bool]:
        relative = Path("objects") / "sha256" / digest[:2] / digest
        destination = self.output_root / relative
        _ensure_storage_directory(self.output_root, relative.parent)
        deduplicated = False
        try:
            if not _same_file(working.lstat(), working_status):
                raise AcquisitionError("working file was replaced before immutable installation")
            os.link(working, destination, follow_symlinks=False)
            installed_descriptor = _open_regular_read(destination)
            try:
                installed_status = os.fstat(installed_descriptor)
                if not _same_file(installed_status, working_status):
                    _unlink_if_same(destination, installed_status)
                    raise AcquisitionError("installed object is not the downloaded file")
                installed_digest, installed_size = _hash_descriptor(
                    installed_descriptor,
                    self.source.max_object_bytes,
                )
                if installed_digest != digest or installed_size != size:
                    _unlink_if_same(destination, installed_status)
                    raise AcquisitionError("installed object changed before verification")
            finally:
                os.close(installed_descriptor)
            _fsync_directory(destination.parent)
        except FileExistsError:
            existing_digest, existing_size = _hash_file(destination, self.source.max_object_bytes)
            if existing_digest != digest or existing_size != size:
                raise AcquisitionError(f"immutable object collision at {destination}") from None
            deduplicated = True
        except OSError as error:
            raise AcquisitionError(f"cannot install immutable object: {error}") from error
        return relative.as_posix(), deduplicated

    def _verify_completed_object(self, record: CompletedObject) -> None:
        try:
            path = _resolve_existing_storage_file(self.output_root, record.object_path)
            digest, size = _hash_file(path, self.source.max_object_bytes)
        except (ManifestError, OSError) as error:
            raise AcquisitionError(f"cannot verify completed object: {error}") from error
        if digest != record.sha256 or size != record.size:
            raise AcquisitionError(f"completed object failed integrity check: {path}")

    def _validate_completed_record(
        self,
        item: CatalogObject,
        record: CompletedObject,
    ) -> None:
        expected_object_path = (
            Path("objects") / "sha256" / record.sha256[:2] / record.sha256
        ).as_posix()
        expected = (
            self.source.source_id,
            item.object_id,
            item.url,
            item.filename,
            item.data_format,
            item.compression,
            self.source.license,
            self.source.license_evidence,
            self.source.redistributable,
            self.source.machine_learning_allowed,
            expected_object_path,
        )
        observed = (
            record.source_id,
            record.object_id,
            record.url,
            record.original_filename,
            record.data_format,
            record.compression,
            record.license,
            record.license_evidence,
            record.redistributable,
            record.machine_learning_allowed,
            record.object_path,
        )
        if observed != expected:
            raise AcquisitionError(
                f"completed record differs from current audited policy: {item.object_id}"
            )
        self._verify_evidence_snapshots(record.evidence_snapshots)
        self._verify_evidence_scope(record.evidence_snapshots)
        self._verify_completed_object(record)


def _require_source_approval(source: DataSource) -> None:
    if not source.enabled:
        raise AcquisitionError(f"source is disabled: {source.source_id}")
    if not source.approved:
        raise AcquisitionError(f"source is not approved: {source.source_id}")
    if not source.machine_learning_allowed:
        raise AcquisitionError(f"source is not approved for machine learning: {source.source_id}")
    if not source.license_evidence:
        raise AcquisitionError(f"source has no license evidence: {source.source_id}")
    if source.robots_policy != "allowed":
        raise AcquisitionError(f"source robots policy is not allowed: {source.source_id}")
    if source.adapter != "aobazero_csa":
        raise AcquisitionError(f"source adapter cannot acquire data: {source.adapter}")


def _header(headers: Mapping[str, str], name: str) -> str | None:
    value = headers.get(name)
    if value is not None:
        return value
    folded = name.casefold()
    for key, candidate in headers.items():
        if key.casefold() == folded:
            return candidate
    return None


def _safe_header(headers: Mapping[str, str], name: str) -> str | None:
    value = _header(headers, name)
    if value is None:
        return None
    value = value.strip()
    if not value or len(value) > 2_048 or "\r" in value or "\n" in value or "\x00" in value:
        raise AcquisitionError(f"invalid {name} response header")
    return value


def _safe_content_type(headers: Mapping[str, str]) -> str | None:
    value = _safe_header(headers, "Content-Type")
    if value is None:
        return None
    return value.split(";", maxsplit=1)[0].strip().casefold() or None


def _content_length(headers: Mapping[str, str]) -> int | None:
    value = _safe_header(headers, "Content-Length")
    if value is None:
        return None
    if not value.isascii() or not value.isdigit():
        raise AcquisitionError("Content-Length must be a non-negative decimal integer")
    return int(value)


def _read_bounded(stream: BufferedIOBase, maximum_bytes: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = stream.read(min(READ_CHUNK_BYTES, maximum_bytes + 1 - total))
        if not chunk:
            break
        if not isinstance(chunk, bytes):
            raise AcquisitionError("response returned non-byte content")
        chunks.append(chunk)
        total += len(chunk)
        if total > maximum_bytes:
            raise AcquisitionError(f"response exceeds {maximum_bytes} bytes")
    return b"".join(chunks)


def _copy_bounded_to_descriptor(
    source: Path,
    destination_descriptor: int,
    *,
    maximum_bytes: int,
) -> tuple[int, str]:
    try:
        descriptor = _open_regular_read(source)
        with os.fdopen(descriptor, "rb") as input_file:
            copied = 0
            digest = hashlib.sha256()
            while True:
                chunk = input_file.read(READ_CHUNK_BYTES)
                if not chunk:
                    break
                copied += len(chunk)
                if copied > maximum_bytes:
                    raise AcquisitionError("saved partial exceeds the object size cap")
                digest.update(chunk)
                _write_all(destination_descriptor, chunk)
    except OSError as error:
        raise AcquisitionError(f"cannot copy saved partial: {error}") from error
    return copied, digest.hexdigest()


def _hash_file(path: Path, maximum_bytes: int) -> tuple[str, int]:
    descriptor = _open_regular_read(path)
    try:
        return _hash_descriptor(descriptor, maximum_bytes)
    finally:
        os.close(descriptor)


def _hash_descriptor(descriptor: int, maximum_bytes: int) -> tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    offset = 0
    while True:
        chunk = os.pread(descriptor, READ_CHUNK_BYTES, offset)
        if not chunk:
            break
        total += len(chunk)
        offset += len(chunk)
        if total > maximum_bytes:
            raise AcquisitionError("local object exceeds the source object size cap")
        digest.update(chunk)
    return digest.hexdigest(), total


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("file write made no progress")
        view = view[written:]


def _same_file(left: os.stat_result, right: os.stat_result) -> bool:
    return left.st_dev == right.st_dev and left.st_ino == right.st_ino


def _unlink_if_same(path: Path, expected: os.stat_result) -> None:
    try:
        observed = path.lstat()
    except FileNotFoundError:
        return
    except OSError:
        return
    if not _same_file(observed, expected):
        return
    with suppress(FileNotFoundError):
        path.unlink()


def _read_file_bounded(path: Path, maximum_bytes: int) -> bytes:
    descriptor = _open_regular_read(path)
    with os.fdopen(descriptor, "rb") as input_file:
        return _read_bounded(input_file, maximum_bytes)


def _open_regular_read(path: Path) -> int:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise AcquisitionError(f"path is not a regular file: {path}")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _resume_validator(etag: str | None, last_modified: str | None) -> str | None:
    del last_modified
    if etag is not None and _STRONG_ETAG_RE.fullmatch(etag) is not None:
        return etag
    return None


def _parse_retry_after(value: str | None) -> float | None:
    if value is None:
        return None
    if value.isascii() and value.isdigit():
        return float(value)
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return max(0.0, (parsed - datetime.now(UTC)).total_seconds())


def _ensure_storage_directory(root: Path, relative: Path) -> Path:
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise AcquisitionError("storage directory must be a normalized relative path")
    root.mkdir(parents=True, exist_ok=True)
    root_status = root.lstat()
    if not stat.S_ISDIR(root_status.st_mode) or stat.S_ISLNK(root_status.st_mode):
        raise AcquisitionError("output root must be a non-symlink directory")
    current = root
    for part in relative.parts:
        current /= part
        with suppress(FileExistsError):
            current.mkdir(mode=0o700)
        item_status = current.lstat()
        if not stat.S_ISDIR(item_status.st_mode) or stat.S_ISLNK(item_status.st_mode):
            raise AcquisitionError(
                f"storage path component must be a non-symlink directory: {current}"
            )
    return current


def _resolve_existing_storage_file(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise AcquisitionError("storage object path must be normalized and relative")
    root_status = root.lstat()
    if not stat.S_ISDIR(root_status.st_mode) or stat.S_ISLNK(root_status.st_mode):
        raise AcquisitionError("output root must be a non-symlink directory")
    current = root
    for part in path.parts[:-1]:
        current /= part
        item_status = current.lstat()
        if not stat.S_ISDIR(item_status.st_mode) or stat.S_ISLNK(item_status.st_mode):
            raise AcquisitionError(
                f"storage path component must be a non-symlink directory: {current}"
            )
    candidate = current / path.parts[-1]
    item_status = candidate.lstat()
    if not stat.S_ISREG(item_status.st_mode) or stat.S_ISLNK(item_status.st_mode):
        raise AcquisitionError(f"storage object must be a non-symlink regular file: {candidate}")
    return candidate


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
