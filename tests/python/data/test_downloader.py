from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import date
from io import BytesIO
from pathlib import Path

import pytest
from open_shogi_training.data.downloader import (
    AcquisitionError,
    Downloader,
    HttpResponse,
)
from open_shogi_training.data.manifest import ManifestStore
from open_shogi_training.data.registry import (
    CatalogObject,
    DataSource,
    EvidenceObject,
    LicenseEvidence,
)

ROBOTS_URL = "http://source.test/robots.txt"
RIGHTS_URL = "https://evidence.test/readme.txt"
INDEX_URL = "http://source.test/data/sample.html"
GAME_ONE_URL = "http://source.test/data/game-one.csa"
GAME_TWO_URL = "http://source.test/data/game-two.csa"
QUOTE = "These samples are in the public domain."


class FakeResponse:
    def __init__(
        self,
        *,
        status: int = 200,
        url: str,
        body: bytes | BytesIO = b"",
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self.status = status
        self.url = url
        self.headers = dict(headers or {})
        self.body = body if isinstance(body, BytesIO) else BytesIO(body)

    def close(self) -> None:
        self.body.close()


class FailingBody(BytesIO):
    def __init__(self, first: bytes) -> None:
        super().__init__(first)
        self._failed = False

    def read(self, size: int = -1) -> bytes:
        if self.tell() < len(self.getvalue()):
            return super().read(size)
        if not self._failed:
            self._failed = True
            raise OSError("simulated connection loss")
        return b""


class FakeTransport:
    def __init__(self, routes: Mapping[str, list[FakeResponse]]) -> None:
        self.routes = {url: list(responses) for url, responses in routes.items()}
        self.requests: list[tuple[str, dict[str, str]]] = []

    def request(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout: float,
        validate_redirect: Callable[[str], None],
    ) -> HttpResponse:
        del timeout
        self.requests.append((url, dict(headers)))
        responses = self.routes.get(url)
        if not responses:
            raise AssertionError(f"unexpected network request: {url}")
        response = responses.pop(0)
        if response.url != url:
            validate_redirect(response.url)
        return response


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.value += seconds


def _source(*, two_games: bool = False, max_bytes: int = 1024) -> DataSource:
    catalog = [
        CatalogObject(
            object_id="game-one",
            url=GAME_ONE_URL,
            filename="game-one.csa",
            data_format="CSA",
            compression="none",
        )
    ]
    if two_games:
        catalog.append(
            CatalogObject(
                object_id="game-two",
                url=GAME_TWO_URL,
                filename="game-two.csa",
                data_format="CSA",
                compression="none",
            )
        )
    return DataSource(
        source_id="source",
        name="Synthetic source",
        official_base="http://source.test/data/",
        enabled=True,
        approved=True,
        license="Public Domain",
        license_evidence=(
            LicenseEvidence(
                url=RIGHTS_URL,
                local_path="docs/source-audits/source.md",
                quote=QUOTE,
            ),
        ),
        robots_checked=date(2026, 7, 29),
        terms_checked=date(2026, 7, 29),
        max_requests_per_second=2.0,
        concurrency=1,
        allowed_paths=("/data/", "/robots.txt/"),
        denied_paths=("/secret/",),
        redistributable=True,
        machine_learning_allowed=True,
        last_reviewed=date(2026, 7, 29),
        allowed_hosts=("source.test",),
        allow_insecure_http=True,
        max_object_bytes=max_bytes,
        user_agent="OpenShogiAI/0.3 test",
        catalog_path="catalog.yaml",
        evidence_catalog_path="evidence.yaml",
        adapter="aobazero_csa",
        robots_url=ROBOTS_URL,
        robots_policy="allowed",
        catalog=tuple(catalog),
        evidence_catalog=(
            EvidenceObject("robots", ROBOTS_URL, 4096),
            EvidenceObject("rights", RIGHTS_URL, 4096),
            EvidenceObject("official-sample-index", INDEX_URL, 4096),
        ),
    )


def _routes(
    source: DataSource,
    *,
    game_responses: Mapping[str, list[FakeResponse]] | None = None,
    rights: bytes | None = None,
    index: bytes | None = None,
    robots: bytes = b"User-agent: *\nDisallow: /secret/\n",
) -> dict[str, list[FakeResponse]]:
    sample_index = index
    if sample_index is None:
        sample_index = "\n".join(
            f'<script>Kifu.load("./{item.filename}");</script>' for item in source.catalog
        ).encode()
    result = {
        ROBOTS_URL: [
            FakeResponse(
                url=ROBOTS_URL,
                body=robots,
                headers={"Content-Length": str(len(robots)), "Content-Type": "text/plain"},
            )
        ],
        RIGHTS_URL: [
            FakeResponse(
                url=RIGHTS_URL,
                body=rights if rights is not None else QUOTE.encode(),
                headers={"Content-Type": "text/plain"},
            )
        ],
        INDEX_URL: [
            FakeResponse(
                url=INDEX_URL,
                body=sample_index,
                headers={"Content-Type": "text/html"},
            )
        ],
    }
    result.update(game_responses or {})
    return result


def _downloader(
    source: DataSource,
    root: Path,
    transport: FakeTransport,
    *,
    clock: FakeClock | None = None,
) -> Downloader:
    fake_clock = clock or FakeClock()
    return Downloader(
        source,
        root,
        transport=transport,
        sleep=fake_clock.sleep,
        monotonic=fake_clock.monotonic,
        now=lambda: "2026-07-29T00:00:00Z",
        backoff_seconds=0.25,
    )


def test_disabled_source_refuses_before_network_or_output(tmp_path: Path) -> None:
    source = replace(_source(), enabled=False, approved=False)
    output = tmp_path / "raw"
    transport = FakeTransport({})

    with pytest.raises(AcquisitionError, match="disabled"):
        _downloader(source, output, transport).acquire_catalog(limit=1)

    assert transport.requests == []
    assert not output.exists()


def test_robots_is_first_request_and_evidence_is_hashed(tmp_path: Path) -> None:
    source = _source()
    game = b"V2\nN+one\nN-two\n%TORYO\n"
    transport = FakeTransport(
        _routes(
            source,
            game_responses={
                GAME_ONE_URL: [
                    FakeResponse(
                        url=GAME_ONE_URL,
                        body=game,
                        headers={
                            "Content-Length": str(len(game)),
                            "Content-Type": "text/plain; charset=utf-8",
                            "ETag": '"game-v1"',
                        },
                    )
                ]
            },
        )
    )

    outcome = _downloader(source, tmp_path, transport).download(source.catalog[0])

    assert transport.requests[0][0] == ROBOTS_URL
    assert outcome.status == "downloaded"
    assert len(outcome.record.evidence_snapshots) == 3
    for snapshot in outcome.record.evidence_snapshots:
        evidence_path = tmp_path / snapshot.object_path
        assert evidence_path.is_file()
        assert len(snapshot.sha256) == 64
    assert outcome.record.content_type == "text/plain"


def test_changed_rights_quote_refuses_before_game_request(tmp_path: Path) -> None:
    source = _source()
    transport = FakeTransport(_routes(source, rights=b"Different rights text."))

    with pytest.raises(AcquisitionError, match="quotation is absent"):
        _downloader(source, tmp_path, transport).download(source.catalog[0])

    assert GAME_ONE_URL not in [url for url, _ in transport.requests]


def test_missing_sample_link_refuses_before_game_request(tmp_path: Path) -> None:
    source = _source()
    transport = FakeTransport(_routes(source, index=b"<html>no catalog links</html>"))

    with pytest.raises(AcquisitionError, match="does not reference every audited catalog"):
        _downloader(source, tmp_path, transport).download(source.catalog[0])

    assert GAME_ONE_URL not in [url for url, _ in transport.requests]


def test_sample_index_may_reference_unselected_csa_objects(tmp_path: Path) -> None:
    source = _source()
    index = b"""
        <script>Kifu.load("./game-one.csa");</script>
        <script>Kifu.load("./outside-the-bounded-slice.csa");</script>
    """
    transport = FakeTransport(
        _routes(
            source,
            index=index,
            game_responses={
                GAME_ONE_URL: [FakeResponse(url=GAME_ONE_URL, body=b"game")],
            },
        )
    )

    outcome = _downloader(source, tmp_path, transport).download(source.catalog[0])

    assert outcome.status == "downloaded"


def test_cross_host_data_redirect_is_rejected(tmp_path: Path) -> None:
    source = _source()
    redirected = "http://raw.githubusercontent.com/aobazero/no_noise/game-one.csa"
    transport = FakeTransport(
        _routes(
            source,
            game_responses={GAME_ONE_URL: [FakeResponse(url=redirected, body=b"not allowed")]},
        )
    )

    with pytest.raises(AcquisitionError, match="redirect policy"):
        _downloader(source, tmp_path, transport).download(source.catalog[0])


def test_content_length_and_stream_caps_fail_closed(tmp_path: Path) -> None:
    source = _source(max_bytes=4)
    transport = FakeTransport(
        _routes(
            source,
            game_responses={
                GAME_ONE_URL: [
                    FakeResponse(
                        url=GAME_ONE_URL,
                        body=b"12345",
                        headers={"Content-Length": "5"},
                    )
                ]
            },
        )
    )

    with pytest.raises(AcquisitionError, match="Content-Length exceeds"):
        _downloader(source, tmp_path, transport).download(source.catalog[0])

    assert ManifestStore(tmp_path).completed_records() == ()


def test_range_resume_requires_validator_and_complete_content_range(
    tmp_path: Path,
) -> None:
    source = _source()
    transport = FakeTransport(
        _routes(
            source,
            game_responses={
                GAME_ONE_URL: [
                    FakeResponse(
                        url=GAME_ONE_URL,
                        body=FailingBody(b"abcd"),
                        headers={"Content-Length": "10", "ETag": '"v1"'},
                    ),
                    FakeResponse(
                        status=206,
                        url=GAME_ONE_URL,
                        body=b"efghij",
                        headers={
                            "Content-Length": "6",
                            "Content-Range": "bytes 4-9/10",
                            "ETag": '"v1"',
                        },
                    ),
                ]
            },
        )
    )
    downloader = _downloader(source, tmp_path, transport)

    with pytest.raises(AcquisitionError, match="connection loss"):
        downloader.download(source.catalog[0])
    outcome = downloader.download(source.catalog[0])

    data_requests = [headers for url, headers in transport.requests if url == GAME_ONE_URL]
    assert data_requests[-1]["Range"] == "bytes=4-"
    assert data_requests[-1]["If-Range"] == '"v1"'
    assert (tmp_path / outcome.record.object_path).read_bytes() == b"abcdefghij"


def test_same_size_partial_swap_is_rejected_during_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source()
    transport = FakeTransport(
        _routes(
            source,
            game_responses={
                GAME_ONE_URL: [
                    FakeResponse(
                        url=GAME_ONE_URL,
                        body=FailingBody(b"abcd"),
                        headers={"Content-Length": "10", "ETag": '"v1"'},
                    ),
                    FakeResponse(
                        status=206,
                        url=GAME_ONE_URL,
                        body=b"efghij",
                        headers={
                            "Content-Length": "6",
                            "Content-Range": "bytes 4-9/10",
                            "ETag": '"v1"',
                        },
                    ),
                ]
            },
        )
    )
    downloader = _downloader(source, tmp_path, transport)

    with pytest.raises(AcquisitionError, match="connection loss"):
        downloader.download(source.catalog[0])

    original_write_response = downloader._write_response

    def swap_then_write(
        response: HttpResponse,
        working_descriptor: int,
        *,
        existing_partial: Path | None,
        expected_partial_sha256: str | None,
        requested_offset: int,
    ) -> None:
        assert existing_partial is not None
        existing_partial.write_bytes(b"wxyz")
        original_write_response(
            response,
            working_descriptor,
            existing_partial=existing_partial,
            expected_partial_sha256=expected_partial_sha256,
            requested_offset=requested_offset,
        )

    monkeypatch.setattr(downloader, "_write_response", swap_then_write)

    with pytest.raises(AcquisitionError, match="saved partial hash changed"):
        downloader.download(source.catalog[0])

    assert ManifestStore(tmp_path).completed_records() == ()


def test_truncated_206_is_not_installed_as_complete(tmp_path: Path) -> None:
    source = _source()
    transport = FakeTransport(
        _routes(
            source,
            game_responses={
                GAME_ONE_URL: [
                    FakeResponse(
                        url=GAME_ONE_URL,
                        body=FailingBody(b"abcd"),
                        headers={"Content-Length": "10", "ETag": '"v1"'},
                    ),
                    FakeResponse(
                        status=206,
                        url=GAME_ONE_URL,
                        body=b"efgh",
                        headers={
                            "Content-Length": "4",
                            "Content-Range": "bytes 4-7/10",
                            "ETag": '"v1"',
                        },
                    ),
                ]
            },
        )
    )
    downloader = _downloader(source, tmp_path, transport)
    with pytest.raises(AcquisitionError):
        downloader.download(source.catalog[0])

    with pytest.raises(AcquisitionError, match="Content-Range"):
        downloader.download(source.catalog[0])
    assert ManifestStore(tmp_path).completed_records() == ()


def test_range_request_safely_restarts_on_http_200(tmp_path: Path) -> None:
    source = _source()
    transport = FakeTransport(
        _routes(
            source,
            game_responses={
                GAME_ONE_URL: [
                    FakeResponse(
                        url=GAME_ONE_URL,
                        body=FailingBody(b"old"),
                        headers={"Content-Length": "10", "ETag": '"v1"'},
                    ),
                    FakeResponse(
                        url=GAME_ONE_URL,
                        body=b"new-content",
                        headers={
                            "Content-Length": "11",
                            "ETag": '"v2"',
                        },
                    ),
                ]
            },
        )
    )
    downloader = _downloader(source, tmp_path, transport)
    with pytest.raises(AcquisitionError):
        downloader.download(source.catalog[0])

    outcome = downloader.download(source.catalog[0])

    assert (tmp_path / outcome.record.object_path).read_bytes() == b"new-content"


def test_weak_etag_and_last_modified_never_enable_range(tmp_path: Path) -> None:
    source = _source()
    transport = FakeTransport(
        _routes(
            source,
            game_responses={
                GAME_ONE_URL: [
                    FakeResponse(
                        url=GAME_ONE_URL,
                        body=FailingBody(b"old"),
                        headers={
                            "Content-Length": "10",
                            "ETag": 'W/"weak"',
                            "Last-Modified": "Wed, 29 Jul 2026 00:00:00 GMT",
                        },
                    ),
                    FakeResponse(
                        url=GAME_ONE_URL,
                        body=b"replacement",
                        headers={"Content-Length": "11", "ETag": '"strong"'},
                    ),
                ]
            },
        )
    )
    downloader = _downloader(source, tmp_path, transport)
    with pytest.raises(AcquisitionError):
        downloader.download(source.catalog[0])

    outcome = downloader.download(source.catalog[0])

    data_headers = [headers for url, headers in transport.requests if url == GAME_ONE_URL]
    assert "Range" not in data_headers[-1]
    assert "If-Range" not in data_headers[-1]
    assert (tmp_path / outcome.record.object_path).read_bytes() == b"replacement"


def test_whitespace_validator_is_rejected(tmp_path: Path) -> None:
    source = _source()
    transport = FakeTransport(
        _routes(
            source,
            game_responses={
                GAME_ONE_URL: [
                    FakeResponse(
                        url=GAME_ONE_URL,
                        body=b"game",
                        headers={"Content-Length": "4", "ETag": "   "},
                    )
                ]
            },
        )
    )

    with pytest.raises(AcquisitionError, match="invalid ETag"):
        _downloader(source, tmp_path, transport).download(source.catalog[0])


def test_refresh_uses_conditional_etag_and_honors_304(tmp_path: Path) -> None:
    source = _source()
    body = b"game"
    transport = FakeTransport(
        _routes(
            source,
            game_responses={
                GAME_ONE_URL: [
                    FakeResponse(
                        url=GAME_ONE_URL,
                        body=body,
                        headers={"Content-Length": "4", "ETag": '"v1"'},
                    ),
                    FakeResponse(status=304, url=GAME_ONE_URL),
                ]
            },
        )
    )
    downloader = _downloader(source, tmp_path, transport)
    first = downloader.download(source.catalog[0])

    refreshed = downloader.download(source.catalog[0], refresh=True)

    data_headers = [headers for url, headers in transport.requests if url == GAME_ONE_URL]
    assert data_headers[-1]["If-None-Match"] == '"v1"'
    assert refreshed.status == "not-modified"
    assert refreshed.record == first.record
    assert len(ManifestStore(tmp_path).completed_records()) == 1


def test_refresh_falls_back_to_last_modified_condition(tmp_path: Path) -> None:
    source = _source()
    modified = "Wed, 29 Jul 2026 00:00:00 GMT"
    transport = FakeTransport(
        _routes(
            source,
            game_responses={
                GAME_ONE_URL: [
                    FakeResponse(
                        url=GAME_ONE_URL,
                        body=b"game",
                        headers={"Content-Length": "4", "Last-Modified": modified},
                    ),
                    FakeResponse(status=304, url=GAME_ONE_URL),
                ]
            },
        )
    )
    downloader = _downloader(source, tmp_path, transport)
    downloader.download(source.catalog[0])

    outcome = downloader.download(source.catalog[0], refresh=True)

    data_headers = [headers for url, headers in transport.requests if url == GAME_ONE_URL]
    assert data_headers[-1]["If-Modified-Since"] == modified
    assert outcome.status == "not-modified"


def test_retry_after_and_content_deduplication(tmp_path: Path) -> None:
    source = _source(two_games=True)
    body = b"same game bytes"
    clock = FakeClock()
    transport = FakeTransport(
        _routes(
            source,
            game_responses={
                GAME_ONE_URL: [
                    FakeResponse(
                        status=429,
                        url=GAME_ONE_URL,
                        headers={"Retry-After": "2"},
                    ),
                    FakeResponse(url=GAME_ONE_URL, body=body),
                ],
                GAME_TWO_URL: [FakeResponse(url=GAME_TWO_URL, body=body)],
            },
        )
    )

    outcomes = _downloader(source, tmp_path, transport, clock=clock).acquire_catalog(limit=2)

    assert [outcome.status for outcome in outcomes] == [
        "downloaded",
        "deduplicated-content",
    ]
    assert outcomes[0].record.object_path == outcomes[1].record.object_path
    assert any(delay == 2 for delay in clock.sleeps)

    evidence_only = FakeTransport(_routes(source))
    repeated = _downloader(source, tmp_path, evidence_only).acquire_catalog(limit=2)
    assert [outcome.status for outcome in repeated] == ["skipped-url", "skipped-url"]
    assert [url for url, _ in evidence_only.requests] == [
        ROBOTS_URL,
        RIGHTS_URL,
        INDEX_URL,
    ]


def test_acquire_catalog_instance_is_single_use(tmp_path: Path) -> None:
    source = _source()
    transport = FakeTransport(
        _routes(
            source,
            game_responses={
                GAME_ONE_URL: [FakeResponse(url=GAME_ONE_URL, body=b"game")],
            },
        )
    )
    downloader = _downloader(source, tmp_path, transport)

    outcomes = downloader.acquire_catalog(limit=1)

    assert [outcome.status for outcome in outcomes] == ["downloaded"]
    with pytest.raises(AcquisitionError, match="supports one acquisition mode"):
        downloader.acquire_catalog(limit=1)
    with pytest.raises(AcquisitionError, match="direct download is unavailable"):
        downloader.download(source.catalog[0])


def test_standalone_download_cannot_reuse_evidence_for_catalog_run(tmp_path: Path) -> None:
    source = _source(two_games=True)
    transport = FakeTransport(
        _routes(
            source,
            game_responses={
                GAME_ONE_URL: [FakeResponse(url=GAME_ONE_URL, body=b"game-one")],
            },
        )
    )
    downloader = _downloader(source, tmp_path, transport)

    downloader.download(source.catalog[0])
    request_count = len(transport.requests)

    with pytest.raises(AcquisitionError, match="fresh instance before acquire_catalog"):
        downloader.acquire_catalog(limit=2)

    assert len(transport.requests) == request_count


def test_evidence_digest_drift_refuses_before_game_request(tmp_path: Path) -> None:
    source = _source()
    evidence = tuple(
        replace(item, sha256=hashlib.sha256(QUOTE.encode()).hexdigest())
        if item.url == RIGHTS_URL
        else item
        for item in source.evidence_catalog
    )
    source = replace(source, evidence_catalog=evidence)
    transport = FakeTransport(_routes(source, rights=(QUOTE + " changed").encode()))

    with pytest.raises(AcquisitionError, match="audited SHA-256"):
        _downloader(source, tmp_path, transport).download(source.catalog[0])

    assert GAME_ONE_URL not in [url for url, _ in transport.requests]


def test_completed_record_is_rebound_to_current_policy(tmp_path: Path) -> None:
    source = _source()
    body = b"game"
    first_transport = FakeTransport(
        _routes(
            source,
            game_responses={GAME_ONE_URL: [FakeResponse(url=GAME_ONE_URL, body=body)]},
        )
    )
    _downloader(source, tmp_path, first_transport).download(source.catalog[0])
    manifest_path = tmp_path / "manifest.jsonl"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["source_id"] = "wrong-source"
    manifest_path.write_text(
        json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    evidence_transport = FakeTransport(_routes(source))

    with pytest.raises(AcquisitionError, match="current audited policy"):
        _downloader(source, tmp_path, evidence_transport).download(source.catalog[0])

    assert GAME_ONE_URL not in [url for url, _ in evidence_transport.requests]


def test_corrupt_saved_partial_is_not_used_for_range_resume(tmp_path: Path) -> None:
    source = _source()
    replacement = b"replacement"
    transport = FakeTransport(
        _routes(
            source,
            game_responses={
                GAME_ONE_URL: [
                    FakeResponse(
                        url=GAME_ONE_URL,
                        body=FailingBody(b"abcd"),
                        headers={"Content-Length": "10", "ETag": '"v1"'},
                    ),
                    FakeResponse(
                        url=GAME_ONE_URL,
                        body=replacement,
                        headers={"Content-Length": str(len(replacement)), "ETag": '"v2"'},
                    ),
                ]
            },
        )
    )
    downloader = _downloader(source, tmp_path, transport)
    with pytest.raises(AcquisitionError, match="connection loss"):
        downloader.download(source.catalog[0])
    partial_record = ManifestStore(tmp_path).snapshot().partials[-1]
    (tmp_path / partial_record.partial_path).write_bytes(b"wxyz")

    outcome = downloader.download(source.catalog[0])

    data_headers = [headers for url, headers in transport.requests if url == GAME_ONE_URL]
    assert "Range" not in data_headers[-1]
    assert "If-Range" not in data_headers[-1]
    assert (tmp_path / outcome.record.object_path).read_bytes() == replacement


def test_replaced_working_path_never_follows_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source()
    target = tmp_path / "outside-target"
    target.write_bytes(b"safe")
    transport = FakeTransport(
        _routes(
            source,
            game_responses={GAME_ONE_URL: [FakeResponse(url=GAME_ONE_URL, body=b"game")]},
        )
    )
    downloader = _downloader(source, tmp_path / "raw", transport)
    original_create = downloader._create_working_file

    def create_then_replace() -> tuple[int, Path]:
        descriptor, path = original_create()
        path.unlink()
        path.symlink_to(target)
        return descriptor, path

    monkeypatch.setattr(downloader, "_create_working_file", create_then_replace)

    with pytest.raises(AcquisitionError, match="working file was replaced"):
        downloader.download(source.catalog[0])

    assert target.read_bytes() == b"safe"
