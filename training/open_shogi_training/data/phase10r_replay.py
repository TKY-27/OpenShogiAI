"""Execute the frozen Phase 10R replay and identity upgrade.

This module is deliberately a bounded execution command rather than a new data
architecture.  It consumes the frozen C2B candidate ledger, the exact official
archives, and the already approved AobaZero object list; it writes ignored,
source-preserving replay receipts for the external-memory v2 leakage scanner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import zipfile
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from open_shogi_training.data.phase10r_identity import (
    REPLAY_PROOF_SCHEMA,
    canonical_game_hash,
    canonical_position_hash,
    history_id,
    source_game_id,
    transposition_key,
)
from open_shogi_training.data.phase10r_kif import convert_kif_to_csa
from open_shogi_training.data.phase10r_registry import Phase10RArtifact, load_phase10r_registry

REPLAY_SCHEMA = "open_shogiai_phase10r_replay_completion/v2"
POSITION_SCHEMA = "open_shogiai_phase10r_replayed_position/v2"
NORMALIZATION_VERSION = "phase10r-source-preserving/v2"
WCSC_PARSER_VERSION = "phase10r-wcsc-parser/v2"
AOBA_PARSER_VERSION = "phase10r-aobazero-parser/v2"
KIF_PARSER_VERSION = "phase10r-kif-parser/v2"
CLI_DEFAULT = "target/release/open-shogi-cli"
COPY_YAMADA_PATH = "WCSC2003/yosen1/daemon-shogi/copy/YAMADA.CSA"


class Phase10RReplayError(RuntimeError):
    """Raised when a frozen replay proof cannot be completed."""


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        + b"\n"
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise Phase10RReplayError(f"expected a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(value))


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with path.open("wb") as handle:
        for row in rows:
            encoded = _json_bytes(dict(row))
            handle.write(encoded)
            digest.update(encoded)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise Phase10RReplayError(f"missing JSONL receipt: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise Phase10RReplayError(f"non-object JSONL row at {path}:{line_number}")
            rows.append(value)
    return rows


def _safe_member(path: str) -> str:
    normalized = path.replace("\\", "/")
    parts = tuple(part for part in normalized.split("/") if part)
    if not parts or any(part in {".", ".."} or "\x00" in part for part in parts):
        raise Phase10RReplayError(f"unsafe archive member path: {path!r}")
    return "/".join(parts)


def _decode_source(raw: bytes) -> tuple[str, str]:
    for encoding in ("utf-8", "cp932"):
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1"), "binary-structural"


_CSA_MOVE = re.compile(r"^[+-](?:[0-9]{4}[A-Z]{2}|[0-9]{4}[A-Z]{2}\+)$")
_CSA_TERMINAL = re.compile(
    r"^%(?:TORYO|CHUDAN|SENNICHITE|OUTE_SENNICHITE|JISHOGI|TIME_UP|KACHI|ILLEGAL_ACTION|TSUMI|MAX_MOVES|ERROR|OTHER.*)"
)


def normalize_structural_csa(raw: bytes) -> tuple[str, dict[str, Any]]:
    """Keep only replay-relevant CSA statements without repairing moves."""

    text, encoding = _decode_source(raw)
    lines: list[str] = []
    has_position = False
    has_version = False
    has_terminal = False
    terminal: str | None = None
    transformations: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.replace("\x00", "").strip("\r")
        if not line.strip():
            continue
        stripped = line.strip()
        if has_terminal:
            continue
        if stripped.startswith("'"):
            continue
        if stripped.startswith("V"):
            if not has_version:
                lines.append("V3.0")
                has_version = True
            continue
        if stripped.startswith("N+") or stripped.startswith("N-"):
            if stripped.isascii():
                lines.append(line)
            continue
        if stripped.startswith("P"):
            if (
                stripped == "PI"
                or stripped.startswith("P1")
                or stripped.startswith("P2")
                or stripped.startswith("P3")
                or stripped.startswith("P4")
                or stripped.startswith("P5")
                or stripped.startswith("P6")
                or stripped.startswith("P7")
                or stripped.startswith("P8")
                or stripped.startswith("P9")
            ) or stripped in {"P+", "P-"}:
                lines.append(line)
                has_position = True
            continue
        if stripped in {"+", "-"}:
            lines.append(stripped)
            continue
        if _CSA_TERMINAL.match(stripped):
            terminal = stripped
            lines.append(stripped)
            has_terminal = True
            continue
        if stripped.startswith(("+", "-")):
            move = stripped.split(",", 1)[0].strip()
            if _CSA_MOVE.fullmatch(move):
                if move.endswith("+"):
                    move = move[:-1] + "+"
                lines.append(move)
                if move != stripped:
                    transformations.append("source_clock_and_comment_removed")
            continue
    if not has_version:
        lines.insert(0, "V3.0")
        transformations.append("missing_or_legacy_version_to_v3")
    if not has_position:
        lines.insert(1 if lines and lines[0] == "V3.0" else 0, "PI")
        transformations.append("missing_initial_position_to_pi")
    if not lines or not any(
        line in {"+", "-"} or _CSA_MOVE.fullmatch(line or "") for line in lines
    ):
        raise Phase10RReplayError("CSA source contains no replayable move stream")
    return (
        "'CSA encoding=UTF-8\n" + "\n".join(lines) + "\n",
        {
            "schema": "phase10r_csa_normalization/v2",
            "source_encoding": encoding,
            "source_terminal": terminal,
            "normalization_transformations": transformations,
        },
    )


class MemberStore:
    """Read exact members from ZIP and LZH archives without broad extraction."""

    def __init__(self, archive: Path) -> None:
        self.archive = archive
        self.kind = "zip" if zipfile.is_zipfile(archive) else "lzh"
        self.entries: dict[str, dict[str, Any]] = {}
        if self.kind == "zip":
            with zipfile.ZipFile(archive) as handle:
                for info in handle.infolist():
                    name = _safe_member(info.filename)
                    if info.is_dir() or name.endswith("/"):
                        continue
                    self.entries[name.casefold()] = {
                        "path": name,
                        "size_bytes": info.file_size,
                        "packed_size_bytes": info.compress_size,
                        "method": str(info.compress_type),
                        "crc32": f"{info.CRC:08x}",
                    }
        else:
            completed = subprocess.run(
                ["/opt/homebrew/bin/lha", "v", str(archive)],
                capture_output=True,
                check=False,
            )
            if completed.returncode != 0:
                raise Phase10RReplayError(completed.stderr.decode(errors="replace"))
            seen_header = False
            for raw_line in completed.stdout.decode(errors="replace").splitlines():
                if raw_line.startswith(" PERMSSN"):
                    seen_header = True
                    continue
                if not seen_header or not raw_line.strip() or raw_line.startswith("-"):
                    continue
                fields = raw_line.split()
                if len(fields) < 11:
                    continue
                member = _safe_member(" ".join(fields[10:]))
                self.entries[member.casefold()] = {
                    "path": member,
                    "size_bytes": int(fields[3]),
                    "packed_size_bytes": int(fields[2]),
                    "method": fields[5],
                    "crc32": fields[6].lower(),
                }
        if not self.entries:
            raise Phase10RReplayError(f"archive inventory is empty: {archive}")

    def entry(self, requested: str) -> dict[str, Any]:
        try:
            return self.entries[_safe_member(requested).casefold()]
        except KeyError as error:
            raise Phase10RReplayError(
                f"archive member is absent from the official archive: {requested!r}"
            ) from error

    def read(self, requested: str) -> tuple[dict[str, Any], bytes]:
        entry = self.entry(requested)
        if self.kind == "zip":
            with zipfile.ZipFile(self.archive) as handle:
                raw = handle.read(entry["path"])
        else:
            completed = subprocess.run(
                ["/opt/homebrew/bin/lha", "p", str(self.archive), entry["path"]],
                capture_output=True,
                check=False,
            )
            if completed.returncode != 0:
                raise Phase10RReplayError(completed.stderr.decode(errors="replace"))
            prefix = b"::::::::\n" + entry["path"].encode() + b"\n::::::::\n"
            if not completed.stdout.startswith(prefix):
                raise Phase10RReplayError(f"unexpected LHA member framing: {entry['path']}")
            raw = completed.stdout[len(prefix) :]
        if self.kind == "zip" and len(raw) != entry["size_bytes"]:
            raise Phase10RReplayError(f"member size changed: {entry['path']}")
        actual = _sha256_bytes(raw)
        entry = dict(entry)
        entry["payload_size_bytes"] = len(raw)
        entry["sha256"] = actual
        return entry, raw


def _registry_artifact(registry: Any, artifact_id: str) -> Phase10RArtifact:
    return registry.artifact(artifact_id)


def _raw_archive(root: Path, artifact: Phase10RArtifact) -> Path:
    candidates = sorted(
        path
        for path in (root / "local/phase10r-data/raw" / artifact.artifact_id).iterdir()
        if path.is_file() and not path.is_symlink()
    )
    if len(candidates) != 1:
        raise Phase10RReplayError(f"raw archive identity is ambiguous for {artifact.artifact_id}")
    path = candidates[0]
    if artifact.size_bytes is not None and path.stat().st_size != artifact.size_bytes:
        raise Phase10RReplayError(f"raw archive size mismatch for {artifact.artifact_id}")
    digest = _sha256_file(path)
    if artifact.sha256 is not None and digest != artifact.sha256:
        raise Phase10RReplayError(f"raw archive SHA-256 mismatch for {artifact.artifact_id}")
    return path


def _aoba_raw_manifest(root: Path) -> tuple[dict[str, dict[str, Any]], str]:
    manifest_path = (
        root
        / "local/campaign-inputs/data-processed/phase3/aobazero-no-noise-pd-sample100/manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    hashes = list(manifest["rawObjectSha256"])
    names = [f"w{value}.csa" for value in ()]
    del names
    raw_root = root / "local/phase10r-data/raw/aobazero-no-noise-exact100"
    files = sorted(
        path for path in raw_root.glob("*.csa") if path.is_file() and not path.is_symlink()
    )
    if len(files) != 100:
        raise Phase10RReplayError(f"AobaZero reacquisition is not exact100: {len(files)}")
    by_sha: dict[str, dict[str, Any]] = {}
    for path in files:
        digest = _sha256_file(path)
        if digest not in hashes:
            raise Phase10RReplayError(
                f"AobaZero object is outside the frozen object list: {path.name}"
            )
        by_sha[digest] = {
            "path": path.name,
            "path_on_disk": path,
            "size_bytes": path.stat().st_size,
        }
    if len(by_sha) != 100 or set(by_sha) != set(hashes):
        raise Phase10RReplayError(
            "AobaZero object hashes do not reconcile with the frozen manifest"
        )
    return by_sha, _sha256_file(manifest_path)


def _load_aoba_splits(root: Path) -> dict[str, str]:
    positions = (
        root / "local/campaign-inputs/data-processed/phase3/"
        "aobazero-no-noise-pd-sample100/positions-00000.jsonl.gz"
    )
    import gzip

    result: dict[str, str] = {}
    with gzip.open(positions, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            result[str(row["rawSha256"])] = {
                "train": "train",
                "validation": "validation",
                "test": "final_holdout",
            }[row["split"]]
    if len(result) != 100:
        raise Phase10RReplayError(f"AobaZero split map is not exact100: {len(result)}")
    return result


def _mapping_rows(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    staged = _read_jsonl(root / "local/phase10r-data/c2b-explore17/staged-mapping.jsonl")
    rejected = _read_jsonl(root / "local/phase10r-data/c2b-explore9/adapter-rejections.jsonl")
    if len(staged) != 5273 or len(rejected) != 344:
        raise Phase10RReplayError(
            f"frozen C2B ledger changed: staged={len(staged)} rejected={len(rejected)}"
        )
    if any(row.get("status") != "staged" for row in staged):
        raise Phase10RReplayError("staged mapping contains a non-staged row")
    return staged, rejected


def _source_terminal(row: Mapping[str, Any]) -> str:
    value = row.get("source_terminal")
    if isinstance(value, str) and value:
        return value if value.startswith("%") else "%" + value
    conversion = row.get("conversion")
    if (
        isinstance(conversion, Mapping)
        and isinstance(conversion.get("terminal"), str)
        and conversion["terminal"]
    ):
        value = str(conversion["terminal"])
        return value if value.startswith("%") else "%" + value
    return "source_terminal_absent"


def _copy_c2b_stages(root: Path, output: Path, staged_rows: list[dict[str, Any]]) -> None:
    source_root = root / "local/phase10r-data/c2b-explore17/stage"
    destination = output / "stages"
    destination.mkdir(parents=True, exist_ok=True)
    for row in staged_rows:
        source = source_root / str(row["inputFile"])
        if not source.is_file() or source.is_symlink():
            raise Phase10RReplayError(f"C2B stage file is missing: {source}")
        target = destination / str(row["artifact_id"]) / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    yamada = output / "repair-source" / "wcsc13-kifu-copy-yamada.csa"
    yamada.parent.mkdir(parents=True, exist_ok=True)
    raw_path = (
        root / "local/phase10r-data/c2b-explore/extracted/wcsc13-kifu/"
        "WCSC2003/yosen1/daemon-shogi/copy/YAMADA.CSA"
    )
    csa, _ = normalize_structural_csa(raw_path.read_bytes())
    yamada.write_text(csa, encoding="utf-8")
    manual_stage = destination / "wcsc13-kifu" / "wcsc13-kifu-copy-yamada.csa"
    manual_stage.write_text(csa, encoding="utf-8")


def _run_export(cli: Path, input_dir: Path, output: Path) -> list[dict[str, Any]]:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(f".{output.name}.phase10r.tmp")
    temporary_output.unlink(missing_ok=True)
    working_directory = Path.cwd()
    input_argument = (
        input_dir.relative_to(working_directory)
        if input_dir.is_absolute() and working_directory in input_dir.parents
        else input_dir
    )
    output_argument = (
        temporary_output.relative_to(working_directory)
        if temporary_output.is_absolute() and working_directory in temporary_output.parents
        else temporary_output
    )
    completed = subprocess.run(
        [
            str(cli),
            "export-csa-jsonl",
            "--input-dir",
            str(input_argument),
            "--output",
            str(output_argument),
            "--max-games",
            "10000",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise Phase10RReplayError(completed.stderr or completed.stdout)
    temporary_output.replace(output)
    return _read_jsonl(output)


def _proof_and_positions(
    *,
    artifact: Phase10RArtifact,
    source_id: str,
    source_revision: str,
    official_sha: str,
    member_path: str,
    member_sha: str,
    record_sha: str,
    source_terminal: str,
    parser_version: str,
    normalization_version: str,
    input_sha: str,
    exported: Mapping[str, Any],
    split: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if exported.get("status") != "ok":
        raise Phase10RReplayError(f"cannot make proof for rejected replay: {exported}")
    initial = str(exported["initialSfen"])
    moves = [str(move) for move in exported["usiMoves"]]
    sfens = [str(sfen) for sfen in exported["positionSfens"]]
    if len(sfens) != len(moves) + 1:
        raise Phase10RReplayError(f"position/move count mismatch for {member_path}")
    source_id_value = source_game_id(official_sha, member_path, member_sha)
    game_hash = canonical_game_hash(initial, moves)
    canonical_hashes = [canonical_position_hash(sfen) for sfen in sfens]
    transpositions = [transposition_key(sfen) for sfen in sfens]
    histories = [history_id(initial, moves[:index]) for index in range(len(sfens))]
    proof = {
        "schema": REPLAY_PROOF_SCHEMA,
        "official_artifact_sha256": official_sha,
        "archive_member_path": member_path,
        "archive_member_sha256": member_sha,
        "source_game_id": source_id_value,
        "parser_version": parser_version,
        "normalization_version": normalization_version,
        "canonical_initial_sfen": initial,
        "usi_moves": moves,
        "complete_legal_replay": True,
        "canonical_game_hash": game_hash,
        "canonical_position_hashes": canonical_hashes,
        "transposition_keys": transpositions,
        "history_ids": histories,
        "terminal": source_terminal,
    }
    positions: list[dict[str, Any]] = []
    for index, (sfen, canonical, transposition, history) in enumerate(
        zip(sfens, canonical_hashes, transpositions, histories, strict=True)
    ):
        side = sfen.split(" ")[1]
        positions.append(
            {
                "schema": POSITION_SCHEMA,
                "artifact_id": artifact.artifact_id,
                "artifact_sha256": official_sha,
                "source_id": source_id,
                "source_revision": source_revision,
                "archive_member_path": member_path,
                "archive_member_sha256": member_sha,
                "source_game_id": source_id_value,
                "record_id": member_sha,
                "record_sha256": record_sha,
                "input_sha256": input_sha,
                "game_id": game_hash,
                "position_index": index,
                "canonical_position_id": canonical,
                "canonical_sfen": sfen,
                "history_id": history,
                "transposition_key": transposition,
                "side_to_move": "black" if side == "b" else "white",
                "split": split,
                "protected_role": split,
                "parser_version": parser_version,
                "normalization_version": normalization_version,
            }
        )
    return proof, positions


def _source_games_and_splits(
    records: list[dict[str, Any]], aoba_splits: Mapping[str, str]
) -> dict[str, str]:
    """Assign non-Aoba game groups while preserving protected Aoba splits."""

    result: dict[str, str] = {}
    groups_by_source: dict[str, set[str]] = defaultdict(set)
    for record in records:
        game = str(record["game_id"])
        if record["source_id"] == "aobazero":
            result[game] = str(record["split"])
        else:
            groups_by_source[str(record["source_id"])].add(game)
    occupied = set(result)
    for source, groups in sorted(groups_by_source.items()):
        ordered = sorted(
            groups, key=lambda value: _sha256_bytes(f"phase10r-split-v2:{source}:{value}".encode())
        )
        held = max(1, math.ceil(len(ordered) * 0.10))
        validation = max(1, math.ceil(len(ordered) * 0.10))
        for index, game in enumerate(ordered):
            if game in occupied:
                continue
            result[game] = (
                "source_held_out"
                if index < held
                else "validation"
                if index < held + validation
                else "train"
            )
    return result


def _archive_inventory(
    root: Path, artifacts: Iterable[Phase10RArtifact], selected: Mapping[tuple[str, str], str]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for artifact in artifacts:
        archive = _raw_archive(root, artifact)
        store = MemberStore(archive)
        for entry in sorted(store.entries.values(), key=lambda value: value["path"].casefold()):
            key = (artifact.artifact_id, entry["path"].casefold())
            rows.append(
                {
                    "artifact_id": artifact.artifact_id,
                    "official_artifact_sha256": artifact.sha256,
                    "archive_member_path": entry["path"],
                    "size_bytes": entry["size_bytes"],
                    "packed_size_bytes": entry["packed_size_bytes"],
                    "method": entry["method"],
                    "crc32": entry["crc32"],
                    "scope": selected.get(key, "inventory_only_outside_frozen_c2b_scope"),
                }
            )
    return rows


def run_replay(root: Path, cli_path: Path) -> dict[str, Any]:
    registry = load_phase10r_registry(root / "configs/phase10r/source-registry.yaml")
    artifacts = registry.approved_artifacts()
    if len(artifacts) != 33:
        raise Phase10RReplayError(f"frozen approved population changed: {len(artifacts)}")
    output = root / "local/phase10r-data/replay-v2"
    output.mkdir(parents=True, exist_ok=True)
    staged_rows, adapter_rows = _mapping_rows(root)
    selected: dict[tuple[str, str], str] = {}
    for row in [*staged_rows, *adapter_rows]:
        selected[
            (str(row["artifact_id"]), _safe_member(str(row["archive_member_path"])).casefold())
        ] = "frozen_c2b_candidate"
    selected[("wcsc13-kifu", COPY_YAMADA_PATH.casefold())] = "approved_repairable_replacement"
    inventory = _archive_inventory(
        root, [artifact for artifact in artifacts if artifact.source_id != "aobazero"], selected
    )
    _write_jsonl(output / "official-member-inventory.jsonl", inventory)
    aoba_objects, aoba_manifest_sha = _aoba_raw_manifest(root)
    aoba_splits = _load_aoba_splits(root)
    _write_json(
        output / "aobazero-reacquisition.json",
        {
            "schema": "open_shogiai_phase10r_aobazero_reacquisition/v2",
            "manifest_sha256": aoba_manifest_sha,
            "object_count": len(aoba_objects),
            "objects": [
                {
                    "object_sha256": digest,
                    "archive_member_path": value["path"],
                    "size_bytes": value["size_bytes"],
                }
                for digest, value in sorted(aoba_objects.items())
            ],
        },
    )
    _copy_c2b_stages(root, output, staged_rows)
    replay_root = output / "replays"
    proofs_root = output / "proofs"
    positions_root = output / "positions"
    stream_summaries: dict[str, dict[str, Any]] = {}
    accepted_records: list[dict[str, Any]] = []
    for artifact in artifacts:
        if artifact.source_id == "aobazero":
            stage = output / "stages" / artifact.artifact_id
            stage.mkdir(parents=True, exist_ok=True)
            aoba_rows: list[dict[str, Any]] = []
            for _index, (digest, value) in enumerate(sorted(aoba_objects.items())):
                csa, _ = normalize_structural_csa(value["path_on_disk"].read_bytes())
                (stage / value["path"]).write_text(csa, encoding="utf-8")
                aoba_rows.append({"inputFile": value["path"], "object_sha256": digest})
            export_path = replay_root / f"{artifact.artifact_id}.jsonl"
            exported = _run_export(cli_path, stage, export_path)
            by_input = {str(row.get("inputFile")): row for row in exported}
            if len(exported) != 100 or any(row.get("status") != "ok" for row in exported):
                raise Phase10RReplayError("AobaZero replay did not accept all 100 objects")
            proof_rows: list[dict[str, Any]] = []
            position_rows: list[dict[str, Any]] = []
            for source in aoba_rows:
                digest = str(source["object_sha256"])
                exported_row = by_input[str(source["inputFile"])]
                proof, positions = _proof_and_positions(
                    artifact=artifact,
                    source_id="aobazero",
                    source_revision=artifact.source_revision,
                    official_sha=digest,
                    member_path=str(source["inputFile"]),
                    member_sha=digest,
                    record_sha=_sha256_bytes(str(exported_row["normalizedCsa"]).encode()),
                    source_terminal="source_terminal_absent",
                    parser_version=AOBA_PARSER_VERSION,
                    normalization_version=NORMALIZATION_VERSION,
                    input_sha=digest,
                    exported=exported_row,
                    split=aoba_splits[digest],
                )
                proof_rows.append(proof)
                position_rows.extend(positions)
                accepted_records.append(
                    {
                        "artifact_id": artifact.artifact_id,
                        "source_id": "aobazero",
                        "game_id": proof["canonical_game_hash"],
                        "split": aoba_splits[digest],
                    }
                )
            proof_sha = _write_jsonl(proofs_root / f"{artifact.artifact_id}.jsonl", proof_rows)
            position_sha = _write_jsonl(
                positions_root / f"{artifact.artifact_id}.jsonl", position_rows
            )
            stream_summaries[artifact.artifact_id] = {
                "status": "complete",
                "source_id": "aobazero",
                "source_revision": artifact.source_revision,
                "official_artifact_sha256": None,
                "accepted_games": 100,
                "excluded_games": 0,
                "position_count": len(position_rows),
                "proof_count": len(proof_rows),
                "replay_sha256": _sha256_file(export_path),
                "proof_sha256": proof_sha,
                "position_sha256": position_sha,
                "parser_version": AOBA_PARSER_VERSION,
                "normalization_version": NORMALIZATION_VERSION,
            }
            continue
        if artifact.artifact_id == "denryu-dr5-hardware3":
            archive = _raw_archive(root, artifact)
            store = MemberStore(archive)
            kif_members = [
                entry
                for entry in store.entries.values()
                if entry["path"].casefold().startswith("kifu_dr5hdw3/kif/")
                and entry["path"].casefold().endswith(".kif")
            ]
            if len(kif_members) != 132:
                raise Phase10RReplayError(f"Denryu KIF member count changed: {len(kif_members)}")
            stage = output / "stages" / artifact.artifact_id
            stage.mkdir(parents=True, exist_ok=True)
            denryu_rows: list[dict[str, Any]] = []
            for index, entry in enumerate(
                sorted(kif_members, key=lambda value: value["path"].casefold())
            ):
                actual, raw = store.read(entry["path"])
                csa, conversion = convert_kif_to_csa(raw)
                input_file = f"denryu-{index:04d}.csa"
                (stage / input_file).write_text(csa, encoding="utf-8")
                denryu_rows.append(
                    {
                        "inputFile": input_file,
                        "archive_member_path": actual["path"],
                        "archive_member_sha256": _sha256_bytes(raw),
                        "source_terminal": conversion.get("terminal"),
                        "source_format": "kif",
                        "conversion": conversion,
                    }
                )
            export_path = replay_root / f"{artifact.artifact_id}.jsonl"
            exported = _run_export(cli_path, stage, export_path)
            by_input = {str(row.get("inputFile")): row for row in exported}
            if len(exported) != 132 or any(row.get("status") != "ok" for row in exported):
                raise Phase10RReplayError("Denryu replay did not accept all 132 KIF members")
            proof_rows: list[dict[str, Any]] = []
            position_rows: list[dict[str, Any]] = []
            for row in denryu_rows:
                actual = by_input[row["inputFile"]]
                proof, positions = _proof_and_positions(
                    artifact=artifact,
                    source_id=artifact.source_id,
                    source_revision=artifact.source_revision,
                    official_sha=str(artifact.sha256),
                    member_path=str(row["archive_member_path"]),
                    member_sha=str(row["archive_member_sha256"]),
                    record_sha=_sha256_bytes(str(actual["normalizedCsa"]).encode()),
                    source_terminal=_source_terminal(row),
                    parser_version=KIF_PARSER_VERSION,
                    normalization_version=NORMALIZATION_VERSION,
                    input_sha=_sha256_bytes(str(actual["normalizedCsa"]).encode()),
                    exported=actual,
                    split="train",
                )
                proof_rows.append(proof)
                position_rows.extend(positions)
                accepted_records.append(
                    {
                        "artifact_id": artifact.artifact_id,
                        "source_id": artifact.source_id,
                        "game_id": proof["canonical_game_hash"],
                        "split": "train",
                    }
                )
            proof_sha = _write_jsonl(proofs_root / f"{artifact.artifact_id}.jsonl", proof_rows)
            position_sha = _write_jsonl(
                positions_root / f"{artifact.artifact_id}.jsonl", position_rows
            )
            stream_summaries[artifact.artifact_id] = {
                "status": "complete",
                "source_id": artifact.source_id,
                "source_revision": artifact.source_revision,
                "official_artifact_sha256": artifact.sha256,
                "accepted_games": len(proof_rows),
                "excluded_games": 0,
                "candidate_games": len(denryu_rows),
                "position_count": len(position_rows),
                "proof_count": len(proof_rows),
                "replay_sha256": _sha256_file(export_path),
                "proof_sha256": proof_sha,
                "position_sha256": position_sha,
                "parser_version": KIF_PARSER_VERSION,
                "normalization_version": NORMALIZATION_VERSION,
            }
            continue
        archive = _raw_archive(root, artifact)
        store = MemberStore(archive)
        stage = output / "stages" / artifact.artifact_id
        export_path = replay_root / f"{artifact.artifact_id}.jsonl"
        exported = _run_export(cli_path, stage, export_path)
        by_input = {str(row.get("inputFile")): row for row in exported}
        rows_for_artifact = [
            row for row in staged_rows if row["artifact_id"] == artifact.artifact_id
        ]
        if len(exported) != len(rows_for_artifact) + (
            1 if artifact.artifact_id == "wcsc13-kifu" else 0
        ):
            raise Phase10RReplayError(f"replay row count mismatch for {artifact.artifact_id}")
        validation = {
            str(row["inputFile"]): row
            for row in _read_jsonl(root / "local/phase10r-data/c2b-explore17/validation.jsonl")
            if row.get("inputFile") in {r["inputFile"] for r in rows_for_artifact}
        }
        accepted_rows: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
        rejected_rows: list[dict[str, Any]] = []
        for row in rows_for_artifact:
            name = str(row["inputFile"])
            actual = by_input[name]
            expected = validation[name]
            if actual.get("status") != expected.get("status"):
                raise Phase10RReplayError(
                    f"C2B replay result changed for {artifact.artifact_id}/{name}"
                )
            entry, raw = store.read(str(row["archive_member_path"]))
            if _sha256_bytes(raw) != row["archive_member_sha256"]:
                raise Phase10RReplayError(
                    f"C2B member hash mismatch: {artifact.artifact_id}/{name}"
                )
            if actual.get("status") == "ok":
                accepted_rows.append((row, actual, entry))
            else:
                rejected_rows.append(
                    {
                        "artifact_id": artifact.artifact_id,
                        "archive_member_path": entry["path"],
                        "archive_member_sha256": row["archive_member_sha256"],
                        "inputFile": name,
                        "status": "exact_replay_excluded",
                        "reason": actual.get("reason", "rejected"),
                    }
                )
        if artifact.artifact_id == "wcsc13-kifu":
            manual_name = "wcsc13-kifu-copy-yamada.csa"
            actual = by_input.get(manual_name)
            entry, raw = store.read(COPY_YAMADA_PATH)
            if actual is None or actual.get("status") != "ok":
                raise Phase10RReplayError("repairable WCSC13 copy/YAMADA member did not replay")
            accepted_rows.append(
                (
                    {
                        "inputFile": manual_name,
                        "archive_member_path": entry["path"],
                        "archive_member_sha256": _sha256_bytes(raw),
                        "source_terminal": None,
                    },
                    actual,
                    entry,
                )
            )
        proof_rows = []
        position_rows = []
        for row, actual, entry in accepted_rows:
            proof, positions = _proof_and_positions(
                artifact=artifact,
                source_id=artifact.source_id,
                source_revision=artifact.source_revision,
                official_sha=str(artifact.sha256),
                member_path=entry["path"],
                member_sha=str(row["archive_member_sha256"]),
                record_sha=_sha256_bytes(str(actual["normalizedCsa"]).encode()),
                source_terminal=_source_terminal(row),
                parser_version=KIF_PARSER_VERSION
                if str(row.get("source_format")) == "kif"
                else WCSC_PARSER_VERSION,
                normalization_version=NORMALIZATION_VERSION,
                input_sha=_sha256_bytes(str(actual["normalizedCsa"]).encode()),
                exported=actual,
                split="train",
            )
            proof_rows.append(proof)
            position_rows.extend(positions)
            accepted_records.append(
                {
                    "artifact_id": artifact.artifact_id,
                    "source_id": artifact.source_id,
                    "game_id": proof["canonical_game_hash"],
                    "split": "train",
                }
            )
        adapter_attempts = []
        for row in adapter_rows:
            if row["artifact_id"] != artifact.artifact_id:
                continue
            if (
                _safe_member(str(row["archive_member_path"])).casefold()
                == COPY_YAMADA_PATH.casefold()
            ):
                continue
            entry, raw = store.read(str(row["archive_member_path"]))
            if _sha256_bytes(raw) != row["archive_member_sha256"]:
                raise Phase10RReplayError(
                    f"adapter rejection member hash mismatch: {entry['path']}"
                )
            adapter_attempts.append(
                {
                    "artifact_id": artifact.artifact_id,
                    "archive_member_path": entry["path"],
                    "archive_member_sha256": row["archive_member_sha256"],
                    "status": "exact_replay_excluded",
                    "reason": row.get("reason", "adapter_rejected"),
                    "source_format": row.get("source_format"),
                }
            )
        rejected_rows.extend(adapter_attempts)
        _write_jsonl(output / "replay-exclusions" / f"{artifact.artifact_id}.jsonl", rejected_rows)
        proof_sha = _write_jsonl(proofs_root / f"{artifact.artifact_id}.jsonl", proof_rows)
        position_sha = _write_jsonl(positions_root / f"{artifact.artifact_id}.jsonl", position_rows)
        stream_summaries[artifact.artifact_id] = {
            "status": "complete",
            "source_id": artifact.source_id,
            "source_revision": artifact.source_revision,
            "official_artifact_sha256": artifact.sha256,
            "accepted_games": len(proof_rows),
            "excluded_games": len(rejected_rows),
            "candidate_games": len(rows_for_artifact)
            + sum(1 for row in adapter_rows if row["artifact_id"] == artifact.artifact_id),
            "position_count": len(position_rows),
            "proof_count": len(proof_rows),
            "replay_sha256": _sha256_file(export_path),
            "proof_sha256": proof_sha,
            "position_sha256": position_sha,
            "parser_version": WCSC_PARSER_VERSION,
            "normalization_version": NORMALIZATION_VERSION,
        }
    split_map = _source_games_and_splits(accepted_records, aoba_splits)
    for artifact in artifacts:
        path = positions_root / f"{artifact.artifact_id}.jsonl"
        rows = _read_jsonl(path)
        changed = False
        for row in rows:
            split = split_map[row["game_id"]]
            if row["split"] != split or row["protected_role"] != split:
                row["split"] = split
                row["protected_role"] = split
                changed = True
        if changed:
            stream_summaries[artifact.artifact_id]["position_sha256"] = _write_jsonl(path, rows)
    completion = {
        "schema": REPLAY_SCHEMA,
        "status": "complete",
        "approved_stream_count": len(artifacts),
        "completed_stream_count": len(stream_summaries),
        "streams": stream_summaries,
        "c2b_candidate_scope": {
            "staged": len(staged_rows),
            "adapter_rejected": len(adapter_rows),
            "candidate_members": len(staged_rows) + len(adapter_rows),
            "accepted_games": sum(
                v["accepted_games"] for k, v in stream_summaries.items() if k.startswith("wcsc")
            ),
            "denryu_games": sum(
                v["accepted_games"]
                for k, v in stream_summaries.items()
                if k == "denryu-dr5-hardware3"
            ),
            "exactly_excluded_members": sum(
                v["excluded_games"] for k, v in stream_summaries.items() if k.startswith("wcsc")
            ),
        },
        "repairable_replacement": {
            "artifact_id": "wcsc13-kifu",
            "archive_member_path": COPY_YAMADA_PATH,
            "status": "accepted",
        },
        "inventory_member_count": len(inventory),
        "inventory_outside_frozen_c2b_scope": sum(
            1 for row in inventory if row["scope"] == "inventory_only_outside_frozen_c2b_scope"
        ),
        "source_registry_sha256": registry.sha256,
    }
    completion["scan_identity"] = _sha256_bytes(_json_bytes(completion))
    _write_json(output / "replay-completion.json", completion)
    _write_json(
        output / "replay-scan-identity.json",
        {
            "schema": "open_shogiai_phase10r_scan_identity/v2",
            "replay_completion_sha256": _sha256_file(output / "replay-completion.json"),
            "source_registry_sha256": registry.sha256,
        },
    )
    return completion


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--cli", type=Path, default=Path(CLI_DEFAULT))
    args = parser.parse_args()
    root = args.root.resolve()
    cli = args.cli if args.cli.is_absolute() else root / args.cli
    completion = run_replay(root, cli)
    print(
        json.dumps(
            {
                "status": completion["status"],
                "streams": completion["completed_stream_count"],
                "scan_identity": completion["scan_identity"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
