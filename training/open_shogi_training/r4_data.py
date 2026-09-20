"""R4 value data: immutable replay, audited public root values, and local relabels.

All old partitions survive. Public annotations are root-player probabilities, not
leaf cp labels. The final holdout is never opened, including during reacquisition.
"""

from __future__ import annotations

import gzip
import json
import math
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

from .data.aobazero import adapt_aobazero_csa
from .evaluator_data import MAX_FEATURES, atomic, digest, encoded, symmetry_keys
from .evaluator_training import GROUPS
from .phase10v_model import sparse_features

SPLITS = ("train", "validation", "development_test")


def reference(root: Path, path: Path, expected: str | None = None) -> dict:
    requested = path if path.is_absolute() else root / path
    if any(p.is_symlink() for p in (requested, *requested.parents)):
        raise ValueError("linked R4 input")
    path = requested.resolve(strict=True)
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("R4 input outside repository")
    sha = digest(path)
    if expected is not None and sha != expected:
        raise ValueError(f"R4 input changed: {path.relative_to(root)}")
    return {"path": str(path.relative_to(root)), "sha256": sha}


def verify_dataset(root: Path, path: Path, expected: str | None = None) -> dict:
    reference(root, path / "manifest.json", expected)
    report = json.loads((path / "manifest.json").read_text())
    if report.get("schema") != "open_shogiai_r4_data/v1":
        raise ValueError("unknown R4 dataset")
    for ref in report["artifacts"]:
        if Path(ref["path"]).name != ref["path"]:
            raise ValueError("dataset member escapes directory")
        reference(root, path / ref["path"], ref["sha256"])
    for ref in report["inputs"]:
        reference(root, Path(ref["path"]), ref["sha256"])
    return report


def import_dataset(root: Path, output: Path, generation: dict) -> dict:
    ref = generation["prepared_dataset"]
    source = root / ref["path"]
    report = verify_dataset(root, source, ref["manifest_sha256"])
    destination = output / "dataset"
    if destination.exists():
        return verify_dataset(root, destination, ref["manifest_sha256"])
    # Retry only this owned snapshot, rejecting links and unexpected staging files.
    building = output / "dataset-building"
    if building.exists():
        for member in [building, *building.rglob("*")]:
            if member.is_symlink() or (
                member != building and not (source / member.relative_to(building)).is_file()
            ):
                raise ValueError("unsafe R4 staging member")
    shutil.copytree(source, building, dirs_exist_ok=True)
    verify_dataset(root, building, ref["manifest_sha256"])
    building.rename(destination)
    return report


def probability_cp(value: float) -> int:
    """Pinned upstream winrate_to_score convention, from the pre-move player's view."""
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("invalid public search probability")
    if value > 0.9997:
        return 5000
    if value < 0.0003:
        return -5000
    return int(600 * math.log(value / (1 - value)))


def public_rows(root: Path, folder: Path, cli: Path) -> tuple[list[dict], list[dict]]:
    import re

    selected = json.loads((folder / "selected.json").read_text())
    if len(selected["records"]) != 90:
        raise ValueError("R4 expects the 72 historical train and 18 validation public games")
    ledger = root / "local/frozen/metadata/game-partitions.jsonl.gz"
    refs = [
        reference(root, folder / "selected.json"),
        reference(root, ledger, selected["partitions_sha256"]),
    ]
    with gzip.open(ledger, "rt") as stream:
        next(stream)
        partitions = {r[4]: r for line in stream if (r := json.loads(line))[2] == "aobazero"}
    staged = folder / "normalized"
    staged.mkdir(exist_ok=True)
    records, annotations = {}, {}
    for entry in selected["records"]:
        old, record = partitions[entry["name"]], entry["record"]
        if (
            old[8] not in ("train", "validation")
            or old[8] != entry["partition"]
            or old[7] != entry["game_id"]
        ):
            raise ValueError("protected public source or changed source-game split")
        path = folder / record["object_path"]
        refs.append(reference(root, path, old[5]))
        if (
            record["sha256"] != old[5]
            or not record["machine_learning_allowed"]
            or record["license"] != "Public Domain"
        ):
            raise ValueError("public source identity/rights mismatch")
        for evidence in record["evidence_snapshots"]:
            ref = reference(root, folder / evidence["object_path"], evidence["sha256"])
            if ref not in refs:
                refs.append(ref)
        raw = path.read_bytes()
        adapted = adapt_aobazero_csa(raw)
        destination = staged / entry["name"]
        if destination.exists() and destination.read_text() != adapted.csa:
            raise ValueError("public normalized record changed")
        atomic(destination, adapted.csa.encode())
        matches = [
            re.fullmatch(r"([+-][0-9]{4}[A-Z]{2}),?'?v=([0-9.]+),.*", line)
            for line in raw.decode().splitlines()
            if re.match(r"^[+-][0-9]{4}[A-Z]{2}", line)
        ]
        if len(matches) != adapted.move_count or any(m is None for m in matches):
            raise ValueError("unknown public annotation dialect")
        annotations[entry["name"]] = [float(m[2]) for m in matches]
        records[entry["name"]] = entry
    exported = folder / "replayed.jsonl"
    if not exported.exists():
        subprocess.run(
            [
                str(cli),
                "export-csa-jsonl",
                "--input-dir",
                str(staged),
                "--output",
                str(exported),
                "--max-games",
                "90",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
    refs.extend((reference(root, exported), reference(root, cli)))
    rows, seen = [], set()
    for line in exported.read_text().splitlines():
        game = json.loads(line)
        name = Path(game["inputFile"]).name
        if game["status"] != "ok" or name not in records or name in seen:
            raise ValueError("public game replay failed or duplicated")
        seen.add(name)
        entry, values = records[name], annotations[name]
        if len(game["usiMoves"]) != len(values) or len(game["positionSfens"]) != len(values) + 1:
            raise ValueError("public root/move alignment mismatch")
        for ply, (sfen, value) in enumerate(zip(game["positionSfens"][:-1], values, strict=True)):
            # Broad stage strata only; no runtime opening recognition or score adjustment.
            group = "opening" if ply < 40 else "general" if ply % 4 == 0 else "attack_end"
            rows.append(
                {
                    "sfen": sfen,
                    "score": {"kind": "cp", "value": probability_cp(value)},
                    "score_perspective": "side_to_move",
                    "kind": "root",
                    "game": f"aobazero:{entry['game_id']}",
                    "family": f"aobazero:{entry['game_id']}",
                    "group": group,
                    "ply": ply,
                    "move": game["usiMoves"][ply],
                    "split": entry["partition"],
                    "source": "aobazero",
                    "source_sha256": entry["record"]["sha256"],
                    "source_row": ply,
                    "root_win_probability": value,
                    "teacher_label_origin": "public_searched_root_600_logit",
                    "round_source": 1,
                }
            )
    if seen != set(records):
        raise ValueError("public replay omitted a source game")
    return rows, refs


def build_pool(root: Path, plan: dict, output: Path) -> dict:
    """The same R4 assembly path with a pinned external pool instead of C1 relabels."""
    return build(root, plan, output)


def build(root: Path, plan: dict, output: Path) -> dict:
    """Prepare once from actual pinned data; do not reassign any historical split."""
    if output.exists():
        return verify_dataset(root, output)
    replay = root / plan["replay"]["path"]
    refs = [reference(root, replay / "manifest.json", plan["replay"]["manifest_sha256"])]
    refs.extend(reference(root, Path(p)) for p in plan.get("provenance", []))
    old = json.loads((replay / "manifest.json").read_text())
    refs.extend(reference(root, replay / r["path"], r["sha256"]) for r in old["artifacts"])
    external = bool(plan.get("shards"))
    if external:
        from .r4_sources import rows

        incoming = []
        for shard in plan["shards"]:
            path = root / shard["path"]
            refs.append(reference(root, path, shard["sha256"]))
            incoming.extend(rows(path, shard, plan["stride"]))
            print(json.dumps({"decoded": shard["path"], "spaced_rows": len(incoming)}), flush=True)
        focus_data = {"policy": "no mandatory relabel; valid published root labels reused"}
    else:
        incoming, public_refs = public_rows(root, root / plan["public"], root / plan["cli"])
        refs.extend(public_refs)
        focus = root / plan["focus"]
        refs.append(reference(root, focus))
        focus_data = json.loads(focus.read_text())
        incoming.extend(focus_data["rows"])
        for ref in focus_data["inputs"]:
            refs.append(reference(root, Path(ref["path"]), ref["sha256"]))
    guard = root / plan["split_guard"]
    refs.append(reference(root, guard))
    blocked = {
        r["position_sha256"]
        for rows in json.loads(guard.read_text()).values()
        if isinstance(rows, list)
        for r in rows
        if isinstance(r, dict) and "position_sha256" in r
    }
    exclusions = root / plan["exclusions"]
    refs.append(reference(root, exclusions))
    blocked.update(json.loads(exclusions.read_text())["symmetry_keys"])
    owners, conflicts = {}, set()
    retained = {s: set() for s in SPLITS}
    # Existing symmetry keys were certified by the original prepare. Their bytes
    # are pinned above. New states are checked against every old partition.
    for split in SPLITS:
        with gzip.open(replay / f"{split}-rows.jsonl.gz", "rt") as stream:
            for index, line in enumerate(stream):
                row = json.loads(line)
                key = row["symmetry_key"]
                if key in owners:
                    raise ValueError("replay split/dedup invariant failed")
                owners[key] = split
                if not external or split == "train" or int(key[:8], 16) % 24 == 0:
                    retained[split].add(index)
    existing_keys = set(owners)
    for row in incoming:
        keys = symmetry_keys(row["sfen"])
        key = row["symmetry_key"] = min(keys)
        if row["split"] not in (SPLITS if external else ("train", "validation")):
            raise ValueError("new data cannot open or reassign heldout data")
        if keys & blocked or (key in owners and owners[key] != row["split"]):
            conflicts.add(key)
        owners.setdefault(key, row["split"])
    # Never remove old evaluation rows because a new training source overlaps.
    # Instead reject the incoming conflicting state, including both new owners.
    new_rows, used, duplicates = {s: [] for s in SPLITS}, set(), 0
    for row in sorted(
        incoming, key=lambda r: (r["source"] != "r4_focus", str(r["game"]), r["ply"])
    ):
        key = row["symmetry_key"]
        if key in conflicts or key in used:
            duplicates += 1
            continue
        if key in existing_keys and row["source"] != "r4_focus":
            duplicates += 1
            continue
        if row["source"] == "r4_focus" and (
            row["split"] != "train"
            or row["family"]
            not in {f["family"] for f in old["source_games"] if f["split"] == "train"}
        ):
            raise ValueError("focus relabel leaves original train family")
        used.add(key)
        new_rows[row["split"]].append(row)
    if external:
        # Validation is a fixed hash-selected subsample, never a model-selected prefix.
        for split in ("validation", "development_test"):
            new_rows[split] = sorted(new_rows[split], key=lambda r: r["symmetry_key"])[:8192]
    replaced = used & existing_keys
    counts = {s: len(retained[s]) for s in SPLITS}
    counts["train"] -= len(replaced)
    for split in SPLITS:
        counts[split] += len(new_rows[split])
    building = output.with_name(output.name + "-building")
    building.mkdir(parents=True)
    sources, distributions, actual = {}, {}, {}
    for split in SPLITS:
        arrays = {
            "features": np.lib.format.open_memmap(
                building / f"{split}-features.npy",
                mode="w+",
                dtype="uint16",
                shape=(counts[split], 2, MAX_FEATURES),
            ),
            "lengths": np.lib.format.open_memmap(
                building / f"{split}-lengths.npy",
                mode="w+",
                dtype="uint8",
                shape=(counts[split], 2),
            ),
            "targets": np.lib.format.open_memmap(
                building / f"{split}-targets.npy",
                mode="w+",
                dtype="float32",
                shape=(counts[split],),
            ),
            "groups": np.lib.format.open_memmap(
                building / f"{split}-groups.npy", mode="w+", dtype="uint8", shape=(counts[split],)
            ),
            "sources": np.lib.format.open_memmap(
                building / f"{split}-sources.npy", mode="w+", dtype="uint8", shape=(counts[split],)
            ),
        }
        if external:
            for name in ("sequences", "origins"):
                arrays[name] = np.lib.format.open_memmap(
                    building / f"{split}-{name}.npy",
                    mode="w+",
                    dtype="uint32",
                    shape=(counts[split],),
                )
        old_arrays = {
            k: np.load(replay / f"{split}-{k}.npy", mmap_mode="r", allow_pickle=False)
            for k in arrays
            if k not in ("sources", "sequences", "origins")
        }
        source_counts, group_counts, lineages = Counter(), Counter(), set()
        index, sequence_ids = 0, {}

        def lineage(row, new, index, arrays=arrays, sequence_ids=sequence_ids):
            if external:
                key = f"{row['source']}:{row['game']}"
                arrays["sequences"][index] = sequence_ids.setdefault(key, len(sequence_ids))
                arrays["origins"][index] = 2 if new else 0 if row["source"] == "generated" else 1

        with gzip.open(building / f"{split}-rows.jsonl.gz", "wb") as destination:
            with gzip.open(replay / f"{split}-rows.jsonl.gz", "rt") as stream:
                for source_index, line in enumerate(stream):
                    row = json.loads(line)
                    if source_index not in retained[split] or row["symmetry_key"] in replaced:
                        continue
                    for kind, values in old_arrays.items():
                        arrays[kind][index] = values[source_index]
                    arrays["sources"][index] = 0
                    row.update(round_source=0, split=split)
                    lineage(row, False, index)
                    destination.write(encoded(row) + b"\n")
                    source_counts["replay"] += 1
                    group_counts[row["group"]] += 1
                    lineages.add(str(row["family"]))
                    index += 1
            for row in new_rows[split]:
                value = row["score"]
                if (
                    value["kind"] != "cp"
                    or type(value["value"]) is not int
                    or abs(value["value"]) > 28999
                ):
                    raise ValueError("invalid new scalar label")
                black, white, stm = sparse_features(row["sfen"])
                for side, features in enumerate((white, black) if stm else (black, white)):
                    if not 0 < len(features) <= MAX_FEATURES:
                        raise ValueError("invalid feature capacity")
                    arrays["features"][index, side, : len(features)] = features
                    arrays["lengths"][index, side] = len(features)
                arrays["targets"][index] = np.clip(value["value"], -20000, 20000)
                arrays["groups"][index] = GROUPS.index(row["group"])
                arrays["sources"][index] = 1
                lineage(row, True, index)
                destination.write(encoded(row) + b"\n")
                source_counts[row["source"]] += 1
                group_counts[row["group"]] += 1
                lineages.add(str(row["family"]))
                index += 1
        if index != counts[split]:
            raise ValueError("R4 prepared count mismatch")
        for array in arrays.values():
            array.flush()
        sources[split], distributions[split], actual[split] = (
            dict(source_counts),
            dict(group_counts),
            len(lineages),
        )
    report = {
        "schema": "open_shogiai_r4_data/v1",
        "plan": plan,
        "inputs": refs,
        "artifacts": [
            {"path": p.name, "sha256": digest(p), "bytes": p.stat().st_size}
            for p in sorted(building.iterdir())
        ],
        "unique_positions": counts,
        "source_rows": sources,
        "distributions": distributions,
        "source_families": actual,
        "source_games": old["source_games"]
        + (
            list(
                {
                    (r["game"], r["split"]): {
                        "game": r["game"],
                        "family": r["family"],
                        "split": r["split"],
                    }
                    for r in new_rows["development_test"]
                }.values()
            )
            if external
            else []
        ),
        "replay_manifest_sha256": digest(replay / "manifest.json"),
        "new_unique_positions": sum(len(v) for v in new_rows.values()) - len(replaced),
        "relabeled_existing_train": len(replaced),
        "rejected_new_conflicts": len(conflicts),
        "duplicate_or_excluded_new": duplicates,
        "old_eval_rows_in_train": 0,
        "sealed_holdout_opened": False,
        "focus": {k: v for k, v in focus_data.items() if k not in ("rows", "inputs")},
        "selection": (
            "validation only; existing development_test remains development confirmation, "
            "not final holdout"
        ),
    }
    if external:
        report.update(
            supplied_records=sum(s["bytes"] // 40 for s in plan["shards"]),
            spaced_candidate_examples=len(incoming),
            origins={
                "0": "defense_generated_apery",
                "1": "ancestral_replay",
                "2": "nodchip_hao_depth9",
            },
            ancestry="defense best1536 continuation; exact ancestral per-example exposures unknown",
            series_limitations=(
                "PSV has no original game ID/history; resets and first "
                "retained state infer families, not certified independence"
            ),
            source_games_scope=(
                "old sealed ledger plus new retained development games; "
                "all new sequences live in compressed row metadata"
            ),
            teacher_conflicts=(
                "existing labels retained, incoming duplicates excluded; never averaged"
            ),
            evaluation_subsample=(
                "old splits retained; hash modulo24 for old "
                "validation/development; new hash-first8192 each"
            ),
        )
    atomic(building / "manifest.json", encoded(report))
    building.rename(output)
    return report
