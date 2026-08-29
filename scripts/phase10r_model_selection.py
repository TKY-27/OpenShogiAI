#!/usr/bin/env python3
"""Run the frozen Phase 10R architecture-selection measurements.

The script deliberately consumes only the public validation split and the
frozen Arena start pool.  It never opens the final holdout and it writes
generated measurements below ``local/phase10r-selection``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np

VARIANTS = (
    "sparse-pair-policy-wdl",
    "factorized-pair-triple-policy-score",
)
START_GROUPS = (
    "general_opening",
    "ibisha",
    "opponent_furibisha",
    "hard_middlegame_endgame",
)
START_MANIFEST = Path("artifacts/phase10/start-pool-manifest.json")
VALIDATION_ROWS = Path(
    "local/phase10r-data/phase10r-teacher-binding/1m/calibration-validation.jsonl"
)
DATASET_MANIFEST_SHA256 = "9661f52684bffb1ecfb63a13e59e76e1d57cf9dc4c2b181140fe0fbdc3f395d0"
START_MANIFEST_SHA256 = "491a3d54d3d67002fc03fc3644c16b405f5b1dbb662e88e4af260bd45757a1f1"
VALIDATION_ROWS_SHA256 = "2ce60cbb87195da3ee68d32a8ae8514ff16ac434741ee8717b63b7c8ca34e62c"
TEACHER_BINDING_IDENTITY_SHA256 = "781a45570ce6c88e29e4c9f7f3acb96e3981bb74d7da6fb10ff80cdd76f51cbf"
ARENA_SEED = 20260821
ARENA_DEPTH = 8
ARENA_HASH_MB = 32
ARENA_MAX_PLIES = 128
BOOTSTRAP_RESAMPLES = 100_000
OFFLINE_OUTPUT_VERSION = "v5"
Z95 = 1.959963984540054
_MAX_OUTPUT_BYTES = 32 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def ref(root: Path, path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def write_immutable(path: Path, value: Any) -> None:
    encoded = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or path.read_bytes() != encoded:
            raise RuntimeError(f"refusing to overwrite an existing measurement: {path}")
        return
    with path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()


def write_immutable_text(path: Path, text: str) -> None:
    encoded = text.encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or path.read_bytes() != encoded:
            raise RuntimeError(f"refusing to overwrite an existing measurement: {path}")
        return
    with path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()


def run_checked(
    command: list[str], root: Path, *, timeout: int | None = None
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=root,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout)[-4_096:]
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n{detail}"
        )
    return completed


def model_paths(root: Path, variant: str) -> dict[str, Path]:
    base = root / "local/phase10r-data/checkpoints/phase10r/1m" / variant / "teacher-bound-v1"
    int8 = root / "local/phase10r-selection/int8" / f"{variant}.osaval02"
    paths = {"float32": base / "teacher-bound-v1.osaval02", "int8": int8}
    for quantization, path in paths.items():
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"missing {quantization} artifact for {variant}: {path}")
    return paths


def inspect_model(root: Path, path: Path) -> dict[str, Any]:
    output = run_checked(
        ["target/release/open-shogi-cli", "model", "inspect", "--model", str(path)], root
    ).stdout.strip()
    value = json.loads(output)
    if not isinstance(value, dict) or value.get("schema") != "phase10r_osaval02_inspection/v1":
        raise RuntimeError(f"model inspection is not OSAVAL02: {path}")
    if value.get("quantization") not in {"float32", "int8"}:
        raise RuntimeError(f"model inspection quantization is invalid: {path}")
    return value


def validation_rows(root: Path) -> list[dict[str, Any]]:
    path = root / VALIDATION_ROWS
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise RuntimeError(f"validation input has a blank line: {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(f"validation row is not an object: {path}:{line_number}")
            if value.get("split") != "validation":
                raise RuntimeError(f"validation row has an unexpected split: {line_number}")
            rows.append(value)
    if len(rows) != 1_880:
        raise RuntimeError(f"validation row count is {len(rows)}, expected 1880")
    return rows


def ensure_sfen_input(root: Path, rows: list[dict[str, Any]]) -> Path:
    path = root / "local/phase10r-selection/inputs/calibration-validation.sfen"
    content = "".join(f"{row['sfen']}\n" for row in rows)
    write_immutable_text(path, content)
    return path


def peak_rss_bytes(stderr: str) -> int | None:
    match = re.search(r"\s+([0-9]+)\s+maximum resident set size", stderr)
    return int(match.group(1)) if match else None


def run_inference(root: Path, model: Path, input_path: Path, output_path: Path) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = output_path.with_suffix(".log")
    if output_path.exists() or output_path.is_symlink():
        if output_path.is_symlink() or not output_path.is_file():
            raise RuntimeError(f"inference output is not a regular file: {output_path}")
        return {"output": ref(root, output_path), "peak_rss_bytes": None, "resumed": True}
    command = [
        "/usr/bin/time",
        "-l",
        "target/release/open-shogi-cli",
        "model",
        "infer",
        "--model",
        str(model),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
    ]
    completed = subprocess.run(
        command,
        cwd=root,
        capture_output=True,
        text=True,
        timeout=1_800,
        check=False,
    )
    write_immutable_text(log_path, (completed.stdout or "") + (completed.stderr or ""))
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout)[-4_096:]
        raise RuntimeError(f"model inference failed for {model}: {detail}")
    if not output_path.is_file():
        raise RuntimeError(f"model inference did not publish output: {output_path}")
    return {
        "output": ref(root, output_path),
        "log": ref(root, log_path),
        "peak_rss_bytes": peak_rss_bytes(completed.stderr),
        "resumed": False,
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise RuntimeError(f"JSONL output has a blank line: {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(f"JSONL output row is not an object: {path}:{line_number}")
            rows.append(value)
    return rows


def rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        rank = (cursor + 1 + end) / 2.0
        for index in order[cursor:end]:
            ranks[index] = rank
        cursor = end
    return ranks


def correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or not left:
        return None
    left_mean = mean(left)
    right_mean = mean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum((a - left_mean) ** 2 for a in left))
    right_norm = math.sqrt(sum((b - right_mean) ** 2 for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    return numerator / (left_norm * right_norm)


def spearman(left: list[float], right: list[float]) -> float | None:
    return correlation(rankdata(left), rankdata(right))


def percentile(values: list[float], value: float) -> float:
    if not values:
        return value
    return float(np.percentile(np.asarray(values, dtype=np.float64), value))


def latency_summary(values_ns: list[int]) -> dict[str, Any]:
    values = [float(value) for value in values_ns]
    if not values:
        return {"count": 0, "mean_ns": None, "p50_ns": None, "p95_ns": None, "p99_ns": None}
    return {
        "count": len(values),
        "mean_ns": mean(values),
        "p50_ns": percentile(values, 50),
        "p95_ns": percentile(values, 95),
        "p99_ns": percentile(values, 99),
        "min_ns": min(values),
        "max_ns": max(values),
        "throughput_per_second": 1_000_000_000.0 / mean(values),
    }


def ece(probabilities: list[list[float]], targets: list[int]) -> float:
    bins = [[0, 0.0, 0] for _ in range(10)]
    for values, target in zip(probabilities, targets, strict=True):
        confidence = max(values)
        predicted = max(range(3), key=lambda index: (values[index], -index))
        index = min(9, int(confidence * 10.0))
        bins[index][0] += 1
        bins[index][1] += confidence
        bins[index][2] += int(predicted == target)
    total = len(targets)
    return sum(
        count / max(total, 1) * abs(confidence / count - correct / count)
        for count, confidence, correct in bins
        if count
    )


def output_inference(row: dict[str, Any]) -> dict[str, Any]:
    inference = row.get("inference")
    if (
        not isinstance(inference, dict)
        or inference.get("schema") != "open_shogiai_osaval02_inference/v1"
    ):
        raise RuntimeError("inference row is not OSAVAL02")
    legal = inference.get("legalMoves")
    wdl = inference.get("wdl")
    score = inference.get("score")
    mate = inference.get("mate")
    if (
        not isinstance(legal, list)
        or not isinstance(wdl, dict)
        or not isinstance(score, dict)
        or not isinstance(mate, dict)
    ):
        raise RuntimeError("inference row is incomplete")
    moves = [item.get("move") for item in legal if isinstance(item, dict)]
    logits = [float(item["logit"]) for item in legal if isinstance(item, dict)]
    if any(not isinstance(move, str) for move in moves) or len(moves) != len(set(moves)):
        raise RuntimeError("inference legal move list is invalid")
    probabilities = [float(wdl[key]) for key in ("loss", "draw", "win")]
    if any(not math.isfinite(value) or value < 0.0 for value in probabilities):
        raise RuntimeError("inference WDL probabilities are invalid")
    if not math.isclose(sum(probabilities), 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise RuntimeError("inference WDL probabilities do not sum to one")
    return {
        "moves": moves,
        "logits": logits,
        "wdl": probabilities,
        "score_cp": float(score["calibratedCp"]),
        "mate_class": mate.get("class"),
        "elapsed_ns": int(row["elapsedNs"]),
        "sfen": row.get("sfen"),
    }


def offline_metrics(rows: list[dict[str, Any]], outputs: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != len(outputs):
        raise RuntimeError(f"offline row count mismatch: {len(rows)} != {len(outputs)}")
    policy_top1 = policy_top3 = teacher_top1 = teacher_top3_recall = teacher_top3_exact = 0
    policy_nll = 0.0
    brier = nll = value_mse = value_mae = 0.0
    targets: list[int] = []
    probabilities: list[list[float]] = []
    cp_prediction: list[float] = []
    cp_target: list[float] = []
    rank_agreement: list[float] = []
    mate_correct = 0
    mate_total = 0
    latency: list[int] = []
    for source, output in zip(rows, outputs, strict=True):
        parsed = output_inference(output)
        legal_moves = source.get("legal_moves")
        played = source.get("played_move")
        ranking_scores = source.get("ranking_scores")
        teacher = source.get("raw_targets", {}).get("teacher_binding", {})
        candidates = teacher.get("candidates", []) if isinstance(teacher, dict) else []
        teacher_moves = [
            candidate.get("root_move") for candidate in candidates if isinstance(candidate, dict)
        ]
        if (
            not isinstance(legal_moves, list)
            or not isinstance(played, str)
            or not isinstance(ranking_scores, list)
        ):
            raise RuntimeError("validation row lacks policy labels")
        if set(parsed["moves"]) != set(legal_moves) or len(parsed["moves"]) != len(legal_moves):
            raise RuntimeError("runtime legal move coverage differs from validation labels")
        if parsed["moves"][0] == played:
            policy_top1 += 1
        if played in parsed["moves"][:3]:
            policy_top3 += 1
        if teacher_moves:
            if parsed["moves"][0] == teacher_moves[0]:
                teacher_top1 += 1
            prediction_top3 = set(parsed["moves"][:3])
            teacher_top3_set = set(teacher_moves[:3])
            teacher_top3_recall += len(prediction_top3 & teacher_top3_set) / len(teacher_top3_set)
            teacher_top3_exact += int(prediction_top3 == teacher_top3_set)
        if len(ranking_scores) != len(legal_moves):
            raise RuntimeError("ranking label length differs from legal move count")
        logits_by_move = dict(zip(parsed["moves"], parsed["logits"], strict=True))
        rank_agreement_value = spearman(
            [float(logits_by_move[move]) for move in legal_moves],
            [float(score) for score in ranking_scores],
        )
        if rank_agreement_value is not None:
            rank_agreement.append(rank_agreement_value)
        target = int(source["wdl"])
        values = parsed["wdl"]
        targets.append(target)
        probabilities.append(values)
        if played not in parsed["moves"]:
            raise RuntimeError("played move is absent from runtime legal moves")
        largest_logit = max(parsed["logits"])
        log_normalizer = largest_logit + math.log(
            sum(math.exp(logit - largest_logit) for logit in parsed["logits"])
        )
        policy_nll += log_normalizer - parsed["logits"][parsed["moves"].index(played)]
        one_hot = [float(index == target) for index in range(3)]
        brier += (
            sum((actual - expected) ** 2 for actual, expected in zip(values, one_hot, strict=True))
            / 3.0
        )
        nll -= math.log(max(values[target], 1.0e-12))
        expected_value = values[2] - values[0]
        target_value = target / 2.0 - 1.0
        value_mse += (expected_value - target_value) ** 2
        value_mae += abs(expected_value - target_value)
        binding_score = teacher.get("score", {}) if isinstance(teacher, dict) else {}
        if isinstance(binding_score, dict) and binding_score.get("kind") == "cp":
            cp_prediction.append(parsed["score_cp"])
            cp_target.append(float(binding_score["value"]))
        if isinstance(binding_score, dict) and binding_score.get("kind") == "mate":
            expected_mate = (
                "mating"
                if float(binding_score["value"]) > 0
                else ("mated" if float(binding_score["value"]) < 0 else "no_mate_label")
            )
            mate_total += 1
            mate_correct += int(parsed["mate_class"] == expected_mate)
        latency.append(parsed["elapsed_ns"])
    return {
        "rows": len(rows),
        "policy_examples": len(rows),
        "policy_top1": policy_top1 / len(rows),
        "policy_top3": policy_top3 / len(rows),
        "policy_nll": policy_nll / len(rows),
        "teacher_top1": teacher_top1 / max(len(rows), 1),
        "teacher_top3_recall": teacher_top3_recall / max(len(rows), 1),
        "teacher_top3_exact_set": teacher_top3_exact / max(len(rows), 1),
        "ranking_spearman": mean(rank_agreement),
        "ranking_examples": len(rank_agreement),
        "wdl_examples": len(rows),
        "wdl_brier": brier / len(rows),
        "wdl_nll": nll / len(rows),
        "value_mse": value_mse / len(rows),
        "value_mae": value_mae / len(rows),
        "calibration_ece": ece(probabilities, targets),
        "mate_sign_accuracy": mate_correct / max(mate_total, 1),
        "mate_examples": mate_total,
        "score_correlation_pearson": correlation(cp_prediction, cp_target),
        "score_correlation_spearman": spearman(cp_prediction, cp_target),
        "score_examples": len(cp_prediction),
        "score_mae_cp": mean(
            abs(prediction - target)
            for prediction, target in zip(cp_prediction, cp_target, strict=True)
        )
        if cp_prediction
        else None,
        "inference_latency": latency_summary(latency),
    }


def quantization_parity(
    float_outputs: list[dict[str, Any]], int8_outputs: list[dict[str, Any]]
) -> dict[str, Any]:
    if len(float_outputs) != len(int8_outputs):
        raise RuntimeError("float/int8 output row count differs")
    max_wdl = max_score = max_logit = 0.0
    top1 = exact_order = mate = 0
    for left, right in zip(float_outputs, int8_outputs, strict=True):
        a = output_inference(left)
        b = output_inference(right)
        if a["sfen"] != b["sfen"] or set(a["moves"]) != set(b["moves"]):
            raise RuntimeError("float/int8 position or legal move coverage differs")
        max_wdl = max(max_wdl, *(abs(x - y) for x, y in zip(a["wdl"], b["wdl"], strict=True)))
        max_score = max(max_score, abs(a["score_cp"] - b["score_cp"]))
        a_logits = dict(zip(a["moves"], a["logits"], strict=True))
        b_logits = dict(zip(b["moves"], b["logits"], strict=True))
        max_logit = max(
            max_logit,
            *(abs(a_logits[move] - b_logits[move]) for move in a_logits),
        )
        top1 += int(a["moves"][:1] == b["moves"][:1])
        exact_order += int(a["moves"] == b["moves"])
        mate += int(a["mate_class"] == b["mate_class"])
    count = max(len(float_outputs), 1)
    return {
        "rows": len(float_outputs),
        "wdl_max_abs_delta": max_wdl,
        "calibrated_score_cp_max_abs_delta": max_score,
        "policy_logit_max_abs_delta": max_logit,
        "policy_top1_agreement": top1 / count,
        "policy_order_exact_agreement": exact_order / count,
        "mate_class_agreement": mate / count,
    }


def start_pool(root: Path) -> list[dict[str, Any]]:
    path = root / START_MANIFEST
    if sha256(path) != START_MANIFEST_SHA256:
        raise RuntimeError("frozen Arena start manifest SHA-256 differs")
    document = read_json(path)
    positions = document.get("positions")
    if (
        document.get("schema") != "open_shogiai_phase10_start_pool/v2"
        or not isinstance(positions, list)
        or len(positions) != 800
    ):
        raise RuntimeError("frozen Arena start pool is invalid")
    counts = {group: 0 for group in START_GROUPS}
    for position in positions:
        group = position.get("assignedGroup")
        if group not in counts or not isinstance(position.get("sfen"), str):
            raise RuntimeError("start pool row is invalid")
        counts[group] += 1
    if counts != {group: 200 for group in START_GROUPS}:
        raise RuntimeError(f"start pool group counts are invalid: {counts}")
    return positions


def start_pool_metrics(
    root: Path, artifacts: dict[str, dict[str, Any]], positions: list[dict[str, Any]]
) -> dict[str, Any]:
    input_path = root / "local/phase10r-selection/inputs/start-pool.sfen"
    content = "".join(f"{position['sfen']} 1\n" for position in positions)
    write_immutable_text(input_path, content)
    results: dict[str, Any] = {}
    for variant in VARIANTS:
        model_path = root / artifacts[variant]["float32"]["artifact"]["path"]
        output_path = (
            root
            / "local/phase10r-selection"
            / f"offline-{OFFLINE_OUTPUT_VERSION}"
            / f"{variant}-start-pool.jsonl"
        )
        run = run_inference(root, model_path, input_path, output_path)
        rows = read_jsonl(output_path)
        if len(rows) != len(positions):
            raise RuntimeError("start-pool inference row count differs")
        by_group: dict[str, list[dict[str, Any]]] = {group: [] for group in START_GROUPS}
        for position, row in zip(positions, rows, strict=True):
            parsed = output_inference(row)
            by_group[position["assignedGroup"]].append(parsed)
        results[variant] = {
            "input": ref(root, input_path),
            "run": run,
            "groups": {
                group: {
                    "rows": len(group_rows),
                    "mean_score_cp": mean(item["score_cp"] for item in group_rows),
                    "mean_expected_wdl": mean(
                        item["wdl"][2] - item["wdl"][0] for item in group_rows
                    ),
                    "inference_latency": latency_summary(
                        [item["elapsed_ns"] for item in group_rows]
                    ),
                }
                for group, group_rows in by_group.items()
            },
        }
    return results


def run_offline(root: Path) -> dict[str, Any]:
    rows = validation_rows(root)
    validation_path = root / VALIDATION_ROWS
    if sha256(validation_path) != VALIDATION_ROWS_SHA256:
        raise RuntimeError("frozen calibration-validation input SHA-256 differs")
    input_path = ensure_sfen_input(root, rows)
    artifacts: dict[str, dict[str, Any]] = {}
    outputs: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for variant in VARIANTS:
        artifacts[variant] = {}
        outputs[variant] = {}
        for quantization, model_path in model_paths(root, variant).items():
            inspection = inspect_model(root, model_path)
            if (
                inspection.get("variantId") != variant
                or inspection.get("datasetManifestSha256") != DATASET_MANIFEST_SHA256
            ):
                raise RuntimeError(
                    f"artifact identity is not bound to the frozen candidate: {model_path}"
                )
            output_path = (
                root
                / "local/phase10r-selection"
                / f"offline-{OFFLINE_OUTPUT_VERSION}"
                / f"{variant}-{quantization}.jsonl"
            )
            run = run_inference(root, model_path, input_path, output_path)
            output_rows = read_jsonl(output_path)
            if len(output_rows) != len(rows):
                raise RuntimeError(f"inference output row count differs for {model_path}")
            artifacts[variant][quantization] = {
                "artifact": ref(root, model_path),
                "inspection": inspection,
                "run": run,
                "output": ref(root, output_path),
            }
            outputs[variant][quantization] = output_rows
    positions = start_pool(root)
    candidates = {
        variant: offline_metrics(rows, outputs[variant]["float32"]) for variant in VARIANTS
    }
    parity = {
        variant: quantization_parity(outputs[variant]["float32"], outputs[variant]["int8"])
        for variant in VARIANTS
    }
    result = {
        "schema": "open_shogiai_phase10r_model_selection_offline/v1",
        "controls": {
            "validation_rows": ref(root, validation_path),
            "validation_input": ref(root, input_path),
            "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
            "teacher_binding_identity_sha256": TEACHER_BINDING_IDENTITY_SHA256,
            "final_holdout_opened": False,
        },
        "artifacts": artifacts,
        "candidates": candidates,
        "float_int8_parity": parity,
        "start_pool": start_pool_metrics(root, artifacts, positions),
    }
    output = root / "local/phase10r-selection" / f"offline-results-{OFFLINE_OUTPUT_VERSION}.json"
    write_immutable(output, result)
    result["result"] = ref(root, output)
    return result


def canonical_sfen(value: str) -> str:
    parts = value.split()
    if len(parts) == 3:
        parts.append("1")
    if len(parts) != 4 or parts[3] != "1":
        raise RuntimeError(f"Arena start SFEN must be a move-one state: {value}")
    return " ".join(parts)


def matchup_specs() -> list[dict[str, Any]]:
    specs = []
    for variant in VARIANTS:
        for games, movetime in ((400, 250), (800, 1_000), (200, 5_000)):
            specs.append(
                {
                    "id": f"{variant}-vs-handcrafted-{movetime}ms",
                    "a": variant,
                    "b": "handcrafted-experimental",
                    "games": games,
                    "movetime_ms": movetime,
                }
            )
    specs.append(
        {
            "id": "sparse-pair-policy-wdl-vs-factorized-pair-triple-policy-score-1000ms",
            "a": VARIANTS[0],
            "b": VARIANTS[1],
            "games": 400,
            "movetime_ms": 1_000,
        }
    )
    return specs


def arena_command(
    root: Path, spec: dict[str, Any], a_model: Path, b_model: Path | None, sfen: str, output: Path
) -> list[str]:
    command = [
        "target/release/open-shogi-cli",
        "arena",
        "--games",
        "2",
        "--player-a",
        "neural",
        "--a-model",
        str(a_model),
        "--player-b",
        "neural" if b_model is not None else "handcrafted-experimental",
        "--a-depth",
        str(ARENA_DEPTH),
        "--b-depth",
        str(ARENA_DEPTH),
        "--a-hash-mb",
        str(ARENA_HASH_MB),
        "--b-hash-mb",
        str(ARENA_HASH_MB),
        "--movetime-ms",
        str(spec["movetime_ms"]),
        "--sfen",
        sfen,
        "--max-plies",
        str(ARENA_MAX_PLIES),
        "--seed",
        str(ARENA_SEED),
        "--git-commit",
        git_commit(root),
        "--output-dir",
        str(output),
    ]
    if b_model is not None:
        command[command.index("--player-b") + 1] = "neural"
        command.extend(["--b-model", str(b_model)])
    return command


def git_commit(root: Path) -> str:
    return run_checked(["git", "rev-parse", "HEAD"], root).stdout.strip()


def run_arena_pair(
    root: Path,
    spec: dict[str, Any],
    pair_index: int,
    position: dict[str, Any],
    model_paths_by_variant: dict[str, Path],
) -> dict[str, Any]:
    output = root / "local/phase10r-selection/arena" / spec["id"] / f"pair-{pair_index:04d}"
    report_path = output / "arena-report.json"
    if report_path.is_file():
        report = read_json(report_path)
        resumed = True
    else:
        output.mkdir(parents=True, exist_ok=True)
        log_path = output / "arena.log"
        a_model = model_paths_by_variant[spec["a"]]
        b_model = (
            None if spec["b"] == "handcrafted-experimental" else model_paths_by_variant[spec["b"]]
        )
        command = arena_command(
            root, spec, a_model, b_model, canonical_sfen(position["sfen"]), output
        )
        resume = output.joinpath("arena.state").exists() or output.joinpath("games").exists()
        if resume:
            command.append("--resume")
        with log_path.open("a", encoding="utf-8") as log:
            completed = subprocess.run(
                command, cwd=root, stdout=log, stderr=subprocess.STDOUT, text=True, check=False
            )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Arena pair failed ({spec['id']} pair {pair_index}); see {log_path}"
            )
        report = read_json(report_path)
        resumed = False
    expected_a_sha = sha256(model_paths_by_variant[spec["a"]])
    expected_b_sha = (
        None
        if spec["b"] == "handcrafted-experimental"
        else sha256(model_paths_by_variant[spec["b"]])
    )
    validate_arena_report(
        report,
        spec,
        position,
        expected_a_sha=expected_a_sha,
        expected_b_sha=expected_b_sha,
        expected_git_commit=git_commit(root),
    )
    return {
        "pair_index": pair_index,
        "position_id": position["positionId"],
        "assigned_group": position["assignedGroup"],
        "initial_sfen": canonical_sfen(position["sfen"]),
        "report": ref(root, report_path),
        "report_sha256": sha256(report_path),
        "report_data": report,
        "resumed": resumed,
    }


def validate_arena_report(
    report: dict[str, Any],
    spec: dict[str, Any],
    position: dict[str, Any],
    *,
    expected_a_sha: str,
    expected_b_sha: str | None,
    expected_git_commit: str,
) -> None:
    run = report.get("run")
    metrics = report.get("metrics")
    games = report.get("games")
    if not isinstance(run, dict) or not isinstance(metrics, dict) or not isinstance(games, list):
        raise RuntimeError("Arena report is incomplete")
    if (
        run.get("gameLimit") != 2
        or run.get("seed") != ARENA_SEED
        or run.get("maxPlies") != ARENA_MAX_PLIES
        or run.get("gitCommit") != expected_git_commit
    ):
        raise RuntimeError("Arena report violates fixed game controls")
    if run.get("initialSfen") != canonical_sfen(position["sfen"]):
        raise RuntimeError("Arena report start position differs from the start manifest")
    if (
        run.get("budget", {}).get("kind") != "movetime_ms"
        or run.get("budget", {}).get("value") != spec["movetime_ms"]
    ):
        raise RuntimeError("Arena report time control differs")
    if run.get("opening", {}).get("enabled") is not False:
        raise RuntimeError("Arena report has an enabled opening book")
    player_a = run.get("playerA")
    player_b = run.get("playerB")
    if not isinstance(player_a, dict) or not isinstance(player_b, dict):
        raise RuntimeError("Arena player configuration is missing")
    for player, expected_sha in ((player_a, expected_a_sha), (player_b, expected_b_sha)):
        if (
            player.get("searchDepth") != ARENA_DEPTH
            or player.get("hashMegabytes") != ARENA_HASH_MB
            or player.get("transposition") is not True
            or player.get("openingEnabled") is not False
        ):
            raise RuntimeError("Arena player search controls differ")
        if player.get("modelArtifactSha256") != expected_sha:
            raise RuntimeError("Arena player artifact identity differs")
    if player_a.get("evaluatorKind") != "neural":
        raise RuntimeError("Arena player A is not neural")
    if expected_b_sha is None:
        if player_b.get("evaluatorKind") != "handcrafted-experimental":
            raise RuntimeError("Arena player B is not handcrafted-experimental")
    elif player_b.get("evaluatorKind") != "neural":
        raise RuntimeError("Arena player B is not neural")
    if (
        metrics.get("games") != 2
        or metrics.get("finishedGames") != 2
        or metrics.get("illegalMoves") != 0
        or len(games) != 2
    ):
        raise RuntimeError("Arena pair did not complete two legal games")
    a_label = player_a.get("label")
    b_label = player_b.get("label")
    if not isinstance(a_label, str) or not isinstance(b_label, str):
        raise RuntimeError("Arena player identity is missing")
    if {game.get("black") for game in games} != {a_label, b_label}:
        raise RuntimeError("Arena pair did not reverse colors")
    for game in games:
        expected_white = b_label if game.get("black") == a_label else a_label
        if game.get("white") != expected_white:
            raise RuntimeError("Arena pair has an invalid color assignment")
    allowed = {"black_win", "white_win", "draw", "max_plies"}
    if any(game.get("result") not in allowed for game in games):
        raise RuntimeError("Arena report contains an unexplained game result")


def game_points(result: str, candidate_color: str) -> tuple[float, str, bool]:
    if result == "max_plies":
        return 0.0, "capped", True
    if result == "draw":
        return 0.5, "draw", False
    winner = "black" if result == "black_win" else "white"
    if winner == candidate_color:
        return 1.0, "win", False
    return 0.0, "loss", False


def wilson(score: float, games: int) -> tuple[float, float]:
    if games <= 0:
        return (math.nan, math.nan)
    denominator = 1.0 + Z95 * Z95 / games
    center = (score + Z95 * Z95 / (2.0 * games)) / denominator
    margin = (
        Z95
        / denominator
        * math.sqrt(score * (1.0 - score) / games + Z95 * Z95 / (4.0 * games * games))
    )
    return (max(0.0, center - margin), min(1.0, center + margin))


def elo(score: float) -> float:
    bounded = min(1.0 - 0.5 / 10_000.0, max(0.5 / 10_000.0, score))
    return 400.0 * math.log10(bounded / (1.0 - bounded))


def paired_bootstrap(
    pair_points: list[float], pair_games: list[int], seed: int
) -> tuple[float, float]:
    if not pair_points:
        return (math.nan, math.nan)
    generator = np.random.default_rng(seed)
    points = np.asarray(pair_points, dtype=np.float64)
    games = np.asarray(pair_games, dtype=np.float64)
    samples: list[np.ndarray] = []
    for _ in range(0, BOOTSTRAP_RESAMPLES, 1_000):
        indexes = generator.integers(0, len(points), size=(1_000, len(points)))
        sample_points = points[indexes].sum(axis=1)
        sample_games = games[indexes].sum(axis=1)
        samples.append(sample_points / np.maximum(sample_games, 1.0))
    values = np.concatenate(samples)[:BOOTSTRAP_RESAMPLES]
    return (float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975)))


def aggregate_arena(
    pair_results: list[dict[str, Any]], spec: dict[str, Any], *, include_groups: bool = True
) -> dict[str, Any]:
    wins = draws = losses = capped = illegal = 0
    points = 0.0
    finished_games = 0
    pair_points: list[float] = []
    pair_games: list[int] = []
    latency_ms: list[float] = []
    nps: list[float] = []
    groups: dict[str, list[dict[str, Any]]] = {group: [] for group in START_GROUPS}
    for pair in pair_results:
        report = pair["report_data"]
        run = report["run"]
        metrics = report["metrics"]
        a_label = run["playerA"]["label"]
        pair_point = 0.0
        pair_finished = 0
        for game in report["games"]:
            candidate_color = "black" if game["black"] == a_label else "white"
            point, classification, is_capped = game_points(game["result"], candidate_color)
            pair_point += point
            pair_finished += int(not is_capped)
            points += point
            capped += int(is_capped)
            if classification == "win":
                wins += 1
            elif classification == "draw":
                draws += 1
            elif classification == "loss":
                losses += 1
            illegal += int(metrics.get("illegalMoves", 0) > 0)
        finished_games += pair_finished
        pair_points.append(pair_point)
        pair_games.append(pair_finished)
        candidate_nodes = sum(int(game["playerASearchNodes"]) for game in report["games"])
        candidate_elapsed = sum(float(game["playerASearchElapsedMs"]) for game in report["games"])
        candidate_searches = sum(int(game["playerASearches"]) for game in report["games"])
        if candidate_elapsed > 0:
            nps.append(candidate_nodes / (candidate_elapsed / 1_000.0))
        if candidate_searches > 0:
            latency_ms.append(candidate_elapsed / candidate_searches)
        groups[pair["assigned_group"]].append(pair)
    score = points / finished_games if finished_games else math.nan
    wilson_ci = wilson(score, finished_games)
    bootstrap_seed = (
        ARENA_SEED + int(hashlib.sha256(spec["id"].encode()).hexdigest()[:8], 16) % 10_000
    )
    bootstrap_ci = paired_bootstrap(pair_points, pair_games, bootstrap_seed)
    group_results: dict[str, Any] = {}
    if include_groups:
        for group, entries in groups.items():
            if not entries:
                continue
            group_result = aggregate_arena(
                entries, {**spec, "id": f"{spec['id']}-{group}"}, include_groups=False
            )
            group_results[group] = group_result
    return {
        "matchup": spec["id"],
        "candidate": spec["a"],
        "opponent": spec["b"],
        "movetime_ms": spec["movetime_ms"],
        "games": len(pair_results) * 2,
        "pairs": len(pair_results),
        "finished_wld_games": finished_games,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "capped_games": capped,
        "illegal_games": illegal,
        "unexplained_crashes": 0,
        "score_rate": score,
        "wilson_95_ci": {"lower": wilson_ci[0], "upper": wilson_ci[1]},
        "paired_color_bootstrap_95_ci": {"lower": bootstrap_ci[0], "upper": bootstrap_ci[1]},
        "elo_vs_opponent": elo(score) if math.isfinite(score) else math.nan,
        "candidate_search_nps": {
            "mean": mean(nps) if nps else None,
            "p50": median(nps) if nps else None,
            "p95": percentile(nps, 95) if nps else None,
        },
        "candidate_search_latency_ms": {
            "mean": mean(latency_ms) if latency_ms else None,
            "p50": median(latency_ms) if latency_ms else None,
            "p95": percentile(latency_ms, 95) if latency_ms else None,
        },
        "opening_groups": group_results if include_groups else None,
    }


def run_arena(root: Path, workers: int) -> dict[str, Any]:
    positions = start_pool(root)
    model_paths_by_variant = {
        variant: model_paths(root, variant)["float32"] for variant in VARIANTS
    }
    specifications = matchup_specs()
    aggregates: list[dict[str, Any]] = []
    all_pairs: list[dict[str, Any]] = []
    for spec in specifications:
        count = spec["games"] // 2
        if count > len(positions):
            raise RuntimeError("Arena campaign requests more starts than the frozen pool")
        selected_positions = positions[:count]
        pairs: list[dict[str, Any]] = []
        print(f"Arena {spec['id']}: {count} paired starts", flush=True)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    run_arena_pair, root, spec, index, position, model_paths_by_variant
                ): index
                for index, position in enumerate(selected_positions, 1)
            }
            for future in as_completed(futures):
                pair = future.result()
                pairs.append(pair)
                print(f"  {spec['id']} pair {pair['pair_index']}/{count}", flush=True)
        pairs.sort(key=lambda pair: pair["pair_index"])
        aggregate = aggregate_arena(pairs, spec)
        aggregates.append(aggregate)
        all_pairs.extend(pairs)
        print(
            json.dumps(
                {
                    key: aggregate[key]
                    for key in (
                        "matchup",
                        "games",
                        "wins",
                        "draws",
                        "losses",
                        "capped_games",
                        "score_rate",
                    )
                },
                sort_keys=True,
            ),
            flush=True,
        )
    result = {
        "schema": "open_shogiai_phase10r_model_selection_arena/v1",
        "controls": {
            "opponent": "handcrafted-experimental",
            "candidate_mode": "pure_value",
            "equal_wall_clock": True,
            "paired_reversed_colors": True,
            "opening_book": "disabled",
            "depth_cap": ARENA_DEPTH,
            "hash_mib_per_player": ARENA_HASH_MB,
            "max_plies": ARENA_MAX_PLIES,
            "seed": ARENA_SEED,
            "start_manifest": ref(root, root / START_MANIFEST),
            "paired_color_bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "finished_wld_score_excludes_capped_games": True,
            "git_commit": git_commit(root),
        },
        "matchups": aggregates,
        "pairs": [
            {key: value for key, value in pair.items() if key != "report_data"}
            for pair in all_pairs
        ],
    }
    output = root / "local/phase10r-selection/arena-results.json"
    write_immutable(output, result)
    result["result"] = ref(root, output)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--mode", choices=("offline", "arena", "all"), default="all")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 8:
        parser.error("--workers must be in 1..8")
    root = args.root.resolve()
    result: dict[str, Any] = {"schema": "open_shogiai_phase10r_model_selection/v1"}
    if args.mode in {"offline", "all"}:
        result["offline"] = run_offline(root)
    if args.mode in {"arena", "all"}:
        result["arena"] = run_arena(root, args.workers)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
