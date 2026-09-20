"""Bounded post-training local registration and the same browser path for Luna/Astra."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from .evaluator_data import START, atomic, digest, encoded


def ui_identity(root: Path) -> dict:
    ui = root.parent / "OpenShogiUI"
    scope = [
        "src",
        "scripts",
        "core-prototype-dev.ts",
        "model-build.ts",
        "vite.config.ts",
        "package.json",
        "package-lock.json",
        "release-model.json",
    ]
    if subprocess.check_output(["git", "status", "--porcelain", "--", *scope], cwd=ui):
        raise ValueError("commit the OSUI integration before sealing")
    names = subprocess.check_output(
        ["git", "ls-files", "--", *scope], cwd=ui, text=True
    ).splitlines()
    return {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ui, text=True).strip(),
        "files": {p: digest(ui / p) for p in names},
    }


def register(
    root: Path, run: Path, config: dict, model: Path, output: Path, *, rehearsal=False
) -> dict:
    """Register one descriptor; failure or rehearsal restores its previous bytes."""
    from .evaluator_run import RUNTIME_SOURCES, _audit_report, inside

    policy = config["development_integration"]
    ui = root.parent / "OpenShogiUI"
    for name, expected in policy["ui_identity"]["files"].items():
        path = ui / name
        if path.is_symlink() or digest(path) != expected:
            raise ValueError("OSUI differs from the reviewed integration: " + name)
    for name in ("module", "wasm"):
        if digest(root / RUNTIME_SOURCES[name]) != config["runtime"][name]["sha256"]:
            raise ValueError("development runtime differs from measured arena runtime")
    model = inside(model)
    output.mkdir(parents=True, exist_ok=True)
    sha = digest(model)
    report = output / "model-audit.json"
    if not report.exists():
        subprocess.run(
            [
                "node",
                str(root / "scripts/check_evaluator_model.mjs"),
                str(root / config["runtime"]["module"]["path"]),
                str(model),
                sha,
                str(root / config["runtime"]["replay"]["path"]),
                str(report),
            ],
            cwd=root,
            check=True,
            timeout=180,
        )
    _audit_report(run, config, report=report, model=model)
    selection = policy.get("selection", "r4c2")
    if selection not in {"r4c2", "r4c3"}:
        raise ValueError("unreviewed registration target")
    descriptor = inside(f"local/core-prototype/{selection}.json", exists=False)
    previous = descriptor.read_bytes() if descriptor.exists() else None
    if previous is not None:
        atomic(output / "previous-descriptor.json", previous)
    value = {
        "schema": "open_shogi_development_candidate/v1",
        "runId": config["run_id"] + ("-prefix-check" if rehearsal else ""),
        "leaf": {"path": str(model.relative_to(root)), "sha256": sha},
        "controller": None,
    }
    descriptor.parent.mkdir(parents=True, exist_ok=True)
    atomic(descriptor, encoded(value))
    published = descriptor.read_bytes()
    success = False
    url = f"http://127.0.0.1:{policy['port']}"
    try:

        def available():
            try:
                with urllib.request.urlopen(url, timeout=2) as response:
                    return response.status == 200
            except urllib.error.URLError:
                return False

        if not available():
            with (output / "server.log").open("ab") as log:
                process = subprocess.Popen(
                    [
                        "node",
                        "node_modules/vite/bin/vite.js",
                        "--host",
                        "127.0.0.1",
                        "--port",
                        str(policy["port"]),
                        "--strictPort",
                    ],
                    cwd=ui,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=log,
                    start_new_session=True,
                )
            atomic(
                output / "server.json",
                encoded(
                    {
                        "pid": process.pid,
                        "url": url,
                        "purpose": "local interactive OSUI; retained after verification",
                    }
                ),
            )
            for _ in range(60):
                if available():
                    break
                if process.poll() is not None:
                    raise RuntimeError("local OSUI startup failed; see server.log")
                time.sleep(0.25)
            else:
                raise TimeoutError("local OSUI startup timed out")
        subprocess.run(
            [
                "node",
                "scripts/verify-development-candidate.mjs",
                url + "/#/match",
                selection,
                sha,
                str(output / "browser"),
            ],
            cwd=ui,
            check=True,
            timeout=300,
        )
        browser = json.loads((output / "browser/browser.json").read_text())
        if browser["status"] != "PASS" or browser["expectedHash"] != sha:
            raise ValueError("browser candidate identity failed")
        result = {
            "schema": "open_shogiai_development_registration/v1",
            "status": "PASS",
            "run_sha256": digest(run / "run.json"),
            "model": value["leaf"],
            "url": url + "/#/match",
            "browser_sha256": digest(output / "browser/browser.json"),
            "rehearsal": rehearsal,
            "adoption": "unverified; no promotion",
            "next_owner": "user; no Astra restart required solely for integration",
        }
        atomic(output / "result.json", encoded(result))
        success = True
        return result
    finally:
        if not success or rehearsal:
            if descriptor.read_bytes() != published:
                raise ValueError("descriptor changed concurrently; refusing to overwrite it")
            if previous is None:
                descriptor.unlink()
            else:
                atomic(descriptor, previous)
            atomic(
                output / "rollback.json",
                encoded(
                    {
                        "restored_previous": previous is not None,
                        "reason": "rehearsal" if success else "registration_failed",
                    }
                ),
            )


def rehearsal(root: Path, run: Path, config: dict) -> dict:
    """Evidence only: no training/Arena completion receipts or adoption decisions."""
    import torch

    from . import evaluator_arena as arena
    from .defense_evaluation import screen
    from .evaluator_run import _training_identity
    from .phase10v_model import Phase10VModel, _snapshot

    ref = json.loads((run / "fit/resume.json").read_text())
    checkpoint = run / "fit" / ref["path"]
    if (
        checkpoint.parent != run / "fit"
        or checkpoint.is_symlink()
        or digest(checkpoint) != ref["sha256"]
    ):
        raise ValueError("invalid prefix checkpoint")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state["identity"] != _training_identity(run, config):
        raise ValueError("prefix checkpoint belongs to another run/data identity")
    if not 1 <= state["step"] <= 64 or state["step"] != ref["step"]:
        raise ValueError("rehearsal requires preserved initial optimizer updates")
    output = run / "post-training-rehearsal"
    output.mkdir(exist_ok=True)
    model = output / f"prefix-step{state['step']:06d}.osaval03"
    exported = _snapshot(
        state["parameters"], Phase10VModel.read(root / config["generation"]["leaf_path"]).seed
    )
    if model.exists() and digest(model) != exported.sha256:
        raise ValueError("prefix export changed")
    if not model.exists():
        exported.write(model)
    optional = screen(
        root, run, config, output=output / "missing-screen", model=model, mode="retained_only"
    )
    if optional["screen_pass"] is not None:
        raise ValueError("missing screen was represented as scored")
    folder = output / "short-pair"
    folder.mkdir(exist_ok=True)
    plan = {
        "config": {"max_plies": 64},
        "artifacts": {
            "probe": config["runtime"]["probe"],
            "baseline": {
                "path": config["generation"]["leaf_path"],
                "sha256": config["generation"]["leaf_sha256"],
            },
            "candidate": {"path": str(model.relative_to(root)), "sha256": digest(model)},
        },
        "purpose": "2 color-reversed 30-second pipeline checks; not strength or main Arena",
    }
    plan_path = folder / "plan.json"
    if plan_path.exists() and plan_path.read_bytes() != encoded(plan):
        raise ValueError("rehearsal plan changed")
    atomic(plan_path, encoded(plan))
    spec = importlib.util.spec_from_file_location(
        "rehearsal_probe", root / "scripts/compare_core_prototype.py"
    )
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    results = []
    for side in ("black", "white"):
        game = {
            "id": f"prefix-{side}",
            "pair": 0,
            "group": "pipeline",
            "clock_ms": 30000,
            "candidate_side": side,
            "initial_sfen": START,
        }
        attempt = folder / game["id"]
        attempt.mkdir(exist_ok=True)
        receipt = attempt / "rehearsal-result.json"
        if receipt.exists():
            result = json.loads(receipt.read_text())
        else:
            result = arena._play_game(
                root,
                folder,
                game,
                plan,
                digest(plan_path),
                attempt,
                helper,
                None,
                time.monotonic() + 70,
            )
            atomic(receipt, encoded(result))
        if (
            result["status"] not in ("completed", "incomplete")
            or result["absolute_deadline_violations"]
        ):
            raise ValueError("short paired execution failed")
        results.append(result)
    registered = register(root, run, config, model, output / "registration", rehearsal=True)
    report = {
        "schema": "open_shogiai_post_training_rehearsal/v1",
        "status": "PASS",
        "checkpoint": ref,
        "step": state["step"],
        "exposures": state["exposures"],
        "games": results,
        "optional_screen": optional,
        "registration": registered,
        "main_arena_complete": False,
        "strength_validated": False,
    }
    atomic(output / "result.json", encoded(report))
    return report
