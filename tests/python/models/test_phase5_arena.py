import hashlib
import math
from pathlib import Path

import open_shogi_training.selfplay.execution as execution_module
import pytest
from open_shogi_training.models.phase5_arena import (
    PHASE5_BOOTSTRAP_SAMPLES,
    PHASE5_GIT_COMMIT,
    PHASE5_PAIRS,
    _pair_seed,
    _run_engine_command,
    _safe_ratio,
    phase5_comparisons,
)
from open_shogi_training.selfplay.common import ArtifactRef, ContractError


def test_phase5_comparison_matrix_is_closed_and_seeded_in_original_order() -> None:
    f32 = Path("weights/value.f32.osaval")
    int8 = Path("weights/value.int8.osaval")
    comparisons = phase5_comparisons(f32_model=f32, int8_model=int8)

    assert PHASE5_GIT_COMMIT == ""
    assert PHASE5_PAIRS == 20
    assert PHASE5_BOOTSTRAP_SAMPLES == 100_000
    assert [comparison.name for comparison in comparisons] == [
        "material-vs-baseline",
        "baseline-vs-experimental",
        "experimental-vs-neural-f32",
        "neural-f32-vs-int8",
        "neural-int8-opening-off-vs-on",
    ]
    assert [comparison.seed_index for comparison in comparisons] == list(range(5))
    assert len({comparison.name for comparison in comparisons}) == 5
    assert comparisons[2].model_b == f32
    assert comparisons[3].model_a == f32
    assert comparisons[3].model_b == int8
    assert comparisons[4].player_b_opening is True
    assert _pair_seed(comparisons[0], 0) == 20_260_810
    assert _pair_seed(comparisons[-1], 19) == 20_261_229


def test_empty_phase5_measurement_buckets_emit_finite_zero_rates() -> None:
    for numerator in (0, 1, 10**12):
        value = _safe_ratio(numerator, 0)
        assert value == 0.0
        assert math.isfinite(value)


@pytest.mark.parametrize("failure_index", [3, 4, 5])
def test_phase5_subprocess_fails_closed_on_timeout_output_or_tree_memory_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_index: int
) -> None:
    engine = tmp_path / "open-shogi-cli"
    engine.write_bytes(b"#!/bin/sh\nexit 0\n")
    engine.chmod(0o700)
    engine_bytes = engine.read_bytes()
    engine_ref = ArtifactRef(
        path="open-shogi-cli",
        sha256=hashlib.sha256(engine_bytes).hexdigest(),
        size=len(engine_bytes),
    )
    outcome = [b"", b"bounded failure", 0, False, False, False, 1024, "process_tree_ps_rss_sum"]
    outcome[failure_index] = True
    observed: dict[str, object] = {}

    def bounded(argv, *, cwd, timeout_seconds, pass_fds, memory_limit_bytes, launch_guard):
        launch_guard()
        observed.update(
            argv=argv,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            pass_fds=pass_fds,
            memory_limit_bytes=memory_limit_bytes,
        )
        launch_guard()
        return tuple(outcome)

    monkeypatch.setattr(execution_module, "_run_bounded_process", bounded)
    with pytest.raises(ContractError, match="failed safely"):
        _run_engine_command(
            engine,
            ["arena"],
            repository_root=tmp_path,
            expected_engine=engine_ref,
            timeout_seconds=17,
            memory_limit_mib=384,
        )

    assert observed["cwd"] == tmp_path
    assert observed["timeout_seconds"] == 17
    assert observed["memory_limit_bytes"] == 384 * 1024 * 1024
    assert observed["argv"][0] != engine
    assert Path(observed["argv"][0]).is_relative_to(tmp_path / "local/runtime-snapshots")


def test_phase5_subprocess_never_runs_an_engine_outside_the_validated_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = tmp_path / "open-shogi-cli"
    engine.write_bytes(b"#!/bin/sh\nexit 0\n")
    engine.chmod(0o700)
    called = False

    def bounded(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("unverified engine must not reach process creation")

    monkeypatch.setattr(execution_module, "_run_bounded_process", bounded)
    with pytest.raises(ContractError, match="validated build receipt"):
        _run_engine_command(
            engine,
            ["arena"],
            repository_root=tmp_path,
            expected_engine=ArtifactRef(
                path="open-shogi-cli",
                sha256="0" * 64,
                size=engine.stat().st_size,
            ),
            timeout_seconds=17,
            memory_limit_mib=384,
        )

    assert called is False
